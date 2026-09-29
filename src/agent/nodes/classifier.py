"""Classifier node -- routes user queries to specialist agents via Haiku.

This node is wired as a conditional edge from ``context_loader``. It calls Haiku
to classify the user's query, selects 1-3 specialist agents, generates targeted
sub-questions, and returns ``Send`` objects so LangGraph dispatches the chosen
specialists in parallel.

Falls back to the general ``cofounder`` agent when classification fails (LLM error,
invalid JSON, JSON that is not a list, or no valid agents returned).
"""

import json
import logging

from langgraph.types import Send

from src.agent.agents.registry import AGENT_REGISTRY
from src.agent.prompts.classifier import CLASSIFIER_PROMPT
from src.agent.state import AgentInput, CofounderState
from src.config.llm_config import get_haiku_client

logger = logging.getLogger(__name__)


def classify_and_route(state: CofounderState) -> list[Send]:
    """Classify the user query and return Send objects to dispatch specialists.

    Uses Haiku (cheap/fast) to determine which agents should handle the query.
    The classifier selects up to 3 agents and generates a domain-specific
    sub-question for each. Results are returned as ``Send`` objects that
    LangGraph uses for parallel dispatch.

    Args:
        state: Current graph state with ``messages`` and ``company_context``.

    Returns:
        List of ``Send`` objects targeting ``specialist_{agent_id}`` nodes.
    """
    # Get latest user message
    messages = state.get("messages", [])
    if not messages:
        return _fallback_send(state, "No message provided")
    user_message = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # Build agent descriptions for the classifier
    agent_descriptions = "\n".join(
        f"- **{cfg.agent_id}** ({cfg.name}): {cfg.description}"
        for cfg in AGENT_REGISTRY.values()
    )

    # Build classifier prompt
    prompt = CLASSIFIER_PROMPT.format(
        company_context=state.get("company_context", ""),
        agent_descriptions=agent_descriptions,
        user_message=user_message,
    )

    # Call Haiku
    try:
        client, model = get_haiku_client()
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text.strip()

        # Parse JSON — handle potential markdown code block wrapping
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        classifications = json.loads(response_text)
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        logger.warning("Classifier JSON parse failed: %s — falling back to cofounder", e)
        return _fallback_send(state, user_message)
    except Exception as e:
        logger.error("Classifier LLM call failed: %s — falling back to cofounder", e)
        return _fallback_send(state, user_message)

    # Valid JSON that is not a list (e.g. an object wrapping the list) cannot be routed
    if not isinstance(classifications, list):
        logger.warning(
            "Classifier returned %s instead of a list — falling back to cofounder",
            type(classifications).__name__,
        )
        return _fallback_send(state, user_message)

    # Validate and cap at 3 agents
    valid = []
    for c in classifications[:3]:
        if not isinstance(c, dict):
            logger.warning("Classifier returned a non-object entry: %r — skipping", c)
            continue
        agent_id = c.get("agent_id", "")
        if not isinstance(agent_id, str) or agent_id not in AGENT_REGISTRY:
            logger.warning("Classifier returned unknown agent_id: %s — skipping", agent_id)
            continue
        valid.append(c)

    if not valid:
        return _fallback_send(state, user_message)

    # Ensure at least one primary
    has_primary = any(c.get("relevance") == "primary" for c in valid)
    if not has_primary:
        valid[0]["relevance"] = "primary"

    # Build Send objects
    company_context = state.get("company_context", "")
    user_id = state.get("user_id", "")
    sends = []
    for c in valid:
        agent_id = c["agent_id"]
        sends.append(
            Send(
                f"specialist_{agent_id}",
                AgentInput(
                    query=c.get("sub_question", user_message),
                    original_query=user_message,
                    company_context=company_context,
                    user_id=user_id,
                    source_hints=c.get("source_hints", []),
                ),
            )
        )

    return sends


def _fallback_send(state: CofounderState, user_message: str) -> list[Send]:
    """Fallback: route everything to the general cofounder agent.

    Args:
        state: Current graph state for extracting context fields.
        user_message: The user's raw message text.

    Returns:
        Single-element list with a ``Send`` to ``specialist_cofounder``.
    """
    return [
        Send(
            "specialist_cofounder",
            AgentInput(
                query=user_message,
                original_query=user_message,
                company_context=state.get("company_context", ""),
                user_id=state.get("user_id", ""),
                source_hints=[],
            ),
        )
    ]
