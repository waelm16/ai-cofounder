"""Synthesizer node -- merges specialist outputs into a unified cofounder response.

Receives all specialist ``AgentOutput`` entries (accumulated by the ``operator.add``
reducer) and calls Sonnet to produce a single coherent response that resolves
contradictions and prioritizes actionability.

On LLM failure, falls back to concatenating the raw specialist responses.

Also exports ``build_synthesis_prompt`` for the streaming endpoint, which needs
to build the same prompt but stream the Sonnet response token-by-token.
"""

import logging

from src.agent.prompts.synthesizer import SYNTHESIZER_PROMPT
from src.agent.state import CofounderState
from src.config.llm_config import get_sonnet_client

logger = logging.getLogger(__name__)


def build_synthesis_prompt(state: CofounderState) -> tuple[str, list[dict], list[str]]:
    """Build the synthesis prompt and collect metadata from specialist outputs.

    Extracted so the streaming endpoint can reuse the prompt building logic
    without duplicating it.

    Args:
        state: Full graph state with ``agent_outputs``, ``messages``, and
            ``company_context``.

    Returns:
        Tuple of (prompt string, all_sources list, agents_consulted list).
    """
    agent_outputs = state.get("agent_outputs", [])
    messages = state.get("messages", [])
    original_query = (
        messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    )

    specialist_lines = []
    for output in agent_outputs:
        specialist_lines.append(
            f"### {output['agent_name']} ({output['agent_id']})\n"
            f"**Confidence**: {output.get('confidence', 'N/A')}\n\n"
            f"{output['response']}"
        )
    specialist_outputs = "\n\n---\n\n".join(specialist_lines) if specialist_lines else "No specialist input received."

    user_profile = state.get("user_profile", {})
    user_context = f"User: {user_profile.get('name', 'Unknown')} ({user_profile.get('role', 'Unknown role')})"

    prompt = SYNTHESIZER_PROMPT.format(
        company_context=state.get("company_context", ""),
        knowledge_context="See specialist outputs above for sourced knowledge.",
        specialist_outputs=specialist_outputs,
        original_query=original_query,
        user_context=user_context,
    )

    all_sources = []
    for output in agent_outputs:
        all_sources.extend(output.get("sources", []))
    agents_consulted = [output["agent_id"] for output in agent_outputs]

    return prompt, all_sources, agents_consulted


def synthesize(state: CofounderState) -> dict:
    """Combine all specialist outputs into a single cofounder response.

    Args:
        state: Full graph state with ``agent_outputs``, ``messages``, and
            ``company_context``.

    Returns:
        Partial state dict with ``final_response``, ``sources_cited``, and
        ``agents_consulted``.
    """
    prompt, all_sources, agents_consulted = build_synthesis_prompt(state)

    try:
        client, model = get_sonnet_client()
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        final_response = response.content[0].text
    except Exception as e:
        logger.error("Synthesizer LLM call failed: %s", e)
        agent_outputs = state.get("agent_outputs", [])
        final_response = "I encountered an error during synthesis. Here are the raw specialist inputs:\n\n"
        for output in agent_outputs:
            final_response += f"**{output['agent_name']}**: {output['response']}\n\n"

    return {
        "final_response": final_response,
        "sources_cited": all_sources,
        "agents_consulted": agents_consulted,
    }
