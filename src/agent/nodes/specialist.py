"""Specialist node factory -- creates a LangGraph node function for any registered agent.

Uses the factory pattern: ``make_agent_node(agent_id)`` returns a closure that:

1. Uses the knowledge router to detect methodology matches and classifier source hints.
2. Retrieves domain-specific context from the agent's dedicated FAISS index,
   filtered to matched methodology sources when applicable.
3. Builds a prompt from ``SPECIALIST_PROMPT`` with the agent's expertise, company
   context, retrieved knowledge, and methodology application instructions.
4. Calls Sonnet for reasoning and returns a structured ``AgentOutput``.

This means adding a new specialist requires only a registry entry and knowledge
content -- no new node code.
"""

import logging
from typing import Callable

from src.agent.agents.registry import AGENT_REGISTRY
from src.agent.knowledge_router import MethodologyMatch, route_from_hints, route_query
from src.agent.prompts.specialist import SPECIALIST_PROMPT
from src.agent.state import AgentInput, AgentOutput
from src.config.llm_config import get_sonnet_client
from src.config.settings import get_settings
from src.rag.embeddings import embed_query
from src.rag.retriever import format_results_for_citation
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _retrieve_for_methodology(
    match: MethodologyMatch, query: str, settings
) -> tuple[str, list[dict]]:
    """Retrieve knowledge filtered to a specific methodology's sources.

    Args:
        match: The matched methodology with source filters and top_k.
        query: The search query to embed.
        settings: Application settings for vector store directory.

    Returns:
        Tuple of (formatted knowledge context, list of source metadata dicts).
    """
    store = VectorStore(name=match.store_name, store_dir=settings.vector_store_dir)
    if store.size == 0:
        return "", []

    query_vec = embed_query(query)
    results = store.search(query_vec, top_k=match.top_k, source_type=match.source_type)
    # Post-filter to only the specific source names for this methodology
    results = [
        r for r in results if r.get("source_name", "") in match.source_names
    ]
    if not results:
        return "", []

    knowledge_context = format_results_for_citation(results)
    sources = [
        {
            "source_type": r.get("source_type"),
            "source_name": r.get("source_name"),
            "title": r.get("title"),
            "score": r.get("score"),
        }
        for r in results
    ]
    return knowledge_context, sources


def _retrieve_default(
    store_name: str, query: str, settings, top_k: int = 5
) -> tuple[str, list[dict]]:
    """Retrieve knowledge from the agent's default FAISS store (no methodology filter).

    Args:
        store_name: FAISS index name from the agent's config.
        query: The search query to embed.
        settings: Application settings for vector store directory.
        top_k: Number of results to retrieve.

    Returns:
        Tuple of (formatted knowledge context, list of source metadata dicts).
    """
    store = VectorStore(name=store_name, store_dir=settings.vector_store_dir)
    if store.size == 0:
        return "", []

    query_vec = embed_query(query)
    results = store.search(query_vec, top_k=top_k)
    if not results:
        return "", []

    knowledge_context = format_results_for_citation(results)
    sources = [
        {
            "source_type": r.get("source_type"),
            "source_name": r.get("source_name"),
            "title": r.get("title"),
            "score": r.get("score"),
        }
        for r in results
    ]
    return knowledge_context, sources


def make_agent_node(agent_id: str) -> Callable:
    """Factory: return a LangGraph node function for the given specialist agent.

    The returned closure captures the ``AgentConfig`` for ``agent_id`` and
    performs methodology-aware RAG retrieval + Sonnet reasoning each time it
    is invoked.

    Args:
        agent_id: Registry key (must exist in ``AGENT_REGISTRY``).

    Returns:
        A callable ``(AgentInput) -> dict`` suitable for ``StateGraph.add_node``.
    """
    config = AGENT_REGISTRY[agent_id]

    def specialist_node(state: AgentInput) -> dict:
        """Process a routed sub-question and return AgentOutput.

        Args:
            state: ``AgentInput`` with the sub-question, original query, and context.

        Returns:
            Dict with ``agent_outputs`` key containing a single-element list,
            compatible with the ``operator.add`` reducer on ``CofounderState``.
        """
        query = state["query"]
        original_query = state["original_query"]
        company_context = state["company_context"]
        source_hints = state.get("source_hints", [])

        # --- Methodology detection ---
        # 1. Try keyword-based routing from query text
        match = route_query(query, original_query)

        # 2. If no keyword match, try classifier source_hints as fallback
        if match is None and source_hints:
            hint_matches = route_from_hints(source_hints)
            if hint_matches:
                match = hint_matches[0]  # Use the first (highest priority) hint

        # --- Knowledge retrieval ---
        settings = get_settings()
        knowledge_context = "No knowledge base available for this agent."
        methodology_context = ""
        sources: list[dict] = []

        try:
            if match is not None:
                # Methodology-filtered retrieval
                knowledge_context_result, sources = _retrieve_for_methodology(
                    match, query, settings
                )
                if knowledge_context_result:
                    knowledge_context = knowledge_context_result
                else:
                    # Fallback to agent's default store if methodology filter returned nothing
                    logger.info(
                        "Methodology %s returned no results for %s, falling back to default",
                        match.methodology_id, agent_id,
                    )
                    knowledge_context_result, sources = _retrieve_default(
                        config.knowledge_store_name, query, settings
                    )
                    if knowledge_context_result:
                        knowledge_context = knowledge_context_result
                methodology_context = (
                    f"\n## Methodology: {match.methodology_id.replace('_', ' ').title()}\n"
                    f"Apply the following framework principles when answering:\n"
                    f"{match.methodology_prompt}\n\n"
                    f"IMPORTANT: Ground your answer in this methodology. "
                    f"Apply its specific principles rather than giving generic advice.\n"
                )
                logger.info(
                    "Agent %s using methodology %s", agent_id, match.methodology_id
                )
            else:
                # Default: search the agent's own knowledge store
                knowledge_context_result, sources = _retrieve_default(
                    config.knowledge_store_name, query, settings
                )
                if knowledge_context_result:
                    knowledge_context = knowledge_context_result
        except Exception as e:
            logger.warning("Knowledge retrieval failed for %s: %s", agent_id, e)

        # Build prompt
        prompt = SPECIALIST_PROMPT.format(
            agent_name=config.name,
            agent_expertise=config.description,
            company_context=company_context,
            knowledge_context=knowledge_context,
            methodology_context=methodology_context,
            sub_question=query,
            original_query=original_query,
        )

        # Call Sonnet
        try:
            client, model = get_sonnet_client()
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
        except Exception as e:
            logger.error("Specialist %s LLM call failed: %s", agent_id, e)
            response_text = (
                f"I apologize, but I encountered an error processing this query. "
                f"Error: {str(e)}"
            )

        # Return ONLY agent_outputs — critical for reducer compatibility
        return {
            "agent_outputs": [
                AgentOutput(
                    agent_id=agent_id,
                    agent_name=config.name,
                    response=response_text,
                    sources=sources,
                    confidence="medium",
                )
            ]
        }

    specialist_node.__name__ = f"specialist_{agent_id}"
    return specialist_node
