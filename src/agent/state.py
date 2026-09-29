"""State schema for the multi-agent cofounder graph.

Defines the TypedDict contracts that flow through the LangGraph pipeline:

- ``CofounderState`` -- the top-level graph state shared across all nodes.
- ``AgentClassification`` -- routing decision produced by the classifier.
- ``AgentOutput`` -- structured response returned by each specialist node.
- ``AgentInput`` -- the payload dispatched to specialist nodes via ``Send``.

The ``agent_outputs`` field uses an ``operator.add`` reducer so that parallel
specialist nodes can independently append their results without overwriting
each other.
"""

import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentClassification(TypedDict):
    """Routing decision for a single specialist agent.

    Produced by the classifier node and consumed when building ``Send`` objects.

    Args:
        agent_id: Registry key of the target specialist (e.g. ``"gtm"``).
        sub_question: Targeted sub-question tailored to the specialist's domain.
        relevance: ``"primary"`` for the main agent, ``"secondary"`` for supporting.
        source_hints: Methodology IDs the classifier detected for this agent.
    """

    agent_id: str
    sub_question: str
    relevance: str  # "primary" | "secondary"
    source_hints: list[str]


class AgentOutput(TypedDict):
    """Structured response returned by a specialist node.

    Multiple ``AgentOutput`` dicts accumulate in ``CofounderState.agent_outputs``
    via the ``operator.add`` reducer.

    Args:
        agent_id: Registry key of the specialist that produced this output.
        agent_name: Human-readable name (e.g. ``"GTM Strategist"``).
        response: The specialist's full text response.
        sources: List of source metadata dicts from RAG retrieval.
        confidence: Self-assessed confidence level: ``"high"``, ``"medium"``, or ``"low"``.
    """

    agent_id: str
    agent_name: str
    response: str
    sources: list[dict]
    confidence: str  # "high" | "medium" | "low"


class AgentInput(TypedDict):
    """Payload dispatched to a specialist node via LangGraph ``Send``.

    Created by ``classify_and_route`` and received by each specialist node function.

    Args:
        query: The targeted sub-question for this specialist.
        original_query: The user's original unmodified message.
        company_context: Enriched company context string.
        user_id: Authenticated user identifier.
        source_hints: Optional source names/methodologies the classifier detected
            that the specialist should prioritize during RAG retrieval.
    """

    query: str
    original_query: str
    company_context: str
    user_id: str
    source_hints: list[str]


class CofounderState(TypedDict):
    """Top-level state for the multi-agent cofounder graph.

    This TypedDict flows through all graph nodes. Each node reads the fields
    it needs and returns a partial dict to update specific fields.

    Sections:
        Input: User message, identity, and session tracking.
        Context: Business context loaded from DB by ``context_loader``.
        Routing: Agent classifications produced by ``classify_and_route``.
        Agent results: Specialist outputs accumulated via ``operator.add`` reducer.
        Final output: Synthesized response and metadata.
        Learning extraction: Structured learnings extracted post-synthesis.
    """

    # Input
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    session_id: str

    # Context (loaded by context_loader)
    user_profile: dict
    active_hypotheses: list[dict]
    problem_statement: dict | None
    recent_metrics: dict | None
    recent_decisions: list[dict]
    recent_timeline: list[dict]
    company_context: str

    # Agent results (accumulated via operator.add reducer)
    agent_outputs: Annotated[list[AgentOutput], operator.add]

    # Final output
    final_response: str
    sources_cited: list[dict]
    agents_consulted: list[str]

    # Learning extraction
    extracted_learnings: dict | None
