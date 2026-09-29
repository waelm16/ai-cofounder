"""LangGraph multi-agent cofounder graph -- Router pattern with parallel specialist dispatch.

Assembles the full LangGraph ``StateGraph`` using the Router pattern:

1. ``context_loader`` -- loads user profile, hypotheses, decisions, metrics from DB.
2. ``classify_and_route`` -- Haiku classifies the query and returns ``Send`` objects
   to dispatch 1-3 specialist nodes in parallel.
3. ``specialist_{agent_id}`` -- factory-generated nodes (one per registered agent)
   that retrieve from per-agent FAISS indices and reason with Sonnet.
4. ``synthesize`` -- merges all specialist outputs into a unified cofounder response.

Learning extraction runs as a background task in the API layer, not in the graph.

The ``classify_and_route`` node is wired as a conditional edge from ``context_loader``.
It returns ``Send`` objects, so LangGraph dispatches the selected specialist nodes in
parallel. All specialist edges converge on ``synthesize``.

Exports:
    graph: Full pipeline (with synthesis), for the sync /api/chat endpoint.
    graph_pre_synthesis: Stops after specialists collect, for the streaming endpoint.
        That endpoint runs synthesis itself, because ``synthesize`` calls the
        Anthropic SDK directly and its tokens are not visible to LangGraph.
"""

from langgraph.graph import END, START, StateGraph

from src.agent.agents.registry import AGENT_REGISTRY
from src.agent.nodes.classifier import classify_and_route
from src.agent.nodes.context_loader import context_loader
from src.agent.nodes.specialist import make_agent_node
from src.agent.nodes.synthesizer import synthesize
from src.agent.state import CofounderState


def _build_base(builder: StateGraph) -> None:
    """Add shared nodes and edges (context_loader, classifier, specialists)."""
    builder.add_node("context_loader", context_loader)
    for agent_id in AGENT_REGISTRY:
        builder.add_node(f"specialist_{agent_id}", make_agent_node(agent_id))
    builder.add_edge(START, "context_loader")
    builder.add_conditional_edges("context_loader", classify_and_route)


def build_graph() -> StateGraph:
    """Full pipeline: context → classify → specialists → synthesize → END."""
    builder = StateGraph(CofounderState)
    _build_base(builder)
    builder.add_node("synthesize", synthesize)
    for agent_id in AGENT_REGISTRY:
        builder.add_edge(f"specialist_{agent_id}", "synthesize")
    builder.add_edge("synthesize", END)
    return builder


def build_pre_synthesis_graph() -> StateGraph:
    """Stops after specialists: context → classify → specialists → collect → END.

    Used by the streaming endpoint which handles synthesis separately so it
    can stream the Sonnet output token-by-token.
    """
    builder = StateGraph(CofounderState)
    _build_base(builder)
    builder.add_node("collect", lambda state: {})
    for agent_id in AGENT_REGISTRY:
        builder.add_edge(f"specialist_{agent_id}", "collect")
    builder.add_edge("collect", END)
    return builder


graph = build_graph().compile()
graph_pre_synthesis = build_pre_synthesis_graph().compile()
