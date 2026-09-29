"""Node functions for the multi-agent cofounder pipeline.

The first four are part of the LangGraph ``StateGraph``, in this order:

1. ``context_loader`` -- hydrates state with DB context.
2. ``classify_and_route`` -- routes to specialists via ``Send``.
3. ``make_agent_node`` -- factory that creates specialist node functions.
4. ``synthesize`` -- merges specialist outputs into a unified response.

``extract_learnings`` is not a graph node. The API layer calls it in a
background thread after the response has been saved.
"""

from src.agent.nodes.classifier import classify_and_route
from src.agent.nodes.context_loader import context_loader
from src.agent.nodes.learning_extractor import extract_learnings
from src.agent.nodes.specialist import make_agent_node
from src.agent.nodes.synthesizer import synthesize

__all__ = [
    "classify_and_route",
    "context_loader",
    "extract_learnings",
    "make_agent_node",
    "synthesize",
]
