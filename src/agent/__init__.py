"""Multi-agent AI cofounder system.

This package implements a LangGraph-based multi-agent pipeline. A classifier routes
each question to one to three of the 11 agents in ``AGENT_REGISTRY`` (a general
Cofounder, GTM, Finance, Marketing, Fintech, Healthcare, Product, Legal,
Data & Analytics, Business Development, and Devil's Advocate), and a synthesizer
merges their answers.

Graph flow::

    START -> context_loader -> classify_and_route -> [parallel specialists] -> synthesize -> END

Learning extraction is not a graph node. The API layer runs it in a background
thread after the response has been saved.

Exports:
    graph: The compiled LangGraph StateGraph ready for invocation.
"""

from src.agent.graph import graph

__all__ = ["graph"]
