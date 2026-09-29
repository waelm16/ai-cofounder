"""Agent registry package.

Exports:
    AGENT_REGISTRY: Dict mapping agent IDs to their ``AgentConfig`` instances.
    AgentConfig: Dataclass defining a specialist agent's identity, prompt, and knowledge store.
"""

from src.agent.agents.registry import AGENT_REGISTRY, AgentConfig

__all__ = ["AGENT_REGISTRY", "AgentConfig"]
