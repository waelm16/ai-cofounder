"""LLM client factory functions for tiered Anthropic Claude usage.

The system uses two model tiers:
- **Haiku**: Cheap and fast, used for classification, routing, and extraction tasks.
- **Sonnet**: More capable, used for specialist reasoning, synthesis, and advice.

Each factory function returns a ``(client, model_name)`` tuple ready for use
with the Anthropic Messages API. The Anthropic client is cached as a singleton
to avoid connection overhead on every call.
"""

from anthropic import Anthropic

from src.config.settings import get_settings

_client_cache: Anthropic | None = None


def _get_client() -> Anthropic:
    """Return a cached Anthropic client singleton."""
    global _client_cache
    if _client_cache is None:
        settings = get_settings()
        _client_cache = Anthropic(api_key=settings.anthropic_api_key)
    return _client_cache


def get_haiku_client() -> tuple[Anthropic, str]:
    """Create an Anthropic client configured for Haiku (extraction/classification).

    Returns:
        Tuple of (Anthropic client instance, Haiku model ID string).
    """
    settings = get_settings()
    return _get_client(), settings.haiku_model


def get_sonnet_client() -> tuple[Anthropic, str]:
    """Create an Anthropic client configured for Sonnet (advice/reasoning).

    Returns:
        Tuple of (Anthropic client instance, Sonnet model ID string).
    """
    settings = get_settings()
    return _get_client(), settings.sonnet_model
