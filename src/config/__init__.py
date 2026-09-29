"""Application configuration: settings, LLM client factories, and constants.

Modules:
    settings: Pydantic BaseSettings singleton loaded from env / .env file.
    llm_config: Factory functions for tiered Anthropic Claude clients (Haiku / Sonnet).
"""
