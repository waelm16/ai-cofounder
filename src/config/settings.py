"""Application settings loaded from environment variables and ``.env`` file.

Uses Pydantic BaseSettings for typed, validated configuration with automatic
environment-variable binding.  A single cached instance is shared application-wide
via :func:`get_settings`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the AI Cofounder application.

    Values are loaded from environment variables (or a ``.env`` file).
    Defaults are provided for local development.

    Attributes:
        anthropic_api_key: API key for Anthropic Claude models.
        database_url: SQLAlchemy connection string (SQLite for dev).
        embedding_model: HuggingFace model ID for sentence-transformers.
        haiku_model: Anthropic model ID used for cheap extraction/classification.
        sonnet_model: Anthropic model ID used for reasoning/synthesis.
        vector_store_dir: Directory for persisted FAISS indices.
        chunk_size: Target character count per document chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.
        log_level: Python logging level.
    """

    anthropic_api_key: str = "your-key-here"
    database_url: str = "sqlite:///./data/consultant.db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6"
    vector_store_dir: str = "vector_stores"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Returns:
        The shared :class:`Settings` instance (created once, cached forever).
    """
    return Settings()
