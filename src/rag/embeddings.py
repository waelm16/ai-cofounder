"""Embedding utilities using sentence-transformers with a lazy-loaded singleton model.

Uses BAAI/bge-small-en-v1.5 (384-dimensional embeddings) by default.  The model
is loaded on first use and reused across all subsequent calls to avoid repeated
initialization overhead.  All embeddings are L2-normalized so that inner-product
similarity equals cosine similarity (matching the FAISS IndexFlatIP used in
:mod:`src.rag.vector_store`).
"""

import numpy as np

from src.config.settings import get_settings

_model = None


def _get_model():
    """Return the singleton SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed multiple texts into normalized vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of 384 floats).
    """
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string into a normalized vector.

    Args:
        text: The query text to embed.

    Returns:
        A single embedding vector (list of 384 floats).
    """
    model = _get_model()
    embedding = model.encode([text], normalize_embeddings=True)
    return embedding[0].tolist()


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the configured model.

    Returns:
        384 (the output dimension of bge-small-en-v1.5).
    """
    return 384
