"""FAISS vector store with JSON metadata sidecar.

Each VectorStore instance manages a single FAISS IndexFlatIP index (inner-product
similarity on L2-normalized vectors) plus a companion JSON file that stores
per-document metadata. The pair is persisted as ``{name}.faiss`` and ``{name}.json``
inside *store_dir*.

Typical usage::

    store = VectorStore(name="unified_knowledge")
    store.add_documents(texts, embeddings, metadata_list)
    results = store.search(query_embedding, top_k=5, source_type="book")
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np

from src.config.settings import get_settings


class VectorStore:
    """FAISS-backed vector store with JSON sidecar for document metadata.

    Uses ``IndexFlatIP`` (inner product) which, combined with L2-normalized
    embeddings from bge-small-en-v1.5, is equivalent to cosine similarity.
    Metadata (source type, titles, authors, etc.) is stored in a parallel JSON
    file so it can be filtered and returned alongside search results.

    Args:
        name: Logical name of the store, used as filename prefix.
        dimension: Embedding vector dimension (384 for bge-small-en-v1.5).
        store_dir: Directory where ``.faiss`` and ``.json`` files are persisted.
            Defaults to the ``VECTOR_STORE_DIR`` setting (``vector_stores``).
    """

    def __init__(
        self,
        name: str = "unified_knowledge",
        dimension: int = 384,
        store_dir: str | None = None,
    ):
        self.name = name
        self.dimension = dimension
        self.store_dir = Path(store_dir or get_settings().vector_store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.store_dir / f"{name}.faiss"
        self._meta_path = self.store_dir / f"{name}.json"

        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []
        self._load_or_create()

    def _load_or_create(self):
        """Load an existing index + metadata from disk, or create a fresh pair."""
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r") as f:
                self._metadata = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._metadata = []

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata_list: list[dict],
    ) -> int:
        """Add documents to the FAISS index and persist to disk.

        Each document's text is stored in the metadata sidecar (under the
        ``content`` key) alongside any caller-supplied metadata fields.

        Args:
            texts: Raw text content for each document chunk.
            embeddings: Pre-computed embedding vectors (one per text).
            metadata_list: Per-document metadata dicts (source_type, title, etc.).

        Returns:
            Number of documents successfully added.
        """
        if not texts:
            return 0

        vectors = np.array(embeddings, dtype=np.float32)
        self._index.add(vectors)

        for text, meta in zip(texts, metadata_list):
            entry = {**meta, "content": text}
            self._metadata.append(entry)

        self.save()
        return len(texts)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[dict]:
        """Search the index for the closest vectors to *query_embedding*.

        When *source_type* is specified, the index is over-fetched (3x) and
        results are post-filtered to return only matching documents.

        Args:
            query_embedding: Normalized query vector (384-d for bge-small-en-v1.5).
            top_k: Maximum number of results to return.
            source_type: Optional filter — only return results whose metadata
                ``source_type`` matches this value.

        Returns:
            List of dicts, each containing the original metadata fields plus
            ``content`` (text) and ``score`` (inner-product similarity).
        """
        if self._index.ntotal == 0:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)

        # Over-fetch if filtering by source_type
        fetch_k = top_k * 3 if source_type else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            if source_type and meta.get("source_type") != source_type:
                continue
            result = {**meta, "score": float(score)}
            results.append(result)
            if len(results) >= top_k:
                break

        return results

    def save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._metadata, f)

    def has_source(self, source_name: str) -> bool:
        """Check if a source has already been ingested (by ``source_name`` metadata).

        Args:
            source_name: The source identifier to look for (typically a filename stem).

        Returns:
            True if any document in the metadata has a matching ``source_name``.
        """
        return any(m.get("source_name") == source_name for m in self._metadata)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal
