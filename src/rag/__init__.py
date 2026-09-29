"""RAG (Retrieval-Augmented Generation) pipeline for the AI Cofounder system.

Provides document ingestion, embedding, vector storage (FAISS), and multi-source
retrieval with citation formatting. The unified knowledge store indexes books,
Paul Graham essays, YC content, research papers, and customer call transcripts.
Per-agent specialist stores are managed separately.

Key public API:
    - retrieve: Embed a query and search the unified knowledge store.
    - retrieve_multi_source: Balanced retrieval across multiple source types.
    - format_results_for_citation: Format search results for LLM prompt injection.
"""

from src.rag.retriever import (
    format_results_for_citation,
    retrieve,
    retrieve_multi_source,
)

__all__ = ["retrieve", "retrieve_multi_source", "format_results_for_citation"]
