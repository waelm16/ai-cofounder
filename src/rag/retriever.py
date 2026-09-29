"""High-level retrieval interface for the RAG pipeline.

Provides convenience functions that combine embedding and vector search in a
single call, plus multi-source balanced retrieval and citation formatting for
LLM prompt injection.  The unified knowledge store is cached as a singleton
per directory to avoid re-loading the FAISS index on every query.
"""

from src.config.settings import get_settings
from src.rag.embeddings import embed_query
from src.rag.vector_store import VectorStore

_store_cache: dict[str, VectorStore] = {}


def get_knowledge_store(store_dir: str | None = None) -> VectorStore:
    """Get or create the unified knowledge vector store (singleton per directory).

    Args:
        store_dir: Directory containing the FAISS index files. Defaults to the
            ``VECTOR_STORE_DIR`` setting.

    Returns:
        A cached VectorStore instance for ``unified_knowledge``.
    """
    store_dir = store_dir or get_settings().vector_store_dir
    if store_dir not in _store_cache:
        _store_cache[store_dir] = VectorStore(
            name="unified_knowledge", store_dir=store_dir
        )
    return _store_cache[store_dir]


def retrieve(
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    store_dir: str | None = None,
) -> list[dict]:
    """Embed a query and search the unified knowledge store.

    Args:
        query: Natural-language query string.
        top_k: Maximum number of results to return.
        source_type: Optional filter (e.g. ``"book"``, ``"pg_essay"``).
        store_dir: Directory containing FAISS index files. Defaults to the
            ``VECTOR_STORE_DIR`` setting.

    Returns:
        List of result dicts with ``content``, ``score``, and metadata fields.
    """
    store = get_knowledge_store(store_dir)
    if store.size == 0:
        return []
    query_embedding = embed_query(query)
    return store.search(query_embedding, top_k=top_k, source_type=source_type)


def retrieve_multi_source(
    query: str,
    top_k_per_source: int = 3,
    source_types: list[str] | None = None,
    store_dir: str | None = None,
) -> dict[str, list[dict]]:
    """Retrieve results from each source type independently for balanced coverage.

    Ensures the LLM prompt contains knowledge from books, essays, YC content,
    research papers, and call transcripts rather than being dominated by a
    single source type.

    Args:
        query: Natural-language query string.
        top_k_per_source: Max results per source type.
        source_types: Source types to query. Defaults to all five built-in types.
        store_dir: Directory containing FAISS index files. Defaults to the
            ``VECTOR_STORE_DIR`` setting.

    Returns:
        Dict mapping each source type to its list of result dicts.
    """
    if source_types is None:
        source_types = ["book", "pg_essay", "yc_content", "research_paper", "customer_call"]

    results = {}
    for st in source_types:
        results[st] = retrieve(
            query, top_k=top_k_per_source, source_type=st, store_dir=store_dir
        )
    return results


def _build_citation_header(index: int, r: dict) -> str:
    """Build a bracketed citation header from available metadata fields.

    Assembles a human-readable citation line like::

        [Source 1: book: "The Mom Test", by Rob Fitzpatrick, Ch. 3, p.42]

    Args:
        index: 1-based source number for display.
        r: Result dict containing metadata (title, author, chapter, etc.).

    Returns:
        A formatted citation string enclosed in square brackets.
    """
    source_type = r.get("source_type", "")
    title = r.get("title") or r.get("source_name", "unknown")

    parts = [f"Source {index}: {source_type}: \"{title}\""]

    author = r.get("author")
    if author:
        parts.append(f"by {author}")

    series = r.get("series")
    if series:
        parts.append(f"({series})")

    chapter = r.get("chapter")
    if chapter:
        parts.append(f"Ch. {chapter}")

    page = r.get("page_number")
    if page is not None:
        parts.append(f"p.{page}")

    date = r.get("date")
    if date:
        parts.append(date)

    customer_name = r.get("customer_name")
    customer_company = r.get("customer_company")
    if customer_name:
        label = f"Call with {customer_name}"
        if customer_company:
            label += f" @ {customer_company}"
        parts.append(label)

    url = r.get("url")
    if url:
        parts.append(url)

    return "[" + ", ".join(parts) + "]"


def format_results_for_citation(results: list[dict]) -> str:
    """Format search results as numbered, cited passages for LLM prompts.

    Each result is rendered with a citation header followed by its content,
    separated by blank lines. Suitable for direct injection into system or
    user messages.

    Args:
        results: List of result dicts from :func:`retrieve` or similar.

    Returns:
        Formatted multi-line string, or a fallback message if no results.
    """
    if not results:
        return "No relevant sources found."

    formatted = []
    for i, r in enumerate(results, 1):
        header = _build_citation_header(i, r)
        formatted.append(f"{header}\n{r['content']}")

    return "\n\n".join(formatted)
