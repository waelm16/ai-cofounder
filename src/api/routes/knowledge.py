"""Knowledge search API route.

Exposes the RAG retriever as a REST endpoint so users and the front-end
can query the unified FAISS knowledge base directly.

Endpoints:
    GET /api/knowledge/search — semantic search over the knowledge base
"""

from fastapi import APIRouter, Depends, Query

from src.api.auth import get_current_user
from src.api.schemas import KnowledgeSearchResponse, KnowledgeSearchResult
from src.memory.database import User
from src.rag.retriever import format_results_for_citation, retrieve

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/search", response_model=KnowledgeSearchResponse)
def search_knowledge(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
    source_type: str | None = Query(None, description="Filter by source type"),
    current_user: User = Depends(get_current_user),
):
    """Perform a semantic search over the unified FAISS knowledge base.

    Returns the top-k most relevant chunks with metadata and a
    pre-formatted citation string suitable for LLM context injection.

    Args:
        q: Natural-language search query.
        top_k: Number of results to return (1--20, default 5).
        source_type: Optional filter (e.g. ``"book"``, ``"pg_essay"``).

    Returns:
        KnowledgeSearchResponse: Matching chunks, scores, and formatted
            citation text.
    """
    results = retrieve(q, top_k=top_k, source_type=source_type)
    formatted = format_results_for_citation(results)
    return KnowledgeSearchResponse(
        query=q,
        results=[KnowledgeSearchResult(**r) for r in results],
        formatted=formatted,
    )
