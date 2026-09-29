"""Ingest call transcripts from ``data/knowledge/call_transcripts/`` into the unified knowledge store.

When possible, enriches chunk metadata by looking up the corresponding
``CustomerCall`` record in the database (matched by filename via ``ILIKE``).
This allows retrieval results to display customer name, company, call type,
and date alongside the transcript content.

Usage::

    python scripts/ingest/ingest_call_transcript.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import embed_texts
from src.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = Path("data/knowledge/call_transcripts")
SOURCE_TYPE = "customer_call"


def _lookup_call_metadata(source_name: str) -> dict | None:
    """Look up a CustomerCall record in the DB and return enrichment metadata.

    Searches for a ``CustomerCall`` whose ``customer_name`` contains
    *source_name* (case-insensitive), returning the most recent match.

    Args:
        source_name: Filename stem used as a fuzzy match against customer names.

    Returns:
        Dict with ``customer_name``, ``customer_company``, ``call_type``, and
        ``date`` fields (only keys with non-None values), or None if no match
        is found or the database is unavailable.
    """
    try:
        from src.memory.database import SessionLocal, CustomerCall

        session = SessionLocal()
        try:
            # Try matching by customer_name (often the file is named after the customer)
            call = (
                session.query(CustomerCall)
                .filter(CustomerCall.customer_name.ilike(f"%{source_name}%"))
                .order_by(CustomerCall.call_date.desc())
                .first()
            )
            if call is None:
                return None

            meta: dict = {}
            if call.customer_name:
                meta["customer_name"] = call.customer_name
            if call.customer_company:
                meta["customer_company"] = call.customer_company
            if call.call_type:
                meta["call_type"] = call.call_type
            if call.call_date:
                meta["date"] = str(call.call_date.date())
            return meta or None
        finally:
            session.close()
    except Exception as exc:
        logger.debug(f"Could not look up call metadata for '{source_name}': {exc}")
        return None


def main():
    """Discover, chunk, embed, and store all new call transcript files."""
    if not TRANSCRIPTS_DIR.exists():
        logger.warning(f"Directory {TRANSCRIPTS_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(TRANSCRIPTS_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".txt", ".md"):
            continue
        if store.has_source(file_path.stem):
            continue

        extra_metadata = _lookup_call_metadata(file_path.stem)
        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All call transcripts already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from call transcripts...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
