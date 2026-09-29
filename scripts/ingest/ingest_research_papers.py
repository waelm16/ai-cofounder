"""Ingest research papers from ``data/knowledge/research_papers/`` into the unified knowledge store.

Currently focuses on KPMG Venture Pulse quarterly reports.  Each paper is
chunked, embedded, and stored with ``source_type="research_paper"`` plus
publisher and date metadata when available.

Usage::

    python scripts/ingest/ingest_research_papers.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import embed_texts
from src.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PAPERS_DIR = Path("data/knowledge/research_papers")
SOURCE_TYPE = "research_paper"

#: Filename-stem to metadata mapping for known research papers.
PAPER_METADATA = {
    "kpmg-venture-pulse-q1-2025": {
        "title": "KPMG Venture Pulse Q1 2025",
        "publisher": "KPMG",
        "date": "Q1 2025",
    },
    "kpmg-venture-pulse-q2-2025": {
        "title": "KPMG Venture Pulse Q2 2025",
        "publisher": "KPMG",
        "date": "Q2 2025",
    },
    "kpmg-venture-pulse-q3-2025": {
        "title": "KPMG Venture Pulse Q3 2025",
        "publisher": "KPMG",
        "date": "Q3 2025",
    },
    "kpmg-venture-pulse-q4-2025": {
        "title": "KPMG Venture Pulse Q4 2025",
        "publisher": "KPMG",
        "date": "Q4 2025",
    },
    "Q4 Pulse Survey Deck FINAL": {
        "title": "Q4 Pulse Survey Deck",
        "publisher": "KPMG",
        "date": "Q4 2025",
    },
}


def main():
    """Discover, chunk, embed, and store all new research papers."""
    if not PAPERS_DIR.exists():
        logger.warning(f"Directory {PAPERS_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(PAPERS_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        if store.has_source(file_path.stem):
            continue

        extra_metadata = PAPER_METADATA.get(file_path.stem)
        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All research papers already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from research papers...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
