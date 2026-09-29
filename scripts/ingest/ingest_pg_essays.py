"""Ingest Paul Graham essays from ``data/knowledge/paul_graham/`` into the unified knowledge store.

Each ``.txt`` / ``.md`` / ``.html`` file is chunked, embedded, and stored with
``source_type="pg_essay"`` and ``author="Paul Graham"``.  The essay title is
extracted from the first non-empty line of the file.

Usage::

    python scripts/ingest/ingest_pg_essays.py
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

PG_DIR = Path("data/knowledge/paul_graham")
SOURCE_TYPE = "pg_essay"


def _extract_title(file_path: Path) -> str:
    """Return the first non-empty line of the file as the essay title.

    Args:
        file_path: Path to the essay text file.

    Returns:
        The first non-blank line, or the filename stem as a fallback.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                return stripped
    return file_path.stem


def main():
    """Discover, chunk, embed, and store all new Paul Graham essays."""
    if not PG_DIR.exists():
        logger.warning(f"Directory {PG_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(PG_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".txt", ".md", ".html", ".htm"):
            continue
        if store.has_source(file_path.stem):
            continue

        title = _extract_title(file_path)
        extra_metadata = {"title": title, "author": "Paul Graham"}
        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All PG essays already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from PG essays...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
