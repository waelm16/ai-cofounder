"""Ingest the sample knowledge pack from ``data/knowledge/sample/`` into the unified knowledge store.

The sample pack is a handful of short original notes shipped with the repository
so the system can answer questions on first install.  Each ``.md`` / ``.txt``
file is chunked, embedded, and stored with ``source_type="sample_note"``.  The
note title is taken from the first markdown heading in the file.

Usage::

    python scripts/ingest/ingest_sample.py
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

SAMPLE_DIR = Path("data/knowledge/sample")
SOURCE_TYPE = "sample_note"


def _extract_title(file_path: Path) -> str:
    """Return the first non-empty line of the file, without markdown heading marks.

    Args:
        file_path: Path to the note.

    Returns:
        The first non-blank line, or the filename stem as a fallback.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped
    return file_path.stem


def main():
    """Discover, chunk, embed, and store all new sample notes."""
    if not SAMPLE_DIR.exists():
        logger.warning(f"Directory {SAMPLE_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(SAMPLE_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".txt", ".md"):
            continue
        if store.has_source(file_path.stem):
            continue

        extra_metadata = {"title": _extract_title(file_path)}
        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All sample notes already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from sample notes...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
