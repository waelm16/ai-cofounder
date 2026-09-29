"""Ingest books (PDF/EPUB/TXT/MD) from ``data/knowledge/books/`` into the unified knowledge store.

Reads each book file, chunks it via :class:`~src.rag.document_processor.DocumentProcessor`,
embeds all chunks with bge-small-en-v1.5, and writes them into the ``unified_knowledge``
FAISS index.  Books already present in the store (matched by filename stem) are skipped.

Usage::

    python scripts/ingest/ingest_books.py
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

BOOKS_DIR = Path("data/knowledge/books")
SOURCE_TYPE = "book"

#: Map filename substrings to rich metadata (title, author).
#: Keys are checked with ``in stem`` so partial matches work for long filenames.
BOOK_METADATA = {
    "100M_Offers": {"title": "$100M Offers", "author": "Alex Hormozi"},
    "The_Hard_Thing_About_Hard_Things": {
        "title": "The Hard Thing About Hard Things",
        "author": "Ben Horowitz",
    },
    "The_Lean_Startup": {"title": "The Lean Startup", "author": "Eric Ries"},
    "The_Mom_Test": {"title": "The Mom Test", "author": "Rob Fitzpatrick"},
}


def _lookup_metadata(stem: str) -> dict:
    """Find matching book metadata by checking if any key is a substring of the stem."""
    for key, meta in BOOK_METADATA.items():
        if key in stem:
            return dict(meta)
    return {}


def main():
    """Discover, chunk, embed, and store all new books in the knowledge directory."""
    if not BOOKS_DIR.exists():
        logger.warning(f"Directory {BOOKS_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(BOOKS_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".pdf", ".epub", ".txt", ".md"):
            continue
        if store.has_source(file_path.stem):
            continue

        extra_metadata = _lookup_metadata(file_path.stem)
        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata or None)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All books already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from books...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
