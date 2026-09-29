"""Ingest Y Combinator content from ``data/knowledge/yc_content/`` into the unified knowledge store.

Processes Startup School transcripts and guides.  Filenames are converted from
slug format (e.g. ``how-to-build-an-mvp-startup-school``) to human-readable
titles.  Source URLs are extracted from file headers when available.

Usage::

    python scripts/ingest/ingest_yc_content.py
"""

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rag.document_processor import DocumentProcessor
from src.rag.embeddings import embed_texts
from src.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

YC_DIR = Path("data/knowledge/yc_content")
SOURCE_TYPE = "yc_content"


def _title_from_slug(slug: str) -> str:
    """Convert a filename slug to a human-readable title.

    Strips the trailing ``-startup-school`` suffix and title-cases the
    remaining hyphen-separated words.

    Args:
        slug: Filename stem (e.g. ``"how-to-build-an-mvp-startup-school"``).

    Returns:
        Title-cased string (e.g. ``"How To Build An Mvp"``).
    """
    cleaned = re.sub(r"-startup-school$", "", slug)
    return cleaned.replace("-", " ").title()


def _extract_url(file_path: Path) -> str | None:
    """Extract a source URL from the file header, if present.

    Looks for a line starting with ``http`` or ``Source: http`` near the
    top of the file (skipping blank lines and markdown headings).

    Args:
        file_path: Path to the content file.

    Returns:
        The URL string, or None if no URL is found.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.lower().startswith("source:"):
                url = stripped.split(":", 1)[1].strip()
                if url.startswith("http"):
                    return url
            if stripped.startswith("http"):
                return stripped
            break
    return None


def main():
    """Discover, chunk, embed, and store all new YC content files."""
    if not YC_DIR.exists():
        logger.warning(f"Directory {YC_DIR} does not exist, nothing to ingest.")
        return

    store = VectorStore()
    processor = DocumentProcessor()

    all_chunks = []
    for file_path in sorted(YC_DIR.iterdir()):
        if file_path.name.startswith(".") or file_path.name == ".gitkeep":
            continue
        if not file_path.is_file() or file_path.suffix.lower() not in (".txt", ".md", ".html", ".htm", ".pdf"):
            continue
        if store.has_source(file_path.stem):
            continue

        title = _title_from_slug(file_path.stem)
        url = _extract_url(file_path)
        extra_metadata = {"title": title, "series": "YC Startup School"}
        if url:
            extra_metadata["url"] = url

        chunks = processor.process_file(file_path, SOURCE_TYPE, extra_metadata=extra_metadata)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.info("All YC content already ingested, nothing to do.")
        return

    logger.info(f"Embedding {len(all_chunks)} chunks from YC content...")
    texts = [c.content for c in all_chunks]
    embeddings = embed_texts(texts)
    metadata = [c.metadata for c in all_chunks]

    added = store.add_documents(texts, embeddings, metadata)
    logger.info(f"Added {added} chunks. Store size: {store.size}")


if __name__ == "__main__":
    main()
