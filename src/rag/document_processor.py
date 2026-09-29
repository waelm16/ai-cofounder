"""Document processing: file reading and chunking with sentence-aware overlap.

Supports PDF, EPUB, HTML, TXT, and Markdown files.  Text is split into
overlapping chunks whose boundaries are snapped to the nearest sentence ending,
producing cleaner context windows for embedding and retrieval.

Classes:
    DocumentChunk: Lightweight container for a text chunk and its metadata.
    DocumentProcessor: Reads files in various formats and produces overlapping,
        sentence-boundary-aware chunks ready for embedding.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentChunk:
    """A single text chunk extracted from a source document.

    Attributes:
        content: The chunk text.
        metadata: Dict of metadata (source_type, source_name, chunk_index, etc.).
    """

    content: str
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """Reads documents in multiple formats and splits them into overlapping chunks.

    Chunk boundaries are snapped to sentence endings (``". "``, ``"? "``, ``"! "``,
    or paragraph breaks) so that individual chunks are more coherent for embedding.

    Args:
        chunk_size: Target character count per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_file(
        self,
        file_path: str | Path,
        source_type: str,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Read a file and split it into overlapping chunks with metadata.

        The file format is detected from its extension. PDF page-break positions
        are tracked so that chunk metadata can include a ``page_number`` field.

        Args:
            file_path: Path to the source file.
            source_type: Label stored in each chunk's metadata (e.g. ``"book"``).
            extra_metadata: Additional key-value pairs merged into chunk metadata.

        Returns:
            List of :class:`DocumentChunk` instances, or an empty list if the
            file is empty or unreadable.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            text, page_breaks = self._read_pdf(file_path)
        elif ext == ".epub":
            text = self._read_epub(file_path)
            page_breaks = []
        elif ext in (".html", ".htm"):
            text = self._read_html(file_path)
            page_breaks = []
        else:
            text = self._read_text(file_path)
            page_breaks = []

        if not text or not text.strip():
            return []

        return self._chunk_text(
            text,
            source_type=source_type,
            source_name=file_path.stem,
            page_breaks=page_breaks,
            extra_metadata=extra_metadata,
        )

    def process_text(
        self,
        text: str,
        source_type: str,
        source_name: str,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Chunk pre-loaded text that does not originate from a file on disk.

        Useful for content built programmatically (e.g. call transcripts
        assembled from database records).

        Args:
            text: The raw text to chunk.
            source_type: Label stored in each chunk's metadata.
            source_name: Identifier for the source (used for deduplication).
            extra_metadata: Additional key-value pairs merged into chunk metadata.

        Returns:
            List of :class:`DocumentChunk` instances.
        """
        if not text or not text.strip():
            return []
        return self._chunk_text(
            text,
            source_type=source_type,
            source_name=source_name,
            extra_metadata=extra_metadata,
        )

    def process_directory(
        self,
        directory: str | Path,
        source_type: str,
        extensions: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """Process all matching files in a directory.

        Files are processed in sorted order. Hidden files and ``.gitkeep``
        sentinels are skipped.

        Args:
            directory: Path to the directory to scan.
            source_type: Label applied to every chunk's metadata.
            extensions: Allowed file extensions (with leading dot). Defaults to
                common document formats.

        Returns:
            Combined list of :class:`DocumentChunk` from all processed files.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        if extensions is None:
            extensions = [".pdf", ".epub", ".txt", ".md", ".html", ".htm"]

        chunks = []
        for file_path in sorted(directory.iterdir()):
            if file_path.name.startswith(".") or file_path.name == ".gitkeep":
                continue
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                chunks.extend(self.process_file(file_path, source_type))
        return chunks

    def _read_pdf(self, file_path: Path) -> tuple[str, list[int]]:
        """Read PDF, return full text and character positions of page breaks."""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(file_path))
        text_parts = []
        page_breaks = []
        offset = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            offset += len(page_text)
            page_breaks.append(offset)
        return "\n".join(text_parts), page_breaks

    def _read_epub(self, file_path: Path) -> str:
        """Extract text from EPUB by reading each HTML document in spine order."""
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(file_path), options={"ignore_ncx": True})
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _read_html(self, file_path: Path) -> str:
        """Extract text from HTML."""
        from bs4 import BeautifulSoup

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n", strip=True)

    def _read_text(self, file_path: Path) -> str:
        """Read plain text or markdown."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @staticmethod
    def _find_break_point(text: str, target_pos: int, window: int = 100) -> int:
        """Snap *target_pos* to the nearest sentence boundary within *window*.

        Looks for `. `, `? `, `! `, or `\\n\\n` in the region
        [target_pos - window, target_pos + window].  Returns the position
        **after** the delimiter so the chunk ends at a clean sentence.
        Falls back to nearest whitespace, then to target_pos itself.
        """
        if target_pos >= len(text):
            return len(text)

        search_start = max(0, target_pos - window)
        search_end = min(len(text), target_pos + window)
        region = text[search_start:search_end]

        # Sentence-ending delimiters (position is right after the delimiter)
        best = None
        best_dist = window + 1
        for delim in (". ", "? ", "! ", "\n\n"):
            pos = 0
            while True:
                idx = region.find(delim, pos)
                if idx == -1:
                    break
                abs_pos = search_start + idx + len(delim)
                dist = abs(abs_pos - target_pos)
                if dist < best_dist:
                    best_dist = dist
                    best = abs_pos
                pos = idx + 1

        if best is not None:
            return best

        # Fallback: nearest space
        for offset in range(0, window):
            for candidate in (target_pos + offset, target_pos - offset):
                if 0 <= candidate < len(text) and text[candidate] == " ":
                    return candidate + 1  # include the space in previous chunk
        return target_pos

    def _chunk_text(
        self,
        text: str,
        source_type: str,
        source_name: str,
        page_breaks: list[int] | None = None,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Split text into overlapping, sentence-boundary-aware chunks.

        The algorithm advances by ``chunk_size`` characters, then snaps the
        end position to the nearest sentence boundary using
        :meth:`_find_break_point`.  Each subsequent chunk starts
        ``chunk_overlap`` characters before the previous chunk's end to
        preserve context across boundaries.

        Args:
            text: Full document text.
            source_type: Label for chunk metadata.
            source_name: Identifier for deduplication.
            page_breaks: Character offsets of PDF page boundaries (used to
                compute ``page_number`` metadata).
            extra_metadata: Additional fields merged into every chunk's metadata.

        Returns:
            List of :class:`DocumentChunk` instances.
        """
        if not page_breaks:
            page_breaks = []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            raw_end = start + self.chunk_size

            # Snap end to a sentence boundary (unless we're at the tail)
            if raw_end < len(text):
                end = self._find_break_point(text, raw_end)
            else:
                end = len(text)

            chunk_text = text[start:end]

            if not chunk_text.strip():
                start = end - self.chunk_overlap
                if start <= (chunks[-1].metadata["_start_offset"] if chunks else 0):
                    break
                continue

            page_number = None
            if page_breaks:
                page_number = 1
                for i, pb in enumerate(page_breaks):
                    if start < pb:
                        page_number = i + 1
                        break
                else:
                    page_number = len(page_breaks)

            # _start_offset is an internal field used to detect infinite loops
            # in empty-chunk handling; it is stripped before returning.
            metadata = {
                "source_type": source_type,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "_start_offset": start,
            }
            if page_number is not None:
                metadata["page_number"] = page_number
            if extra_metadata:
                metadata.update(extra_metadata)

            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))
            chunk_index += 1

            if end >= len(text):
                break
            start = end - self.chunk_overlap

        # Remove internal tracking field
        for chunk in chunks:
            chunk.metadata.pop("_start_offset", None)

        return chunks
