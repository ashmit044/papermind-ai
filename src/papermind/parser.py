"""PDF parser and document sectioning engine for PaperMind AI.

Extracts text, metadata, and section boundaries from academic papers using PyMuPDF.
"""

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF
from pydantic import BaseModel

from papermind.config import config
from papermind.utils.logger import logger


def document_fingerprint(content: bytes) -> str:
    """Return a stable identifier for document bytes, independent of its filename."""
    return hashlib.sha256(content).hexdigest()


class TextChunk(BaseModel):
    """Represents a text chunk with positional metadata."""

    chunk_id: int
    text: str
    page_num: int
    section_name: str
    char_count: int


class DocumentMetadata(BaseModel):
    """Metadata extracted from research paper."""

    title: str
    author: str
    page_count: int
    sections: list[str]


class PaperParser:
    """Extracts, cleans, and partitions PDF research papers into structured chunks."""

    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_pdf(
        self, pdf_path: str | Path, fallback_title: str | None = None
    ) -> tuple[DocumentMetadata, list[TextChunk]]:
        """Parse a PDF file, extract text sections, and return metadata and text chunks."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        logger.info(f"Extracting document text from PDF: {pdf_path.name}")
        with fitz.open(pdf_path) as doc:
            return self._parse_open_document(
                doc, fallback_title or pdf_path.stem.replace("_", " ").title()
            )

    def parse_pdf_bytes(
        self, content: bytes, fallback_title: str = "Uploaded Document"
    ) -> tuple[DocumentMetadata, list[TextChunk]]:
        """Parse uploaded PDF bytes in memory without persisting the original file."""
        if not content:
            raise ValueError("PDF content must not be empty.")

        try:
            document = fitz.open(stream=content, filetype="pdf")
        except fitz.FileDataError as error:
            raise ValueError("Uploaded content is not a valid PDF.") from error

        logger.info("Extracting uploaded PDF content in memory")
        with document:
            return self._parse_open_document(document, fallback_title)

    def _parse_open_document(
        self, doc: fitz.Document, fallback_title: str
    ) -> tuple[DocumentMetadata, list[TextChunk]]:
        """Extract metadata and chunks from an already-open PDF document."""

        metadata_dict = doc.metadata or {}
        title = metadata_dict.get("title") or fallback_title
        author = metadata_dict.get("author") or "Unknown Author"

        raw_pages: list[tuple[int, str]] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            cleaned_text = self._clean_page_text(text)
            if cleaned_text:
                raw_pages.append((page_idx + 1, cleaned_text))

        # Extract structured sections
        chunks, detected_sections = self._chunk_document(raw_pages, title)

        doc_meta = DocumentMetadata(
            title=title,
            author=author,
            page_count=len(raw_pages),
            sections=detected_sections if detected_sections else ["Full Document"],
        )

        logger.info(
            f"Successfully parsed '{title}' ({doc_meta.page_count} pages, {len(chunks)} chunks)"
        )
        return doc_meta, chunks

    def parse_raw_text(
        self, text: str, title: str = "Raw Document"
    ) -> tuple[DocumentMetadata, list[TextChunk]]:
        """Parse raw text string directly without PDF input (useful for testing & text files)."""
        cleaned = self._clean_page_text(text)
        raw_pages = [(1, cleaned)]
        chunks, detected_sections = self._chunk_document(raw_pages, title)

        doc_meta = DocumentMetadata(
            title=title,
            author="User Input",
            page_count=1,
            sections=detected_sections if detected_sections else ["Full Document"],
        )
        return doc_meta, chunks

    def _clean_page_text(self, text: str) -> str:
        """Remove redundant whitespace, headers, footers, and line numbers."""
        # Normalize each line without losing meaningful section boundaries.
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        # Remove common academic header page indicators like "Page 1 of 12"
        text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
        return text.strip()

    def _chunk_document(
        self, raw_pages: list[tuple[int, str]], default_title: str
    ) -> tuple[list[TextChunk], list[str]]:
        """Split page texts into overlapping chunks with section context."""
        chunks: list[TextChunk] = []
        chunk_id = 0
        detected_sections: set[str] = set()

        current_section = "Abstract / Intro"

        # Regex for detecting common academic headings
        section_pattern = re.compile(
            r"^(?:\d+[\.\s]+)?(Abstract|Introduction|Related Work|Methodology|Methods|System Model|Experiments|Results|Discussion|Conclusion|References)",
            re.IGNORECASE,
        )

        for page_num, page_text in raw_pages:
            lines = page_text.split("\n")
            current_paragraph = ""

            for line in lines:
                match = section_pattern.search(line.strip())
                if match:
                    # Finish the preceding section before changing its label;
                    # otherwise a short section can be incorrectly attributed
                    # to the next heading on the same page.
                    if current_paragraph.strip():
                        chunk_text = current_paragraph.strip()
                        chunks.append(
                            TextChunk(
                                chunk_id=chunk_id,
                                text=chunk_text,
                                page_num=page_num,
                                section_name=current_section,
                                char_count=len(chunk_text),
                            )
                        )
                        chunk_id += 1
                        current_paragraph = ""

                    header = match.group(1).title()
                    current_section = header
                    detected_sections.add(header)

                current_paragraph += " " + line.strip()

                if len(current_paragraph) >= self.chunk_size:
                    chunk = TextChunk(
                        chunk_id=chunk_id,
                        text=current_paragraph.strip(),
                        page_num=page_num,
                        section_name=current_section,
                        char_count=len(current_paragraph.strip()),
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    # Keep overlap
                    overlap_chars = self.chunk_overlap
                    current_paragraph = (
                        current_paragraph[-overlap_chars:]
                        if len(current_paragraph) > overlap_chars
                        else ""
                    )

            if current_paragraph.strip():
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=current_paragraph.strip(),
                    page_num=page_num,
                    section_name=current_section,
                    char_count=len(current_paragraph.strip()),
                )
                chunks.append(chunk)
                chunk_id += 1

        return chunks, list(detected_sections)
