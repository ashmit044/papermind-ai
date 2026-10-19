"""Unit tests for PaperParser."""

import fitz
import pytest

from papermind.parser import PaperParser, document_fingerprint


def test_document_fingerprint_tracks_content_not_filename():
    first_document = b"paper contents"
    same_document = b"paper contents"
    different_document = b"different paper contents"

    assert document_fingerprint(first_document) == document_fingerprint(same_document)
    assert document_fingerprint(first_document) != document_fingerprint(
        different_document
    )


def test_clean_page_text():
    parser = PaperParser()
    raw_text = "Page  1   of 10\n\n\nAbstract\n  Transformers   are great.  "
    cleaned = parser._clean_page_text(raw_text)
    assert "Page 1 of 10" not in cleaned
    assert "Abstract\nTransformers are great." in cleaned


def test_parse_raw_text():
    parser = PaperParser(chunk_size=100, chunk_overlap=20)
    sample_text = """
    ABSTRACT
    This paper introduces a novel deep neural architecture for efficient graph processing.
    
    METHODOLOGY
    We formulate graph convolution using sparse matrix multiplication.
    
    RESULTS
    Our method achieves 99.2% accuracy on benchmark tasks.
    """
    meta, chunks = parser.parse_raw_text(sample_text, title="Test Paper")

    assert meta.title == "Test Paper"
    assert meta.page_count == 1
    assert len(chunks) > 0
    assert any("Abstract" in c.section_name for c in chunks)


def test_parse_pdf_uses_upload_name_when_pdf_has_no_title(tmp_path):
    pdf_path = tmp_path / "content-hash.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "A short research paper.")
    document.save(pdf_path)
    document.close()

    metadata, _chunks = PaperParser().parse_pdf(
        pdf_path, fallback_title="Uploaded Paper"
    )

    assert metadata.title == "Uploaded Paper"


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [(0, 0), (-1, 0), (100, -1), (100, 100)],
)
def test_parser_rejects_invalid_chunk_configuration(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        PaperParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
