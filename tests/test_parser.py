"""Unit tests for PaperParser."""

import pytest
from papermind.parser import PaperParser


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
