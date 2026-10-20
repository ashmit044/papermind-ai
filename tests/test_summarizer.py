"""Unit tests for PaperSummarizer."""

from papermind.parser import DocumentMetadata, TextChunk
from papermind.summarizer import PaperSummarizer


def test_summarizer_reports_missing_claims_instead_of_filling_them():
    metadata = DocumentMetadata(
        title="A Paper Without Explicit Findings",
        author="Unknown Author",
        page_count=1,
        sections=["Full Document"],
    )
    chunks = [
        TextChunk(
            chunk_id=0,
            text="This document contains a neutral description of a research topic.",
            page_num=1,
            section_name="Full Document",
            char_count=65,
        )
    ]

    digest = PaperSummarizer(metadata, chunks).generate_digest()

    assert digest["methodology"] == "Not identified in the extracted text."
    assert digest["limitations"] == "Not identified in the extracted text."
    assert digest["key_insights"] == [
        "No explicit contribution statements were identified by the current "
        "keyword-based extractor. Review the extracted paper text directly."
    ]


def test_paper_summarizer_digest():
    meta = DocumentMetadata(
        title="Attention Is All You Need",
        author="Vaswani et al.",
        page_count=10,
        sections=["Abstract", "Methodology", "Experiments"],
    )
    chunks = [
        TextChunk(
            chunk_id=0,
            text="Abstract: We propose the Transformer, a model architecture based solely on attention mechanisms.",
            page_num=1,
            section_name="Abstract",
            char_count=100,
        ),
        TextChunk(
            chunk_id=1,
            text="Methodology: Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            page_num=3,
            section_name="Methodology",
            char_count=120,
        ),
    ]

    summarizer = PaperSummarizer(meta, chunks)
    digest = summarizer.generate_digest()

    assert digest["title"] == "Attention Is All You Need"
    assert digest["author"] == "Vaswani et al."
    assert len(digest["overview"]) > 0
    assert len(digest["key_insights"]) > 0
    assert digest["limitations"] == "Not identified in the extracted text."
