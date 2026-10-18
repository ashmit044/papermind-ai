"""Unit tests for RAGEngine."""

import pytest

from papermind.parser import TextChunk
from papermind.rag_engine import RAGEngine, SourceRef


class StubVectorIndex:
    """Return fixed matches without loading a sentence-transformer model."""

    def __init__(self, matches):
        self.matches = matches

    def search(self, _query, top_k=4):
        return self.matches[:top_k]


def test_rag_engine_query():
    chunks = [
        TextChunk(
            chunk_id=0,
            text="FlashMind reduces spatial complexity to O(N log N) with 3.4x speedup.",
            page_num=1,
            section_name="Abstract",
            char_count=70,
        ),
        TextChunk(
            chunk_id=1,
            text="ResNet uses residual skip connections for deep gradient flow.",
            page_num=2,
            section_name="Related Work",
            char_count=60,
        ),
    ]
    index_mgr = StubVectorIndex(
        [(chunk, 0.9 - index * 0.1) for index, chunk in enumerate(chunks)]
    )
    rag = RAGEngine(index_mgr, llm_provider="offline")
    resp = rag.query("What complexity reduction does FlashMind achieve?")

    assert resp.query == "What complexity reduction does FlashMind achieve?"
    assert "FlashMind" in resp.answer or "O(N log N)" in resp.answer
    assert "[S1]" in resp.answer
    assert len(resp.sources) > 0
    assert resp.sources[0].source_id == 1
    assert resp.sources[0].page_num == 1


def test_rag_engine_rejects_empty_questions_and_invalid_top_k():
    rag = RAGEngine(vector_index=object(), llm_provider="offline")

    for question, top_k in [
        ("  ", 3),
        ("question", 0),
        ("question", -1),
        ("question", True),
    ]:
        with pytest.raises(ValueError):
            rag.query(question, top_k=top_k)


def test_rag_engine_flags_unknown_or_missing_inline_citations():
    source = SourceRef(
        source_id=1,
        chunk_id=0,
        page_num=1,
        section_name="Abstract",
        similarity_score=0.9,
        snippet="Evidence",
    )

    unknown = RAGEngine._check_answer_citations("Claim [S9]", [source])
    uncited = RAGEngine._check_answer_citations("Claim without a citation", [source])

    assert "[unknown source]" in unknown
    assert "verifiable inline citation" in uncited
