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


def test_rag_engine_filters_passages_below_relevance_threshold():
    first = TextChunk(
        chunk_id=0,
        text="Transformer retrieval uses semantic similarity.",
        page_num=1,
        section_name="Methodology",
        char_count=48,
    )
    second = TextChunk(
        chunk_id=1,
        text="A separate passage with weak similarity.",
        page_num=2,
        section_name="Discussion",
        char_count=40,
    )
    index = StubVectorIndex([(first, 0.85), (second, 0.32)])

    response = RAGEngine(
        index, llm_provider="offline", min_similarity_score=0.5
    ).query("How does Transformer retrieval work?")

    assert [source.chunk_id for source in response.sources] == [0]
    assert "[S1]" in response.answer
    assert "[S2]" not in response.answer


def test_rag_engine_returns_no_sources_when_all_passages_are_below_threshold():
    chunk = TextChunk(
        chunk_id=0,
        text="The source is unrelated to this question.",
        page_num=1,
        section_name="Results",
        char_count=40,
    )
    response = RAGEngine(
        StubVectorIndex([(chunk, 0.1)]),
        llm_provider="offline",
        min_similarity_score=0.5,
    ).query("What is the orbital period of Mars?")

    assert response.sources == []
    assert "relevance threshold" in response.answer


@pytest.mark.parametrize("threshold", [-1.1, 1.1, True, "0.2"])
def test_rag_engine_rejects_invalid_similarity_threshold(threshold):
    with pytest.raises(ValueError, match="min_similarity_score"):
        RAGEngine(object(), min_similarity_score=threshold)


def test_openai_provider_without_key_does_not_silently_fall_back(monkeypatch):
    from papermind import rag_engine

    monkeypatch.setattr(rag_engine.config, "openai_api_key", None)
    engine = RAGEngine(object(), llm_provider="openai")
    answer = engine._generate_answer("question", "source context")

    assert "OPENAI_API_KEY" in answer
    assert "offline synthesizer" in answer


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
