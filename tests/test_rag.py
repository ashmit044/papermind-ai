"""Unit tests for RAGEngine."""

import pytest
from papermind.parser import TextChunk
from papermind.embeddings import VectorIndex
from papermind.rag_engine import RAGEngine


def test_rag_engine_query():
    index_mgr = VectorIndex()
    chunks = [
        TextChunk(chunk_id=0, text="FlashMind reduces spatial complexity to O(N log N) with 3.4x speedup.", page_num=1, section_name="Abstract", char_count=70),
        TextChunk(chunk_id=1, text="ResNet uses residual skip connections for deep gradient flow.", page_num=2, section_name="Related Work", char_count=60),
    ]
    index_mgr.build_index(chunks)

    rag = RAGEngine(index_mgr, llm_provider="offline")
    resp = rag.query("What complexity reduction does FlashMind achieve?")

    assert resp.query == "What complexity reduction does FlashMind achieve?"
    assert "FlashMind" in resp.answer or "O(N log N)" in resp.answer
    assert len(resp.sources) > 0
    assert resp.sources[0].page_num == 1
