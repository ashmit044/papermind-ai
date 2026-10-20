"""Unit tests for VectorIndex."""

import pytest

from papermind.embeddings import VectorIndex
from papermind.parser import TextChunk


def test_empty_rebuild_clears_old_vector_index():
    index_mgr = VectorIndex.__new__(VectorIndex)
    index_mgr.model_name = "test-model"
    index_mgr.model = object()
    index_mgr.dimension = 2
    index_mgr.index = object()
    index_mgr.chunks = [
        TextChunk(
            chunk_id=0,
            text="old document",
            page_num=1,
            section_name="Intro",
            char_count=12,
        )
    ]

    index_mgr.build_index([])

    assert index_mgr.index is None
    assert index_mgr.chunks == []


@pytest.mark.parametrize(
    "query,top_k",
    [("", 1), ("  ", 1), ("valid query", 0), ("valid query", -1), ("valid query", True)],
)
def test_vector_search_rejects_invalid_inputs(query, top_k):
    index_mgr = VectorIndex.__new__(VectorIndex)
    index_mgr.index = None
    index_mgr.chunks = []

    with pytest.raises(ValueError):
        index_mgr.search(query, top_k=top_k)


def test_vector_index_build_and_search():
    index_mgr = VectorIndex()

    chunks = [
        TextChunk(
            chunk_id=0,
            text="Transformer models use self-attention mechanism.",
            page_num=1,
            section_name="Intro",
            char_count=50,
        ),
        TextChunk(
            chunk_id=1,
            text="Convolutional neural networks process images.",
            page_num=1,
            section_name="Related",
            char_count=45,
        ),
        TextChunk(
            chunk_id=2,
            text="Linear attention reduces quadratic complexity.",
            page_num=2,
            section_name="Method",
            char_count=45,
        ),
    ]

    index_mgr.build_index(chunks)
    assert index_mgr.index is not None
    assert index_mgr.index.ntotal == 3

    results = index_mgr.search("attention complexity", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert "attention" in top_chunk.text.lower()
    assert score > 0.0
