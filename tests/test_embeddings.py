"""Unit tests for VectorIndex."""

from papermind.embeddings import VectorIndex
from papermind.parser import TextChunk


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
