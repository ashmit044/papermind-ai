"""Tests for retrieval evaluation metrics."""

import pytest

from papermind.parser import TextChunk
from scripts.evaluate_retrieval import score_retrieval


class StubIndex:
    def __init__(self, results):
        self.results = results

    def search(self, question, top_k):
        return self.results[question][:top_k]


def chunk(chunk_id):
    return TextChunk(
        chunk_id=chunk_id,
        text=f"passage {chunk_id}",
        page_num=chunk_id + 1,
        section_name="Results",
        char_count=10,
    )


def test_score_retrieval_computes_recall_at_k_and_mrr():
    index = StubIndex(
        {
            "first": [(chunk(2), 0.9), (chunk(1), 0.8)],
            "second": [(chunk(3), 0.9), (chunk(1), 0.8)],
        }
    )
    queries = [
        {"question": "first", "relevant_chunk_ids": [1]},
        {"question": "second", "relevant_chunk_ids": [3]},
    ]

    scores = score_retrieval(index, queries, top_k=2)

    assert scores["recall@2"] == 1.0
    assert scores["mrr@2"] == pytest.approx(0.75)


@pytest.mark.parametrize("top_k", [0, -1])
def test_score_retrieval_requires_positive_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        score_retrieval(
            StubIndex([]), [{"question": "q", "relevant_chunk_ids": [1]}], top_k
        )


def test_score_retrieval_requires_queries():
    with pytest.raises(ValueError, match="At least one query"):
        score_retrieval(StubIndex([]), [], top_k=1)
