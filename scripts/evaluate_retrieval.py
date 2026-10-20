"""Evaluate the current dense retriever on a small labeled JSON benchmark."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running the benchmark directly from a source checkout without installing
# the package in editable mode.
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from papermind.embeddings import VectorIndex
from papermind.parser import TextChunk


def score_retrieval(
    index: Any, queries: list[dict[str, Any]], top_k: int
) -> dict[str, float]:
    """Return macro Recall@k and mean reciprocal rank for labeled queries."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not queries:
        raise ValueError("At least one query is required")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in queries:
        relevant_ids = set(case["relevant_chunk_ids"])
        if not relevant_ids:
            raise ValueError("Each query must include a relevant chunk ID")

        results = index.search(case["question"], top_k=top_k)
        retrieved_ids = [chunk.chunk_id for chunk, _score in results]
        recalls.append(
            len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)
        )

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(1 / first_relevant_rank if first_relevant_rank else 0.0)

    return {
        f"recall@{top_k}": sum(recalls) / len(recalls),
        f"mrr@{top_k}": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluations/retrieval_smoke.json"),
        help="Path to a benchmark JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Retrieval cutoff (default: 3)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence Transformers model name (default: configured PaperMind model)",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    chunks = [
        TextChunk(char_count=len(chunk["text"]), **chunk) for chunk in dataset["chunks"]
    ]
    index = VectorIndex(model_name=args.model) if args.model else VectorIndex()
    index.build_index(chunks)
    scores = score_retrieval(index, dataset["queries"], args.top_k)

    print(f"Dataset: {args.dataset} ({len(dataset['queries'])} queries)")
    print(f"Model: {index.model_name}")
    for metric, value in scores.items():
        print(f"{metric}: {value:.3f}")
    print(
        "Note: this synthetic smoke set is for regression checks, not a quality claim."
    )


if __name__ == "__main__":
    main()
