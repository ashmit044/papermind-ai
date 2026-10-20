"""Vector embedding and FAISS index manager for PaperMind AI.

Generates dense vector embeddings using sentence-transformers and builds similarity search indices.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from papermind.config import config
from papermind.parser import TextChunk
from papermind.utils.logger import logger


class VectorIndex:
    """Manages dense vector generation and FAISS similarity indexing."""

    def __init__(self, model_name: str = config.embedding_model_name):
        self.model_name = model_name
        logger.info(f"Initializing embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[TextChunk] = []

    def build_index(self, chunks: list[TextChunk]) -> None:
        """Compute embeddings for text chunks and replace the current FAISS index."""
        # Clear old state first so a failed or empty rebuild cannot leak stale results.
        self.index = None
        self.chunks = []
        valid_chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not valid_chunks:
            logger.warning(
                "No non-empty text chunks provided; vector index was cleared."
            )
            return

        texts = [chunk.text for chunk in valid_chunks]
        logger.info(f"Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings_np = np.asarray(embeddings, dtype=np.float32)

        if embeddings_np.ndim != 2 or embeddings_np.shape[0] != len(valid_chunks):
            raise ValueError(
                "Embedding model returned an unexpected number of vectors."
            )
        if embeddings_np.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"received {embeddings_np.shape[1]}."
            )
        if not np.isfinite(embeddings_np).all():
            raise ValueError("Embedding model returned non-finite vector values.")

        embeddings_np = np.ascontiguousarray(embeddings_np)
        # Inner Product on normalized vectors equals Cosine Similarity.
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings_np)
        self.index = index
        self.chunks = valid_chunks

        logger.info(
            f"Successfully built FAISS index with {self.index.ntotal} vectors (dim={self.dimension})"
        )

    def search(self, query: str, top_k: int = 4) -> list[tuple[TextChunk, float]]:
        """Perform semantic search for query and return top_k matching chunks with similarity scores."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if self.index is None or self.index.ntotal == 0:
            logger.error(
                "FAISS index is uninitialized or empty. Call build_index first."
            )
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec_np = np.array(query_vec, dtype=np.float32)

        scores, indices = self.index.search(
            query_vec_np, k=min(top_k, self.index.ntotal)
        )

        results: list[tuple[TextChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
