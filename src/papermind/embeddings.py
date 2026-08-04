"""Vector embedding and FAISS index manager for PaperMind AI.

Generates dense vector embeddings using sentence-transformers and builds similarity search indices.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
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
        self.chunks: List[TextChunk] = []

    def build_index(self, chunks: List[TextChunk]) -> None:
        """Compute embeddings for text chunks and populate FAISS index."""
        if not chunks:
            logger.warning("Empty chunk list provided for indexing.")
            return

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} document chunks...")
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Inner Product on normalized vectors equals Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_np)
        
        logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors (dim={self.dimension})")

    def search(self, query: str, top_k: int = 4) -> List[Tuple[TextChunk, float]]:
        """Perform semantic search for query and return top_k matching chunks with similarity scores."""
        if self.index is None or self.index.ntotal == 0:
            logger.error("FAISS index is uninitialized or empty. Call build_index first.")
            return []

        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec_np = np.array(query_vec, dtype=np.float32)

        scores, indices = self.index.search(query_vec_np, k=min(top_k, self.index.ntotal))

        results: List[Tuple[TextChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results
