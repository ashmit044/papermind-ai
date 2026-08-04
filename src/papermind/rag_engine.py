"""RAG Q&A Engine for PaperMind AI.

Combines vector retrieval with local or remote LLMs to answer questions grounded in paper context.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from papermind.config import config
from papermind.embeddings import VectorIndex
from papermind.parser import TextChunk
from papermind.utils.logger import logger


class SourceRef(BaseModel):
    """Reference to source document snippet."""
    chunk_id: int
    page_num: int
    section_name: str
    similarity_score: float
    snippet: str


class RAGResponse(BaseModel):
    """Structured Q&A answer response."""
    query: str
    answer: str
    sources: List[SourceRef]


class RAGEngine:
    """Retrieval-Augmented Generation query engine."""

    def __init__(self, vector_index: VectorIndex, llm_provider: str = config.llm_provider):
        self.vector_index = vector_index
        self.llm_provider = llm_provider

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """Retrieve relevant paper chunks and answer question using context grounding."""
        logger.info(f"Processing RAG Query: '{question}'")
        
        matches = self.vector_index.search(question, top_k=top_k)
        
        sources: List[SourceRef] = []
        context_str = ""

        for chunk, score in matches:
            sources.append(
                SourceRef(
                    chunk_id=chunk.chunk_id,
                    page_num=chunk.page_num,
                    section_name=chunk.section_name,
                    similarity_score=round(score, 4),
                    snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                )
            )
            context_str += f"\n--- [Page {chunk.page_num} | Section: {chunk.section_name}] ---\n{chunk.text}\n"

        if not matches:
            return RAGResponse(
                query=question,
                answer="No relevant text context could be retrieved from the document to answer this query.",
                sources=[],
            )

        answer = self._generate_answer(question, context_str)

        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
        )

    def _generate_answer(self, question: str, context: str) -> str:
        """Synthesize answer using selected LLM provider or smart offline extraction."""
        if self.llm_provider == "openai" and config.openai_api_key:
            return self._query_openai(question, context)
        elif self.llm_provider == "ollama":
            return self._query_ollama(question, context)
        else:
            # Standalone smart offline response synthesis
            return self._offline_synthesis(question, context)

    def _offline_synthesis(self, question: str, context: str) -> str:
        """Standalone fallback synthesizer for offline execution without external API keys."""
        # Find key sentences from context that align with query terms
        words = [w.lower() for w in question.split() if len(w) > 3]
        paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]

        best_paragraphs = []
        for p in paragraphs:
            score = sum(1 for w in words if w in p.lower())
            if score > 0:
                best_paragraphs.append((score, p))

        best_paragraphs.sort(key=lambda x: x[0], reverse=True)

        if best_paragraphs:
            synthesized = "Based on the paper context retrieved:\n\n"
            for _, para in best_paragraphs[:2]:
                synthesized += f"• {para.strip()}\n\n"
            return synthesized.strip()
        else:
            first_p = paragraphs[0] if paragraphs else context[:300]
            return f"According to the paper findings:\n\n{first_p.strip()}"

    def _query_openai(self, question: str, context: str) -> str:
        """Query OpenAI ChatCompletion API if configured."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=config.openai_api_key)
            prompt = f"You are a research paper assistant. Answer the user question using ONLY the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content or "No response generated."
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Falling back to offline synthesis.")
            return self._offline_synthesis(question, context)

    def _query_ollama(self, question: str, context: str) -> str:
        """Query local Ollama instance if available."""
        try:
            import requests
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt, "stream": False},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            else:
                return self._offline_synthesis(question, context)
        except Exception as e:
            logger.error(f"Ollama connection error: {e}. Falling back to offline synthesis.")
            return self._offline_synthesis(question, context)
