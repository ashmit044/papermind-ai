"""Academic Paper Summarizer module for PaperMind AI.

Generates structured digests, executive summaries, methodology breakdowns, and key insights.
"""

from typing import Any

from papermind.parser import DocumentMetadata, TextChunk
from papermind.utils.logger import logger


class PaperSummarizer:
    """Summarizes research papers into structured executive digests."""

    def __init__(self, metadata: DocumentMetadata, chunks: list[TextChunk]):
        self.metadata = metadata
        self.chunks = chunks

    def generate_digest(self) -> dict[str, Any]:
        """Generate a complete paper digest including overview, takeaways, and methodology."""
        logger.info(f"Generating research paper digest for '{self.metadata.title}'...")

        overview = self._extract_overview()
        insights = self._extract_key_insights()
        methodology = (
            self._extract_section_summary("Methodology")
            or self._extract_section_summary("Methods")
            or "Not identified in the extracted text."
        )
        limitations = (
            self._extract_section_summary("Discussion")
            or "Not identified in the extracted text."
        )

        digest = {
            "title": self.metadata.title,
            "author": self.metadata.author,
            "page_count": self.metadata.page_count,
            "sections": self.metadata.sections,
            "overview": overview,
            "key_insights": insights,
            "methodology": methodology,
            "limitations": limitations,
        }

        return digest

    def _extract_overview(self) -> str:
        """Extract overview from Abstract or first chunks."""
        abstract_chunks = [
            c for c in self.chunks if "abstract" in c.section_name.lower()
        ]
        if abstract_chunks:
            text = " ".join([c.text for c in abstract_chunks[:2]])
            return text[:600] + "..." if len(text) > 600 else text

        first_chunks = self.chunks[:2]
        combined = " ".join([c.text for c in first_chunks])
        return combined[:500] + "..." if len(combined) > 500 else combined

    def _extract_key_insights(self) -> list[str]:
        """Extract top 3-4 key findings/contributions from text chunks."""
        insights = []
        keywords = [
            "propose",
            "demonstrate",
            "outperform",
            "achieve",
            "introduce",
            "novel",
            "results show",
        ]

        for chunk in self.chunks:
            for line in chunk.text.split("."):
                line_clean = line.strip()
                if (
                    any(kw in line_clean.lower() for kw in keywords)
                    and len(line_clean) > 30
                    and line_clean not in insights
                    and len(insights) < 4
                ):
                    insights.append(line_clean + ".")

        if not insights:
            insights = [
                (
                    "No explicit contribution statements were identified by the current "
                    "keyword-based extractor. Review the extracted paper text directly."
                )
            ]

        return insights

    def _extract_section_summary(self, section_keyword: str) -> str:
        """Extract text snippet for a targeted section keyword."""
        relevant_chunks = [
            c for c in self.chunks if section_keyword.lower() in c.section_name.lower()
        ]
        if relevant_chunks:
            text = " ".join([c.text for c in relevant_chunks[:2]])
            return text[:400] + "..." if len(text) > 400 else text
        return ""
