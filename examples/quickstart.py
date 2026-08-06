"""Quickstart example script demonstrating programmatically parsing a paper and executing RAG queries."""

from papermind.parser import PaperParser
from papermind.embeddings import VectorIndex
from papermind.rag_engine import RAGEngine
from papermind.summarizer import PaperSummarizer
from papermind.graph import ConceptGraphExtractor


def main():
    sample_text = """
    ABSTRACT
    Deep Learning models for multi-agent reinforcement learning face scalability challenges.
    We present SwarmMind, an asynchronous policy optimization algorithm.

    METHODOLOGY
    SwarmMind utilizes decentralized actor-critic networks with factorized value functions.

    RESULTS
    Our empirical results across StarCraft II micro-benchmarks show a 28% win-rate increase.
    """

    print("--- 1. Parsing Document ---")
    parser = PaperParser()
    meta, chunks = parser.parse_raw_text(sample_text, title="SwarmMind: Multi-Agent RL")
    print(f"Parsed document: '{meta.title}' into {len(chunks)} chunks.")

    print("\n--- 2. Generating Paper Digest ---")
    summarizer = PaperSummarizer(meta, chunks)
    digest = summarizer.generate_digest()
    print("Overview:", digest["overview"])
    print("Key Insights:", digest["key_insights"])

    print("\n--- 3. Vector Indexing & RAG Q&A ---")
    vector_index = VectorIndex()
    vector_index.build_index(chunks)

    rag = RAGEngine(vector_index, llm_provider="offline")
    response = rag.query("What performance increase was achieved?")
    print("Answer:", response.answer)
    print("Source Section:", response.sources[0].section_name)

    print("\n--- 4. Extracting Concept Graph ---")
    extractor = ConceptGraphExtractor(top_n_concepts=5)
    G, nodes = extractor.build_graph(chunks)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == "__main__":
    main()
