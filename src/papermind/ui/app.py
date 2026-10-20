"""Streamlit Dashboard for PaperMind AI.

Interactive web UI featuring PDF upload, paper digests, grounded RAG Q&A, and concept graph visualization.
"""

import sys
from pathlib import Path

# Add src to sys.path if running standalone
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

from papermind.config import config
from papermind.embeddings import VectorIndex
from papermind.graph import ConceptGraphExtractor
from papermind.parser import PaperParser, document_fingerprint
from papermind.rag_engine import RAGEngine
from papermind.summarizer import PaperSummarizer
from papermind.utils.exporter import DigestExporter

# Streamlit Page Config
st.set_page_config(
    page_title="PaperMind AI — Academic Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Glassmorphism Styling
st.markdown(
    """
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp header {
        background-color: transparent;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


def sample_document_text() -> str:
    """Fallback sample text for instant demo execution."""
    return """
    ABSTRACT
    Transformers have revolutionized natural language processing and computer vision. However, standard self-attention exhibits quadratic computation and memory complexity O(N^2) with sequence length N. In this paper, we propose FlashMind, an efficient sparse linear attention mechanism that reduces spatial complexity to O(N log N) while maintaining competitive representation quality.

    INTRODUCTION
    Deep neural networks trained on massive multi-modal corpora demonstrate remarkable emergent intelligence. Yet, processing long-context documents such as complete academic books, codebase repositories, and high-resolution genomic sequences remains constrained by memory latency. FlashMind addresses this bottleneck by introducing hierarchical memory token compression.

    METHODOLOGY
    FlashMind relies on dual-stream sparse cross-attention layers. Given input embeddings X, we compute key-value pairs using randomized Hadamard projections. Loss is computed using cross-entropy alongside auxiliary contrastive representation regularization.

    EXPERIMENTS & RESULTS
    We evaluate FlashMind on the Long Range Arena (LRA) benchmark and arXiv CS paper QA datasets. Results demonstrate a 3.4x throughput speedup over standard Transformer baselines and a 42% decrease in peak GPU memory usage, achieving state-of-the-art performance.

    CONCLUSION
    We present FlashMind, an efficient architecture for long-context research reasoning. Future work involves extending FlashMind to real-time multimodal video stream processing.
    """


def main():
    st.sidebar.title("🧠 PaperMind AI")
    st.sidebar.caption("Academic Paper Digest & Local RAG Assistant")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    llm_option = st.sidebar.selectbox(
        "LLM Synthesis Backend",
        ["Offline Synthesizer (Fast)", "Ollama (Local Llama3)", "OpenAI GPT-3.5"],
        index=0,
    )

    provider_map = {
        "Offline Synthesizer (Fast)": "offline",
        "Ollama (Local Llama3)": "ollama",
        "OpenAI GPT-3.5": "openai",
    }
    selected_provider = provider_map[llm_option]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 **Tip:** Upload any PDF or use the sample paper below for instant testing."
    )

    # Main Title Header
    st.title("📄 PaperMind AI Research Digest")
    st.write(
        "Upload a research paper PDF to extract executive summaries, query using RAG, and visualize knowledge graphs."
    )

    uploaded_file = st.file_uploader("Upload Academic Paper (PDF)", type=["pdf"])
    use_sample = st.checkbox("Use Demo Sample Paper", value=uploaded_file is None)

    if uploaded_file is not None or use_sample:
        parser = PaperParser()

        with st.spinner("Processing paper and extracting section chunks..."):
            if uploaded_file is not None:
                file_contents = uploaded_file.getvalue()
                document_id = document_fingerprint(file_contents)
                safe_filename = Path(uploaded_file.name).name or "uploaded_document.pdf"
                fallback_title = Path(safe_filename).stem.replace("_", " ").title()
                meta, chunks = parser.parse_pdf_bytes(
                    file_contents, fallback_title=fallback_title
                )
            else:
                sample_text = sample_document_text()
                document_id = document_fingerprint(sample_text.encode("utf-8"))
                meta, chunks = parser.parse_raw_text(
                    sample_text,
                    title="FlashMind: Efficient Long-Context Transformers",
                )

        # Top Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Document Title",
            meta.title[:25] + "..." if len(meta.title) > 25 else meta.title,
        )
        col2.metric("Total Pages", meta.page_count)
        col3.metric("Text Chunks", len(chunks))
        col4.metric("Sections Detected", len(meta.sections))

        st.markdown("---")

        # Tabs Navigation
        tab_summary, tab_rag, tab_graph = st.tabs(
            [
                "📌 Executive Digest",
                "💬 Grounded RAG Q&A",
                "🕸️ Concept Graph",
            ]
        )

        # TAB 1: Executive Digest
        with tab_summary:
            summarizer = PaperSummarizer(meta, chunks)
            digest = summarizer.generate_digest()

            st.subheader("📌 Executive Overview")
            st.info(digest["overview"])

            st.subheader("💡 Key Takeaways & Contributions")
            for idx, insight in enumerate(digest["key_insights"], 1):
                st.markdown(f"**{idx}.** {insight}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("⚙️ Methodology")
                st.write(digest["methodology"])
            with col_b:
                st.subheader("⚠️ Limitations & Future Work")
                st.write(digest["limitations"])

            st.markdown("### 📥 Export Digest")
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                md_path = config.cache_dir / f"{meta.title[:15]}_digest.md"
                DigestExporter.export_to_markdown(digest, md_path)
                with open(md_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "Download Markdown Report",
                        f.read(),
                        file_name="paper_digest.md",
                        mime="text/markdown",
                    )
            with exp_col2:
                json_path = config.cache_dir / f"{meta.title[:15]}_digest.json"
                DigestExporter.export_to_json(digest, json_path)
                with open(json_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        "Download JSON Data",
                        f.read(),
                        file_name="paper_digest.json",
                        mime="application/json",
                    )

        # TAB 2: Grounded RAG Q&A
        with tab_rag:
            st.subheader("💬 Ask Questions Grounded in Paper Context")

            # Initialize Vector Index
            if (
                "vector_index" not in st.session_state
                or st.session_state.get("current_doc_id") != document_id
            ):
                with st.spinner(
                    "Building FAISS vector index with sentence-transformers..."
                ):
                    idx_mgr = VectorIndex()
                    idx_mgr.build_index(chunks)
                    st.session_state.vector_index = idx_mgr
                    st.session_state.current_doc_id = document_id

            rag = RAGEngine(
                st.session_state.vector_index, llm_provider=selected_provider
            )

            user_query = st.text_input(
                "Enter your question about the paper:",
                placeholder="e.g. What memory complexity improvement does FlashMind achieve?",
            )

            if user_query:
                with st.spinner(
                    "Retrieving semantic context and synthesizing answer..."
                ):
                    response = rag.query(user_query, top_k=3)
                    st.markdown("#### 🤖 Grounded AI Answer")
                    st.success(response.answer)

                    st.markdown("#### 📚 Source Citations & Snippets")
                    for src in response.sources:
                        with st.expander(
                            f"[S{src.source_id}] Page {src.page_num} | Section: {src.section_name} (Similarity: {src.similarity_score})"
                        ):
                            st.write(src.snippet)

        # TAB 3: Concept Graph
        with tab_graph:
            st.subheader("🕸️ Concept Co-occurrence & Entity Graph")
            st.write(
                "Extracts technical concepts, model names, and relationships directly from the document."
            )

            with st.spinner("Generating concept network graph..."):
                extractor = ConceptGraphExtractor(top_n_concepts=15)
                G, _ = extractor.build_graph(chunks)
                fig = extractor.create_plotly_figure(G)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
