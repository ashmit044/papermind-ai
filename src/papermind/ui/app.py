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
    """Return explicitly fictional content for exploring the interface safely."""
    return """
    DEMO NOTICE: This is a fictional example created only to demonstrate PaperMind AI.
    ExampleStudy and all methods, descriptions, and outcomes below are hypothetical.
    This is not a real research paper and contains no experimentally verified results.

    ABSTRACT
    This illustrative document describes a hypothetical approach called ExampleStudy.
    It is included to demonstrate PDF section parsing, retrieval, citations, and the
    concept graph. No actual model was trained and no experiment was conducted.

    INTRODUCTION
    Research papers often describe a problem, explain the motivation, and situate a
    contribution relative to prior work. A research reader can use section structure
    to locate those parts and compare statements with the original passage.

    METHODOLOGY
    In this fictional walkthrough, the hypothetical ExampleStudy organizes a problem
    statement, a proposed workflow, and a discussion of assumptions. These statements
    are illustrative placeholders rather than evidence of a working method.

    RESULTS
    No experiments, datasets, performance numbers, or empirical results are reported
    in this demonstration document. A responsible research digest should say when
    evidence is missing rather than fill the gap with an invented claim.

    CONCLUSION
    This sample demonstrates interface behavior only. It should never be cited as a
    scientific publication or used as evidence for a real research or engineering claim.
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
    if selected_provider == "openai":
        st.sidebar.warning(
            "OpenAI mode sends retrieved paper passages to OpenAI for answer generation. "
            "Do not select it for sensitive documents unless this data transfer is acceptable."
        )
    elif selected_provider == "ollama":
        st.sidebar.caption(
            "Ollama mode sends prompts to the local Ollama service on this machine."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "💡 **Tip:** Upload a text-based PDF or try the clearly labelled fictional demo."
    )

    # Main Title Header
    st.title("📄 PaperMind AI Research Digest")
    st.write(
        "Upload a research paper PDF to extract executive summaries, query using RAG, and visualize knowledge graphs."
    )

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    use_sample = st.checkbox("Try the fictional demo document", value=False)

    if uploaded_file is None and not use_sample:
        st.info(
            "Upload a PDF to begin, or select the fictional demo to explore the interface."
        )
        st.caption(
            "PaperMind processes uploaded PDF bytes in memory. Scanned/image-only PDFs "
            "are not supported yet because OCR is not included."
        )
        return

    parser = PaperParser()
    try:
        with st.spinner("Processing the document and extracting page-aware text..."):
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
                    title="Fictional ExampleStudy (demo content)",
                )
    except (OSError, RuntimeError, ValueError) as error:
        st.error(f"PaperMind could not process this document: {error}")
        st.caption(
            "Try a valid, non-password-protected PDF with selectable text. "
            "Scanned PDFs need OCR, which is not currently supported."
        )
        st.stop()

    if not chunks:
        st.error("No text chunks were extracted, so the document cannot be analyzed.")
        st.stop()

    if uploaded_file is None and use_sample:
        st.warning(
            "FICTIONAL DEMO ONLY — ExampleStudy is invented content. No real experiment "
            "or research finding is represented here; do not cite the sample."
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
            try:
                with st.spinner(
                    "Loading the local embedding model and building the FAISS index..."
                ):
                    idx_mgr = VectorIndex()
                    idx_mgr.build_index(chunks)
                    st.session_state.vector_index = idx_mgr
                    st.session_state.current_doc_id = document_id
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                st.error(
                    "The local embedding model or vector index could not be initialized. "
                    "Check model-download access and available memory."
                )
                st.caption(f"Details: {error}")
                st.stop()

        rag = RAGEngine(st.session_state.vector_index, llm_provider=selected_provider)

        user_query = st.text_input(
            "Enter your question about the paper:",
            placeholder="e.g. What memory complexity improvement does FlashMind achieve?",
        )

        if user_query:
            with st.spinner("Retrieving semantic context and synthesizing answer..."):
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

        with st.spinner("Extracting concept co-occurrences from the document..."):
            extractor = ConceptGraphExtractor(top_n_concepts=15)
            G, _ = extractor.build_graph(chunks)
            if G.number_of_nodes() == 0:
                st.info(
                    "No concepts were extracted by the current rule-based heuristic. "
                    "Try a paper with clearly named methods, datasets, models, or acronyms."
                )
            else:
                fig = extractor.create_plotly_figure(G)
                st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Edges represent terms appearing in the same text chunk; they do not "
                "establish causation or a formal scientific relationship."
            )


if __name__ == "__main__":
    main()
