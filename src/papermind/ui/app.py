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

# Paper-like surfaces, restrained color, and clear reading hierarchy.
st.markdown(
    """
<style>
    :root {
        --paper-bg: #f7f9fc;
        --paper-surface: #ffffff;
        --paper-ink: #17243a;
        --paper-muted: #52627a;
        --paper-line: #dce4ee;
        --paper-accent: #1769aa;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--paper-bg);
        color: var(--paper-ink);
    }
    .stApp {
        --primary-color: #1769aa;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .main .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.92);
    }
    [data-testid="stSidebar"] {
        background: var(--paper-surface);
        border-right: 1px solid var(--paper-line);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--paper-ink);
        letter-spacing: -0.025em;
    }
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    .stMarkdown h1, .stMarkdown h2 {
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 500;
    }
    p, label, .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--paper-muted);
    }
    [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--paper-line);
    }
    [data-baseweb="tab"] {
        color: var(--paper-muted);
        padding-left: 0.15rem;
        padding-right: 0.15rem;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: var(--paper-accent);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--paper-surface);
        border: 1px dashed #b9c8d9;
        border-radius: 8px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid #b8c8d9;
        font-weight: 600;
        color: var(--paper-ink);
        background: var(--paper-surface);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--paper-accent);
        color: var(--paper-accent);
    }
    [data-testid="stTextInput"] input {
        background: var(--paper-surface);
        border-color: #c8d4e2;
        border-radius: 6px;
    }
    [data-testid="stMetric"] {
        background: transparent;
        border-left: 2px solid var(--paper-line);
        padding-left: 0.85rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--paper-muted);
    }
    [data-testid="stAlert"] {
        border-radius: 6px;
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.25rem 1rem 2rem;
        }
        [data-baseweb="tab-list"] {
            gap: 0.6rem;
        }
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
    st.sidebar.title("PaperMind")
    st.sidebar.caption("Research paper workspace")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Answer settings")
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
    st.sidebar.subheader("Paper")
    uploaded_file = st.sidebar.file_uploader("Upload academic PDF", type=["pdf"])
    use_sample = st.sidebar.checkbox(
        "Explore the synthetic demo paper", value=uploaded_file is None
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Uploaded PDF originals are parsed in memory. Provider settings determine "
        "whether retrieved passages leave your device."
    )

    st.title("PaperMind")
    st.write(
        "Read, question, and explore a research paper with its sources close at hand."
    )

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

        st.subheader(meta.title)
        if uploaded_file is None:
            st.caption(
                f"Demonstration content — not a published research paper · "
                f"{meta.page_count} page · {len(chunks)} text chunks"
            )
        else:
            st.caption(
                f"{Path(uploaded_file.name).name} · {meta.page_count} pages · "
                f"{len(meta.sections)} sections detected"
            )

        st.markdown("---")

        tab_workspace, tab_graph = st.tabs(["Paper workspace", "Concept graph"])

        with tab_workspace:
            digest_column, source_column = st.columns([1.8, 1], gap="large")
            with digest_column:
                summarizer = PaperSummarizer(meta, chunks)
                digest = summarizer.generate_digest()

                st.markdown("### Executive digest")
                st.write(digest["overview"])

                st.markdown("### Key insights")
                for idx, insight in enumerate(digest["key_insights"], 1):
                    st.markdown(f"**{idx}**&nbsp;&nbsp; {insight}")

                st.markdown("### Ask about this paper")
                if (
                    "vector_index" not in st.session_state
                    or st.session_state.get("current_doc_id") != document_id
                ):
                    with st.spinner("Preparing local search…"):
                        idx_mgr = VectorIndex()
                        idx_mgr.build_index(chunks)
                        st.session_state.vector_index = idx_mgr
                        st.session_state.current_doc_id = document_id

                rag = RAGEngine(
                    st.session_state.vector_index, llm_provider=selected_provider
                )
                with st.form("paper_question", clear_on_submit=False):
                    question_col, ask_col = st.columns(
                        [5, 1], vertical_alignment="bottom"
                    )
                    with question_col:
                        user_query = st.text_input(
                            "Question",
                            placeholder="Ask about the paper's methods, findings, or limitations…",
                        )
                    with ask_col:
                        submitted = st.form_submit_button(
                            "Ask", use_container_width=True
                        )

                response_key = (document_id, user_query.strip(), llm_option)
                if (
                    submitted
                    and user_query.strip()
                    and st.session_state.get("query_response_key") != response_key
                ):
                    with st.spinner(
                        "Finding relevant passages and preparing an answer…"
                    ):
                        st.session_state.query_response = rag.query(
                            user_query.strip(), top_k=3
                        )
                        st.session_state.query_response_key = response_key

                response = None
                if st.session_state.get("query_response_key") == response_key:
                    response = st.session_state.query_response
                if submitted and not user_query.strip():
                    st.warning("Enter a question about the paper first.")
                if response:
                    st.markdown("#### Answer")
                    st.write(response.answer)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("### Methodology")
                    st.write(digest["methodology"])
                with col_b:
                    st.markdown("### Limitations and future work")
                    st.write(digest["limitations"])

                st.markdown("### Export digest")
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

            with source_column:
                st.markdown("### Retrieved sources")
                if response:
                    for src in response.sources:
                        with st.expander(
                            f"[S{src.source_id}] · Page {src.page_num} · {src.section_name}"
                        ):
                            st.write(src.snippet)
                else:
                    st.caption(
                        "Ask a question to see the paper passages used for the answer, "
                        "with page and section details."
                    )

        with tab_graph:
            st.markdown("### Concept co-occurrence graph")
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
