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
from papermind.learning import PaperMemory
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
        max-width: 1080px;
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
    .stMarkdown h1, .stMarkdown h2 { font-weight: 600; }
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
    .stButton > button[kind="secondary"] {
        min-height: 3rem;
        white-space: normal;
        text-align: left;
        line-height: 1.35;
    }
    [data-testid="stChatMessage"] {
        max-width: 780px;
        margin-left: auto;
        margin-right: auto;
        padding: 0.7rem 0.5rem;
        border: 0;
        background: transparent;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        justify-content: flex-end;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
        flex: 0 1 auto;
        max-width: 78%;
        padding: 0.72rem 1rem;
        border-radius: 1.25rem;
        background: #eef1f5;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stChatMessageContent"] { padding-left: 0; }
    [data-testid="stChatInput"] textarea {
        border-radius: 20px;
        padding: 0.85rem 1rem;
        background: #ffffff;
        box-shadow: 0 2px 12px rgba(23, 36, 58, 0.08);
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

if st.session_state.get("dark_mode", False):
    st.markdown(
        """
<style>
    :root {
        --paper-bg: #212121;
        --paper-surface: #2f2f2f;
        --paper-ink: #ececec;
        --paper-muted: #b4b4b4;
        --paper-line: #454545;
        --paper-accent: #8ab4f8;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--paper-bg);
        color: var(--paper-ink);
    }
    [data-testid="stSidebar"], [data-testid="stChatInput"] textarea,
    [data-testid="stFileUploaderDropzone"] {
        background: var(--paper-surface);
        color: var(--paper-ink);
        border-color: var(--paper-line);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] { background: #303030; }
    .stButton > button, .stDownloadButton > button {
        background: var(--paper-surface);
        color: var(--paper-ink);
        border-color: var(--paper-line);
    }
    [data-baseweb="tab"], [data-testid="stExpander"] {
        color: var(--paper-ink);
        border-color: var(--paper-line);
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


def render_chat(meta, chunks, document_id, selected_provider, saved_notes):
    """Render the paper conversation as the primary, full-width workspace view."""
    if (
        "vector_index" not in st.session_state
        or st.session_state.get("current_doc_id") != document_id
    ):
        with st.spinner("Preparing paper search…"):
            idx_mgr = VectorIndex()
            idx_mgr.build_index(chunks)
            st.session_state.vector_index = idx_mgr
            st.session_state.current_doc_id = document_id

    rag = RAGEngine(st.session_state.vector_index, llm_provider=selected_provider)
    history_key = f"paper_chat_{document_id}"
    st.session_state.setdefault(history_key, [])
    history = st.session_state[history_key]

    st.markdown("### Chat with your paper")
    st.caption("Ask a question, then explore the cited passages behind each answer.")

    if not history:
        st.markdown("#### A good place to start")
        prompt_columns = st.columns(3)
        suggested_questions = [
            "What problem does this paper address?",
            "Explain the method in simple terms.",
            "What evidence supports the main result?",
        ]
        for column, prompt in zip(prompt_columns, suggested_questions):
            with column:
                if st.button(
                    prompt,
                    key=f"suggest_{document_id}_{prompt}",
                    use_container_width=True,
                ):
                    st.session_state["pending_chat_question"] = prompt
                    st.rerun()

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                with st.expander(
                    f"[{source['source_id']}] Page {source['page_num']} · "
                    f"{source['section_name']}"
                ):
                    st.write(source["snippet"])
            if message.get("user_notes"):
                with st.expander("Relevant saved notes · not paper citations"):
                    for note in message["user_notes"]:
                        st.markdown(f"- {note}")

    user_query = st.chat_input(
        "Ask a follow-up about the paper…",
        key=f"chat_input_{document_id}",
    )
    user_query = user_query or st.session_state.pop("pending_chat_question", None)
    if user_query and user_query.strip():
        user_query = user_query.strip()
        history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.status("Searching your paper…", expanded=True) as work_status:
                response = rag.query(
                    user_query,
                    top_k=3,
                    user_notes=saved_notes,
                    progress_callback=lambda message: work_status.update(
                        label=message, state="running", expanded=True
                    ),
                )
                work_status.update(
                    label="Answer ready · grounded in retrieved passages",
                    state="complete",
                    expanded=False,
                )
            st.markdown(response.answer)
            source_data = []
            for source in response.sources:
                source_data.append(source.model_dump())
                with st.expander(
                    f"[{source.source_id}] Page {source.page_num} · "
                    f"{source.section_name}"
                ):
                    st.write(source.snippet)
        history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": source_data,
                "user_notes": response.user_notes,
            }
        )
        if response.user_notes:
            with st.expander("Relevant saved notes · not paper citations"):
                for note in response.user_notes:
                    st.markdown(f"- {note}")


def render_digest(meta, chunks):
    """Render the paper summary and its export actions."""
    summarizer = PaperSummarizer(meta, chunks)
    digest = summarizer.generate_digest()
    st.markdown("### Executive digest")
    st.write(digest["overview"])
    st.markdown("### Key insights")
    for idx, insight in enumerate(digest["key_insights"], 1):
        st.markdown(f"**{idx}**&nbsp;&nbsp; {insight}")

    detail_col, limitation_col = st.columns(2)
    with detail_col:
        st.markdown("### Methodology")
        st.write(digest["methodology"])
    with limitation_col:
        st.markdown("### Limitations and future work")
        st.write(digest["limitations"])

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        md_path = config.cache_dir / f"{meta.title[:15]}_digest.md"
        DigestExporter.export_to_markdown(digest, md_path)
        with open(md_path, "r", encoding="utf-8") as f:
            st.download_button(
                "Download Markdown Report",
                f.read(),
                file_name="paper_digest.md",
                mime="text/markdown",
            )
    with export_col2:
        json_path = config.cache_dir / f"{meta.title[:15]}_digest.json"
        DigestExporter.export_to_json(digest, json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            st.download_button(
                "Download JSON Data",
                f.read(),
                file_name="paper_digest.json",
                mime="application/json",
            )


def render_concept_map(chunks):
    """Render the paper concept co-occurrence graph."""
    st.markdown("### Concept co-occurrence graph")
    st.write(
        "Extracts technical concepts, model names, and relationships directly from the document."
    )
    with st.spinner("Generating concept network graph..."):
        extractor = ConceptGraphExtractor(top_n_concepts=15)
        graph, _ = extractor.build_graph(chunks)
        figure = extractor.create_plotly_figure(graph)
        st.plotly_chart(figure, use_container_width=True)


def main():
    st.sidebar.title("PaperMind")
    st.sidebar.caption("Your research workspace")

    st.sidebar.toggle(
        "Dark mode", key="dark_mode", help="Switch between light and dark themes."
    )
    new_chat_requested = st.sidebar.button("＋  New chat", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Answer settings")
    llm_option = st.sidebar.selectbox(
        "LLM Synthesis Backend",
        ["Offline Synthesizer (Fast)", "Ollama (Local Llama3)", "OpenAI GPT-3.5"],
        index=0,
        help=(
            "Offline mode retrieves evidence and answers immediately with excerpts. "
            "Ollama or OpenAI can compose fuller responses when configured."
        ),
    )
    if llm_option == "Offline Synthesizer (Fast)":
        st.sidebar.caption("Fast evidence mode · no generative model is running.")
    elif llm_option == "OpenAI GPT-3.5" and not config.openai_api_key:
        st.sidebar.warning(
            "OPENAI_API_KEY is not configured; answers will use offline evidence mode."
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
        "PDFs are parsed in memory. Selected model providers may receive retrieved "
        "passages and relevant notes."
    )

    st.title("PaperMind")
    st.caption("Your research copilot · answers grounded in the paper you choose")

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

        paper_memory = PaperMemory(config.cache_dir / "paper_memory.json")
        try:
            saved_notes = paper_memory.get_notes(document_id)
        except (OSError, TypeError, ValueError) as error:
            st.sidebar.error(f"Could not read saved paper notes: {error}")
            saved_notes = []

        with st.sidebar.expander("Paper memory", expanded=False):
            st.caption(
                "Add a note or correction for future questions on this paper. "
                "Notes stay on this device and are not model training."
            )
            with st.form(f"paper_memory_form_{document_id}"):
                new_note = st.text_area(
                    "What should PaperMind remember?",
                    placeholder="Add a definition, correction, or research note…",
                    max_chars=PaperMemory.MAX_NOTE_LENGTH,
                )
                save_note = st.form_submit_button("Save note")
            if save_note:
                try:
                    paper_memory.add_note(document_id, new_note)
                except (OSError, TypeError, ValueError) as error:
                    st.error(f"Could not save note: {error}")
                else:
                    st.success("Saved for this paper on this device.")
                    st.rerun()
            if saved_notes:
                st.caption(f"{len(saved_notes)} saved note(s)")
                if st.button(
                    "Clear notes for this paper", key=f"clear_notes_{document_id}"
                ):
                    try:
                        paper_memory.clear_notes(document_id)
                    except (OSError, TypeError, ValueError) as error:
                        st.error(f"Could not clear notes: {error}")
                    else:
                        st.rerun()

        history_key = f"paper_chat_{document_id}"
        st.session_state.setdefault(history_key, [])
        if new_chat_requested:
            st.session_state[history_key] = []
            st.session_state.pop("pending_chat_question", None)
            st.rerun()
        workspace_view = st.sidebar.radio(
            "Workspace",
            ["Chat", "Paper digest", "Concept map"],
            key=f"workspace_{document_id}",
        )

        if workspace_view == "Chat":
            render_chat(meta, chunks, document_id, selected_provider, saved_notes)
        elif workspace_view == "Paper digest":
            render_digest(meta, chunks)
        else:
            render_concept_map(chunks)


if __name__ == "__main__":
    main()
