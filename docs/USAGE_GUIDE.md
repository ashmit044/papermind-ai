# 📖 PaperMind AI — User Guide

Welcome to the **PaperMind AI** user guide. This document provides step-by-step instructions for using both the CLI tool and the Streamlit web dashboard.

---

## 💻 1. CLI Usage Guide

PaperMind AI provides a feature-rich command-line interface powered by `typer` and `rich`.

### Commands

#### Generate Executive Paper Digest
```bash
papermind digest path/to/paper.pdf
```
Options:
- `--md` / `--no-md`: Export summary report to Markdown (default: enabled).
- `--json`: Export raw digest data to JSON format.
- `--output-dir PATH`: Directory for saved outputs (default: `./output`).

#### Grounded RAG Question Answering
```bash
papermind ask path/to/paper.pdf "What speedup does the proposed model achieve?"
```
Options:
- `-k`, `--top-k INT`: Number of context chunks to retrieve (default: 3).

#### Display Version
```bash
papermind version
```

---

## 🌐 2. Streamlit Web Dashboard

To launch the web user interface:

```bash
streamlit run src/papermind/ui/app.py
```

### Dashboard Tabs
1. **📌 Executive Digest**: View paper title, pages, chunk count, overview, methodology, key findings, and download Markdown/JSON reports.
2. **💬 Grounded RAG Q&A**: Interactive chat interface to ask questions about the document with inline page citations.
3. **🕸️ Concept Graph**: Interactive Plotly network visualizer showing concept centrality and co-occurrence nodes.

---

## ⚙️ 3. Configuration & Environment Variables

Copy `.env.example` to `.env` to configure custom model parameters:

```env
PAPERMIND_LLM_PROVIDER=offline
PAPERMIND_EMBEDDING_MODEL=all-MiniLM-L6-v2
PAPERMIND_CHUNK_SIZE=500
PAPERMIND_CHUNK_OVERLAP=50
PAPERMIND_MIN_SIMILARITY=0.20
OPENAI_API_KEY=your_key_here
```

`PAPERMIND_MIN_SIMILARITY` is a cosine-similarity threshold between -1 and 1. Only retrieved passages at or above the threshold are used as evidence; lower the value cautiously if useful passages are being missed.

OpenAI mode sends retrieved passages to OpenAI for answer generation. Ollama mode uses a local Ollama service. Install the optional adapters with `python -m pip install -e ".[openai,ollama]"` when using those providers.
