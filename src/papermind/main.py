"""CLI Entrypoint for PaperMind AI.

Powered by Typer and Rich for beautiful terminal outputs.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from papermind.config import config
from papermind.embeddings import VectorIndex
from papermind.parser import PaperParser
from papermind.rag_engine import RAGEngine
from papermind.summarizer import PaperSummarizer
from papermind.utils.exporter import DigestExporter

app = typer.Typer(
    name="papermind",
    help="📄 PaperMind AI: Academic Paper Digest, RAG Assistant & Knowledge Graph Synthesizer",
    add_completion=False,
)
console = Console()


@app.command()
def digest(
    pdf_path: str = typer.Argument(..., help="Path to PDF research paper"),
    export_md: bool = typer.Option(
        True, "--md/--no-md", help="Export summary to Markdown report"
    ),
    export_json: bool = typer.Option(
        False, "--json", help="Export digest data to JSON file"
    ),
    output_dir: str = typer.Option("./output", help="Directory for exported reports"),
):
    """Generate an executive summary & digest of an academic paper."""
    console.print(
        Panel(
            "[bold cyan]PaperMind AI — Research Digest Generator[/bold cyan]",
            expand=False,
        )
    )

    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {pdf_path}")
        raise typer.Exit(1)

    parser = PaperParser()
    meta, chunks = parser.parse_pdf(path)

    summarizer = PaperSummarizer(meta, chunks)
    digest_data = summarizer.generate_digest()

    console.print(f"\n[bold green]Title:[/bold green] {meta.title}")
    console.print(
        f"[bold yellow]Pages:[/bold yellow] {meta.page_count} | [bold yellow]Chunks:[/bold yellow] {len(chunks)}\n"
    )

    console.print(
        Panel(
            Markdown(f"### 📌 Executive Overview\n\n{digest_data['overview']}"),
            title="Overview",
        )
    )

    table = Table(
        title="💡 Key Contributions & Insights",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Key Finding", style="white")

    for idx, insight in enumerate(digest_data["key_insights"], 1):
        table.add_row(str(idx), insight)

    console.print(table)

    if export_md:
        out_file = Path(output_dir) / f"{path.stem}_digest.md"
        DigestExporter.export_to_markdown(digest_data, out_file)
        console.print(
            f"\n[bold green]✓ Report exported to:[/bold green] [underline]{out_file}[/underline]"
        )

    if export_json:
        out_json = Path(output_dir) / f"{path.stem}_digest.json"
        DigestExporter.export_to_json(digest_data, out_json)
        console.print(
            f"[bold green]✓ JSON exported to:[/bold green] [underline]{out_json}[/underline]"
        )


@app.command()
def ask(
    pdf_path: str = typer.Argument(..., help="Path to PDF research paper"),
    question: str = typer.Argument(..., help="Question to ask about the paper"),
    top_k: int = typer.Option(
        3, "--top-k", "-k", help="Number of context chunks to retrieve"
    ),
):
    """Query a research paper using local RAG semantic search."""
    console.print(
        Panel("[bold cyan]PaperMind AI — Grounded Paper Q&A[/bold cyan]", expand=False)
    )

    path = Path(pdf_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {pdf_path}")
        raise typer.Exit(1)

    parser = PaperParser()
    _meta, chunks = parser.parse_pdf(path)

    vector_index = VectorIndex()
    vector_index.build_index(chunks)

    rag = RAGEngine(vector_index)
    result = rag.query(question, top_k=top_k)

    console.print(f"\n[bold cyan]Question:[/bold cyan] {result.query}\n")
    console.print(
        Panel(result.answer, title="🤖 AI Grounded Answer", border_style="green")
    )

    console.print("\n[bold yellow]📚 Retrieved Source Snips:[/bold yellow]")
    for src in result.sources:
        console.print(
            f'• [dim]Page {src.page_num} | Section: {src.section_name} | Similarity: {src.similarity_score:.2f}[/dim]\n  "{src.snippet}"\n'
        )


@app.command()
def version():
    """Display PaperMind AI CLI version."""
    console.print(
        f"[bold green]{config.app_name}[/bold green] version [yellow]{config.version}[/yellow]"
    )


if __name__ == "__main__":
    app()
