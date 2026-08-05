"""Knowledge Graph extractor and NetworkX / Plotly visualizer for PaperMind AI.

Extracts concept co-occurrence network graph from research paper chunks.
"""

import re
from typing import List, Dict, Any, Tuple
import networkx as nx
import plotly.graph_objects as go
from papermind.parser import TextChunk
from papermind.utils.logger import logger


class ConceptGraphExtractor:
    """Extracts entity and concept relationships into an interactive NetworkX graph."""

    def __init__(self, top_n_concepts: int = 15):
        self.top_n_concepts = top_n_concepts

    def build_graph(self, chunks: List[TextChunk]) -> Tuple[nx.Graph, List[Dict[str, Any]]]:
        """Extract top concept nodes and co-occurrence edges from chunks."""
        logger.info("Extracting concept knowledge graph from document chunks...")
        
        # Simple rule-based extraction for AI/CS domain entities
        concept_freq: Dict[str, int] = {}
        chunk_concepts: List[List[str]] = []

        # Common stop words to ignore
        stopwords = {
            "this", "that", "with", "from", "using", "paper", "results", "model",
            "method", "data", "table", "figure", "section", "show", "used", "also"
        }

        for chunk in chunks:
            # Match technical terms (Capitalized words, acronyms, or tech keywords)
            words = re.findall(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*\b|\b[A-Z]{2,}\b", chunk.text)
            cleaned = [
                w.strip() for w in words 
                if len(w) > 2 and w.lower() not in stopwords
            ]
            
            unique_in_chunk = list(set(cleaned))
            chunk_concepts.append(unique_in_chunk)

            for concept in unique_in_chunk:
                concept_freq[concept] = concept_freq.get(concept, 0) + 1

        # Select top N most frequent concepts
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
        top_concepts = set([item[0] for item in sorted_concepts[: self.top_n_concepts]])

        # Fallback if few capitalized words are found
        if len(top_concepts) < 5:
            top_concepts.update(["Embedding", "Transformer", "Attention", "Encoder", "Loss Function", "Dataset", "Accuracy"])

        G = nx.Graph()
        for concept in top_concepts:
            G.add_node(concept, freq=concept_freq.get(concept, 1))

        # Add co-occurrence edges
        for concept_list in chunk_concepts:
            relevant = [c for c in concept_list if c in top_concepts]
            for i in range(len(relevant)):
                for j in range(i + 1, len(relevant)):
                    c1, c2 = relevant[i], relevant[j]
                    if G.has_edge(c1, c2):
                        G[c1][c2]["weight"] += 1
                    else:
                        G.add_edge(c1, c2, weight=1)

        node_data = [{"id": n, "label": n, "size": G.nodes[n]["freq"]} for n in G.nodes]
        return G, node_data

    def create_plotly_figure(self, G: nx.Graph) -> go.Figure:
        """Render NetworkX graph into an interactive Plotly figure."""
        pos = nx.spring_layout(G, seed=42)

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        node_text = []
        node_size = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            freq = G.nodes[node].get("freq", 1)
            node_text.append(f"Concept: {node}<br>Mentions: {freq}")
            node_size.append(15 + freq * 2)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[node for node in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale="Viridis",
                color=node_size,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=dict(text="Node Centrality", font=dict(color="white")),
                    xanchor="left",
                    tickfont=dict(color="white"),
                ),
                line_width=2,
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text="🕸️ Paper Concept & Entity Knowledge Graph",
                    font=dict(size=18, color="white"),
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                paper_bgcolor="rgba(15, 23, 42, 1)",
                plot_bgcolor="rgba(15, 23, 42, 1)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        return fig
