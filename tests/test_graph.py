"""Tests for source-grounded concept graph construction."""

from papermind.graph import ConceptGraphExtractor
from papermind.parser import TextChunk


def make_chunk(text: str) -> TextChunk:
    return TextChunk(
        chunk_id=0,
        text=text,
        page_num=1,
        section_name="Methods",
        char_count=len(text),
    )


def test_graph_does_not_inject_placeholder_concepts():
    graph, nodes = ConceptGraphExtractor().build_graph(
        [make_chunk("all words in this document are lowercase and generic")]
    )

    assert graph.number_of_nodes() == 0
    assert nodes == []


def test_graph_contains_only_extracted_terms_and_labels_frequency_truthfully():
    graph, nodes = ConceptGraphExtractor(top_n_concepts=10).build_graph(
        [make_chunk("Transformer FAISS Retrieval Transformer")]
    )

    assert set(graph.nodes) == {"Transformer", "FAISS", "Retrieval"}
    assert len(nodes) == 3
    assert graph.number_of_edges() == 3

    figure = ConceptGraphExtractor().create_plotly_figure(graph)
    node_trace = figure.data[1]
    assert list(node_trace.marker.color) == [1, 1, 1]
    assert node_trace.marker.colorbar.title.text == "Chunks Containing Concept"


def test_graph_requires_positive_concept_limit():
    for value in (0, -1, True):
        try:
            ConceptGraphExtractor(top_n_concepts=value)
        except ValueError as error:
            assert "positive integer" in str(error)
        else:
            raise AssertionError("Invalid concept limit should raise ValueError")
