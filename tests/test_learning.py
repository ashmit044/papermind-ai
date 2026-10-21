"""Tests for explicitly approved per-paper memory."""

import pytest

from papermind.learning import PaperMemory


def test_paper_memory_persists_notes_per_document(tmp_path):
    memory = PaperMemory(tmp_path / "nested" / "paper-memory.json")
    memory.add_note("paper-a", "The authors use a held-out test set.")
    memory.add_note("paper-a", "  The authors use a held-out test set.  ")
    memory.add_note("paper-b", "Check the appendix for dataset details.")

    assert memory.get_notes("paper-a") == [
        "The authors use a held-out test set."
    ]
    memory.clear_notes("paper-a")
    assert memory.get_notes("paper-a") == []
    assert memory.get_notes("paper-b") == ["Check the appendix for dataset details."]


def test_paper_memory_rejects_empty_or_oversized_notes(tmp_path):
    memory = PaperMemory(tmp_path / "paper-memory.json")

    with pytest.raises(ValueError, match="cannot be empty"):
        memory.add_note("paper-a", "  ")
    with pytest.raises(ValueError, match="3000 characters or fewer"):
        memory.add_note("paper-a", "x" * 3001)


def test_paper_memory_rejects_malformed_existing_data(tmp_path):
    memory_file = tmp_path / "paper-memory.json"
    memory_file.write_text('{"paper-a": "not a list"}', encoding="utf-8")
    memory = PaperMemory(memory_file)

    with pytest.raises(ValueError, match="invalid note format"):
        memory.get_notes("paper-a")
