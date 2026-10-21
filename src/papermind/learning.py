"""Local, user-approved notes that supplement a paper's retrieved evidence."""

import json
from pathlib import Path


class PaperMemory:
    """Persist short learning notes per document without modifying model weights."""

    MAX_NOTE_LENGTH = 3000

    def __init__(self, path: Path):
        self.path = path

    def get_notes(self, document_id: str) -> list[str]:
        """Return saved notes for one document fingerprint."""
        data = self._read()
        notes = data.get(document_id, [])
        if not isinstance(notes, list) or not all(
            isinstance(note, str) for note in notes
        ):
            raise ValueError("Paper memory file has an invalid note format.")
        return notes

    def add_note(self, document_id: str, note: str) -> None:
        """Save an explicitly supplied note for future questions about this paper."""
        normalized_note = note.strip()
        if not normalized_note:
            raise ValueError("A learning note cannot be empty.")
        if len(normalized_note) > self.MAX_NOTE_LENGTH:
            raise ValueError(
                f"A learning note must be {self.MAX_NOTE_LENGTH} characters or fewer."
            )

        data = self._read()
        notes = data.setdefault(document_id, [])
        if normalized_note not in notes:
            notes.append(normalized_note)
        self._write(data)

    def clear_notes(self, document_id: str) -> None:
        """Remove saved notes for one paper while retaining other paper memories."""
        data = self._read()
        data.pop(document_id, None)
        self._write(data)

    def _read(self) -> dict[str, list[str]]:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(document_id, str)
            and isinstance(notes, list)
            and all(isinstance(note, str) for note in notes)
            for document_id, notes in data.items()
        ):
            raise ValueError("Paper memory file has an invalid note format.")
        return data

    def _write(self, data: dict[str, list[str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
