from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    """A retrievable text unit with source location metadata."""

    chunk_id: str
    doc_id: str
    text: str
    chapter: str | None
    chapter_index: int | None
    paragraph_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Evidence:
    """A retrieved chunk plus the scores used to select it."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    chapter: str | None = None
    chapter_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    """A generated answer that can be traced back to evidence."""

    question: str
    answer: str
    evidences: list[Evidence]
    citations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

