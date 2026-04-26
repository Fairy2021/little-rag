from __future__ import annotations

import re
from dataclasses import dataclass

from rag_tutor.data.schema import Chunk, Document


CHAPTER_RE = re.compile(
    r"^(第[一二三四五六七八九十百千万零〇两\d]+回\s*[^\n]*)$|^(#{1,6}\s+.+)$"
)


@dataclass(slots=True)
class Paragraph:
    text: str
    start_char: int
    end_char: int
    paragraph_index: int
    chapter: str | None
    chapter_index: int | None


def detect_chapter(line: str) -> str | None:
    match = CHAPTER_RE.match(line.strip())
    if not match:
        return None
    title = match.group(1) or match.group(2) or ""
    return title.lstrip("#").strip()


def split_paragraphs(text: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    chapter: str | None = None
    chapter_index: int | None = None
    next_chapter_index = 0

    for idx, match in enumerate(re.finditer(r"\S(?:.*\S)?", text)):
        para_text = match.group(0).strip()
        title = detect_chapter(para_text)
        if title:
            chapter = title
            chapter_index = next_chapter_index
            next_chapter_index += 1
        paragraphs.append(
            Paragraph(
                text=para_text,
                start_char=match.start(),
                end_char=match.end(),
                paragraph_index=idx,
                chapter=chapter,
                chapter_index=chapter_index,
            )
        )
    return paragraphs


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = split_paragraphs(document.text)
    chunks: list[Chunk] = []
    current: list[Paragraph] = []

    def make_chunk(parts: list[Paragraph]) -> Chunk:
        chunk_text = "\n\n".join(part.text for part in parts)
        first = parts[0]
        last = parts[-1]
        chunk_id = f"{document.doc_id}:{len(chunks):05d}"
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            text=chunk_text,
            chapter=first.chapter,
            chapter_index=first.chapter_index,
            paragraph_index=first.paragraph_index,
            start_char=first.start_char,
            end_char=last.end_char,
            metadata={
                "source_path": document.source_path,
                "paragraph_count": len(parts),
            },
        )

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        flushed = current
        chunks.append(make_chunk(flushed))
        current = _tail_overlap_paragraphs(flushed, overlap)

    for para in paragraphs:
        if len(para.text) > chunk_size:
            flush_current()
            current = []
            chunks.extend(_split_long_paragraph(document, para, chunk_size, overlap, len(chunks)))
            continue

        candidate_len = _joined_len(current + [para])
        if current and candidate_len > chunk_size:
            flush_current()
            if current and _joined_len(current + [para]) > chunk_size:
                current = []
        current.append(para)

    if current:
        chunks.append(make_chunk(current))
    return chunks


def _joined_len(parts: list[Paragraph]) -> int:
    if not parts:
        return 0
    return sum(len(part.text) for part in parts) + 2 * (len(parts) - 1)


def _tail_overlap_paragraphs(parts: list[Paragraph], overlap: int) -> list[Paragraph]:
    if overlap <= 0:
        return []
    selected: list[Paragraph] = []
    total = 0
    for part in reversed(parts):
        selected.append(part)
        total += len(part.text)
        if total >= overlap:
            break
    return list(reversed(selected))


def _split_long_paragraph(
    document: Document,
    para: Paragraph,
    chunk_size: int,
    overlap: int,
    start_index: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    offset = 0
    while offset < len(para.text):
        piece = para.text[offset : offset + chunk_size]
        chunk_id = f"{document.doc_id}:{start_index + len(chunks):05d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                text=piece,
                chapter=para.chapter,
                chapter_index=para.chapter_index,
                paragraph_index=para.paragraph_index,
                start_char=para.start_char + offset,
                end_char=para.start_char + offset + len(piece),
                metadata={
                    "source_path": document.source_path,
                    "paragraph_count": 1,
                    "split_long_paragraph": True,
                },
            )
        )
        if offset + chunk_size >= len(para.text):
            break
        offset += step
    return chunks
