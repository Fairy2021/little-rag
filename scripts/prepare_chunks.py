from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_tutor.data.chunker import chunk_document
from rag_tutor.data.io import read_text_file, write_jsonl
from rag_tutor.data.normalize import normalize_text
from rag_tutor.data.schema import Document


def make_doc_id(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def summarize_lengths(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {"min": 0, "p50": 0, "max": 0}
    return {
        "min": min(lengths),
        "p50": int(statistics.median(lengths)),
        "max": max(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest, normalize, and chunk txt/md files.")
    parser.add_argument("--input", default="orig_txt/hongloumeng.txt", help="Input .txt or .md file.")
    parser.add_argument("--out", default="rag_tutor/data/chunks.jsonl", help="Output chunks jsonl.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target max characters per chunk.")
    parser.add_argument("--overlap", type=int, default=80, help="Paragraph overlap target in characters.")
    args = parser.parse_args()

    input_path = Path(args.input)
    raw_text = read_text_file(input_path)
    normalized = normalize_text(raw_text)
    document = Document(
        doc_id=make_doc_id(input_path),
        text=normalized,
        source_path=str(input_path),
        metadata={"raw_chars": len(raw_text), "normalized_chars": len(normalized)},
    )
    chunks = chunk_document(document, chunk_size=args.chunk_size, overlap=args.overlap)
    written = write_jsonl(args.out, (chunk.to_dict() for chunk in chunks))

    lengths = [len(chunk.text) for chunk in chunks]
    length_stats = summarize_lengths(lengths)
    chapter_indexes = sorted({chunk.chapter_index for chunk in chunks if chunk.chapter_index is not None})

    print(f"input: {input_path}")
    print(f"output: {args.out}")
    print(f"chunks: {written}")
    print(f"length_min: {length_stats['min']}")
    print(f"length_p50: {length_stats['p50']}")
    print(f"length_max: {length_stats['max']}")
    print(f"chapters_detected: {len(chapter_indexes)}")
    if chunks:
        first = chunks[0]
        print(
            "first_chunk: "
            f"id={first.chunk_id} chapter_index={first.chapter_index} "
            f"paragraph_index={first.paragraph_index} chars={first.start_char}-{first.end_char}"
        )


if __name__ == "__main__":
    main()
