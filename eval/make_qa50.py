from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_tutor.data.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 50-item QA bootstrap file from chunks.")
    parser.add_argument("--chunks", default="rag_tutor/data/chunks.jsonl")
    parser.add_argument("--out", default="eval/qa_50.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    chunks = read_jsonl(args.chunks)
    rows = []
    for chunk in chunks:
        sentence = _first_sentence(chunk["text"])
        if len(sentence) < 18 or len(sentence) > 120:
            continue
        rows.append(_row(len(rows) + 1, chunk, sentence))
        if len(rows) >= args.limit:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote: {args.out}")
    print(f"items: {len(rows)}")


def _first_sentence(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r"^第.+回", line):
            continue
        parts = [p.strip() for p in re.split(r"(?<=[。！？!?])", line) if p.strip()]
        if parts:
            return parts[0]
    return ""


def _row(i: int, chunk: dict, sentence: str) -> dict:
    chapter = chunk.get("chapter") or "未知章节"
    return {
        "id": f"auto_{i:03d}",
        "question": f"这段情节出自哪一回：{sentence[:40]}",
        "answer": chapter,
        "expected_keywords": [chapter[: min(len(chapter), 8)]],
        "relevant_chunk_ids": [chunk["chunk_id"]],
        "relevance": {chunk["chunk_id"]: 3},
        "question_type": "chapter_location",
        "difficulty": "easy",
        "notes": "Auto-generated bootstrap item; review before treating as gold.",
    }


if __name__ == "__main__":
    main()

