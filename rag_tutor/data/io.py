from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".txt", ".md"}


def read_text_file(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Use one of: {allowed}")
    return file_path.read_text(encoding="utf-8-sig")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

