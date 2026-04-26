from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Apply light cleanup while preserving paragraph and chapter boundaries."""

    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")
    text = text.replace("\t", " ")
    lines = [re.sub(r"[ ]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"

