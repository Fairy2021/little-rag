from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rag_tutor.data.schema import Evidence
from rag_tutor.retrieval.bm25 import tokenize


class LLM(ABC):
    @abstractmethod
    def generate(self, question: str, evidences: list[Evidence], prompt: str) -> str:
        raise NotImplementedError


class DummyLLM(LLM):
    """Offline LLM replacement that echoes evidence-backed snippets."""

    def generate(self, question: str, evidences: list[Evidence], prompt: str) -> str:
        del prompt
        query_terms = _content_terms(question)
        evidence_terms = set()
        for evidence in evidences:
            evidence_terms.update(tokenize(evidence.text))
        if query_terms and len(query_terms - evidence_terms) / len(query_terms) > 0.35:
            return "不知道。"
        lines = []
        for evidence in evidences:
            sentence, score = _best_sentence(evidence.text, query_terms)
            if sentence and score > 0:
                lines.append(f"- {sentence} [{evidence.chunk_id}]")
        if not lines:
            return "不知道。"
        return "\n".join(lines)


def _content_terms(text: str) -> set[str]:
    stop = set("的了呢吗么谁何何人什么为何如何怎样怎么是否有在上后中")
    return {term for term in tokenize(text) if term not in stop}


def _best_sentence(text: str, query_terms: set[str]) -> tuple[str, int]:
    rough_parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r"^第.+回", line):
            continue
        rough_parts.extend(re.split(r"(?<=[。！？!?])", line))
    sentences = [part.strip() for part in rough_parts if part.strip()]
    if not sentences:
        sentences = [text.strip()]
    scored = []
    for sent in sentences:
        terms = set(tokenize(sent))
        score = len(query_terms & terms)
        scored.append((score, len(sent), sent))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best = scored[0][2]
    return best[:160].rstrip(), scored[0][0]

