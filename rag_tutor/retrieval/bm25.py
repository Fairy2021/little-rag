from __future__ import annotations

import math
import re
from collections import Counter


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._compute_idf()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = []
        for index, freqs in enumerate(self.term_freqs):
            score = self._score_doc(query_tokens, freqs, self.doc_lens[index])
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        results = []
        for index, score in scores[:top_k]:
            results.append({"score": float(score), "chunk": self.chunks[index]})
        return results

    def _compute_idf(self) -> dict[str, float]:
        doc_count = len(self.doc_tokens)
        dfs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            dfs.update(set(tokens))
        return {
            term: math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def _score_doc(self, query_tokens: list[str], freqs: Counter[str], doc_len: int) -> float:
        score = 0.0
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[a-z0-9]+", part):
            tokens.append(part)
            continue
        tokens.extend(part)
        tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tokens
