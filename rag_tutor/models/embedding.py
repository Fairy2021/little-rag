from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class FakeEmbedding(EmbeddingModel):
    """Deterministic lexical hash embedding for fully offline demos."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.vstack([self._embed_one(text) for text in texts]).astype("float32")
        return vectors

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little", signed=False)
            index = raw % self.dim
            weight = 1.0 + min(len(token), 4) * 0.15
            vec[index] += weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


def _tokens(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", "", text.lower())
    if not cleaned:
        return []
    chars = list(cleaned)
    bigrams = [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
    latin_words = re.findall(r"[a-z0-9]+", text.lower())
    return chars + bigrams + latin_words
