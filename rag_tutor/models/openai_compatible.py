from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

from rag_tutor.models.embedding import EmbeddingModel


class OpenAICompatibleEmbedding(EmbeddingModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        embed_url: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model or os.getenv("OPENAI_COMPAT_EMBED_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY")
        self.url = embed_url or os.getenv("OPENAI_COMPAT_EMBED_URL") or _join_url(
            base_url or os.getenv("OPENAI_COMPAT_BASE_URL", "https://api.openai.com/v1"),
            "embeddings",
        )
        self.timeout = timeout
        self._dim: int | None = None
        if not self.api_key:
            raise RuntimeError("OPENAI_COMPAT_API_KEY is required for OpenAICompatibleEmbedding")

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        data = _post_json(
            self.url,
            self.api_key,
            {"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        array = np.asarray(vectors, dtype="float32")
        self._dim = int(array.shape[1])
        return array


def _post_json(url: str, api_key: str, payload: dict, timeout: int = 60) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI-compatible request failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI-compatible request failed: {exc.reason}") from exc


def _join_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")

