from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _try_import_faiss() -> Any | None:
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:
        return None


class DenseIndex:
    def __init__(self, vectors: np.ndarray, chunks: list[dict], backend: str = "auto") -> None:
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must have the same length")
        self.vectors = _normalize(vectors.astype("float32"))
        self.chunks = chunks
        self.backend = self._choose_backend(backend)
        self._faiss_index = self._build_faiss_index() if self.backend == "faiss" else None

    @classmethod
    def build(cls, chunks: list[dict], vectors: np.ndarray, backend: str = "auto") -> "DenseIndex":
        return cls(vectors=vectors, chunks=chunks, backend=backend)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            return []
        query = _normalize(query_vector.reshape(1, -1).astype("float32"))
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"query dim {query.shape[1]} does not match index dim {self.vectors.shape[1]}"
            )
        limit = min(top_k, len(self.chunks))
        if self.backend == "faiss" and self._faiss_index is not None:
            scores, indexes = self._faiss_index.search(query, limit)
            pairs = zip(indexes[0].tolist(), scores[0].tolist(), strict=False)
        else:
            scores = self.vectors @ query[0]
            indexes = np.argsort(-scores)[:limit]
            pairs = ((int(i), float(scores[i])) for i in indexes)
        results = []
        for index, score in pairs:
            if index < 0:
                continue
            chunk = self.chunks[index]
            results.append({"score": float(score), "chunk": chunk})
        return results

    def save(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        meta = {"backend": self.backend, "dim": int(self.vectors.shape[1]), "count": len(self.chunks)}
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.backend == "faiss" and self._faiss_index is not None:
            faiss = _try_import_faiss()
            if faiss is not None:
                faiss.write_index(self._faiss_index, str(path / "faiss.index"))

    @classmethod
    def load(cls, index_dir: str | Path, backend: str = "auto") -> "DenseIndex":
        path = Path(index_dir)
        vectors = np.load(path / "vectors.npy")
        chunks = []
        with (path / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        meta_path = path / "meta.json"
        saved_backend = "auto"
        if meta_path.exists():
            saved_backend = json.loads(meta_path.read_text(encoding="utf-8")).get("backend", "auto")
        selected_backend = saved_backend if backend == "auto" else backend
        index = cls(vectors=vectors, chunks=chunks, backend=selected_backend)
        faiss_path = path / "faiss.index"
        faiss = _try_import_faiss()
        if index.backend == "faiss" and faiss is not None and faiss_path.exists():
            index._faiss_index = faiss.read_index(str(faiss_path))
        return index

    def _choose_backend(self, backend: str) -> str:
        if backend not in {"auto", "faiss", "numpy"}:
            raise ValueError("backend must be one of: auto, faiss, numpy")
        if backend == "numpy":
            return "numpy"
        if _try_import_faiss() is not None:
            return "faiss"
        if backend == "faiss":
            raise RuntimeError("faiss is not installed")
        return "numpy"

    def _build_faiss_index(self) -> Any:
        faiss = _try_import_faiss()
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        return index


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
