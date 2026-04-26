from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_tutor.data.io import read_jsonl
from rag_tutor.models import FakeEmbedding, OpenAICompatibleEmbedding
from rag_tutor.retrieval import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense vector index from chunks.jsonl.")
    parser.add_argument("--chunks", default="rag_tutor/data/chunks.jsonl", help="Input chunks jsonl.")
    parser.add_argument("--out-dir", default="rag_tutor/index/dense", help="Output index directory.")
    parser.add_argument("--embedding", choices=["fake", "openai"], default="fake")
    parser.add_argument("--dim", type=int, default=256, help="FakeEmbedding dimension.")
    parser.add_argument("--embed-model", default=None, help="OpenAI-compatible embedding model.")
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    if args.embedding == "openai":
        embedder = OpenAICompatibleEmbedding(model=args.embed_model)
    else:
        embedder = FakeEmbedding(dim=args.dim)
    vectors = embedder.embed_texts([chunk["text"] for chunk in chunks])
    index = DenseIndex.build(chunks=chunks, vectors=vectors, backend=args.backend)
    index.save(args.out_dir)

    print(f"chunks: {len(chunks)}")
    print(f"embedding: {embedder.__class__.__name__} dim={embedder.dim}")
    print(f"backend: {index.backend}")
    print(f"index_dir: {args.out_dir}")


if __name__ == "__main__":
    main()
