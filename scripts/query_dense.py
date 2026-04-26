from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_tutor.models import FakeEmbedding, OpenAICompatibleEmbedding
from rag_tutor.retrieval import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the dense index.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--index-dir", default="rag_tutor/index/dense", help="Dense index directory.")
    parser.add_argument("--embedding", choices=["fake", "openai"], default="fake")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=256, help="Must match the build_index dimension.")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    args = parser.parse_args()

    if args.embedding == "openai":
        embedder = OpenAICompatibleEmbedding(model=args.embed_model)
    else:
        embedder = FakeEmbedding(dim=args.dim)
    index = DenseIndex.load(args.index_dir, backend=args.backend)
    results = index.search(embedder.embed_query(args.query), top_k=args.top_k)

    print(f"query: {args.query}")
    print(f"backend: {index.backend}")
    print(f"top_k: {len(results)}")
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        text = chunk["text"].replace("\n", " ")
        preview = text[:100] + ("..." if len(text) > 100 else "")
        print(
            f"{rank}. score={item['score']:.4f} "
            f"chunk_id={chunk['chunk_id']} chapter_index={chunk['chapter_index']} "
            f"chapter={chunk['chapter']}"
        )
        print(f"   {preview}")


if __name__ == "__main__":
    main()
