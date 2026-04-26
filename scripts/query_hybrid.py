from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_tutor.models import FakeEmbedding, OpenAICompatibleEmbedding
from rag_tutor.retrieval import BM25Index, DenseIndex, reciprocal_rank_fusion


def print_results(title: str, results: list[dict], show_sources: bool = False) -> None:
    print(f"\n[{title}]")
    if not results:
        print("no results")
        return
    for rank, item in enumerate(results, start=1):
        chunk = item["chunk"]
        preview = chunk["text"].replace("\n", " ")
        preview = preview[:100] + ("..." if len(preview) > 100 else "")
        source_text = ""
        if show_sources:
            source_text = " sources=" + ",".join(sorted(item.get("sources", {}).keys()))
        print(
            f"{rank}. score={item['score']:.4f} "
            f"chunk_id={chunk['chunk_id']} chapter_index={chunk['chapter_index']}"
            f"{source_text}"
        )
        print(f"   chapter={chunk['chapter']}")
        print(f"   {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, BM25, and RRF hybrid retrieval.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--index-dir", default="rag_tutor/index/dense", help="Dense index directory.")
    parser.add_argument("--embedding", choices=["fake", "openai"], default="fake")
    parser.add_argument("--dense-k", type=int, default=8)
    parser.add_argument("--bm25-k", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dim", type=int, default=256, help="Must match build_index dimension.")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    args = parser.parse_args()

    dense_index = DenseIndex.load(args.index_dir, backend=args.backend)
    if args.embedding == "openai":
        embedder = OpenAICompatibleEmbedding(model=args.embed_model)
    else:
        embedder = FakeEmbedding(dim=args.dim)
    dense_results = dense_index.search(embedder.embed_query(args.query), top_k=args.dense_k)

    bm25_index = BM25Index(dense_index.chunks)
    bm25_results = bm25_index.search(args.query, top_k=args.bm25_k)

    hybrid_results = reciprocal_rank_fusion(
        {"dense": dense_results, "bm25": bm25_results},
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )

    print(f"query: {args.query}")
    print(f"dense_backend: {dense_index.backend}")
    print_results("dense", dense_results[: args.top_k])
    print_results("bm25", bm25_results[: args.top_k])
    print_results("hybrid_rrf", hybrid_results, show_sources=True)


if __name__ == "__main__":
    main()
