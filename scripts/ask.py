from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_tutor.generation import DummyLLM, OpenAICompatibleLLM, answer_question
from rag_tutor.models import FakeEmbedding, OpenAICompatibleEmbedding
from rag_tutor.retrieval import BM25Index, DenseIndex, heuristic_rerank, reciprocal_rank_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question with hybrid retrieval + rerank + citations.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", default="rag_tutor/index/dense")
    parser.add_argument("--embedding", choices=["fake", "openai"], default="fake")
    parser.add_argument("--llm", choices=["dummy", "openai"], default="dummy")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--backend", choices=["auto", "faiss", "numpy"], default="auto")
    parser.add_argument("--dense-k", type=int, default=12)
    parser.add_argument("--bm25-k", type=int, default=12)
    parser.add_argument("--fusion-k", type=int, default=10)
    parser.add_argument("--rerank-k", type=int, default=4)
    parser.add_argument("--context-chars", type=int, default=1800)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dense_index = DenseIndex.load(args.index_dir, backend=args.backend)
    if args.embedding == "openai":
        embedder = OpenAICompatibleEmbedding(model=args.embed_model)
    else:
        embedder = FakeEmbedding(dim=args.dim)
    llm = OpenAICompatibleLLM(model=args.chat_model) if args.llm == "openai" else DummyLLM()
    dense = dense_index.search(embedder.embed_query(args.question), top_k=args.dense_k)
    bm25 = BM25Index(dense_index.chunks).search(args.question, top_k=args.bm25_k)
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, top_k=args.fusion_k)
    reranked = heuristic_rerank(args.question, fused, top_k=args.rerank_k)
    answer = answer_question(args.question, reranked, llm, max_context_chars=args.context_chars)

    if args.json:
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"question: {answer.question}")
    print("\nanswer:")
    print(answer.answer)
    print("\ncitations:")
    print(", ".join(answer.citations) if answer.citations else "none")
    print("\nevidence:")
    for idx, evidence in enumerate(answer.evidences, start=1):
        print(
            f"{idx}. chunk_id={evidence.chunk_id} score={evidence.score:.4f} "
            f"chapter_index={evidence.chapter_index} chapter={evidence.chapter}"
        )
    if args.show_prompt:
        print("\nprompt:")
        print(answer.metadata["prompt"])


if __name__ == "__main__":
    main()
