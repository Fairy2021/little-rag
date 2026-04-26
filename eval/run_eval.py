from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_tutor.eval.runner import build_report, generation_grounding_eval, retrieval_eval

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all eval layers and write eval_report.md.")
    parser.add_argument("--qa", default="eval/demo_qa.jsonl")
    parser.add_argument("--index-dir", default="rag_tutor/index/demo_dense")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="eval_report.md")
    parser.add_argument("--judge", choices=["keyword", "llm"], default="keyword")
    parser.add_argument("--support-judge", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()
    retrieval = retrieval_eval(args.qa, args.index_dir, args.dim, args.k)
    generation = generation_grounding_eval(
        args.qa,
        args.index_dir,
        args.dim,
        args.k,
        judge_kind=args.judge,
        support_judge_kind=args.support_judge,
        judge_model=args.judge_model,
    )
    report = build_report(retrieval, generation)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
