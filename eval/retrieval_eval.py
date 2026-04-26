from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_tutor.eval.runner import retrieval_eval

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval Recall@k, MRR, and latency.")
    parser.add_argument("--qa", default="eval/demo_qa.jsonl")
    parser.add_argument("--index-dir", default="rag_tutor/index/demo_dense")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(retrieval_eval(args.qa, args.index_dir, args.dim, args.k), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
