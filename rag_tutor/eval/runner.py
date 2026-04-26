from __future__ import annotations

import time

from rag_tutor.data.io import read_jsonl
from rag_tutor.eval.judge import Judge, make_judge
from rag_tutor.eval.metrics import latency_stats, ndcg_at_k
from rag_tutor.generation import DummyLLM, answer_question
from rag_tutor.models import FakeEmbedding
from rag_tutor.retrieval import BM25Index, DenseIndex, heuristic_rerank, reciprocal_rank_fusion
from rag_tutor.retrieval.bm25 import tokenize


def retrieve(
    index: DenseIndex,
    embedder: FakeEmbedding,
    question: str,
    k: int = 5,
    bm25_index: BM25Index | None = None,
) -> tuple[list[dict], float]:
    start = time.perf_counter()
    dense = index.search(embedder.embed_query(question), top_k=max(12, k))
    bm25 = (bm25_index or BM25Index(index.chunks)).search(question, top_k=max(12, k))
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, top_k=max(10, k))
    return heuristic_rerank(question, fused, top_k=k), (time.perf_counter() - start) * 1000


def retrieval_eval(qa_path: str, index_dir: str, dim: int = 256, k: int = 5) -> dict:
    rows = _load_qa(qa_path)
    index = DenseIndex.load(index_dir)
    embedder = FakeEmbedding(dim=dim)
    bm25_index = BM25Index(index.chunks)
    hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0
    latencies = []
    failures = []
    for row in rows:
        results, latency = retrieve(index, embedder, row["question"], k=k, bm25_index=bm25_index)
        latencies.append(latency)
        ranked_ids = [item["chunk"]["chunk_id"] for item in results]
        gains = _gains(row)
        relevant = {cid for cid, gain in gains.items() if gain > 0}
        first_rank = next((i + 1 for i, cid in enumerate(ranked_ids) if cid in relevant), None)
        hits += int(first_rank is not None)
        rr_sum += 1 / first_rank if first_rank else 0
        ndcg_sum += ndcg_at_k(ranked_ids, gains, k)
        if not first_rank:
            failures.append({"question": row["question"], "expected": sorted(relevant), "got": ranked_ids})
    total = max(len(rows), 1)
    stats = latency_stats(latencies)
    return {
        "total": len(rows),
        "recall_at_k": hits / total,
        "mrr": rr_sum / total,
        "ndcg_at_k": ndcg_sum / total,
        "latency_ms_avg": stats["avg"],
        "latency_ms_p50": stats["p50"],
        "latency_ms_p95": stats["p95"],
        "failures": failures,
    }


def generation_grounding_eval(
    qa_path: str,
    index_dir: str,
    dim: int = 256,
    k: int = 4,
    judge_kind: str = "keyword",
    support_judge_kind: str = "heuristic",
    judge_model: str | None = None,
) -> dict:
    rows = _load_qa(qa_path)
    index = DenseIndex.load(index_dir)
    embedder = FakeEmbedding(dim=dim)
    bm25_index = BM25Index(index.chunks)
    judge = make_judge(judge_kind, judge_model)
    support_judge = make_judge("llm", judge_model) if support_judge_kind == "llm" else None
    correct = cited = total_citations = unsupported_h = unsupported_llm = total_claims = 0
    failures = []
    for row in rows:
        results, _ = retrieve(index, embedder, row["question"], k=k, bm25_index=bm25_index)
        answer = answer_question(row["question"], results, DummyLLM())
        ref = " ".join(row.get("expected_keywords", [])) if judge_kind == "keyword" else row.get("answer", "")
        verdict = judge.score_answer(row["question"], answer.answer, ref)
        correct += int(verdict.get("score", 0))
        evidence_ids = {ev.chunk_id for ev in answer.evidences}
        relevant_ids = set(row.get("relevant_chunk_ids", []))
        for cid in answer.citations:
            total_citations += 1
            cited += int(cid in evidence_ids and (not relevant_ids or cid in relevant_ids))
        for line in _claim_lines(answer.answer):
            total_claims += 1
            cid = _citation_id(line)
            ev_text = next((ev.text for ev in answer.evidences if ev.chunk_id == cid), "")
            unsupported_h += int(not _supported(line, ev_text))
            if support_judge:
                unsupported_llm += int(not support_judge.score_support(line, ev_text).get("supported", 0))
        if not verdict.get("score", 0):
            failures.append({"question": row["question"], "answer": answer.answer, "reason": verdict["reason"]})
    total = max(len(rows), 1)
    return {
        "total": len(rows),
        "judge": judge_kind,
        "support_judge": support_judge_kind,
        "accuracy": correct / total,
        "citation_accuracy": cited / max(total_citations, 1),
        "unsupported_claim_rate": unsupported_h / max(total_claims, 1),
        "unsupported_claim_rate_llm": unsupported_llm / max(total_claims, 1) if support_judge else None,
        "failures": failures,
    }


def build_report(retrieval: dict, generation: dict) -> str:
    lines = [
        "# Eval Report",
        "",
        "## Retrieval",
        f"- total: {retrieval['total']}",
        f"- Recall@k: {retrieval['recall_at_k']:.3f}",
        f"- MRR: {retrieval['mrr']:.3f}",
        f"- nDCG@k: {retrieval['ndcg_at_k']:.3f}",
        f"- latency_ms_avg: {retrieval['latency_ms_avg']:.2f}",
        f"- latency_ms_p50: {retrieval['latency_ms_p50']:.2f}",
        f"- latency_ms_p95: {retrieval['latency_ms_p95']:.2f}",
        "",
        "## Generation",
        f"- judge: {generation['judge']}",
        f"- judge_accuracy: {generation['accuracy']:.3f}",
        "",
        "## Grounding",
        f"- citation_accuracy: {generation['citation_accuracy']:.3f}",
        f"- unsupported_claim_rate_heuristic: {generation['unsupported_claim_rate']:.3f}",
        f"- unsupported_claim_rate_llm: {generation['unsupported_claim_rate_llm']}",
        "",
        "## Failure Samples",
    ]
    failures = retrieval["failures"] + generation["failures"]
    lines.extend([f"- {item}" for item in failures[:10]] or ["- none"])
    return "\n".join(lines) + "\n"


def _load_qa(path: str) -> list[dict]:
    return [row for row in read_jsonl(path) if row.get("question")]


def _gains(row: dict) -> dict[str, int]:
    if row.get("relevance"):
        return {cid: int(gain) for cid, gain in row["relevance"].items()}
    return {cid: 1 for cid in row.get("relevant_chunk_ids", [])}


def _claim_lines(answer: str) -> list[str]:
    if answer.strip() == "不知道。":
        return []
    return [line.strip("- ").strip() for line in answer.splitlines() if line.strip()]


def _citation_id(line: str) -> str:
    if "[" not in line or "]" not in line:
        return ""
    return line.rsplit("[", 1)[-1].split("]", 1)[0]


def _supported(claim: str, evidence: str) -> bool:
    claim_terms = {t for t in tokenize(claim) if len(t) > 1}
    evidence_terms = set(tokenize(evidence))
    return not claim_terms or len(claim_terms - evidence_terms) / len(claim_terms) <= 0.25
