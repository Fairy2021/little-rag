from __future__ import annotations

from collections import Counter

from rag_tutor.retrieval.bm25 import tokenize


def heuristic_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    query_terms = Counter(tokenize(query))
    if not query_terms:
        return candidates[:top_k]
    reranked = []
    for item in candidates:
        chunk = item["chunk"]
        text_terms = Counter(tokenize(chunk["text"]))
        overlap = sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items())
        coverage = overlap / max(sum(query_terms.values()), 1)
        source_bonus = 0.03 * len(item.get("sources", {}))
        final_score = float(item["score"]) + coverage + source_bonus
        new_item = dict(item)
        new_item["rerank_score"] = final_score
        new_item["features"] = {"query_coverage": coverage, "source_bonus": source_bonus}
        reranked.append(new_item)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

