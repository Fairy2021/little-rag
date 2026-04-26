from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[dict]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    sources: dict[str, dict[str, dict[str, float]]] = {}

    for source_name, results in ranked_lists.items():
        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            chunks[chunk_id] = chunk
            sources.setdefault(chunk_id, {})[source_name] = {
                "rank": float(rank),
                "score": float(item["score"]),
            }

    fused_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {
            "score": scores[chunk_id],
            "chunk": chunks[chunk_id],
            "sources": sources[chunk_id],
        }
        for chunk_id in fused_ids
    ]

