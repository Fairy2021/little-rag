from __future__ import annotations

import math
import statistics


def dcg_at_k(ranked_ids: list[str], gains: dict[str, int], k: int) -> float:
    total = 0.0
    for i, chunk_id in enumerate(ranked_ids[:k], start=1):
        gain = gains.get(chunk_id, 0)
        total += (2**gain - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_ids: list[str], gains: dict[str, int], k: int) -> float:
    ideal_ids = sorted(gains, key=gains.get, reverse=True)
    ideal = dcg_at_k(ideal_ids, gains, k)
    return dcg_at_k(ranked_ids, gains, k) / ideal if ideal > 0 else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def latency_stats(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }

