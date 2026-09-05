"""Small, dependency-free ranking metrics for held-out implicit-feedback items."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import islice


def _ranked_hits(recommendations: Iterable[int], relevant: set[int], k: int):
    """Credit an item only at its first rank; duplicate slots still cost rank."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    seen = set()
    for rank, item in enumerate(islice(recommendations, k), start=1):
        if item in relevant and item not in seen:
            yield rank
        seen.add(item)


def precision_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    return sum(1 for _ in _ranked_hits(recommendations, relevant, k)) / k


def average_precision_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    hits, score = 0, 0.0
    for rank in _ranked_hits(recommendations, relevant, k):
        hits += 1
        score += hits / rank
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    actual = sum(1 / math.log2(rank + 1) for rank in _ranked_hits(recommendations, relevant, k))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return actual / ideal if ideal else 0.0

