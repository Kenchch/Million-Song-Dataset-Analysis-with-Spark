"""Small, dependency-free ranking metrics for held-out implicit-feedback items."""

from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    top_k = list(recommendations)[:k]
    return 0.0 if not top_k else len(set(top_k) & relevant) / k


def average_precision_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    hits, score = 0, 0.0
    for rank, item in enumerate(list(recommendations)[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommendations: Iterable[int], relevant: set[int], k: int = 10) -> float:
    actual = sum(1 / math.log2(rank + 1) for rank, item in enumerate(list(recommendations)[:k], start=1) if item in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return actual / ideal if ideal else 0.0

