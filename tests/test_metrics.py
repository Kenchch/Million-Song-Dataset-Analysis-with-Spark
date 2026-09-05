from src.metrics import average_precision_at_k, ndcg_at_k, precision_at_k
import pytest
from itertools import repeat


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {2, 3}, 3) == 2 / 3


def test_average_precision_rewards_early_hit():
    assert average_precision_at_k([2, 1, 3], {2, 3}, 3) == (1 + 2 / 3) / 2


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k([2, 3, 1], {2, 3}, 3) == 1.0


@pytest.mark.parametrize("metric", [precision_at_k, average_precision_at_k, ndcg_at_k])
@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_cutoff(metric, k):
    with pytest.raises(ValueError, match="positive integer"):
        metric([], set(), k)


@pytest.mark.parametrize("metric", [precision_at_k, average_precision_at_k, ndcg_at_k])
def test_duplicates_cannot_inflate_scores(metric):
    assert 0 <= metric([1, 1, 1], {1}, 3) <= 1
    assert metric([1, 1, 2], {1, 2}, 3) == metric([1, 99, 2], {1, 2}, 3)
    assert metric([], {1}, 3) == 0
    assert metric([1], set(), 3) == 0
    assert metric(repeat(1), {1}, 3) == metric([1, 1, 1], {1}, 3)

