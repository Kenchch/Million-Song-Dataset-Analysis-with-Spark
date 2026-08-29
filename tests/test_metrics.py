from src.metrics import average_precision_at_k, ndcg_at_k, precision_at_k


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {2, 3}, 3) == 2 / 3


def test_average_precision_rewards_early_hit():
    assert average_precision_at_k([2, 1, 3], {2, 3}, 3) == (1 + 2 / 3) / 2


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k([2, 3, 1], {2, 3}, 3) == 1.0

