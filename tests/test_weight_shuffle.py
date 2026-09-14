"""Weight-shuffle / control integrity tests (task book §37 B1/B2)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.controls import (  # noqa: E402
    cf_weights, random_sparse_weights, shuffle_weights,
)


def test_shuffle_preserves_multiset_and_sum():
    w = {0: 0.5, 1: 0.3, 2: 0.2}
    for seed in range(10):
        s = shuffle_weights(w, random.Random(seed))
        assert sorted(s.values()) == sorted(w.values())
        assert abs(sum(s.values()) - 1.0) < 1e-9


def test_shuffle_does_not_add_or_remove_positions():
    w = {i: 1 / 5 for i in range(5)}
    s = shuffle_weights(w, random.Random(1))
    assert set(s) == set(w)


def test_random_sparse_never_exceeds_positions():
    credits = {i: float(i % 3) for i in range(12)}
    w = random_sparse_weights(credits, n_diff=12, rng=random.Random(2))
    assert set(w) <= set(credits)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    true_w, _ = cf_weights(credits, n_diff=12)
    assert set(true_w) == set(credits)
