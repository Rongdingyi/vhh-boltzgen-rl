"""Preference weight construction tests (task book §32-§33)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.controls import (  # noqa: E402
    cf_weights, random_sparse_weights, region_weights, weight_entropy,
)

ETA = 0.75


def test_cf_weights_sum_and_floor():
    credits = {0: 2.0, 1: 0.0, 2: 2.0, 3: 0.0}
    w, fallback = cf_weights(credits, n_diff=4)
    assert not fallback
    assert abs(sum(w.values()) - 1.0) < 1e-9
    floor = (1 - ETA) / 4
    assert abs(w[1] - floor) < 1e-12
    assert w[0] > w[1] and abs(w[0] - w[2]) < 1e-12


def test_cf_weights_fallback_uniform():
    credits = {0: 0.0, 1: -1.0, 2: 0.0}
    w, fallback = cf_weights(credits, n_diff=3)
    assert fallback
    assert all(abs(v - 1 / 3) < 1e-12 for v in w.values())


def test_random_sparse_selection_and_count():
    import random

    credits = {i: (1.0 if i < 3 else 0.0) for i in range(10)}
    w = random_sparse_weights(credits, n_diff=10, rng=random.Random(0))
    assert len(w) == 3
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(abs(v - 1 / 3) < 1e-12 for v in w.values())


def test_region_weights_uniform_inside_region():
    credits = {"cdr1": 1.0, "cdr2": 1.0, "cdr3": 2.0}
    positions = {"cdr1": [0, 1], "cdr2": [2], "cdr3": [3, 4, 5, 6]}
    w = region_weights(credits, positions)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w[0] - w[1]) < 1e-12
    assert abs(w[0] - 0.25 / 2) < 1e-9
    assert abs(w[3] - 0.5 / 4) < 1e-9


def test_weight_entropy():
    assert weight_entropy({0: 1.0}) == 0.0
    assert abs(weight_entropy({0: 0.5, 1: 0.5}) - 0.693147) < 1e-5
