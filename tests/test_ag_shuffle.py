"""Adaptive weight shuffle control (task book §44/§77)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    shuffle_pair_weights, weight_entropy,
)


def test_shuffle_keeps_multiset_entropy_and_active_count():
    weights = {10: 0.5, 11: 0.3, 12: 0.2}
    diff = [10, 11, 12, 13, 14]
    out = shuffle_pair_weights(weights, diff, random.Random(0))
    assert sorted(out.values()) == sorted(weights.values())
    assert len(out) == len(weights)
    assert sum(out.values()) == 0.5 + 0.3 + 0.2
    assert weight_entropy(out) == weight_entropy(weights)
    assert set(out) <= set(diff)


def test_shuffle_changes_mapping_with_fixed_seed():
    weights = {10: 0.5, 11: 0.3, 12: 0.2}
    out = shuffle_pair_weights(weights, [10, 11, 12, 13, 14], random.Random(7))
    assert {p for p, w in out.items() if w == 0.5} != {10}


def test_shuffle_deterministic_for_same_seed():
    weights = {1: 0.25, 2: 0.75}
    a = shuffle_pair_weights(weights, [1, 2, 3, 4], random.Random(3))
    b = shuffle_pair_weights(weights, [1, 2, 3, 4], random.Random(3))
    assert a == b


def test_shuffle_full_support_is_a_real_permutation():
    """When every differing position is active the mapping must still change
    across seeds (review item 5): zeros are part of the shuffled multiset."""
    weights = {10: 0.5, 11: 0.3, 12: 0.2}
    diff = [10, 11, 12]
    mappings = {tuple(sorted(shuffle_pair_weights(weights, diff,
                                                  random.Random(seed)).items()))
                for seed in range(20)}
    assert len(mappings) > 1
    for mapping in mappings:
        assert sorted(w for _, w in mapping) == [0.2, 0.3, 0.5]
        assert len(mapping) == 3


def test_shuffle_preserves_zeros_and_histogram():
    weights = {10: 0.6, 11: 0.4}
    diff = [10, 11, 12, 13]
    out = shuffle_pair_weights(weights, diff, random.Random(11))
    assert sorted(out.values()) == [0.4, 0.6]
    assert set(out) <= set(diff)


def test_shuffle_of_empty_weights_is_empty():
    assert shuffle_pair_weights({}, [1, 2, 3], random.Random(0)) == {}


def test_shuffle_rejects_more_weights_than_positions():
    try:
        shuffle_pair_weights({1: 0.5, 2: 0.5}, [1], random.Random(0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
