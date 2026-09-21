"""All-changed pair construction (task book §6/§6.2)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch  # noqa: E402

from _online_fixtures import make_group  # noqa: E402
from vhh_rl.online_pref.pair_builder import (  # noqa: E402
    build_all_changed_pairs, changed_design_positions,
)


def test_changed_positions_only_design_positions():
    assert changed_design_positions("AAAA", "ABBA", (1, 2, 3)) == (1, 2)
    assert changed_design_positions("AAAA", "ABBA", (3,)) == ()


def test_pairs_use_all_changed_support_not_verified():
    group = make_group(rewards=(0.0, 1.0, 2.0), sequences=("AAAA", "AAAB", "ABBB"))
    pairs = build_all_changed_pairs(group, (1, 2, 3), min_reward_gap=0.30)
    assert {p.meta["peer_type"] for p in pairs} == {"worst", "median"}
    for pair in pairs:
        assert pair.changed_positions_verified is None
        assert pair.changed_positions_all == changed_design_positions(
            pair.winner_sequence, pair.loser_sequence, (1, 2, 3))
        assert pair.winner_reward > pair.loser_reward
        assert pair.reward_gap >= 0.30
        assert pair.winner_coords.shape == (5, 3)
        assert pair.meta["design_positions"] == (1, 2, 3)


def test_small_gap_or_identical_sequences_are_skipped():
    group = make_group(rewards=(1.0, 1.1, 1.2), sequences=("AAAA", "AAAB", "AABB"))
    assert build_all_changed_pairs(group, (1, 2, 3), min_reward_gap=0.30) == []
    same = make_group(rewards=(0.0, 5.0, 9.0), sequences=("AAAA", "AAAA", "AAAA"))
    assert build_all_changed_pairs(same, (1, 2, 3)) == []
