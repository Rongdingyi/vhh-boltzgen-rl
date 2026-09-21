"""Frozen update schedule and shared randomness contract (§9/§10/§53)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _online_fixtures import make_group  # noqa: E402
from vhh_rl.online_pref.pair_builder import build_all_changed_pairs  # noqa: E402
from vhh_rl.online_pref.step_randomness import (  # noqa: E402
    build_update_schedule, sample_standard_noise_like,
)


def _pairs():
    group = make_group()
    return build_all_changed_pairs(group, (1, 2, 3))


def test_schedule_is_deterministic_and_complete():
    pairs = _pairs()
    a = build_update_schedule(pairs, updates=25, seed=7, round_index=1)
    b = build_update_schedule(pairs, updates=25, seed=7, round_index=1)
    assert [s.__dict__ for s in a] == [s.__dict__ for s in b]
    assert len(a) == 25
    assert all(s.pair_id in {p.pair_id for p in pairs} for s in a)
    assert all(s.noise_seed != s.augmentation_seed for s in a)
    other = build_update_schedule(pairs, updates=25, seed=8, round_index=1)
    # schedule seeds (not the tiny pair order) are what differ across seeds
    assert other[0].sigma_seed != a[0].sigma_seed
    assert other[0].augmentation_seed != a[0].augmentation_seed


def test_same_update_same_noise_tensor_across_arms():
    coords = torch.zeros(5, 3)
    first = sample_standard_noise_like(coords, seed=123)
    second = sample_standard_noise_like(coords, seed=123)
    third = sample_standard_noise_like(coords, seed=124)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert first.shape == (1, 5, 3)
