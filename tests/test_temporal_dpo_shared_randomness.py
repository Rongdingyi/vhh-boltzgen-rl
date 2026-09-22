"""Temporal arms share pair, noise, augmentation; only sigma differs (§23/§52)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _online_fixtures import make_group  # noqa: E402
from vhh_rl.online_pref.pair_builder import build_all_changed_pairs  # noqa: E402
from vhh_rl.online_pref.sigma_sampler import sample_region_sigma  # noqa: E402
from vhh_rl.online_pref.step_randomness import (  # noqa: E402
    build_update_schedule, read_schedule, sample_standard_noise_like,
    write_schedule,
)


class Stub:
    sigma_data = 16.0
    P_mean = 1.2
    P_std = 0.8
    device = "cpu"


def test_same_schedule_and_noise_across_arms(tmp_path):
    pairs = build_all_changed_pairs(make_group(), (1, 2, 3))
    schedule = build_update_schedule(pairs, updates=25, seed=7, round_index=1)
    path = tmp_path / "update_schedule_r1.json"
    write_schedule(path, schedule)
    loaded = read_schedule(path)
    assert [s.pair_id for s in loaded] == [s.pair_id for s in schedule]
    assert [s.noise_seed for s in loaded] == [s.noise_seed for s in schedule]

    spec = schedule[0]
    sigmas = {}
    noises = {}
    for region in ("full", "suffix", "prefix"):
        sigmas[region] = float(sample_region_sigma(
            Stub(), boundary_sigma=10.0, region=region, seed=spec.sigma_seed,
            device="cpu"))
        noises[region] = sample_standard_noise_like(torch.zeros(5, 3),
                                                    spec.noise_seed)
    for region in ("suffix", "prefix"):
        assert torch.equal(noises[region], noises["full"])
    assert sigmas["suffix"] <= 10.0 < sigmas["prefix"]
    assert sigmas["suffix"] != sigmas["prefix"]
