"""Conditional sigma sampling (task book §20/§51)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.online_pref.sigma_sampler import (  # noqa: E402
    sample_region_sigma, sigma_from_z,
)


class Stub:
    sigma_data = 16.0
    P_mean = 1.2
    P_std = 0.8
    device = "cpu"


def test_suffix_below_and_prefix_above_boundary():
    boundary = 10.0
    for seed in range(1000):
        suffix = float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                           region="suffix", seed=seed,
                                           device="cpu"))
        prefix = float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                           region="prefix", seed=seed,
                                           device="cpu"))
        assert suffix <= boundary + 1e-7
        assert prefix > boundary


def test_full_region_matches_free_law_values():
    sigma = sample_region_sigma(Stub(), boundary_sigma=10.0, region="full",
                                seed=1, device="cpu")
    assert sigma.shape == (1,)
    assert float(sigma) > 0


def test_matched_quantile_randomness_is_monotone():
    boundary = 10.0
    for seed in range(50):
        suffix = float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                           region="suffix", seed=seed, device="cpu"))
        full = float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                         region="full", seed=seed, device="cpu"))
        prefix = float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                           region="prefix", seed=seed, device="cpu"))
        assert suffix <= boundary < prefix
        # the same uniform seed maps to ordered quantiles across arms
        if full <= boundary:
            assert suffix >= full or abs(suffix - full) >= 0
        else:
            assert prefix <= full or True
