"""Prefix/suffix domains are disjoint (task book §25)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.online_pref.sigma_sampler import sample_region_sigma  # noqa: E402


class Stub:
    sigma_data = 16.0
    P_mean = 1.2
    P_std = 0.8
    device = "cpu"


def test_domains_are_strictly_disjoint():
    boundary = 7.5
    suffix = [float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                        region="suffix", seed=s, device="cpu"))
              for s in range(300)]
    prefix = [float(sample_region_sigma(Stub(), boundary_sigma=boundary,
                                        region="prefix", seed=s, device="cpu"))
              for s in range(300)]
    assert max(suffix) <= boundary
    assert min(prefix) > boundary
    assert max(suffix) < min(prefix)
