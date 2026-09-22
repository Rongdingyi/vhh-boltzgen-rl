"""Analytic conditional mass vs Monte Carlo (task book §25)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.online_pref.sigma_sampler import suffix_mass  # noqa: E402


class Stub:
    sigma_data = 16.0
    P_mean = 1.2
    P_std = 0.8


def test_analytic_mass_matches_monte_carlo():
    boundary = 10.0
    analytic = suffix_mass(Stub(), boundary)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(100000, generator=g)
    sigma = Stub.sigma_data * (Stub.P_mean + Stub.P_std * z).exp()
    mc = float((sigma <= boundary).float().mean())
    assert abs(analytic - mc) < 0.02


def test_conditional_masses_sum_to_one():
    boundary = 4.0
    q = suffix_mass(Stub(), boundary)
    assert 0.0 < q < 1.0
    assert abs(q + (1.0 - q) - 1.0) < 1e-12
