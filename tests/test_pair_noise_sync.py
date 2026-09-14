"""Paired noising sync (task book §49)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.paired_noise import paired_noising  # noqa: E402


def test_shared_sigma_noise_transform():
    torch.manual_seed(0)
    n = 40
    w = torch.randn(n, 3)
    l = torch.randn(n, 3)
    mask = torch.ones(n, dtype=torch.bool)
    sigma = torch.tensor([1.7])
    noise = torch.randn(1, n, 3)
    out = paired_noising(w, l, mask, sigma, noise=noise)
    assert torch.equal(out["noise"], noise)
    assert torch.allclose(
        out["x_t_w"], out["x0_w_aug"] + sigma.reshape(1, 1, 1) * noise, atol=1e-6)
    assert torch.allclose(
        out["x_t_l"], out["x0_l_aug"] + sigma.reshape(1, 1, 1) * noise, atol=1e-6)
    # identical inputs share the exact same transform
    twin = paired_noising(w, w, mask, sigma, noise=noise)
    assert torch.allclose(twin["x0_w_aug"], twin["x0_l_aug"], atol=1e-6)
    # different inputs must not be identical after the same transform
    assert not torch.allclose(out["x0_w_aug"], out["x0_l_aug"], atol=1e-4)
