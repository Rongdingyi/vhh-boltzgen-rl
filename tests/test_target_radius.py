"""Target displacement RMS <= configured radius (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.target_builder import build_target


def _feats():
    N = 12
    feats = {
        "atom_to_token": torch.zeros(N, 2),
        "fake_atom_mask": torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]).float(),
        "atom_pad_mask": torch.ones(N),
        "backbone_mask": torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]).float(),
    }
    feats["atom_to_token"][0:6, 0] = 1
    feats["atom_to_token"][6:12, 1] = 1
    return feats


def test_radius_bound():
    g = torch.Generator().manual_seed(1)
    anchor = torch.randn(12, 3, generator=g)
    winner = anchor + torch.randn(12, 3, generator=torch.Generator().manual_seed(2)) * 3
    feats = _feats()
    for radius in (0.25, 0.5, 1.0):
        bt = build_target(anchor, winner, feats, [0, 1], {0: 2.0, 1: 0.0}, radius)
        if bt.touched.any():
            diff = (bt.target_coords - anchor)[bt.touched]
            rms = float((diff ** 2).sum(-1).mean().sqrt())
            assert rms <= radius + 1e-4, (radius, rms)


def test_noop_control_is_anchor():
    anchor = torch.randn(12, 3, generator=torch.Generator().manual_seed(3))
    feats = _feats()
    bt = build_target(anchor, anchor + 1.0, feats, [0, 1], {0: 1.0}, 0.5, control="noop")
    assert torch.equal(bt.target_coords, anchor)
    assert not bt.touched.any()
