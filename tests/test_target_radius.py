"""Target displacement RMS <= configured radius (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.target_builder import build_target


def _feats():
    feats = {
        "atom_to_token": torch.zeros(20, 2),
        "fake_atom_mask": torch.tensor([0, 0, 1, 1, 1, 1] + [0] * 14).float(),
        "atom_pad_mask": torch.ones(20),
        "backbone_mask": torch.tensor([1, 1, 1, 0, 0, 0] + [0] * 14).float(),
    }
    feats["atom_to_token"][0:6, 0] = 1
    feats["atom_to_token"][6:20, 1] = 1
    return feats


def test_radius_bound():
    g = torch.Generator().manual_seed(1)
    anchor = torch.randn(20, 3, generator=g)
    winner = anchor + torch.randn(20, 3, generator=torch.Generator().manual_seed(2)) * 3
    feats = _feats()
    for radius in (0.25, 0.5, 1.0):
        bt = build_target(anchor, winner, feats, [0, 1], {0: 2.0, 1: 0.0}, radius)
        if bt.touched.any():
            diff = (bt.target_coords - anchor)[bt.touched]
            rms = float((diff ** 2).sum(-1).mean().sqrt())
            assert rms <= radius + 1e-4, (radius, rms)


def test_noop_control_is_anchor():
    anchor = torch.randn(20, 3, generator=torch.Generator().manual_seed(3))
    feats = _feats()
    bt = build_target(anchor, anchor + 1.0, feats, [0, 1], {0: 1.0}, 0.5, control="noop")
    assert torch.equal(bt.target_coords, anchor)
    assert not bt.touched.any()
