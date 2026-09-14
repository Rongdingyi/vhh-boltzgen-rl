"""Only credited CDR fake atoms may be modified (task book §82)."""
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


def test_only_credited_atoms_change():
    g = torch.Generator().manual_seed(4)
    anchor = torch.randn(12, 3, generator=g)
    winner = anchor + torch.randn(12, 3, generator=torch.Generator().manual_seed(5))
    feats = _feats()
    bt = build_target(anchor, winner, feats, [0, 1], {0: 1.0, 1: 0.0}, 1.0)
    changed = (bt.target_coords - anchor).abs().sum(-1) > 1e-6
    assert set(torch.where(changed)[0].tolist()) <= set(torch.where(bt.touched)[0].tolist())
    # token 0 has only backbone atoms -> no change allowed there
    assert not changed[0:3].any()
