"""Target == anchor implies zero fit loss at initialization (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.target_builder import build_target


def test_noop_target_loss_zero():
    anchor = torch.randn(20, 3, generator=torch.Generator().manual_seed(6))
    feats = {"atom_to_token": torch.zeros(20, 1), "fake_atom_mask": torch.ones(20),
             "atom_pad_mask": torch.ones(20)}
    feats["atom_to_token"][:, 0] = 1
    bt = build_target(anchor, anchor, feats, [0], {0: 1.0}, 0.5, control="noop")
    diff = (bt.target_coords - anchor) ** 2
    assert float(diff.max()) == 0.0


def test_ema_imports():
    from vhh_rl.cf_opsd.behavior_ema import ema_update  # noqa: F401
