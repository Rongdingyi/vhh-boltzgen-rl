"""Feat squeezing helper (task book §18)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from _branch_fixtures import make_feats  # noqa: E402
from vhh_rl.branch_distill.decode import squeeze_singleton_feats  # noqa: E402


def test_squeeze_only_drops_leading_singletons():
    feats = make_feats()
    batched = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
               for k, v in feats.items()}
    out = squeeze_singleton_feats(batched)
    for key, value in feats.items():
        assert out[key].shape == value.shape, key


def test_squeeze_keeps_real_batch_dims():
    feats = make_feats()
    stacked = {k: (v.unsqueeze(0).repeat(2, *([1] * v.dim())) if torch.is_tensor(v) else v)
               for k, v in feats.items()}
    out = squeeze_singleton_feats(stacked)
    assert out["atom_to_token"].shape[0] == 2
