"""Target-residue fake-atom mask semantics (task book §25/§26/§75)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.signed_local.local_mask import (  # noqa: E402
    assert_mask_invariants, target_residue_mask,
)


def _feats() -> dict:
    """8 atoms, 5 tokens, token 2 is the design target with fake atoms {2,3}."""
    n_tok, n_atom = 5, 8
    a2t = torch.zeros(n_atom, n_tok, dtype=torch.long)
    owners = [0, 1, 2, 2, 3, 3, 4, 4]
    for atom, tok in enumerate(owners):
        a2t[atom, tok] = 1
    fake = torch.zeros(n_atom, dtype=torch.bool)
    fake[[2, 3, 4, 5]] = True
    pad = torch.ones(n_atom, dtype=torch.bool)
    pad[7] = False
    return {
        "token_index": torch.arange(n_tok),
        "design_mask": torch.tensor([False, False, True, False, False]),
        "atom_to_token": a2t,
        "fake_atom_mask": fake,
        "atom_pad_mask": pad,
    }


def test_mask_selects_exactly_target_fake_atoms():
    feats = _feats()
    mask = target_residue_mask(feats, position=2, design_positions=(2, 3))
    assert mask.tolist() == [False, False, True, True, False, False, False, False]
    assert_mask_invariants(feats, mask, target_token=2)


def test_non_design_position_rejected():
    feats = _feats()
    with pytest.raises(ValueError):
        target_residue_mask(feats, position=3, design_positions=(2,))
    with pytest.raises(ValueError):
        target_residue_mask(feats, position=1, design_positions=(2,))


def test_invariants_reject_leaky_mask():
    feats = _feats()
    leaky = torch.zeros(8, dtype=torch.bool)
    leaky[[2, 4]] = True
    with pytest.raises(ValueError):
        assert_mask_invariants(feats, leaky, target_token=3)
    real_atom = torch.zeros(8, dtype=torch.bool)
    real_atom[0] = True
    with pytest.raises(ValueError):
        assert_mask_invariants(feats, real_atom, target_token=0)
    empty = torch.zeros(8, dtype=torch.bool)
    with pytest.raises(ValueError):
        assert_mask_invariants(feats, empty, target_token=2)
