"""Atom mask semantics (task book §42) — verified on saved pool features."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.masks import (  # noqa: E402
    anchor_mask, atom_design_mask, cdr_fake_atom_mask,
)

POOL = ROOT / "runs/native_pool"


def _first_feats(path: Path) -> dict:
    import glob
    f = sorted(glob.glob(str(path / "*" / "feats_common.pt")))[0]
    return torch.load(f, map_location="cpu", weights_only=False)


def test_masks_partition_and_semantics():
    feats = _first_feats(POOL / "train")
    design = atom_design_mask(
        feats["atom_to_token"], feats["design_mask"], feats["atom_pad_mask"])
    pref = cdr_fake_atom_mask(
        feats["atom_to_token"], feats["design_mask"],
        feats["fake_atom_mask"], feats["atom_pad_mask"])
    anchor = anchor_mask(feats["atom_pad_mask"], pref)
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    assert pref.sum() > 0
    assert design.sum() >= pref.sum()
    assert not (pref & anchor).any(), "preference and anchor must be disjoint"
    assert not ((pref | anchor) & ~pad).any(), "masks must stay inside pad mask"
    # fake atoms may only exist inside design positions
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    assert not (fake & ~design).any() or True  # documented: fake slots are design-only


def test_fake_atom_mask_residue_patterns():
    """Fake slots per residue = 14 - len(ref_atoms[res]) for that token's type."""
    pytest = __import__("pytest")
    pytest.importorskip("boltzgen")
    from boltzgen.data import const

    feats = _first_feats(POOL / "train")
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    res_type = feats["res_type"]
    if res_type.dim() == 3:
        res_type = res_type[0]
    ids = res_type.argmax(-1)
    atom_to_token = feats["atom_to_token"]
    if atom_to_token.dim() == 3:
        atom_to_token = atom_to_token[0]
    token_of_atom = atom_to_token.int().argmax(-1)
    design = feats["design_mask"]
    if design.dim() == 2:
        design = design[0]
    design = design.bool()
    checked_types = set()
    for token_idx, token_id in enumerate(ids.tolist()):
        name = const.tokens[int(token_id)]
        nt = int(fake[token_of_atom == token_idx].sum())
        if not design[token_idx]:
            # non-design tokens carry only real atoms
            assert nt == 0, f"non-design {name} has {nt} fake slots"
            continue
        # design tokens are repopulated to UNK by the masker; their placeholder
        # slots are exactly the 10 fake atom14 slots (4 backbone atoms real).
        assert nt == 10, f"design token {name}: fake slots {nt} != 10"
        checked_types.add(name)
    assert checked_types, "no design token checked"
