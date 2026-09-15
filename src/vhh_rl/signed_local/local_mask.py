"""Target-residue fake-atom mask (task book §25/§26)."""
from __future__ import annotations

import torch

from ..native_atom14.masks import design_token_offset, residue_atom_masks


def target_residue_mask(feats: dict, position: int,
                        design_positions: tuple[int, ...]) -> torch.Tensor:
    """[N_atom] bool: fake atoms of exactly the target design residue.

    Invariants enforced by ``assert_mask_invariants``: non-empty, only the
    target token, only fake atoms, only padded/real atoms.
    """
    if int(position) not in {int(p) for p in design_positions}:
        raise ValueError(f"target position {position} is not a design position")
    offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                 design_positions)
    masks = residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                               feats["atom_pad_mask"], [position + offset])
    mask = masks[0].bool()
    assert_mask_invariants(feats, mask, position + offset)
    return mask


def assert_mask_invariants(feats: dict, mask: torch.Tensor, target_token: int) -> None:
    if not bool(mask.any()):
        raise ValueError("target residue mask is empty")
    a2t = feats["atom_to_token"]
    if a2t.dim() == 3:
        a2t = a2t.squeeze(0)
    token_of_atom = a2t.int().argmax(-1)
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    sel_tokens = set(token_of_atom[mask].tolist())
    if sel_tokens != {int(target_token)}:
        raise ValueError(f"mask selects tokens {sel_tokens}, expected {{{target_token}}}")
    if not bool((mask & ~fake).sum() == 0):
        raise ValueError("mask selects non-fake atoms")
    if not bool((mask & ~pad).sum() == 0):
        raise ValueError("mask selects padded atoms")
    # no atoms of any other token
    if bool((mask & (token_of_atom != int(target_token))).any()):
        raise ValueError("mask leaks atoms from other tokens")
