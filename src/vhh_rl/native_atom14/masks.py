"""Atom-level masks for native atom14 preference/DPO (task book §35/41/44).

Semantics audited against featurizer.py:1108-1109 and writer.py:273:
  - fake_atom_mask == 1  -> fake (placeholder) atom slot of a design residue
  - design_mask          -> token-level, True for CDR design positions
  - atom_to_token        -> [N_atom, N_tok] one-hot mapping
The semantics are enforced by tests/test_atom_masks.py (GLY/ALA/TRP patterns),
not assumed from the variable name (task book §42).
"""
from __future__ import annotations

import torch


def _squeeze_single_batch(tensor: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    if tensor.dim() == expected_ndim + 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def atom_to_token_index(atom_to_token: torch.Tensor) -> torch.Tensor:
    """[N_atom, N_tok] (or [1, N_atom, N_tok]) one-hot -> [N_atom] token index."""
    if atom_to_token.dim() == 3 and atom_to_token.shape[0] == 1:
        atom_to_token = atom_to_token.squeeze(0)
    return atom_to_token.int().argmax(dim=-1)


def atom_design_mask(
    atom_to_token: torch.Tensor,
    design_mask: torch.Tensor,
    atom_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Boolean [N_atom]: atom belongs to a design (CDR) token and is real."""
    design_mask = _squeeze_single_batch(design_mask, 1)
    atom_pad_mask = _squeeze_single_batch(atom_pad_mask, 1)
    token_of_atom = atom_to_token_index(atom_to_token)
    mask = design_mask.bool()[token_of_atom]
    return mask & atom_pad_mask.bool()


def cdr_fake_atom_mask(
    atom_to_token: torch.Tensor,
    design_mask: torch.Tensor,
    fake_atom_mask: torch.Tensor,
    atom_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """M_fake_cdr (task book §41): CDR design positions' fake side-chain atoms."""
    fake_atom_mask = _squeeze_single_batch(fake_atom_mask, 1)
    atom_pad_mask = _squeeze_single_batch(atom_pad_mask, 1)
    return (
        atom_design_mask(atom_to_token, design_mask, atom_pad_mask)
        & fake_atom_mask.bool()
        & atom_pad_mask.bool()
    )


def anchor_mask(atom_pad_mask: torch.Tensor, preference_mask: torch.Tensor) -> torch.Tensor:
    """M_anchor (task book §44): everything real except the preference atoms."""
    atom_pad_mask = _squeeze_single_batch(atom_pad_mask, 1)
    preference_mask = _squeeze_single_batch(preference_mask, 1)
    pad = atom_pad_mask.bool()
    return pad & ~preference_mask.bool()


def design_token_offset(token_index: torch.Tensor, design_mask: torch.Tensor,
                        design_positions) -> int:
    """Offset between featurizer token ids and manifest sequence positions.

    The manifest positions index the chain sequence; the featurizer token table
    is global.  For all design positions the offset must be identical, which we
    assert instead of assuming chain ordering.
    """
    token_index = _squeeze_single_batch(token_index, 1).long()
    design_mask = _squeeze_single_batch(design_mask, 1).bool()
    idx = token_index[design_mask].tolist()
    positions = sorted(int(p) for p in design_positions)
    if len(idx) != len(positions):
        raise ValueError(f"design token count {len(idx)} != manifest {len(positions)}")
    offsets = {i - p for i, p in zip(idx, positions)}
    if len(offsets) != 1:
        raise ValueError(f"non-constant token offset: {sorted(offsets)[:6]}")
    return offsets.pop()


def residue_atom_masks(atom_to_token: torch.Tensor, fake_atom_mask: torch.Tensor,
                       atom_pad_mask: torch.Tensor,
                       token_ids: list[int]) -> torch.Tensor:
    """[R, N_atom] bool: fake*real atoms of each requested token id."""
    token_of_atom = atom_to_token_index(atom_to_token)
    fake = _squeeze_single_batch(fake_atom_mask, 1).bool()
    pad = _squeeze_single_batch(atom_pad_mask, 1).bool()
    masks = []
    for tid in token_ids:
        masks.append((token_of_atom == int(tid)) & fake & pad)
    return torch.stack(masks, dim=0) if masks else torch.zeros((0, pad.numel()), dtype=torch.bool)


def mask_stats(mask: torch.Tensor) -> dict[str, int]:
    mask = mask.bool()
    return {"total": int(mask.numel()), "selected": int(mask.sum().item())}
