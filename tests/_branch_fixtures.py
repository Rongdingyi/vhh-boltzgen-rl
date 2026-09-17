"""Synthetic feats/coords fixtures shared by branch_distill tests."""
from __future__ import annotations

import torch


def make_feats(design_positions=(2,), n_tok=4):
    """17 atoms / 4 tokens; token 2 is the design residue with fake atoms 11,12."""
    owners = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    n_atom = len(owners)
    a2t = torch.zeros(n_atom, n_tok, dtype=torch.long)
    for atom, tok in enumerate(owners):
        a2t[atom, tok] = 1
    fake = torch.zeros(n_atom, dtype=torch.bool)
    fake[[3, 7, 11, 12, 16]] = True
    backbone = torch.zeros(n_atom, dtype=torch.bool)
    backbone[[0, 1, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15]] = True
    design_mask = torch.zeros(n_tok, dtype=torch.bool)
    for p in design_positions:
        design_mask[p] = True
    return {
        "token_index": torch.arange(n_tok),
        "design_mask": design_mask,
        "atom_to_token": a2t,
        "fake_atom_mask": fake,
        "atom_pad_mask": torch.ones(n_atom, dtype=torch.bool),
        "backbone_mask": backbone,
        "atom_resolved_mask": torch.ones(n_atom, dtype=torch.bool),
        "coords": torch.zeros(n_atom, 3),
    }


def make_coords(seed: int = 0, shift_fake: float = 0.0, n_atom: int = 17):
    g = torch.Generator().manual_seed(seed)
    coords = torch.zeros(n_atom, 3)
    for res in range(4):
        base = torch.tensor([3.0 * res, 0.0, 0.0])
        coords[4 * res + 0] = base + torch.tensor([0.0, 0.0, 0.0])
        coords[4 * res + 1] = base + torch.tensor([1.45, 0.0, 0.0])
        coords[4 * res + 2] = base + torch.tensor([1.45, 1.52, 0.0])
    coords[11] = torch.tensor([6.0, 1.0, 1.0]) + shift_fake
    coords[12] = torch.tensor([6.0, 2.0, 1.0]) + shift_fake
    coords[3] = torch.tensor([0.0, 1.0, 1.0])
    coords[7] = torch.tensor([3.0, 1.0, 1.0])
    coords[16] = torch.tensor([9.0, 1.0, 1.0])
    return coords + 0.01 * torch.randn(n_atom, 3, generator=g)
