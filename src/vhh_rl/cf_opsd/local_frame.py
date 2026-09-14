"""Residue-local backbone frames and winner->anchor geometry transfer (§20-§24).

Frame convention: for residue i with backbone atoms N, CA, C:
    e1 = normalize(CA - N)
    e2 = normalize((C - CA) - ((C - CA).e1) e1)
    e3 = e1 x e2
    F = [e1 e2 e3]  (columns)
    local(X)  = F^T (X - N)
    world(U)  = N + F U
Only side-chain / fake geometry is transferred; the winner backbone is never
moved onto the anchor.
"""
from __future__ import annotations

import torch

from .types import OPSDTrajectory


def backbone_indices(feats: dict, token_ids: list[int]) -> list[tuple[int, int, int]]:
    """Return (N, CA, C) atom indices for each token id (residue)."""
    a2t = feats["atom_to_token"]
    if a2t.dim() == 3 and a2t.shape[0] == 1:
        a2t = a2t.squeeze(0)
    token_of_atom = a2t.int().argmax(-1)
    backbone = feats["backbone_mask"].reshape(-1).bool()
    out = []
    for tid in token_ids:
        idx = torch.where((token_of_atom == int(tid)) & backbone)[0]
        if idx.numel() < 3:
            raise ValueError(f"token {tid} has <3 backbone atoms")
        out.append((int(idx[0]), int(idx[1]), int(idx[2])))
    return out


def frame_from_backbone(coords: torch.Tensor, n_idx: int, ca_idx: int, c_idx: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    n = coords[n_idx].float()
    ca = coords[ca_idx].float()
    c = coords[c_idx].float()
    e1 = ca - n
    e1 = e1 / e1.norm().clamp_min(1e-8)
    v = c - ca
    e2 = v - (v @ e1) * e1
    e2 = e2 / e2.norm().clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=0)
    F = torch.stack([e1, e2, e3], dim=1)  # columns
    return F, n


def to_local(coords: torch.Tensor, F: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    return (coords.float() - origin) @ F


def to_world(local: torch.Tensor, F: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    return local @ F.T + origin


def transfer_winner_to_anchor(
    anchor_coords: torch.Tensor,
    winner_coords: torch.Tensor,
    feats: dict,
    positions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move winner fake-atom geometry into the anchor's residue-local frames.

    Returns (transferred_coords [N,3], touched_mask [N] bool).  Only fake atoms
    of the requested design positions that are valid under atom_pad &
    fake_atom_mask are touched.
    """
    a2t = feats["atom_to_token"]
    if a2t.dim() == 3 and a2t.shape[0] == 1:
        a2t = a2t.squeeze(0)
    token_of_atom = a2t.int().argmax(-1)
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    pad = feats["atom_pad_mask"].reshape(-1).bool()

    transferred = anchor_coords.clone().float()
    touched = torch.zeros(anchor_coords.shape[0], dtype=torch.bool)
    backbone = backbone_indices(feats, positions)
    for tid, (n_i, ca_i, c_i) in zip(positions, backbone):
        F_a, o_a = frame_from_backbone(anchor_coords, n_i, ca_i, c_i)
        F_w, o_w = frame_from_backbone(winner_coords, n_i, ca_i, c_i)
        atoms = torch.where((token_of_atom == int(tid)) & fake & pad)[0]
        if atoms.numel() == 0:
            continue
        local = to_local(winner_coords[atoms], F_w, o_w)
        transferred[atoms] = to_world(local, F_a, o_a)
        touched[atoms] = True
    return transferred, touched
