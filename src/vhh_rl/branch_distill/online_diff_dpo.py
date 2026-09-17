"""Arm B: online refreshed changed-position Diffusion-DPO (§44/§45).

Uniform weights over the changed positions; no credit file is read, and the
winner/loser pair must come from the same BranchGroup.
"""
from __future__ import annotations

import torch

from ..native_atom14.global_cf_step import compute_weighted_cf_dpo_step
from ..native_atom14.masks import design_token_offset, residue_atom_masks


def uniform_changed_weights(changed_positions) -> dict[int, float]:
    positions = sorted(int(p) for p in changed_positions)
    if not positions:
        raise ValueError("no changed positions")
    w = 1.0 / len(positions)
    return {p: w for p in positions}


def online_dpo_step(policy_sm, ref_sm, feats: dict, teacher_coords: torch.Tensor,
                    peer_coords: torch.Tensor, changed_positions,
                    network_condition_kwargs: dict, *, design_positions,
                    beta: float = 10.0):
    """One weighted-DPO step between two endpoints of the same group."""
    weights = uniform_changed_weights(changed_positions)
    positions = sorted(weights)
    offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                 design_positions)
    masks = residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                               feats["atom_pad_mask"],
                               [p + offset for p in positions])
    w = torch.tensor([weights[p] for p in positions],
                     device=teacher_coords.device, dtype=torch.float32)
    return compute_weighted_cf_dpo_step(
        policy_sm, ref_sm, feats, teacher_coords, peer_coords, masks, w,
        network_condition_kwargs, beta=beta)
