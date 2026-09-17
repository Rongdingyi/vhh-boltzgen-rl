"""Arms C/D: online sibling distillation (task book §46).

Identical data and trainer; the only difference is the supervision mask
(all design fake atoms vs changed-residue fake atoms).
"""
from __future__ import annotations

import torch

from .query_fit import forward_peer_prediction, local_distill_loss

MASK_MODES = ("all_design", "changed_only")


def positions_for_mask(mask_mode: str, record, design_positions):
    if mask_mode == "all_design":
        return sorted(int(p) for p in design_positions)
    if mask_mode == "changed_only":
        return sorted(int(p) for p in record.changed_positions)
    raise ValueError(f"unknown mask_mode {mask_mode}")


def supervision_mask(feats: dict, positions, design_positions) -> torch.Tensor:
    from ..native_atom14.masks import design_token_offset, residue_atom_masks

    offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                 design_positions)
    masks = residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                               feats["atom_pad_mask"],
                               [int(p) + offset for p in positions])
    return masks.any(dim=0)


def online_distill_step(student, record, mask_mode: str, design_positions,
                        *, device="cuda") -> tuple[torch.Tensor, dict]:
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}")
    positions = positions_for_mask(mask_mode, record, design_positions)
    feats = record.conditioning["feats"]
    mask = supervision_mask(feats, positions, design_positions)
    pred = forward_peer_prediction(
        student, full_query_batch=record.full_query_batch,
        sigma=record.meta["sigma"], conditioning=record.conditioning,
        multiplicity=record.branch_count, peer_index=record.peer_index,
        device=device)
    loss = local_distill_loss(pred, record.target_coords, mask)
    return loss, {"n_supervised_positions": len(positions),
                  "n_supervised_atoms": int(mask.sum()),
                  "loss": float(loss.detach())}
