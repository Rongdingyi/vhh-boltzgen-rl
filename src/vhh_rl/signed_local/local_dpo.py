"""Signed local Diffusion-DPO loss (task book §27/§28).

The preferred/rejected orientation comes from the counterfactual dR sign.
Reward magnitude is never used.  Thin wrapper over the shared machinery in
``native_atom14.global_cf_step`` so the local branch cannot drift from the
validated DPO math.
"""
from __future__ import annotations

import torch

from ..native_atom14.global_cf_step import signed_local_dpo_step


def signed_local_dpo_loss(
    policy_sm,
    ref_sm,
    feats: dict,
    preferred_coords: torch.Tensor,   # [N_atom, 3]
    rejected_coords: torch.Tensor,    # [N_atom, 3]
    position_mask: torch.Tensor,      # [N_atom] bool, target residue fake atoms
    cond_kwargs: dict,
    beta: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """Returns (loss, metrics); loss == log 2 when policy == reference."""
    out = signed_local_dpo_step(
        policy_sm, ref_sm, feats, preferred_coords, rejected_coords,
        position_mask, cond_kwargs, beta=beta)
    metrics = {
        "z": float(out.dpo.z.mean().detach()),
        "implicit_acc": float(out.dpo.implicit_acc),
        "model_diff": float(out.dpo.model_diff.mean()),
        "ref_diff": float(out.dpo.ref_diff.mean()),
        "sigma": float(out.sigma.reshape(-1).mean()),
        "preferred_loss": float(out.winner_loss.mean()),
        "rejected_loss": float(out.loser_loss.mean()),
    }
    return out.loss, metrics
