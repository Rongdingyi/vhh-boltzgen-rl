"""Minimal shared global-CF-DPO step (task book SL-CF-DPO §36).

This helper is the *only* implementation of the frozen global branch:

    winner/loser -> paired noise (same sigma/noise/rigid aug)
                 -> per-residue fake-atom denoising losses
                 -> credit weights w_i (eta=0.75 uniform floor)
                 -> standard diffusion DPO (beta=10)

Both the validated `native_atom14.weighted_dpo` trainer and the new
`signed_local.trainer` call it, so the global branch cannot drift.

The local correction branch must NOT use this module's global-energy readings
for any reward regression; it only reuses the same mechanism with a single
target-residue mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .denoise_loss import per_residue_denoising_loss
from .dpo_loss import DPOLossOutput, diffusion_dpo_loss
from .paired_noise import paired_noising


@dataclass
class GlobalCFStepOutput:
    loss: torch.Tensor
    dpo: DPOLossOutput
    sigma: torch.Tensor
    winner_loss: torch.Tensor  # weighted policy loss [1]
    loser_loss: torch.Tensor
    ref_winner_loss: torch.Tensor
    ref_loser_loss: torch.Tensor


def compute_weighted_cf_dpo_step(
    policy_sm,
    ref_sm,
    feats: dict,
    winner_coords: torch.Tensor,
    loser_coords: torch.Tensor,
    residue_masks: torch.Tensor,  # [R, N_atom] bool
    weights: torch.Tensor,  # [R] non-negative, normalized inside
    network_condition_kwargs: dict,
    *,
    beta: float = 10.0,
    sigma: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> GlobalCFStepOutput:
    atom_mask = feats["atom_pad_mask"]
    if atom_mask.dim() > 1:
        atom_mask = atom_mask.reshape(-1)
    if sigma is None:
        sigma = policy_sm.noise_distribution(1)
    if noise is None:
        noise = torch.randn_like(winner_coords.unsqueeze(0))
    paired = paired_noising(
        winner_coords, loser_coords, atom_mask, sigma,
        augmentation=policy_sm.coordinate_augmentation, noise=noise,
    )
    lw_p, _ = per_residue_denoising_loss(
        policy_sm, feats, paired["x0_w_aug"], paired["x_t_w"], sigma,
        network_condition_kwargs, residue_masks, require_grad=True)
    ll_p, _ = per_residue_denoising_loss(
        policy_sm, feats, paired["x0_l_aug"], paired["x_t_l"], sigma,
        network_condition_kwargs, residue_masks, require_grad=True)
    lw_r, _ = per_residue_denoising_loss(
        ref_sm, feats, paired["x0_w_aug"], paired["x_t_w"], sigma,
        network_condition_kwargs, residue_masks, require_grad=False)
    ll_r, _ = per_residue_denoising_loss(
        ref_sm, feats, paired["x0_l_aug"], paired["x_t_l"], sigma,
        network_condition_kwargs, residue_masks, require_grad=False)

    w = weights.to(lw_p.device).float()
    w = w / w.sum().clamp_min(1e-12)
    loss_w = policy_sm.loss_weight(sigma.reshape(-1))
    winner_loss = (lw_p.reshape(-1) * w).sum() * loss_w
    loser_loss = (ll_p.reshape(-1) * w).sum() * loss_w
    ref_winner_loss = (lw_r.reshape(-1) * w).sum() * loss_w
    ref_loser_loss = (ll_r.reshape(-1) * w).sum() * loss_w
    dpo = diffusion_dpo_loss(winner_loss, loser_loss, ref_winner_loss, ref_loser_loss, beta)
    return GlobalCFStepOutput(
        loss=dpo.total_loss, dpo=dpo, sigma=sigma,
        winner_loss=winner_loss.detach(), loser_loss=loser_loss.detach(),
        ref_winner_loss=ref_winner_loss.detach(), ref_loser_loss=ref_loser_loss.detach(),
    )


def signed_local_dpo_step(
    policy_sm,
    ref_sm,
    feats: dict,
    preferred_coords: torch.Tensor,
    rejected_coords: torch.Tensor,
    position_mask: torch.Tensor,  # [N_atom] bool, target residue fake atoms
    network_condition_kwargs: dict,
    *,
    beta: float = 10.0,
    sigma: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> GlobalCFStepOutput:
    """Signed local correction step: identical machinery, single-residue mask.

    The preferred/rejected orientation already encodes the counterfactual
    direction; reward magnitude is never used in the loss.
    """
    atom_mask = feats["atom_pad_mask"]
    if atom_mask.dim() > 1:
        atom_mask = atom_mask.reshape(-1)
    if sigma is None:
        sigma = policy_sm.noise_distribution(1)
    if noise is None:
        noise = torch.randn_like(preferred_coords.unsqueeze(0))
    paired = paired_noising(
        preferred_coords, rejected_coords, atom_mask, sigma,
        augmentation=policy_sm.coordinate_augmentation, noise=noise,
    )
    mask = position_mask.float()
    model_kwargs = dict(network_condition_kwargs)
    p_p = _masked_policy_loss(policy_sm, feats, paired["x0_w_aug"], paired["x_t_w"],
                              sigma, model_kwargs, mask, require_grad=True)
    r_p = _masked_policy_loss(policy_sm, feats, paired["x0_l_aug"], paired["x_t_l"],
                              sigma, model_kwargs, mask, require_grad=True)
    p_r = _masked_policy_loss(ref_sm, feats, paired["x0_w_aug"], paired["x_t_w"],
                              sigma, model_kwargs, mask, require_grad=False)
    r_r = _masked_policy_loss(ref_sm, feats, paired["x0_l_aug"], paired["x_t_l"],
                              sigma, model_kwargs, mask, require_grad=False)
    dpo = diffusion_dpo_loss(p_p, r_p, p_r, r_r, beta)
    return GlobalCFStepOutput(
        loss=dpo.total_loss, dpo=dpo, sigma=sigma,
        winner_loss=p_p.detach(), loser_loss=r_p.detach(),
        ref_winner_loss=p_r.detach(), ref_loser_loss=r_r.detach(),
    )


def _masked_policy_loss(structure_module, feats, x0, noised, sigma, kwargs,
                        mask: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
    """Per-sample denoising loss restricted to one target residue's fake atoms."""
    from .denoise_loss import per_sample_denoising_loss

    out = per_sample_denoising_loss(
        structure_module, feats, x0, noised, sigma, kwargs,
        preference_mask=mask, require_grad=require_grad)
    return out.per_sample_loss
