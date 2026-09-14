"""Reference/backbone anchor loss (task book §43-47).

    L_anchor = 1/2 [ MSE_M(D_theta^w, D_ref^w) + MSE_M(D_theta^l, D_ref^l) ]

where D is the denoised coordinate output on the same noisy state, M = the
anchor mask (all real atoms except the preference atoms).  Reference output is
detached; current output keeps gradients.  Mask-normalized by
``3 * mask.sum`` so protein length does not change the loss scale.
"""
from __future__ import annotations

import torch


def anchor_loss(
    policy_denoised_w: torch.Tensor,  # [1, N_atom, 3]
    ref_denoised_w: torch.Tensor,  # [1, N_atom, 3] detached
    policy_denoised_l: torch.Tensor,
    ref_denoised_l: torch.Tensor,
    mask: torch.Tensor,  # [N_atom] or [1, N_atom] bool
) -> torch.Tensor:
    m = mask.float()
    if m.dim() == 1:
        m = m.unsqueeze(0)

    def one(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        sq = ((pred.float() - ref.float().detach()) ** 2).sum(dim=-1)  # [B, N_atom]
        per_sample = (sq * m).sum(dim=-1) / (3.0 * m.sum(dim=-1) + 1e-8)
        return per_sample.mean()

    return 0.5 * (one(policy_denoised_w, ref_denoised_w) + one(policy_denoised_l, ref_denoised_l))
