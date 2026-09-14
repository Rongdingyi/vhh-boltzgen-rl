"""Diffusion-DPO loss (task book §36-37).

    z = -beta/2 * ((l_theta_w - l_theta_l) - (l_ref_w - l_ref_l))
    L = -log sigmoid(z)

At init (policy == reference) z == 0 and L == log 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    total_loss: torch.Tensor
    dpo_loss: torch.Tensor
    z: torch.Tensor
    implicit_acc: torch.Tensor
    model_diff: torch.Tensor  # l_theta_w - l_theta_l
    ref_diff: torch.Tensor  # l_ref_w - l_ref_l


def diffusion_dpo_loss(
    winner_policy_loss: torch.Tensor,  # [B]
    loser_policy_loss: torch.Tensor,  # [B]
    winner_ref_loss: torch.Tensor,  # [B]
    loser_ref_loss: torch.Tensor,  # [B]
    beta: float,
) -> DPOLossOutput:
    model_diff = winner_policy_loss - loser_policy_loss
    ref_diff = winner_ref_loss - loser_ref_loss
    z = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(z)
    return DPOLossOutput(
        total_loss=loss.mean(),
        dpo_loss=loss.mean(),
        z=z.detach(),
        implicit_acc=(z > 0).float().mean().detach(),
        model_diff=model_diff.detach(),
        ref_diff=ref_diff.detach(),
    )
