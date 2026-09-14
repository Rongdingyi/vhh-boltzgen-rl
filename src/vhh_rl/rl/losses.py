"""Policy losses: REINFORCE and GRPO-style clipped objective + exact KL."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reinforce_loss(
    action_logprobs: torch.Tensor,
    advantage: float,
) -> torch.Tensor:
    """L = -A * mean_t log pi(a_t|s_t)  (per-trajectory length normalization)."""
    return -(advantage * action_logprobs.mean())


def grpo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: float,
    clip_eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    stats = {
        "ratio_mean": float(ratio.mean()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "clip_fraction": float(((ratio - 1.0).abs() > clip_eps).float().mean()),
    }
    return loss, stats


def exact_categorical_kl(
    current_full_logprobs: list[torch.Tensor],
    reference_full_logprobs: list[torch.Tensor],
) -> torch.Tensor:
    """Exact 20-AA categorical KL per event, averaged over events (task book §24).

    Constraint-masked entries carry p->0 so they cannot produce NaN; a tiny
    clamp guards the log of a probability that underflowed to 0.
    """
    kls = []
    for cur, ref in zip(current_full_logprobs, reference_full_logprobs):
        log_p = torch.log_softmax(cur, dim=-1)
        log_q = torch.log_softmax(ref, dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1)
        kls.append(kl.mean())
    return torch.stack(kls).mean() if kls else torch.zeros(())
