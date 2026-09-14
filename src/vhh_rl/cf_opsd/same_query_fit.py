"""Phase C: same-query finite realization (task book §35-§45).

Student clean prediction P_theta(X_q, sigma_q) is trained towards the positive
target Y+ on the target mask (V0-A) and optionally with a weak hold term over
the remaining resolved atoms towards the behavior anchor (V0-B).
"""
from __future__ import annotations

import torch


def student_prediction(structure_module, query_state: torch.Tensor, sigma: torch.Tensor,
                       network_condition_kwargs: dict) -> torch.Tensor:
    with torch.no_grad():
        denoised, _ = structure_module.preconditioned_network_forward(
            query_state, sigma.reshape(-1), training=False,
            network_condition_kwargs=network_condition_kwargs,
        )
    return denoised.float()


def fit_steps(student_sm, query_state: torch.Tensor, sigma: torch.Tensor,
              target: torch.Tensor, anchor: torch.Tensor,
              target_mask: torch.Tensor, hold_mask: torch.Tensor,
              network_condition_kwargs: dict, params, *,
              steps: int, lr: float = 1e-5, max_grad_norm: float = 1.0,
              variant: str = "target_mask_only", hold_lambda: float = 0.05,
              ) -> tuple[list[torch.Tensor], list[dict]]:
    """Train and record the clean prediction after every step (incl. step 0)."""
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    preds = [student_prediction(student_sm, query_state, sigma, network_condition_kwargs)]
    logs = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        denoised, _ = student_sm.preconditioned_network_forward(
            query_state, sigma.reshape(-1), training=False,
            network_condition_kwargs=network_condition_kwargs,
        )
        denoised = denoised.float()
        diff = (denoised - target.float()) ** 2
        loss = diff[target_mask].sum(dim=-1).mean() if target_mask.any() else diff.mean()
        hold_value = torch.zeros((), device=denoised.device)
        if variant == "target_mask_weak_hold" and hold_mask.any():
            hold_value = ((denoised - anchor.float()) ** 2)[hold_mask].sum(dim=-1).mean()
            loss = loss + hold_lambda * hold_value
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        preds.append(student_prediction(student_sm, query_state, sigma, network_condition_kwargs))
        logs.append({"step": step + 1, "loss": float(loss.detach()),
                     "hold": float(hold_value.detach()),
                     "grad_norm": float(grad_norm)})
    return preds, logs


def distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    return float(((a[mask].float() - b[mask].float()) ** 2).sum(-1).mean().sqrt())
