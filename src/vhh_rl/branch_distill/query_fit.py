"""Exact multi-branch query replay + finite local distillation fit (§31-§39).

The trained forward replays the FULL K-branch batch exactly as captured
(§31/§72: multiplicity K must not be squashed to 1); only the peer branch
slice enters the loss.
"""
from __future__ import annotations

import torch

DEVICE = "cuda"


def network_kwargs(conditioning: dict, multiplicity: int, device=DEVICE) -> dict:
    feats = conditioning["feats"]
    return {
        "s_inputs": conditioning["s_inputs"].to(device),
        "s_trunk": conditioning["s_trunk"].to(device),
        "feats": {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in feats.items()} if isinstance(feats, dict)
        else feats.to(device),
        "multiplicity": multiplicity,
        "diffusion_conditioning": conditioning["diffusion_conditioning"].to(device)
        if torch.is_tensor(conditioning["diffusion_conditioning"])
        else conditioning["diffusion_conditioning"],
    }


def forward_peer_prediction(model, *, full_query_batch: torch.Tensor,
                            sigma: float, conditioning: dict,
                            multiplicity: int, peer_index: int,
                            device=DEVICE) -> torch.Tensor:
    """Exact full-batch replay; returns the peer branch clean prediction [N,3]."""
    query = full_query_batch.float()
    if query.dim() == 3:
        query = query.unsqueeze(0)             # [1, K, N, 3]
    if query.shape[1] != multiplicity:
        raise ValueError(f"query batch {query.shape[1]} != multiplicity {multiplicity}")
    query = query.to(device)
    kwargs = network_kwargs(conditioning, multiplicity, device=device)
    sigma_t = torch.full((multiplicity,), float(sigma), device=device)
    denoised, _ = model.structure_module.preconditioned_network_forward(
        query, sigma_t, training=False, network_condition_kwargs=kwargs)
    return denoised.float()[0, int(peer_index)]


def local_distill_loss(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """§36: MSE over touched * fake * pad, no hold term in v1."""
    diff = (pred - target.to(pred.device).float()) ** 2
    mask = mask.to(pred.device).reshape(-1).bool()
    if not bool(mask.any()):
        raise ValueError("empty target mask")
    return diff[mask].sum(dim=-1).mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return float(local_distill_loss(pred, target, mask).detach())


def record_target_mask(record) -> torch.Tensor:
    """§36 target mask: touched & fake_atom & atom_pad."""
    feats = record.conditioning["feats"]
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    return record.touched_mask.reshape(-1).bool() & fake & pad


def fit_record(base_checkpoint, record, *, updates: int = 40, lr: float = 1e-5,
               max_grad_norm: float = 1.0, device=DEVICE) -> dict:
    """One-record overfit from a fresh base student (§35); returns the student."""
    from ..native_atom14.checkpoint import (
        load_base_model, make_policy_reference, parameter_drift,
        trainable_score_params,
    )

    base = load_base_model(base_checkpoint, device=device)
    student, reference = make_policy_reference(base, device=device)
    params = trainable_score_params(student)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    mask = record_target_mask(record)
    pred = forward_peer_prediction(student, full_query_batch=record.full_query_batch,
                                   sigma=record.meta["sigma"], conditioning=record.conditioning,
                                   multiplicity=record.branch_count,
                                   peer_index=record.peer_index, device=device)
    initial = masked_mse(pred, record.target_coords, mask)
    history = []
    for step in range(1, int(updates) + 1):
        pred = forward_peer_prediction(student, full_query_batch=record.full_query_batch,
                                       sigma=record.meta["sigma"],
                                       conditioning=record.conditioning,
                                       multiplicity=record.branch_count,
                                       peer_index=record.peer_index, device=device)
        loss = local_distill_loss(pred, record.target_coords, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach()),
                        "grad_norm": float(grad_norm)})
    pred = forward_peer_prediction(student, full_query_batch=record.full_query_batch,
                                   sigma=record.meta["sigma"],
                                   conditioning=record.conditioning,
                                   multiplicity=record.branch_count,
                                   peer_index=record.peer_index, device=device)
    final = masked_mse(pred, record.target_coords, mask)
    drift = parameter_drift(student, reference)
    return {
        "student": student,
        "reference": reference,
        "initial_masked_mse": initial,
        "final_masked_mse": final,
        "mse_ratio": (final / initial) if initial > 0 else None,
        "param_drift": drift["total"],
        "history": history,
    }
