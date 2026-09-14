"""Phase D: static CF-OPSD trainer over a fixed target set (task book §46-§53)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from ..native_atom14.checkpoint import (
    load_base_model, make_policy_reference, parameter_drift,
    save_native_checkpoint, trainable_score_params,
)
from ..native_atom14.dpo_trainer import move_conditioning
from .same_query_fit import fit_steps

DEVICE = "cuda"


def run_static(name: str, targets_path: Path, conditioning_dir: Path, base_checkpoint: Path,
               output_dir: Path, *, updates: int, lr: float = 1e-5,
               variant: str = "target_mask_only", hold_lambda: float = 0.05,
               seed: int = 20260914, log_tag: str = "cf_opsd_static") -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")  # sampling context used TF32
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = torch.load(targets_path, map_location="cpu", weights_only=False)
    if not targets:
        raise RuntimeError("empty static target set")
    cond_path = Path(targets_path).parent / "cond_kwargs.pt"
    cond_kwargs = (torch.load(cond_path, map_location="cpu", weights_only=False)
                   if cond_path.is_file() else {})

    base = load_base_model(base_checkpoint, device=DEVICE)
    student, reference = make_policy_reference(base, device=DEVICE)
    params = trainable_score_params(student)
    n_train = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    cond_cache: dict[str, dict] = {}
    history = []
    t0 = time.time()
    for step in range(1, updates + 1):
        rec = random.choice(targets)
        cid = rec["case_id"]
        if cid not in cond_cache:
            if cid in cond_kwargs:
                cond_cache[cid] = move_conditioning(cond_kwargs[cid])
            else:  # legacy fallback: round-1 conditioning cache
                from ..native_atom14.dpo_trainer import load_conditioning
                cond_cache[cid] = move_conditioning(load_conditioning(conditioning_dir, cid))
        cond = cond_cache[cid]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": cond["feats"], "multiplicity": mult,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        full_query = rec.get("full_query_coords", rec["query_coords"]).to(DEVICE).float()
        if full_query.dim() == 2:
            full_query = full_query.unsqueeze(0)
        query = full_query.unsqueeze(0)                      # [1, B, N, 3]
        mult = int(rec.get("multiplicity", query.shape[1]))
        design_index = int(rec.get("design_index", 0))
        sigma = torch.full((query.shape[1],), float(rec["sigma"]), device=DEVICE)
        target = rec["target_coords"].to(DEVICE).float()
        full_anchor = rec.get("full_anchor_coords", rec["anchor_coords"]).to(DEVICE).float()
        if full_anchor.dim() == 2:
            full_anchor = full_anchor.unsqueeze(0)
        anchor = full_anchor[design_index]                   # [N, 3]
        feats = cond["feats"]
        token_of_atom = feats["atom_to_token"]
        if token_of_atom.dim() == 3:
            token_of_atom = token_of_atom.squeeze(0)
        token_of_atom = token_of_atom.int().argmax(-1)
        pad = feats["atom_pad_mask"].reshape(-1).bool()
        fake = feats["fake_atom_mask"].reshape(-1).bool()
        credited = set(rec["credits"].keys())
        target_mask = torch.zeros(pad.shape[0], dtype=torch.bool, device=DEVICE)
        for p in credited:
            target_mask |= (token_of_atom == int(p))
        target_mask &= fake & pad
        resolved = feats["atom_resolved_mask"].reshape(-1).bool()
        hold_mask = pad & resolved & ~target_mask

        optimizer.zero_grad(set_to_none=True)
        denoised, _ = student.structure_module.preconditioned_network_forward(
            query, sigma, training=False, network_condition_kwargs=kwargs)
        pred = denoised.float()[0, design_index]             # [N, 3]
        diff = (pred - target) ** 2
        loss = diff[target_mask, :].sum(dim=-1).mean()
        hold_value = torch.zeros((), device=DEVICE)
        if variant == "target_mask_weak_hold" and hold_mask.any():
            hold_value = ((pred - anchor) ** 2)[hold_mask, :].sum(dim=-1).mean()
            loss = loss + hold_lambda * hold_value
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        rec_row = {"step": step, "case_id": cid, "loss": float(loss.detach()),
                   "hold": float(hold_value.detach()), "grad_norm": float(grad_norm),
                   "seconds": time.time() - t0}
        history.append(rec_row)
        with (output_dir / "train_metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(rec_row) + "\n")
        if step % 10 == 0 or step == 1:
            print(f"[{log_tag}] step {step}/{updates} loss={rec_row['loss']:.6f} "
                  f"grad={rec_row['grad_norm']:.3f}", flush=True)
    ckpt = output_dir / f"checkpoint_{updates:04d}.pt"
    drift = parameter_drift(student, reference)
    save_native_checkpoint(base_checkpoint, student, ckpt, {
        "method": log_tag, "updates": updates, "variant": variant,
        "targets": str(targets_path), "seed": seed,
        "param_drift_total": drift["total"],
    })
    summary = {"updates": updates, "n_targets": len(targets),
               "n_trainable_params": n_train,
               "final_loss": history[-1]["loss"] if history else None,
               "param_drift_total": drift["total"],
               "checkpoint": str(ckpt), "elapsed_seconds": time.time() - t0}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
