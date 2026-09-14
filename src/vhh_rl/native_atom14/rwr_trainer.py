"""N1 Reward-Weighted (top-quartile) diffusion fine-tuning (task book §23-27).

Continues training the score model on the top-25% native samples (by CDR
classifier reward) using the official diffusion forward-noising + coordinate
MSE.  No reward regression; the high-reward samples are pseudo-targets.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    load_base_model,
    make_policy_reference,
    parameter_drift,
    save_native_checkpoint,
    trainable_score_params,
)
from .denoise_loss import per_sample_denoising_loss
from .dpo_trainer import load_conditioning, move_conditioning
from .sample_dataset import load_case_samples

DEVICE = "cuda"


def _load_coords(pool_root: Path, sample_id: str) -> torch.Tensor:
    case_id = sample_id.rsplit("_s", 1)[0]
    for split_dir in Path(pool_root).iterdir():
        candidate = split_dir / case_id / "coords" / f"{sample_id}.pt"
        if candidate.is_file():
            return torch.load(candidate, map_location=DEVICE, weights_only=True).float()
    raise FileNotFoundError(sample_id)


def run_rwr(
    base_checkpoint: str | Path,
    pool_root: str | Path,
    split: str,
    conditioning_dir: str | Path,
    output_dir: str | Path,
    *,
    positive_quantile: float = 0.25,
    lr: float = 1e-5,
    max_steps: int = 500,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 50,
    seed: int = 20260913,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    # positive samples = top quartile per case
    positives: dict[str, list[str]] = {}
    for case_dir in sorted((Path(pool_root) / split).iterdir()):
        if not case_dir.is_dir():
            continue
        samples = [s for s in load_case_samples(case_dir)
                   if s.reward_raw is not None and not s.contains_invalid and s.fr_mismatch == 0]
        if not samples:
            continue
        k = max(1, int(len(samples) * positive_quantile))
        ranked = sorted(samples, key=lambda s: s.reward_raw, reverse=True)[:k]
        positives[case_dir.name] = [s.sample_id for s in ranked]
    if not positives:
        raise RuntimeError("no positive samples")
    print(f"[rwr] {len(positives)} cases, "
          f"{sum(len(v) for v in positives.values())} positive samples", flush=True)

    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    train_params = trainable_score_params(policy)
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.0)

    cond_cache = {cid: move_conditioning(load_conditioning(conditioning_dir, cid))
                  for cid in positives}
    case_ids = sorted(positives)
    step = 0
    history = []
    t0 = time.time()
    while step < max_steps:
        case_id = case_ids[step % len(case_ids)]
        sample_id = random.choice(positives[case_id])
        cond = cond_cache[case_id]
        feats = cond["feats"]
        network_condition_kwargs = {
            "s_inputs": cond["s_inputs"],
            "s_trunk": cond["s_trunk"],
            "feats": feats,
            "multiplicity": 1,
            "diffusion_conditioning": cond["diffusion_conditioning"],
        }
        x0 = _load_coords(pool_root, sample_id).unsqueeze(0)
        atom_mask = feats["atom_pad_mask"].reshape(-1)
        from boltzgen.model.modules.utils import center_random_augmentation

        sigma = policy.structure_module.noise_distribution(1)
        noise = torch.randn_like(x0)
        x0_aug = center_random_augmentation(
            x0, atom_mask.unsqueeze(0).float(),
            augmentation=policy.structure_module.coordinate_augmentation,
        )
        noised = x0_aug + sigma.reshape(-1, 1, 1) * noise

        optimizer.zero_grad(set_to_none=True)
        out = per_sample_denoising_loss(
            policy.structure_module, feats, x0_aug, noised, sigma,
            network_condition_kwargs, preference_mask=None, require_grad=True,
        )
        loss = out.per_sample_loss.mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        step += 1

        record = {
            "step": step,
            "case_id": case_id,
            "sample_id": sample_id,
            "loss": float(loss.detach()),
            "denoise/loss": float(out.per_sample_loss.mean().detach()),
            "sigma": float(sigma.mean()),
            "train/grad_norm": float(grad_norm),
            "train/lr": lr,
            "seconds": time.time() - t0,
        }
        if step % 10 == 0 or step == 1:
            record["param/drift_total"] = parameter_drift(policy, reference)["total"]
        history.append(record)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % 10 == 0 or step == 1:
            print(f"[rwr] step {step}/{max_steps} loss={record['loss']:.4f} "
                  f"grad={record['train/grad_norm']:.3f}", flush=True)
        if step == max_steps or (checkpoint_every and step % checkpoint_every == 0):
            ckpt = output_dir / f"checkpoint_{step:04d}.pt"
            sha = save_native_checkpoint(
                base_checkpoint, policy, ckpt,
                {"method": "rwr", "step": step, "seed": seed, "split": split},
            )
            print(f"[rwr] saved {ckpt.name} sha256={sha[:12]}", flush=True)

    summary = {"steps": step, "final_loss": history[-1]["loss"] if history else None,
               "elapsed_seconds": time.time() - t0}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
