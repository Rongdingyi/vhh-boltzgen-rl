"""N2/N3/N4 Diffusion-DPO trainer (task book §28-52, §70-81).

Offline preference DPO over the native atom14 design model:
  - only ``structure_module.score_model`` is trained;
  - trunk conditioning is precomputed per case (frozen);
  - winner/loser share sigma, Gaussian noise and rigid augmentation;
  - per-sample coordinate weighted-MSE replicates the official loss;
  - preference mask: all design atoms (N2) or CDR fake atoms (N3);
  - optional reference/backbone anchor on non-preference atoms (N4).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from .anchor import anchor_loss
from .checkpoint import (
    load_base_model,
    make_policy_reference,
    parameter_drift,
    save_native_checkpoint,
    trainable_score_params,
)
from .denoise_loss import per_sample_denoising_loss
from .dpo_loss import diffusion_dpo_loss
from .masks import anchor_mask, atom_design_mask, cdr_fake_atom_mask
from .paired_noise import paired_noising
from .preference_dataset import build_pairs

DEVICE = "cuda"


def _log(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _compute_masks(feats: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    design = atom_design_mask(
        feats["atom_to_token"], feats["design_mask"], feats["atom_pad_mask"]
    )
    if mode == "all_design_atoms":
        pref = design
    elif mode == "cdr_fake_atoms":
        pref = cdr_fake_atom_mask(
            feats["atom_to_token"], feats["design_mask"],
            feats["fake_atom_mask"], feats["atom_pad_mask"],
        )
    else:
        raise ValueError(f"unknown preference mask {mode}")
    anchor = anchor_mask(feats["atom_pad_mask"], pref)
    return pref, anchor


def load_conditioning(conditioning_dir: Path, case_id: str) -> dict[str, Any]:
    path = Path(conditioning_dir) / f"{case_id}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def _move_value(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict) and value.get("__partial__"):
        import functools

        from boltzgen.model.modules.encoders import single_to_keys

        funcs = {"single_to_keys": single_to_keys}
        keywords = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in value["keywords"].items()
        }
        return functools.partial(funcs[value["func"]], **keywords)
    if callable(value) and hasattr(value, "keywords") and hasattr(value, "func"):
        # legacy cache format: raw functools.partial
        import functools

        keywords = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in value.keywords.items()
        }
        return functools.partial(value.func, **keywords)
    return value


def move_conditioning(payload: dict[str, Any], device: str = DEVICE) -> dict[str, Any]:
    out: dict[str, Any] = {
        "s_inputs": payload["s_inputs"].to(device),
        "s_trunk": payload["s_trunk"].to(device),
        "feats": {
            k: _move_value(v, device)
            for k, v in payload["feats"].items()
        },
        "diffusion_conditioning": {
            k: _move_value(v, device)
            for k, v in payload["diffusion_conditioning"].items()
        },
    }
    return out


def run_dpo(
    base_checkpoint: str | Path,
    pairs_path: str | Path,
    pool_root: str | Path,
    conditioning_dir: str | Path,
    output_dir: str | Path,
    *,
    preference_mask: str = "cdr_fake_atoms",
    beta: float = 1.0,
    anchor_lambda: float = 0.0,
    lr: float = 1e-5,
    max_steps: int = 500,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 50,
    eval_every: int = 50,
    seed: int = 20260913,
    smoke_steps: int | None = None,
    log_tag: str = "dpo",
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    pairs = [json.loads(l) for l in Path(pairs_path).open()]
    if not pairs:
        raise RuntimeError("empty preference dataset")
    with (output_dir / "pairs_used.jsonl").open("w") as handle:
        for p in pairs:
            handle.write(json.dumps(p) + "\n")

    print(f"[{log_tag}] loading base model from {base_checkpoint}", flush=True)
    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    train_params = trainable_score_params(policy)
    n_train = sum(p.numel() for p in train_params)
    print(f"[{log_tag}] trainable score_model params: {n_train:,}", flush=True)
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.0)

    # group pairs by case and preload conditioning to GPU
    by_case: dict[str, list[dict]] = {}
    for pair in pairs:
        by_case.setdefault(pair["case_id"], []).append(pair)
    cond_cache: dict[str, dict[str, Any]] = {}
    for case_id in by_case:
        cond_cache[case_id] = move_conditioning(load_conditioning(conditioning_dir, case_id))
    print(f"[{log_tag}] conditioning cached for {len(cond_cache)} cases", flush=True)

    case_ids = sorted(by_case)
    step = 0
    history = []
    t0 = time.time()
    while step < max_steps:
        case_id = case_ids[step % len(case_ids)]
        pair = random.choice(by_case[case_id])
        cond = cond_cache[case_id]
        feats = cond["feats"]
        network_condition_kwargs = {
            "s_inputs": cond["s_inputs"],
            "s_trunk": cond["s_trunk"],
            "feats": feats,
            "multiplicity": 1,
            "diffusion_conditioning": cond["diffusion_conditioning"],
        }
        pref_mask, anc_mask = _compute_masks(feats, preference_mask)

        winner = _load_coords(pool_root, pair["winner_sample_id"])
        loser = _load_coords(pool_root, pair["loser_sample_id"])
        atom_mask = feats["atom_pad_mask"]
        if atom_mask.dim() > 1:
            atom_mask = atom_mask.reshape(-1)

        sigma = policy.structure_module.noise_distribution(1)
        noise = torch.randn_like(winner.unsqueeze(0))
        paired = paired_noising(
            winner, loser, atom_mask, sigma,
            augmentation=policy.structure_module.coordinate_augmentation,
            noise=noise,
        )

        optimizer.zero_grad(set_to_none=True)
        out_w = per_sample_denoising_loss(
            policy.structure_module, feats,
            paired["x0_w_aug"], paired["x_t_w"], sigma,
            network_condition_kwargs, preference_mask=pref_mask, require_grad=True,
        )
        out_l = per_sample_denoising_loss(
            policy.structure_module, feats,
            paired["x0_l_aug"], paired["x_t_l"], sigma,
            network_condition_kwargs, preference_mask=pref_mask, require_grad=True,
        )
        ref_w = per_sample_denoising_loss(
            reference.structure_module, feats,
            paired["x0_w_aug"], paired["x_t_w"], sigma,
            network_condition_kwargs, preference_mask=pref_mask, require_grad=False,
        )
        ref_l = per_sample_denoising_loss(
            reference.structure_module, feats,
            paired["x0_l_aug"], paired["x_t_l"], sigma,
            network_condition_kwargs, preference_mask=pref_mask, require_grad=False,
        )
        dpo = diffusion_dpo_loss(
            out_w.per_sample_loss, out_l.per_sample_loss,
            ref_w.per_sample_loss, ref_l.per_sample_loss, beta,
        )
        loss = dpo.total_loss
        anc_value = torch.zeros((), device=DEVICE)
        if anchor_lambda > 0.0:
            anc_value = anchor_loss(
                out_w.denoised_coords, ref_w.denoised_coords,
                out_l.denoised_coords, ref_l.denoised_coords, anc_mask,
            )
            loss = loss + anchor_lambda * anc_value
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        step += 1

        record = {
            "step": step,
            "case_id": case_id,
            "loss": float(loss.detach()),
            "dpo/loss": float(dpo.dpo_loss.detach()),
            "dpo/z_mean": float(dpo.z.mean()),
            "dpo/z_std": float(dpo.z.std()) if dpo.z.numel() > 1 else 0.0,
            "dpo/implicit_acc": float(dpo.implicit_acc),
            "dpo/model_diff_mean": float(dpo.model_diff.mean()),
            "dpo/ref_diff_mean": float(dpo.ref_diff.mean()),
            "denoise/winner_current": float(out_w.per_sample_loss.mean().detach()),
            "denoise/loser_current": float(out_l.per_sample_loss.mean().detach()),
            "denoise/winner_ref": float(ref_w.per_sample_loss.mean().detach()),
            "denoise/loser_ref": float(ref_l.per_sample_loss.mean().detach()),
            "anchor/loss": float(anc_value.detach()),
            "train/grad_norm": float(grad_norm),
            "train/lr": lr,
            "sigma": float(sigma.mean()),
            "reward_gap": pair["reward_gap"],
            "seconds": time.time() - t0,
        }
        if step % 10 == 0 or step == 1:
            drift = parameter_drift(policy, reference)
            record["param/drift_total"] = drift["total"]
        history.append(record)
        _log(metrics_path, record)
        if step % 10 == 0 or step == 1:
            print(f"[{log_tag}] step {step}/{max_steps} loss={record['loss']:.4f} "
                  f"z={record['dpo/z_mean']:+.4f} acc={record['dpo/implicit_acc']:.2f} "
                  f"grad={record['train/grad_norm']:.3f}", flush=True)
        if step == max_steps or (checkpoint_every and step % checkpoint_every == 0):
            ckpt_path = output_dir / f"checkpoint_{step:04d}.pt"
            sha = save_native_checkpoint(
                base_checkpoint, policy, ckpt_path,
                {
                    "method": log_tag,
                    "step": step,
                    "beta": beta,
                    "anchor_lambda": anchor_lambda,
                    "preference_mask": preference_mask,
                    "pairs_path": str(pairs_path),
                    "seed": seed,
                },
            )
            print(f"[{log_tag}] saved {ckpt_path.name} sha256={sha[:12]}", flush=True)
        if smoke_steps is not None and step >= smoke_steps:
            break

    summary = {
        "steps": step,
        "n_pairs": len(pairs),
        "n_trainable_params": n_train,
        "final_loss": history[-1]["loss"] if history else None,
        "final_implicit_acc": history[-1]["dpo/implicit_acc"] if history else None,
        "elapsed_seconds": time.time() - t0,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _load_coords(pool_root: str | Path, sample_id: str) -> torch.Tensor:
    """Find the sample coords by id under pool_root/*/<case>/coords/<id>.pt."""
    pool_root = Path(pool_root)
    case_id = sample_id.rsplit("_s", 1)[0]
    for split_dir in pool_root.iterdir():
        candidate = split_dir / case_id / "coords" / f"{sample_id}.pt"
        if candidate.is_file():
            return torch.load(candidate, map_location=DEVICE, weights_only=True).float()
    raise FileNotFoundError(f"coords for {sample_id} not found under {pool_root}")
