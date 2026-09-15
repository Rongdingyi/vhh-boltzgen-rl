"""Phase C/D trainer: CF-weighted (+ temporal-weighted) fake-atom DPO.

Same DPO math, same paired noise/rigid augmentation, same trainable scope as
N3; the only change is how the per-residue denoising losses are aggregated
(task book §31-§47).

  variant  : "cf" | "random_sparse" | "shuffle" | "region"
  temporal : "none" | "hard" | "smooth"
             L = g(sigma) * L_DPO, g from sigma_seq (tau=0.5, no sweep)
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
from .dpo_trainer import _load_coords, load_conditioning, move_conditioning
from .global_cf_step import compute_weighted_cf_dpo_step
from .masks import design_token_offset, residue_atom_masks

DEVICE = "cuda"


def _log(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def _pair_weights(all_weights: dict, pair_id: str, variant: str) -> dict[int, float]:
    entry = all_weights["pairs"].get(pair_id)
    if entry is None:
        raise KeyError(f"no weights for pair {pair_id}")
    raw = entry[variant]
    return {int(k): float(v) for k, v in raw.items()}


def _load_design_positions(manifest_path: str | Path) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for line in Path(manifest_path).open():
        row = json.loads(line)
        out[row["case_id"]] = tuple(sorted(int(p) for p in row["design_positions"]))
    return out


def run_weighted_dpo(
    base_checkpoint: str | Path,
    pairs_path: str | Path,
    pool_root: str | Path,
    conditioning_dir: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    variant: str = "cf",
    temporal: str = "none",
    sigma_seq: float | None = None,
    tau: float = 0.5,
    beta: float = 10.0,
    lr: float = 1e-5,
    max_steps: int = 500,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 50,
    seed: int = 20260913,
    smoke_steps: int | None = None,
    max_cases: int | None = None,
    log_tag: str = "cf-dpo",
) -> dict[str, Any]:
    if temporal != "none" and sigma_seq is None:
        raise ValueError("sigma_seq required for temporal weighting")
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    pairs = [json.loads(l) for l in Path(pairs_path).open()]
    if max_cases:
        keep = sorted({p["case_id"] for p in pairs})[:max_cases]
        pairs = [p for p in pairs if p["case_id"] in set(keep)]
    if not pairs:
        raise RuntimeError("empty preference dataset")
    weights_all = json.loads(Path(weights_path).read_text())
    print(f"[{log_tag}] variant={variant} temporal={temporal} "
          f"pairs={len(pairs)} cases={len({p['case_id'] for p in pairs})}", flush=True)

    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    train_params = trainable_score_params(policy)
    n_train = sum(p.numel() for p in train_params)
    print(f"[{log_tag}] trainable score_model params: {n_train:,}", flush=True)
    optimizer = torch.optim.AdamW(train_params, lr=lr, weight_decay=0.0)

    by_case: dict[str, list[dict]] = {}
    for pair in pairs:
        by_case.setdefault(pair["case_id"], []).append(pair)
    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[3] / "runs/round1_rl_split/rl_manifest_split.jsonl"
    design_positions = _load_design_positions(manifest_path)
    uniform_all = variant == "uniform_all"  # temporal-only ablation uses N3's mask
    cond_cache: dict[str, dict[str, Any]] = {}
    offsets: dict[str, int] = {}
    for case_id in by_case:
        cond_cache[case_id] = move_conditioning(load_conditioning(conditioning_dir, case_id))
        feats = cond_cache[case_id]["feats"]
        full_design = design_positions.get(case_id)
        if full_design is None:
            raise KeyError(f"{case_id} missing from {manifest_path}")
        if not uniform_all:
            for pair in by_case[case_id]:
                bad = set(_pair_weights(weights_all, _pid(pair), variant)) - set(full_design)
                if bad:
                    raise ValueError(f"{_pid(pair)}: weight positions outside design set {sorted(bad)[:5]}")
        offsets[case_id] = design_token_offset(
            feats["token_index"], feats["design_mask"], full_design)
    print(f"[{log_tag}] conditioning cached; token offsets {offsets}", flush=True)

    case_ids = sorted(by_case)
    step = 0
    history = []
    t0 = time.time()
    while step < max_steps:
        case_id = case_ids[step % len(case_ids)]
        pair = random.choice(by_case[case_id])
        pid = _pid(pair)
        if uniform_all:
            full_design = design_positions[case_id]
            weights = {p: 1.0 / len(full_design) for p in full_design}
        else:
            weights = _pair_weights(weights_all, pid, variant)
        positions = sorted(weights)
        cond = cond_cache[case_id]
        feats = cond["feats"]
        network_condition_kwargs = {
            "s_inputs": cond["s_inputs"],
            "s_trunk": cond["s_trunk"],
            "feats": feats,
            "multiplicity": 1,
            "diffusion_conditioning": cond["diffusion_conditioning"],
        }
        token_ids = [p + offsets[case_id] for p in positions]
        residue_masks = residue_atom_masks(
            feats["atom_to_token"], feats["fake_atom_mask"],
            feats["atom_pad_mask"], token_ids,
        ).to(DEVICE)
        w = torch.tensor([weights[p] for p in positions], device=DEVICE, dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)

        winner = _load_coords(pool_root, pair["winner_sample_id"])
        loser = _load_coords(pool_root, pair["loser_sample_id"])

        optimizer.zero_grad(set_to_none=True)
        step_out = compute_weighted_cf_dpo_step(
            policy.structure_module, reference.structure_module, feats,
            winner, loser, residue_masks, w, network_condition_kwargs, beta=beta)
        dpo = step_out.dpo
        raw_loss = step_out.loss
        lw_loss, ll_loss = step_out.winner_loss, step_out.loser_loss
        sigma_value = float(step_out.sigma.reshape(-1).mean())
        g_sigma = 1.0
        if temporal == "smooth":
            from vhh_rl.credit.temporal_schedule import g_smooth
            g_sigma = g_smooth(sigma_value, float(sigma_seq), tau)
        elif temporal == "hard":
            from vhh_rl.credit.temporal_schedule import g_hard
            g_sigma = g_hard(sigma_value, float(sigma_seq))
        loss = g_sigma * raw_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(train_params, max_grad_norm)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        optimizer.step()
        step += 1

        record = {
            "step": step, "case_id": case_id, "pair_id": pid,
            "loss": float(loss.detach()),
            "raw_dpo_loss": float(raw_loss.detach()),
            "weighted_dpo_loss": float(loss.detach()),
            "g_sigma": float(g_sigma),
            "sigma": sigma_value,
            "dpo/z_mean": float(dpo.z.mean()),
            "dpo/implicit_acc": float(dpo.implicit_acc),
            "dpo/model_diff_mean": float(dpo.model_diff.mean()),
            "dpo/ref_diff_mean": float(dpo.ref_diff.mean()),
            "denoise/winner_current": float(lw_loss.detach()),
            "denoise/loser_current": float(ll_loss.detach()),
            "n_weight_residues": len(positions),
            "weight_entropy": float(-(w * w.clamp_min(1e-12).log()).sum()),
            "train/grad_norm": float(grad_norm),
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
                  f"raw={record['raw_dpo_loss']:.4f} g={g_sigma:.3f} "
                  f"z={record['dpo/z_mean']:+.4f} acc={record['dpo/implicit_acc']:.2f} "
                  f"grad={record['train/grad_norm']:.3f}", flush=True)
        if step == max_steps or (checkpoint_every and step % checkpoint_every == 0):
            ckpt_path = output_dir / f"checkpoint_{step:04d}.pt"
            sha = save_native_checkpoint(
                base_checkpoint, policy, ckpt_path,
                {
                    "method": log_tag, "step": step, "beta": beta,
                    "variant": variant, "temporal": temporal,
                    "sigma_seq": sigma_seq, "tau": tau,
                    "weights_path": str(weights_path),
                    "pairs_path": str(pairs_path), "seed": seed,
                },
            )
            print(f"[{log_tag}] saved {ckpt_path.name} sha256={sha[:12]}", flush=True)
        if smoke_steps is not None and step >= smoke_steps:
            break

    summary = {
        "steps": step, "n_pairs": len(pairs), "variant": variant,
        "temporal": temporal, "sigma_seq": sigma_seq,
        "n_trainable_params": n_train,
        "final_loss": history[-1]["loss"] if history else None,
        "final_raw_dpo_loss": history[-1]["raw_dpo_loss"] if history else None,
        "final_implicit_acc": history[-1]["dpo/implicit_acc"] if history else None,
        "elapsed_seconds": time.time() - t0,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _pid(pair: dict) -> str:
    return f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
