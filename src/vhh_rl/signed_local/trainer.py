"""SL-CF-DPO trainer (task book §34-§41, §70-§72).

Branch schedule is the deterministic G-G-G-L cycle (75% global CF-DPO updates
frozen exactly as in `weighted_dpo` + 25% signed local correction updates).
The local branch uses only target-residue fake atoms and never touches reward
magnitude.  Output dir is wiped at start; protocol hashes are recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from ..native_atom14.checkpoint import (
    load_base_model, make_policy_reference, parameter_drift,
    save_native_checkpoint, trainable_score_params,
)
from ..native_atom14.dpo_trainer import (
    _load_coords, load_conditioning, load_case_conditioning, move_conditioning,
)
from ..native_atom14.global_cf_step import compute_weighted_cf_dpo_step
from ..native_atom14.masks import design_token_offset, residue_atom_masks
from ..signed_local.local_dpo import signed_local_dpo_loss
from ..signed_local.local_mask import target_residue_mask
from ..signed_local.scheduler import ThreeToOneScheduler
from ..signed_local.edge_validator import load_edges
from ..signed_local.types import TOL

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _load_global_pairs(pairs_path: Path, cases: set[str],
                       weights_path: Path) -> dict[str, list[dict]]:
    weights_all = json.loads(weights_path.read_text())
    by_case: dict[str, list[dict]] = defaultdict(list)
    for line in pairs_path.open():
        pair = json.loads(line)
        if pair["case_id"] not in cases:
            continue
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        entry = weights_all["pairs"].get(pid)
        if entry is None:
            continue
        pair["_weights"] = entry["cf"]
        by_case[pair["case_id"]].append(pair)
    return dict(by_case)


def _load_design_positions(manifest_path: Path) -> dict[str, tuple[int, ...]]:
    out = {}
    for line in manifest_path.open():
        row = json.loads(line)
        out[row["case_id"]] = tuple(sorted(int(p) for p in row["design_positions"]))
    return out


class LocalEdgeSampler:
    """Case-balanced / class-balanced / context-balanced edge sampling (§38-40)."""

    def __init__(self, edges, seed: int, class_ratio: float = 0.5) -> None:
        self.by_case: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for e in edges:
            self.by_case[e.case_id][e.event_class].append(e)
        self.case_ids = sorted(self.by_case)
        self.rng = random.Random(seed)
        self.class_names = sorted({e.event_class for e in edges}) or ["sign_flip"]
        self.class_ratio = class_ratio

    def sample(self, case_hint: str | None = None):
        cid = case_hint if case_hint and case_hint in self.by_case else self.rng.choice(self.case_ids)
        pool = self.by_case[cid]
        if self.rng.random() < self.class_ratio and pool.get("both_negative"):
            cls = "both_negative"
        elif pool.get("sign_flip"):
            cls = "sign_flip"
        elif pool.get("both_negative"):
            cls = "both_negative"
        else:
            cls = self.rng.choice(list(pool))
        cands = pool[cls]
        if cls == "sign_flip" and len({e.context for e in cands}) == 2:
            ctx = "winner_drop" if self.rng.random() < 0.5 else "loser_gain"
            ctx_cands = [e for e in cands if e.context == ctx]
            cands = ctx_cands or cands
        return self.rng.choice(cands)


def run_signed_local(
    base_checkpoint: str | Path,
    edges_path: str | Path,
    pairs_path: str | Path,
    weights_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "mixed",                  # mixed | global_only | local_only
    local_classes: tuple[str, ...] = ("both_negative", "sign_flip"),
    shuffle_direction: bool = False,      # A5 direction-shuffle control
    updates: int = 100,
    schedule: tuple[int, int] = (3, 1),
    beta_global: float = 10.0,
    beta_local: float = 10.0,
    lr: float = 1e-5,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 50,
    seed: int = 20260915,
    conditioning_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    log_tag: str = "slcf",
) -> dict:
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    random.seed(seed)
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)   # §70: no mixed old/new logs
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    edges = [e for e in load_edges(edges_path) if e.event_class in local_classes]
    if not edges:
        raise RuntimeError("no local correction edges for the requested classes")
    sampler = LocalEdgeSampler(edges, seed=seed)
    case_ids_edges = set(sampler.case_ids)

    manifest = Path(manifest_path or ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    design_positions = _load_design_positions(manifest)
    global_pairs = _load_global_pairs(Path(pairs_path), case_ids_edges, Path(weights_path))
    if mode in ("mixed", "global_only") and not global_pairs:
        raise RuntimeError("no global pairs for the local-edge cases")

    cond_dir = Path(conditioning_dir) if conditioning_dir else None
    cond_cache: dict[str, dict] = {}

    def cond_for(cid: str) -> dict:
        if cid not in cond_cache:
            cond = load_case_conditioning(cid, cond_dir, ROOT / "runs/cf_opsd/rollouts/train")
            cond_cache[cid] = move_conditioning(cond, device=DEVICE)
        return cond_cache[cid]

    def feats_for(cid: str) -> dict:
        return cond_for(cid)["feats"]

    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    params = trainable_score_params(policy)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    coords_cache: dict[str, torch.Tensor] = {}

    def coords_of(path: str) -> torch.Tensor:
        if path not in coords_cache:
            coords_cache[path] = torch.load(path, map_location=DEVICE,
                                            weights_only=True).float()
        return coords_cache[path]

    # residue groups for the global branch (same as weighted_dpo)
    residue_cache: dict[str, torch.Tensor] = {}

    def residue_masks_for(cid: str, positions: list[int]) -> torch.Tensor:
        key = f"{cid}:{','.join(map(str, positions))}"
        if key not in residue_cache:
            feats = feats_for(cid)
            offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                         design_positions[cid])
            residue_cache[key] = residue_atom_masks(
                feats["atom_to_token"], feats["fake_atom_mask"],
                feats["atom_pad_mask"], [p + offset for p in positions]).to(DEVICE)
        return residue_cache[key]

    protocol = {
        "method": "sl_cf_dpo", "mode": mode,
        "local_classes": list(local_classes),
        "shuffle_direction": bool(shuffle_direction),
        "updates": updates, "schedule": list(schedule),
        "beta_global": beta_global, "beta_local": beta_local, "lr": lr,
        "seed": seed, "tolerance": TOL,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": _sha256(Path(base_checkpoint)),
        "edges_path": str(edges_path), "edges_sha256": _sha256(Path(edges_path)),
        "pairs_path": str(pairs_path), "pairs_sha256": _sha256(Path(pairs_path)),
        "weights_path": str(weights_path), "weights_sha256": _sha256(Path(weights_path)),
        "credit_csv_sha256": _sha256(ROOT / "runs/next_stage/counterfactual/residue_credit.csv"),
        "cf_scores_sha256": _sha256(ROOT / "runs/next_stage/counterfactual/counterfactual_scores.jsonl"),
        "n_local_edges": len(edges),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=1))

    scheduler = ThreeToOneScheduler(*schedule)
    history = []
    stats = defaultdict(int)
    local_acc = defaultdict(list)
    global_acc: list[float] = []
    local_acc_all: list[float] = []
    local_acc_class: dict[str, list[float]] = defaultdict(list)
    shuffle_rng = random.Random(seed + 1)
    case_cycle = sorted(global_pairs) if global_pairs else sorted(sampler.case_ids)
    t0 = time.time()
    for step in range(1, updates + 1):
        if mode == "global_only":
            branch = "global"
        elif mode == "local_only":
            branch = "local"
        else:
            branch = scheduler.branch(step)
        stats[f"{branch}_steps"] += 1
        record: dict[str, Any] = {"step": step, "branch": branch}

        if branch == "global":
            cid = case_cycle[(step - 1) % len(case_cycle)]
            pair = random.choice(global_pairs[cid])
            pid = f"{cid}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
            positions = sorted(int(p) for p in pair["_weights"])
            w = torch.tensor([pair["_weights"][str(p)] for p in positions],
                             device=DEVICE, dtype=torch.float32)
            feats = feats_for(cid)
            kwargs = {"s_inputs": cond_for(cid)["s_inputs"], "s_trunk": cond_for(cid)["s_trunk"],
                      "feats": feats, "multiplicity": 1,
                      "diffusion_conditioning": cond_for(cid)["diffusion_conditioning"]}
            masks = residue_masks_for(cid, positions)
            winner = _load_coords(pairs_path.parent, pair["winner_sample_id"])
            loser = _load_coords(pairs_path.parent, pair["loser_sample_id"])
            optimizer.zero_grad(set_to_none=True)
            out = compute_weighted_cf_dpo_step(
                policy.structure_module, reference.structure_module, feats,
                winner, loser, masks, w, kwargs, beta=beta_global)
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"non-finite grad norm at global step {step}")
            optimizer.step()
            acc = float(out.dpo.implicit_acc)
            global_acc.append(acc)
            record.update({
                "case_id": cid, "pair_id": pid, "loss": float(out.loss.detach()),
                "z": float(out.dpo.z.mean()), "implicit_acc": acc,
                "grad_norm": float(grad_norm),
                "sigma": float(out.sigma.reshape(-1).mean()),
                "n_weight_residues": len(positions),
            })
        else:
            edge = sampler.sample()
            cid = edge.case_id
            feats = feats_for(cid)
            mask = target_residue_mask(feats, edge.position, design_positions[cid])
            kwargs = {"s_inputs": cond_for(cid)["s_inputs"], "s_trunk": cond_for(cid)["s_trunk"],
                      "feats": feats, "multiplicity": 1,
                      "diffusion_conditioning": cond_for(cid)["diffusion_conditioning"]}
            anchor = coords_of(edge.anchor_coords_path)
            cf = coords_of(edge.cf_coords_path)
            if edge.preferred_side == "cf":
                preferred, rejected = cf, anchor
            else:
                preferred, rejected = anchor, cf
            flipped = False
            if shuffle_direction and shuffle_rng.random() < 0.5:
                preferred, rejected = rejected, preferred
                flipped = True
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = signed_local_dpo_loss(
                policy.structure_module, reference.structure_module, feats,
                preferred, rejected, mask, kwargs, beta=beta_local)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"non-finite grad norm at local step {step}")
            optimizer.step()
            local_acc_all.append(metrics["implicit_acc"])
            local_acc_class[edge.event_class].append(metrics["implicit_acc"])
            local_acc[edge.context].append(metrics["implicit_acc"])
            record.update({
                "case_id": cid, "edge_id": edge.edge_id, "pair_id": edge.pair_id,
                "position": edge.position, "event_class": edge.event_class,
                "context": edge.context, "preferred_side": edge.preferred_side,
                "direction_shuffled": flipped, "dR": edge.dR,
                "loss": float(loss.detach()), "z": metrics["z"],
                "implicit_acc": metrics["implicit_acc"],
                "grad_norm": float(grad_norm), "sigma": metrics["sigma"],
                "moved_rms": edge.moved_rms,
                "lift_ref_percentile": edge.lift_ref_percentile,
            })

        if step % 10 == 0 or step == 1:
            record["param/drift_total"] = parameter_drift(policy, reference)["total"]
        history.append(record)
        _log(metrics_path, record)
        if step % 10 == 0 or step == 1:
            print(f"[{log_tag}] step {step}/{updates} {branch} "
                  f"z={record['z']:+.4f} loss={record['loss']:.4f} "
                  f"grad={record['grad_norm']:.3f}", flush=True)
        if checkpoint_every and (step % checkpoint_every == 0 or step == updates):
            ckpt = output_dir / f"checkpoint_{step:04d}.pt"
            save_native_checkpoint(base_checkpoint, policy, ckpt, {
                "method": "sl_cf_dpo", "mode": mode, "step": step, "seed": seed,
                "edges_path": str(edges_path), "shuffle_direction": shuffle_direction,
            })
            print(f"[{log_tag}] saved {ckpt.name}", flush=True)

    def _mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    summary = {
        "mode": mode,
        "n_global_steps": stats.get("global_steps", 0),
        "n_local_steps": stats.get("local_steps", 0),
        "global_implicit_acc": _mean(global_acc),
        "local_implicit_acc": _mean(local_acc_all),
        "local_acc_by_class": {k: _mean(v) for k, v in local_acc_class.items()},
        "local_acc_by_context": {k: _mean(v) for k, v in local_acc.items()},
        "n_edges_available": len(edges),
        "elapsed_seconds": time.time() - t0,
        "final_loss": history[-1]["loss"] if history else None,
        "protocol": protocol,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
