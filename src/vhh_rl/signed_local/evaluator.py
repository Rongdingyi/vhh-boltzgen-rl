"""SL-CF-DPO evaluation: held-out reward + local preference accuracy (§47/§50-§53)."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from ..native_atom14.checkpoint import load_base_model, make_policy_reference
from ..native_atom14.dpo_trainer import move_conditioning
from ..native_atom14.global_cf_step import signed_local_dpo_step
from .edge_validator import load_edges
from .local_mask import target_residue_mask
from .trainer import _load_design_positions

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")


def evaluate_reward(ckpt: Path | None, cases: list[str], tag: str, *,
                    num_samples: int = 8, run_root: Path | None = None,
                    manifest: Path | None = None) -> dict:
    """Held-out reward / invalid / FR / unique via the shared evaluator."""
    from ..cf_opsd.evaluator import evaluate

    return evaluate(cases, ckpt, tag, num_samples=num_samples, run_root=run_root,
                    manifest=manifest)


def local_preference_accuracy(base_checkpoint, ckpt: Path | None, edges_path,
                              *, beta: float = 10.0, n_sigma: int = 8,
                              seed: int = 12345, cond_dir: Path | None = None,
                              rollout_root: Path | None = None) -> dict:
    """Fraction of held-out local edges with z > 0 (policy prefers the scorer's side)."""
    rng = random.Random(seed)
    design_positions = _load_design_positions(
        ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    edges = load_edges(edges_path)
    policy = load_base_model(ckpt or base_checkpoint, device=DEVICE)
    reference = load_base_model(base_checkpoint, device=DEVICE)
    for p in reference.parameters():
        p.requires_grad_(False)
    policy.eval()
    reference.eval()
    from ..native_atom14.dpo_trainer import load_case_conditioning

    per_edge = []
    for e in edges:
        cond = move_conditioning(
            load_case_conditioning(e.case_id, cond_dir,
                                   rollout_root or ROOT / "runs/cf_opsd/rollouts/train"),
            device=DEVICE)
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        anchor = torch.load(e.anchor_coords_path, map_location=DEVICE,
                            weights_only=True).float()
        cf = torch.load(e.cf_coords_path, map_location=DEVICE, weights_only=True).float()
        preferred, rejected = (cf, anchor) if e.preferred_side == "cf" else (anchor, cf)
        mask = target_residue_mask(feats, e.position, design_positions[e.case_id]).to(DEVICE)
        zs = []
        with torch.no_grad():
            for _ in range(n_sigma):
                out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                            feats, preferred, rejected, mask, kwargs,
                                            beta=beta)
                zs.append(float(out.dpo.z.mean()))
        per_edge.append({"edge_id": e.edge_id, "event_class": e.event_class,
                         "context": e.context, "z": sum(zs) / len(zs)})
    tol = 1e-12

    def label(z: float) -> float:
        if z > tol:
            return 1.0
        if z < -tol:
            return 0.0
        return 0.5  # exact ties count as chance

    acc = defaultdict(list)
    for r in per_edge:
        acc["all"].append(label(r["z"]))
        acc[r["event_class"]].append(label(r["z"]))
        acc[r["context"]].append(label(r["z"]))
    zs_all = sorted(r["z"] for r in per_edge)
    n_ties = sum(1 for r in per_edge if abs(r["z"]) <= tol)
    return {
        "n_edges": len(per_edge),
        "n_ties": n_ties,
        "accuracy": {k: (sum(v) / len(v) if v else None) for k, v in acc.items()},
        "median_z": zs_all[len(zs_all) // 2] if zs_all else None,
        "per_edge": per_edge,
    }
