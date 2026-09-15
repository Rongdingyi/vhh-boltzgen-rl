"""Local SNR / gradient-direction audits (task book §54-§56).

These run *before* any training and answer: does one tiny signed-local DPO
step actually push z in the preference direction, and does that hold across
sigma draws?  This is the gate that the failed v2 global-energy proxy could
not pass.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from ..native_atom14.checkpoint import load_base_model, make_policy_reference, trainable_score_params
from ..native_atom14.dpo_trainer import move_conditioning
from .local_dpo import signed_local_dpo_loss
from .local_mask import target_residue_mask
from .trainer import _load_design_positions
from .edge_validator import load_edges

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")


def _cond_bundle(cid: str, cond_dir: Path | None, rollout_root: Path | None,
                 dev: str = DEVICE) -> tuple[dict, dict]:
    """Exact per-case conditioning; feats come from the same captured payload
    the sampler used (single source of truth, no disk re-load drift)."""
    from ..native_atom14.dpo_trainer import load_case_conditioning

    cond = move_conditioning(load_case_conditioning(cid, cond_dir, rollout_root), device=dev)
    feats = cond["feats"]
    kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
              "feats": feats, "multiplicity": 1,
              "diffusion_conditioning": cond["diffusion_conditioning"]}
    return kwargs, feats


def _edge_pair(edge, design_positions, feats, dev: str = DEVICE):
    anchor = torch.load(edge.anchor_coords_path, map_location=dev, weights_only=True).float()
    cf = torch.load(edge.cf_coords_path, map_location=dev, weights_only=True).float()
    if edge.preferred_side == "cf":
        preferred, rejected = cf, anchor
    else:
        preferred, rejected = anchor, cf
    mask = target_residue_mask(feats, edge.position, design_positions[edge.case_id]).to(dev)
    return preferred, rejected, mask


def gradient_direction_audit(base_checkpoint, edges_path, *, n_edges: int = 32,
                             lr: float = 1e-6, beta: float = 10.0,
                             seed: int = 12345, cond_dir: Path | None = None,
                             rollout_root: Path | None = None) -> dict:
    rng = random.Random(seed)
    design_positions = _load_design_positions(
        ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    edges = load_edges(edges_path)
    rng.shuffle(edges)
    edges = edges[:n_edges]
    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    pristine = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    rows = []
    for e in edges:
        kwargs, feats = _cond_bundle(e.case_id, cond_dir, rollout_root)
        preferred, rejected, mask = _edge_pair(e, design_positions, feats)
        params = trainable_score_params(policy)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        sigma = policy.structure_module.noise_distribution(1)
        noise = torch.randn_like(preferred.unsqueeze(0))
        from ..native_atom14.global_cf_step import signed_local_dpo_step

        before = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                       feats, preferred, rejected, mask, kwargs,
                                       beta=beta, sigma=sigma, noise=noise)
        optimizer.zero_grad(set_to_none=True)
        before.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        after = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                      feats, preferred, rejected, mask, kwargs,
                                      beta=beta, sigma=sigma, noise=noise)
        rows.append({"edge_id": e.edge_id, "event_class": e.event_class,
                     "context": e.context, "z_before": float(before.dpo.z.mean()),
                     "z_after": float(after.dpo.z.mean()),
                     "dz": float(after.dpo.z.mean() - before.dpo.z.mean())})
        policy.load_state_dict(pristine)  # restore base for the next edge
    n_pos = sum(1 for r in rows if r["dz"] > 0)
    dzs = [r["dz"] for r in rows]
    out = {
        "n_edges": len(rows),
        "positive_fraction": n_pos / len(rows) if rows else None,
        "median_dz": sorted(dzs)[len(dzs) // 2] if dzs else None,
        "rows": rows,
        "gate": (n_pos / len(rows) >= 0.75) if rows else False,
    }
    return out


def sigma_robustness_audit(base_checkpoint, edges_path, *, n_edges: int = 32,
                           n_sigma: int = 16, lr: float = 1e-6, beta: float = 10.0,
                           seed: int = 12345, cond_dir: Path | None = None,
                           rollout_root: Path | None = None) -> dict:
    rng = random.Random(seed)
    design_positions = _load_design_positions(
        ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    edges = load_edges(edges_path)
    rng.shuffle(edges)
    edges = edges[:n_edges]
    base = load_base_model(base_checkpoint, device=DEVICE)
    policy, reference = make_policy_reference(base, device=DEVICE)
    pristine = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    from ..native_atom14.global_cf_step import signed_local_dpo_step

    per_edge = []
    for e in edges:
        kwargs, feats = _cond_bundle(e.case_id, cond_dir, rollout_root)
        preferred, rejected, mask = _edge_pair(e, design_positions, feats)
        frac_pos = []
        for _ in range(n_sigma):
            params = trainable_score_params(policy)
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
            sigma = policy.structure_module.noise_distribution(1)
            noise = torch.randn_like(preferred.unsqueeze(0))
            before = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                           feats, preferred, rejected, mask, kwargs,
                                           beta=beta, sigma=sigma, noise=noise)
            optimizer.zero_grad(set_to_none=True)
            before.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            after = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                          feats, preferred, rejected, mask, kwargs,
                                          beta=beta, sigma=sigma, noise=noise)
            frac_pos.append(1.0 if float(after.dpo.z.mean()) > float(before.dpo.z.mean()) else 0.0)
            policy.load_state_dict(pristine)
        per_edge.append({"edge_id": e.edge_id, "positive_fraction": sum(frac_pos) / len(frac_pos)})
    fracs = sorted(r["positive_fraction"] for r in per_edge)
    median = fracs[len(fracs) // 2] if fracs else None
    return {"n_edges": len(per_edge), "median_positive_fraction": median,
            "gate": (median is not None and median >= 0.70), "rows": per_edge}
