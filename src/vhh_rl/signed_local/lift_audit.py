"""Lift manifold audit (task book §20-§24).

Compares the reference-model local denoising loss of lifted counterfactual
endpoints against the same-case native pool distribution.  Nothing is filtered
before the audit; the gate decides.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from ..native_atom14.checkpoint import load_base_model
from ..native_atom14.denoise_loss import per_residue_denoising_loss
from ..native_atom14.dpo_trainer import (
    load_case_conditioning, load_conditioning, move_conditioning,
)
from ..native_atom14.masks import design_token_offset, residue_atom_masks
from .edge_validator import load_edges

DEVICE = "cuda"
ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"


def _site_mask(feats: dict, position: int, design_positions) -> torch.Tensor:
    offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                 design_positions)
    return residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                              feats["atom_pad_mask"], [position + offset])[0].bool()


def _local_loss(sm, feats: dict, coords: torch.Tensor, mask: torch.Tensor,
                kwargs: dict, n_sigma: int, rng: random.Random) -> float:
    vals = []
    for _ in range(n_sigma):
        sigma = sm.noise_distribution(1)
        noise = torch.randn_like(coords.unsqueeze(0))
        out = per_residue_denoising_loss(
            sm, feats, coords.unsqueeze(0), coords.unsqueeze(0) + sigma.reshape(-1, 1, 1) * noise,
            sigma, kwargs, mask.unsqueeze(0).float().to(coords.device),
            require_grad=False)
        vals.append(float(out[0].mean()))
    return sum(vals) / len(vals)


def native_local_loss_distribution(base_checkpoint, cases, design_positions,
                                   *, split: str = "train", samples_per_case: int = 16,
                                   max_sites: int = 6, n_sigma: int = 8,
                                   seed: int = 12345, rollout_root: Path | None = None,
                                   conditioning_dir: Path | None = None) -> dict:
    rng = random.Random(seed)
    model = load_base_model(base_checkpoint, device=DEVICE)
    model.eval()
    sm = model.structure_module
    out = {}
    for cid in cases:
        cond = move_conditioning(
            load_case_conditioning(cid, conditioning_dir, rollout_root), device=DEVICE)
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        pos_all = list(design_positions[cid])
        rows = []
        meta_rows = [json.loads(l) for l in (POOL / split / cid / "metadata.jsonl").open()]
        meta_rows = [r for r in meta_rows if r["reward_raw"] is not None][: samples_per_case]
        for r in meta_rows:
            coords = torch.load(r["coords_path"], map_location=DEVICE,
                                weights_only=True).float()
            sites = rng.sample(pos_all, min(max_sites, len(pos_all)))
            for pos in sites:
                mask = _site_mask(feats, pos, design_positions[cid]).to(DEVICE)
                rows.append(_local_loss(sm, feats, coords, mask, kwargs, n_sigma, rng))
        out[cid] = rows
    return out


def lifted_local_losses(base_checkpoint, edges, design_positions, *,
                        n_sigma: int = 8, seed: int = 12345,
                        rollout_root: Path | None = None,
                        conditioning_dir: Path | None = None) -> dict[str, float]:
    rng = random.Random(seed)
    model = load_base_model(base_checkpoint, device=DEVICE)
    model.eval()
    sm = model.structure_module
    out = {}
    for e in edges:
        cond = move_conditioning(
            load_case_conditioning(e.case_id, conditioning_dir, rollout_root), device=DEVICE)
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        coords = torch.load(e.cf_coords_path, map_location=DEVICE, weights_only=True).float()
        mask = _site_mask(feats, e.position, design_positions[e.case_id]).to(DEVICE)
        out[e.edge_id] = _local_loss(sm, feats, coords, mask, kwargs, n_sigma, rng)
    return out


def run_manifold_audit(base_checkpoint, edges_path, out_dir, design_positions,
                       *, n_sigma: int = 8, seed: int = 12345,
                       rollout_root: Path | None = None,
                       conditioning_dir: Path | None = None) -> dict:
    edges = load_edges(edges_path)
    cases = sorted({e.case_id for e in edges})
    native = native_local_loss_distribution(
        base_checkpoint, cases, design_positions, n_sigma=n_sigma, seed=seed,
        rollout_root=rollout_root, conditioning_dir=conditioning_dir)
    lifted = lifted_local_losses(
        base_checkpoint, edges, design_positions, n_sigma=n_sigma, seed=seed,
        rollout_root=rollout_root, conditioning_dir=conditioning_dir)
    p99 = {cid: sorted(v)[max(0, int(0.99 * (len(v) - 1)))] for cid, v in native.items() if v}
    per_edge = []
    for e in edges:
        thr = p99.get(e.case_id)
        if thr is None or e.edge_id not in lifted:
            continue
        val = lifted[e.edge_id]
        # percentile of the lifted value inside the same-case native distribution
        vals = native[e.case_id]
        pct = sum(1 for v in vals if v <= val) / len(vals)
        per_edge.append({"edge_id": e.edge_id, "case_id": e.case_id,
                         "lift_loss": val, "native_p99": thr, "percentile": pct,
                         "outlier": pct > 0.99})
    frac = (sum(1 for r in per_edge if r["outlier"]) / len(per_edge)) if per_edge else None
    if frac is None:
        gate = "NO_DATA"
    elif frac <= 0.20:
        gate = "PASS"
    elif frac <= 0.40:
        gate = "FILTERED_PILOT"
    else:
        gate = "STOP"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"n_edges": len(edges), "n_audited": len(per_edge),
               "p99_outlier_fraction": frac, "gate": gate,
               "native_mean_by_case": {c: (sum(v) / len(v) if v else None)
                                       for c, v in native.items()}}
    (out_dir / "manifold_audit.json").write_text(json.dumps(
        {"summary": summary, "per_edge": per_edge}, indent=1))
    return summary
