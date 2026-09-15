#!/usr/bin/env python
"""Mechanism audit: global-CF vs signed-local gradient conflict (task book §52-§56).

Diagnostic only; nothing here enters training and no gradient surgery is
implemented.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

import _common as C  # noqa: E402

SL_EDGES = C.ROOT / "runs/signed_local/edges/train_edges.jsonl"
OUT = C.AUDIT_DIR / "gradient_conflict.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sites", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)

    from vhh_rl.native_atom14.checkpoint import (
        load_base_model, make_policy_reference, trainable_score_params,
    )
    from vhh_rl.native_atom14.dpo_trainer import (
        _load_coords, load_case_conditioning, move_conditioning,
    )
    from vhh_rl.native_atom14.global_cf_step import (
        compute_weighted_cf_dpo_step, signed_local_dpo_step,
    )
    from vhh_rl.native_atom14.masks import design_token_offset, residue_atom_masks
    from vhh_rl.signed_local.edge_validator import load_edges
    from vhh_rl.signed_local.local_mask import target_residue_mask

    edges = [e for e in load_edges(SL_EDGES) if e.event_class == "sign_flip"]
    rng.shuffle(edges)
    edges = edges[: args.n_sites]
    pairs = {C.pair_id(p): p for p in C.load_pairs()}
    credit = {pid: {r["position"]: r for r in rows}
              for pid, rows in C.load_residue_rows().items()}
    design = {json.loads(l)["case_id"]: tuple(json.loads(l)["design_positions"])
              for l in C.MANIFEST.open()}

    model = load_base_model(C.BASE_CKPT, device="cuda")
    policy, reference = make_policy_reference(model, device="cuda")
    params = trainable_score_params(policy)

    cond_cache: dict[str, dict] = {}

    def cond_for(case_id: str) -> dict:
        if case_id not in cond_cache:
            cond_cache[case_id] = move_conditioning(
                load_case_conditioning(case_id, C.POOL / "conditioning",
                                       C.ROOT / "runs/cf_opsd/rollouts/train"),
                device="cuda")
        return cond_cache[case_id]

    rows = []
    for edge in edges:
        pair = pairs.get(edge.pair_id)
        if pair is None:
            continue
        cond = cond_for(edge.case_id)
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        cf = C.load_cf_pair_weights(edge.pair_id)
        positions = sorted(cf)
        offset = design_token_offset(feats["token_index"], feats["design_mask"],
                                     design[edge.case_id])
        masks = residue_atom_masks(feats["atom_to_token"], feats["fake_atom_mask"],
                                   feats["atom_pad_mask"],
                                   [p + offset for p in positions]).to("cuda")
        w = torch.tensor([cf[p] for p in positions], device="cuda")
        w = w / w.sum().clamp_min(1e-12)
        winner = _load_coords(C.POOL, pair["winner_sample_id"])
        loser = _load_coords(C.POOL, pair["loser_sample_id"])
        g_step = compute_weighted_cf_dpo_step(policy.structure_module,
                                              reference.structure_module, feats,
                                              winner, loser, masks, w, kwargs,
                                              beta=10.0)
        g_grads = torch.autograd.grad(g_step.loss, params, retain_graph=False,
                                      allow_unused=True)

        mask = target_residue_mask(feats, edge.position, design[edge.case_id]).to("cuda")
        anchor = torch.load(edge.anchor_coords_path, map_location="cuda",
                            weights_only=True).float()
        cfc = torch.load(edge.cf_coords_path, map_location="cuda",
                         weights_only=True).float()
        preferred, rejected = ((cfc, anchor) if edge.preferred_side == "cf"
                               else (anchor, cfc))
        l_step = signed_local_dpo_step(policy.structure_module,
                                       reference.structure_module, feats,
                                       preferred, rejected, mask, kwargs, beta=10.0)
        l_grads = torch.autograd.grad(l_step.loss, params, allow_unused=True)

        dot = sum((g * h).sum() for g, h in zip(g_grads, l_grads)
                  if g is not None and h is not None)
        ng = sum(g.pow(2).sum() for g in g_grads if g is not None).sqrt()
        nl = sum(h.pow(2).sum() for h in l_grads if h is not None).sqrt()
        denom = float(ng * nl)
        cosine = float(dot) / denom if denom > 0 else None
        cred = credit.get(edge.pair_id, {}).get(edge.position, {})
        rows.append({
            "edge_id": edge.edge_id, "pair_id": edge.pair_id,
            "case_id": edge.case_id, "region": cred.get("region"),
            "c_drop": cred.get("c_drop"), "c_gain": cred.get("c_gain"),
            "cosine": cosine,
            "z_global": float(g_step.dpo.z.mean()),
            "z_local": float(l_step.dpo.z.mean()),
        })

    cosines = [r["cosine"] for r in rows if r["cosine"] is not None]
    cosines_sorted = sorted(cosines)
    payload = {
        "n_sites": len(rows),
        "median_cosine": statistics.median(cosines) if cosines else None,
        "p25": cosines_sorted[len(cosines_sorted) // 4] if cosines else None,
        "p75": cosines_sorted[(3 * len(cosines_sorted)) // 4] if cosines else None,
        "negative_fraction": (sum(1 for c in cosines if c < 0) / len(cosines))
        if cosines else None,
        "strong_negative_fraction": (sum(1 for c in cosines if c < -0.1) / len(cosines))
        if cosines else None,
        "by_region": _group(rows, "region"),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in
                      ("n_sites", "median_cosine", "p25", "p75",
                       "negative_fraction", "strong_negative_fraction")}, indent=1))


def _group(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["cosine"] is None:
            continue
        groups[str(row.get(key))].append(row["cosine"])
    return {k: {"n": len(v), "median_cosine": statistics.median(v),
                "negative_fraction": sum(1 for c in v if c < 0) / len(v)}
            for k, v in sorted(groups.items())}


if __name__ == "__main__":
    main()
