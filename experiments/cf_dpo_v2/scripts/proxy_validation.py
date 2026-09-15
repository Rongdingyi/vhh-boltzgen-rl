#!/usr/bin/env python
"""CF-DPO v2 proxy validation (proposal §11.2/§11.3, Exp3 supplement).

For trained/untrained checkpoints: does the denoising-energy proxy order the
edge endpoints the same way the classifier reward does?  Also reports the
across-sigma noise of the proxy (how much of the signal is proxy noise).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_dpo_v2.signed_trainer import edge_loss  # noqa: E402
from vhh_rl.native_atom14.checkpoint import load_base_model, make_policy_reference  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import move_conditioning  # noqa: E402

OUT = ROOT / "runs/cf_dpo_v2/proxy_validation"
DOC = ROOT / "docs/cf_dpo_v2/EXP3_PROXY_VALIDATION.md"
GRAPH = ROOT / "runs/cf_dpo_v2/graph/pilot_graph.pt"
BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n-sigma", type=int, default=8)
    parser.add_argument("--max-edges", type=int, default=80)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)

    graph = torch.load(GRAPH, map_location="cpu", weights_only=False)
    nodes, edges = graph["nodes"], graph["edges"]
    edges = [e for e in edges if e["kind"] != "same_seq"]
    # stratified random sampling (not head-of-list)
    rng = random.Random(0)
    by_kind: dict[str, list] = defaultdict(list)
    for e in edges:
        by_kind[e["kind"]].append(e)
    per_kind = max(1, args.max_edges // max(1, len(by_kind)))
    sampled = []
    for kind, lst in sorted(by_kind.items()):
        rng.shuffle(lst)
        sampled.extend(lst[:per_kind])
    edges = sampled[: args.max_edges]

    # policy = evaluated checkpoint (or base); reference = the ORIGINAL frozen base
    policy = load_base_model(args.ckpt or BASE, device="cuda")
    reference = load_base_model(BASE, device="cuda")
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    for p in policy.parameters():
        p.requires_grad_(False)

    cond_cache: dict[str, dict] = {}
    rows = []
    for edge in edges:
        node_a, node_b = nodes[edge["a"]], nodes[edge["b"]]
        cid = node_a["case_id"]
        if cid not in cond_cache:
            rollout_dir = ROOT / "runs/cf_opsd/rollouts/train" / cid
            payload = torch.load(sorted(rollout_dir.glob("seed*.pt"))[0],
                                 map_location="cpu", weights_only=False)
            cond_cache[cid] = move_conditioning(payload["cond_kwargs"], device="cuda")
        cond = cond_cache[cid]
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        coords_a = node_a["coords"].to("cuda").float()
        coords_b = node_b["coords"].to("cuda").float()
        diffs = []
        for _ in range(args.n_sigma):
            sigma = policy.structure_module.noise_distribution(1)
            noise = torch.randn_like(coords_a.unsqueeze(0))
            z, info = edge_loss(policy.structure_module, reference.structure_module,
                                feats, coords_a, coords_b, sigma, noise, kwargs,
                                tau=1.0, kappa=1.0, require_grad=False)
            diffs.append(info["h_b"] - info["h_a"])
        # ground truth recomputed from node rewards (not the stored edge label)
        dR_true = None
        if node_a.get("reward") is not None and node_b.get("reward") is not None:
            dR_true = node_b["reward"] - node_a["reward"]
        rows.append({"kind": edge["kind"], "dR_stored": edge["dR"],
                     "dR_node": dR_true,
                     "delta_h_mean": st.mean(diffs),
                     "delta_h_std": st.stdev(diffs) if len(diffs) > 1 else 0.0})
    mismatch = sum(1 for r in rows
                   if r["dR_node"] is not None
                   and abs(r["dR_node"] - r["dR_stored"]) > 1e-6)
    agree = [r for r in rows if r["dR_node"] not in (None, 0)]
    sign_ok = sum(1 for r in agree
                  if (r["delta_h_mean"] > 0) == (r["dR_node"] > 0))
    sign_acc = sign_ok / max(1, len(agree))
    noise = st.median([r["delta_h_std"] for r in rows])
    summary = {"checkpoint": str(args.ckpt or "base"), "n_edges": len(rows),
               "n_sign_evaluated": len(agree),
               "stored_vs_node_label_mismatches": mismatch,
               "sign_agreement_vs_node_reward": sign_acc, "median_sigma_noise": noise,
               "median_abs_delta_h": st.median([abs(r["delta_h_mean"]) for r in rows])}
    (OUT / f"proxy_{args.ckpt.stem if args.ckpt else 'base'}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    doc = (f"# CF-DPO v2 proxy validation\n\n"
           f"- checkpoint: {summary['checkpoint']}\n"
           f"- edges: {summary['n_edges']} (stratified random sample)\n"
           f"- sign-evaluated edges (dR != 0): {summary['n_sign_evaluated']}\n"
           f"- stored-vs-node label mismatches: {summary['stored_vs_node_label_mismatches']}\n"
           f"- **sign agreement vs node reward: {sign_acc:.3f}**\n"
           f"- median across-sigma noise (std of dh): {noise:.3e}\n"
           f"- median |dh|: {summary['median_abs_delta_h']:.3e}\n")
    DOC.write_text(doc)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
