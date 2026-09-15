#!/usr/bin/env python
"""One-edge overfit gate (task book §32): 12 edges x 40 updates."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.native_atom14.checkpoint import load_base_model, make_policy_reference, trainable_score_params  # noqa: E402
from vhh_rl.native_atom14.global_cf_step import signed_local_dpo_step  # noqa: E402
from vhh_rl.native_atom14.dpo_trainer import load_case_conditioning, move_conditioning  # noqa: E402
from vhh_rl.signed_local.edge_validator import load_edges  # noqa: E402
from vhh_rl.signed_local.local_mask import target_residue_mask  # noqa: E402
from vhh_rl.signed_local.trainer import _load_design_positions  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
OUT = ROOT / "runs/signed_local/one_edge"
STEPS = (0, 10, 20, 40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path,
                        default=ROOT / "runs/signed_local/edges/train_edges.jsonl")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)
    design = _load_design_positions(ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    edges = load_edges(args.edges)
    neg = [e for e in edges if e.event_class == "both_negative"][:4]
    flip_w = [e for e in edges if e.event_class == "sign_flip" and e.context == "winner_drop"][:4]
    flip_l = [e for e in edges if e.event_class == "sign_flip" and e.context == "loser_gain"][:4]
    chosen = neg + flip_w + flip_l
    base = load_base_model(BASE, device="cuda")
    policy, reference = make_policy_reference(base, device="cuda")
    pristine = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    results = []
    for e in chosen:
        cond = move_conditioning(load_case_conditioning(
            e.case_id, ROOT / "runs/native_pool/conditioning",
            ROOT / "runs/cf_opsd/rollouts/train"), device="cuda")
        feats = cond["feats"]
        kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
                  "feats": feats, "multiplicity": 1,
                  "diffusion_conditioning": cond["diffusion_conditioning"]}
        anchor = torch.load(e.anchor_coords_path, map_location="cuda", weights_only=True).float()
        cf = torch.load(e.cf_coords_path, map_location="cuda", weights_only=True).float()
        preferred, rejected = (cf, anchor) if e.preferred_side == "cf" else (anchor, cf)
        mask = target_residue_mask(feats, e.position, design[e.case_id]).to("cuda")
        policy.load_state_dict(pristine)
        params = trainable_score_params(policy)
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        zs = {}
        for step in range(0, args.updates + 1):
            out = signed_local_dpo_step(policy.structure_module, reference.structure_module,
                                        feats, preferred, rejected, mask, kwargs, beta=10.0)
            if step in STEPS:
                zs[step] = float(out.dpo.z.mean())
            if step == args.updates:
                break
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        results.append({"edge_id": e.edge_id, "event_class": e.event_class,
                        "context": e.context, "z": zs})
        print(f"[one-edge] {e.edge_id} z={ {k: round(v,5) for k,v in zs.items()} }", flush=True)
    finals = [r["z"][args.updates] for r in results]
    inits = [r["z"].get(0, 0.0) for r in results]
    dz = sorted(f - i for f, i in zip(finals, inits))
    report = {
        "n_edges": len(results),
        "n_positive_final": sum(1 for z in finals if z > 0),
        "median_dz": dz[len(dz) // 2] if dz else None,
        "nonfinite": sum(1 for z in finals if z != z or abs(z) == float("inf")),
        "results": results,
    }
    report["gate_pass"] = (report["n_positive_final"] >= 10 and (report["median_dz"] or 0) > 0.001
                           and report["nonfinite"] == 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "one_edge_overfit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("n_edges", "n_positive_final", "median_dz",
                                             "nonfinite", "gate_pass")}, indent=1))


if __name__ == "__main__":
    main()
