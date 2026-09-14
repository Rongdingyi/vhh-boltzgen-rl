#!/usr/bin/env python
"""Phase D: static CF-OPSD target set (8 targets) + 50/100-update training."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.credit import pair_credit  # noqa: E402
from vhh_rl.cf_opsd.rollout import (  # noqa: E402
    load_cond_kwargs, load_contexts, load_trajectories,
)
from vhh_rl.cf_opsd.static_trainer import run_static  # noqa: E402
from vhh_rl.cf_opsd.target_builder import build_target  # noqa: E402
from vhh_rl.cf_opsd.target_metrics import decode_and_score  # noqa: E402
from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.native_atom14.decode import decode_atom14, sequence_from_feat  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
ROLL = ROOT / "runs/cf_opsd/rollouts"
OUT = ROOT / "runs/cf_opsd/static_v0"
COND = ROOT / "runs/native_pool/conditioning"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]),
            split=row.get("split", "train"), seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def decode_seq(coords, feats):
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = coords.clone()
    out = decode_atom14(feat)
    seq, _tok, invalid = sequence_from_feat(out)
    return seq, invalid


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    q_star = float(json.loads((ROOT / "runs/cf_opsd/query_probe.json").read_text())["q_star_progress"])

    gate_b = json.loads((ROOT / "runs/cf_opsd/target_probe/gate_b.json").read_text())
    if gate_b.get("selected_radius") is None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "STOPPED.md").write_text(
            "# CF-OPSD Static Pilot（Phase D）\n\nGate B FAIL（无通过半径）→ "
            "按任务书 §46（Gate C 未过不得进入 Phase D）停止。\n")
        print("Gate B FAIL -> Phase D not run (task book §46)")
        return
    radius = float(gate_b["selected_radius"])
    variant = "target_mask_only"
    cases = load_cases()
    scorer = make_reward_adapter(cache_path=ROOT / "runs/cf_opsd/reward_cache.sqlite")

    targets = []
    cond_kwargs_by_case = {}
    for cid in cfg["train_cases"]:
        case = cases[cid]
        seed = case.seed_base + 800000
        traj_path = ROLL / "train" / cid / f"seed{seed}.pt"
        all_trajs = sorted(load_trajectories(traj_path), key=lambda t: t.sample_index)
        contexts = load_contexts(traj_path)
        cond_kwargs_by_case[cid] = load_cond_kwargs(traj_path)
        mult = int(contexts[0]["multiplicity"]) if contexts else 1
        trajs = [t for t in all_trajs
                 if t.endpoint_reward is not None and not t.contains_invalid and t.fr_mismatch == 0]
        ranked = sorted(trajs, key=lambda t: t.endpoint_reward, reverse=True)
        n_steps = len(ranked[0].sigmas)
        step = max(0, min(n_steps - 1, int(round(q_star * n_steps)) - 1))
        pairs = [(ranked[0], ranked[-1]), (ranked[1], ranked[-2])]
        for pi, (winner, loser) in enumerate(pairs):
            credit = pair_credit(scorer, case, f"{cid}_w{pi}", f"{cid}_l{pi}",
                                 winner.endpoint_sequence, loser.endpoint_sequence,
                                 case.design_positions, case.fr_positions)
            anchor = loser.anchors[step]
            bt = build_target(anchor, winner.endpoint_coords, loser.feats_common,
                              sorted(credit["credits"]), credit["credits"], radius)
            if not bt.touched.any():
                print(f"[static] {cid} pair {pi}: empty target, skipped")
                continue
            ts, invalid = decode_seq(bt.target_coords, loser.feats_common)
            anchor_seq, _ = decode_seq(anchor, loser.feats_common)
            anchor_reward = float(scorer.score_sequences(
                [anchor_seq], spec=case_spec_for(case)).raw_scores[0]) if anchor_seq else None
            target_reward = None
            if not invalid:
                target_reward = float(scorer.score_sequences(
                    [ts], spec=case_spec_for(case)).raw_scores[0])
            targets.append({
                "case_id": cid, "pair_index": pi,
                "query_index": step, "query_progress": q_star,
                "sigma": loser.sigmas[step],
                "full_query_coords": contexts[step]["query"],
                "full_anchor_coords": contexts[step]["anchors"],
                "design_index": int(loser.meta.get("batch_index", 0)), "multiplicity": mult,
                "query_coords": loser.query_states[step],
                "anchor_coords": anchor,
                "winner_coords": winner.endpoint_coords,
                "loser_coords": loser.endpoint_coords,
                "winner_sequence": winner.endpoint_sequence,
                "loser_sequence": loser.endpoint_sequence,
                "anchor_sequence": anchor_seq,
                "anchor_reward": anchor_reward,
                "target_coords": bt.target_coords,
                "target_sequence": ts,
                "target_reward": target_reward,
                "credits": credit["credits"],
                "radius": radius,
                "feats_common": loser.feats_common,
            })
            print(f"[static] {cid} pair {pi}: target_gain="
                  f"{None if target_reward is None else target_reward - anchor_reward:+.3f}")
    scorer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(cond_kwargs_by_case, OUT / "cond_kwargs.pt")
    torch.save(targets, OUT / "targets_static.pt")
    summary = {"n_targets": len(targets), "radius": radius, "q_star": q_star,
               "variant": variant}
    (OUT / "targets_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    for updates in (50, 100):
        run_dir = OUT / f"u{updates:03d}"
        result = run_static("cf_opsd_static", OUT / "targets_static.pt", COND, BASE,
                            run_dir, updates=updates, variant=variant,
                            log_tag=f"cf_opsd_static_u{updates}")
        print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
