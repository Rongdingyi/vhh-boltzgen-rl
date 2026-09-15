#!/usr/bin/env python
"""Run one SL-CF-DPO pilot arm (task book §43-§46, §84)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.edge_validator import load_edges  # noqa: E402
from vhh_rl.signed_local.trainer import run_signed_local  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
PILOT_CASES = ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"]
ARMS = {
    "cf":     dict(mode="global_only", updates=100, schedule=(3, 1), classes=("both_negative", "sign_flip"), shuffle=False),
    "local":  dict(mode="local_only",  updates=100, schedule=(3, 1), classes=("both_negative", "sign_flip"), shuffle=False),
    "neg":    dict(mode="mixed",       updates=100, schedule=(3, 1), classes=("both_negative",), shuffle=False),
    "main":   dict(mode="mixed",       updates=100, schedule=(3, 1), classes=("both_negative", "sign_flip"), shuffle=False),
    "shuffle": dict(mode="mixed",      updates=100, schedule=(3, 1), classes=("both_negative", "sign_flip"), shuffle=True),
    # conditional arms (§46): A6 = 100 global CF + 25 local add-on (125 updates),
    # A7 = compute-matched 125 global CF updates.
    "addon":  dict(mode="mixed",       updates=125, schedule=(4, 1), classes=("both_negative", "sign_flip"), shuffle=False),
    "cf125":  dict(mode="global_only", updates=125, schedule=(3, 1), classes=("both_negative", "sign_flip"), shuffle=False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--edges", type=Path,
                        default=ROOT / "runs/signed_local/edges/train_edges.jsonl")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    spec = ARMS[args.arm]
    out_dir = ROOT / "runs/signed_local/pilot" / args.arm
    # pilot subset: keep only the 4 pilot train cases
    edges = [e for e in load_edges(args.edges) if e.case_id in PILOT_CASES]
    sub = ROOT / "runs/signed_local/pilot" / f"pilot_edges_{args.arm}.jsonl"
    sub.parent.mkdir(parents=True, exist_ok=True)
    with sub.open("w") as fh:
        for e in edges:
            fh.write(json.dumps(e.__dict__) + "\n")
    summary = run_signed_local(
        BASE, sub, ROOT / "runs/native_pool/pairs_train.jsonl",
        ROOT / "runs/next_stage/weights/residue_weights.json", out_dir,
        mode=spec["mode"], local_classes=spec["classes"],
        shuffle_direction=spec["shuffle"], updates=spec["updates"],
        schedule=spec["schedule"], seed=args.seed,
        conditioning_dir=ROOT / "runs/native_pool/conditioning",
        log_tag=f"slcf-{args.arm}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
