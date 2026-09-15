#!/usr/bin/env python
"""SL-CF-DPO full 24-train/8-heldout training (task book §57-§59)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.trainer import run_signed_local  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
ARMS = {
    "f0": dict(mode="global_only", shuffle=False, classes=("both_negative", "sign_flip")),
    "f1": dict(mode="mixed", shuffle=False, classes=("both_negative", "sign_flip")),
    "f2": dict(mode="mixed", shuffle=True, classes=("both_negative", "sign_flip")),
    "f3": dict(mode="mixed", shuffle=False, classes=("both_negative",)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    spec = ARMS[args.arm]
    summary = run_signed_local(
        BASE, ROOT / "runs/signed_local/edges/train_edges.jsonl",
        ROOT / "runs/native_pool/pairs_train.jsonl",
        ROOT / "runs/next_stage/weights/residue_weights.json",
        ROOT / "runs/signed_local/full" / args.arm,
        mode=spec["mode"], local_classes=spec["classes"],
        shuffle_direction=spec["shuffle"], updates=args.updates,
        schedule=(3, 1), seed=args.seed, checkpoint_every=100,
        conditioning_dir=ROOT / "runs/native_pool/conditioning",
        log_tag=f"slcf-{args.arm}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
