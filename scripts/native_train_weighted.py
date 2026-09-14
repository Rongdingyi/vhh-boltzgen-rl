#!/usr/bin/env python
"""Phase C/D: CF-weighted and/or temporal-weighted native DPO.

Examples:
  native_train_weighted.py --variant cf  --max-steps 100 --max-cases 4
  native_train_weighted.py --variant shuffle --max-steps 500
  native_train_weighted.py --variant cf --temporal smooth --sigma-seq 1.7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
POOL = ROOT / "runs/native_pool"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["cf", "random_sparse", "shuffle", "region", "uniform_all",
                                 "diff_only", "drop_only", "gain_only",
                                 "0.25", "0.50", "0.75", "1.00"])
    parser.add_argument("--temporal", default="none", choices=["none", "hard", "smooth"])
    parser.add_argument("--sigma-seq", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "runs/next_stage/weights/residue_weights.json")
    parser.add_argument("--pairs", type=Path, default=POOL / "pairs_train.jsonl")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl")
    parser.add_argument("--pool-root", type=Path, default=POOL)
    parser.add_argument("--conditioning-dir", type=Path, default=POOL / "conditioning")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    tag = args.tag or f"cf-dpo-{args.variant}" + (
        f"-{args.temporal}" if args.temporal != "none" else ""
    )
    summary = run_weighted_dpo(
        base_checkpoint=BASE_CKPT,
        pairs_path=args.pairs,
        pool_root=args.pool_root,
        conditioning_dir=args.conditioning_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        variant=args.variant,
        temporal=args.temporal,
        sigma_seq=args.sigma_seq,
        tau=args.tau,
        beta=args.beta,
        lr=args.lr,
        max_steps=args.max_steps,
        checkpoint_every=args.checkpoint_every,
        smoke_steps=args.smoke_steps,
        max_cases=args.max_cases,
        seed=args.seed,
        log_tag=tag,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
