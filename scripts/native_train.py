#!/usr/bin/env python
"""Train native atom14 DPO (N2/N3/N4) or RWR (N1).

Examples:
  native_train_dpo.py --method n2 --beta 1.0 --max-steps 50   # smoke
  native_train_dpo.py --method n4 --beta 1.0 --anchor-lambda 0.1 --max-steps 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
POOL = ROOT / "runs/native_pool"

METHOD_DEFAULTS = {
    "n2": dict(preference_mask="all_design_atoms", anchor_lambda=0.0),
    "n3": dict(preference_mask="cdr_fake_atoms", anchor_lambda=0.0),
    "n4": dict(preference_mask="cdr_fake_atoms", anchor_lambda=0.1),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["n1", "n2", "n3", "n4"])
    parser.add_argument("--summarize-only", action="store_true",
                        help="print the exact command that would run")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--anchor-lambda", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--pool-root", type=Path, default=POOL)
    parser.add_argument("--conditioning-dir", type=Path, default=POOL / "conditioning")
    parser.add_argument("--pairs", type=Path, default=POOL / "pairs_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or (ROOT / "runs" / f"native_{args.method}")
    if args.method == "n1":
        from vhh_rl.native_atom14.rwr_trainer import run_rwr

        summary = run_rwr(
            base_checkpoint=BASE_CKPT,
            pool_root=args.pool_root,
            split="train",
            conditioning_dir=args.conditioning_dir,
            output_dir=output_dir,
            lr=args.lr,
            max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
        )
    else:
        from vhh_rl.native_atom14.dpo_trainer import run_dpo

        defaults = METHOD_DEFAULTS[args.method]
        anchor_lambda = (
            defaults["anchor_lambda"] if args.anchor_lambda is None else args.anchor_lambda
        )
        summary = run_dpo(
            base_checkpoint=BASE_CKPT,
            pairs_path=args.pairs,
            pool_root=args.pool_root,
            conditioning_dir=args.conditioning_dir,
            output_dir=output_dir,
            preference_mask=defaults["preference_mask"],
            beta=args.beta,
            anchor_lambda=anchor_lambda,
            lr=args.lr,
            max_steps=args.max_steps,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            log_tag=args.method,
        )
    import json
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
