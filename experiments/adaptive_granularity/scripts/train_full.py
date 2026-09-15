#!/usr/bin/env python
"""Phase D full training: 24 train cases x 500 updates (task book §57-§59)."""
from __future__ import annotations
import argparse
import json

import _common as C  # noqa: E402

ARMS = {
    "f0": dict(kind="current_cf", variant="cf"),
    "f1": dict(kind="ag", variant=None),          # best simple, pass --variant
    "f2": dict(kind="ag", variant="adaptive"),
    "f3": dict(kind="ag", variant="adaptive_shuffle"),
    "f4": dict(kind="eligible_cf", variant="cf"),
}
STEPS = 500
SEED = 20260913


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--variant", default=None,
                        help="for f1: no_floor|strict_consensus|strict_region")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = ARMS[args.arm]
    if spec["kind"] == "current_cf":
        pairs_path, weights_path, variant = C.PAIRS, C.CURRENT_WEIGHTS, "cf"
    elif spec["kind"] == "eligible_cf":
        pairs_path = C.WEIGHTS_DIR / "pairs_adaptive_eligible_cf.jsonl"
        weights_path, variant = C.CURRENT_WEIGHTS, "cf"
    else:
        variant = args.variant or spec["variant"]
        if variant is None:
            raise SystemExit("f1 requires --variant no_floor|strict_consensus|strict_region")
        pairs_path = C.WEIGHTS_DIR / f"pairs_{variant}.jsonl"
        weights_path = C.WEIGHTS_DIR / "ag_weights.json"
    if not pairs_path.is_file():
        raise SystemExit(f"missing {pairs_path}")

    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    out_dir = (C.FULL_DIR / args.arm if args.seed == SEED
               else C.FULL_DIR / f"{args.arm}_seed{args.seed}")
    summary = run_weighted_dpo(
        base_checkpoint=C.BASE_CKPT,
        pairs_path=pairs_path,
        pool_root=C.POOL,
        conditioning_dir=C.POOL / "conditioning",
        weights_path=weights_path,
        output_dir=out_dir,
        manifest_path=C.MANIFEST,
        variant=variant,
        beta=10.0,
        lr=1e-5,
        max_steps=args.steps,
        checkpoint_every=100,
        seed=args.seed,
        log_tag=f"ag-full-{args.arm}",
    )
    (out_dir / "full_meta.json").write_text(json.dumps(
        {"arm": args.arm, "variant": variant, "pairs_path": str(pairs_path),
         "weights_path": str(weights_path), "steps": args.steps,
         "seed": args.seed}, indent=1))
    print(json.dumps({"arm": args.arm, "variant": variant, "summary": summary},
                     indent=1))


if __name__ == "__main__":
    main()
