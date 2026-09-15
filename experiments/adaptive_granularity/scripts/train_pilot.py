#!/usr/bin/env python
"""Phase B/C pilot training: variants -> run_weighted_dpo (task book §29-§31)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import _common as C  # noqa: E402

ARMS = {
    "current": dict(kind="current_cf", variant="cf"),
    "nofloor": dict(kind="ag", variant="no_floor"),
    "strict": dict(kind="ag", variant="strict_consensus"),
    "region": dict(kind="ag", variant="strict_region"),
    "adaptive": dict(kind="ag", variant="adaptive"),
    "shuffle": dict(kind="ag", variant="adaptive_shuffle"),
    "eligible-cf": dict(kind="eligible_cf", variant="cf"),
}
STEPS = 100
SEED = 20260913


def _materialize_current_cf(pairs: list[dict]) -> tuple[Path, Path]:
    """Filtered current-CF pilot support (same protocol as cf_dpo_mini, §30)."""
    out_dir = C.PILOT_DIR / "current"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(C.PILOT_TRAIN_CASES)
    pilot_pairs = [p for p in pairs if p["case_id"] in keep]
    weights_all = C.load_current_weights()
    weights = {"eta": weights_all.get("eta", 0.75), "seed": weights_all.get("seed"),
               "pairs": {k: v for k, v in weights_all["pairs"].items()
                         if v["case_id"] in keep}}
    pairs_path = out_dir / "pairs_pilot4.jsonl"
    weights_path = out_dir / "weights_pilot4.json"
    pairs_path.write_text("".join(json.dumps(p) + "\n" for p in pilot_pairs))
    weights_path.write_text(json.dumps(weights, indent=1))
    return pairs_path, weights_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = ARMS[args.arm]
    pairs = C.load_pairs()

    if spec["kind"] == "current_cf":
        pairs_path, weights_path = _materialize_current_cf(pairs)
        variant = "cf"
    else:
        pairs_path = C.WEIGHTS_DIR / f"pairs_{spec['variant']}_pilot4.jsonl"
        weights_path = C.WEIGHTS_DIR / "ag_weights.json"
        variant = spec["variant"]
        if not pairs_path.is_file():
            raise SystemExit(f"missing {pairs_path}; run ag-build-weights + "
                             f"ag-validate-weights first")
    pilot_pairs = [json.loads(l) for l in pairs_path.open()]
    counts = {case: sum(1 for p in pilot_pairs if p["case_id"] == case)
              for case in C.PILOT_TRAIN_CASES}
    missing = [case for case, n in counts.items() if n == 0]
    if missing:
        raise SystemExit(f"arm={args.arm}: pilot cases with 0 pairs: {missing} "
                         f"(no silent case drop)")

    from vhh_rl.native_atom14.weighted_dpo import run_weighted_dpo

    out_dir = C.PILOT_DIR / args.arm
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
        checkpoint_every=50,
        seed=args.seed,
        log_tag=f"ag-{args.arm}",
    )
    (out_dir / "pilot_meta.json").write_text(json.dumps(
        {"arm": args.arm, "variant": variant, "pairs_path": str(pairs_path),
         "weights_path": str(weights_path), "steps": args.steps, "seed": args.seed,
         "pilot_case_counts": counts}, indent=1))
    print(json.dumps({"arm": args.arm, "variant": variant,
                      "pilot_case_counts": counts,
                      "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
