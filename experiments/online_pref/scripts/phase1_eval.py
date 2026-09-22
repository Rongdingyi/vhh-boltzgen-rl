#!/usr/bin/env python
"""Phase 1 evaluation: base + 3 seeds x 3 temporal arms (§25/§26)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _common as C  # noqa: E402

REGIONS = ("full", "suffix", "prefix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=C.TRAIN_SEEDS)
    parser.add_argument("--regions", nargs="*", default=list(REGIONS))
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()
    from vhh_rl.cf_opsd.evaluator import evaluate

    out_path = C.RUN_ROOT / "phase1" / "eval" / "phase1_eval.json"
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    if not args.skip_base and "base" not in results:
        run_root = C.RUN_ROOT / "phase1" / "eval" / "base"
        shutil.rmtree(run_root, ignore_errors=True)
        payload = evaluate(C.heldout8(), None, "online_pref_p1_base",
                           num_samples=8, seed_offset=C.EVAL_SEED_OFFSET,
                           run_root=run_root)
        results["base"] = _summarize(payload)
    for seed in args.seeds:
        for region in args.regions:
            key = f"{seed}:{region}"
            if key in results:
                continue
            ckpt = C.RUN_ROOT / "phase1" / f"seed_{seed}" / f"T-{region}" / "student_r4.pt"
            if not Path(ckpt).is_file():
                print(f"[skip] {key}: missing {ckpt}")
                continue
            run_root = C.RUN_ROOT / "phase1" / "eval" / f"seed{seed}_{region}"
            shutil.rmtree(run_root, ignore_errors=True)
            payload = evaluate(C.heldout8(), ckpt, f"online_pref_p1_{region}_{seed}",
                               num_samples=8, seed_offset=C.EVAL_SEED_OFFSET,
                               run_root=run_root)
            results[key] = _summarize(payload)
            print(f"[phase1-eval] {key}: reward={payload['reward_mean']:+.3f}",
                  flush=True)
    C.write_json(out_path, results)


def _summarize(payload: dict) -> dict:
    cases = payload["cases"]
    return {"reward_mean": payload["reward_mean"],
            "per_case": {k: v["reward_mean"] for k, v in cases.items()},
            "invalid": sum(v["n_invalid"] for v in cases.values()),
            "fr": sum(v["n_fr_mismatch"] for v in cases.values()),
            "n": sum(v["n"] for v in cases.values())}


if __name__ == "__main__":
    main()
