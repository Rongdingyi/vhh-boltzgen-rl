#!/usr/bin/env python
"""Phase 0 evaluation: 8 held-out cases x 8 samples per arm/seed (§13)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import _common as C  # noqa: E402

ARMS = {"A": "A", "B": "B", "V": "V"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=C.TRAIN_SEEDS)
    parser.add_argument("--arms", nargs="*", default=["A", "B", "V"])
    args = parser.parse_args()
    from vhh_rl.cf_opsd.evaluator import evaluate

    results = {}
    base_root = C.EVAL / "base"
    shutil.rmtree(base_root, ignore_errors=True)
    base = evaluate(C.heldout8(), None, "online_pref_base",
                    num_samples=8, seed_offset=C.EVAL_SEED_OFFSET,
                    run_root=base_root)
    results["base"] = _summarize(base)
    for seed in args.seeds:
        for arm in args.arms:
            ckpt = C.seed_dir(seed) / arm / "student_r4.pt"
            if not ckpt.is_file():
                print(f"[skip] seed {seed} arm {arm}: missing {ckpt}")
                continue
            run_root = C.EVAL / f"seed{seed}_{arm}"
            shutil.rmtree(run_root, ignore_errors=True)
            payload = evaluate(C.heldout8(), ckpt, f"online_pref_{arm}_{seed}",
                               num_samples=8, seed_offset=C.EVAL_SEED_OFFSET,
                               run_root=run_root)
            results[f"{seed}:{arm}"] = _summarize(payload)
            print(f"[phase0-eval] seed {seed} arm {arm}: "
                  f"reward={payload['reward_mean']:+.3f}", flush=True)
    C.write_json(C.EVAL / "phase0_eval.json", results)


def _summarize(payload: dict) -> dict:
    cases = payload["cases"]
    return {
        "reward_mean": payload["reward_mean"],
        "per_case": {k: v["reward_mean"] for k, v in cases.items()},
        "invalid": sum(v["n_invalid"] for v in cases.values()),
        "fr": sum(v["n_fr_mismatch"] for v in cases.values()),
        "n": sum(v["n"] for v in cases.values()),
    }


if __name__ == "__main__":
    main()
