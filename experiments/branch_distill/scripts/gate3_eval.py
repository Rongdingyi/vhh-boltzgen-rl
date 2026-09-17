#!/usr/bin/env python
"""Gate 3 evaluation on the 8 held-out cases (task book §50)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import _common as C  # noqa: E402

ARMS = {"base": None, "a": C.GATE3_DIR / "a" / "student_r4.pt",
        "b": C.GATE3_DIR / "b" / "student_r4.pt",
        "c": C.GATE3_DIR / "c" / "student_r4.pt",
        "d": C.GATE3_DIR / "d" / "student_r4.pt"}
NUM_SAMPLES = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    args = parser.parse_args()
    from vhh_rl.cf_opsd.evaluator import evaluate

    out = C.GATE3_DIR / "eval"
    summary = {}
    for arm in args.arms:
        ckpt = ARMS.get(arm)
        if arm != "base" and (ckpt is None or not Path(ckpt).is_file()):
            print(f"[skip] arm {arm}: missing checkpoint")
            continue
        run_root = out / arm
        shutil.rmtree(run_root, ignore_errors=True)
        payload = evaluate(C.HELDOUT_CASES, ckpt, f"branch_{arm}",
                           num_samples=NUM_SAMPLES,
                           seed_offset=C.EVAL_SEED_OFFSET, run_root=run_root)
        n = sum(v["n"] for v in payload["cases"].values())
        summary[arm] = {
            "reward_mean": payload["reward_mean"],
            "per_case": {k: v["reward_mean"] for k, v in payload["cases"].items()},
            "invalid": sum(v["n_invalid"] for v in payload["cases"].values()),
            "fr": sum(v["n_fr_mismatch"] for v in payload["cases"].values()),
            "unique": sum(v.get("n_unique", 0) for v in payload["cases"].values()),
            "n": n,
        }
        print(f"[gate3-eval] {arm}: reward={payload['reward_mean']:+.3f} "
              f"invalid={summary[arm]['invalid']}", flush=True)
    C.write_json(out / "gate3_eval.json", summary)


if __name__ == "__main__":
    main()
