#!/usr/bin/env python
"""Borderline 3-seed micro-confirmation (task book §49/§50, review item 8).

Runs only after gate_c.json says BORDERLINE_3SEED: evaluates the 100-step
pilot checkpoints of Current CF and Adaptive for seeds 42/43/44 on the 8
held-out cases and writes gate_c_confirm.json with the pre-registered rule
mean(Adaptive-CF) >= +0.30 and >= 2/3 seeds Adaptive >= CF.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import _common as C  # noqa: E402

SEEDS = [42, 43, 44]
ARMS = ["current", "adaptive"]
GATE_C = C.PILOT_DIR / "gate_c.json"
GATE_C_CONFIRM = C.PILOT_DIR / "gate_c_confirm.json"
STEP = 100
SEED_OFFSET = 800000
MEAN_THRESHOLD = 0.30
MIN_SEEDS = 2


def confirm_verdict(per_seed_delta: dict[int, float]) -> dict:
    """Pre-registered §50 rule; pure function so it can be unit-tested."""
    deltas = [per_seed_delta[s] for s in sorted(per_seed_delta)]
    if not deltas:
        return {"verdict": "NOT_RUN", "mean_delta": None,
                "seeds_adaptive_ge_cf": 0, "n_seeds": 0}
    mean_delta = sum(deltas) / len(deltas)
    non_negative = sum(1 for d in deltas if d >= -1e-9)
    passed = mean_delta >= MEAN_THRESHOLD and non_negative >= MIN_SEEDS
    return {
        "verdict": "GO" if passed else "NO_GO",
        "criterion": f"mean(Adaptive-CF) >= +{MEAN_THRESHOLD} and "
                     f">= {MIN_SEEDS}/{len(deltas)} seeds non-negative",
        "mean_delta": mean_delta,
        "delta_per_seed": {str(s): per_seed_delta[s] for s in sorted(per_seed_delta)},
        "seeds_adaptive_ge_cf": non_negative,
        "n_seeds": len(deltas),
    }


def require_borderline(override: str | None) -> None:
    payload = json.loads(GATE_C.read_text()) if GATE_C.is_file() else None
    verdict = (payload or {}).get("verdict")
    if verdict == "BORDERLINE_3SEED":
        return
    if verdict == "STRONG_GO":
        raise SystemExit("gate C is STRONG_GO; 3-seed confirmation is unnecessary")
    if override:
        print(f"[warn] Gate C confirmation override: {override}", flush=True)
        return
    raise SystemExit(f"gate C verdict is {verdict!r} ({GATE_C}); "
                     "3-seed confirmation requires BORDERLINE_3SEED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--arms", nargs="*", default=ARMS)
    parser.add_argument("--override-gate", default=None)
    args = parser.parse_args()
    require_borderline(args.override_gate)

    from vhh_rl.cf_opsd.evaluator import evaluate

    results: dict[str, dict] = {}
    per_seed_delta: dict[int, float] = {}
    for seed in args.seeds:
        arm_scores: dict[str, float | None] = {}
        for arm in args.arms:
            ckpt = C.PILOT_DIR / f"seed{seed}" / arm / f"checkpoint_{STEP:04d}.pt"
            if not ckpt.is_file():
                raise SystemExit(f"missing {ckpt}; run the 3-seed training first")
            run_root = C.PILOT_DIR / "eval_3seed" / f"seed{seed}_{arm}"
            shutil.rmtree(run_root, ignore_errors=True)
            summary = evaluate(C.heldout8(), ckpt, f"ag_3seed_{arm}_{seed}",
                               num_samples=8, seed_offset=SEED_OFFSET,
                               run_root=run_root)
            invalid = sum(v["n_invalid"] for v in summary["cases"].values())
            fr = sum(v["n_fr_mismatch"] for v in summary["cases"].values())
            n = sum(v["n"] for v in summary["cases"].values())
            arm_scores[arm] = summary["reward_mean"]
            results[f"seed{seed}_{arm}"] = {
                "reward_mean": summary["reward_mean"], "invalid": invalid,
                "fr": fr, "n": n,
                "per_case": {c: v["reward_mean"] for c, v in summary["cases"].items()},
            }
        if "current" in arm_scores and "adaptive" in arm_scores:
            per_seed_delta[seed] = arm_scores["adaptive"] - arm_scores["current"]
    confirm = confirm_verdict(per_seed_delta)
    confirm["results"] = results
    GATE_C_CONFIRM.write_text(json.dumps(confirm, indent=1))
    print(json.dumps({k: v for k, v in confirm.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    main()
