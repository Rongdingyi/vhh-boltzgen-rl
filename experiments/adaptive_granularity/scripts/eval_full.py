#!/usr/bin/env python
"""Phase D held-out evaluation + checkpoint selection (task book §60-§62)."""
from __future__ import annotations
import argparse
import json
import shutil

import _common as C  # noqa: E402

STEPS = [100, 200, 300, 400, 500]
SEED_OFFSET = 800000


def _eval(tag: str, ckpt, run_tag: str) -> dict:
    from vhh_rl.cf_opsd.evaluator import evaluate
    run_root = C.FULL_DIR / "eval" / run_tag
    shutil.rmtree(run_root, ignore_errors=True)   # stale-dir guard (review 7)
    return evaluate(C.heldout8(), ckpt, tag, num_samples=8,
                    seed_offset=SEED_OFFSET, run_root=run_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=["f0", "f1", "f2", "f3", "f4"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[20260913])
    args = parser.parse_args()
    base = _eval("ag_full_base", None, "base")
    base_invalid = sum(v["n_invalid"] for v in base["cases"].values())
    base_n = sum(v["n"] for v in base["cases"].values())
    report: dict = {"base": {"reward_mean": base["reward_mean"],
                             "invalid": base_invalid, "n": base_n}}
    for seed in args.seeds:
        for arm in args.arms:
            arm_dir = C.FULL_DIR / arm if seed == 20260913 else C.FULL_DIR / f"{arm}_seed{seed}"
            if not arm_dir.is_dir():
                continue
            candidates = {}
            for step in STEPS:
                ckpt = arm_dir / f"checkpoint_{step:04d}.pt"
                if not ckpt.is_file():
                    continue
                summary = _eval(f"ag_full_{arm}_{seed}_{step}", ckpt,
                                f"{arm}_seed{seed}_{step}")
                invalid = sum(v["n_invalid"] for v in summary["cases"].values())
                fr = sum(v["n_fr_mismatch"] for v in summary["cases"].values())
                n = sum(v["n"] for v in summary["cases"].values())
                candidates[step] = {
                    "reward_mean": summary["reward_mean"], "invalid": invalid,
                    "fr": fr, "n": n,
                    "valid": (fr == 0 and invalid <= base_invalid + 0.01 * base_n),
                    "per_case": {c: v["reward_mean"] for c, v in summary["cases"].items()},
                }
            valid = {s: v for s, v in candidates.items() if v["valid"]}
            if not valid:
                report[f"{arm}_seed{seed}"] = {"selected": None,
                                               "candidates": candidates}
                continue
            best = max(valid, key=lambda s: valid[s]["reward_mean"])
            report[f"{arm}_seed{seed}"] = {
                "selected_step": best,
                "selected": valid[best],
                "candidates": candidates,
                "selection_rule": "held-out reward highest, validity enforced",
            }
    out = C.FULL_DIR / "eval/full_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v.get("selected", {}) or {}).get("reward_mean")
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
