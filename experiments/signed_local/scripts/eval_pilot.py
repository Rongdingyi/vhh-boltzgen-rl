#!/usr/bin/env python
"""Held-out 4-case evaluation for SL-CF-DPO pilot arms (task book §47)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.evaluator import evaluate_reward  # noqa: E402

HELDOUT = ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"]
ARMS = {"cf": 100, "local": 100, "neg": 100, "main": 100, "shuffle": 100,
        "addon": 125, "cf125": 125}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--with-base", action="store_true")
    args = parser.parse_args()
    out_dir = ROOT / "runs/signed_local/pilot"
    results = {}
    if args.with_base:
        results["base"] = evaluate_reward(None, HELDOUT, "slcf_base",
                                          run_root=out_dir / "eval_base")
    for arm in (args.arms or list(ARMS)):
        ckpt = out_dir / arm / f"checkpoint_{ARMS[arm]:04d}.pt"
        if not ckpt.is_file():
            print(f"[skip] {arm}: missing {ckpt}")
            continue
        results[arm] = evaluate_reward(ckpt, HELDOUT, f"slcf_{arm}",
                                       run_root=out_dir / f"eval_{arm}")
    merged_path = out_dir / "pilot_eval.json"
    merged = json.loads(merged_path.read_text()) if merged_path.is_file() else {}
    merged.update(results)
    merged_path.write_text(json.dumps(merged, indent=1))
    results = merged
    print(json.dumps({k: v.get("reward_mean") for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
