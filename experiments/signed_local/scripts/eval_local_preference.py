#!/usr/bin/env python
"""Held-out local preference accuracy (task book §50-§53)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.signed_local.evaluator import local_preference_accuracy  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")
OUT = ROOT / "runs/signed_local/pilot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="*", default=["cf", "local", "neg", "main", "shuffle"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[12345],
                        help="sigma-draw seeds; per-arm mean/std over seeds is reported")
    args = parser.parse_args()
    edges = ROOT / "runs/signed_local/edges/heldout_edges.jsonl"
    if not edges.is_file():
        raise SystemExit(f"missing {edges}; run build_heldout_edges.py first")
    steps = {"cf": 100, "local": 100, "neg": 100, "main": 100, "shuffle": 100,
             "addon": 125, "cf125": 125}
    runs: dict[str, list[dict]] = {}
    for seed in args.seeds:
        runs.setdefault("base", []).append(local_preference_accuracy(
            BASE, None, edges, seed=seed,
            cond_dir=ROOT / "runs/native_pool/conditioning"))
        for arm in args.arms:
            ckpt = OUT / arm / f"checkpoint_{steps[arm]:04d}.pt"
            if not ckpt.is_file():
                print(f"[skip] {arm}: missing {ckpt}")
                continue
            runs.setdefault(arm, []).append(local_preference_accuracy(
                BASE, ckpt, edges, seed=seed,
                cond_dir=ROOT / "runs/native_pool/conditioning"))
    results = {}
    for arm, rs in runs.items():
        alls = [r["accuracy"]["all"] for r in rs]
        zs = [r["median_z"] for r in rs]
        mean = sum(alls) / len(alls)
        var = sum((x - mean) ** 2 for x in alls) / len(alls)
        results[arm] = {
            "seeds": list(args.seeds),
            "accuracy_all_mean": mean,
            "accuracy_all_std": var ** 0.5,
            "accuracy_all_per_seed": alls,
            "median_z_mean": sum(zs) / len(zs),
            "accuracy": rs[0]["accuracy"],
            "per_seed": [{"seed": s, "accuracy": r["accuracy"], "median_z": r["median_z"]}
                         for s, r in zip(args.seeds, rs)],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_preference_accuracy.json").write_text(json.dumps(results, indent=1))
    for arm, r in results.items():
        print(f"{arm}: acc={r['accuracy_all_mean']:.3f} ± {r['accuracy_all_std']:.3f} "
              f"(seeds {r['accuracy_all_per_seed']})")


if __name__ == "__main__":
    main()
