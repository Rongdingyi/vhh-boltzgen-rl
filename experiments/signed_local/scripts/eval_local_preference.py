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
    args = parser.parse_args()
    edges = ROOT / "runs/signed_local/edges/heldout_edges.jsonl"
    if not edges.is_file():
        raise SystemExit(f"missing {edges}; run build_heldout_edges.py first")
    results = {"base": local_preference_accuracy(BASE, None, edges,
                                                 cond_dir=ROOT / "runs/native_pool/conditioning")}
    steps = {"cf": 100, "local": 100, "neg": 100, "main": 100, "shuffle": 100}
    for arm in args.arms:
        ckpt = OUT / arm / f"checkpoint_{steps[arm]:04d}.pt"
        if not ckpt.is_file():
            print(f"[skip] {arm}: missing {ckpt}")
            continue
        results[arm] = local_preference_accuracy(
            BASE, ckpt, edges, cond_dir=ROOT / "runs/native_pool/conditioning")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_preference_accuracy.json").write_text(json.dumps(results, indent=1))
    for arm, r in results.items():
        print(f"{arm}: acc={r['accuracy']} median_z={r['median_z']}")


if __name__ == "__main__":
    main()
