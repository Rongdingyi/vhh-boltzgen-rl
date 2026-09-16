#!/usr/bin/env python
"""Correctness audit of the frozen diff_only ablation weights (task book §6).

Checks every pair in runs/paper_stage/weights/ablation_weights.json:
  * weights only on winner/loser-differing CDR positions (same-residue = absent)
  * all active weights identical (uniform within the differing support)
  * n_active == n_diff and weight value == 1/n_diff  (no c_cons magnitude)
  * sum == 1 +- 1e-6
  * no all-CDR support (support must equal the differing set, and any
    non-differing design position must have zero weight)
Gate: 20/20 sampled pairs must pass; the full 192-pair sweep is also reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
CREDIT = ROOT / "runs/next_stage/counterfactual/residue_credit.csv"
WEIGHTS = ROOT / "runs/paper_stage/weights/ablation_weights.json"
MANIFEST = ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl"
OUT = ROOT / "runs/paper_stage/diffonly_audit"
N_SAMPLE = 20
TOL = 1e-6


def load_diff_positions() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for row in csv.DictReader(CREDIT.open()):
        out.setdefault(row["pair_id"], []).append(int(row["position"]))
    for rows in out.values():
        rows.sort()
    return out


def load_design_positions() -> dict[str, set[int]]:
    out = {}
    for line in MANIFEST.open():
        row = json.loads(line)
        out[row["case_id"]] = {int(p) for p in row["design_positions"]}
    return out


def check_pair(pair_id: str, diff_positions: list[int], weights: dict,
               design_positions: set[int]) -> dict:
    diff = {int(p) for p in diff_positions}
    keys = {int(k) for k in weights}
    values = [float(v) for v in weights.values()]
    positive = sorted({round(v, 12) for v in values if v > 0})
    checks = {
        "support_is_differing": keys == diff,
        "no_same_residue_weight": keys <= diff,
        "no_non_design_weight": keys <= design_positions,
        "n_active_eq_n_diff": len(keys) == len(diff),
        "uniform_weights": len(positive) == 1,
        "weight_eq_uniform": (not values) or all(
            abs(v - 1.0 / len(values)) <= TOL for v in values),
        "sum_is_one": abs(sum(values) - 1.0) <= TOL,
        "no_cons_reading": True,   # implied by uniform == 1/n_diff
    }
    return {
        "pair_id": pair_id,
        "n_diff": len(diff),
        "n_active": len(keys),
        "unique_positive_weights": len(positive),
        "weight_sum": sum(values),
        "uniform_value": positive[0] if len(positive) == 1 else None,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=N_SAMPLE)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    diff_positions = load_diff_positions()
    design_positions = load_design_positions()
    payload = json.loads(WEIGHTS.read_text())
    pairs = payload["pairs"]

    rows = []
    for pair_id in sorted(pairs):
        weights = pairs[pair_id].get("diff_only")
        if weights is None:
            rows.append({"pair_id": pair_id, "pass": False,
                         "checks": {"missing_diff_only": False}})
            continue
        case_id = pair_id.split(":")[0]
        rows.append(check_pair(pair_id, diff_positions.get(pair_id, []), weights,
                               design_positions.get(case_id, set())))
    failed = [r for r in rows if not r["pass"]]

    rng = random.Random(args.seed)
    sample_ids = sorted({r["pair_id"] for r in rows})
    sample = rng.sample(sample_ids, min(args.sample, len(sample_ids)))
    sample_rows = [r for r in rows if r["pair_id"] in set(sample)]
    sample_pass = sum(1 for r in sample_rows if r["pass"])

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "n_diff", "n_active", "unique_positive_weights",
              "weight_sum", "uniform_value", "pass"]
    with (OUT / f"audit_{args.sample}pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample_rows)
    summary = {
        "n_pairs": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": len(failed),
        "failed_pairs": [r["pair_id"] for r in failed][:10],
        "sample_size": len(sample_rows),
        "sample_pass": sample_pass,
        "sample_seed": args.seed,
        "gate": sample_pass == len(sample_rows) and not failed,
        "meta": payload.get("meta", {}),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not summary["gate"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
