#!/usr/bin/env python
"""Phase A2: build all AG weight variants (task book §19-§23, §70).

Writes runs/adaptive_granularity/weights/ag_weights.json in a layout that the
existing trainer can consume directly:
  weights["pairs"][pid][variant] = {position: weight}
with variants: no_floor, strict_consensus, strict_region, adaptive,
adaptive_shuffle.  Current CF is NOT rebuilt (frozen baseline, §18).
"""
from __future__ import annotations
import csv
import hashlib
import json
import random
from pathlib import Path

import _common as C  # noqa: E402
from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    adaptive_weights, no_floor_weights, shuffle_pair_weights,
    strict_consensus_weights, strict_region_weights, validate_weights,
    weight_entropy,
)

OUT = C.WEIGHTS_DIR
RHO = 0.70
TOL = 0.05


def main() -> None:
    pairs = C.load_pairs()
    residue = C.load_residue_rows()
    regions = C.load_region_rows()
    rng = random.Random(20260913)
    OUT.mkdir(parents=True, exist_ok=True)

    ag: dict[str, dict] = {"meta": {"tol": TOL, "rho": RHO,
                                    "source": "runs/next_stage/counterfactual",
                                    "eta_current": 0.75, "seed": 20260913},
                           "pairs": {}}
    summary_rows = []
    for pair in pairs:
        pid = C.pair_id(pair)
        rows = residue.get(pid)
        if not rows:
            continue
        diff = [r["position"] for r in rows]
        res_region = regions.get(pid, {})
        built = {
            "no_floor": no_floor_weights(rows, tol=TOL),
            "strict_consensus": strict_consensus_weights(rows, tol=TOL),
            "strict_region": strict_region_weights(rows, res_region, tol=TOL),
            "adaptive": adaptive_weights(rows, res_region, rho=RHO, tol=TOL),
        }
        for name, result in list(built.items()):
            if result.eligible:
                ok, reason = validate_weights(result.weights, diff)
                if not ok:
                    raise ValueError(f"{pid}:{name}: {reason}")
        rng_seed = random.Random(int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16))
        built["adaptive_shuffle"] = type(built["adaptive"])(
            weights=shuffle_pair_weights(built["adaptive"].weights, diff, rng_seed)
            if built["adaptive"].eligible else {},
            eligible=built["adaptive"].eligible,
            reason=built["adaptive"].reason,
        )
        entry = {"case_id": pair["case_id"]}
        for name, result in built.items():
            if result.eligible:
                entry[name] = {str(p): w for p, w in result.weights.items()}
            if name == "adaptive":
                entry["adaptive_modes"] = result.modes
                entry["adaptive_diagnostics"] = result.diagnostics
        ag["pairs"][pid] = entry
        for name, result in built.items():
            summary_rows.append({
                "pair_id": pid, "case_id": pair["case_id"], "variant": name,
                "eligible": int(result.eligible),
                "n_diff": len(diff),
                "n_active": len(result.weights),
                "active_fraction": (len(result.weights) / len(diff)) if diff else 0.0,
                "weight_entropy": weight_entropy(result.weights),
                "reason": result.reason,
                "n_residue_regions": result.diagnostics.get("n_residue_mode_regions"),
                "n_region_regions": result.diagnostics.get("n_region_mode_regions"),
                "n_abstain_regions": result.diagnostics.get("n_abstain_mode_regions",
                    result.diagnostics.get("n_abstain_regions")),
            })

    (OUT / "ag_weights.json").write_text(json.dumps(ag, indent=1))
    fields = list(summary_rows[0])
    with (OUT / "weights_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    cases = sorted({row["case_id"] for row in summary_rows})
    variant_stats = {}
    for name in ["no_floor", "strict_consensus", "strict_region",
                 "adaptive", "adaptive_shuffle"]:
        rows = [r for r in summary_rows if r["variant"] == name and r["eligible"]]
        variant_stats[name] = {
            "eligible_pairs": len(rows),
            "total_pairs": len(summary_rows) // 5,
            "eligible_cases": len({r["case_id"] for r in rows}),
            "total_cases": len(cases),
            "median_active_positions": _median([r["n_active"] for r in rows]),
            "median_active_fraction": _median([r["active_fraction"] for r in rows]),
            "median_weight_entropy": _median([r["weight_entropy"] for r in rows]),
            "pilot_case_eligible": {
                case: sum(1 for r in rows if r["case_id"] == case)
                for case in C.PILOT_TRAIN_CASES},
        }
    payload = {
        "meta": ag["meta"],
        "variant_stats": variant_stats,
        "n_pairs": len(pairs),
        "pairs_without_credit": [C.pair_id(p) for p in pairs
                                 if C.pair_id(p) not in residue],
        "cases_without_eligible": {
            "no_floor": [c for c in cases if not any(
                r["eligible"] and r["case_id"] == c and r["variant"] == "no_floor"
                for r in summary_rows)],
        },
    }
    (OUT / "weights_summary.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["variant_stats"], indent=1))


def _median(values: list[float]) -> float | None:
    import statistics as st
    return st.median(values) if values else None


if __name__ == "__main__":
    main()
