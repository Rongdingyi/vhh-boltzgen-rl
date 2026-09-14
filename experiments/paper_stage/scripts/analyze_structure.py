#!/usr/bin/env python
"""Paper-stage Step 3: case-level structure validation statistics (task book §26).

Input : runs/paper_stage/structure_boltz2 per-sample CSV (from analyze_refold.py)
Output: results/paper_stage/structure_boltz2_per_case.csv
        runs/paper_stage/structure_boltz2/stats.json + stats.md

Statistics are case-level (mean over the 2 sequences), paired vs CF:
mean/median delta, W/T/L, Wilcoxon, bootstrap 95% CI (seed 12345), effect size.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
DESIGN = ROOT / "runs/paper_stage/structure_boltz2"
RESULTS = ROOT / "results/paper_stage"
METRICS = ["cdr_rmsd", "cdr1_rmsd", "cdr2_rmsd", "cdr3_rmsd", "fr_rmsd",
           "plddt_cdr", "cdr_recovery"]
ARMS = ["base", "n3", "shuffle", "cf"]


def load() -> list[dict]:
    manifest = {r["sample_id"]: r["arm"] for r in
                csv.DictReader((DESIGN / "manifest.csv").open())}
    path = DESIGN / "results_per_sample.csv"
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["arm"] = manifest[r["sample_id"]]
        for m in METRICS:
            r[m] = float(r[m]) if r[m] not in ("", None) else float("nan")
    return rows


def case_means(rows: list[dict], metric: str) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["case_id"], r["arm"]), []).append(r[metric])
    return {k: st.mean(v) for k, v in acc.items()}


def bootstrap_ci(deltas: np.ndarray, n: int = 10000, seed: int = 12345) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def main() -> None:
    rows = load()
    per_case_rows = []
    all_cases = sorted({r["case_id"] for r in rows})
    by_metric = {m: case_means(rows, m) for m in METRICS}

    stats = {"n_cases": len(all_cases), "n_seqs_per_case": 2, "metrics": {}}
    for metric in METRICS:
        cm = by_metric[metric]
        entry = {}
        for arm in ARMS:
            vals = [cm[(c, arm)] for c in all_cases if (c, arm) in cm]
            entry[arm] = {"mean": st.mean(vals), "median": st.median(vals),
                          "std": st.stdev(vals) if len(vals) > 1 else 0.0}
        for ref in ("base", "n3", "shuffle"):
            d = np.array([cm[(c, "cf")] - cm[(c, ref)] for c in all_cases
                          if (c, "cf") in cm and (c, ref) in cm])
            lo, hi = bootstrap_ci(d)
            try:
                from scipy.stats import wilcoxon
                p = float(wilcoxon(d, zero_method="wilcox").pvalue) if any(d != 0) else 1.0
            except Exception:
                p = None
            entry[f"cf_vs_{ref}"] = {
                "n": int(len(d)), "delta_mean": float(d.mean()),
                "delta_median": float(np.median(d)),
                "wins": int((d < 0).sum()), "ties": int((d == 0).sum()), "losses": int((d > 0).sum()),
                "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p,
                "cliffs_delta": cliffs_delta(
                    np.array([cm[(c, "cf")] for c in all_cases if (c, "cf") in cm and (c, ref) in cm]),
                    np.array([cm[(c, ref)] for c in all_cases if (c, "cf") in cm and (c, ref) in cm])),
                "case_improved": int((d < 0).sum()),
            }
        stats["metrics"][metric] = entry

    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "structure_boltz2_per_case.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id"] + [f"{m}_{arm}" for arm in ARMS for m in METRICS])
        for c in all_cases:
            w.writerow([c] + [by_metric[m].get((c, arm), float("nan"))
                              for arm in ARMS for m in METRICS])
    (DESIGN / "stats.json").write_text(json.dumps(stats, indent=1))

    md = "# Boltz-2 expanded structure validation (100 cases x 2 seqs, case-level)\n\n"
    md += "| metric | base | n3 | shuffle | cf | cf-base | cf-n3 | cf-shuffle |\n|---|---|---|---|---|---|---|---|\n"
    for m in ("cdr_rmsd", "cdr3_rmsd", "fr_rmsd", "plddt_cdr"):
        e = stats["metrics"][m]
        md += (f"| {m} | {e['base']['mean']:.3f} | {e['n3']['mean']:.3f} | "
               f"{e['shuffle']['mean']:.3f} | {e['cf']['mean']:.3f} | "
               f"{e['cf_vs_base']['delta_mean']:+.3f} | {e['cf_vs_n3']['delta_mean']:+.3f} | "
               f"{e['cf_vs_shuffle']['delta_mean']:+.3f} |\n")
    md += "\n## Paired CDR RMSD (case-level, n=100)\n\n"
    for ref in ("base", "n3", "shuffle"):
        v = stats["metrics"]["cdr_rmsd"][f"cf_vs_{ref}"]
        md += (f"- **CF vs {ref}**: mean {v['delta_mean']:+.3f} A, median {v['delta_median']:+.3f}, "
               f"W/T/L {v['wins']}/{v['ties']}/{v['losses']}, "
               f"95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}], "
               f"Wilcoxon p={v['wilcoxon_p']:.3g}, Cliff's d={v['cliffs_delta']:+.3f}\n")
    (DESIGN / "stats.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
