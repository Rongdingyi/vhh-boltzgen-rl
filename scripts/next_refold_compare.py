#!/usr/bin/env python
"""Phase C refold comparison: b1/b2/b3 vs base/n3 on the shared 20 cases.

Old per-sample CSV: runs/native_pool/native_refold_results_per_sample.csv (round-1 arms)
New per-sample CSV: runs/native_pool/native_refold_next_results_per_sample.csv (b1/b2/b3)
Outputs: runs/next_stage/refold_phaseC.{json,md}
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage"
METRICS = ["cdr_rmsd", "cdr1_rmsd", "cdr2_rmsd", "cdr3_rmsd", "fr_rmsd",
           "plddt_cdr", "cdr_recovery"]


def load(path: Path, manifest_path: Path) -> list[dict]:
    manifest = {r["sample_id"]: r["arm"] for r in csv.DictReader(manifest_path.open())}
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        r["arm"] = manifest[r["sample_id"]]
        for m in METRICS:
            r[m] = float(r[m]) if r[m] not in ("", None) else float("nan")
    return rows


def per_case_mean(rows: list[dict], metric: str) -> dict[str, float]:
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        out.setdefault(r["case_id"], {}).setdefault(r["arm"], []).append(r[metric])
    return {cid: {arm: vals for arm, vals in arms.items()} for cid, arms in out.items()}


def main() -> None:
    old = load(POOL / "native_refold_results_per_sample.csv",
               POOL / "native_refold_design/manifest.csv")
    new = load(POOL / "native_refold_next_results_per_sample.csv",
               POOL / "native_refold_next/manifest.csv")
    old_arms = {"base": [r for r in old if r["arm"] == "base"],
                "n3": [r for r in old if r["arm"] == "n3"]}
    new_arms = {arm: [r for r in new if r["arm"] == arm] for arm in ("b1", "b2", "b3")}
    arms = {**old_arms, **new_arms}

    table = {}
    for arm, rows in arms.items():
        table[arm] = {"n": len(rows)}
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] == r[m]]
            table[arm][f"{m}_mean"] = st.mean(vals) if vals else None
            table[arm][f"{m}_median"] = st.median(vals) if vals else None

    # per-case paired deltas vs base and vs n3
    per_case = {m: per_case_mean(old + new, m) for m in METRICS}
    paired = {}
    for arm in ("b1", "b2", "b3"):
        entry = {}
        for ref in ("base", "n3"):
            deltas = []
            for cid, vals in per_case["cdr_rmsd"].items():
                if ref in vals and arm in vals:
                    deltas.append(st.mean(vals[arm]) - st.mean(vals[ref]))
            p = None
            try:
                from scipy.stats import wilcoxon
                if len(deltas) >= 5 and any(d != 0 for d in deltas):
                    p = float(wilcoxon(deltas, zero_method="wilcox").pvalue)
            except Exception:
                pass
            entry[f"vs_{ref}"] = {
                "n_cases": len(deltas),
                "delta_cdr_rmsd_mean": st.mean(deltas) if deltas else None,
                "delta_cdr_rmsd_median": st.median(deltas) if deltas else None,
                "n_cases_improved": sum(1 for d in deltas if d < 0),
                "wilcoxon_p": p,
            }
        paired[arm] = entry

    payload = {"table": table, "paired": paired}
    (OUT / "refold_phaseC.json").write_text(json.dumps(payload, indent=1))

    md = "# Phase C refold（固定 20 cases x 4 seqs，Boltz-2 recycling=3, steps=200）\n\n"
    md += "| arm | n | CDR RMSD mean | CDR3 RMSD | FR RMSD | pLDDT(CDR) | recovery |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for arm, t in table.items():
        md += (f"| {arm} | {t['n']} | {t['cdr_rmsd_mean']:.3f} | {t['cdr3_rmsd_mean']:.3f} | "
               f"{t['fr_rmsd_mean']:.3f} | {t['plddt_cdr_mean']:.3f} | {t['cdr_recovery_mean']:.3f} |\n")
    md += "\n## Paired CDR RMSD deltas（per case, 4 seqs/case）\n\n"
    md += "| arm | vs | delta mean (A) | delta median | improved | Wilcoxon p |\n|---|---|---|---|---|---|\n"
    for arm, entry in paired.items():
        for ref, v in entry.items():
            p = "n/a" if v["wilcoxon_p"] is None else f"{v['wilcoxon_p']:.2e}"
            md += (f"| {arm} | {ref} | {v['delta_cdr_rmsd_mean']:+.3f} | "
                   f"{v['delta_cdr_rmsd_median']:+.3f} | {v['n_cases_improved']}/{v['n_cases']} | {p} |\n")
    (OUT / "refold_phaseC.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
