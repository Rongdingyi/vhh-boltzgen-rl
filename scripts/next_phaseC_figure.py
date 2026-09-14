#!/usr/bin/env python
"""Phase C dashboard figure: valid100 deltas + refold CDR RMSD per arm."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage/figures"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["base", "n3", "b1", "b2", "b3"]
LABELS = {"base": "base", "n3": "N3 (round-1)", "b1": "B1 random-sparse",
          "b2": "B2 shuffle", "b3": "B3 CF-DPO"}


def main() -> None:
    summary = json.loads((ROOT / "runs/next_stage/valid100_arms_summary.json").read_text())
    deltas = {
        "base": 0.0,
        "n3": 2.796,  # round-1 reported N3 delta
        "b1": summary["full_b1_s0450"]["delta_vs_base_mean"],
        "b2": summary["full_b2_s0450"]["delta_vs_base_mean"],
        "b3": summary["full_b3_s0500"]["delta_vs_base_mean"],
    }
    old = list(csv.DictReader((POOL / "native_refold_results_per_sample.csv").open()))
    old_man = {r["sample_id"]: r["arm"] for r in csv.DictReader(
        (POOL / "native_refold_design/manifest.csv").open())}
    new = list(csv.DictReader((POOL / "native_refold_next_results_per_sample.csv").open()))
    new_man = {r["sample_id"]: r["arm"] for r in csv.DictReader(
        (POOL / "native_refold_next/manifest.csv").open())}
    rmsd = {a: [] for a in ARMS}
    for r in old:
        a = old_man[r["sample_id"]]
        if a in rmsd:
            rmsd[a].append(float(r["cdr_rmsd"]))
    for r in new:
        a = new_man[r["sample_id"]]
        rmsd[a].append(float(r["cdr_rmsd"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    vals = [deltas[a] for a in ARMS]
    colors = ["C7", "C0", "C2", "C1", "C3"]
    ax.bar(range(len(ARMS)), vals, color=colors)
    ax.set_xticks(range(len(ARMS)), [LABELS[a] for a in ARMS], rotation=15)
    ax.set_ylabel("valid100 mean reward delta vs base")
    ax.axhline(2.5, ls="--", color="gray", lw=1)
    ax.set_title("valid100 (100 cases x 8)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, f"{v:+.2f}", ha="center", fontsize=8)

    ax = axes[1]
    means = [st.mean(rmsd[a]) for a in ARMS]
    sems = [st.stdev(rmsd[a]) / len(rmsd[a]) ** 0.5 for a in ARMS]
    ax.bar(range(len(ARMS)), means, yerr=sems, color=colors, capsize=3)
    ax.set_xticks(range(len(ARMS)), [LABELS[a] for a in ARMS], rotation=15)
    ax.set_ylabel("refold CDR RMSD (A)")
    ax.axhline(2.0, ls="--", color="gray", lw=1)
    ax.set_title("fixed 20 cases x 4 seqs")
    for i, v in enumerate(means):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "figC_dashboard.png", dpi=200)
    print(f"wrote {OUT / 'figC_dashboard.png'}")


if __name__ == "__main__":
    main()
