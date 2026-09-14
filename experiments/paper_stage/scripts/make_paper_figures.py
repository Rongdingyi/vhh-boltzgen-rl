#!/usr/bin/env python
"""Paper-stage main figures (Figure A/B/C + multi-seed + query budget)."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
PS = ROOT / "runs/paper_stage"
RESULTS = ROOT / "results/paper_stage"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def jload(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    # Figure A: credit sparsity
    stats = jload(ROOT / "runs/next_stage/counterfactual/credit_stats.json")
    g = stats["gate"]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ks = ["median_top10_mass", "median_top20_mass", "median_top30_mass", "median_top50_mass"]
    vals = [g[k] for k in ks]
    ax.bar(["top10%", "top20%", "top30%", "top50%"], vals, color=["C0", "C0", "C3", "C7"])
    ax.axhline(0.60, ls="--", color="gray", lw=1)
    ax.set_ylabel("median positive-credit mass")
    ax.set_title(f"Credit sparsity (n_eff/n_diff={g['median_neff_ratio']:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "figA_credit_sparsity.png", dpi=200)
    plt.close(fig)

    # Figure B: attribution ablation bars
    abl = {"N3": 2.796, "Shuffle": 3.531, "Diff-only": 4.221,
           "Drop-only": 4.932, "Gain-only": 5.258, "CF-DPO": 5.566}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    colors = ["C7", "C1", "C0", "C2", "C2", "C3"]
    ax.bar(list(abl), list(abl.values()), color=colors)
    for i, (k, v) in enumerate(abl.items()):
        ax.text(i, v + 0.06, f"{v:+.2f}", ha="center", fontsize=8)
    ax.set_ylabel("valid100 $\\Delta$ reward vs base")
    ax.set_ylim(0, 6.4)
    ax.set_title("Credit attribution ablations (seed 42)")
    fig.tight_layout()
    fig.savefig(FIG / "figB_ablation.png", dpi=200)
    plt.close(fig)

    # Figure C: reward–structure Pareto (Boltz-2 and Protenix panels)
    boltz = jload(ROOT / "runs/paper_stage/structure_boltz2/stats.json")
    prox = jload(ROOT / "runs/paper_stage/structure_protenix/stats.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    arms = [("base", 0.0, "gray"), ("n3", 2.796, "C0"),
            ("shuffle", 3.531, "C1"), ("cf", 5.566, "C3")]
    if boltz:
        for arm, d, c in arms:
            ax = axes[0]
            ax.scatter(d, boltz["metrics"]["cdr_rmsd"][arm]["mean"], color=c, s=45)
            ax.annotate(arm, (d, boltz["metrics"]["cdr_rmsd"][arm]["mean"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[0].set_xlabel("valid100 $\\Delta$ reward"); axes[0].set_ylabel("CDR RMSD (A)")
    axes[0].set_title("Boltz-2 (100 cases x 2)")
    if prox:
        for arm, d, c in arms:
            if arm in ("base", "n3", "cf"):
                axes[1].scatter(d, prox[arm]["cdr_rmsd_mean"], color=c, s=45)
                axes[1].annotate(arm, (d, prox[arm]["cdr_rmsd_mean"]),
                                 textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_xlabel("valid100 $\\Delta$ reward"); axes[1].set_ylabel("CDR RMSD (A)")
    axes[1].set_title("Protenix-v2 (50 cases x 2)")
    fig.tight_layout()
    fig.savefig(FIG / "figC_pareto.png", dpi=200)
    plt.close(fig)

    # Figure D: query budget
    qb = None
    with (RESULTS / "query_budget.csv").open() as fh:
        qb = list(csv.DictReader(fh))
    fig, ax = plt.subplots(figsize=(5, 3.8))
    xs = [float(r["queries_per_pair_mean"]) for r in qb]
    ys = [float(r["delta_mean"]) for r in qb]
    ax.plot(xs, ys, marker="o", color="C3")
    for x, y, r in zip(xs, ys, qb):
        ax.annotate(f"{float(r['budget'])*100:.0f}%", (x, y), textcoords="offset points",
                    xytext=(4, -8), fontsize=8)
    ax.set_xlabel("scorer queries / pair"); ax.set_ylabel("valid100 $\\Delta$ reward")
    ax.set_title("Query-budget curve")
    fig.tight_layout()
    fig.savefig(FIG / "figD_query_budget.png", dpi=200)
    plt.close(fig)

    # Figure E: multi-seed
    with (RESULTS / "multiseed_summary.csv").open() as fh:
        ms = list(csv.DictReader(fh))
    per_arm = {}
    for r in ms:
        if r.get("valid100_delta"):
            per_arm.setdefault(r["arm"], []).append(float(r["valid100_delta"]))
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    order = ["cf", "n3", "shuffle"]
    means = [st.mean(per_arm[a]) for a in order]
    stds = [st.stdev(per_arm[a]) if len(per_arm[a]) > 1 else 0.0 for a in order]
    ax.bar(order, means, yerr=stds, capsize=4, color=["C3", "C0", "C1"])
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + 0.06, f"{m:+.2f}±{s:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("valid100 $\\Delta$ reward (3 seeds)")
    ax.set_title("Multi-seed robustness")
    fig.tight_layout()
    fig.savefig(FIG / "figE_multiseed.png", dpi=200)
    plt.close(fig)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
