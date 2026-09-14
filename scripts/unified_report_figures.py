#!/usr/bin/env python
"""Unified comparison dashboard: all methods (legacy + both RL routes) on one figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
RU = ROOT / "runs/report_unified"
OUT = ROOT / "docs/combined_figures"

METHODS = ["G0", "M1", "M2", "M3", "M3v2", "baseIF", "IF-RL_R1", "IF-RL_R2",
           "nativeBase", "N1", "N2", "N3", "N4"]
COLORS = {
    "G0": "#8172B2", "M1": "#55A868", "M2": "#2E7D32", "M3": "#66BB6A",
    "M3v2": "#937860", "baseIF": "#4C72B0", "IF-RL_R1": "#6BAED6",
    "IF-RL_R2": "#C44E52", "nativeBase": "#969696", "N1": "#FDBB84",
    "N2": "#FD8D3C", "N3": "#D94801", "N4": "#F16913",
}
PANELS = [
    ("cdr_pll", "VHH-ESM-C CDR PLL  (higher better)", "{:.2f}"),
    ("cdr_camelid_margin", "classifier CDR margin  (higher better)", "{:.2f}"),
    ("recovery", "CDR recovery  (higher better)", "{:.2f}"),
    ("mpnn_nll", "ProteinMPNN CDR NLL  (lower better)", "{:.2f}"),
    ("refold_cdr_rmsd", "refold CDR RMSD (A)  (lower better)", "{:.2f}"),
    ("nativeness", "classifier global nativeness", "{:.3f}"),
]


def main() -> None:
    table = json.loads((RU / "master_table.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    for ax, (key, title, fmt) in zip(axes.ravel(), PANELS):
        vals = [table.get(m, {}).get(key) for m in METHODS]
        plotted = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(range(len(METHODS)), plotted,
                      color=[COLORS[m] for m in METHODS])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        for b, v in zip(bars, vals):
            label = fmt.format(v) if v is not None else "n/a"
            y = v if v is not None else 0.0
            ax.text(b.get_x() + b.get_width() / 2, y, label,
                    ha="center", va="bottom", fontsize=7.5)
        lo = 0.0 if key == "nativeness" else min(0.0, min(v for v in plotted if v < 0) if any(v < 0 for v in plotted) else 0.0)
        hi = max(plotted) * 1.25 if max(plotted) > 0 else 1.0
        if key == "nativeness":
            hi = 1.12
        ax.set_ylim(lo if key != "nativeness" else 0.0, hi)
    fig.suptitle("VHH CDR post-training: unified comparison across legacy methods and both RL routes "
                 "(valid100, per-method standard pools)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig5_unified_dashboard.png", dpi=150)
    plt.close(fig)

    # refold comparison (two reference families)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    frozen = ["G0", "M1", "M2", "M3", "baseIF", "IF-RL_R1", "IF-RL_R2"]
    native = ["nativeBase", "N1", "N2", "N3", "N4"]
    for ax, group, title in [
        (axes[0], frozen, "fixed (BoltzGen) backbone reference"),
        (axes[1], native, "native generated backbone (own reference)"),
    ]:
        vals = [table.get(m, {}).get("refold_cdr_rmsd") for m in group]
        bars = ax.bar(range(len(group)), [v or 0 for v in vals],
                      color=[COLORS[m] for m in group])
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group, rotation=25, ha="right", fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v or 0, f"{v:.2f}" if v else "n/a",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"refold CDR RMSD (A) — {title}", fontsize=10)
        ax.set_ylim(0, max([v for v in vals if v] or [1]) * 1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_unified_refold.png", dpi=150)
    plt.close(fig)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
