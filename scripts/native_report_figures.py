#!/usr/bin/env python
"""Native round-1 report figures (task book §87)."""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "docs/native_round1_figures"
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["n1", "n2", "n3", "n4"]
LABELS = {"n1": "N1 RWR", "n2": "N2 full DPO", "n3": "N3 fake DPO", "n4": "N4 fake+anchor"}
COLORS = {"n1": "#55A868", "n2": "#4C72B0", "n3": "#C44E52", "n4": "#DD8452"}


def fig1_training():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for arm in ARMS:
        path = ROOT / f"runs/native_{arm}/train_metrics.jsonl"
        if not path.is_file():
            continue
        rows = [json.loads(l) for l in path.open()]
        if arm != "n1":
            x = [r["step"] for r in rows]
            axes[0].plot(x, [r["dpo/z_mean"] for r in rows], color=COLORS[arm], label=LABELS[arm])
            axes[1].plot(x, [r["dpo/implicit_acc"] for r in rows], color=COLORS[arm], label=LABELS[arm])
        axes[2].plot([r["step"] for r in rows], [r["train/grad_norm"] for r in rows],
                     color=COLORS[arm], label=LABELS[arm])
    axes[0].set_title("DPO z (mean per step)"); axes[0].set_xlabel("step")
    axes[1].set_title("DPO implicit accuracy"); axes[1].set_xlabel("step")
    axes[2].set_title("grad norm"); axes[2].set_xlabel("step")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training.png", dpi=150)
    plt.close(fig)


def load_pool(d: Path):
    per = {}
    for f in d.glob("*/metadata.jsonl"):
        rows = [json.loads(l) for l in f.open()]
        rs = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        if rs:
            per[f.parent.name] = rs
    return per


def fig2_valid100():
    base = load_pool(POOL / "valid100")
    sel = {"n1": "valid100_n1_s0250", "n2": "valid100_n2_s0500",
           "n3": "valid100_n3_s0350", "n4": "valid100_n4_s0500"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # per-case delta histograms
    for arm in ARMS:
        per = load_pool(POOL / sel[arm])
        common = sorted(set(per) & set(base))
        d = [st.mean(per[c]) - st.mean(base[c]) for c in common]
        axes[0].hist(d, bins=25, alpha=0.5, color=COLORS[arm],
                     label=f"{LABELS[arm]} (Δ={st.mean(d):+.2f})")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title("valid100 per-case delta (reward)")
    axes[0].set_xlabel("per-case Δ CDR margin"); axes[0].set_ylabel("# cases")
    axes[0].legend(fontsize=8)
    # reward box summary
    data = [ [st.mean(v) for v in base.values()] ]
    names = ["base"]
    for arm in ARMS:
        per = load_pool(POOL / sel[arm])
        data.append([st.mean(v) for v in per.values()])
        names.append(LABELS[arm])
    bp = axes[1].boxplot(data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        arm = next((k for k, v in LABELS.items() if v == name), None)
        patch.set_facecolor(COLORS.get(arm, "#999999"))
        patch.set_alpha(0.7)
    axes[1].set_title("valid100 per-case mean reward")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_valid100.png", dpi=150)
    plt.close(fig)


def fig3_geometry():
    rows = list(csv.DictReader(open(POOL / "native_refold_results_per_sample.csv")))
    man = {r["sample_id"]: r["arm"] for r in csv.DictReader(open(POOL / "native_refold_design/manifest.csv"))}
    by = {}
    for r in rows:
        by.setdefault(man[r["sample_id"]], []).append(r)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, key, title in [(axes[0], "cdr_rmsd", "CDR RMSD (A)"),
                           (axes[1], "cdr3_rmsd", "CDR3 RMSD (A)"),
                           (axes[2], "plddt_cdr", "pLDDT (CDR)")]:
        names = ["base"] + ARMS
        vals = [st.mean(float(x[key]) for x in by["base"])]
        vals += [st.mean(float(x[key]) for x in by[a]) for a in ARMS]
        bars = ax.bar(range(len(names)), vals,
                      color=["#999999"] + [COLORS[a] for a in ARMS])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(["base"] + [LABELS[a] for a in ARMS], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Refold diagnostics (20 valid100 cases x 4 seq/arm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_geometry.png", dpi=150)
    plt.close(fig)


def fig4_reward_vs_rmsd():
    base = load_pool(POOL / "valid100")
    sel = {"n1": "valid100_n1_s0250", "n2": "valid100_n2_s0500",
           "n3": "valid100_n3_s0350", "n4": "valid100_n4_s0500"}
    rows = list(csv.DictReader(open(POOL / "native_refold_results_per_sample.csv")))
    man = {r["sample_id"]: (r["case_id"], r["arm"]) for r in csv.DictReader(open(POOL / "native_refold_design/manifest.csv"))}
    by = {}
    for r in rows:
        by.setdefault(man[r["sample_id"]][1], []).append(r)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    pts = []
    for arm in ["base"] + ARMS:
        pool_dir = POOL / "valid100" if arm == "base" else POOL / sel[arm]
        per_case = {}
        for f in pool_dir.glob("*/metadata.jsonl"):
            rs = [json.loads(l)["reward_raw"] for l in f.open()]
            rs = [x for x in rs if x is not None]
            if rs:
                per_case[f.parent.name] = st.mean(rs)
        cases = sorted({man[r["sample_id"]][0] for r in by[arm]})
        rw = [per_case[c] for c in cases if c in per_case]
        rm = [st.mean(float(r["cdr_rmsd"]) for r in by[arm] if man[r["sample_id"]][0] == c)
              for c in cases if c in per_case]
        ax.scatter(rw, rm, color=COLORS.get(arm, "#999999"), alpha=0.6, s=22,
                   label=LABELS.get(arm, "base native"))
        pts += list(zip(rw, rm))
    if len(pts) > 3:
        a = np.array(pts)
        corr = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        ax.set_title(f"reward vs refold CDR RMSD (r={corr:+.3f}, n={len(pts)})")
    ax.set_xlabel("valid100 mean CDR margin"); ax.set_ylabel("refold CDR RMSD (A)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_reward_vs_rmsd.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1_training()
    fig2_valid100()
    fig3_geometry()
    fig4_reward_vs_rmsd()
    print("figures ->", OUT)
