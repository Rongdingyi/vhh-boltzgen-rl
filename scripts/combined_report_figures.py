#!/usr/bin/env python
"""Combined IF-RL + diffusion-RL report figures.

Outputs to docs/combined_figures/.
  fig1_scores.png    : valid100 CDR margin + global nativeness for all methods
  fig2_curves.png    : training curves of both branches
  fig3_tradeoff.png  : reward vs refold geometry (arm-level + per-sample both branches)
"""
from __future__ import annotations

import collections
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
OUT = ROOT / "docs/combined_figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "base_if": "#4C72B0", "if_r1": "#55A868", "if_r2": "#C44E52",
    "native_base": "#969696", "n1": "#FDBB84", "n2": "#FD8D3C",
    "n3": "#D94801", "n4": "#F16913", "g0": "#8172B2", "m1": "#937860",
}
LABELS = {
    "base_if": "base IF", "if_r1": "IF-RL R1", "if_r2": "IF-RL R2",
    "native_base": "native base", "n1": "N1 RWR", "n2": "N2 full DPO",
    "n3": "N3 fake DPO", "n4": "N4 +anchor", "g0": "G0 (MPNN)", "m1": "M1-beta1",
}
ORDER = ["base_if", "if_r1", "if_r2", "native_base", "n1", "n2", "n3", "n4", "g0", "m1"]


def load_if_panel() -> dict:
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(ROOT / "runs/round1_rl_split/arm3_scores.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("base", "rl", "r2s1", "g0", "m1"):
            per[r["arm"]][r["case_id"]].append(r)
    out = {}
    arm_map = {"base": "base_if", "rl": "if_r1", "r2s1": "if_r2", "g0": "g0", "m1": "m1"}
    for arm, key in arm_map.items():
        if arm not in per:
            continue
        rows = [r for cid in per[arm] for r in per[arm][cid]]
        out[key] = {
            "cdr_margin": st.mean(r["cdr_camelid_margin"] for r in rows),
            "nativeness": st.mean(r["camelid_native_likeness_score"] for r in rows),
        }
    return out


def load_native_panel() -> dict:
    path = POOL / "native_panel_valid100_summary.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for arm, key in [("base", "native_base"), ("n1", "n1"), ("n2", "n2"),
                     ("n3", "n3"), ("n4", "n4")]:
        if arm in data:
            out[key] = {
                "cdr_margin": data[arm]["cdr_camelid_margin"],
                "nativeness": data[arm]["camelid_native_likeness_score"],
            }
    return out


def fig1_scores(panels: dict) -> None:
    methods = [k for k in ORDER if k in panels]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, field, title, fmt in [
        (axes[0], "cdr_margin", "valid100 CDR camelid margin (reward)", "{:.2f}"),
        (axes[1], "nativeness", "valid100 global nativeness", "{:.3f}"),
    ]:
        vals = [panels[k][field] for k in methods]
        bars = ax.bar(range(len(methods)), vals, color=[COLORS[k] for k in methods])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS[k] for k in methods], rotation=25, ha="right", fontsize=8)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
        ylo = 0 if field == "nativeness" else min(0, min(vals) - 1)
        ax.set_ylim(ylo, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_scores.png", dpi=150)
    plt.close(fig)


def fig2_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    # A: IF-RL both rounds
    ax = axes[0]
    r1 = [json.loads(l) for l in open(ROOT / "runs/round1_rl_split/gate6_pilot/train_metrics.jsonl")]
    up1 = [r["update"] for r in r1]
    ax.plot(up1, [r["reward/raw_mean"] for r in r1], color=COLORS["if_r1"], lw=2,
            label="R1 nativeness")
    ax2 = ax.twinx()
    for s, c, lb in [("round2_s1", COLORS["if_r2"], "R2 CDR margin (seedA)"),
                     ("round2_s2", "#E5989B", "R2 CDR margin (seedB)")]:
        rows = [json.loads(l) for l in open(ROOT / f"runs/round1_rl_split/{s}/train_metrics.jsonl")]
        ax2.plot([r["update"] for r in rows], [r["reward/raw_mean"] for r in rows],
                 color=c, lw=2, label=lb)
    ax.set_xlabel("update"); ax.set_ylabel("R1 nativeness", color=COLORS["if_r1"])
    ax2.set_ylabel("R2 CDR margin", color=COLORS["if_r2"])
    ax.set_title("IF-RL training (round 1 + round 2)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="lower right")
    # B: native DPO z
    ax = axes[1]
    for arm, c in [("n2", COLORS["n2"]), ("n3", COLORS["n3"]), ("n4", COLORS["n4"])]:
        rows = [json.loads(l) for l in open(ROOT / f"runs/native_{arm}/train_metrics.jsonl")]
        ax.plot([r["step"] for r in rows], [r["dpo/z_mean"] for r in rows], color=c,
                lw=1.5, label=LABELS[arm])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("DPO z (mean)")
    ax.set_title("Diffusion-RL: DPO preference signal")
    ax.legend(fontsize=8)
    # C: native implicit accuracy
    ax = axes[2]
    for arm, c in [("n2", COLORS["n2"]), ("n3", COLORS["n3"]), ("n4", COLORS["n4"])]:
        rows = [json.loads(l) for l in open(ROOT / f"runs/native_{arm}/train_metrics.jsonl")]
        acc = np.array([r["dpo/implicit_acc"] for r in rows])
        sm = np.convolve(acc, np.ones(25) / 25, mode="valid")
        ax.plot([r["step"] for r in rows][len(acc) - len(sm):], sm, color=c, lw=1.8,
                label=LABELS[arm])
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xlabel("step"); ax.set_ylabel("implicit accuracy (MA25)")
    ax.set_title("Diffusion-RL: implicit preference accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_curves.png", dpi=150)
    plt.close(fig)


def fig3_tradeoff(panels: dict) -> None:
    # ---- IF per-sample scatter (nativeness vs refold CDR RMSD)
    if_rows = list(csv.DictReader(open(ROOT / "runs/round1_rl_split/refold5_results_per_sample.csv")))
    if_man = {r["sample_id"]: (r["case_id"], r["arm"]) for r in
              csv.DictReader(open(ROOT / "runs/round1_rl_split/refold5_design/manifest.csv"))}
    if_nat = {}
    for line in open(ROOT / "runs/round1_rl_split/arm3_scores.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("base", "rl", "r2s1", "m1"):
            if_nat[(r["case_id"], r["arm"], r["idx"])] = r["camelid_native_likeness_score"]
    for fp in sorted(glob.glob("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/outputs/cdr_all/g0_t1.00/g0_scores/g0_scores_shard*.csv")):
        for r in csv.DictReader(open(fp)):
            if str(r.get("selected_in_top8", "")).strip().lower() == "true":
                k = (r["case_id"], "g0", int(r["selection_rank"]) - 1)
                if 0 <= k[2] < 4:
                    if_nat[k] = float(r["classifier_native_probability"])
    ix, iy = [], []
    for r in if_rows:
        c, a = if_man[r["sample_id"]]
        k = (c, a, int(r["sample_id"][-1]))
        if k in if_nat:
            ix.append(if_nat[k]); iy.append(float(r["cdr_rmsd"]))
    ix, iy = np.array(ix), np.array(iy)
    if_corr = float(np.corrcoef(ix, iy)[0, 1])

    # ---- native per-sample scatter (reward vs refold CDR RMSD)
    base_rows = {}
    for d in sorted((POOL / "valid100").iterdir()):
        if not d.is_dir():
            continue
        rows = [json.loads(l) for l in (d / "metadata.jsonl").open()]
        rs = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        if rs:
            base_rows[d.name] = st.mean(rs)
    ranked = sorted(base_rows, key=base_rows.get)
    cases = ranked[:10] + ranked[-10:]
    arm_dirs = {"b": "valid100", "1": "valid100_n1_s0250", "2": "valid100_n2_s0500",
                "3": "valid100_n3_s0350", "4": "valid100_n4_s0500"}
    reward_by_key = {}
    for tag, dirname in arm_dirs.items():
        for ci, cid in enumerate(cases):
            rows = [json.loads(l) for l in (POOL / dirname / cid / "metadata.jsonl").open()]
            scored = [r for r in rows if r["reward_raw"] is not None and not r["contains_UNK"]][:4]
            for k, r in enumerate(scored):
                reward_by_key[f"n{ci:02d}{tag}{k}"] = r["reward_raw"]
    nat_rows = list(csv.DictReader(open(POOL / "native_refold_results_per_sample.csv")))
    nx, ny = [], []
    for r in nat_rows:
        if r["sample_id"] in reward_by_key:
            nx.append(reward_by_key[r["sample_id"]]); ny.append(float(r["cdr_rmsd"]))
    nx, ny = np.array(nx), np.array(ny)
    nat_corr = float(np.corrcoef(nx, ny)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    # Panel A: arm level tradeoff
    ax = axes[0]
    points = [
        ("base_if", 0.0, 1.0), ("if_r1", 0.060, 1.202 / 1.137), ("if_r2", 4.994, 1.337 / 1.137),
        ("g0", 7.586, 1.925 / 1.137), ("m1", 11.311, 4.140 / 1.137),
        ("native_base", 0.0, 1.0), ("n1", 0.583, 1.839 / 1.860), ("n2", 2.702, 1.999 / 1.860),
        ("n3", 2.796, 2.090 / 1.860), ("n4", 2.512, 2.206 / 1.860),
    ]
    for key, dx, ratio in points:
        ax.scatter(dx, ratio, color=COLORS[key], s=60, zorder=3)
        ax.annotate(LABELS[key], (dx, ratio), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("reward gain vs own base (CDR margin)")
    ax.set_ylabel("refold CDR RMSD / own base")
    ax.set_title("reward vs geometry trade-off (all methods)")
    # Panel B: IF per-sample
    ax = axes[1]
    ax.scatter(ix, iy, s=10, alpha=0.5, color=COLORS["if_r2"])
    ax.set_xlabel("global nativeness"); ax.set_ylabel("refold CDR RMSD (A)")
    ax.set_title(f"IF branch per-sample (r={if_corr:+.3f}, n={len(ix)})")
    # Panel C: native per-sample
    ax = axes[2]
    ax.scatter(nx, ny, s=10, alpha=0.5, color=COLORS["n3"])
    ax.set_xlabel("CDR margin (reward)"); ax.set_ylabel("refold CDR RMSD (A)")
    ax.set_title(f"Diffusion branch per-sample (r={nat_corr:+.3f}, n={len(nx)})")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_tradeoff.png", dpi=150)
    plt.close(fig)
    print("correlations: IF %.4f  native %.4f" % (if_corr, nat_corr))


if __name__ == "__main__":
    panels = {}
    panels.update(load_if_panel())
    panels.update(load_native_panel())
    fi = fig1_scores(panels)
    fig2_curves()
    fig3_tradeoff(panels)
    print("figures ->", OUT)
