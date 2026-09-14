#!/usr/bin/env python3
"""Generate group-meeting figures for the RL report.

Outputs to docs/round1_figures/.
"""
from __future__ import annotations

import csv
import collections
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RL = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1_rl_split")
OUT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/docs/round1_figures")
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["base", "rl", "r2s1", "g0", "m1", "design"]
ARMS_REFOLD = ["base", "rl", "r2s1", "g0", "m1"]
ARM_LABEL = {
    "base": "base IF",
    "rl": "RL-R1",
    "r2s1": "RL-R2",
    "g0": "G0",
    "m1": "M1-beta1",
    "design": "design readout",
}
ARM_SUB = {
    "base": "",
    "rl": "(nativeness)",
    "r2s1": "(CDR+guard)",
    "g0": "(ProteinMPNN)",
    "m1": "(pure guidance)",
    "design": "(atom14)",
}
COLORS = {"base": "#4C72B0", "rl": "#55A868", "r2s1": "#C44E52",
          "g0": "#8172B2", "m1": "#937860", "design": "#DD8452"}


def load_metrics(path):
    return [json.loads(l) for l in open(path)]


def style_bars(ax, bars, arms, fmt="{:.2f}"):
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=4)
    for b, a in zip(bars, arms):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(
            b.get_height()), ha="center", va="bottom", fontsize=8)


def fig1_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    r1 = load_metrics(RL / "gate6_pilot/train_metrics.jsonl")
    ax = axes[0]
    up = [r["update"] for r in r1]
    ax.plot(up, [r["reward/raw_mean"] for r in r1], color=COLORS["rl"], lw=2, label="nativeness (raw)")
    ax.set_xlabel("update"); ax.set_ylabel("nativeness (mean)")
    ax.set_title("Round 1: nativeness reward (24 cases, GRPO)")
    ax2 = ax.twinx()
    ax2.plot(up, [r["policy/kl"] for r in r1], color="gray", ls="--", lw=1.2, label="KL")
    ax2.set_ylabel("KL (policy || reference)")
    ax2.tick_params(axis="y", colors="gray")
    ax.legend(loc="lower right", fontsize=8); ax2.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    for s, c, lb in [("round2_s1", "#C44E52", "seed A"), ("round2_s2", "#E5989B", "seed B")]:
        rows = load_metrics(RL / s / "train_metrics.jsonl")
        up = [r["update"] for r in rows]
        ax.plot(up, [r["reward/raw_mean"] for r in rows], color=c, lw=2, label=f"CDR margin ({lb})")
        ax.plot(up, [r.get("reward/guard_nat_mean", float("nan")) for r in rows],
                color=c, ls=":", lw=1.5, label=f"global nat guard ({lb})")
    ax.set_xlabel("update"); ax.set_ylabel("score")
    ax.set_title("Round 2: CDR margin + nativeness guardrail")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training_curves.png", dpi=150)
    plt.close(fig)


def load_arm3():
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(RL / "arm3_scores.jsonl"):
        r = json.loads(line)
        per[r["case_id"]][r["arm"]].append(r)
    for line in open(RL / "design_arm_scores.jsonl"):
        r = json.loads(line)
        per[r["case_id"]][r["arm"]].append(r)
    return per


def arm_means(per, field, arms=ARMS):
    return {a: st.mean(r[field] for c in per for r in per[c].get(a, [])) for a in arms}


def fig2_scores(per):
    fields = [("camelid_native_likeness_score", "global nativeness"),
              ("cdr_camelid_margin", "CDR camelid margin"),
              ("p_cdr_camelid", "P(CDR camelid)"),
              ("background_discordance_jsd", "FR-CDR discordance (JS)")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, (f, title) in zip(axes, fields):
        m = arm_means(per, f)
        bars = ax.bar(range(len(ARMS)), [m[a] for a in ARMS],
                      color=[COLORS[a] for a in ARMS])
        ax.set_title(title, fontsize=10)
        style_bars(ax, bars, ARMS, fmt="{:.3f}" if f == "background_discordance_jsd" else "{:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_arm_scores.png", dpi=150)
    plt.close(fig)


def fig3_ood(per):
    fields = [("nativeness_ensemble_js", "ensemble disagreement (nat JS)"),
              ("cdr_energy", "CDR energy (lower=better)"),
              ("cdr_embedding_distance", "CDR embedding distance")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (f, title) in zip(axes, fields):
        m = arm_means(per, f)
        bars = ax.bar(range(len(ARMS)), [m[a] for a in ARMS],
                      color=[COLORS[a] for a in ARMS])
        ax.set_title(title, fontsize=10)
        style_bars(ax, bars, ARMS, fmt="{:.3f}")
    # OOD review rate as 4th panel
    fig2, ax = plt.subplots(figsize=(7.5, 4.6))
    rates = {a: sum(1 for c in per for r in per[c].get(a, []) if r["ood_flag"] == "review")
             / max(1, sum(1 for c in per for r in per[c].get(a, []))) for a in ARMS}
    bars = ax.bar(range(len(ARMS)), [rates[a] for a in ARMS],
                  color=[COLORS[a] for a in ARMS])
    style_bars(ax, bars, ARMS, fmt="{:.0%}")
    ax.set_title("OOD review rate", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig3b_ood_review.png", dpi=150)
    plt.close(fig2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ood.png", dpi=150)
    plt.close(fig)


def fig4_refold():
    rows = list(csv.DictReader(open(RL / "refold5_results_per_sample.csv")))
    man = {r["sample_id"]: (r["case_id"], r["arm"]) for r in
           csv.DictReader(open(RL / "refold5_design/manifest.csv"))}
    by = collections.defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            by[man[r["sample_id"]][1]].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for ax, (f, title) in zip(axes, [("cdr_rmsd", "CDR RMSD vs backbone (A)"),
                                     ("cdr3_rmsd", "CDR3 RMSD (A)"),
                                     ("plddt_cdr", "pLDDT (CDR)")]):
        m = {a: st.mean(float(x[f]) for x in by[a]) for a in ARMS_REFOLD}
        bars = ax.bar(range(len(ARMS_REFOLD)), [m[a] for a in ARMS_REFOLD],
                      color=[COLORS[a] for a in ARMS_REFOLD])
        ax.set_title(title, fontsize=10)
        style_bars(ax, bars, ARMS_REFOLD)
    fig.suptitle("Refold self-consistency (Boltz-2, 20 cases x 4 seqs/arm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_refold_rmsd.png", dpi=150)
    plt.close(fig)
    return by


def fig5_scatter(per):
    # nativeness vs cdr_rmsd, colored by arm (refold arms only)
    rows = list(csv.DictReader(open(RL / "refold5_results_per_sample.csv")))
    man = {r["sample_id"]: (r["case_id"], r["arm"]) for r in
           csv.DictReader(open(RL / "refold5_design/manifest.csv"))}
    nat = {}
    for line in open(RL / "arm3_scores.jsonl"):
        r = json.loads(line)
        if r["arm"] in ("base", "rl", "r2s1", "m1"):
            nat[(r["case_id"], r["arm"], r["idx"])] = r["camelid_native_likeness_score"]
    for fp in sorted(glob.glob("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/outputs/cdr_all/g0_t1.00/g0_scores/g0_scores_shard*.csv")):
        for r in csv.DictReader(open(fp)):
            if str(r.get("selected_in_top8", "")).strip().lower() == "true":
                k = (r["case_id"], "g0", int(r["selection_rank"]) - 1)
                if 0 <= k[2] < 4:
                    nat[k] = float(r["classifier_native_probability"])
    xs, ys, cs = [], [], []
    for r in rows:
        if r["status"] != "ok":
            continue
        c, a = man[r["sample_id"]]
        k = (c, a, int(r["sample_id"][-1]))
        if k in nat:
            xs.append(nat[k]); ys.append(float(r["cdr_rmsd"])); cs.append(COLORS[a])
    xs, ys = np.array(xs), np.array(ys)
    corr = float(np.corrcoef(xs, ys)[0, 1])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(xs, ys, c=cs, s=14, alpha=0.65)
    ax.set_xlabel("global nativeness (classifier)")
    ax.set_ylabel("CDR RMSD (A)")
    ax.set_title(f"nativeness does not predict refold fidelity (r = {corr:+.3f}, n={len(xs)})")
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a],
                          label=f"{ARM_LABEL[a]} {ARM_SUB[a]}".strip())
               for a in ARMS_REFOLD]
    ax.legend(handles=handles, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_nat_vs_rmsd.png", dpi=150)
    plt.close(fig)
    return corr


def fig6_valid100_r1():
    # per-case delta histogram for round 1 (valid100)
    per = collections.defaultdict(lambda: {"base": [], "rl": []})
    for i in range(3):
        with open(RL / f"v100eval_shard{i}/eval_per_case.csv") as f:
            for r in csv.DictReader(f):
                if r["arm"] in ("base", "rl"):
                    per[r["case_id"]][r["arm"]].append(float(r["raw_score"]))
    deltas = [st.mean(v["rl"]) - st.mean(v["base"]) for v in per.values() if len(v["base"]) > 2]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=24, color=COLORS["rl"], alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("per-case delta (RL - base IF), nativeness")
    ax.set_ylabel("# cases")
    wins = sum(d > 0 for d in deltas)
    ax.set_title(f"Round-1 on valid100 (n={len(deltas)}): {wins} wins, mean {st.mean(deltas):+.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_valid100_delta.png", dpi=150)
    plt.close(fig)


def fig7_design_vs_arms(per):
    # per-case paired deltas: design readout vs base IF, two metrics
    pairs = []
    for c in per:
        if per[c].get("design") and per[c].get("base"):
            pairs.append((
                per[c]["design"][0]["cdr_camelid_margin"]
                - st.mean(x["cdr_camelid_margin"] for x in per[c]["base"]),
                per[c]["design"][0]["camelid_native_likeness_score"]
                - st.mean(x["camelid_native_likeness_score"] for x in per[c]["base"]),
            ))
    d_cdr = [p[0] for p in pairs]; d_nat = [p[1] for p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, d, title in [(axes[0], d_cdr, "design readout vs base IF: CDR margin"),
                         (axes[1], d_nat, "design readout vs base IF: global nativeness")]:
        ax.hist(d, bins=24, color=COLORS["design"], alpha=0.85)
        ax.axvline(0, color="k", lw=1)
        wins = sum(x > 0 for x in d)
        ax.set_title(f"{title}\n{wins}/{len(d)} wins, mean {st.mean(d):+.3f}", fontsize=9)
        ax.set_xlabel("per-case delta (design - base)"); ax.set_ylabel("# cases")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_design_vs_base.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    per = load_arm3()
    fig1_training_curves()
    fig2_scores(per)
    fig3_ood(per)
    fig4_refold()
    corr = fig5_scatter(per)
    fig6_valid100_r1()
    fig7_design_vs_arms(per)
    print(f"figures written to {OUT}; corr={corr:+.4f}")
