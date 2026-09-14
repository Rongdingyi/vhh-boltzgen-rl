#!/usr/bin/env python
"""Phase B1 analysis from saved artifacts (no GPU; can rerun cheaply).

Inputs : runs/next_stage/trajectory/{trajectory_metrics.csv,trajectory_sequences.jsonl}
Outputs: trajectory_summary.json, figures/figB1_*.png, docs/TEMPORAL_CREDIT_AUDIT.md
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.trajectory_audit import find_sigma_threshold  # noqa: E402

OUT = ROOT / "runs/next_stage/trajectory"
DOC = ROOT / "docs/TEMPORAL_CREDIT_AUDIT.md"


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def main() -> None:
    metric_rows = list(csv.DictReader((OUT / "trajectory_metrics.csv").open()))
    for r in metric_rows:
        r["step"] = int(r["step"])
        r["design_index"] = int(r["design_index"])
        for k in ("sigma", "cdr_identity", "seq_identity", "stable_fraction",
                  "rmsd_fr_bb", "rmsd_cdr_bb", "rmsd_cdr_fake"):
            r[k] = float(r[k]) if r[k] != "" else float("nan")
        r["valid"] = r["valid"] == "True"
    seq_rows = [json.loads(l) for l in (OUT / "trajectory_sequences.jsonl").open()]
    reward_by_seq = {r["sequence"]: r["reward"] for r in seq_rows if r.get("reward") is not None}

    steps = sorted({r["step"] for r in metric_rows})
    def agg(key, fn=st.median):
        return [fn([r[key] for r in metric_rows if r["step"] == t]) for t in steps]
    sigmas = agg("sigma")
    med_cdr_id = agg("cdr_identity")
    med_seq_id = agg("seq_identity")
    valid_rate = [sum(1 for r in metric_rows if r["step"] == t and r["valid"])
                  / max(1, sum(1 for r in metric_rows if r["step"] == t)) for t in steps]
    med_stable = agg("stable_fraction")
    med_fr = agg("rmsd_fr_bb")
    med_cdrbb = agg("rmsd_cdr_bb")
    med_fake = agg("rmsd_cdr_fake")

    final_seq = {(r["case_id"], r["design_index"]): r["sequence"]
                 for r in seq_rows if r["step"] == steps[-1]}
    final_reward = {k: reward_by_seq.get(s) for k, s in final_seq.items()}
    seq_lookup = {(r["case_id"], r["design_index"], r["step"]): r["sequence"]
                  for r in seq_rows}

    reward_corr, reward_mae = [], []
    for t in steps:
        pairs = []
        for r in metric_rows:
            if r["step"] != t:
                continue
            seq_t = seq_lookup[(r["case_id"], r["design_index"], t)]
            rt = reward_by_seq.get(seq_t)
            rf = final_reward.get((r["case_id"], r["design_index"]))
            if rt is not None and rf is not None:
                pairs.append((rt, rf))
        reward_corr.append(pearson([a for a, _ in pairs], [b for _, b in pairs]))
        reward_mae.append(st.mean(abs(a - b) for a, b in pairs) if pairs else None)

    by_traj: dict[tuple, dict[int, dict]] = {}
    for r in metric_rows:
        by_traj.setdefault((r["case_id"], r["design_index"]), {})[r["step"]] = r
    conv = {"fr_bb": [], "cdr_bb": [], "cdr_fake": []}
    for key, rmsd_key in (("fr_bb", "rmsd_fr_bb"), ("cdr_bb", "rmsd_cdr_bb"),
                          ("cdr_fake", "rmsd_cdr_fake")):
        for t in steps:
            traj_convs = []
            for series in by_traj.values():
                vals = [series[s][rmsd_key] for s in series]
                finite = [v for v in vals if v == v]
                scale = max(finite) if finite and max(finite) > 0 else 1.0
                traj_convs.append(max(0.0, 1.0 - series[t][rmsd_key] / scale))
            conv[key].append(st.median(traj_convs) if traj_convs else float("nan"))

    sigma_seq = find_sigma_threshold(sigmas, med_cdr_id, valid_rate, 0.80, 0.90)
    sigma_90 = find_sigma_threshold(sigmas, med_cdr_id, valid_rate, 0.90, 0.90)
    gate_window = [
        {"step": t, "sigma": sigmas[t], "conv_cdr_bb": conv["cdr_bb"][t],
         "cdr_identity": med_cdr_id[t]}
        for t in steps
        if conv["cdr_bb"][t] > 0.8 and med_cdr_id[t] < 0.5
    ]
    h3 = "STRONG SUPPORT" if gate_window else "WEAK"

    summary = {
        "n_cases": len({r["case_id"] for r in metric_rows}),
        "n_trajectories": len({(r["case_id"], r["design_index"]) for r in metric_rows}),
        "n_steps": len(steps),
        "curves": {
            "step": steps, "sigma": sigmas,
            "median_cdr_identity": med_cdr_id, "median_seq_identity": med_seq_id,
            "valid_rate": valid_rate, "median_stable_fraction": med_stable,
            "median_rmsd_fr_bb": med_fr, "median_rmsd_cdr_bb": med_cdrbb,
            "median_rmsd_cdr_fake": med_fake,
            "norm_conv_fr_bb": conv["fr_bb"], "norm_conv_cdr_bb": conv["cdr_bb"],
            "norm_conv_cdr_fake": conv["cdr_fake"],
            "reward_corr_to_final": reward_corr, "reward_mae_to_final": reward_mae,
        },
        "sigma_seq": sigma_seq, "sigma_90": sigma_90,
        "gate_b_window_steps": gate_window, "h3": h3,
    }
    (OUT / "trajectory_summary.json").write_text(json.dumps(summary, indent=1))

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    log_sigma = [(float("nan") if s <= 0 else math.log(s)) for s in sigmas]
    x = list(range(len(steps)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    panels = [
        (axes[0, 0], med_cdr_id, "median CDR identity to final", 0.80),
        (axes[0, 1], med_stable, "median stable residue fraction", None),
        (axes[0, 2], reward_corr, "corr(R_t, R_final)", None),
        (axes[1, 0], conv["fr_bb"], "normalized FR-bb convergence", 0.8),
        (axes[1, 1], conv["cdr_bb"], "normalized CDR-bb convergence", 0.8),
        (axes[1, 2], conv["cdr_fake"], "normalized fake-atom convergence", None),
    ]
    for ax, vals, title, line in panels:
        vals = [v if v is not None else float("nan") for v in vals]
        ax.plot(x, vals)
        if line is not None:
            ax.axhline(line, ls="--", color="gray", lw=1)
        if sigma_seq is not None:
            ax.axvline(sigmas.index(sigma_seq), ls=":", color="C3", lw=1)
        ax.set_title(title)
        ax.set_xlabel("diffusion step")
    fig.tight_layout()
    fig.savefig(OUT / "figures/figB1_emergence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(log_sigma, med_cdr_id, marker=".", label="median CDR identity")
    ax.plot(log_sigma, conv["cdr_bb"], marker=".", label="norm. CDR-bb convergence")
    ax.plot(log_sigma, valid_rate, ls="--", label="valid decode rate")
    if sigma_seq is not None:
        ax.axvline(math.log(sigma_seq), ls=":", color="C3",
                   label=f"sigma_seq={sigma_seq:.2f}")
    ax.set_xlabel("log sigma (t_hat)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures/figB1_vs_sigma.png", dpi=200)
    plt.close(fig)

    md = f"""# Temporal Credit Audit（Phase B1，Gate B）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §20–§29
设置：{summary['n_cases']} cases × {summary['n_trajectories'] // summary['n_cases']} trajectories
（frozen native base，50 steps，官方 sampler + res_from_atom14，未训练）。

## 1. 关键阈值

| 量 | 值 |
|---|---|
| sigma_seq（median CDR identity >= 0.80 且 valid >= 0.90） | {sigma_seq} |
| sigma_90（identity >= 0.90） | {sigma_90} |
| Gate B window（CDR-bb conv > 0.8 且 identity < 0.5 的 step 数） | {len(gate_window)} |

**Gate B decision: {h3}**

## 2. 曲线（median across trajectories）

| step | sigma | CDR identity | stable frac | valid | FR-bb RMSD | CDR-bb RMSD | fake RMSD | reward corr |
|---|---|---|---|---|---|---|---|---|
"""
    for i in steps:
        rc = reward_corr[i]
        rc_s = f"{rc:+.3f}" if rc is not None else "n/a"
        md += (f"| {i} | {sigmas[i]:.2f} | {med_cdr_id[i]:.3f} | {med_stable[i]:.3f} | "
               f"{valid_rate[i]:.3f} | {med_fr[i]:.2f} | {med_cdrbb[i]:.2f} | "
               f"{med_fake[i]:.2f} | {rc_s} |\n")
    md += f"""
图：`runs/next_stage/trajectory/figures/figB1_emergence.png`、`figB1_vs_sigma.png`。

## 3. 结论

- backbone 收敛与 sequence identity 的先后：gate window = {len(gate_window)} step →
  H3 = {h3}。
- 若 STRONG：用 §29 的 g(sigma)（tau=0.5，不做 sweep）做 temporal weighting；
  否则 temporal DPO 仅作小 ablation。

*脚本：`scripts/next_audit_trajectory.py`（采集）、`scripts/next_analyze_trajectory.py`（分析）*
"""
    DOC.write_text(md)
    print(json.dumps({"sigma_seq": sigma_seq, "sigma_90": sigma_90, "h3": h3,
                      "gate_window_steps": len(gate_window)}, indent=1))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
