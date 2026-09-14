#!/usr/bin/env python
"""Phase A1/A2 analysis: residue + region credit, sparsity, Gate A.

Reads runs/next_stage/counterfactual/{counterfactual_sequences,scores}.jsonl
Writes residue_credit.csv / region_credit.csv / pair_credit_summary.csv /
credit_stats.json / figures/ and docs/COUNTERFACTUAL_CREDIT_AUDIT.md.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.cdr_regions import region_map  # noqa: E402
from vhh_rl.credit.credit_metrics import effective_n, topk_mass  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402

POOL = ROOT / "runs/native_pool"
CF = ROOT / "runs/next_stage/counterfactual"
FIG = CF / "figures"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split="train",
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def median_or_none(xs):
    return st.median(xs) if xs else None


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(rank(xs), rank(ys))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    seqs = [json.loads(l) for l in (CF / "counterfactual_sequences.jsonl").open()]
    scores = {r["cf_id"]: r["score"] for r in
              (json.loads(l) for l in (CF / "counterfactual_scores.jsonl").open())}
    pairs = [json.loads(l) for l in (POOL / "pairs_train.jsonl").open()]
    for s in seqs:
        s["score"] = scores[s["cf_id"]]

    n_scored = len({s["sequence_sha1"] for s in seqs})
    score_drift = []
    residue_rows, region_rows, pair_rows = [], [], []
    per_pair_cons: list[list[float]] = []
    per_pair_topk: list[dict] = []
    c_drop_all, c_gain_all = [], []
    c_cons_region = {"cdr1": [], "cdr2": [], "cdr3": []}

    for pair in pairs:
        pid = f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
        case = cases[pair["case_id"]]
        regions = region_map(case)
        w_rows = {s["position"]: s for s in seqs
                  if s["pair_id"] == pid and s["kind"] == "winner_drop"}
        l_rows = {s["position"]: s for s in seqs
                  if s["pair_id"] == pid and s["kind"] == "loser_gain"}
        r_win = next(s["score"] for s in seqs if s["cf_id"] == f"{pid}:orig_w")
        r_lose = next(s["score"] for s in seqs if s["cf_id"] == f"{pid}:orig_l")
        score_drift += [abs(r_win - pair["winner_reward"]), abs(r_lose - pair["loser_reward"])]

        entries = []
        for pos in sorted(w_rows):
            c_drop = r_win - w_rows[pos]["score"]
            c_gain = l_rows[pos]["score"] - r_lose
            c_avg = 0.5 * (c_drop + c_gain)
            c_cons = min(c_drop, c_gain) if (c_drop > 0 and c_gain > 0) else 0.0
            region = next((k for k, v in regions.items() if pos in v), None)
            e = {
                "pair_id": pid, "case_id": pair["case_id"], "position": pos,
                "region": region, "winner_aa": case.full_sequence[pos],
                "c_drop": c_drop, "c_gain": c_gain, "c_avg": c_avg, "c_cons": c_cons,
                "disagreement": abs(c_drop - c_gain),
                "sign_agree": (c_drop > 0 and c_gain > 0) or (c_drop < 0 and c_gain < 0)
                              or (c_drop == 0 and c_gain == 0),
            }
            entries.append(e)
            residue_rows.append(e)
            c_drop_all.append(c_drop)
            c_gain_all.append(c_gain)
            if region:
                c_cons_region[region].append(c_cons)

        n_diff = len(entries)
        cons = [e["c_cons"] for e in entries]
        pair_rows.append({
            "pair_id": pid, "case_id": pair["case_id"], "n_diff": n_diff,
            "n_positive_consistent": sum(1 for c in cons if c > 0),
            "positive_fraction": sum(1 for c in cons if c > 0) / max(n_diff, 1),
            "sign_agreement_rate": sum(1 for e in entries if e["sign_agree"]) / max(n_diff, 1),
            "credit_abs_sum": sum(e["disagreement"] for e in entries),
            "credit_positive_sum": sum(cons),
            "c_avg_sum": sum(e["c_avg"] for e in entries),
            "reward_gap": pair["reward_gap"],
            "n_eff": effective_n(cons),
            "n_eff_ratio": effective_n(cons) / max(n_diff, 1),
            **{f"topk_{k}": v for k, v in topk_mass(cons, n_diff).items()},
        })
        per_pair_cons.append(cons)
        per_pair_topk.append(topk_mass(cons, n_diff))

        for region in regions:
            drop = next((s for s in seqs if s["pair_id"] == pid and s["region"] == region
                         and s["kind"] == "region_drop"), None)
            gain = next((s for s in seqs if s["pair_id"] == pid and s["region"] == region
                         and s["kind"] == "region_gain"), None)
            if drop is None or gain is None:
                continue
            g_drop = r_win - drop["score"]
            g_gain = gain["score"] - r_lose
            g_avg = 0.5 * (g_drop + g_gain)
            g_cons = min(g_drop, g_gain) if (g_drop > 0 and g_gain > 0) else 0.0
            resid = [e for e in entries if e["region"] == region]
            region_rows.append({
                "pair_id": pid, "case_id": pair["case_id"], "region": region,
                "n_changed_positions": len(resid), "G_drop": g_drop, "G_gain": g_gain,
                "G_avg": g_avg, "G_cons": g_cons,
                "sum_residue_c_avg": sum(e["c_avg"] for e in resid),
                "sum_residue_c_cons": sum(e["c_cons"] for e in resid),
                "epistasis_avg": g_avg - sum(e["c_avg"] for e in resid),
                "epistasis_cons": g_cons - sum(e["c_cons"] for e in resid),
                "sign_agree": (g_drop > 0 and g_gain > 0) or (g_drop < 0 and g_gain < 0),
            })

    # ------------------------------------------------------------------ stats
    top30 = [t["top30"] for t in per_pair_topk]
    gate = {
        "median_top10_mass": median_or_none([t["top10"] for t in per_pair_topk]),
        "median_top20_mass": median_or_none([t["top20"] for t in per_pair_topk]),
        "median_top30_mass": median_or_none(top30),
        "median_top50_mass": median_or_none([t["top50"] for t in per_pair_topk]),
        "median_neff_ratio": median_or_none([r["n_eff_ratio"] for r in pair_rows]),
        "median_positive_fraction": median_or_none([r["positive_fraction"] for r in pair_rows]),
        "median_sign_agreement_rate": median_or_none([r["sign_agreement_rate"] for r in pair_rows]),
    }
    if gate["median_top30_mass"] >= 0.60 and gate["median_neff_ratio"] <= 0.60:
        gate["h1"] = "STRONG SUPPORT"
    elif gate["median_top30_mass"] >= 0.45:
        gate["h1"] = "PARTIAL SUPPORT"
    elif gate["median_top30_mass"] < 0.45 and gate["median_neff_ratio"] > 0.75:
        gate["h1"] = "WEAK SUPPORT"
    else:
        gate["h1"] = "PARTIAL SUPPORT"

    region_stats = {}
    for region in ("cdr1", "cdr2", "cdr3"):
        rows_r = [r for r in region_rows if r["region"] == region]
        region_stats[region] = {
            "n_pairs_with_changes": len(rows_r),
            "n_residue_values": len(c_cons_region[region]),
            "residue_c_cons_median": median_or_none([v for v in c_cons_region[region]]),
            "region_G_cons_median": median_or_none([r["G_cons"] for r in rows_r]),
            "region_G_avg_median": median_or_none([r["G_avg"] for r in rows_r]),
            "sign_agreement_rate": (sum(1 for r in rows_r if r["sign_agree"]) / len(rows_r))
                                   if rows_r else None,
            "epistasis_avg_median": median_or_none([r["epistasis_avg"] for r in rows_r]),
        }
    dom: dict[str, int] = {}
    for pid in {r["pair_id"] for r in region_rows}:
        rows_p = [r for r in region_rows if r["pair_id"] == pid]
        best = max(rows_p, key=lambda r: abs(r["G_avg"]))
        dom[best["region"]] = dom.get(best["region"], 0) + 1

    stats = {
        "n_pairs": len(pairs),
        "n_differing_positions": len(residue_rows),
        "n_unique_sequences_scored": n_scored,
        "max_abs_original_score_drift": max(score_drift) if score_drift else None,
        "gate": gate,
        "region": region_stats,
        "dominant_region_counts": dom,
        "corr_cavg_sum_reward_gap_pearson": pearson(
            [r["c_avg_sum"] for r in pair_rows], [r["reward_gap"] for r in pair_rows]),
        "corr_cavg_sum_reward_gap_spearman": spearman(
            [r["c_avg_sum"] for r in pair_rows], [r["reward_gap"] for r in pair_rows]),
        "corr_cdrop_cgain_pearson": pearson(c_drop_all, c_gain_all),
    }

    # ------------------------------------------------------------------ write
    def write_csv(path, rows):
        if not rows:
            return
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv(CF / "residue_credit.csv", residue_rows)
    write_csv(CF / "region_credit.csv", region_rows)
    write_csv(CF / "pair_credit_summary.csv", pair_rows)
    (CF / "credit_stats.json").write_text(json.dumps(stats, indent=1))

    # ------------------------------------------------------------------ figs
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [i / 30 for i in range(31)]
    curves = []
    for cons in per_pair_cons:
        if sum(cons) <= 0:
            continue
        srt = sorted(cons, reverse=True)
        tot = sum(srt)
        curves.append([sum(srt[:max(1, int(round(f * len(srt))))]) / tot for f in xs])
    if curves:
        ax.plot(xs, [st.mean(c[i] for c in curves) for i in range(len(xs))], label="mean")
        ax.fill_between(
            xs,
            [sorted(c[i] for c in curves)[len(curves) // 4] for i in range(len(xs))],
            [sorted(c[i] for c in curves)[3 * len(curves) // 4] for i in range(len(xs))],
            alpha=0.25, label="IQR")
    ax.set_xlabel("fraction of differing residues (sorted by c_cons)")
    ax.set_ylabel("cumulative positive-credit mass")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig1_credit_lorenz.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([r["n_diff"] for r in pair_rows], [r["n_eff"] for r in pair_rows], s=12)
    lim = max([r["n_diff"] for r in pair_rows] + [1])
    ax.plot([0, lim], [0, lim], "--", color="gray", lw=1)
    ax.set_xlabel("n_diff")
    ax.set_ylabel("n_eff (c_cons)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ndiff_neff.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for region, color in zip(("cdr1", "cdr2", "cdr3"), ("C0", "C1", "C2")):
        vals = [v for v in c_cons_region[region] if v != 0]
        if vals:
            ax.hist(vals, bins=40, alpha=0.5, label=region, color=color)
    ax.set_xlabel("c_cons (positive only)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig3_region_credit.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(c_drop_all, c_gain_all, s=4, alpha=0.3)
    lo = min(c_drop_all + c_gain_all + [-1])
    hi = max(c_drop_all + c_gain_all + [1])
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("c_drop")
    ax.set_ylabel("c_gain")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cdrop_cgain.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter([r["c_avg_sum"] for r in pair_rows], [r["reward_gap"] for r in pair_rows], s=12)
    r = stats["corr_cavg_sum_reward_gap_pearson"]
    ax.set_xlabel("C_sum (sum c_avg)")
    ax.set_ylabel("reward gap")
    ax.set_title(f"pearson = {r:+.3f}" if r is not None else "")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_rewardgap_vs_credit.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    data = [[t[k] for t in per_pair_topk] for k in ("top10", "top20", "top30", "top50")]
    ax.boxplot(data, tick_labels=["top10", "top20", "top30", "top50"], showfliers=True)
    ax.set_ylabel("positive-credit mass")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_topk_boxplot.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------ doc
    g = stats["gate"]
    md = f"""# Counterfactual Credit Audit（Phase A1/A2，Gate A）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §10–§19
数据：24 train cases / {stats['n_pairs']} pairs；{stats['n_differing_positions']} 个 differing positions；
{n_scored} 条 unique counterfactual sequences + originals（batch scorer + SHA1 cache）。

## 1. Residue credit（双向 intervention）

| 指标 | 数值 |
|---|---|
| c_drop vs c_gain Pearson | {stats['corr_cdrop_cgain_pearson']:+.4f} |
| sign agreement rate (median) | {g['median_sign_agreement_rate']:.3f} |
| positive_fraction (median) | {g['median_positive_fraction']:.3f} |
| original score drift (max abs diff) | {stats['max_abs_original_score_drift']:.2e} |

## 2. Sparsity（Gate A）

| 指标 | median |
|---|---|
| top10% positive-credit mass | {g['median_top10_mass']:.3f} |
| top20% | {g['median_top20_mass']:.3f} |
| **top30%** | **{g['median_top30_mass']:.3f}** |
| top50% | {g['median_top50_mass']:.3f} |
| **n_eff / n_diff** | **{g['median_neff_ratio']:.3f}** |

**Gate A decision: {g['h1']}**
（Strong: top30 >= 0.60 且 n_eff/n_diff <= 0.60；Partial: top30 0.45–0.60；
Weak: top30 < 0.45 且 n_eff/n_diff > 0.75）

## 3. Region credit（A2）

| region | pairs w/ changes | residue c_cons median | G_cons median | G_avg median | sign agree | epistasis (G_avg - sum c_avg) median |
|---|---|---|---|---|---|---|
"""
    for region in ("cdr1", "cdr2", "cdr3"):
        rs = stats["region"][region]
        if rs["n_pairs_with_changes"] == 0:
            continue
        md += (f"| {region} | {rs['n_pairs_with_changes']} | "
               f"{rs['residue_c_cons_median']:.3f} | {rs['region_G_cons_median']:.3f} | "
               f"{rs['region_G_avg_median']:.3f} | {rs['sign_agreement_rate']:.3f} | "
               f"{rs['epistasis_avg_median']:+.3f} |\n")
    md += f"""
Dominant region（|G_avg| 最大）：{json.dumps(stats['dominant_region_counts'])}

## 4. Reward-gap explanation（仅诊断）

corr(C_sum(c_avg), reward_gap)：Pearson {stats['corr_cavg_sum_reward_gap_pearson']:+.3f}，
Spearman {stats['corr_cavg_sum_reward_gap_spearman']:+.3f}（不要求加性）。

## 5. 结论

- 依据 Gate A 决定 Phase C 采用 residue-level sparse CF weighting 还是 region-level /
  temporal 优先（任务书 §16/§33）。
- 图：`runs/next_stage/counterfactual/figures/fig1–fig6`。

*脚本：`scripts/next_build_counterfactuals.py`、`next_score_counterfactuals.py`、
`next_analyze_credit.py`；credit 公式：`src/vhh_rl/credit/credit_metrics.py`*
"""
    (ROOT / "docs/COUNTERFACTUAL_CREDIT_AUDIT.md").write_text(md)
    print(json.dumps(stats, indent=1))
    print("wrote docs/COUNTERFACTUAL_CREDIT_AUDIT.md")


if __name__ == "__main__":
    main()
