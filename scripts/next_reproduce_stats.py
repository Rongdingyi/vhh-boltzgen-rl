#!/usr/bin/env python
"""Phase A0: reproduce the round-1 diagnostic statistics (task book §7).

Outputs:
  runs/next_stage/repro/pair_stats.json
  runs/next_stage/repro/refold_correlations.csv
  docs/NEXT_STAGE_DIAGNOSTIC_REPRO.md
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.pair_analysis import hamming_stats  # noqa: E402

POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage/repro"


def load_manifest() -> dict:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        cases[row["case_id"]] = row
    return cases


def load_pool_sequences(pool_dir: Path) -> tuple[dict[str, str], dict[str, float]]:
    seqs, rewards = {}, {}
    for meta in pool_dir.glob("*/metadata.jsonl"):
        for line in meta.open():
            r = json.loads(line)
            seqs[r["sample_id"]] = r["decoded_sequence"]
            if r["reward_raw"] is not None:
                rewards[r["decoded_sequence"]] = r["reward_raw"]
    return seqs, rewards


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
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_manifest()
    pairs = [json.loads(l) for l in (POOL / "pairs_train.jsonl").open()]

    # ---------- pair stats
    seqs, _ = load_pool_sequences(POOL / "train")
    design = {c: tuple(cases[c]["design_positions"]) for c in cases}
    stats = hamming_stats(pairs, seqs, design)
    (OUT / "pair_stats.json").write_text(json.dumps(stats, indent=1))
    print("pair_stats:", json.dumps(stats, indent=1))

    # ---------- refold correlations
    refold = list(csv.DictReader((POOL / "native_refold_results_per_sample.csv").open()))
    refold_man = list(csv.DictReader((POOL / "native_refold_design/manifest.csv").open()))
    arm_pool = {
        "base": POOL / "valid100",
        "n1": POOL / "valid100_n1_s0250",
        "n2": POOL / "valid100_n2_s0500",
        "n3": POOL / "valid100_n3_s0350",
        "n4": POOL / "valid100_n4_s0500",
    }
    rewards = {}
    for arm, d in arm_pool.items():
        _, rw = load_pool_sequences(d)
        rewards[arm] = rw

    refold_by_id = {r["sample_id"]: r for r in refold}
    rows = []
    for m in refold_man:
        sid, arm, seq = m["sample_id"], m["arm"], m["sequence"]
        if sid not in refold_by_id:
            continue
        rr = refold_by_id[sid]
        reward = rewards[arm].get(seq)
        if reward is None:
            continue
        rows.append({
            "arm": arm, "case_id": m["case_id"], "reward": reward,
            "cdr_rmsd": float(rr["cdr_rmsd"]), "cdr3_rmsd": float(rr["cdr3_rmsd"]),
            "plddt_cdr": float(rr["plddt_cdr"]), "cdr_recovery": float(rr["cdr_recovery"]),
        })
    metrics = ["cdr_rmsd", "cdr3_rmsd", "plddt_cdr", "cdr_recovery"]
    out_rows = []
    for group in ["all"] + sorted({r["arm"] for r in rows}):
        sub = rows if group == "all" else [r for r in rows if r["arm"] == group]
        for met in metrics:
            xs = [r["reward"] for r in sub]
            ys = [r[met] for r in sub]
            row = {"group": group, "metric": met, "n": len(sub),
                   "pearson": pearson(xs, ys), "spearman": spearman(xs, ys)}
            out_rows.append(row)
    with (OUT / "refold_correlations.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["group", "metric", "n", "pearson", "spearman"])
        w.writeheader()
        w.writerows(out_rows)
    print("refold correlations:")
    for r in out_rows:
        if r["group"] in ("all", "n3"):
            print(f"  {r['group']:5s} {r['metric']:12s} n={r['n']:3d} "
                  f"pearson={r['pearson']:+.3f} spearman={r['spearman']:+.3f}")

    # ---------- markdown
    md = f"""# 下一阶段诊断复现（Phase A0）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §2/§7
数据：`runs/native_pool/pairs_train.jsonl`、`runs/native_pool/train/*/metadata.jsonl`、
`runs/native_pool/native_refold_results_per_sample.csv`、`native_refold_design/manifest.csv`

## 1. Pair Hamming 统计（192 pairs）

| 统计量 | 复现值 |
|---|---|
| n_pairs | {stats['n_pairs']} |
| 总差异位点 | {stats['total_differing_positions']} |
| Hamming mean / median | {stats['hamming_mean']:.2f} / {stats['hamming_median']:.1f} |
| Hamming min / max | {stats['hamming_min']} / {stats['hamming_max']} |
| Hamming p10/p25/p75/p90 | {stats['hamming_p10']:.1f} / {stats['hamming_p25']:.1f} / {stats['hamming_p75']:.1f} / {stats['hamming_p90']:.1f} |
| Pearson(Hamming, reward gap) | {stats['pearson_hamming_gap']:+.4f} |
| Spearman(Hamming, reward gap) | {stats['spearman_hamming_gap']:+.4f} |

## 2. Refold 相关性（400 samples）

| group | metric | n | Pearson | Spearman |
|---|---|---|---|---|
"""
    for r in out_rows:
        md += (f"| {r['group']} | {r['metric']} | {r['n']} | "
               f"{r['pearson']:+.3f} | {r['spearman']:+.3f} |\n")
    md += """
## 3. 结论

- 一个 preference pair 平均约 `%.1f` 个 CDR 残基不同，但差异数与 reward gap
  基本无关（Pearson %+.3f）——credit 分配问题成立。
- classifier reward 与 refold 结构指标几乎不相关（CDR RMSD Pearson %+.3f），
  与 pLDDT 呈明显反向——单靠 sequence reward 无法提供结构自洽性信号。

*复现脚本：`scripts/next_reproduce_stats.py`*
""" % (stats['hamming_mean'], stats['pearson_hamming_gap'],
       next((r['pearson'] for r in out_rows if r['group'] == 'all' and r['metric'] == 'cdr_rmsd'), 0.0))
    (ROOT / "docs/NEXT_STAGE_DIAGNOSTIC_REPRO.md").write_text(md)
    print("wrote docs/NEXT_STAGE_DIAGNOSTIC_REPRO.md")


if __name__ == "__main__":
    main()
