#!/usr/bin/env python
"""Aggregate valid100 evaluations for the next-stage arms vs base / N3.

Reads eval_summary_<arm>_valid100_shard*.json + base and n3 pools, computes
per-case mean rewards, paired deltas, win counts and Wilcoxon p-value.
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/next_stage"

ARMS = {
    "full_b1_s0450": ("B1 random_sparse", "valid100_full_b1_s0450_s0450"),
    "full_b2_s0450": ("B2 shuffle", "valid100_full_b2_s0450_s0450"),
    "full_b3_s0500": ("B3 cf", "valid100_full_b3_s0500_s0500"),
}
BASE_ARM = "base"
N3_ARM = "n3"


def pool_case_means(pool_dir: Path) -> dict[str, float]:
    out = {}
    for d in sorted(pool_dir.iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            out[d.name] = st.mean(rs)
    return out


def summarize(tag: str, label: str, case_means: dict[str, float],
              base: dict[str, float], n3: dict[str, float]) -> dict:
    common = sorted(set(case_means) & set(base))
    deltas = [case_means[c] - base[c] for c in common]
    wins = sum(1 for d in deltas if d > 0)
    p_value = None
    try:
        from scipy.stats import wilcoxon
        if len(deltas) >= 5 and any(d != 0 for d in deltas):
            p_value = float(wilcoxon(deltas, zero_method="wilcox").pvalue)
    except Exception:
        pass
    d_n3 = [case_means[c] - n3[c] for c in sorted(set(case_means) & set(n3))]
    return {
        "arm": label, "tag": tag, "n_cases": len(common),
        "reward_mean": st.mean(case_means[c] for c in common),
        "delta_vs_base_mean": st.mean(deltas) if deltas else None,
        "delta_vs_base_median": st.median(deltas) if deltas else None,
        "wins_vs_base": wins,
        "wilcoxon_p": p_value,
        "delta_vs_n3_mean": st.mean(d_n3) if d_n3 else None,
        "per_case": {c: {"arm": case_means[c], "base": base[c],
                         "delta": case_means[c] - base[c]} for c in common},
    }


def main() -> None:
    base = pool_case_means(POOL / "valid100")
    n3 = pool_case_means(POOL / "valid100_n3_s0350")
    base_summary = summarize(BASE_ARM, "native base", base, base, n3)
    results = {"base": base_summary}
    for tag, (label, pool_name) in ARMS.items():
        pool_dir = POOL / pool_name
        if not pool_dir.is_dir():
            print(f"[skip] {tag}: {pool_dir} missing")
            continue
        results[tag] = summarize(tag, label, pool_case_means(pool_dir), base, n3)
    (OUT / "valid100_arms_summary.json").write_text(json.dumps(results, indent=1))

    md = "# valid100 对照（Phase C，100 cases x 8 samples）\n\n"
    md += ("| arm | reward mean | Δ vs base | median Δ | wins vs base | Wilcoxon p | Δ vs N3 |\n"
           "|---|---|---|---|---|---|---|\n")
    for tag, r in results.items():
        from math import isnan
        p = "n/a" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.2e}"
        d = r["delta_vs_base_mean"]
        dn3 = r["delta_vs_n3_mean"]
        md += (f"| {r['arm']} | {r['reward_mean']:.3f} | "
               f"{d:+.3f} | {r['delta_vs_base_median']:+.3f} | {r['wins_vs_base']}/{r['n_cases']} | "
               f"{p} | {dn3:+.3f} |\n")
    (OUT / "valid100_arms_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
