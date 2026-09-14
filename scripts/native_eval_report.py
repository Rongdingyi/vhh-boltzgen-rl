#!/usr/bin/env python
"""Aggregate native arm evaluations (task book §56/§86)."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"


def load(split_dir: Path) -> dict[str, dict]:
    per = {}
    for f in sorted(split_dir.glob("*/metadata.jsonl")):
        cid = f.parent.name
        rows = [json.loads(l) for l in f.open()]
        rs = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        per[cid] = {
            "mean": st.mean(rs) if rs else None,
            "median": st.median(rs) if rs else None,
            "best": max(rs) if rs else None,
            "n": len(rows),
            "invalid": sum(r["contains_UNK"] for r in rows),
            "fr_mismatch": sum(r["FR_mismatch_count"] > 0 for r in rows),
            "unique": len({r["decoded_sequence"] for r in rows}) / max(len(rows), 1),
        }
    return per


def bootstrap_ci(deltas, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array(deltas)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "valid100"
    base = load(POOL / which)
    base = {c: v for c, v in base.items() if v["mean"] is not None}
    arms = sorted({p.name.split("_")[1] + "_" + p.name.split("_")[2]
                   for p in POOL.glob(f"{which}_*") if p.is_dir()})
    rows = []
    summary = {}
    for arm in arms:
        d = POOL / f"{which}_{arm}"
        if not d.is_dir():
            continue
        per = load(d)
        common = sorted(set(per) & set(base))
        if not common:
            continue
        deltas = [per[c]["mean"] - base[c]["mean"] for c in common]
        inv = sum(per[c]["invalid"] for c in common)
        fr = sum(per[c]["fr_mismatch"] for c in common)
        n_samp = sum(per[c]["n"] for c in common)
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(deltas).pvalue)
        except Exception:
            p = None
        lo, hi = bootstrap_ci(deltas) if len(deltas) > 1 else (deltas[0], deltas[0])
        summary[arm] = {
            "n_cases": len(common),
            "mean_reward": st.mean(per[c]["mean"] for c in common),
            "base_mean_reward": st.mean(base[c]["mean"] for c in common),
            "mean_delta": st.mean(deltas),
            "median_delta": st.median(deltas),
            "wins": sum(x > 0 for x in deltas),
            "ties": sum(x == 0 for x in deltas),
            "losses": sum(x < 0 for x in deltas),
            "wilcoxon_p": p,
            "bootstrap_ci95": [lo, hi],
            "invalid_rate": inv / n_samp,
            "fr_mismatch_rate": fr / n_samp,
            "mean_unique_rate": st.mean(per[c]["unique"] for c in common),
        }
        for c in common:
            rows.append({
                "case_id": c,
                "base_reward": base[c]["mean"],
                f"{arm}_reward": per[c]["mean"],
                f"{arm}_delta": per[c]["mean"] - base[c]["mean"],
            })
    out_dir = POOL
    csv_path = out_dir / f"eval_{which}_per_case.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    (out_dir / f"eval_{which}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"per-case -> {csv_path}")


if __name__ == "__main__":
    main()
