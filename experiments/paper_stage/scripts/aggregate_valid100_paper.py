#!/usr/bin/env python
"""Aggregate valid100 evaluations for arbitrary paper-stage arm tags.

usage: aggregate_valid100_paper.py tag1 tag2 ...
Reads  runs/native_pool/eval_summary_<tag>_valid100_shard*.json
Writes runs/paper_stage/valid100/<tag>.json and results/paper_stage/valid100_summary.csv
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/paper_stage/valid100"
RESULTS = ROOT / "results/paper_stage"


def base_case_means() -> dict[str, float]:
    out = {}
    for d in sorted((POOL / "valid100").iterdir()):
        if not d.is_dir():
            continue
        rs = [json.loads(l)["reward_raw"] for l in (d / "metadata.jsonl").open()]
        rs = [r for r in rs if r is not None]
        if rs:
            out[d.name] = st.mean(rs)
    return out


def arm_case_means(tag: str) -> dict[str, float]:
    cases: dict[str, list[float]] = {}
    n_invalid = n_total = 0
    for path in sorted(glob.glob(str(POOL / f"eval_summary_{tag}_valid100_shard*.json"))):
        summary = json.loads(Path(path).read_text())
        for cid, v in summary["cases"].items():
            n_invalid += v["n_invalid"]
            n_total += v["n_samples"]
            if v["reward_mean"] is not None:
                cases.setdefault(cid, []).append(v["reward_mean"])
    return {c: st.mean(v) for c, v in cases.items()}, {
        "invalid_rate": n_invalid / max(1, n_total), "n": n_total}


def paired(delta: list[float]) -> dict:
    d = np.array(delta)
    if len(d) == 0:
        return {}
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d, zero_method="wilcox").pvalue) if any(d != 0) else 1.0
    except Exception:
        p = None
    rng = np.random.default_rng(12345)
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n": int(len(d)), "delta_mean": float(d.mean()),
        "delta_median": float(np.median(d)),
        "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
        "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)),
        "wilcoxon_p": p,
    }


def main() -> None:
    tags = sys.argv[1:]
    base = base_case_means()
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [{
        "arm": "base", "reward_mean": st.mean(base.values()), "delta_mean": 0.0,
        "delta_median": 0.0, "wins": 0, "ties": 100, "losses": 0,
        "ci95_low": 0.0, "ci95_high": 0.0, "wilcoxon_p": None, "invalid_rate": 0.0,
        "n_cases": len(base),
    }]
    for tag in tags:
        means, extra = arm_case_means(tag)
        if not means:
            print(f"[skip] {tag}: no valid100 summaries yet")
            continue
        common = sorted(set(means) & set(base))
        stats = paired([means[c] - base[c] for c in common])
        payload = {"tag": tag, "reward_mean": st.mean(means.values()),
                   "per_case": means, "delta_vs_base": stats, **extra}
        (OUT / f"{tag}.json").write_text(json.dumps(payload, indent=1))
        rows.append({
            "arm": tag, "reward_mean": st.mean(means.values()), **stats,
            "invalid_rate": extra["invalid_rate"], "n_cases": len(common),
        })
        print(f"{tag}: reward={st.mean(means.values()):.3f} "
              f"delta={stats.get('delta_mean', float('nan')):+.3f} "
              f"wins={stats.get('wins')}/{stats.get('n')} p={stats.get('wilcoxon_p')}")
    with (RESULTS / "valid100_summary.csv").open("w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS / 'valid100_summary.csv'}")


if __name__ == "__main__":
    main()
