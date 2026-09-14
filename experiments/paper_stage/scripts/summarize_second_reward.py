#!/usr/bin/env python
"""Second-reward pipeline 3/3b: summarize held-out ESM-C rewards per arm.

Inputs : runs/paper_stage/second_reward/{base_heldout.jsonl,<tag>_heldout_scored.jsonl}
Base   : the frozen base held-out pool scored with ESM-C (same generation seeds).
Output : runs/paper_stage/second_reward/heldout_summary.json + .md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SEC = ROOT / "runs/paper_stage/second_reward"


def load_scored(path: Path) -> dict[tuple[str, int], float]:
    out = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("cdr_pll") is not None and not r.get("contains_invalid"):
            out[(r["case_id"], r["sample_index"])] = float(r["cdr_pll"])
    return out


def case_means(scored: dict[tuple[str, int], float]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for (cid, _i), v in scored.items():
        acc.setdefault(cid, []).append(v)
    return {c: st.mean(v) for c, v in acc.items()}


def paired(delta: list[float]) -> dict:
    import numpy as np
    d = np.array(delta)
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d, zero_method="wilcox").pvalue) if any(d != 0) else 1.0
    except Exception:
        p = None
    return {"n": len(d), "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "wilcoxon_p": p}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True,
                        help="arm tags with <tag>_heldout_scored.jsonl")
    args = parser.parse_args()
    base = case_means(load_scored(SEC / "base_heldout_scored.jsonl"))
    result = {"base_cdr_pll_mean": st.mean(base.values()), "arms": {}}
    md = "# Second-reward (VHH-ESM-C CDR PLL) held-out comparison\n\n"
    md += f"Base: cdr_pll mean = {st.mean(base.values()):.4f}\n\n"
    md += "| arm | cdr_pll mean | Δ vs base | wins | Wilcoxon p |\n|---|---|---|---|---|\n"
    for tag in args.tags:
        scored = load_scored(SEC / f"{tag}_heldout_scored.jsonl")
        cm = case_means(scored)
        common = sorted(set(cm) & set(base))
        delta = [cm[c] - base[c] for c in common]
        stats = paired(delta)
        result["arms"][tag] = {"cdr_pll_mean": st.mean(cm.values()), **stats}
        p = "n/a" if stats["wilcoxon_p"] is None else f"{stats['wilcoxon_p']:.3g}"
        md += (f"| {tag} | {st.mean(cm.values()):.4f} | {stats['delta_mean']:+.4f} | "
               f"{stats['wins']}/{stats['n']} | {p} |\n")
    # CF vs Uniform if both present
    if "sr_uniform" in result["arms"] and "sr_cf" in result["arms"]:
        u = case_means(load_scored(SEC / "sr_uniform_heldout_scored.jsonl"))
        c = case_means(load_scored(SEC / "sr_cf_heldout_scored.jsonl"))
        common = sorted(set(u) & set(c))
        delta = [c[x] - u[x] for x in common]
        stats = paired(delta)
        result["cf_vs_uniform"] = stats
        md += (f"\n**CF vs Uniform (second reward)**: delta mean {stats['delta_mean']:+.4f}, "
               f"wins {stats['wins']}/{stats['n']}, p={stats['wilcoxon_p']}\n")
    (SEC / "heldout_summary.json").write_text(json.dumps(result, indent=1))
    (SEC / "heldout_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
