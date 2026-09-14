#!/usr/bin/env python
"""Recompute Gate B / write CF_OPSD_TARGET_PROBE.md from existing probe CSVs.

Useful when the probe's reporting tail fails or when Gate B fails for all
radii (no controls are produced in that case).
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
OUT = ROOT / "runs/cf_opsd/target_probe"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_TARGET_PROBE.md"
RADII = (0.25, 0.50, 1.00)


def _f(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def main() -> None:
    rows = [{k: _f(v) for k, v in r.items()}
            for r in csv.DictReader((OUT / "target_metrics.csv").open())]
    radius_rows = []
    for radius in RADII:
        rr = [r for r in rows if r["radius"] == radius and r["control"] == "cf"]
        if not rr:
            continue
        gains = [r["target_minus_anchor"] for r in rr if r["target_minus_anchor"] is not None]
        matches = [r["credit_match"] for r in rr if r["credit_match"] is not None]
        radius_rows.append({
            "radius": radius, "n": len(rr),
            "invalid_rate": sum(1 for r in rr if r["invalid"]) / len(rr),
            "fr_mismatch_total": int(sum(r["fr_mismatch"] for r in rr)),
            "median_target_minus_anchor": st.median(gains) if gains else None,
            "positive_cases": sum(1 for g in gains if g > 0),
            "median_credit_match": st.median(matches) if matches else None,
            "median_third_aa_rate": st.median([r["third_aa_rate"] for r in rr
                                               if r["third_aa_rate"] is not None]),
            "median_actual_rms": st.median([r["actual_rms"] for r in rr]),
        })
    gate = {}
    for r in radius_rows:
        gate[str(r["radius"])] = bool(
            r["invalid_rate"] <= 0.05 and r["fr_mismatch_total"] == 0
            and (r["median_target_minus_anchor"] or 0) > 0
            and (r["median_credit_match"] or 0) >= 0.50
            and r["positive_cases"] >= 3)
    passing = [r["radius"] for r in radius_rows if gate[str(r["radius"])]]
    selected = min(passing) if passing else None
    query = json.loads((ROOT / "runs/cf_opsd/query_probe.json").read_text())
    summary = {"q_star_progress": query["q_star_progress"], "query_status": query["status"],
               "radii": radius_rows, "gate_b": gate, "selected_radius": selected}
    (OUT / "gate_b.json").write_text(json.dumps(summary, indent=1, default=str))
    (OUT / "radius_summary.csv").write_text(
        "radius,n,invalid_rate,fr_mismatch_total,median_target_minus_anchor,"
        "positive_cases,median_credit_match,median_third_aa_rate,median_actual_rms\n"
        + "\n".join(
            f"{r['radius']},{r['n']},{r['invalid_rate']},{r['fr_mismatch_total']},"
            f"{r['median_target_minus_anchor']},{r['positive_cases']},"
            f"{r['median_credit_match']},{r['median_third_aa_rate']},{r['median_actual_rms']}"
            for r in radius_rows) + "\n")

    radius_table = "\n".join(
        f"| {r['radius']} | {fmt(r['median_target_minus_anchor'])} | {r['positive_cases']}/4 | "
        f"{fmt(r['median_credit_match'], '.2f')} | {r['invalid_rate']:.2f} | "
        f"{r['fr_mismatch_total']} | {'PASS' if gate[str(r['radius'])] else 'FAIL'} |"
        for r in radius_rows)
    probe_section = ((ROOT / "runs/cf_opsd/query_probe_section.md").read_text()
                     if (ROOT / "runs/cf_opsd/query_probe_section.md").is_file() else "")
    doc = f"""# CF-OPSD Target Probe

{probe_section}

## Target construction（q* = {query['q_star_progress']:.0%}）

### Radius sweep（CF target）

| rho (A) | median target-anchor | positive cases | median credit match | invalid | FR mismatch | Gate B |
|---|---|---|---|---|---|---|
{radius_table}

- selected radius：**{selected}**
- Gate B：**{'PASS' if selected is not None else 'FAIL'}**

### 失败模式

- rho=0.25 Å：小位移不足以翻转 credited residue 的硬解码身份（credit match ≈ 0），
  且 1/4 anchor 本身无法 decode（invalid 25%）。
- rho=0.5/1.0 Å：位移跨过 hard-decode 边界后目标整体 decode 失败（invalid 100%）。
- 结论：counterfactual sequence credit 不能稳定转换为 bounded atom14 geometric
  target（§32/§73/§74 预警的失败模式）→ 按任务书停止 CF-OPSD。
"""
    DOC.write_text(doc)
    print(json.dumps({k: v for k, v in summary.items() if k != "radii"}, indent=1, default=str))
    print(f"wrote {DOC}")


def fmt(v, spec="+.3f"):
    return "n/a" if v is None else format(v, spec)


if __name__ == "__main__":
    main()
