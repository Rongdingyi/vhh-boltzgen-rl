#!/usr/bin/env python
"""Gate 1 report -> docs/branch_distill/GATE1_BRANCHING.md (§27/§63)."""
from __future__ import annotations

import csv
import json
import statistics as st

import _common as C  # noqa: E402


def main() -> None:
    gate = json.loads((C.GATE1_DIR / "gate1.json").read_text())
    rows = list(csv.DictReader((C.GATE1_DIR / "group_metrics.csv").open()))
    by_progress = {}
    for row in rows:
        by_progress.setdefault(float(row["progress"]), []).append(row)

    def med(rs, key):
        vals = [float(r[key]) for r in rs if r[key] not in ("", "None")]
        return st.median(vals) if vals else None

    lines = ["# Gate 1 — Same-state sibling branching", "",
             "## 1. K=8 是否产生不同序列？", ""]
    for progress in sorted(by_progress):
        rs = by_progress[progress]
        lines.append(f"- progress {progress:.2f}: unique median "
                     f"{med(rs, 'n_unique_valid_sequences')}, "
                     f"teacher-vs-median hamming median "
                     f"{med(rs, 'teacher_vs_median_hamming')}")
    lines += ["", "## 2. reward spread", ""]
    for progress in sorted(by_progress):
        rs = by_progress[progress]
        lines.append(f"- progress {progress:.2f}: reward_std median "
                     f"{med(rs, 'reward_std'):.3f}, best_minus_median median "
                     f"{med(rs, 'best_minus_median'):.3f}")
    lines += ["", "## 3. 最早可用 progress", "",
              f"selected_progress = {gate['selected_progress']}", "",
              "## 4. teacher vs median 平均差几个设计位点", ""]
    if gate["selected_progress"] is not None:
        rs = by_progress[gate["selected_progress"]]
        lines.append(f"{med(rs, 'teacher_vs_median_hamming')} (median across groups)")
    lines += ["", "## 5. validity", ""]
    for progress in sorted(by_progress):
        rs = by_progress[progress]
        lines.append(f"- progress {progress:.2f}: valid_rate median "
                     f"{med(rs, 'valid_rate'):.3f}")
    lines += ["", "## 6. FAIL 的直接原因", ""]
    if not gate["pass"]:
        for progress, verdict in sorted(gate["progress"].items()):
            lines.append(f"- progress {progress}: groups pass "
                         f"{verdict['n_groups_pass']}/{verdict['n_groups']}, "
                         f"median valid {verdict['median_valid_rate']:.3f}, "
                         f"median unique {verdict['median_unique']}, "
                         f"median best-minus-median "
                         f"{verdict['median_best_minus_median']:.3f}")
    else:
        lines.append("- Gate 1 PASS")
    lines += ["", f"protocol_sha256: `{gate['protocol_sha256']}`", ""]
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "GATE1_BRANCHING.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
