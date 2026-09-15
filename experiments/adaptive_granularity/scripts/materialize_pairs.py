#!/usr/bin/env python
"""Phase A2: materialize filtered pair files per variant (task book §23/§71).

The trainer must never see a pair whose variant weight is missing: every
eligible pair appears exactly once in the variant's pair file, ineligible
pairs are dropped (not silently skipped later).  Also emits the fixed-4
pilot subsets and the eligible-CF control pair file (§45/§72).
"""
from __future__ import annotations
import json
from collections import defaultdict

import _common as C  # noqa: E402

VARIANTS = ["no_floor", "strict_consensus", "strict_region",
            "adaptive", "adaptive_shuffle"]
CONTROL = "adaptive_eligible_cf"


def main() -> None:
    pairs = C.load_pairs()
    ag = C.load_ag_weights()
    C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for variant in VARIANTS + [CONTROL]:
        weight_key = "adaptive" if variant == CONTROL else variant
        eligible, pilot, by_case = [], [], defaultdict(int)
        for pair in pairs:
            pid = C.pair_id(pair)
            entry = ag["pairs"].get(pid)
            if entry is None or weight_key not in entry:
                continue
            eligible.append(pair)
            by_case[pair["case_id"]] += 1
            if pair["case_id"] in C.PILOT_TRAIN_CASES:
                pilot.append(pair)
        for suffix, rows in (("", eligible), ("_pilot4", pilot)):
            path = C.WEIGHTS_DIR / f"pairs_{variant}{suffix}.jsonl"
            path.write_text("".join(json.dumps(p) + "\n" for p in rows))
        pilot_counts = {case: by_case.get(case, 0) for case in C.PILOT_TRAIN_CASES}
        missing = [case for case, n in pilot_counts.items() if n == 0]
        report[variant] = {
            "eligible_pairs": len(eligible),
            "eligible_cases": len(by_case),
            "pilot4_pairs": len(pilot),
            "pilot4_case_counts": pilot_counts,
            "pilot4_missing_cases": missing,
        }
        if missing:
            print(f"[warn] {variant}: pilot cases without eligible pairs: {missing}")
    (C.WEIGHTS_DIR / "materialize_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
