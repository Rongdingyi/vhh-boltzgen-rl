#!/usr/bin/env python
"""Phase A2 gate: validate AG weights + pair materialization (task book §17/§71)."""
from __future__ import annotations
import json
import sys

import _common as C  # noqa: E402
from vhh_rl.credit.adaptive_granularity import validate_weights  # noqa: E402

VARIANTS = ["no_floor", "strict_consensus", "strict_region",
            "adaptive", "adaptive_shuffle"]


def main() -> None:
    ag = C.load_ag_weights()
    residue = C.load_residue_rows()
    pairs = C.load_pairs()
    failures: list[str] = []
    stats = {}
    for variant in VARIANTS:
        n_eligible = n_checked = 0
        for pair in pairs:
            pid = C.pair_id(pair)
            entry = ag["pairs"].get(pid)
            rows = residue.get(pid)
            if entry is None or rows is None:
                continue
            diff = [r["position"] for r in rows]
            weights = entry.get(variant)
            if weights is None:
                continue
            n_eligible += 1
            parsed = {int(k): float(v) for k, v in weights.items()}
            ok, reason = validate_weights(parsed, diff)
            n_checked += 1
            if not ok:
                failures.append(f"{pid}:{variant}: {reason}")
        pair_file = C.WEIGHTS_DIR / f"pairs_{variant}.jsonl"
        file_rows = [json.loads(l) for l in pair_file.open()] if pair_file.is_file() else []
        file_expect = sum(1 for p in pairs
                          if ag["pairs"].get(C.pair_id(p), {}).get(variant) is not None)
        stats[variant] = {"eligible": n_eligible, "checked": n_checked,
                          "pair_file_rows": len(file_rows),
                          "pair_file_expected": file_expect}
        if len(file_rows) != file_expect:
            failures.append(f"{variant}: pair file rows {len(file_rows)} != {file_expect}")
        for row in file_rows:
            pid = C.pair_id(row)
            entry = ag["pairs"].get(pid, {})
            if variant not in entry:
                failures.append(f"{variant}: {pid} in pair file without weights")
    # current CF baseline is untouched
    current = C.load_current_weights()
    cf_pairs = sum(1 for p in pairs if "cf" in current["pairs"].get(C.pair_id(p), {}))
    stats["current_cf"] = {"pairs_with_cf_weights": cf_pairs}
    if cf_pairs != len(pairs):
        failures.append(f"current CF weights missing for {len(pairs) - cf_pairs} pairs")
    payload = {"ok": not failures, "failures": failures, "stats": stats}
    (C.WEIGHTS_DIR / "validation_summary.json").write_text(json.dumps(payload, indent=1))
    if failures:
        print(json.dumps(payload, indent=1))
        sys.exit(1)
    print(json.dumps({"ok": True, "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
