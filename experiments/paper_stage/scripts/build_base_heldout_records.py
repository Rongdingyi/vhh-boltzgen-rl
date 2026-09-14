#!/usr/bin/env python
"""Build + score the base held-out pool with the second reward (ESM-C cdr_pll).

Writes runs/paper_stage/second_reward/base_heldout.jsonl and
base_heldout_scored.jsonl (via esmc_reward.py scoring pass).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
SEC = ROOT / "runs/paper_stage/second_reward"
POOL = ROOT / "runs/native_pool"


def main() -> None:
    rows = []
    for meta in sorted((POOL / "heldout").glob("*/metadata.jsonl")):
        for line in meta.open():
            r = json.loads(line)
            rows.append({
                "key": f"{r['case_id']}:{r['sample_index']}",
                "case_id": r["case_id"],
                "sample_index": r["sample_index"],
                "sequence": r["decoded_sequence"],
                "contains_invalid": bool(r["contains_UNK"]),
                "fr_mismatch": int(r["FR_mismatch_count"]),
            })
    with (SEC / "base_heldout.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} base held-out records")


if __name__ == "__main__":
    main()
