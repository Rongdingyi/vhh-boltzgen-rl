#!/usr/bin/env python
"""Build refold inputs for the Phase C arms (b1/b2/b3) on the SAME 20 valid100
cases and 4-sequence-per-case protocol as the round-1 native refold, so results
are directly paired (task book §42).

Output: runs/native_pool/native_refold_next (240 inputs: b1/b2/b3 x 20 x 4)
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance/scripts/cdr_all")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(GUID))

from build_refold_inputs import Job, write_job  # noqa: E402
from common_cdr_all import load_cases  # noqa: E402

ARMS = {
    "b1": POOL / "valid100_full_b1_s0450_s0450",
    "b2": POOL / "valid100_full_b2_s0450_s0450",
    "b3": POOL / "valid100_full_b3_s0500_s0500",
}
OUT = POOL / "native_refold_next"
N_CASES = 20
N_SEQ = 4
TAG = {"b1": "5", "b2": "6", "b3": "7"}


def main() -> None:
    base_rows = {}
    for d in sorted((POOL / "valid100").iterdir()):
        if not d.is_dir():
            continue
        rows = [json.loads(l) for l in (d / "metadata.jsonl").open()]
        rs = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        if rs:
            base_rows[d.name] = st.mean(rs)
    ranked = sorted(base_rows, key=base_rows.get)
    cases = ranked[:10] + ranked[-10:]  # identical selection to round 1
    print("cases:", cases)

    loaded = {c.case_id: c for c in load_cases(cases)}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for ci, cid in enumerate(cases):
        case = loaded[cid]
        for arm, pool_dir in ARMS.items():
            d = pool_dir / cid
            seq_rows = [json.loads(l) for l in (d / "metadata.jsonl").open()]
            seqs = [r["decoded_sequence"] for r in seq_rows
                    if r["reward_raw"] is not None and not r["contains_UNK"]][:N_SEQ]
            assert len(seqs) >= N_SEQ, (cid, arm, len(seqs))
            for k in range(N_SEQ):
                sid = f"n{ci:02d}{TAG[arm]}{k}"
                job = Job(sample_id=sid, case_id=cid, label=arm, beta="", rank=str(k),
                          sequence=seqs[k])
                write_job(job, case, OUT, strip_design_sidechains=True)
                rows.append({"sample_id": sid, "case_id": cid, "arm": arm, "idx": k})
                n += 1
    with (OUT / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "case_id", "arm", "idx"])
        w.writeheader()
        w.writerows(rows)
    print(f"DONE {n} inputs -> {OUT}")


if __name__ == "__main__":
    main()
