#!/usr/bin/env python
"""Build refold inputs for native post-training arms (task book §58).

20 valid100 cases x 5 arms (base,n1,n2,n3,n4) x 4 sequences = 400 inputs.
Reuses the vhh_esmc_guidance write_job machinery (official CIF rewriting +
read-back verification). Sample ids are underscore-free (required by the
dataset loader).
"""
from __future__ import annotations

import csv
import json
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
    "base": POOL / "valid100",
    "n1": POOL / "valid100_n1_s0250",
    "n2": POOL / "valid100_n2_s0500",
    "n3": POOL / "valid100_n3_s0350",
    "n4": POOL / "valid100_n4_s0500",
}
OUT = POOL / "native_refold_design"
N_CASES = 20
N_SEQ = 4
TAG = {"base": "b", "n1": "1", "n2": "2", "n3": "3", "n4": "4"}


def main() -> None:
    # pick 20 cases by base difficulty spread? simple: first 20 sorted by base mean
    import statistics as st
    base_rows = {}
    for d in sorted((POOL / "valid100").iterdir()):
        if not d.is_dir():
            continue
        rows = [json.loads(l) for l in (d / "metadata.jsonl").open()]
        rs = [r["reward_raw"] for r in rows if r["reward_raw"] is not None]
        if rs:
            base_rows[d.name] = st.mean(rs)
    ranked = sorted(base_rows, key=base_rows.get)
    cases = ranked[:10] + ranked[-10:]  # 10 hardest + 10 easiest
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
