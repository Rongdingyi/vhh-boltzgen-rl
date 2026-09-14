#!/usr/bin/env python
"""Build refold inputs for M2 and M3 (unified panel candidates, frozen backbone).

Same 20 cases as the IF-RL refold5 study, 4 sequences per case per method,
so the refold table can include G0/M1/M2/M3/IF-RL on one common reference.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
GUID = Path("/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance")
UNI = GUID / "outputs/unified_valid100__g1_g2_g6__m1_m2_m3__g0_512_v1"
REFOLD5 = ROOT / "runs/round1_rl_split/refold5_design/manifest.csv"
OUT = ROOT / "runs/report_unified/refold_m2m3_design"
sys.path.insert(0, str(GUID / "scripts/cdr_all"))

from build_refold_inputs import Job, write_job  # noqa: E402
from common_cdr_all import load_cases  # noqa: E402

N_SEQ = 4


def main() -> None:
    cases = []
    for r in csv.DictReader(REFOLD5.open()):
        if r["case_id"] not in cases:
            cases.append(r["case_id"])
    loaded = {c.case_id: c for c in load_cases(cases)}

    # collect unified panel sequences per method/case (first 4 in file order)
    seqs: dict[tuple[str, str], list[str]] = {}
    for r in csv.DictReader((UNI / "candidates.csv").open()):
        if r["method"] in ("M2", "M3"):
            seqs.setdefault((r["method"], r["case_id"]), []).append(r["sequence"])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for ci, cid in enumerate(cases):
        case = loaded[cid]
        for method, tag in (("M2", "2"), ("M3", "3")):
            pool = seqs[(method, cid)][:N_SEQ]
            assert len(pool) == N_SEQ, (method, cid, len(pool))
            for k, seq in enumerate(pool):
                sid = f"u{ci:02d}{tag}{k}"
                job = Job(sample_id=sid, case_id=cid, label={"M2": "m2", "M3": "m3"}[method],
                          beta="", rank=str(k), sequence=seq)
                write_job(job, case, OUT, strip_design_sidechains=True)
                rows.append({"sample_id": sid, "case_id": cid, "arm": method, "idx": k})
                n += 1
    with (OUT / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "case_id", "arm", "idx"])
        w.writeheader()
        w.writerows(rows)
    print(f"DONE {n} inputs -> {OUT}")


if __name__ == "__main__":
    main()
