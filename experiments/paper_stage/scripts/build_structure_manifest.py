#!/usr/bin/env python
"""Paper-stage Step 3: build the 100-case Boltz-2 structure-validation manifest.

Arms: Base / N3 / Shuffle / CF.  Two sequences per case per arm, taken
deterministically from the frozen valid100 pools (first two scored, FR-clean
samples by sample_id order; same generation protocol as the reward evals).

Output: runs/paper_stage/structure_boltz2/{*.cif,*.npz,manifest.csv}
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

OUT = ROOT / "runs/paper_stage/structure_boltz2"
ARMS = {
    "base": POOL / "valid100",
    "n3": POOL / "valid100_n3_s0350",
    "shuffle": POOL / "valid100_full_b2_s0450_s0450",
    "cf": POOL / "valid100_full_b3_s0500_s0500",
}
TAG = {"base": "b", "n3": "n", "shuffle": "s", "cf": "c"}
N_SEQ = 2


def main() -> None:
    valid100 = []
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl").open():
        valid100.append(json.loads(line)["case_id"])
    valid100 = sorted(valid100)
    print(f"{len(valid100)} valid100 cases")
    loaded = {c.case_id: c for c in load_cases(valid100)}
    missing = [c for c in valid100 if c not in loaded]
    assert not missing, missing

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cid in enumerate(valid100):
        case = loaded[cid]
        for arm, pool_dir in ARMS.items():
            d = pool_dir / cid
            seq_rows = sorted(
                (json.loads(l) for l in (d / "metadata.jsonl").open()),
                key=lambda r: r["sample_id"],
            )
            seqs = [r["decoded_sequence"] for r in seq_rows
                    if r["reward_raw"] is not None and not r["contains_UNK"]][:N_SEQ]
            assert len(seqs) >= N_SEQ, (cid, arm, len(seqs))
            for k in range(N_SEQ):
                sid = f"p{ci:03d}{TAG[arm]}{k}"
                job = Job(sample_id=sid, case_id=cid, label=arm, beta="", rank=str(k),
                          sequence=seqs[k])
                write_job(job, case, OUT, strip_design_sidechains=True)
                rows.append({"sample_id": sid, "case_id": cid, "arm": arm,
                             "idx": k, "sequence": seqs[k]})
    with (OUT / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "case_id", "arm", "idx", "sequence"])
        w.writeheader()
        w.writerows(rows)
    print(f"DONE {len(rows)} inputs -> {OUT}")


if __name__ == "__main__":
    main()
