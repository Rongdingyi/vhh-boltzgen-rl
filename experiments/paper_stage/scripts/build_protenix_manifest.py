#!/usr/bin/env python
"""Paper-stage Step 4: build the 50-case Protenix-v2 independent-refold manifest.

Arms: Base / N3 / CF; 2 sequences per case per arm (same selection rule as the
Boltz-2 validation). Sample ids are underscore-free for the Protenix loader.

Output: runs/paper_stage/structure_protenix/{manifest.csv,manifest.json}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/paper_stage/structure_protenix"
N_CASES = 50
N_SEQ = 2
ARMS = {
    "base": POOL / "valid100",
    "n3": POOL / "valid100_n3_s0350",
    "cf": POOL / "valid100_full_b3_s0500_s0500",
}
TAG = {"base": "b", "n3": "n", "cf": "c"}


def main() -> None:
    valid100 = sorted(json.loads(l)["case_id"]
                      for l in (ROOT / "runs/round1_rl_split/rl_manifest_valid100test.jsonl").open())
    cases = valid100[:N_CASES]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ci, cid in enumerate(cases):
        for arm, pool_dir in ARMS.items():
            seq_rows = sorted((json.loads(l) for l in (pool_dir / cid / "metadata.jsonl").open()),
                              key=lambda r: r["sample_id"])
            seqs = [r["decoded_sequence"] for r in seq_rows
                    if r["reward_raw"] is not None and not r["contains_UNK"]][:N_SEQ]
            assert len(seqs) == N_SEQ, (cid, arm)
            for k, seq in enumerate(seqs):
                rows.append({"case_id": f"px{ci:03d}{TAG[arm]}{k}",
                             "sequence": seq, "arm": arm, "valid100_case": cid, "idx": k})
    with (OUT / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "sequence"])
        w.writeheader()
        for r in rows:
            w.writerow({"case_id": r["case_id"], "sequence": r["sequence"]})
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} rows -> {OUT / 'manifest.csv'}")
    print(f"submit: bash scripts/submit_protenix_v2_array.sh {OUT / 'manifest.csv'} valid100")


if __name__ == "__main__":
    main()
