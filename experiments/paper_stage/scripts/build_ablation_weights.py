#!/usr/bin/env python
"""Paper-stage Step 2: build Diff-only / Drop-only / Gain-only weights.

Reads  runs/next_stage/counterfactual/residue_credit.csv   (c_drop/c_gain/c_cons)
Writes runs/paper_stage/weights/ablation_weights.json
Schema matches runs/next_stage/weights/residue_weights.json so the same
weighted-DPO trainer can load it with --variant {diff_only,drop_only,gain_only}.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.controls import cf_weights  # noqa: E402

CF = ROOT / "runs/next_stage/counterfactual"
OUT = ROOT / "runs/paper_stage/weights"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    drop: dict[str, dict[int, float]] = defaultdict(dict)
    gain: dict[str, dict[int, float]] = defaultdict(dict)
    n_diff: dict[str, int] = {}
    for row in csv.DictReader((CF / "residue_credit.csv").open()):
        pid = row["pair_id"]
        pos = int(row["position"])
        drop[pid][pos] = float(row["c_drop"])
        gain[pid][pos] = float(row["c_gain"])
        n_diff[pid] = n_diff.get(pid, 0) + 1

    out = {"eta": 0.75, "pairs": {}}
    fb = {"drop_only": 0, "gain_only": 0}
    for pid in sorted(n_diff):
        n = n_diff[pid]
        w_diff = {str(p): 1.0 / n for p in sorted(drop[pid])}
        u_drop = {p: max(c, 0.0) for p, c in drop[pid].items()}
        u_gain = {p: max(c, 0.0) for p, c in gain[pid].items()}
        w_drop, f_d = cf_weights(u_drop, n)
        w_gain, f_g = cf_weights(u_gain, n)
        fb["drop_only"] += int(f_d)
        fb["gain_only"] += int(f_g)
        out["pairs"][pid] = {
            "case_id": pid.split(":")[0], "n_diff": n,
            "diff_only": w_diff,
            "drop_only": {str(k): v for k, v in w_drop.items()},
            "gain_only": {str(k): v for k, v in w_gain.items()},
        }
    meta = {
        "n_pairs": len(out["pairs"]),
        "eta": 0.75,
        "fallback_rate_drop_only": fb["drop_only"] / max(1, len(out["pairs"])),
        "fallback_rate_gain_only": fb["gain_only"] / max(1, len(out["pairs"])),
        "queries_per_pair": {
            "diff_only": 0.0,
            "drop_only": st.mean(n_diff.values()),
            "gain_only": st.mean(n_diff.values()),
            "cf": 2.0 * st.mean(n_diff.values()),
        },
        "definitions": {
            "diff_only": "uniform over differing residues, no counterfactual queries",
            "drop_only": "u_i=max(c_drop_i,0), eta=0.75 uniform floor",
            "gain_only": "u_i=max(c_gain_i,0), eta=0.75 uniform floor",
        },
    }
    OUT.joinpath("ablation_weights.json").write_text(json.dumps(out, indent=1))
    OUT.joinpath("ablation_weights_summary.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
