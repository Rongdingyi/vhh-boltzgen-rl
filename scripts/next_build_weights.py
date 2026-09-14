#!/usr/bin/env python
"""Build Phase C preference weights (CF / B1 random-sparse / B2 shuffle / B4 region).

Inputs : runs/next_stage/counterfactual/{residue_credit.csv, region_credit.csv}
Outputs: runs/next_stage/weights/residue_weights.json
         runs/next_stage/weights/weights_summary.json
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.controls import (  # noqa: E402
    cf_weights, random_sparse_weights, region_weights, shuffle_weights, weight_entropy,
)

CF = ROOT / "runs/next_stage/counterfactual"
OUT = ROOT / "runs/next_stage/weights"
SEED = 20260913


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    credits: dict[str, dict[int, float]] = defaultdict(dict)
    pos_by_region: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader((CF / "residue_credit.csv").open()):
        credits[row["pair_id"]][int(row["position"])] = float(row["c_cons"])
        if row["region"]:
            pos_by_region[row["pair_id"]][row["region"]].append(int(row["position"]))
    region_credits: dict[str, dict[str, float]] = defaultdict(dict)
    for row in csv.DictReader((CF / "region_credit.csv").open()):
        region_credits[row["pair_id"]][row["region"]] = float(row["G_cons"])

    rng = random.Random(SEED)
    out = {"eta": 0.75, "seed": SEED, "pairs": {}}
    fallback_count = 0
    ent_cf, ent_b1 = [], []
    for pid in sorted(credits):
        cred = credits[pid]
        n_diff = len(cred)
        w_cf, fallback = cf_weights(cred, n_diff)
        if fallback:
            fallback_count += 1
        w_b1 = random_sparse_weights(cred, n_diff, rng)
        w_b2 = shuffle_weights(w_cf, rng)
        regions = region_credits.get(pid, {})
        w_b4 = (region_weights(regions, pos_by_region.get(pid, {}))
                if regions else dict(w_cf))
        out["pairs"][pid] = {
            "case_id": pid.split(":")[0], "n_diff": n_diff,
            "fallback": fallback,
            "cf": {str(k): v for k, v in w_cf.items()},
            "random_sparse": {str(k): v for k, v in w_b1.items()},
            "shuffle": {str(k): v for k, v in w_b2.items()},
            "region": {str(k): v for k, v in w_b4.items()},
        }
        ent_cf.append(weight_entropy(w_cf))
        ent_b1.append(weight_entropy(w_b1))

    summary = {
        "n_pairs": len(out["pairs"]),
        "cf_fallback_pair_rate": fallback_count / max(1, len(out["pairs"])),
        "eta": 0.75,
        "entropy_cf_median": st.median(ent_cf),
        "entropy_b1_median": st.median(ent_b1),
        "cf_fallback_formula": "uniform over differing residues when sum(c_cons)=0",
        "b1_doc": "k=#positive-credit residues, random positions, uniform 1/k",
        "b2_doc": "true cf weights shuffled across the pair's differing positions",
        "b4_doc": "region G_cons normalized, uniform inside region",
    }
    (OUT / "residue_weights.json").write_text(json.dumps(out, indent=1))
    (OUT / "weights_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if summary["cf_fallback_pair_rate"] > 0.40:
        print("[weights] WARNING: fallback rate > 40% -> consider signed-average "
              "positive credit fallback (task book §33)")


if __name__ == "__main__":
    main()
