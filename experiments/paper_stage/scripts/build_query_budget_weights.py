#!/usr/bin/env python
"""Paper-stage Step 5: subsampled bidirectional CF weights (task book §33).

For budgets {0.25, 0.50, 0.75, 1.00} select that fraction of each pair's
differing residues (deterministic budget_seed=123), keep the conservative
credit only for queried positions (0 otherwise), uniform floor eta=0.75.

Writes runs/paper_stage/weights/query_budget_weights.json
Schema: pairs[pid][budget_tag] -> {pos: weight}
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

from vhh_rl.credit.controls import cf_weights  # noqa: E402

CF = ROOT / "runs/next_stage/counterfactual"
OUT = ROOT / "runs/paper_stage/weights"
BUDGETS = [0.25, 0.50, 0.75, 1.00]
BUDGET_SEED = 123


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cons: dict[str, dict[int, float]] = defaultdict(dict)
    n_diff: dict[str, int] = {}
    for row in csv.DictReader((CF / "residue_credit.csv").open()):
        pid = row["pair_id"]
        cons[pid][int(row["position"])] = float(row["c_cons"])
        n_diff[pid] = n_diff.get(pid, 0) + 1

    out = {"eta": 0.75, "budget_seed": BUDGET_SEED, "budgets": {}, "pairs": {}}
    queries = {f"{b:.2f}": [] for b in BUDGETS}
    fb = {f"{b:.2f}": 0 for b in BUDGETS}
    for pid in sorted(n_diff):
        n = n_diff[pid]
        positions = sorted(cons[pid])
        rng = random.Random(f"{BUDGET_SEED}:{pid}")
        order = positions[:]
        rng.shuffle(order)
        entry = {"case_id": pid.split(":")[0], "n_diff": n}
        for b in BUDGETS:
            k = max(1, int(round(b * n)))
            queried = set(order[:k])
            u = {p: (cons[pid][p] if p in queried else 0.0) for p in positions}
            w, f = cf_weights(u, n)
            tag = f"{b:.2f}"
            entry[tag] = {str(p): v for p, v in w.items()}
            queries[tag].append(2 * k)  # bidirectional
            fb[tag] += int(f)
        out["pairs"][pid] = entry
    out["budgets"] = {
        tag: {
            "queries_per_pair_mean": st.mean(queries[tag]),
            "queries_fraction_of_full": st.mean(queries[tag]) / (2 * st.mean(n_diff.values())),
            "fallback_rate": fb[tag] / max(1, len(out["pairs"])),
        }
        for tag in queries
    }
    (OUT / "query_budget_weights.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["budgets"], indent=1))


if __name__ == "__main__":
    main()
