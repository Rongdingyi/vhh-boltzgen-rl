#!/usr/bin/env python
"""CF-DPO v2: exact two-residue sign-reversal counterexample (proposal §3.1).

F(z1,z2) = 2 z1 + 2 z2 - 3 z1 z2  ->  F(00)=0, F(01)=2, F(10)=2, F(11)=1.
Shows that the averaged bidirectional decomposition is exact yet carries no
usable local direction, while signed local comparisons do.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
OUT = ROOT / "runs/cf_dpo_v2/small_model"
DOC = ROOT / "docs/cf_dpo_v2/EXP2_COUNTEREXAMPLE.md"


def F(z1: int, z2: int) -> float:
    return 2 * z1 + 2 * z2 - 3 * z1 * z2


def main() -> None:
    table = {f"{z1}{z2}": F(z1, z2) for z1 in (0, 1) for z2 in (0, 1)}
    winner, loser = "11", "00"
    rows = []
    for i in (0, 1):
        win_move = "01" if i == 0 else "10"
        lose_move = "10" if i == 0 else "01"
        drop = F(*map(int, winner)) - F(*map(int, win_move))      # winner -> loser residue
        gain = F(*map(int, lose_move)) - F(*map(int, loser))      # loser -> winner residue
        avg = 0.5 * (drop + gain)
        cons = min(drop, gain) if (drop > 0 and gain > 0) else 0.0
        rows.append({"site": i + 1, "c_drop": drop, "c_gain": gain,
                     "c_avg": avg, "c_cons": cons,
                     "signed_direction": "both endpoint edits improve"})
    endpoint_gap = table[winner] - table[loser]
    residual = sum(r["c_avg"] for r in rows) - endpoint_gap
    payload = {
        "reward_table": table,
        "endpoint_gap": endpoint_gap,
        "decomposition_rows": rows,
        "endpoint_residual": residual,
        "claim": {
            "avg_decomposition_exact": abs(residual) < 1e-12,
            "cons_credit_zero_everywhere": all(r["c_cons"] == 0 for r in rows),
            "signed_edges_capture_direction": "winner 11 -> 01 gives +1; 11 -> 10 gives +1",
            "conclusion": "A residual-shrinkage / adaptive-eta scheme sees residual 0 "
                          "and c_cons 0 (uniform floor only); the most valuable "
                          "information (both single edits improve from the global "
                          "winner) is lost. Signed local edges keep it.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "counterexample.json").write_text(json.dumps(payload, indent=1))
    lines = ["# CF-DPO v2 — Experiment 2: two-residue counterexample (proposal §3.1)", "",
             f"Reward table: {table}", "",
             "| site | c_drop | c_gain | c_avg | c_cons |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['site']} | {r['c_drop']:+.1f} | {r['c_gain']:+.1f} | "
                     f"{r['c_avg']:+.1f} | {r['c_cons']:.1f} |")
    lines += ["", f"- endpoint gap F(11)-F(00) = {endpoint_gap:+.1f}",
              f"- averaged decomposition residual = {residual:+.1e} (exact)",
              f"- c_cons is zero at every site -> uniform-floor fallback",
              "- signed local edges keep: 11->01 ΔR=+1, 11->10 ΔR=+1 (both improve),",
              "  so a signed comparison model learns that the global winner is not",
              "  locally optimal, which no nonnegative weighting can express."]
    DOC.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=1))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
