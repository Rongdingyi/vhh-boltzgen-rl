#!/usr/bin/env python
"""Gate 3 report -> docs/branch_distill/GATE3_REPORT.md (§53-§56/§65)."""
from __future__ import annotations

import json
import statistics as st

import _common as C  # noqa: E402
from vhh_rl.branch_distill import gates  # noqa: E402

LABELS = {"base": "Base", "a": "A Offline changed DPO",
          "b": "B Online changed DPO", "c": "C Sibling distill all CDR",
          "d": "D Sibling distill changed residues"}


def main() -> None:
    payload = json.loads((C.GATE3_DIR / "eval" / "gate3_eval.json").read_text())
    query_counts = {}
    for arm in "bcd":
        rounds = C.GATE3_DIR / arm / "rounds.json"
        if rounds.is_file():
            rows = json.loads(rounds.read_text())
            # P0 correction: prefer the corrected endpoint scorer-query count
            query_counts[arm] = sum(
                r.get("reward_queries_cumulative",
                      r.get("reward_queries_round", r.get("queries_cumulative", 0)))
                for r in rows)
    rewards = {arm.upper() if arm != "base" else "base": v["reward_mean"]
               for arm, v in payload.items()}
    per_case = {}
    for case_id in C.heldout8():
        per_case[case_id] = {arm.upper() if arm != "base" else "base":
                             v["per_case"].get(case_id)
                             for arm, v in payload.items()}
    invalid = sum(v["invalid"] for v in payload.values())
    n_total = sum(v["n"] for v in payload.values())
    verdict = gates.gate3_verdict(rewards, per_case,
                                  invalid_rate=invalid / max(1, n_total),
                                  fr_mismatch=sum(v["fr"] for v in payload.values()))
    lines = ["# Gate 3 — component comparison", "",
             "> **Correction note (P0, online-pref task book §2.1).** "
             "Historical Gate3 reward values unchanged. The old 'queries' field "
             "counted pair records rather than scored sibling endpoints; the "
             "endpoint scorer-query count is recomputed from the round logs.", "",
             "| arm | reward8 | Δbase | ΔA | W/T/L vs A | invalid | FR | queries |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    base = rewards.get("base")
    arm_a = rewards.get("A")
    for arm in ["base", "a", "b", "c", "d"]:
        if arm not in payload:
            continue
        v = payload[arm]
        delta_base = v["reward_mean"] - base if base is not None else None
        delta_a = v["reward_mean"] - arm_a if arm_a is not None else None
        wins = ties = losses = 0
        for case_id in C.heldout8():
            if arm == "a":
                continue
            left = v["per_case"].get(case_id)
            right = per_case[case_id].get("A")
            if left is None or right is None:
                continue
            d = left - right
            wins += int(d > 1e-9)
            losses += int(d < -1e-9)
            ties += int(abs(d) <= 1e-9)
        wtl = "-" if arm == "a" else f"{wins}/{ties}/{losses}"
        lines.append(f"| {LABELS[arm]} | {v['reward_mean']:+.3f} | "
                     f"{'-' if delta_base is None else f'{delta_base:+.3f}'} | "
                     f"{'-' if delta_a is None else f'{delta_a:+.3f}'} | {wtl} | "
                     f"{v['invalid']} | {v['fr']} | "
                     f"{query_counts.get(arm, 0 if arm in 'bcd' else 0)} |")
    lines += ["", "## 组件解释（§55）", ""]
    def rel(x, y):
        return None if rewards.get(x) is None or rewards.get(y) is None else rewards[x] - rewards[y]
    if rel("B", "A") is not None:
        lines.append(f"- B - A = {rel('B','A'):+.3f}（on-policy pair refresh）")
        lines.append(f"- D - B = {rel('D','B'):+.3f}")
        lines.append(f"- D - C = {rel('D','C'):+.3f}")
        lines.append(f"- D - A = {rel('D','A'):+.3f}")
    lines += ["", "## Gate（§54/§56）", "", "```json",
              json.dumps(verdict, indent=1), "```", "",
              f"protocol_sha256: `{C.protocol_hash()}`", ""]
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "GATE3_REPORT.md").write_text("\n".join(lines))
    C.write_json(C.GATE3_DIR / "gate3.json", {"protocol_sha256": C.protocol_hash(),
                                              **verdict})
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
