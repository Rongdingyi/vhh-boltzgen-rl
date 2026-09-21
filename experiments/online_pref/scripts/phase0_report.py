#!/usr/bin/env python
"""Phase 0 report + Gate 0 verdict (task book §14/§15/§59)."""
from __future__ import annotations

import csv
import json
import statistics as st

import _common as C  # noqa: E402
from vhh_rl.online_pref import gates  # noqa: E402


def main() -> None:
    ev = json.loads((C.EVAL / "phase0_eval.json").read_text())
    base = ev["base"]
    rows, per_case_rows = [], []
    delta_b_minus_a, cases_ge = {}, {}
    delta_v_minus_b = {}
    invalid_total = fr_total = n_total = 0
    for seed in C.TRAIN_SEEDS:
        a = ev.get(f"{seed}:A")
        b = ev.get(f"{seed}:B")
        v = ev.get(f"{seed}:V")
        if not (a and b and v):
            raise SystemExit(f"seed {seed}: missing arm results in phase0_eval.json")
        for arm_payload in (a, b, v):
            invalid_total += arm_payload["invalid"]
            fr_total += arm_payload["fr"]
            n_total += arm_payload["n"]
        b_minus_a = b["reward_mean"] - a["reward_mean"]
        delta_b_minus_a[seed] = b_minus_a
        delta_v_minus_b[seed] = v["reward_mean"] - b["reward_mean"]
        wins = sum(1 for c, r in b["per_case"].items()
                   if r is not None and a["per_case"].get(c) is not None and r >= a["per_case"][c])
        cases_ge[seed] = wins
        rows.append({"seed": seed, "A": a["reward_mean"], "B": b["reward_mean"],
                     "V": v["reward_mean"], "B_minus_A": b_minus_a,
                     "V_minus_B": v["reward_mean"] - b["reward_mean"],
                     "cases_B_ge_A": wins,
                     "B_wins_ties_losses": _wtl(b, a),
                     "V_wins_ties_losses": _wtl(v, b)})
        for case_id in sorted(a["per_case"]):
            per_case_rows.append({
                "seed": seed, "case_id": case_id,
                "A": a["per_case"].get(case_id), "B": b["per_case"].get(case_id),
                "V": v["per_case"].get(case_id),
                "B_minus_A": (b["per_case"].get(case_id, float("nan"))
                              - a["per_case"].get(case_id, float("nan"))),
            })
    invalid_rate = invalid_total / max(1, n_total)
    verdict = gates.phase0_verdict(delta_b_minus_a, cases_ge, invalid_rate, fr_total)
    mean_ba, std_ba = gates.signed_seed_mean_std(list(delta_b_minus_a.values()))
    mean_vb, std_vb = gates.signed_seed_mean_std(list(delta_v_minus_b.values()))
    query_stats = _query_stats()

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    _write_csv(C.RESULTS / "phase0_seed_summary.csv", rows)
    _write_csv(C.RESULTS / "phase0_per_case.csv", per_case_rows)
    gate_payload = gates.write_gate(
        C.PHASE0 / "gate.json", pass_=verdict["pass"],
        protocol_sha256=C.protocol_hash(), train_seeds=C.TRAIN_SEEDS,
        metrics={"mean_B_minus_A": mean_ba, "std_B_minus_A": std_ba,
                 "per_seed_B_minus_A": delta_b_minus_a,
                 "mean_V_minus_B": mean_vb, "std_V_minus_B": std_vb,
                 "cases_B_ge_A": cases_ge,
                 "invalid_rate": invalid_rate, "fr_mismatch": fr_total,
                 "query_accounting": query_stats,
                 **{k: v for k, v in verdict.items() if k != "pass"}},
        decision=("PROCEED to Phase 1" if verdict["pass"]
                  else "STOP online route (Gate 0 failed)"))

    doc = f"""# Phase 0 — Online refresh with all-changed support (§12-§14)

## 1. Online all-changed vs Offline（3 seeds）

| seed | A offline | B online | V verified | B−A | V−B | B≥A cases | B W/T/L vs A |
|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(f"| {r['seed']} | {r['A']:+.3f} | {r['B']:+.3f} | {r['V']:+.3f} | {r['B_minus_A']:+.3f} | {r['V_minus_B']:+.3f} | {r['cases_B_ge_A']}/8 | {r['B_wins_ties_losses']} |" for r in rows)}

seed-level mean ± sample std：B−A = {mean_ba:+.3f} ± {std_ba:.3f}；V−B = {mean_vb:+.3f} ± {std_vb:.3f}

## 2. Gate 0（§14）

```json
{json.dumps(verdict, indent=1)}
```

## 3. Query accounting（§56）

```json
{json.dumps(query_stats, indent=1)}
```

## 4. 旧 +0.594 的复现（§59 Q2）

all-changed support 下 B−A 的 seed-level 均值见上表（旧 Gate3 的 +0.594 混入了
carrier-verified filtering 与 4-case 口径，本表是干净口径）。

## 5. verified control 解释（§14.1）

{"V−B 均值 >= +0.30 且 2/3 seeds 正：记录 'geometry filtering may contribute'，不并入主方法" if mean_vb >= 0.30 else "verified filtering 未显示明显贡献"}

## 6. 结论

{gate_payload['decision']}
"""
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "PHASE0_ONLINE_REFRESH.md").write_text(doc)
    print(json.dumps({"gate0_pass": verdict["pass"], "mean_B_minus_A": mean_ba,
                      "per_seed": delta_b_minus_a,
                      "mean_V_minus_B": mean_vb}, indent=1))


def _wtl(left: dict, right: dict) -> str:
    wins = ties = losses = 0
    for case_id, value in left["per_case"].items():
        other = right["per_case"].get(case_id)
        if value is None or other is None:
            continue
        delta = value - other
        wins += int(delta > 1e-9)
        losses += int(delta < -1e-9)
        ties += int(abs(delta) <= 1e-9)
    return f"{wins}/{ties}/{losses}"


def _query_stats() -> dict:
    out = {}
    for seed in C.TRAIN_SEEDS:
        for arm in ("B", "V"):
            rounds = C.seed_dir(seed) / arm / "rounds.json"
            if rounds.is_file():
                payload = json.loads(rounds.read_text())
                out[f"{seed}:{arm}"] = {
                    "reward_queries_total": payload["reward_queries_total"],
                    "pair_records_total": payload["pair_records_total"],
                    "updates": payload["updates"],
                }
    return out


def _write_csv(path, rows) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
