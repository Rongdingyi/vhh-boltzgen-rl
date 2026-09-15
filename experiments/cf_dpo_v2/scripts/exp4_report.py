#!/usr/bin/env python
"""CF-DPO v2 experiment 4 pilot report: base / CF-DPO-mini / signed / v2."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
PILOT = ROOT / "runs/cf_opsd/pilot_eval/pilot_eval.json"
V2 = ROOT / "runs/cf_dpo_v2"
DOC = ROOT / "docs/cf_dpo_v2/EXP4_PILOT.md"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def per_case(res: dict | None) -> dict[str, float]:
    if not res:
        return {}
    return {c: v["reward_mean"] for c, v in res.get("cases", {}).items()
            if v.get("reward_mean") is not None}


def main() -> None:
    pilot = load(PILOT) or {}
    base_cases = per_case(pilot.get("base"))
    arms: dict[str, dict] = {}
    for key in ("cf_dpo_mini_u50", "cf_dpo_mini_u100"):
        cases = per_case(pilot.get(key))
        if cases:
            arms[key] = {"reward_mean": st.mean(cases.values()), "cases": cases}
    for variant in ("signed", "v2"):
        for step in (50, 100):
            res = load(V2 / f"{variant}_u100/eval_u{step}/eval_summary.json")
            cases = per_case(res)
            if cases:
                arms[f"cf_dpo_v2_{variant}_u{step}"] = {
                    "reward_mean": st.mean(cases.values()), "cases": cases}

    rows = []
    if base_cases:
        rows.append({"arm": "base", "reward_mean": st.mean(base_cases.values()),
                     "delta": 0.0, "wins": 0, "n": len(base_cases)})
    for name, info in arms.items():
        if not base_cases:
            break
        common = sorted(set(info["cases"]) & set(base_cases))
        deltas = [info["cases"][c] - base_cases[c] for c in common]
        rows.append({"arm": name, "reward_mean": info["reward_mean"],
                     "delta": st.mean(deltas) if deltas else None,
                     "wins": sum(1 for d in deltas if d > 0), "n": len(common)})
    lines = ["| arm | held-out reward | delta vs base | wins | n |", "|---|---|---|---|---|"]
    for r in rows:
        d = "n/a" if r["delta"] is None else f"{r['delta']:+.3f}"
        lines.append(f"| {r['arm']} | {r['reward_mean']:+.3f} | {d} | "
                     f"{r['wins']}/{r['n']} | {r['n']} |")
    doc = ("# CF-DPO v2 — Experiment 4 pilot（4 train / 4 held-out，matched updates）\n\n"
           + "\n".join(lines) + "\n\n"
           "所有臂使用相同的 4 个训练 case、相同 held-out 4 cases、相同生成 seeds"
           "（seed_base+800000，8 samples/case）。CF-DPO-mini 与 v2/signed 在相同"
           "optimizer updates（50/100）下对比。\n")
    DOC.write_text(doc)
    (V2 / "exp4_pilot.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(doc)


if __name__ == "__main__":
    main()
