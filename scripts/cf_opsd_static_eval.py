#!/usr/bin/env python
"""Phase D evaluation + report: base / CF-DPO-mini / CF-OPSD-static (§50-§55)."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.evaluator import evaluate  # noqa: E402

CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
OUT = ROOT / "runs/cf_opsd/pilot_eval"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_PILOT_RESULTS.md"


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    heldout = cfg["heldout_cases"]
    arms = {
        "base": None,
        "cf_dpo_mini_u50": ROOT / "runs/cf_opsd/cf_dpo_mini/run/checkpoint_0050.pt",
        "cf_dpo_mini_u100": ROOT / "runs/cf_opsd/cf_dpo_mini/run/checkpoint_0100.pt",
        "cf_opsd_static_u50": ROOT / "runs/cf_opsd/static_v0/u050/checkpoint_0050.pt",
        "cf_opsd_static_u100": ROOT / "runs/cf_opsd/static_v0/u100/checkpoint_0100.pt",
    }
    results = {}
    for tag, ckpt in arms.items():
        if ckpt is not None and not Path(ckpt).is_file():
            print(f"[skip] {tag}: missing {ckpt}")
            continue
        results[tag] = evaluate(heldout, ckpt, tag, run_root=OUT / tag)
    (OUT / "pilot_eval.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_eval.json").write_text(json.dumps(results, indent=1))

    base = results.get("base", {}).get("reward_mean")
    rows = []
    for tag, res in results.items():
        if tag == "base" or base is None:
            continue
        rows.append({
            "arm": tag, "reward_mean": res["reward_mean"],
            "delta_vs_base": res["reward_mean"] - base,
        })
    cfdpo = {r["step"]: r for r in rows if r["arm"].startswith("cf_dpo_mini")}
    opsd = {r["step"]: r for r in rows if r["arm"].startswith("cf_opsd_static")}
    # gains at matched updates
    gate_d = {}
    for step in (50, 100):
        cd = next((r["delta_vs_base"] for r in rows
                   if r["arm"] == f"cf_dpo_mini_u{step}"), None)
        op = next((r["delta_vs_base"] for r in rows
                   if r["arm"] == f"cf_opsd_static_u{step}"), None)
        if cd is None or op is None:
            gate_d[step] = None
            continue
        gate_d[step] = {
            "cfdpo_delta": cd, "opsd_delta": op,
            "relative": (op / cd) if cd not in (None, 0) else None,
            "reward_superiority": op >= 1.10 * cd,
        }
    json.dump({"gate_d": gate_d}, (OUT / "gate_d.json").open("w"), indent=1, default=str)

    lines = "\n".join(
        f"| {r['arm']} | {r['reward_mean']:+.3f} | {r['delta_vs_base']:+.3f} |" for r in rows)
    doc = f"""# CF-OPSD Pilot Results（Phase D）

Held-out = {len(heldout)} cases × 8 samples；base reward = {base if base is None else f'{base:+.3f}'}。

| arm | held-out reward | Δ vs base |
|---|---|---|
{lines}

## Gate D（§54）

| updates | CF-DPO Δ | CF-OPSD Δ | ratio | ≥1.10× ? |
|---|---|---|---|---|
""" + "\n".join(
        f"| {s} | {gate_d[s]['cfdpo_delta']:+.3f} | {gate_d[s]['opsd_delta']:+.3f} | "
        f"{gate_d[s]['relative']:.2f} | {'YES' if gate_d[s]['reward_superiority'] else 'NO'} |"
        if gate_d[s] else f"| {s} | n/a | n/a | n/a | n/a |"
        for s in (50, 100)) + "\n"
    DOC.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
