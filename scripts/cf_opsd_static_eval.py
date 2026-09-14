#!/usr/bin/env python
"""Phase D evaluation + report: base / CF-DPO-mini / CF-OPSD-static (§50-§55)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

CFG = ROOT / "configs/cf_opsd/fixed_cases.yaml"
OUT = ROOT / "runs/cf_opsd/pilot_eval"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_PILOT_RESULTS.md"


def parse_arm(tag: str) -> tuple[str, int | None]:
    m = re.match(r"^(cf_dpo_mini|cf_opsd_static)_u(\d+)$", tag)
    if not m:
        return tag, None
    return m.group(1), int(m.group(2))


def build_rows(results: dict) -> list[dict]:
    """Pure helper (regression-tested): results tag -> summary rows with step."""
    base = results.get("base", {}).get("reward_mean")
    rows = []
    for tag, res in results.items():
        if tag == "base":
            continue
        method, step = parse_arm(tag)
        rows.append({
            "arm": tag, "method": method, "step": step,
            "reward_mean": res.get("reward_mean"),
            "delta_vs_base": (None if base is None or res.get("reward_mean") is None
                              else res["reward_mean"] - base),
        })
    return rows


def gate_d_rows(rows: list[dict]) -> dict:
    """Gate D (§54): require positive gains on both sides (audit fix D).

    reward_superiority := G_opsd > 0 and G_cfdpo > 0 and G_opsd/G_cfdpo >= 1.10
    """
    out = {}
    for step in (50, 100):
        cd = next((r["delta_vs_base"] for r in rows
                   if r["method"] == "cf_dpo_mini" and r["step"] == step), None)
        op = next((r["delta_vs_base"] for r in rows
                   if r["method"] == "cf_opsd_static" and r["step"] == step), None)
        if cd is None or op is None:
            out[step] = None
            continue
        positive = cd > 0.0 and op > 0.0
        out[step] = {
            "cfdpo_delta": cd, "opsd_delta": op,
            "relative": (op / cd) if cd > 0 else None,
            "positive_gains": positive,
            "reward_superiority": bool(positive and op >= 1.10 * cd),
        }
    return out


def main() -> None:
    from vhh_rl.cf_opsd.evaluator import evaluate  # lazy: keeps module importable without BoltzGen

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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_eval.json").write_text(json.dumps(results, indent=1))

    base = results.get("base", {}).get("reward_mean")
    rows = build_rows(results)
    gate_d = gate_d_rows(rows)
    json.dump({"gate_d": gate_d}, (OUT / "gate_d.json").open("w"), indent=1, default=str)

    lines = "\n".join(
        f"| {r['arm']} | {r['reward_mean']:+.3f} | {r['delta_vs_base']:+.3f} |"
        for r in rows if r["reward_mean"] is not None and r["delta_vs_base"] is not None)
    gate_lines = "\n".join(
        (f"| {s} | {v['cfdpo_delta']:+.3f} | {v['opsd_delta']:+.3f} | {v['relative']:.2f} | "
         f"{'YES' if v['reward_superiority'] else 'NO'} |") if v else
        f"| {s} | n/a | n/a | n/a | n/a |"
        for s, v in gate_d.items())
    doc = f"""# CF-OPSD Pilot Results（Phase D）

Held-out = {len(heldout)} cases × 8 samples；base reward = {base if base is None else f'{base:+.3f}'}。

| arm | held-out reward | Δ vs base |
|---|---|---|
{lines}

## Gate D（§54）

| updates | CF-DPO Δ | CF-OPSD Δ | ratio | ≥1.10× ? |
|---|---|---|---|---|
{gate_lines}
"""
    DOC.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
