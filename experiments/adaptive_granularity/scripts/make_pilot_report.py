#!/usr/bin/env python
"""Assemble AG_PRE_PILOT.md (Phase B) and AG_PILOT.md (Phase C) reports."""
from __future__ import annotations
import json
from pathlib import Path

import _common as C  # noqa: E402

DOCS = C.ROOT / "docs/adaptive_granularity"
PILOT_SUMMARY = C.PILOT_DIR / "eval/pilot_summary.json"
SIMPLE_ARMS = ["nofloor", "strict", "region"]


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _case_wins(cases_a: dict, cases_b: dict, n_min: int) -> str:
    wins = ties = losses = 0
    for cid, a in cases_a.items():
        b = cases_b.get(cid)
        if a is None or b is None:
            continue
        d = a - b
        wins += int(d > 1e-9)
        losses += int(d < -1e-9)
        ties += int(abs(d) <= 1e-9)
    return f"{wins}/{ties}/{losses}"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    summary = load(PILOT_SUMMARY) or {}
    audit = load(C.AUDIT_DIR / "audit_summary.json") or {}
    weights = load(C.WEIGHTS_DIR / "weights_summary.json") or {}
    materialized = load(C.WEIGHTS_DIR / "materialize_summary.json") or {}
    grad = load(C.AUDIT_DIR / "gradient_conflict.json")
    base = summary.get("base", {})
    current = summary.get("current", {})
    base8, cur8 = base.get("reward_mean"), current.get("reward_mean")

    def cell(arm: str, key: str):
        payload = summary.get(arm, {})
        return payload.get(key)

    simple_rows = []
    for arm in SIMPLE_ARMS:
        if arm not in summary:
            continue
        v = summary[arm]
        label = C.VARIANT_LABEL[{"nofloor": "no_floor", "strict": "strict_consensus",
                                 "region": "strict_region"}[arm]]
        simple_rows.append((label, v))
    def variant_of(arm: str) -> str:
        return C.ARM_TO_VARIANT.get(arm, arm)

    pre_table = "\n".join(
        f"| {label} | {materialized.get(variant_of(arm), {}).get('eligible_pairs')} | "
        f"{weights.get('variant_stats', {}).get(variant_of(arm), {}).get('median_active_fraction')} | "
        f"{v.get('reward_mean_hist4'):+.3f} | {v.get('reward_mean'):+.3f} | "
        f"{v.get('delta_vs_current'):+.3f} | {v.get('wins_vs_current')} |"
        for arm, (label, v) in zip(SIMPLE_ARMS, simple_rows))

    gate_b = _gate_b(summary, audit)
    gate_b_txt = json.dumps(gate_b, indent=1)

    # Phase C report
    arms_present = [a for a in ["current", "nofloor", "strict", "region",
                                "adaptive", "shuffle", "eligible-cf"] if a in summary]
    best_simple = gate_b.get("best_simple")
    c_rows = []
    for arm in arms_present:
        v = summary[arm]
        label = C.VARIANT_LABEL.get(
            {"nofloor": "no_floor", "strict": "strict_consensus",
             "region": "strict_region", "eligible-cf": "adaptive_eligible_cf",
             "current": "current_cf"}.get(arm, arm), arm)
        d_cf = v.get("delta_vs_current", 0.0 if arm == "current" else None)
        c_rows.append(f"| {label} | {v.get('reward_mean'):+.3f} | "
                      f"{v.get('delta_vs_base', float('nan')):+.3f} | "
                      f"{'-' if d_cf is None else f'{d_cf:+.3f}'} | "
                      f"{'-' if arm == 'current' else _delta(v, summary.get('shuffle'))} | "
                      f"{'-' if arm == 'current' else _delta(v, summary.get('eligible-cf'))} | "
                      f"{v.get('invalid')} |")
    gate_c = _gate_c(summary)
    mech = {
        "current_cf_conflict_mass_median": audit.get("conflict_mass", {}).get("median"),
        "adaptive_mode_counts": audit.get("adaptive_mode_counts"),
        "rescue_class_counts": audit.get("rescue_class_counts"),
        "eligible_like_coverage": audit.get("rescue_class_counts"),
        "gradient_conflict": None if grad is None else {
            k: grad.get(k) for k in ("n_sites", "median_cosine", "negative_fraction",
                                     "strong_negative_fraction")},
    }

    pre_md = f"""# AG-CF-DPO Pre-Pilot（Phase B, task book §29-§33, §80）

协议：4 train cases × 100 updates，beta=10，lr=1e-5，seed=20260913；
eval = all-heldout-8 ×8 samples（historical-4 为同一批样本的子集均值）。

| arm | eligible pairs | median active frac | reward 4 | reward 8 | ΔCF(8) | wins/ties/losses vs CF |
|---|---:|---:|---:|---:|---:|---:|
{pre_table}

Base heldout8 reward：{base8}
Current CF heldout8 reward：{cur8}

## Gate B（§33）

```json
{gate_b_txt}
```

## Weights summary

```json
{json.dumps(weights.get('variant_stats'), indent=1)}
```

## Audit（Gate A 结论）

```json
{json.dumps({'gate_a_pass': audit.get('gate_a_pass'),
             'gate_a': audit.get('gate_a')}, indent=1)}
```
"""
    (DOCS / "AG_PRE_PILOT.md").write_text(pre_md)
    (C.PILOT_DIR / "gate_b.json").write_text(json.dumps(gate_b, indent=1))

    pilot_md = f"""# AG-CF-DPO Pilot（Phase C, task book §46-§51, §81-§82）

| arm | reward8 | Δbase | ΔCF | vs shuffle | vs eligible-CF | invalid |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(c_rows)}

## Gate C（§48-§51）

```json
{json.dumps(gate_c, indent=1)}
```

## Mechanism（§82）

```json
{json.dumps(mech, indent=1)}
```

## Notes

- best simple arm（Gate B3 进入时取 Region-only）：{best_simple}
- 3-seed confirmation（§49/§50）仅对 Current CF / Adaptive 运行，结果见 `runs/adaptive_granularity/pilot/seed_*`。
"""
    (DOCS / "AG_PILOT.md").write_text(pilot_md)
    (C.PILOT_DIR / "gate_c.json").write_text(json.dumps(gate_c, indent=1))
    print(json.dumps({"gate_b": gate_b["pass"], "gate_c": gate_c["verdict"]}, indent=1))


def _case_better(summary: dict, arm: str, ref_arm: str) -> int:
    arm_cases = summary.get(arm, {}).get("per_case", {})
    ref_cases = summary.get(ref_arm, {}).get("per_case", {})
    return sum(1 for c, r in arm_cases.items()
               if r is not None and ref_cases.get(c) is not None and r > ref_cases[c])


def _delta(arm_payload: dict, ref_payload: dict | None) -> str:
    if not ref_payload:
        return "-"
    d = arm_payload["reward_mean"] - ref_payload["reward_mean"]
    return f"{d:+.3f}"


def _gate_b(summary: dict, audit: dict) -> dict:
    base = summary.get("base", {}).get("reward_mean")
    cur = summary.get("current", {}).get("reward_mean")
    current_cases = summary.get("current", {}).get("per_case", {})
    out = {"B1": None, "B2": None, "B3": None}
    simple_candidates = {}
    for arm in SIMPLE_ARMS:
        if arm not in summary:
            continue
        v = summary[arm]
        d = (v["reward_mean"] - cur) if cur is not None else None
        not_worse = sum(1 for c, r in v.get("per_case", {}).items()
                        if r is not None and current_cases.get(c) is not None
                        and r >= current_cases[c])
        simple_candidates[arm] = {
            "delta_vs_current": d,
            "n_cases_not_worse": not_worse,
            "gain_vs_base": (v["reward_mean"] - base) if base is not None else None,
        }
    b1 = {arm: info for arm, info in simple_candidates.items()
          if arm in ("nofloor", "strict")}
    b1_pass = any(info["delta_vs_current"] is not None
                  and info["delta_vs_current"] >= 0.20
                  and info["n_cases_not_worse"] >= 5
                  for info in b1.values())
    region = simple_candidates.get("region")
    b2_pass = False
    if region and base is not None and cur is not None:
        cf_gain = cur - base
        b2_pass = (cf_gain > 0 and region["gain_vs_base"] >= 0.80 * cf_gain
                   and _case_better(summary, "region", "base") >= 5)
    n_regions = audit.get("n_regions") or 0
    rescue_share = (audit.get("rescue_class_counts", {}).get("coarse_rescue", 0) / n_regions
                    if n_regions else 0.0)
    b3_pass = False
    if region and region["delta_vs_current"] is not None:
        b3_pass = rescue_share >= 0.25 and region["delta_vs_current"] >= -0.75
    best = None
    if b2_pass:
        best = "region"
    if b1_pass:
        cand = [a for a in ("nofloor", "strict") if a in simple_candidates
                and simple_candidates[a]["delta_vs_current"] is not None]
        if cand:
            best = max(cand, key=lambda a: simple_candidates[a]["delta_vs_current"])
    if best is None:
        for arm in ("nofloor", "strict", "region"):
            if arm in simple_candidates:
                d = simple_candidates[arm]["delta_vs_current"]
                if d is not None and (best is None
                                      or d > simple_candidates[best]["delta_vs_current"]):
                    best = arm
    return {"B1": {"pass": b1_pass, "details": b1},
            "B2": {"pass": b2_pass, "region": region,
                   "region_vs_base_wins": _case_better(summary, "region", "base")},
            "B3": {"pass": b3_pass, "coarse_rescue_share": rescue_share},
            "pass": bool(b1_pass or b2_pass or b3_pass),
            "best_simple": best}


def _gate_c(summary: dict) -> dict:
    cur = summary.get("current")
    adaptive = summary.get("adaptive")
    shuffle = summary.get("shuffle")
    eligible = summary.get("eligible-cf")
    if not adaptive:
        return {"verdict": "PHASE_C_NOT_RUN"}
    d_cf = adaptive["reward_mean"] - cur["reward_mean"]
    d_shuffle = (adaptive["reward_mean"] - shuffle["reward_mean"]) if shuffle else None
    d_elig = (adaptive["reward_mean"] - eligible["reward_mean"]) if eligible else None
    wins = sum(1 for c, r in adaptive["per_case"].items()
               if cur["per_case"].get(c) is not None and r is not None
               and r >= cur["per_case"][c])
    invalid_ok = adaptive["invalid"] <= 0.01 * adaptive["n"]
    fr_ok = adaptive["fr"] == 0
    strong = (d_cf >= 0.50 and wins >= 5
              and d_shuffle is not None and d_shuffle >= 0.50
              and d_elig is not None and d_elig >= 0.30
              and invalid_ok and fr_ok)
    borderline = (-0.2 <= d_cf <= 0.5
                  and d_shuffle is not None and d_shuffle > 0
                  and d_elig is not None and d_elig > 0)
    nogo = ((d_shuffle is not None and d_shuffle <= 0)
            or (d_elig is not None and d_elig <= 0)
            or d_cf < -0.5)
    verdict = ("STRONG_GO" if strong else
               "BORDERLINE_3SEED" if borderline else
               "NO_GO" if nogo else "INCONCLUSIVE")
    return {"verdict": verdict, "delta_vs_current": d_cf,
            "delta_vs_shuffle": d_shuffle, "delta_vs_eligible_cf": d_elig,
            "cases_not_worse_vs_current": wins,
            "invalid": adaptive["invalid"], "fr": adaptive["fr"]}


if __name__ == "__main__":
    main()
