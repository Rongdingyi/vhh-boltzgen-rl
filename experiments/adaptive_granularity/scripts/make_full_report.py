#!/usr/bin/env python
"""Phase D report -> docs/adaptive_granularity/AG_FULL.md (task book §58-§62)."""
from __future__ import annotations
import json
from pathlib import Path

import _common as C  # noqa: E402

DOC = C.ROOT / "docs/adaptive_granularity/AG_FULL.md"
GATE_FULL = C.FULL_DIR / "gate_full.json"
GATE_MULTISEED = C.FULL_DIR / "gate_multiseed.json"


def main() -> None:
    payload = json.loads((C.FULL_DIR / "eval/full_summary.json").read_text())
    base = payload["base"]["reward_mean"]
    lines = ["| arm | seed | selected step | reward8 | Δbase | invalid | valid |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for key, value in payload.items():
        if key == "base":
            continue
        selected = value.get("selected")
        if not selected:
            lines.append(f"| {key} | - | - | - | - | - | no valid ckpt |")
            continue
        lines.append(f"| {key} | 20260913 | {value['selected_step']} | "
                     f"{selected['reward_mean']:+.3f} | "
                     f"{selected['reward_mean'] - base:+.3f} | {selected['invalid']} | "
                     f"{selected['valid']} |")
    gate = _gate(payload)
    gate_multi = _gate_multiseed(payload)
    GATE_FULL.parent.mkdir(parents=True, exist_ok=True)
    GATE_FULL.write_text(json.dumps(gate, indent=1))
    GATE_MULTISEED.write_text(json.dumps(gate_multi, indent=1))
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(f"""# AG-CF-DPO Full（task book §57-§66）

| arm | seed | selected step | reward8 | Δbase | invalid | valid |
|---|---|---|---|---|---|---|
{chr(10).join(lines[2:])}

Base reward8：{base}

## Full GO Gate（§61）

```json
{json.dumps(gate, indent=1)}
```

## Notes

- checkpoint 选择只用 8 held-out reward + validity（§60），未使用 valid100。
- multi-seed（§62）只对 Current CF / Adaptive 运行：`bash run.sh ag-full-seeds`。
- valid100 / structure / second reward 需人工触发（§63-§66）。
""")
    print(json.dumps(gate, indent=1))


def _gate(payload: dict) -> dict:
    base = payload["base"]["reward_mean"]
    current = payload.get("f0_seed20260913", {}).get("selected")
    adaptive = payload.get("f2_seed20260913", {}).get("selected")
    shuffle = payload.get("f3_seed20260913", {}).get("selected")
    eligible = payload.get("f4_seed20260913", {}).get("selected")
    if not current or not adaptive:
        return {"verdict": "NOT_RUN"}
    d_cf = adaptive["reward_mean"] - current["reward_mean"]
    wins = sum(1 for c, r in adaptive["per_case"].items()
               if current["per_case"].get(c) is not None
               and r is not None and r >= current["per_case"][c])
    d_shuffle = (adaptive["reward_mean"] - shuffle["reward_mean"]) if shuffle else None
    d_elig = (adaptive["reward_mean"] - eligible["reward_mean"]) if eligible else None
    passed = (d_cf >= 0.50 and wins >= 6
              and d_shuffle is not None and d_shuffle > 0
              and d_elig is not None and d_elig > 0)
    return {"verdict": "FULL_GO" if passed else "NOT_PASSED",
            "stage": "single_seed_full",
            "criterion": "Adaptive-CF >= +0.50, >=6/8 cases not worse, "
                         "Adaptive > Shuffle, Adaptive > Eligible-CF",
            "delta_vs_current": d_cf, "cases_not_worse_vs_current": wins,
            "delta_vs_shuffle": d_shuffle, "delta_vs_eligible_cf": d_elig,
            "base": base}


def _gate_multiseed(payload: dict) -> dict:
    """True 3-seed full gate (§62); only runs after FULL_GO."""
    seeds = [42, 43, 44]
    deltas = {}
    for seed in seeds:
        current = payload.get(f"f0_seed{seed}", {}).get("selected")
        adaptive = payload.get(f"f2_seed{seed}", {}).get("selected")
        if not current or not adaptive:
            continue
        deltas[seed] = adaptive["reward_mean"] - current["reward_mean"]
    if not deltas:
        return {"verdict": "NOT_RUN", "stage": "full_multiseed"}
    mean_delta = sum(deltas.values()) / len(deltas)
    non_negative = sum(1 for d in deltas.values() if d >= -1e-9)
    passed = mean_delta >= 0.50 and non_negative == len(deltas)
    return {"verdict": "MULTISEED_GO" if passed else "NOT_PASSED",
            "stage": "full_multiseed",
            "criterion": "mean(Adaptive-CF) >= +0.50 and 3/3 non-negative",
            "mean_delta": mean_delta, "delta_per_seed": deltas,
            "seeds_non_negative": non_negative, "n_seeds": len(deltas)}


if __name__ == "__main__":
    main()
