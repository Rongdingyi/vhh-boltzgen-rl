#!/usr/bin/env python
"""Assemble docs/signed_local/SL_CF_DPO_PILOT.md from run artifacts (§88/§89)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
E = ROOT / "runs/signed_local/edges"
P = ROOT / "runs/signed_local/pilot"
DOC = ROOT / "docs/signed_local/SL_CF_DPO_PILOT.md"


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    edge_summary = load(E / "train_edge_summary.json")
    heldout_summary = load(E / "heldout_edge_summary.json")
    validation = load(ROOT / "runs/signed_local/edge_validation.json")
    manifold = load(ROOT / "runs/signed_local/manifold/manifold_audit.json")
    one_edge = load(ROOT / "runs/signed_local/one_edge/one_edge_overfit.json")
    grad_audit = load(ROOT / "runs/signed_local/diagnostics/gradient_direction.json")
    sigma_audit = load(ROOT / "runs/signed_local/diagnostics/sigma_robustness.json")
    pilot = load(P / "pilot_eval.json") or {}
    local_acc = load(P / "local_preference_accuracy.json") or {}

    def reward(arm):
        r = pilot.get(arm)
        return None if not r else r.get("reward_mean")

    base = reward("base")
    cf = reward("cf")
    rows = []
    for arm in ("base", "cf", "local", "neg", "main", "shuffle", "addon", "cf125"):
        r = reward(arm)
        if r is None:
            continue
        d_base = None if base is None else r - base
        d_cf = None if cf is None else r - cf
        acc = (local_acc.get(arm) or {}).get("accuracy", {}).get("all")
        rows.append((arm, r, d_base, d_cf, acc))
    table = "\n".join(
        f"| {a} | {'-' if d is None else f'{d:+.3f}'} | "
        f"{'-' if dc is None else f'{dc:+.3f}'} | {r:+.3f} | "
        f"{'-' if acc is None else f'{acc:.3f}'} |"
        for a, r, d, dc, acc in rows)
    md = f"""# SL-CF-DPO Pilot（task book §88）

## 1. Edge dataset

```json
{json.dumps(edge_summary, indent=1)}
```

Held-out edges：

```json
{json.dumps(heldout_summary, indent=1)}
```

Edge validation：`{json.dumps(validation, indent=1) if validation else 'not run'}`

## 2. Manifold audit

```json
{json.dumps(manifold.get('summary') if manifold else None, indent=1)}
```

## 3. One-edge overfit

```json
{json.dumps({k: v for k, v in (one_edge or {}).items() if k != 'results'}, indent=1)}
```

## 4. Local SNR audits

gradient-direction：`{json.dumps({k: v for k, v in (grad_audit or {}).items() if k != 'rows'})}`

sigma robustness：`{json.dumps({k: v for k, v in (sigma_audit or {}).items() if k != 'rows'})}`

## 5. Pilot arms（held-out 4 cases x 8 samples）

| arm | Δbase | ΔCF | reward | local acc |
|---|---|---|---|---|
{table}

## 6. Decision

（见 `SL_CF_DPO_DECISION.md`；GO 条件见 task book §48/§49）
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
