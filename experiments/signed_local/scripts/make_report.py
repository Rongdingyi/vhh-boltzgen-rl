#!/usr/bin/env python
"""Assemble docs/signed_local/SL_CF_DPO_PILOT.md from run artifacts (§88/§89)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
E = ROOT / "runs/signed_local/edges"
P = ROOT / "runs/signed_local/pilot"
DOC = ROOT / "docs/signed_local/SL_CF_DPO_PILOT.md"

ARM_ORDER = ["base", "cf", "local", "neg", "main", "shuffle", "addon", "cf125"]
ARM_LABEL = {"base": "Base", "cf": "CF-mini (A1)", "local": "Local-only (A2)",
             "neg": "CF+Neg", "main": "**SL-CF-DPO (A4)**",
             "shuffle": "DirectionShuffle (A5)", "addon": "CF+Local add-on (A6)",
             "cf125": "CF-125 compute-matched (A7)"}
STEPS = {"cf": (100, 0), "local": (0, 100), "neg": (75, 25), "main": (75, 25),
         "shuffle": (75, 25), "addon": (100, 25), "cf125": (125, 0)}
CONFIG = {"base": (0, 0), "addon": (100, 25), "cf125": (125, 0)}


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def validity(pilot: dict, arm: str) -> dict:
    v = pilot.get(arm)
    if not v:
        return {"n": 0, "invalid": 0, "fr": 0}
    n = sum(c["n"] for c in v["cases"].values())
    return {"n": n,
            "invalid": sum(c["n_invalid"] for c in v["cases"].values()),
            "fr": sum(c["n_fr_mismatch"] for c in v["cases"].values())}


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

    base, cf = reward("base"), reward("cf")
    rows = []
    for arm in ARM_ORDER:
        r = reward(arm)
        if r is None:
            continue
        n_g, n_l = CONFIG.get(arm, STEPS.get(arm, (0, 0)))
        d_base = None if base is None else r - base
        d_cf = None if cf is None else r - cf
        acc = (local_acc.get(arm) or {}).get("accuracy_all_mean")
        if acc is None:
            acc = (local_acc.get(arm) or {}).get("accuracy", {}).get("all")
        val = validity(pilot, arm)
        inv = (val["invalid"] / val["n"]) if val["n"] else None
        rows.append((ARM_LABEL.get(arm, arm), n_g, n_l, r, d_base, d_cf, acc,
                     inv, val["fr"], val["n"]))
    table = "\n".join(
        f"| {lab} | {ng} | {nl} | {r:+.3f} | "
        f"{'-' if d is None else f'{d:+.3f}'} | "
        f"{'-' if dc is None else f'{dc:+.3f}'} | "
        f"{'-' if acc is None else f'{acc:.3f}'} | "
        f"{'-' if inv is None else f'{inv:.2%}'} | "
        f"{fr} | {n} |"
        for lab, ng, nl, r, d, dc, acc, inv, fr, n in rows)

    acc_table = ""
    for arm in ARM_ORDER:
        a = local_acc.get(arm)
        if not a:
            continue
        acc = a.get("accuracy") or {}
        acc_table += (
            f"| {ARM_LABEL.get(arm, arm)} | {a.get('accuracy_all_mean', float('nan')):.3f} "
            f"± {a.get('accuracy_all_std', float('nan')):.3f} | "
            f"{a.get('median_z_mean', float('nan')):+.5f} | "
            f"{acc.get('both_negative', float('nan')):.3f} | "
            f"{acc.get('sign_flip', float('nan')):.3f} | "
            f"{acc.get('winner_drop', float('nan')):.3f} | "
            f"{acc.get('loser_gain', float('nan')):.3f} |\n")

    proto_txt = (ROOT / "experiments/signed_local/configs/FROZEN_PROTOCOL.yaml").read_text() \
        if (ROOT / "experiments/signed_local/configs/FROZEN_PROTOCOL.yaml").is_file() else "missing"

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

## 5. Pilot main table（held-out 4 cases x 8 samples, 固定 eval seed）

| arm | global steps | local steps | reward | Δbase | ΔCF | local acc | invalid | FR | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## 6. Held-out local preference accuracy（{len(local_acc.get('base', {}).get('seeds', [12345])) if local_acc else 1} seeds）

| arm | acc (mean ± sd) | median z | both-neg | sign-flip | winner-drop | loser-gain |
|---|---:|---:|---:|---:|---:|---:|
{acc_table}

## 7. Frozen protocol

```yaml
{proto_txt}
```

## 8. Decision

见 `SL_CF_DPO_DECISION.md`（GO/NO-GO 条件见 task book §48/§49）。
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(md)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
