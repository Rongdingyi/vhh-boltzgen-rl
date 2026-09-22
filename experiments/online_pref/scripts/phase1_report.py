#!/usr/bin/env python
"""Phase 1 report + Gate 1 verdict (task book §26/§27/§60)."""
from __future__ import annotations

import csv
import json
import statistics as st

import _common as C  # noqa: E402
from vhh_rl.online_pref import gates  # noqa: E402

REGIONS = ("full", "suffix", "prefix")


def main() -> None:
    ev = json.loads((C.RUN_ROOT / "phase1" / "eval" / "phase1_eval.json").read_text())
    audit = json.loads((C.RUN_ROOT / "phase1" / "sigma_domain_audit.json").read_text())
    rows, delta_sf, delta_sp, delta_pf = [], {}, {}, {}
    invalid_total = fr_total = n_total = 0
    for seed in C.TRAIN_SEEDS:
        full = ev[f"{seed}:full"]
        suffix = ev[f"{seed}:suffix"]
        prefix = ev[f"{seed}:prefix"]
        for payload in (full, suffix, prefix):
            invalid_total += payload["invalid"]
            fr_total += payload["fr"]
            n_total += payload["n"]
        delta_sf[seed] = suffix["reward_mean"] - full["reward_mean"]
        delta_sp[seed] = suffix["reward_mean"] - prefix["reward_mean"]
        delta_pf[seed] = prefix["reward_mean"] - full["reward_mean"]
        rows.append({"seed": seed, "T_full": full["reward_mean"],
                     "T_suffix": suffix["reward_mean"],
                     "T_prefix": prefix["reward_mean"],
                     "suffix_minus_full": delta_sf[seed],
                     "suffix_minus_prefix": delta_sp[seed],
                     "prefix_minus_full": delta_pf[seed]})
    invalid_rate = invalid_total / max(1, n_total)
    mean_sf, std_sf = gates.signed_seed_mean_std(list(delta_sf.values()))
    mean_sp, std_sp = gates.signed_seed_mean_std(list(delta_sp.values()))
    mean_pf, std_pf = gates.signed_seed_mean_std(list(delta_pf.values()))
    positive_sf = sum(1 for v in delta_sf.values() if v > 0)
    verdict = {
        "pass": (mean_sf >= gates.PHASE1_MIN_SUFFIX_FULL
                 and positive_sf >= gates.PHASE0_MIN_SEEDS_POSITIVE
                 and mean_sp >= gates.PHASE1_MIN_SUFFIX_PREFIX
                 and mean_pf <= gates.PHASE1_MAX_PREFIX_FULL
                 and invalid_rate <= gates.MAX_INVALID and fr_total == 0),
        "mean_suffix_full": mean_sf, "std_suffix_full": std_sf,
        "seeds_suffix_gt_full": positive_sf,
        "mean_suffix_prefix": mean_sp, "std_suffix_prefix": std_sp,
        "mean_prefix_full": mean_pf, "std_prefix_full": std_pf,
        "invalid_rate": invalid_rate, "fr_mismatch": fr_total,
        "thresholds": {"min_suffix_full": gates.PHASE1_MIN_SUFFIX_FULL,
                       "min_suffix_prefix": gates.PHASE1_MIN_SUFFIX_PREFIX,
                       "max_prefix_full": gates.PHASE1_MAX_PREFIX_FULL},
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with (C.RESULTS / "phase1_seed_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    gate = gates.write_gate(
        C.RUN_ROOT / "phase1" / "gate.json", pass_=verdict["pass"],
        protocol_sha256=C.protocol_hash(), train_seeds=C.TRAIN_SEEDS,
        metrics={**{k: v for k, v in verdict.items() if k != "pass"},
                 "sigma_audit": {k: audit[k] for k in
                                 ("branch_sigma_median", "suffix_mass_analytic",
                                  "prefix_mass_analytic")}},
        decision=("PROCEED to Phase 2" if verdict["pass"]
                  else "STOP temporal-localization idea (Gate 1 failed)"))
    doc = f"""# Phase 1 — Branch-aware sigma restriction (shared pair bank)

## 1. sigma domain（§21）

- branch sigma (median over Phase 0 B groups)：{audit['branch_sigma_median']:.3f}
- suffix mass = P_train(σ ≤ σ_b)：{audit['suffix_mass_analytic']:.3f}（MC {audit['suffix_mass_mc']:.3f}）
- prefix mass：{audit['prefix_mass_analytic']:.3f}
- conditional suffix median/P10/P90：{audit['conditional_suffix']['median']:.3f} / {audit['conditional_suffix']['p10']:.3f} / {audit['conditional_suffix']['p90']:.3f}
- conditional prefix median：{audit['conditional_prefix']['median']:.3f}

## 2. Temporal arms（same pairs / schedules / ε / augmentation）

| seed | T-full | T-suffix | T-prefix | suffix−full | suffix−prefix | prefix−full |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['seed']} | {r['T_full']:+.3f} | {r['T_suffix']:+.3f} | {r['T_prefix']:+.3f} | {r['suffix_minus_full']:+.3f} | {r['suffix_minus_prefix']:+.3f} | {r['prefix_minus_full']:+.3f} |" for r in rows)}

seed-level mean ± sample std：suffix−full = {mean_sf:+.3f} ± {std_sf:.3f}；suffix−prefix = {mean_sp:+.3f} ± {std_sp:.3f}；prefix−full = {mean_pf:+.3f} ± {std_pf:.3f}

## 3. Gate 1（§26）

```json
{json.dumps(verdict, indent=1)}
```

## 4. 解释边界（§27）

当前 DPO 仍是 endpoint denoising surrogate；本 Gate 只检验
"branch-aware restriction of the diffusion training noise domain"，
不得写成 exact causal trajectory DPO。

## 5. 结论

{gate['decision']}
"""
    (C.DOCS / "PHASE1_TEMPORAL.md").write_text(doc)
    print(json.dumps({"gate1_pass": verdict["pass"], "mean_suffix_full": mean_sf,
                      "mean_suffix_prefix": mean_sp, "mean_prefix_full": mean_pf},
                     indent=1))


if __name__ == "__main__":
    main()
