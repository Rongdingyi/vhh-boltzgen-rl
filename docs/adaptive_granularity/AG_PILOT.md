# AG-CF-DPO Pilot（Phase C, task book §46-§51, §81-§82）

| arm | reward8 | Δbase | ΔCF | vs shuffle | vs eligible-CF | invalid |
|---|---:|---:|---:|---:|---:|---:|
| Current CF | +1.883 | +2.993 | +0.000 | - | - | 0 |
| No-floor | +1.651 | +2.761 | -0.232 | - | - | 0 |
| Strict consensus | +1.740 | +2.850 | -0.143 | - | - | 0 |
| Strict region | +1.262 | +2.372 | -0.621 | - | - | 0 |

## Gate C（§48-§51）

```json
{
 "verdict": "PHASE_C_NOT_RUN"
}
```

## Mechanism（§82）

```json
{
 "current_cf_conflict_mass_median": 0.12499999999999996,
 "adaptive_mode_counts": {
  "region": 299,
  "residue": 137,
  "abstain": 112
 },
 "rescue_class_counts": {
  "coarse_rescue": 299,
  "residue_candidate": 137,
  "abstain": 112
 },
 "eligible_like_coverage": {
  "coarse_rescue": 299,
  "residue_candidate": 137,
  "abstain": 112
 },
 "gradient_conflict": null
}
```

## Notes

- best simple arm（Gate B3 进入时取 Region-only）：region
- 3-seed confirmation（§49/§50）仅对 Current CF / Adaptive 运行，结果见 `runs/adaptive_granularity/pilot/seed_*`。
