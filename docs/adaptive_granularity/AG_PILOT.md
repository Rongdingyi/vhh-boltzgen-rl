# AG-CF-DPO Pilot（Phase C, task book §46-§51, §81-§82）

| arm | reward8 | Δbase | ΔCF | vs shuffle | vs eligible-CF | invalid |
|---|---:|---:|---:|---:|---:|---:|
| Current CF | +1.883 | +2.993 | +0.000 | - | - | 0 |
| No-floor | +1.651 | +2.761 | -0.232 | +1.516 | -0.295 | 0 |
| Strict consensus | +1.740 | +2.850 | -0.143 | +1.605 | -0.205 | 0 |
| Strict region | +1.262 | +2.372 | -0.621 | +1.127 | -0.684 | 0 |
| Adaptive | +1.614 | +2.723 | -0.270 | +1.479 | -0.332 | 0 |
| shuffle | +0.135 | +1.245 | -1.748 | +0.000 | -1.811 | 0 |
| Eligible-CF | +1.946 | +3.055 | +0.062 | +1.811 | +0.000 | 0 |

## Gate C（§48-§51）

```json
{
 "verdict": "NO_GO",
 "delta_vs_current": -0.2696427028859034,
 "delta_vs_shuffle": 1.478525380953215,
 "delta_vs_eligible_cf": -0.3320314890006557,
 "cases_not_worse_vs_current": 3,
 "invalid": 0,
 "fr": 0
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
