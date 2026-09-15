# AG-CF-DPO Pre-Pilot（Phase B, task book §29-§33, §80）

协议：4 train cases × 100 updates，beta=10，lr=1e-5，seed=20260913；
eval = all-heldout-8 ×8 samples（historical-4 为同一批样本的子集均值）。

| arm | eligible pairs | median active frac | reward 4 | reward 8 | ΔCF(8) | wins/ties/losses vs CF |
|---|---:|---:|---:|---:|---:|---:|
| No-floor | 192 | 0.47058823529411764 | +2.993 | +1.651 | -0.232 | 3/0/5 |
| Strict consensus | 192 | 0.4444444444444444 | +2.908 | +1.740 | -0.143 | 4/0/4 |
| Strict region | 191 | 0.9444444444444444 | +2.636 | +1.262 | -0.621 | 3/0/5 |

Base heldout8 reward：-1.1096771021257155
Current CF heldout8 reward：1.883255540393293

## Gate B（§33）

```json
{
 "B1": {
  "pass": false,
  "details": {
   "nofloor": {
    "delta_vs_current": -0.23225917015224695,
    "n_cases_not_worse": 3,
    "gain_vs_base": 2.7606734723667614
   },
   "strict": {
    "delta_vs_current": -0.14285491313785315,
    "n_cases_not_worse": 4,
    "gain_vs_base": 2.8500777293811552
   }
  }
 },
 "B2": {
  "pass": false,
  "region": {
   "delta_vs_current": -0.6212568142218515,
   "n_cases_not_worse": 3,
   "gain_vs_base": 2.371675828297157
  },
  "region_vs_base_wins": 7
 },
 "B3": {
  "pass": true,
  "coarse_rescue_share": 0.5456204379562044
 },
 "pass": true,
 "best_simple": "strict"
}
```

## Weights summary

```json
{
 "no_floor": {
  "eligible_pairs": 192,
  "total_pairs": 192,
  "eligible_cases": 24,
  "total_cases": 24,
  "median_active_positions": 7.0,
  "median_active_fraction": 0.47058823529411764,
  "median_weight_entropy": 1.5628868149399557,
  "pilot_case_eligible": {
   "sab2_6u52_c": 8,
   "sab2_7sl5_d": 8,
   "sab2_7nqk_b": 8,
   "sab2_6mqe_h": 8
  }
 },
 "strict_consensus": {
  "eligible_pairs": 192,
  "total_pairs": 192,
  "eligible_cases": 24,
  "total_cases": 24,
  "median_active_positions": 7.0,
  "median_active_fraction": 0.4444444444444444,
  "median_weight_entropy": 1.5590738009833416,
  "pilot_case_eligible": {
   "sab2_6u52_c": 8,
   "sab2_7sl5_d": 8,
   "sab2_7nqk_b": 8,
   "sab2_6mqe_h": 8
  }
 },
 "strict_region": {
  "eligible_pairs": 191,
  "total_pairs": 192,
  "eligible_cases": 24,
  "total_cases": 24,
  "median_active_positions": 13,
  "median_active_fraction": 0.9444444444444444,
  "median_weight_entropy": 2.2798071167646885,
  "pilot_case_eligible": {
   "sab2_6u52_c": 8,
   "sab2_7sl5_d": 8,
   "sab2_7nqk_b": 8,
   "sab2_6mqe_h": 8
  }
 },
 "adaptive": {
  "eligible_pairs": 192,
  "total_pairs": 192,
  "eligible_cases": 24,
  "total_cases": 24,
  "median_active_positions": 13.0,
  "median_active_fraction": 0.9230769230769231,
  "median_weight_entropy": 2.1545388619989803,
  "pilot_case_eligible": {
   "sab2_6u52_c": 8,
   "sab2_7sl5_d": 8,
   "sab2_7nqk_b": 8,
   "sab2_6mqe_h": 8
  }
 },
 "adaptive_shuffle": {
  "eligible_pairs": 192,
  "total_pairs": 192,
  "eligible_cases": 24,
  "total_cases": 24,
  "median_active_positions": 13.0,
  "median_active_fraction": 0.9230769230769231,
  "median_weight_entropy": 2.1545388619989803,
  "pilot_case_eligible": {
   "sab2_6u52_c": 8,
   "sab2_7sl5_d": 8,
   "sab2_7nqk_b": 8,
   "sab2_6mqe_h": 8
  }
 }
}
```

## Audit（Gate A 结论）

```json
{
 "gate_a_pass": true,
 "gate_a": {
  "A_median_conflict_mass": {
   "value": 0.12499999999999996,
   "threshold": 0.08,
   "pass": true
  },
  "B_coarse_rescue_region_share": {
   "value": 0.5456204379562044,
   "threshold": 0.2,
   "pass": true
  },
  "C_pairs_with_coarse_rescue_share": {
   "value": 0.9479166666666666,
   "threshold": 0.2,
   "pass": true
  }
 }
}
```
