# Phase 0 — Online refresh with all-changed support (§12-§14)

## 1. Online all-changed vs Offline（3 seeds）

| seed | A offline | B online | V verified | B−A | V−B | B≥A cases | B W/T/L vs A |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20260915 | +2.438 | +2.877 | +3.307 | +0.439 | +0.430 | 6/8 | 6/0/2 |
| 43 | +2.907 | +3.253 | +2.091 | +0.347 | -1.162 | 2/8 | 2/0/6 |
| 44 | +2.913 | +3.371 | +2.888 | +0.458 | -0.483 | 5/8 | 5/0/3 |

seed-level mean ± sample std：B−A = +0.415 ± 0.060；V−B = -0.405 ± 0.799

## 2. Gate 0（§14）

```json
{
 "pass": true,
 "mean_delta": 0.4146015137278785,
 "seeds_positive": 3,
 "median_cases_ge": 5,
 "invalid_rate": 0.0,
 "fr_mismatch": 0,
 "thresholds": {
  "min_mean_delta": 0.3,
  "min_seeds_positive": 2,
  "min_cases_ge": 4,
  "max_invalid": 0.01
 }
}
```

## 3. Query accounting（§56）

```json
{
 "20260915:B": {
  "reward_queries_total": 128,
  "pair_records_total": 32,
  "updates": 100
 },
 "20260915:V": {
  "reward_queries_total": 128,
  "pair_records_total": 32,
  "updates": 100
 },
 "43:B": {
  "reward_queries_total": 128,
  "pair_records_total": 32,
  "updates": 100
 },
 "43:V": {
  "reward_queries_total": 128,
  "pair_records_total": 32,
  "updates": 100
 },
 "44:B": {
  "reward_queries_total": 127,
  "pair_records_total": 32,
  "updates": 100
 },
 "44:V": {
  "reward_queries_total": 127,
  "pair_records_total": 32,
  "updates": 100
 }
}
```

## 4. 旧 +0.594 的复现（§59 Q2）

all-changed support 下 B−A 的 seed-level 均值见上表（旧 Gate3 的 +0.594 混入了
carrier-verified filtering 与 4-case 口径，本表是干净口径）。

## 5. verified control 解释（§14.1）

verified filtering 未显示明显贡献

## 6. 结论

PROCEED to Phase 1
