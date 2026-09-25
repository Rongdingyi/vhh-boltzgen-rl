# Phase 1 — Branch-aware sigma restriction (shared pair bank)

## 1. sigma domain（§21）

- branch sigma (median over Phase 0 B groups)：10.395
- suffix mass = P_train(σ ≤ σ_b)：0.696（MC 0.693）
- prefix mass：0.304
- conditional suffix median/P10/P90：2.657 / 0.524 / 7.870
- conditional prefix median：22.371

## 2. Temporal arms（same pairs / schedules / ε / augmentation）

| seed | T-full | T-suffix | T-prefix | suffix−full | suffix−prefix | prefix−full |
|---:|---:|---:|---:|---:|---:|---:|
| 20260915 | +3.632 | +2.954 | +3.103 | -0.678 | -0.149 | -0.528 |
| 43 | +4.360 | +3.902 | +3.974 | -0.458 | -0.072 | -0.386 |
| 44 | +2.529 | +2.764 | +3.635 | +0.235 | -0.871 | +1.105 |

seed-level mean ± sample std：suffix−full = -0.300 ± 0.476；suffix−prefix = -0.364 ± 0.440；prefix−full = +0.064 ± 0.905

## 3. Gate 1（§26）

```json
{
 "pass": false,
 "mean_suffix_full": -0.3003098184465974,
 "std_suffix_full": 0.4762747032328911,
 "seeds_suffix_gt_full": 1,
 "mean_suffix_prefix": -0.36406400922056664,
 "std_suffix_prefix": 0.440272685143859,
 "mean_prefix_full": 0.06375419077396922,
 "std_prefix_full": 0.9048637301894187,
 "invalid_rate": 0.001736111111111111,
 "fr_mismatch": 0,
 "thresholds": {
  "min_suffix_full": 0.25,
  "min_suffix_prefix": 0.5,
  "max_prefix_full": 0.1
 }
}
```

## 4. 解释边界（§27）

当前 DPO 仍是 endpoint denoising surrogate；本 Gate 只检验
"branch-aware restriction of the diffusion training noise domain"，
不得写成 exact causal trajectory DPO。

## 5. 结论

STOP temporal-localization idea (Gate 1 failed)
