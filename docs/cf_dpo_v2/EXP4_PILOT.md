# CF-DPO v2 — Experiment 4 pilot（4 train / 4 held-out，matched updates）

| arm | held-out reward | delta vs base | wins | n |
|---|---|---|---|---|
| base | -1.206 | +0.000 | 0/4 | 4 |
| cf_dpo_mini_u50 | +1.380 | +2.586 | 4/4 | 4 |
| cf_dpo_mini_u100 | +3.186 | +4.392 | 4/4 | 4 |
| cf_dpo_v2_signed_u50 | -0.900 | +0.307 | 3/4 | 4 |
| cf_dpo_v2_signed_u100 | -1.187 | +0.020 | 2/4 | 4 |
| cf_dpo_v2_v2_u50 | -0.378 | +0.829 | 2/4 | 4 |
| cf_dpo_v2_v2_u100 | +0.044 | +1.250 | 3/4 | 4 |

所有臂使用相同的 4 个训练 case、相同 held-out 4 cases、相同生成 seeds（seed_base+800000，8 samples/case）。CF-DPO-mini 与 v2/signed 在相同optimizer updates（50/100）下对比。
