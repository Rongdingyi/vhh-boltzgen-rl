# CF-DPO v2 — Experiment 4 pilot（4 train / 4 held-out，matched updates）

| arm | held-out reward | delta vs base | wins | n |
|---|---|---|---|---|
| base | -1.206 | +0.000 | 0/4 | 4 |
| cf_dpo_mini_u50 | +1.380 | +2.586 | 4/4 | 4 |
| cf_dpo_mini_u100 | +3.186 | +4.392 | 4/4 | 4 |
| cf_dpo_v2_signed_u50 | -0.922 | +0.285 | 2/4 | 4 |
| cf_dpo_v2_signed_u100 | -0.664 | +0.543 | 2/4 | 4 |
| cf_dpo_v2_v2_u50 | -0.995 | +0.212 | 2/4 | 4 |
| cf_dpo_v2_v2_u100 | -0.512 | +0.694 | 2/4 | 4 |

所有臂使用相同的 4 个训练 case、相同 held-out 4 cases、相同生成 seeds（seed_base+800000，8 samples/case）。CF-DPO-mini 与 v2/signed 在相同optimizer updates（50/100）下对比。

## 说明（修正版）

本表为 E1/E2/E3 修复后的重跑结果（图重建为 565 节点，drop 边 dR 符号修正、
node ID 唯一、same-seq 占比 25%）。与 matched CF-DPO-mini 的对比结论：

- v2 u100 相比 base **+0.69**、signed u100 **+0.54**；
- 同池普通 DPO（CF-DPO-mini）u100 为 **+4.39（4/4 胜）**；
- 即：**修正后 v2/signed 仍显著落后于同预算的普通 DPO 基线**。
