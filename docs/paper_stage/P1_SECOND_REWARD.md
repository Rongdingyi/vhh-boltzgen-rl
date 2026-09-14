# P1-B — Second Black-Box Reward（VHH-ESM-C CDR PLL）

任务书 §36–§43；主 reward 保持不变（`cdr_camelid_margin`），本实验只验证
reward-agnostic：把同一 CF-DPO 流程（同一公式、同一 η、同一训练配置）搬到
第二个独立 black-box sequence reward 上。

## 1. Second reward 选择与审计（§37–§40）

候选 A：**VHH-ESM-C exact masked CDR PLL**（`/share/data/limc/esmc-600m-vhh`，
`vhh_esmc_guidance` pipeline，sequence-only）。

在 24 个 train case、767 条 base-pool 序列上审计：

| 指标 | 值 | gate |
|---|---|---|
| median per-case std | **0.165** | > 1e-4 ✓ |
| global Spearman vs `cdr_camelid_margin` | **−0.0396** | \|·\| < 0.90 ✓ |
| per-case Spearman 中位 | +0.006 | — |
| 选中 | **是** | |

即：该 reward 有足够分辨率，且与主 classifier reward **几乎不相关**
（0.0 附近），是理想的独立 reward。

## 2. 流程（与主实验完全一致）

```text
base pool (24 cases x 32) → cdr_pll 排序
→ rank-symmetric pairs（192 对，top8 vs bottom8）
→ 双向单残基 counterfactual（5,984 条）+ cdr_pll 打分
→ c_cons + η=0.75 floor 权重（fallback 0%）
→ 训练 Uniform（N3 同款）与 CF（同一 weighted-DPO 代码）
```

训练曲线（z，steps 250–450）：Uniform **+0.023**（与主 N3 的 +0.019 一致），
CF **+0.221**（与主 CF 的 +0.225 一致）→ **CF 的 credit 机制在第二个 reward
上以完全相同的幅度分离**。

## 3. 结果

### Checkpoint 选择（held-out 8 cases，按第二 reward 自身选择）

| arm | 选中 | held-out cdr_pll（8 cases × 8） |
|---|---|---|
| Uniform | s500 | −1.7166 |
| CF | s500 | −1.6897 |

两臂都随训练单调改善；CF 在 5 个评测点上始终优于 Uniform。

### Held-out 对照（case-level，base = −2.0560）

| arm | cdr_pll mean | Δ vs base | wins vs base | Wilcoxon p |
|---|---|---|---|---|
| Uniform | −1.7166 | **+0.3394** | 8/8 | 0.0078 |
| **CF** | **−1.6897** | **+0.3663** | 8/8 | 0.0078 |

**CF vs Uniform：Δ = +0.0269，6/8 cases CF 更优，p = 0.38。**

## 4. §42 pilot 判定

```text
CF > Uniform                     ✓（+0.0269）
≥ 6/8 cases CF 不差于 Uniform     ✓（6/8 严格更优）
```

→ 通过，可进入 valid100/完整 test（按任务书 §41 的后续步骤；本阶段
先记录 pilot 结论与完整 held-out 结果）。

## 5. 诚实结论

1. CF 在第二个独立 reward 上复现了"CF > Uniform"的方向，但幅度远小于
   主 classifier reward 上的单 seed 差距（+0.027 vs +2.77/单 seed）。
2. Uniform 在该 reward 上也稳健有效（与主 N3 行为一致），说明两个 reward
   都与当前 generator 兼容（§66 的"Uniform 有效但 CF 增益有限"分支）。
3. 结合 P0-A multi-seed（CF−N3 平均 +1.09）与 P0-C（结构无显著收益），
   论文对 CF 的定位应为：**稳定优于 uniform/shuffle 的 attribution 机制，
   可跨 reward 复现方向性增益**，而不是"大幅超越 uniform"。

## 6. 文件

- `results/paper_stage/second_reward_summary.csv`
- `runs/paper_stage/second_reward/{audit_stats.json,base_pool_scored.jsonl,pairs_train.jsonl,counterfactual_credit.csv,weights.json}`
- `runs/paper_stage/second_reward/sweep/*_scored.jsonl`
- `runs/paper_stage/second_reward/{checkpoint_selection.json,heldout_summary.md}`
