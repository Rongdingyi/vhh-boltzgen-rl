# P2 — Generalization Beyond VHH（决策文档）

任务书 §44–§47、§71：P2 不是当前执行项；只有在 P0/P1 全部成立后才考虑。
本文件记录执行时的 gate 状态与决策。

## 1. 候选方案（未执行）

| 方案 | 内容 | 工程成本 |
|---|---|---|
| P2-A | 同一 BoltzGen 换 generic protein design task（fixed-backbone / binder design）+ 现成 black-box scalar reward | 低-中 |
| P2-B | Pallatom（all-atom joint sequence–structure diffusion）port | 高 |

## 2. Go/No-Go 条件（§47）

```text
CF 3-seed robust
结构验证完成
第二 reward 成功
论文时间允许
```

## 3. 当前状态（本阶段结束时）

| 条件 | 状态 |
|---|---|
| CF 3-seed robust | **部分**：3/3 seed Δ>+4.0，但 CF−N3 均值只有 +1.09（< +2.0 gate）；CF 一致 ≥ 所有 control |
| 结构验证 | **完成，但不利**：Boltz-2 100-case、Protenix-v2 50-case 均未复现 CF 优于 N3 的趋势 |
| 第二 reward | 独立性 gate 通过（Spearman −0.04）；CF vs Uniform 结果见 `P1_SECOND_REWARD.md` |
| 论文时间 | 本任务书范围内 |

## 4. 决策

**不执行 P2（Pallatom / 第二 task）。**

理由：

1. P0-A 的强 claim（CF ≫ N3）未通过 3-seed gate，论文故事需要先重写为
   "credit correspondence > sparsity > uniform"，而不是"大幅超过 N3"。
2. P0-C 表明 CF 的结构收益不显著；在此情况下扩到第二 task/model 只会
   放大验证成本，不解决核心证据缺口。
3. §71 明确要求完成 P0/P1 后停止；Pallatom port 属于下一阶段。

## 5. 若未来重启 P2 的前置条件

```text
1. 论文主线改为 attribution-chain 证据（CF > diff-only > shuffle > N3）且被接受
2. query-budget 与第二 reward 的结果支持方法的 reward-agnostic 定位
3. 有明确的第二 task / reward 资源预算
```
