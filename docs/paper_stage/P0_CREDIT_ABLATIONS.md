# P0-B — Credit Attribution Core Ablations

任务书 §13–§20；协议：`FROZEN_PROTOCOL.yaml`；seed = primary（20260913）；
500 steps；held-out checkpoint 选择按 §54；valid100 = 100 cases × 8 samples。

## 1. 方法定义

| arm | 空间支持（进入 preference 的 residue） | credit 大小 | 方向 | queries/pair |
|---|---|---|---|---|
| N3 Uniform Fake | 全部 CDR fake atoms | 均匀 | — | 0 |
| Diff-only Uniform | 仅 winner≠loser 的 residue | 均匀（1/\|D\|） | — | 0 |
| Drop-only | differing residues | `max(c_drop,0)`+floor | winner→loser | 15.6 |
| Gain-only | differing residues | `max(c_gain,0)`+floor | loser→winner | 15.6 |
| Shuffle (B2) | differing residues | 真 CF 权重随机打乱 | — | 31.2 |
| **CF-DPO** | differing residues | `c_cons`+floor（η=0.75） | **双向** | 31.2 |

## 2. valid100 结果（Δ vs base = 6.578）

| 方法 | reward | Δ vs base | W/T/L | invalid |
|---|---|---|---|---|
| N3 Uniform | 9.374 | +2.796 | 95/0/5 | 0 |
| Shuffle | 10.108 | +3.531 | 96/0/4 | 0 |
| Diff-only Uniform | 10.798 | +4.221 | 98/0/2 | 0 |
| Drop-only | 11.510 | +4.932 | 94/0/6 | 0 |
| Gain-only | 11.836 | +5.258 | 96/0/4 | 0 |
| **CF-DPO** | **12.143** | **+5.566** | **99/0/1** | 0 |

训练信号（z / implicit acc, steps 250–450）：
N3 +0.019/0.64；diff-only +0.037/0.74；drop-only +0.134/0.81；
gain-only +0.131/0.85；CF +0.225/0.86。

## 3. 增益分解（任务书 §15）

| 步骤 | 对比 | Δ reward | 解读 |
|---|---|---|---|
| 1 | N3 → Diff-only | **+1.43** | 去掉 same-residue 的 off-target 更新 |
| 2 | Diff-only → Drop-only | +0.71 | 加入 counterfactual magnitude（单向） |
| 3 | Diff-only → Gain-only | +1.04 | 同上（loser→winner 方向） |
| 4 | 单向 → CF | +0.31 ~ +0.63 | 双向 consolidation |

结论：

1. **必须证明的因果项成立：CF (+5.566) > Diff-only Uniform (+4.221)，
   差距 +1.345**（任务书 §20：若 CF ≈ Diff-only，则主要收益只是"忽略未变残基"，
   创新要收缩；此处不成立，counterfactual magnitude 是真实增益来源）。
2. 单项最大的收益来自 N3 → Diff-only（+1.43），说明 off-target 更新确实是
   uniform DPO 的主要损失来源（呼应 §0 假设 H2）。
3. 单向 credit 已能捕获大部分 counterfactual 增益（+4.93 ~ +5.26），
   双向仅再贡献 +0.31 ~ +0.63；结合主 multi-seed 的噪声水平（CF std 0.56），
   双向的增量在单 seed 上处于噪声边缘，**但方向一致且 CF 的 win-rate 最高
   （99/100）**。
4. 排序完全符合任务书 §20 的理想图景：

```text
CF > one-sided > diff-only ≳ shuffle > N3
```

## 4. 对论文 claim 的影响

- §67 Claim 2（"correct residue-credit correspondence, not sparsity"）：
  证据链成立：Random Sparse ≈ N3（next-stage）、Shuffle < Diff-only <
  one-sided < CF。
- §64 的失败模式（Diff-only ≈ CF）未出现。
- §65（one-sided ≈ bidirectional）：单向非常接近双向（差距 +0.3~0.6），
  按任务书该结果"不坏"——可以作为效率 variant 报告（query 减半），
  但默认主方法保持双向（重复性证据见 P0_MULTISEED：CF 方差最小、
  win-rate 最高）。

## 5. 文件

- `results/paper_stage/ablation_summary.csv`
- `runs/paper_stage/{diff_only,drop_only,gain_only}/train_metrics.jsonl`
- `runs/native_pool/eval_summary_paper_{diff_only,drop_only,gain_only}_s*_valid100_shard*.json`
- `runs/paper_stage/ablations/checkpoint_selection.json`
