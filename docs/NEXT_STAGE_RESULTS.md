# Next-stage Native Atom14 Preference Results

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md`（Phase A–D，完成后停止）

## 1. Starting point（round-1）

| arm | valid100 Δreward | refold CDR RMSD |
|---|---|---|
| N2 | +2.702 | 1.999 |
| **N3（fake-atom DPO）** | **+2.796** | 2.090 |
| N4 | +2.512 | 2.206 |
| base | — | 1.860 |

N3 把**全部 CDR fake atoms** 视为 preference-relevant；pair 平均约 15.6 个
CDR 位点不同，credit 被均匀摊开。

## 2. Motivation

- 192 pairs 的 differing positions 与 reward gap 基本无关
  （Pearson +0.070 / Spearman +0.036）。
- classifier reward 与 refold 结构指标几乎不相关：reward vs CDR RMSD
  r=+0.019（n=400），与 CDR pLDDT r=−0.563、recovery r=−0.447。
- 结论：uniform fake-atom DPO 必然把 preference 更新施加到低/零 credit
  位点；需要从黑盒 sequence reward 恢复 residue credit。

## 3. Reproduced diagnostics（A0）

见 `docs/NEXT_STAGE_DIAGNOSTIC_REPRO.md`：
Hamming mean 15.60 / min 3 / max 31 / 总差异 2996；
corr(Hamming, gap)=+0.070/+0.036；refold 相关性与任务书一致
（+0.019 / −0.563 / −0.447）。

## 4. Counterfactual credit audit（A1/A2, Gate A = STRONG）

数据：24 train cases / 192 pairs；5,992 条单残基双干预 CF + 1,096 条
region CF + 384 原始 = 7,472 条（7,231 unique，batch scorer + SHA1 cache，
原始分数漂移 0）。见 `docs/COUNTERFACTUAL_CREDIT_AUDIT.md`。

| 指标 | median |
|---|---|
| top10 / top20 / **top30** / top50 positive-credit mass | 0.595 / 0.814 / **0.940** / 1.000 |
| **n_eff / n_diff** | **0.270** |
| positive_fraction | 0.471 |
| sign agreement (c_drop vs c_gain) | 0.667 |
| corr(C_sum(c_avg), reward_gap) | Pearson +0.768 / Spearman +0.791 |

Region（A2）：CDR1 G_cons 1.098（sign agree 0.837）、CDR2 1.432（0.871）、
**CDR3 3.685（0.948）**；dominant region：CDR3 105/192、CDR1 45、CDR2 42；
CDR3 存在明显 epistasis（region − Σresidue 中位 −0.64）。

**H1：支持（STRONG）** → 采用 residue-level CF weighting。

## 5. Diffusion temporal audit（B1, Gate B = WEAK）

128 条 trajectory（8 train + 8 held-out cases × 8），frozen base，50 steps，
官方 sampler + `res_from_atom14` 逐步解码。见
`docs/TEMPORAL_CREDIT_AUDIT.md`。

- σ_seq = σ_90 = **0.3137**（median CDR identity ≥0.80 且 valid ≥0.90 的最
  大 σ）。
- σ≳2.8 时 decode 基本无效；σ 2.84→0.64 之间 identity 0.70→1.00、
  valid 0.02→0.88；CDR-bb 归一化收敛在 identity=0.70 时已 0.819。
- 不存在 "backbone 已收敛（>0.8）而 sequence recovery <0.5" 的窗口
  （0 步）→ **H3 不支持**。
- reward 相关性在 σ≈0.64 已达 0.995。

## 6. Methods tested

| 臂 | preference 聚合 | temporal |
|---|---|---|
| N3（round-1） | 全部 CDR fake atoms，uniform | 无 |
| B1 random sparse | 随机 k 个 residue（匹配非零数） | 无 |
| B2 shuffle | 真 CF 权重在 pair 内随机打乱 | 无 |
| B3 **CF（主方法）** | `w̃ᵢ=(1−η)/n_diff+η·c_consᵢ/Σc_cons`，η=0.75 | 无 |
| region（B4） | region G_cons 归一化，region 内 uniform | 可选，未训练（仅审计） |
| T1 / T2（Phase D） | N3 uniform mask | hard / smooth g(σ)，σ_seq=0.3137，τ=0.5 |

DPO 数学形式、paired noise、可训练模块与 N3 完全一致；uniform weights 回归
门通过（`test_weighted_loss_uniform_equivalence`）。

## 7. Held-out results（8 cases × 8，§40 checkpoint 选择）

| arm | best ckpt | reward | Δ vs base (+1.271) |
|---|---|---|---|
| B1 random_sparse | s450 | +4.095 | +2.82 |
| B2 shuffle | s450 | +4.823 | +3.55 |
| **B3 CF** | **s500** | **+7.809** | **+6.54** |

训练信号：B3 z→+0.225 / implicit acc 0.86，control 仅 z≈0.016–0.025 /
acc 0.55–0.68。

## 8. valid100 reward（100 cases × 8）

| arm | reward | Δ vs base | median Δ | wins | Wilcoxon p | Δ vs N3 |
|---|---|---|---|---|---|---|
| base | 6.578 | — | — | — | — | −2.796 |
| B1 random_sparse | 9.361 | +2.784 | +2.270 | 86/100 | 3.8e-15 | −0.012 |
| B2 shuffle | 10.108 | +3.531 | +3.190 | 96/100 | 9.6e-18 | +0.734 |
| **B3 CF** | **12.143** | **+5.566** | **+4.855** | **99/100** | **4.4e-18** | **+2.769** |

invalid：B1 1/600、B2 0/600、B3 1/600（FR mismatch 0）。

## 9. Refold geometry（固定 20 cases × 4 seqs，Boltz-2 steps=200）

| arm | CDR RMSD mean | CDR3 | FR RMSD | pLDDT(CDR) |
|---|---|---|---|---|
| base | 1.860 | 2.347 | 0.559 | 0.623 |
| N3 | 2.090 | 2.709 | 0.578 | 0.614 |
| B1 | 2.154 | 2.803 | 0.567 | 0.609 |
| B2 | 2.064 | 2.635 | 0.554 | 0.616 |
| **B3** | **1.949** | 2.437 | 0.610 | 0.600 |

Paired（per case, 20 cases）：B3 vs base +0.089 Å（7/20, p=0.43）；
B3 vs N3 **−0.141 Å**（11/20, p=0.29）；FR RMSD 不变。

## 10. Reward–geometry Pareto

- B3 同时取得最高 reward（Δ+5.57）与 ≤2.00 Å 的 CDR RMSD（1.949）；N3 的
  reward 只有 B3 的一半且 CDR RMSD 2.090。
- B1/B2 落在 reward 较低但 RMSD 与 N3 相当的区域 → 单纯稀疏化/打乱不能
  同时改善两者。
- 图：`runs/next_stage/figures/figC_dashboard.png`。

**§41 判定：Strong success**（valid100 Δreward ≥ +2.50 且 CDR RMSD ≤ 2.00 Å；
另一分支"reward 不低于 N3 且 RMSD 较 N3 改善 ≥0.10 Å"亦满足：−0.141 Å）。

## 11. Ablations

1. **B1 random sparse ≈ N3**（Δ vs N3 −0.012）：仅减少更新残基数量没有增益。
2. **B2 shuffle > N3**（+0.734）：降低 off-target 已部分有效，但破坏
   residue-credit 对应关系后显著弱于 B3（+2.769）→ credit 对应关系是关键。
3. **Phase C mini pilot**（4 cases / 100 steps）已显示 B3 > B2 > B1 > base，
   全量结果与之一致。
4. **Phase D T1/T2**（held-out 8 cases）：T1 Δ+0.057（4/8）、T2 Δ+0.256
   （3/8），实质无效——σ_seq=0.3137 时训练采样 σ 绝大多数远大于阈值，
   hard gate 200 步中仅 6 步 g=1；smooth 权重整体极小。与 Gate B=WEAK 一致。
5. **combined CF-ST**：因 temporal 单独无效，按 §46/§47 不做主实验。

## 12. Failure analysis

- 结构自洽性（refold RMSD）的 case-level 改善不显著（p≈0.3）：reward 提升
  主要由 CDR 序列层面驱动；CDR3 RMSD 略高于 FR，几何改善集中在总体
  CDR RMSD。
- epitasis：CDR3 的 region credit 与 residue credit 之和差异中位 −0.64，
  单残基 credit 有噪声；B3 仍显著优于全部 control，说明 c_cons 的保守
  聚合有效。
- temporal 无效是**定义问题而非调参问题**：sequence identity 在 diffusion
  轨迹上与 backbone 同步收敛（H3 无窗口），因此任何基于该 σ_seq 的 gate
  都缺乏训练信号。

## 13. Research decision

对应任务书 §66 **情况 B（只有 spatial 有效）**：

- 论文主线暂定为 **counterfactual spatial credit assignment from a
  black-box sequence reward for joint sequence–structure diffusion**；
  不主张 spatiotemporal。
- 已证伪/不支持：H3（temporal separation）、temporal-only 加权（§46）。
- 停止条件满足：Phase A–D 完成 + 本文件；不自动进入 structural critic /
  affinity / Diffusion-GRPO（§65 末）。
- 后续若要恢复 temporal 假设，需要转向 structural proxy reward
  （proteinMPNN LL / BoltzGen IF LL 等，任务书 §50–52）而不是继续调
  classifier reward 的 schedule。

## 产物索引

- credit：`runs/next_stage/counterfactual/{residue_credit.csv,region_credit.csv,pair_credit_summary.csv,credit_stats.json,figures/}`
- 权重：`runs/next_stage/weights/residue_weights.json`
- trajectory：`runs/next_stage/trajectory/{trajectory_metrics.csv,trajectory_sequences.jsonl,trajectory_summary.json,figures/}`
- 训练/选择：`runs/next_stage/{full_cf,full_shuffle,full_random_sparse,d_T1_hard,d_T2_smooth}/`、`checkpoint_selection.{json,md}`
- 评测：`runs/next_stage/valid100_arms_summary.{json,md}`、`refold_phaseC.{json,md}`、`figures/figC_dashboard.png`
