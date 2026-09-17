# CF-DPO Paper-stage Results

任务书：`CF_DPO_PAPER_STAGE_EXPERIMENT_PLAN.md`；执行范围：P0（multi-seed /
credit ablations / expanded structure）+ P1（query budget / second reward）；
P2 不执行（见 `P2_GENERALIZATION.md`）。

## 1. Frozen Protocol

`experiments/paper_stage/configs/FROZEN_PROTOCOL.yaml`（version 1）：
base ckpt sha256 `360af8bd…`、pair dataset sha256 `fd9776fd…`、
BoltzGen commit `a3149cf`、scorer ensemble 6 个权重 + OOD reference 全部
固化；β=10、η=0.75、lr=1e-5、500 steps、trainable = `structure_module.score_model`；
held-out 8 cases、valid100 manifest sha256 `d5c689179…`；
eval generation seeds = `seed_base+700000`、8 samples/case、50 steps；
refold 协议：Boltz-2 recycling=3、steps=200；Protenix-v2 独立折叠。
`run.sh paper-freeze-protocol` 可重复校验。

## 2. Current Starting Point

| 方法 | valid100 Δ vs base（primary seed） |
|---|---|
| N3 Uniform Fake | +2.796 |
| Shuffle (B2) | +3.531 |
| **CF-DPO (B3)** | **+5.566** |

## 3. Multi-seed Robustness（§5–§12）

seeds = primary(20260913) / 43 / 44；评测 seeds 固定。

| seed | CF | N3 | Shuffle |
|---|---|---|---|
| primary | +5.566 | +2.796 | +3.531 |
| 43 | +4.509 | +4.408 | +3.566 |
| 44 | +4.700 | +4.310 | +3.728 |
| **mean ± std** | **+4.925 ± 0.563** | +3.838 ± 0.904 | +3.608 ± 0.105 |

| gate | 结果 |
|---|---|
| 3/3 seeds CF Δ > +4.0 | **PASS** |
| mean(CF − N3) > +2.0 | **FAIL（+1.087）**；seed 43/44 上 CF ≈ N3 |

排查（§63）后确认不是 instability：N3 的训练 seed 方差本身很大
（+2.80/+4.41/+4.31），CF 更稳定。**primary seed 的 +2.77 差距不稳健。**

## 4. Credit Attribution Ablations（§13–§20）

| 方法 | spatial support | credit | 方向 | queries/pair | valid100 Δ |
|---|---|---|---|---|---|
| N3 Uniform | all CDR | uniform | — | 0 | +2.796 |
| Shuffle | differing | 真权重打乱 | — | 31.2 | +3.531 |
| Diff-only Uniform | differing | uniform | — | 0 | +4.221 |
| Drop-only | differing | `max(c_drop,0)` | winner→loser | 15.6 | +4.932 |
| Gain-only | differing | `max(c_gain,0)` | loser→winner | 15.6 | +5.258 |
| **CF-DPO** | differing | `c_cons`+floor | **双向** | 31.2 | **+5.566** |

- **必须证明项：CF (+5.566) > Diff-only (+4.221)，Δ=+1.345 ✓**
- 排序符合 §20 理想图景：CF > one-sided > diff-only ≳ shuffle > N3。
- 增益分解：N3→diff-only +1.43（off-target 更新是主要损失）、
  diff-only→one-sided +0.71/+1.04、one-sided→CF +0.31/+0.63。

## 5. Expanded Structural Validation（Boltz-2，100 cases × 2）

| arm | CDR RMSD | CDR3 | FR | pLDDT(CDR) |
|---|---|---|---|---|
| Base | 1.807 | 2.296 | 0.530 | 0.640 |
| **N3** | **1.780** | **2.237** | 0.552 | 0.630 |
| Shuffle | 1.907 | 2.454 | 0.535 | 0.628 |
| CF | 1.854 | 2.355 | 0.537 | 0.614 |

CF vs N3 = **+0.074 Å**（95% CI [−0.056, +0.202]，Wilcoxon p=0.174）；
CF vs base +0.047（p=0.139）；CF vs shuffle −0.053（p=0.449）。

**20-case 的 favorable trend（−0.141 Å）未能在 100 cases 复现**：CF 不优于
N3（略差、不显著）。按 §62 结构 claim 收缩为"无显著结构退化"。

## 6. Independent Refolder Validation（Protenix-v2，50 cases × 2）

| arm | CDR RMSD |
|---|---|
| Base | 6.827 |
| **N3** | **5.134** |
| CF | 5.915 |

CF vs base −0.911 Å（p=0.046）；CF vs N3 +0.781 Å（p=0.184）。
两个 refolder 一致结论：**N3 的 CDR RMSD 最好，CF 不优于 N3**；
CF-vs-base 方向两个 refolder 不一致（Boltz-2 +0.05，Protenix −0.91），
按要求分别报告、不取平均。

## 7. Query Budget（§31–§35）

| budget | queries/pair | Δ reward | 占全量 CF 增益 | wins |
|---|---|---|---|---|
| 25% | 7.7 | +4.266 | 76.6% | 94/100 |
| 50% | 15.5 | +4.720 | **84.8%** | 98/100 |
| 75% | 23.5 | +4.976 | 89.4% | 96/100 |
| 100% | 31.2 | +5.566 | 100% | 99/100 |

§34 目标（50% → ≥80%）达成。单向变体在 50% query 成本下达到 89–95% 增益。

## 8. Second Reward Generalization（§36–§43）

Second reward = VHH-ESM-C exact masked CDR PLL（与主 reward Spearman −0.04，
median per-case std 0.165 → 独立 gate 通过）。同一 CF-DPO 流程重跑：

| arm | held-out cdr_pll | Δ vs base | wins vs base |
|---|---|---|---|
| Base | −2.056 | — | — |
| Uniform | −1.717 | +0.339 | 8/8 (p=0.0078) |
| **CF** | **−1.690** | +0.366 | 8/8 (p=0.0078) |

**CF vs Uniform：+0.0269，6/8 cases 更优**（p=0.38）→ §42 pilot 通过
（方向性跨 reward 复现；幅度远小于主 reward 的单 seed 差距）。

## 9. Optional Task/Model Generalization

未执行（P2）。决策与理由见 `P2_GENERALIZATION.md`。

## 10. Final Main Table

| Method | Δ reward (3 seeds) | wins | invalid | CDR RMSD (Boltz-2) |
|---|---|---|---|---|
| N3 | +3.838 ± 0.904 | 95-98/100 | ~0 | 1.780 |
| Shuffle | +3.608 ± 0.105 | 96-100/100 | ~0 | 1.907 |
| Diff-only | +4.221 (1 seed) | 98/100 | 0 | — |
| Drop-only | +4.932 (1 seed) | 94/100 | 0 | — |
| Gain-only | +5.258 (1 seed) | 96/100 | 0 | — |
| **CF-DPO** | **+4.925 ± 0.563** | 97-99/100 | ~0 | 1.854 |

## 11. Final Figures

`results/paper_stage/figures/`：
- figA credit sparsity（top-k mass + n_eff/n_diff=0.27）
- figB attribution ablation bars
- figC reward–structure Pareto（Boltz-2 + Protenix 两面板）
- figD query-budget curve
- figE multi-seed robustness

## 12. Statistical Summary

- 单位一律 case-level；valid100 每 case 先平均 8 sequences 再 paired。
- Wilcoxon + paired bootstrap 95% CI（seed 12345，10k 重采样）。
- Multi-seed 报告每 seed 的 case-level 结果 + seed-level mean±std，
  未把 3×100 当作 300 iid 样本（§52）。
- 结构统计 case-level（2 seqs/case），两个 refolder 分开报告（§53）。

## 13. Failure Cases

1. **Multi-seed gate 失败**：CF − N3 平均 +1.09（要求 +2.0），
   2/3 seeds CF ≈ N3。
2. **结构趋势未复现**：100-case Boltz-2 与 50-case Protenix 均显示 N3 的
   CDR RMSD 最优；CF 的 20-case 优势是噪声。
3. **Second reward 增益幅度小**：+0.027（6/8），仅方向性证据；
   完整 valid100 未跑。
4. 800 folds 中 8 个失败（各 arm 均匀），Protenix 用 50 cases。

## 14. Paper Claims Supported / Unsupported

| claim | 状态 |
|---|---|
| C1: global preference hides sparse residue credit（15.6 diff/pair; top30%=94%; n_eff=0.27） | **支持** |
| C2: correct residue-credit correspondence（CF > diff-only > shuffle > N3；random sparse ≈ N3） | **支持（强）** |
| C3a: 跨训练随机性（3 seeds 一致 ≥ controls） | 支持（方向），但幅度不满足原 gate |
| C3b: 跨 reward（第二 reward 上 CF > Uniform） | 支持（pilot，6/8） |
| C3c: 无额外结构退化 | 支持（Boltz-2 不显著；Protenix CF<N3 但 >base） |
| ~~CF 大幅超越 N3（≫）~~ | **不支持**（3-seed 均值 +1.09） |
| ~~CF improves structural consistency~~ | **不支持**（§62） |

## 15. Decision

- **不是 "Ready for paper writing"（按原 headline claim 不可写）；也不是
  "method repair"（方法本身没有失效，credit attribution 因果链完整）。**
- 结论：**Need one more generalization experiment + claim re-scope**：
  1. 论文主线改为 attribution-chain：*counterfactual residue-credit
     correspondence，而非 sparsity/uniform*，证据为 C1+C2+query-budget；
  2. 补第二 reward 的 valid100（当前只到 pilot）与更小方差的多 seed
     （如需定量 gate）；
  3. 结构 claim 按 §62 写"no significant degradation"；
  4. 完成上述后可进入写作；Pallatom/第二 task 仍留到下一阶段。

## 交付物清单（§69）

```text
docs/paper_stage/
├── P0_MULTISEED.md              ✓
├── P0_CREDIT_ABLATIONS.md       ✓
├── P0_STRUCTURE_VALIDATION.md   ✓
├── P1_QUERY_BUDGET.md           ✓
├── P1_SECOND_REWARD.md          ✓
├── P2_GENERALIZATION.md         ✓
└── PAPER_STAGE_RESULTS.md       ✓（本文件）

results/paper_stage/
├── multiseed_summary.csv        ✓（含 multiseed_seed_level.csv）
├── ablation_summary.csv         ✓
├── structure_boltz2_per_case.csv ✓
├── structure_protenix_per_case.csv ✓
├── query_budget.csv             ✓
├── second_reward_summary.csv    ✓
├── paper_main_table.csv         ✓
└── figures/figA–figE.png        ✓
```

**停在此处（§71）：不自动进入 Pallatom / 新 RL / structural critic。**

## Diff-only Multi-seed Re-audit



### 1. Question

CF-DPO 的稳定收益来自 differing-residue support localization，还是来自 counterfactual credit weighting 本身？

### 2. Frozen Protocol

- 复用 paper-stage frozen protocol（BoltzGen `a3149cf`，beta=10，lr=1e-5，500 updates，24 train / 8 heldout / frozen valid100，8 samples/case）
- primary diff-only 直接复用既有 artifact，未重训：`runs/paper_stage/diff_only/checkpoint_0500.pt` sha256 `c92215f74a77632a…`，seed 20260913
- 新增 diff-only seed 43/44，参数与既有 CF/N3 multiseed 完全一致

### 3. Correctness Audit

- `runs/paper_stage/diffonly_audit/audit_summary.json`：192/192 pairs 通过；20/20 抽样通过
- 每个 pair：support == differing positions、active 权重唯一、w = 1/n_diff、sum = 1、无 same-residue/非 design 位置权重

### 4. Training Seeds

| seed | run | selected step | heldout reward | valid100 tag |
|---:|---|---:|---:|---|
| 20260913 | diff_only (primary) | 500 | - | paper_diff_only_s0500 |
| 43 | diff_only_s43 | 500 | +5.526 | paper_diff_only_s43_s0500 |
| 44 | diff_only_s44 | 500 | +6.495 | paper_diff_only_s44_s0500 |

### 5. Held-out Checkpoint Selection

选择规则：8 held-out cases 上 reward 最高，invalid ≤ base+1pp，FR=0；未使用 valid100 选择（见 `runs/paper_stage/diffonly_multiseed/checkpoint_selection.md`）。

### 6. Valid100 Results

| seed | N3 Δ | Diff-only Δ | CF Δ | CF−Diff | Diff−N3 |
|---:|---:|---:|---:|---:|---:|
| 20260913 | +2.796 | +4.221 | +5.566 | +1.345 | +1.424 |
| 43 | +4.408 | +4.521 | +4.509 | -0.012 | +0.112 |
| 44 | +4.310 | +4.599 | +4.700 | +0.101 | +0.289 |
| mean | 3.838 | 4.447 | 4.925 | 0.478 | 0.609 |
| std | 0.738 | 0.163 | 0.460 | 0.615 | 0.581 |

### 7. Seed-level Comparison

| seed | comparison | mean Δ | median Δ | W/T/L | Wilcoxon p | bootstrap 95% CI |
|---:|---|---:|---:|---|---:|---|
| 20260913 | cf_vs_diff | +1.345 | +0.837 | 81/0/19 | 4.16e-11 | [+0.965, +1.731] |
| 20260913 | diff_vs_n3 | +1.424 | +1.181 | 86/0/14 | 1.48e-13 | [+1.128, +1.739] |
| 43 | cf_vs_diff | -0.012 | -0.021 | 48/0/52 | 9.75e-01 | [-0.264, +0.240] |
| 43 | diff_vs_n3 | +0.112 | +0.099 | 55/0/45 | 3.22e-01 | [-0.096, +0.322] |
| 44 | cf_vs_diff | +0.101 | +0.082 | 52/0/48 | 2.11e-01 | [-0.110, +0.301] |
| 44 | diff_vs_n3 | +0.289 | +0.102 | 57/0/43 | 9.72e-03 | [+0.110, +0.488] |

D_s = CF_s − DiffOnly_s：D_20260913 = +1.345, D_43 = -0.012, D_44 = +0.101；mean(D) = +0.478，std(D) = 0.615，min(D) = -0.012，3/3 positive = False

### 8. Case-level Paired Statistics

每 seed 的分 case 文件：`results/paper_stage/diffonly_per_case_s*.csv`（100 cases 配对；Wilcoxon + 10000 次 paired bootstrap，seed=12345）。

### 9. Interpretation

- mean(CF−Diff) = +0.478，3/3 seeds CF > Diff-only = False
- 所有 seed 的 valid100 invalid/FR 见 `runs/paper_stage/valid100/*.json`；训练 sanity：non-finite=0、FR=0（`runs/paper_stage/diff_only_s*/train_metrics.jsonl`）

### 10. Decision

**Case A**：credit weighting 没有稳定独立价值；当前稳定收益主要来自 support localization。

### 11. Next Step

- 若 Case A：停止 credit refinement（不再 sign/region/adaptive/Shapley）
- 若 Case B/C：下一步只允许固定 support 下的 shrinkage curve η ∈ {0, 0.25, 0.5, 0.75, 1.0}（本任务不执行）
- 完成后 STOP，等待人工审阅
