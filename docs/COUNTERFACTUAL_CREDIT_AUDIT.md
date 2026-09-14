# Counterfactual Credit Audit（Phase A1/A2，Gate A）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §10–§19
数据：24 train cases / 192 pairs；2996 个 differing positions；
7231 条 unique counterfactual sequences + originals（batch scorer + SHA1 cache）。

## 1. Residue credit（双向 intervention）

| 指标 | 数值 |
|---|---|
| c_drop vs c_gain Pearson | +0.3965 |
| sign agreement rate (median) | 0.667 |
| positive_fraction (median) | 0.471 |
| original score drift (max abs diff) | 0.00e+00 |

## 2. Sparsity（Gate A）

| 指标 | median |
|---|---|
| top10% positive-credit mass | 0.595 |
| top20% | 0.814 |
| **top30%** | **0.940** |
| top50% | 1.000 |
| **n_eff / n_diff** | **0.270** |

**Gate A decision: STRONG SUPPORT**
（Strong: top30 >= 0.60 且 n_eff/n_diff <= 0.60；Partial: top30 0.45–0.60；
Weak: top30 < 0.45 且 n_eff/n_diff > 0.75）

## 3. Region credit（A2）

| region | pairs w/ changes | residue c_cons median | G_cons median | G_avg median | sign agree | epistasis (G_avg - sum c_avg) median |
|---|---|---|---|---|---|---|
| cdr1 | 178 | 0.094 | 1.098 | 2.171 | 0.837 | +0.000 |
| cdr2 | 178 | 0.091 | 1.432 | 2.438 | 0.871 | +0.000 |
| cdr3 | 192 | 0.000 | 3.685 | 4.408 | 0.948 | -0.641 |

Dominant region（|G_avg| 最大）：{"cdr1": 45, "cdr3": 105, "cdr2": 42}

## 4. Reward-gap explanation（仅诊断）

corr(C_sum(c_avg), reward_gap)：Pearson +0.768，
Spearman +0.791（不要求加性）。

## 5. 结论

- 依据 Gate A 决定 Phase C 采用 residue-level sparse CF weighting 还是 region-level /
  temporal 优先（任务书 §16/§33）。
- 图：`runs/next_stage/counterfactual/figures/fig1–fig6`。

*脚本：`scripts/next_build_counterfactuals.py`、`next_score_counterfactuals.py`、
`next_analyze_credit.py`；credit 公式：`src/vhh_rl/credit/credit_metrics.py`*
