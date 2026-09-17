# Diff-only Multi-seed Verification

## 1. Question

CF-DPO 的稳定收益来自 differing-residue support localization，还是来自 counterfactual credit weighting 本身？

## 2. Frozen Protocol

- 复用 paper-stage frozen protocol（BoltzGen `a3149cf`，beta=10，lr=1e-5，500 updates，24 train / 8 heldout / frozen valid100，8 samples/case）
- primary diff-only 直接复用既有 artifact，未重训：`runs/paper_stage/diff_only/checkpoint_0500.pt` sha256 `c92215f74a77632a…`，seed 20260913
- 新增 diff-only seed 43/44，参数与既有 CF/N3 multiseed 完全一致

## 3. Correctness Audit

- `runs/paper_stage/diffonly_audit/audit_summary.json`：192/192 pairs 通过；20/20 抽样通过
- 每个 pair：support == differing positions、active 权重唯一、w = 1/n_diff、sum = 1、无 same-residue/非 design 位置权重

## 4. Training Seeds

| seed | run | selected step | heldout reward | valid100 tag |
|---:|---|---:|---:|---|
| 20260913 | diff_only (primary) | 500 | - | paper_diff_only_s0500 |
| 43 | diff_only_s43 | 500 | +5.526 | paper_diff_only_s43_s0500 |
| 44 | diff_only_s44 | 500 | +6.495 | paper_diff_only_s44_s0500 |

## 5. Held-out Checkpoint Selection

选择规则：8 held-out cases 上 reward 最高，invalid ≤ base+1pp，FR=0；未使用 valid100 选择（见 `runs/paper_stage/diffonly_multiseed/checkpoint_selection.md`）。

## 6. Valid100 Results

| seed | N3 Δ | Diff-only Δ | CF Δ | CF−Diff | Diff−N3 |
|---:|---:|---:|---:|---:|---:|
| 20260913 | +2.796 | +4.221 | +5.566 | +1.345 | +1.424 |
| 43 | +4.408 | +4.521 | +4.509 | -0.012 | +0.112 |
| 44 | +4.310 | +4.599 | +4.700 | +0.101 | +0.289 |
| mean | 3.838 | 4.447 | 4.925 | 0.478 | 0.609 |
| std | 0.738 | 0.163 | 0.460 | 0.615 | 0.581 |

## 7. Seed-level Comparison

| seed | comparison | mean Δ | median Δ | W/T/L | Wilcoxon p | bootstrap 95% CI |
|---:|---|---:|---:|---|---:|---|
| 20260913 | cf_vs_diff | +1.345 | +0.837 | 81/0/19 | 4.16e-11 | [+0.965, +1.731] |
| 20260913 | diff_vs_n3 | +1.424 | +1.181 | 86/0/14 | 1.48e-13 | [+1.128, +1.739] |
| 43 | cf_vs_diff | -0.012 | -0.021 | 48/0/52 | 9.75e-01 | [-0.264, +0.240] |
| 43 | diff_vs_n3 | +0.112 | +0.099 | 55/0/45 | 3.22e-01 | [-0.096, +0.322] |
| 44 | cf_vs_diff | +0.101 | +0.082 | 52/0/48 | 2.11e-01 | [-0.110, +0.301] |
| 44 | diff_vs_n3 | +0.289 | +0.102 | 57/0/43 | 9.72e-03 | [+0.110, +0.488] |

D_s = CF_s − DiffOnly_s：D_20260913 = +1.345, D_43 = -0.012, D_44 = +0.101；mean(D) = +0.478，std(D) = 0.615，min(D) = -0.012，3/3 positive = False

## 8. Case-level Paired Statistics

每 seed 的分 case 文件：`results/paper_stage/diffonly_per_case_s*.csv`（100 cases 配对；Wilcoxon + 10000 次 paired bootstrap，seed=12345）。

## 9. Interpretation

- mean(CF−Diff) = +0.478，3/3 seeds CF > Diff-only = False
- 所有 seed 的 valid100 invalid/FR 见 `runs/paper_stage/valid100/*.json`；训练 sanity：non-finite=0、FR=0（`runs/paper_stage/diff_only_s*/train_metrics.jsonl`）

## 10. Decision

**Case A**：credit weighting 没有稳定独立价值；当前稳定收益主要来自 support localization。

## 11. Next Step

- 若 Case A：停止 credit refinement（不再 sign/region/adaptive/Shapley）
- 若 Case B/C：下一步只允许固定 support 下的 shrinkage curve η ∈ {0, 0.25, 0.5, 0.75, 1.0}（本任务不执行）
- 完成后 STOP，等待人工审阅
