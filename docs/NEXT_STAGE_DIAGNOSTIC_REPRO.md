# 下一阶段诊断复现（Phase A0）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §2/§7
数据：`runs/native_pool/pairs_train.jsonl`、`runs/native_pool/train/*/metadata.jsonl`、
`runs/native_pool/native_refold_results_per_sample.csv`、`native_refold_design/manifest.csv`

## 1. Pair Hamming 统计（192 pairs）

| 统计量 | 复现值 |
|---|---|
| n_pairs | 192 |
| 总差异位点 | 2996 |
| Hamming mean / median | 15.60 / 15.5 |
| Hamming min / max | 3 / 31 |
| Hamming p10/p25/p75/p90 | 8.0 / 12.0 / 20.0 / 23.0 |
| Pearson(Hamming, reward gap) | +0.0699 |
| Spearman(Hamming, reward gap) | +0.0359 |

## 2. Refold 相关性（400 samples）

| group | metric | n | Pearson | Spearman |
|---|---|---|---|---|
| all | cdr_rmsd | 400 | +0.019 | +0.086 |
| all | cdr3_rmsd | 400 | -0.090 | +0.001 |
| all | plddt_cdr | 400 | -0.563 | -0.593 |
| all | cdr_recovery | 400 | -0.447 | -0.435 |
| base | cdr_rmsd | 80 | -0.043 | -0.016 |
| base | cdr3_rmsd | 80 | -0.174 | -0.108 |
| base | plddt_cdr | 80 | -0.625 | -0.624 |
| base | cdr_recovery | 80 | -0.494 | -0.466 |
| n1 | cdr_rmsd | 80 | +0.052 | +0.128 |
| n1 | cdr3_rmsd | 80 | -0.106 | +0.004 |
| n1 | plddt_cdr | 80 | -0.650 | -0.680 |
| n1 | cdr_recovery | 80 | -0.503 | -0.449 |
| n2 | cdr_rmsd | 80 | +0.039 | +0.098 |
| n2 | cdr3_rmsd | 80 | -0.062 | +0.052 |
| n2 | plddt_cdr | 80 | -0.486 | -0.517 |
| n2 | cdr_recovery | 80 | -0.429 | -0.401 |
| n3 | cdr_rmsd | 80 | -0.101 | +0.064 |
| n3 | cdr3_rmsd | 80 | -0.199 | -0.041 |
| n3 | plddt_cdr | 80 | -0.532 | -0.585 |
| n3 | cdr_recovery | 80 | -0.376 | -0.388 |
| n4 | cdr_rmsd | 80 | +0.071 | +0.123 |
| n4 | cdr3_rmsd | 80 | -0.014 | +0.067 |
| n4 | plddt_cdr | 80 | -0.509 | -0.544 |
| n4 | cdr_recovery | 80 | -0.454 | -0.453 |

## 3. 结论

- 一个 preference pair 平均约 `15.6` 个 CDR 残基不同，但差异数与 reward gap
  基本无关（Pearson +0.070）——credit 分配问题成立。
- classifier reward 与 refold 结构指标几乎不相关（CDR RMSD Pearson +0.019），
  与 pLDDT 呈明显反向——单靠 sequence reward 无法提供结构自洽性信号。

*复现脚本：`scripts/next_reproduce_stats.py`*
