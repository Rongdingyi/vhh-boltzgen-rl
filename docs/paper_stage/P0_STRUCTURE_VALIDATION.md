# P0-C — Expanded Structural Validation（Boltz-2, 100 cases × 2 sequences）

任务书 §21–§30；协议冻结见 `experiments/paper_stage/configs/FROZEN_PROTOCOL.yaml`。

## 1. 设置

- 100 个 valid100 case，每 case 每 arm 取 2 条序列（固定从冻结的 valid100 pool
  按 sample_id 顺序取前 2 条有 reward、FR-clean 的样本；与 reward 评测同源）。
- Arms：Base / N3 / Shuffle(B2) / CF-DPO(B3)，共 800 folds。
- Boltz-2 refold：`boltz2_conf_final.ckpt`，recycling=3，sampling_steps=200，
  diffusion_samples=1（与 round-1 完全一致）。
- 792/800 折叠成功（8 个失败，各 arm 均匀分布）。
- 统计单位：case（先对每 case 的 2 条序列取均值，再做 paired 统计）。

## 2. 主结果（case-level，n=100）

| metric | base | n3 | shuffle | cf | cf-base | cf-n3 | cf-shuffle |
|---|---|---|---|---|---|---|---|
| **CDR RMSD (Å)** | **1.807** | **1.780** | **1.907** | **1.854** | **+0.047** | **+0.074** | **−0.053** |
| CDR3 RMSD | 2.296 | 2.237 | 2.454 | 2.355 | +0.059 | +0.118 | −0.100 |
| FR RMSD | 0.530 | 0.552 | 0.535 | 0.537 | +0.007 | −0.015 | +0.002 |
| pLDDT(CDR) | 0.640 | 0.630 | 0.628 | 0.614 | −0.025 | −0.016 | −0.014 |

Paired CDR RMSD（per case）：

| comparison | mean Δ | median Δ | W/T/L | 95% CI | Wilcoxon p | Cliff's d |
|---|---|---|---|---|---|---|
| CF vs base | +0.047 | +0.047 | 43/0/57 | [−0.109, +0.194] | 0.139 | +0.048 |
| CF vs N3 | +0.074 | +0.081 | 46/0/54 | [−0.056, +0.202] | 0.174 | +0.066 |
| CF vs shuffle | −0.053 | −0.056 | 55/0/45 | [−0.201, +0.092] | 0.449 | −0.009 |

## 3. 与 20-case 结果的关系（关键）

| 指标 | 20 cases（next-stage） | 100 cases（本轮） |
|---|---|---|
| CF CDR RMSD | 1.949 | 1.854 |
| N3 CDR RMSD | 2.090 | 1.780 |
| CF − N3 | **−0.141 Å（CF 更好）** | **+0.074 Å（CF 略差）** |

**结论：20-case 的 favorable trend 未能复现。** 100-case 下 CF 相对 N3 平均
+0.074 Å（未超过 §27 的 0.10 Å 阈值，且 Wilcoxon p=0.17、bootstrap CI 跨 0），
方向由"更好"翻转为"略差但不显著"。

## 4. 按任务书 §62 调整后的结构 claim

禁止写：

```text
CF-DPO improves structural consistency / refold geometry
```

改为：

```text
CF-DPO substantially improves reward alignment
without catastrophic validity collapse
(no significant structural degradation; CDR RMSD within ±0.10 Å of N3)
```

- validity：invalid ≤ 1%、FR mismatch = 0（valid100 与 held-out 均满足）。
- FR RMSD 全部无差异（0.53–0.55 Å），说明架构层未被破坏。
- CF 相比 shuffle 略好（−0.053 Å），说明至少不劣于另一 attribution control。

## 5. 文件

- `results/paper_stage/structure_boltz2_per_case.csv`
- `runs/paper_stage/structure_boltz2/{results_per_sample.csv,stats.json,stats.md}`
- `docs/paper_stage/P0_STRUCTURE_VALIDATION.md`（本文件）
