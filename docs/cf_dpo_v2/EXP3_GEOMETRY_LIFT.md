# CF-DPO v2 — Experiment 3: legal native geometry lift

预登记预算：尝试 32 对训练样本（每对最多 3 个位点、双向），
接受阈值 acceptance ≥ 50%、FR mismatch = 0、同序列 M=2 覆盖率 ≥ 80%。

## 结果

- 尝试边：192；精确 lift：166（acceptance = 86.5%）
- FR mismatch 尝试：0
- 同序列 M=2 覆盖率：100.0%

### 按符号类别

| class | attempts | exact | rate |
|---|---|---|---|
| both_negative | 60 | 55 | 0.92 |
| both_positive | 64 | 56 | 0.88 |
| sign_flip | 68 | 55 | 0.81 |

### 失败原因

| reason | count |
|---|---|
| decode_invalid | 9 |
| exact | 166 |
| third_sequence | 17 |

## Gate（§14：合法几何补全不稳定则不进入训练）

**PASS** — acceptance=86.5%，
FR mismatch=0，same-seq coverage=100.0%。

失败样本未静默丢弃：逐边记录在 `lift_attempts.csv`（含类别、残基对、位移 RMS、失败原因）。
