# CF-OPSD Target Probe


## Query probe（train 4 cases × K=8）

| progress | valid rate | median CDR identity vs endpoint | median full identity |
|---|---|---|---|
| 0.60 | 0.000 | 0.291 | 0.792 |
| 0.70 | 0.000 | 0.405 | 0.820 |
| 0.80 | 0.000 | 0.703 | 0.910 |
| 0.90 | 0.906 | 1.000 | 1.000 |

- 选择：**q\* = 90%**（最早满足 valid ≥ 0.90 且 CDR identity ≥ 0.80）
- 状态：**PASS**


## Target construction（q* = 90%）

### Radius sweep（CF target）

| rho (A) | median target-anchor | positive cases | median credit match | invalid | FR mismatch | Gate B |
|---|---|---|---|---|---|---|
| 0.25 | +0.698 | 2/4 | 0.00 | 1.00 | 0 | FAIL |
| 0.5 | n/a | 0/4 | 0.04 | 1.00 | 0 | FAIL |
| 1.0 | n/a | 0/4 | 0.09 | 1.00 | 0 | FAIL |

- selected radius：**None**
- Gate B：**FAIL**

### 失败模式

- rho=0.25 Å：小位移不足以翻转 credited residue 的硬解码身份（credit match ≈ 0），
  且 1/4 anchor 本身无法 decode（invalid 25%）。
- rho=0.5/1.0 Å：位移跨过 hard-decode 边界后目标整体 decode 失败（invalid 100%）。
- 结论：counterfactual sequence credit 不能稳定转换为 bounded atom14 geometric
  target（§32/§73/§74 预警的失败模式）→ 按任务书停止 CF-OPSD。
