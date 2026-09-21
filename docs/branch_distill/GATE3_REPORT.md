# Gate 3 — component comparison

| arm | reward8 | Δbase | ΔA | W/T/L vs A | invalid | FR | queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | +1.271 | +0.000 | -1.205 | 2/0/6 | 0 | 0 | 0 |
| A Offline changed DPO | +2.476 | +1.205 | +0.000 | - | 0 | 0 | 0 |
| B Online changed DPO | +3.070 | +1.798 | +0.594 | 4/0/4 | 0 | 0 | 32 |
| C Sibling distill all CDR | +1.358 | +0.086 | -1.118 | 1/0/7 | 0 | 0 | 32 |
| D Sibling distill changed residues | +1.556 | +0.285 | -0.920 | 3/0/5 | 0 | 0 | 32 |

## 组件解释（§55）

- B - A = +0.594（on-policy pair refresh）
- D - B = -1.514
- D - C = +0.198
- D - A = -0.920

## Gate（§54/§56）

```json
{
 "verdict": "NO_GO",
 "delta_D_A": -0.9201865532377269,
 "delta_D_B": -1.5138378227420617,
 "delta_D_C": 0.19817563131800853,
 "cases_D_ge_A": 3,
 "n_cases": 8,
 "invalid_rate": 0.0,
 "fr_mismatch": 0
}
```

protocol_sha256: `c45308687969f675664684c48ae5ec0b261c1d47dc5a47bd2cb56c37a9bbdb29`
