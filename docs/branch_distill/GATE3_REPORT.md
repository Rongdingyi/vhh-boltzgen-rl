# Gate 3 — component comparison

| arm | reward8 | Δbase | ΔA | W/T/L vs A | invalid | FR | queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | +0.795 | +0.000 | -1.420 | 1/0/3 | 0 | 0 | 0 |
| A Offline changed DPO | +2.215 | +1.420 | +0.000 | - | 0 | 0 | 0 |
| B Online changed DPO | +3.804 | +3.009 | +1.589 | 3/0/1 | 0 | 0 | 8 |
| C Sibling distill all CDR | +1.065 | +0.270 | -1.150 | 0/0/4 | 0 | 0 | 8 |
| D Sibling distill changed residues | +1.373 | +0.578 | -0.842 | 1/0/3 | 0 | 0 | 8 |

## 组件解释（§55）

- B - A = +1.589（on-policy pair refresh）
- D - B = -2.431
- D - C = +0.307
- D - A = -0.842

## Gate（§54/§56）

```json
{
 "verdict": "NO_GO",
 "delta_D_A": -0.8424436057102866,
 "delta_D_B": -2.4313063612789847,
 "delta_D_C": 0.30738118168665096,
 "cases_D_ge_A": 1,
 "n_cases": 4,
 "invalid_rate": 0.0,
 "fr_mismatch": 0
}
```

protocol_sha256: `c45308687969f675664684c48ae5ec0b261c1d47dc5a47bd2cb56c37a9bbdb29`
