# AG-CF-DPO Audit（task book §8-§15）

来源：`runs/next_stage/counterfactual/{residue,region}_credit.csv` +
`runs/next_stage/weights/residue_weights.json`（只读，未改动）。

## 1. 规模

- pairs 192，regions 548，cases 24
- TOL=0.05，RHO=0.7（预注册，未调参）
- residue 类别比例：`{"stable_positive": 0.4395861148197597, "stable_negative": 0.16154873164218958, "sign_flip": 0.3324432576769025, "weak": 0.0664218958611482}`
- region 类别计数：`{"stable_positive_region": 434, "region_flip": 57, "stable_negative_region": 44, "region_weak": 13}`

## 2. Current-CF conflict floor leakage（§12）

- conflict mass：median 0.1250，
  mean 0.1197，
  P25 0.0962，P75 0.1429
- conflict mass > 0.10 的 pair 比例：0.688
- uncertain mass（含 weak）：median 0.1408，
  > 0.25 比例 0.000

## 3. Coarse-rescue coverage（§13）

- residue candidate regions：137
- **coarse rescue regions：299**
- abstain regions：112
- adaptive 预演 mode counts：`{"region": 299, "residue": 137, "abstain": 112}`

## 4. Gate A（§14）

| condition | value | threshold | verdict |
|---|---:|---:|---|
| A median current-CF conflict mass | 0.1250 | 0.08 | PASS |
| B coarse-rescue region share | 0.5456 | 0.2 | PASS |
| C pairs with coarse-rescue share | 0.9479 | 0.2 | PASS |

**Gate A：PASS**（任意一条成立即通过；FAIL → 不训练）

## 5. Figures

- fig1 conflict weight mass
- fig2 rho_positive by CDR
- fig3 residue-vs-region reliability
- fig4 adaptive mode counts
- fig5 epistasis vs selected granularity
