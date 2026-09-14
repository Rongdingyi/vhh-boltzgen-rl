# CF-OPSD Feasibility Decision

## 1. Motivation
把 counterfactual residue credit 转成显式 atom14 几何 target，以 on-policy self-distillation 替代/补充 CF-DPO。

## 2. Reference Difference
DiffusionOPSD 依赖可微 reward gradient；BoltzGen 的 reward 经由硬 `res_from_atom14` 得到离散序列，不可微。本探索用 black-box counterfactual credit 构造 bounded positive target。

## 3. Query Audit
- q* = 90%，status = PASS

## 4. Target Construction
- rho=0.25: gain median 0.6976213455200195, match 0.0, invalid 1.00, gate=False
- rho=0.5: gain median None, match 0.038461538461538464, invalid 1.00, gate=False
- rho=1.0: gain median None, match 0.08846153846153847, invalid 1.00, gate=False
- selected radius: None

## 5. Same-query Realization
- (未运行)

## 6. Static Pilot
- (未运行)

## 7. On-policy Pilot
- 仅在 Gate D PASS 后运行；见 runs/cf_opsd/onpolicy_v1/。

## 8. Efficiency
- optimizer updates / scorer queries 统计见各 probe 产物。

## 9. Failure Modes
- 详见各阶段文档与 gate JSON。

## 10. Final Decision

### NO-GO
Target 无法从 counterfactual credit 稳定构造（Gate B FAIL，§32）。

## 11. Recommendation
- keep CF-DPO main line（除非上表 GO）
- 不实现 soft decoder / negative target / structural critic（§5/§65/§95）
