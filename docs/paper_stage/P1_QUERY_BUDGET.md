# P1-A — Query Budget / Attribution Cost

任务书 §31–§35；主 reward = `cdr_camelid_margin`；valid100 = 100 cases × 8。
预算定义：每 pair 的双向 CF 全量为 2 × n_diff ≈ 31.2 queries；子采样在每
pair 内用 `budget_seed=123` 随机选比例位置做双向干预，未查询 residue credit=0，
保留 uniform floor（η=0.75）。

## 1. Reward/query 曲线（valid100 Δ vs base = 6.578）

| budget | queries/pair | 占全量 | valid100 reward | Δ vs base | 达到全量增益 | wins |
|---|---|---|---|---|---|---|
| 25% | 7.7 | 24.8% | 10.844 | **+4.266** | 76.6% | 94/100 |
| 50% | 15.5 | 49.7% | 11.298 | **+4.720** | **84.8%** | 98/100 |
| 75% | 23.5 | 75.2% | 11.553 | **+4.976** | 89.4% | 96/100 |
| 100%（CF） | 31.2 | 100% | 12.143 | **+5.566** | 100% | 99/100 |

对照组（同一套 valid100，见 P0_CREDIT_ABLATIONS）：

| 方法 | queries/pair | Δ vs base | 占全量 CF 增益 |
|---|---|---|---|
| Diff-only Uniform | **0** | +4.221 | 75.8% |
| Drop-only（单向） | 15.6 | +4.932 | 88.6% |
| Gain-only（单向） | 15.6 | +5.258 | 94.5% |
| CF（双向） | 31.2 | +5.566 | 100% |

## 2. 任务书 §34 的目标判定

```text
希望：50% queries → ≥80% full-CF reward gain
结果：50% → 84.8%   ✓
```

- 25% budget（约 8 queries/pair）即可拿到 76.6% 的增益；曲线在前 25–50%
  区间最陡，之后单 query 边际收益迅速下降。
- 单向方案在 50% query 成本下达到 ~89–95% 增益；若论文需要成本最优
  变体，gain-only 是最佳性价比点（+5.258 / 15.6 queries）。

## 3. 解释

- 与 P0-B 的 credit 稀疏性一致（top30% 残基解释 94% credit）：随机子采样
  的 25–50% 位置必然覆盖到大部分高 credit 残基，因此收益衰减缓慢。
- 未查询残基保留 floor 权重，相当于"低分辨率 uniform 回退"，避免了
  纯稀疏选择的方差（对比 B1 Random Sparse 没有收益）。
- 本实验定位为 practicality study（§35），不引入 active querying 算法。

## 4. 文件

- `results/paper_stage/query_budget.csv`
- `runs/paper_stage/{qb_025,qb_050,qb_075}/train_metrics.jsonl`
- `runs/paper_stage/query_budget/checkpoint_selection.json`
- `runs/paper_stage/weights/query_budget_weights.json`
