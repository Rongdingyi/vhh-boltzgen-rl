# CF-DPO v2 Errata（审阅发现的 5 个正确性问题与修复）

审阅者（用户）对 `de8f5c9` 的复核发现 4 个会实质改变训练方向的错误，加上由它们
污染的 proxy 校验，共 5 项。全部确认属实并已修复；受影响结果全部标记为无效，
`CF_DPO_V2_DECISION.md` 的 NO-GO 判定撤回。

## E1 — `winner_drop` 的 dR 方向写反（致命）

约定（trainer、small model、proxy 全部一致）：`a -> b, dR = R(b) - R(a)`。
`graph_dataset.build_graph` 中 drop 边为 `a=winner, b=CF`，却保存
`dR = R(winner) - R(CF)`，符号相反 → 该类偏好被反向训练。

**修复**：drop 边改为 `dR = R(CF) - R(winner)`（gain 边原本正确）。

## E2 — counterfactual node ID 冲突导致静默覆盖（致命）

`node_id = f"{cid}:{direction}:{pos}"` 在同一 case 的多条 pair 上会重复；
`Graph.add_node` 直接覆盖，导致 edge 的 label 与 coords/sequence/reward 不匹配。

**修复**：
- `node_id = f"{cid}:{winner_sample}:{loser_sample}:{direction}:{pos}"`（即 cf_id 语义）；
- `Graph.add_node` 对重复 id **抛错**（不再静默覆盖）；
- 新增回归测试 `tests/test_cfd2_graph_conventions.py`（唯一性 + 方向不变量）。

## E3 — same-seq 边被双重采样（约 25% → 实际 ~53%）

`local_edges` 过滤条件是 `kind != "global"`，把 same_seq 也包含进去；显式
same-seq 分支之外，preference 分支又按 75% 抽 local pool → same-seq 总占比约
`0.25 + 0.75*0.75*(167/334) ≈ 53%`，真正 drop/gain 只有 ~28%。

**修复**：
- 采样逻辑抽到 `src/vhh_rl/cf_dpo_v2/sampling.py::choose_edge`，分支互斥：
  `same_seq`（v2，p=ratio）/ `drop|gain`（p=(1-ratio)*0.75）/ `global`（其余）；
- 新增回归测试 `tests/test_cfd2_sampling.py` 断言 25% ± 2%。

## E4 — Exp2 小模型 baseline 与 drop 边方向写反

- `current_cf` / `sign_only`：`z = H(loser) - H(winner)`，而 target 是
  `sigmoid(R(winner)-R(loser)) > 0.5` → 优化方向完全相反；
- small-model 的 drop 边同样保存成 `R(winner) - R(drop_seq)`。

**修复**：weighted 基线改为 `z = H(winner) - H(loser)`；drop 边 dR 取
`-d_drop`（即 `R(drop_seq) - R(winner)`）。原报告的
"weighted 符号正确率 0.18–0.20" 无效，需重跑。

## E5 — proxy 校验的 ground truth 被污染

proxy 脚本用 `edge["dR"]`（含 E1 错误符号、E2 覆盖后的错配）作为真值，
且 `[:max_edges]` 顺序截断（非随机/分层），因此 `sign agreement = 0.475`
混合了真实 proxy mismatch + 错误 drop 标签 + node 覆盖三种因素，不能作为
NO-GO 证据。

**修复**：
- ground truth 改为**从节点 reward 重算** `R(b) - R(a)`，并报告
  stored-vs-node 标签不一致数量（修复后应为 0）；
- 采样改为按 kind 分层随机（固定 seed）。

## 修复后仍然有效 / 需要重跑

| 项目 | 状态 |
|---|---|
| Exp1 数据复核（36.1% flip、94% 稀疏性修正） | 有效（仅用既有 CSV） |
| Exp3 geometry lift 接受率/FR/同序列覆盖 | 有效（不依赖边方向与 ID） |
| CF-DPO-mini +4.39（matched baseline） | 有效（用既有 CF-DPO 训练） |
| Exp2 weighted vs signed 对比 | **无效，需重跑**（E4） |
| Exp4 signed/v2 pilot 表 | **无效，需重跑**（E1/E2/E3） |
| proxy sign agreement 0.475 | **无效，需重跑**（E5 + E1/E2） |
| `CF_DPO_V2_DECISION.md` 的 NO-GO | **撤回**，见该文档新状态 |

## 复现入口

```bash
bash run.sh cfd2-build-graph     # CPU：重建比较图（含方向/ID 断言）
bash run.sh cfd2-exp2            # CPU：Exp2 修正后重跑
bash run.sh cfd2-train-signed    # GPU：signed
bash run.sh cfd2-train-v2        # GPU：v2
bash run.sh cfd2-proxy           # GPU：修正后 proxy 校验
```
