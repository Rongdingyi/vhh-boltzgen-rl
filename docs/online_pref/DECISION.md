# Online-preference route — decision record

## Phase 0（online sibling refresh, all-changed support）

- 3 seeds × {A offline diff_only, B online all-changed, V verified control}
- **B − A = +0.415 ± 0.060（3/3 seeds 正）**，median cases 5/8，invalid 0，FR 0
  → **Gate 0 PASS**（§14）
- V − B = −0.405（1/3 正）→ carrier/geometry filtering 无贡献，仅记录（§14.1）
- query accounting 修正：每 arm 128 条 endpoint scorer query（旧口径把 pair record 当 query）

## Phase 1（branch-aware sigma restriction, frozen pair bank）

- σ_branch = 10.395；suffix mass 0.696 / prefix mass 0.304（§21 通过）
- 三臂共享 pair / schedule / ε / augmentation，仅 sigma 域不同（matched quantile）

| seed | T-full | T-suffix | T-prefix | suffix−full | suffix−prefix | prefix−full |
|---:|---:|---:|---:|---:|---:|---:|
| 20260915 | +3.632 | +2.954 | +3.103 | −0.678 | −0.149 | −0.528 |
| 43 | +4.360 | +3.902 | +3.974 | −0.458 | −0.072 | −0.386 |
| 44 | +2.529 | +2.764 | +3.635 | +0.235 | −0.871 | +1.105 |

- mean(suffix − full) = **−0.300**（要求 ≥ +0.25）；仅 1/3 seeds 正
- mean(suffix − prefix) = **−0.364**（要求 ≥ +0.50）
- mean(prefix − full) = +0.064（≤ +0.10 ✓，negative control 本身符合预期）
- invalid 0.2%，FR 0

→ **Gate 1 FAIL**（§26）：branch-aware noise-domain restriction 没有改善
preference learning。按 §47 停止 temporal 方向，不进入 Phase 2/3/4。
（按 §27：该结果只能表述为 branch-aware restriction of the diffusion training
noise domain，不构成 trajectory-level DPO 证据。）

## 结论与下一步

- 唯一跨 seed 稳定成立的组件的仍然是 **online sibling refresh + changed-position
  支持**（Phase 0 B，+0.415 ± 0.060）。
- §45 的最终 pilot 门槛为 mean(Final − Offline) ≥ +0.50；当前 +0.415 未达到，
  因此不进入 24-case / valid100 / structure / second reward（§46）。
- 建议：若继续这条线，应先解释“为什么 full-sigma 的 online refresh 优于
  suffix/prefix 限制”（例如 preference 信号在中间噪声域更可学），
  而不是继续加 temporal 结构；任何新假设需新的 protocol version。

## 归档（代码内可追溯）

- 汇总表与 artifact 索引：`results/online_pref/README.md`
- gate / audit / eval / round 日志：`results/online_pref/artifacts/`
- 每 arm 的完整训练日志仍在 `runs/online_pref/**`（含 pair banks，未入库）

## 贡献排序（截止本轮，全部有 3-seed 证据）

1. 空间支持定位（只训练好/差样本真正不同的 CDR 位点）——最稳定的核心；
2. 在线 sibling 数据刷新——小而稳定（+0.415 ± 0.060），未达 +0.50 放量门槛；
3. signed local DPO / region / adaptive credit / sibling distillation /
   geometry-carrier filtering / branch-aware sigma restriction——均无跨 seed
   稳定增益。

## 回答“新方向是否又失败”

- 新假设（时间域限制）**失败**：suffix < full，2/3 seeds 为负。
- 但其前置步骤（在线数据刷新 + all-changed 支持）是**目前最干净的正面结果**
  （首次与 geometry 过滤解耦，3/3 seeds 稳定），只是幅度未达放量门槛。
- 因此本轮结论：保留 `online refresh + changed-position support` 作为候选
  增益组件，停止 temporal 结构；不进入 24-case/valid100。
