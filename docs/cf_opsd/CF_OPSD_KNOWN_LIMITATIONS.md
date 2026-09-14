# CF-OPSD Known Limitations（供审计）

## 1. 结果状态

- Phase A（audit + rollout parity）**已实际运行**：parity 序列 100% 一致、
  endpoint 坐标差 0.0。
- Query probe **已实际运行**：60/70/80% anchor 无法 decode，q* = 90%
  （valid 0.906，status PASS）。
- Target probe **已实际运行**：三个半径 Gate B 全部 FAIL
  （0.25 Å: invalid 25%、credit match 0.0、positive 2/4；
  0.5/1.0 Å: invalid 100%）。
- 按任务书 §32/§95：**Phase C（realization）与 Phase D/E 未执行**；
  `CF_OPSD_REALIZATION_PROBE.md`、`CF_OPSD_PILOT_RESULTS.md` 未生成
  （Gate B 失败即停止，这是任务书要求的行为）。

## 2. on-policy 代码状态（两轮审计修复后）

Phase E 闭环已按审计意见修复：

| 审计问题 | 修复 |
|---|---|
| `cfg["outer"/"fit"/"optimizer"]` 缺失导致 KeyError | 新增 `configs/cf_opsd/onpolicy_v1.yaml`；`run_onpolicy` 合并两份配置并显式 `_require` 校验 |
| behavior 未用于下一轮 rollout | 每轮 EMA 后保存 `behavior_r{r}.pt` 并调用 `set_adapter_checkpoint()`，下一轮 rollout 使用该 checkpoint；由 `test_onpolicy_wiring` 验证（round 1 用 base，round 2/3 用 behavior_r1/r2） |
| held-out 评的是 base（ckpt=None） | 每轮保存 `student_r{r}.pt` 并把该 checkpoint 传给 `evaluate()`；由同一测试验证 |
| `cf_opsd_static_eval.py` 的 `r["step"]` KeyError | `build_rows()` 从 arm 名解析 `(method, step)`；新增回归测试 |
| behavior 与 student 是同一对象（额外发现） | `copy.deepcopy(base)` 作为独立 behavior |
| **`pairing.pairs_per_case` 未被读取**（第二轮审计 A） | 训练器按 `pairs_per_case` 构造 (rank1,rankN)、(rank2,rankN-1)…；wiring test 断言 4 endpoints → 2 pairs/case |
| **`fit.loss` 读了但不生效**（第二轮审计 B） | `target_mask_only` / `target_mask_weak_hold`（hold 项 + `hold_lambda`）真正接入训练循环 |
| **Phase E 不校验 Gate B/C/D**（第二轮审计 C） | `check_gates()` 硬校验三个 gate 文件，任一不过即 `RuntimeError` refuse；新增 `test_gate_checks`（含"失败时不得触碰任何依赖"断言） |
| **Gate D 负数边界 bug**（第二轮审计 D） | `reward_superiority` 要求 `G_cfdpo > 0 ∧ G_opsd > 0 ∧ G_opsd/G_cfdpo ≥ 1.10`；新增负增益回归测试 |
| 无 CI | 新增 `.github/workflows/tests.yml`：CPU-only（torch-cpu + pytest）运行 15 个 CI-safe 测试文件；重依赖测试在 CI 中 skip |

**但 Phase E 仍未端到端执行过**：Gate B 已失败，按任务书不应运行。
上述 on-policy 代码路径目前只有集成级 wiring test 覆盖（CPU、fake deps），
没有真实 BoltzGen 端到端 run。审计时请按"已修复但未实测"对待。
CI 只证明这些 CPU 级测试通过，不等于真实 pipeline 已验证。

## 3. 复现性/工程限制

- 仓库仍包含大量集群绝对路径（`/share/home/rongdingyi/...`），
  不能开箱即用；未在本轮处理（属于整仓可移植性工作）。
- CF-OPSD 依赖外部 base checkpoint / 打分 worker / 官方 BoltzGen 仓库，
  均未 vendored。
- `runs/` 已被 `.gitignore` 排除（386G 实验数据），仓库只含代码与文档。
