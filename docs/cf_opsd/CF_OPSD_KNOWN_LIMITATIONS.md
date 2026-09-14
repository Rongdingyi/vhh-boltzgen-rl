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

## 2. on-policy 代码状态（本轮修复后）

Phase E 闭环已按审计意见修复：

| 审计问题 | 修复 |
|---|---|
| `cfg["outer"/"fit"/"optimizer"]` 缺失导致 KeyError | 新增 `configs/cf_opsd/onpolicy_v1.yaml`；`run_onpolicy` 合并两份配置并显式 `_require` 校验 |
| behavior 未用于下一轮 rollout | 每轮 EMA 后保存 `behavior_r{r}.pt` 并调用 `set_adapter_checkpoint()`，下一轮 rollout 使用该 checkpoint；由 `test_onpolicy_wiring` 验证（round 1 用 base，round 2/3 用 behavior_r1/r2） |
| held-out 评的是 base（ckpt=None） | 每轮保存 `student_r{r}.pt` 并把该 checkpoint 传给 `evaluate()`；由同一测试验证 |
| `cf_opsd_static_eval.py` 的 `r["step"]` KeyError | `build_rows()` 从 arm 名解析 `(method, step)`；新增回归测试 `test_static_eval_rows` |
| behavior 与 student 是同一对象（额外发现） | `run_onpolicy` 显式 `copy.deepcopy(base)` 作为 behavior，独立于 student |
| 测试未覆盖 on-policy wiring | 新增 `test_onpolicy_wiring`（注入 fake deps，断言 checkpoint 切换/评测对象/计数器）与 `test_onpolicy_config` |

**但 Phase E 仍未端到端执行过**：Gate B 已失败，按任务书不应运行。
上述 on-policy 代码路径目前只有集成级 wiring test 覆盖（CPU、fake deps），
没有真实 BoltzGen 端到端 run。审计时请按"已修复但未实测"对待。

## 3. 复现性/工程限制

- 仓库仍包含大量集群绝对路径（`/share/home/rongdingyi/...`），
  不能开箱即用；未在本轮处理（属于整仓可移植性工作）。
- CF-OPSD 依赖外部 base checkpoint / 打分 worker / 官方 BoltzGen 仓库，
  均未 vendored。
- `runs/` 已被 `.gitignore` 排除（386G 实验数据），仓库只含代码与文档。
