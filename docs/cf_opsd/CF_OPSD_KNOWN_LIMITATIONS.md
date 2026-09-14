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

### 第三轮（GPU 实跑发现）

| 问题 | 修复 |
|---|---|
| **`test_query_replay` 不通过（diff 0.052）**：design 管线以 `multiplicity=diffusion_batch_size`（=8）调用 sampler，而 replay 用 `multiplicity=1` + 单样本 context，anchor 不可复现 | `rollout.py` 记录每次 network call 的 `multiplicity` 到 trajectory meta；`test_query_replay` 改为：把该 case 的 K 条 trajectory 在 query step 上 stack 成完整 batch，按记录的 multiplicity forward，逐元素与 anchor 比对（FP32 < 1e-5） |
| **同-query 拟合的 batch 语义错误**：静态/on-policy fitting 用单样本 + multiplicity=1，与 anchor 的采样上下文不一致 | 目标记录新增 `full_query_coords` / `full_anchor_coords` / `design_index` / `multiplicity`；`static_trainer` 与 `onpolicy_trainer` 改为在完整 batch 上下文 forward，只在目标 design 的 slice 上计算 loss/hold |
| **`test_student_grad` 断言错误**（测试 bug）：`load_base_model` 默认全参数 requires_grad | 测试改为先 `make_policy_reference`（冻结），断言仅 `structure_module.score_model.*` 可训练、reference 全冻结 |
| **`test_target_decode` IndexError**：conditioning feats 带 batch=1 维度，unbatched 官方 decoder 不兼容 | `target_metrics._squeeze_feats()` / `query_selector.decode_anchor()` 在 decode 前去除 singleton batch 维；测试改用 squeeze 后的 feats |
| **target probe 报告段在 Gate B 失败时崩溃**（None 格式化、controls 为空） | None-safe 格式化 + 独立 `cf_opsd_target_report.py`（可从 CSV 重算 Gate B/文档） |
| **rollout 重跑时 parity 目录已存在 → adapter 拒绝清理** | 脚本在 parity 前清理自有目录 |

### 第四轮（GPU 端到端复现调试，全部定位并修复）

| 问题 | 根因与修复 |
|---|---|
| `test_query_replay` 差 0.213 / 0.062，anchor 无法复现 | 两个独立原因：**(1) TF32**——design CLI 用 `matmul_precision=high`，replay 默认是全精度 fp32（相对差 ~3e-3）；训练器/测试统一 `torch.set_float32_matmul_precision("high")`。**(2) sigma 形状**——sampler 把标量 sigma 扩成 `[B]`；replay 传 `[1]` 会走不同 TF32 kernel（差 0.062）；统一改为 `torch.full((B,), s)`。修复后 50 步 × 2 种 kwargs 全部 **bit-exact (maxdiff=0)** |
| conditioning 来源错误 | round-1 的 conditioning cache 与 rollout 实际 feats 不一致（`ref_pos` 等不同）；rollout 现在直接捕获**该次 run 的精确 conditioning kwargs**（step 0 + 候选 query steps {60/70/80/90%}）并存盘；replay 与训练器只用这份精确 kwargs |
| 采样 batch 上下文丢失 | `multiplicity=8` 时每次 forward 产生完整 batch；每步保存完整 `contexts`（query/anchor [B,N,3] + multiplicity），replay/拟合都用完整 batch，只在目标 design 的 slice 上取损失 |
| 训练脚本的 JSONL/None 格式化崩溃 | 张量过滤 + None-safe 格式化；新增独立 `cf_opsd_target_report.py` |

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
