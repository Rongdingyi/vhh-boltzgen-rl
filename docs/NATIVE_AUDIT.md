# Native Atom14 Audit (Phase A)

> 任务书：`BOLTZGEN_NATIVE_ATOM14_CDR_REWARD_POSTTRAIN_TASK.md` §13
> 审计日期：2026-09-12；审计人：opencode
> 结论：native atom14 design path 可用；reward 固定为现有 CDR classifier（`cdr_camelid_margin`）。
> **本审计不覆盖 IF-GRPO 分支**（`src/vhh_rl/rl/`、`adapters/boltzgen_if.py` 原样保留）。

---

## 1. Repository / checkpoint

| 项 | 值 |
|---|---|
| repo | `/share/home/rongdingyi/programs/proteingen/boltzgen` |
| commit | `a3149cf18eeb58648d1abbb27539bd73f746cdda`（与任务书参考一致，`git status` 干净） |
| design checkpoint | `ckpts/boltzgen1_diverse.ckpt`，sha256 `360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c` |
| IF checkpoint（仅历史对照） | `ckpts/boltzgen1_ifold.ckpt` sha256 `dd4cf108…d56578` |
| folding checkpoint | `ckpts/boltz2_conf_final.ckpt` |
| moldir | `ckpts/moldir`（45229 entries，已校验存在） |
| m3 lineage | 现有 valid100 backbone 由 `vhh_esmc_guidance/scripts/target_free_strict8/generate_backbones.py` 用同一 design path 生成 |

## 2. Native design config（`src/boltzgen/resources/config/design.yaml`）

```yaml
data.cfg:
  backbone_only: false      # ← 原生全原子
  atom14: true
  atom37: false
writer: DesignWriter(atom14=true, inverse_fold=false, write_native=false)
trainer.precision: bf16-mixed
sampling_steps: 500         # default；线下 m3 生产用 --config design sampling_steps=50
diffusion_samples: 1
override.masker_args: {mask: true, mask_backbone: false}
override.step_scale_schedule:  [1.8/0.25, 2.0/0.25, 1.8/0.25, 2.0/0.25]
override.noise_scale_schedule: [0.95/0.25, 0.88/0.25, 0.95/0.25, 0.88/0.25]
override.diffusion_process_args:
  sigma_min: 0.0004, sigma_max: 160.0, sigma_data: 16.0, rho: 7
  P_mean: -1.2, P_std: 1.5
  gamma_0: 0.8, gamma_min: 1.0
  mse_rotational_alignment: true
  coordinate_augmentation: true
  alignment_reverse_diff: true
  sampling_schedule: dilated, time_dilation: 2.667 (start 0.6 / end 0.8)
```

- `predict_res_type: false`（`Boltz.__init__` 默认；checkpoint hyper_parameters 亦为 false）——**没有独立 20-AA 分类头**，序列完全由 atom14 几何读出。
- design mask 来源：spec yaml 序列中的数字 token（如 `CTAS8MGW…` 的 `8/7/19`），`schema.parse_polymer` 展开成 N 个 design 位置（GLY 占位 + `res_design_mask=True`）。
- CLI 生产调用（与 m3 一致，属官方 path）：
  ```
  boltzgen run <spec> --output <dir> --protocol nanobody-anything --num_designs K
    --steps design --skip_inverse_folding --no_subprocess --devices 1 --num_workers 1
    --use_kernels false --design_checkpoints ckpts/boltzgen1_diverse.ckpt
    --inverse_fold_checkpoint ckpts/boltzgen1_ifold.ckpt
    --folding_checkpoint ckpts/boltz2_conf_final.ckpt --moldir ckpts/moldir
    --config design sampling_steps=50
  ```
- 随机性：CLI 无 `--seed`；官方无种子接口。m3 生产用 `_seed_all(seed)`（python/numpy/torch/cuda）+ `np.random.default_rng` 补丁保证可复现（`generate_backbones.py:49-79`），本模块沿用同一方法并在每个 case 生成前重新 seed。

## 3. Sequence readout：`res_from_atom14`

- 实现：`src/boltzgen/data/feature/featurizer.py:1705`（**直接 import 复用，不重写**）。
- 写入调用：`DesignWriter.write_on_batch_end` → `src/boltzgen/task/predict/writer.py:266` `sample = res_from_atom14(sample)`（atom14 分支）。
- 输入：单个 sample 的 feat dict（batch 维已 squeeze），关键字段：
  - `coords` `[N_atom, 3]`（设计位置的 fake atom 与真实侧链槽位按 atom14 顺序，每残基 14 槽）
  - `atom_to_token` `[N_atom, N_tok]` one-hot、`token_index`、`design_mask` `[N_tok]`、`atom_pad_mask` `[N_atom]`
- 算法：把 design 位置的原子按 `N//14 × 14 × 3` 重排 → 侧链 10 个槽位逐个找最近的骨架原子（N/CA/C/O，阈值 0.5Å，`torch.cdist`）→ 统计四个骨架原子各自的最近侧链槽位数 → `const.placement_count_to_token[(n_N,n_CA,n_C,n_O)]` 查表 → residue type；无匹配 → `UNK`。
- `UNK` 处理：任务书 §18 —— UNK sample 不进 classifier、不进 preference、eval 计入 `native_invalid_rate`；base invalid rate >5% 则停止并检查 pipeline。
- 本模块 decode 路径：
  1. 直接在官方 writer 内 hook（保证与官方输出逐字节同源）→ 保存 decoded sequence；
  2. 对保存的坐标独立再跑一次 `res_from_atom14`（decode parity gate 的路径 B）。

## 4. Training / score model

| 项 | 值 |
|---|---|
| `structure_module` | `AtomDiffusion`（`src/boltzgen/model/models/boltz.py:310`） |
| `structure_module.score_model` | `DiffusionModule`（`src/boltzgen/model/modules/diffusion.py:68`，即 Algorithm 20 atom attention encoder/decoder + token transformer） |
| 本轮可训练参数（N1–N4） | 仅 `structure_module.score_model`；trunk/pairformer/embedder/confidence/affinity 全部冻结 |
| forward-noising | `AtomDiffusion.noise_distribution()`（diffusion.py:636）：σ = σ_data·exp(P_mean + P_std·ε)，X_t = X_0 + σ·ε，前置 `center_random_augmentation`（diffusion.py:648-660） |
| 官方 denoising loss | `AtomDiffusion.compute_loss`（diffusion.py:700-856）：per-atom MSE → `align_weights × fake_atom_weight × res_type_weight × resolved_mask` 归一化 → `loss_weight(σ)` → `.mean()`；rigid alignment 用 full resolved atoms（`mse_rotational_alignment=true` → `weighted_rigid_align`） |
| DPO per-sample loss | 复刻上述加权 MSE，但在 `.mean()` 之前按 sample 返回 `[B]`（`native_per_sample_denoising_loss`） |
| EMA | 训练配置 `train/boltzgen_small.yaml: ema: true, decay 0.999`；但 **released checkpoint `boltzgen1_diverse.ckpt` 顶层无 `ema` 键**（`state_dict` 即推理权重），CLI 也未传 `use_ema`（`Predict` 默认 false）→ 官方推理用 raw state_dict。本轮：`policy` 与 `reference` 都从该 state_dict 初始化，训练保存 raw；同时按任务书 §80 维护新 EMA 并在 eval 报告 raw/EMA parity（若实现成本高，第一版以 raw 为准并记录）。 |
| 训练器 | 官方的 diffusion training task（`resources/config/train/boltzgen_small.yaml`）不直接复用（它跑多卡/完整数据），本轮用自写轻量 trainer，但 loss/noising 全部复用官方函数 |

## 5. Reward（固定，禁止修改）

- entrypoint：现有 `vhh_rl` scorer wrapper（`src/vhh_rl/rewards/scorer_worker.py`，经 `ScorerAdapter`）
- objective：`cdr_margin_guarded` 的 `scores` 字段 = **`cdr_camelid_margin`**（CDR classifier logit margin，`vhh_guidance` VHH-source ensemble 输出）
- 方向：higher is better（已审计，无符号翻转；`scorer_worker.py` 内 `scores.append(float(result.cdr_camelid_margin))`）
- 输入：完整 VHH 序列（FR+CDR），classifier 内部按 CDR 区段打分
- 本轮所有 reward 报告字段 `reward_raw = cdr_camelid_margin`；诊断指标（nativeness/discordance/JS/energy…）只做离线评测，不进训练。

## 6. Data split（沿用第一轮，零重叠）

| split | 数量 | case IDs / manifest | sha256 |
|---|---|---|---|
| train | 24 | `sab2_7qxd_kkk … sab2_7nqk_b`（见 `runs/round1_rl_split/rl_manifest_split.jsonl`） | manifest `7d143bed…3fa3e20` |
| held-out (dev) | 8 | `sab2_6e62_b, sab2_8dpm_d, sab2_5mp6_h, sab2_6oq8_d, sab2_4mwf_h, sab2_6cvk_b2, sab2_6s2i_a, sab2_4hf5_h` | 同上 |
| valid100 | 100 | `runs/round1_rl_split/rl_manifest_valid100test.jsonl` | `d5c68917…4860d9` |

- 划分方式：SAbDab2 单链 VHH 晶体结构 → MMseqs2 70% cluster → cluster 级 split（seed 20260911），与 valid100 无 overlap（构建时已做 ≥85% id 泄漏检查并排除）。
- valid100 **只用于最终评测**，禁止参与任何超参/checkpoint 选择。

## 7. 已知约束与风险

1. **采样慢**：官方 config `sampling_steps=500`；m3 生产用 50 步。本轮统一用 50 步（所有 arm 一致），需在报告中记录。
2. **无 seed CLI**：种子必须由外层进程设置（同 m3 `_seed_all` + `default_rng` 补丁）。
3. **UNK 风险**：几何读出的固有失败模式，必须在每个 gate 统计。
4. **FR 不变性**：设计 spec 中 FR 位置是显式序列 token（非数字），理论不可变；仍需对 decoded 序列做 FR hard check。
5. **dtype**：官方 design 推理 bf16-mixed；reward/DPO 训练遵循任务书 §52 `bf16-mixed`，但 denoising loss 复刻时用 fp32 计算（官方 `compute_loss` 内 `torch.autocast("cuda", enabled=False)`）。
