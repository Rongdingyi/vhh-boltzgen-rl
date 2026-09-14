# Next-Stage Implementation（Phase A–D）

任务书：`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md`
（gate 顺序：A0 → A1/A2 → Gate A → B1 → Gate B → C → D → 停止）

## 1. 新增代码

### `src/vhh_rl/credit/`
| 模块 | 作用 |
|---|---|
| `pair_analysis.py` | pair Hamming / reward-gap 统计（A0） |
| `cdr_regions.py` | 复用项目 CDR 分段（`vhh_rl.data.cdr`，无 torch 依赖） |
| `counterfactual.py` | 单残基 / region CF 构建 + 硬检查（长度、FR、单点、canonical） |
| `credit_metrics.py` | `c_drop/c_gain/c_avg/c_cons`、top-k mass、`n_eff`、sparsity |
| `controls.py` | CF 权重 + B1 random-sparse / B2 shuffle / region 权重 |
| `trajectory_audit.py` | Kabsch RMSD、identity、stable fraction、sigma 阈值搜索 |
| `temporal_schedule.py` | `g_smooth`（τ=0.5）/ `g_hard` |

### `src/vhh_rl/native_atom14/`（扩展）
- `adapter.py`：新增 `generate_capture_trajectory`（包装官方
  `AtomDiffusion.sample`，保存 `x0_coords_traj` + `sigmas`/`t_hats` + 每
  sample 的 featurizer feats；不改采样方程）
- `denoise_loss.py`：新增 `per_residue_denoising_loss`（只用该 residue 的
  fake atoms，global rigid alignment 仍用 full resolved atoms）
- `masks.py`：`design_token_offset`（manifest 位置↔token 映射自检）、
  `residue_atom_masks`
- `weighted_dpo.py`：`run_weighted_dpo`（CF/B1/B2/region/uniform_all ×
  none/hard/smooth；DPO 数学形式不变）
- `data/cdr.py`：无 torch 的 CDR 分段（`cli.common` 重新导出）

### `scripts/next_*.py` / `scripts/sbatch_next_*.sh`
A0 复现、CF 构建/评分/分析、权重构建、B1 采集/分析、smoke、加权 DPO、D 臂、
checkpoint 选择、valid100 评测与汇总、refold 构建/对比、dashboard 图；
`run.sh` 新增 `next-*` 入口。

## 2. 任务书要求的 gate / 测试

| Gate | 内容 | 结果 |
|---|---|---|
| A0 | 统计复现 | 与任务书一致（见 `NEXT_STAGE_DIAGNOSTIC_REPRO.md`） |
| A | credit 稀疏性 | **STRONG** |
| B | temporal separation | **WEAK** |
| §56 | `test_weighted_loss_uniform_equivalence` | 通过（22/22 单测） |
| §45 | weighted DPO init：raw DPO loss≈log2 | 通过（gap 1.9e-9）；weighted loss = g(σ)·raw |
| §55 | CF-DPO smoke（5 步、2 cases） | 通过 |

单测：`test_credit_math`、`test_counterfactual_sequence`、
`test_credit_weights`、`test_weight_shuffle`、`test_temporal_schedule`、
`test_residue_atom_mapping`、`test_weighted_loss_uniform_equivalence`。

## 3. 训练设置

- base：`boltzgen1_diverse.ckpt`（frozen trunk conditioning 复用 round-1）
- 可训练：`structure_module.score_model`（2.8 亿参数）
- paired sigma / noise / rigid augmentation 与 N3 相同；β=10、lr=1e-5、
  max_grad_norm=1.0、500 steps、held-out checkpoint 选择（§40）
- 权重量：`runs/next_stage/weights/residue_weights.json`（η=0.75，
  fallback 率 0%，entropy CF 2.11 vs B1 1.95）

## 4. 复现入口

```bash
bash run.sh next-repro            # A0
bash run.sh next-build-cf         # A1/A2 构建
bash run.sh next-score-cf         # GPU 批量打分
bash run.sh next-analyze-cf       # Gate A
bash run.sh next-build-weights
bash run.sh next-audit            # B1 采集
bash run.sh next-analyze-audit    # Gate B
bash run.sh next-smoke            # 单测 + DPO init gate
bash run.sh next-b1-random-sparse # full control
bash run.sh next-b2-shuffle
bash run.sh next-b3-cf            # 主方法
bash run.sh next-select-ckpt
bash run.sh next-valid100
bash run.sh next-v100-agg
bash run.sh next-refold-build     # + 跨仓库 folding/analyze
bash run.sh next-refold-compare
bash run.sh next-d1-hard          # Phase D ablation
bash run.sh next-d2-smooth
```
