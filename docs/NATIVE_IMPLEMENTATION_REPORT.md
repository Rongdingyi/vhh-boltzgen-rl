# Native Atom14 Implementation Report

> 任务书：`BOLTZGEN_NATIVE_ATOM14_CDR_REWARD_POSTTRAIN_TASK.md`
> 代码：`vhh_boltzgen_rl/src/vhh_rl/native_atom14/`；脚本：`vhh_boltzgen_rl/scripts/native_*.py`
> IF-GRPO 分支（`src/vhh_rl/rl/`、`adapters/boltzgen_if.py`）保持原样未动。

## 1. Repository / checkpoint audit

见 `docs/NATIVE_AUDIT.md`。要点：BoltzGen `a3149cf`；design checkpoint
`boltzgen1_diverse.ckpt`（sha256 `360af8bd…f8e3c`）；`predict_res_type=false`，
序列完全由 atom14 几何经 `res_from_atom14` 读出；released checkpoint 只含
raw state_dict（无 EMA payload），官方推理即用 raw 权重——policy/reference 均
从同一 raw state_dict 初始化，并记录 checkpoint sha256。

## 2. Native design path

`NativeDesignAdapter`（`adapter.py`）在进程内调用官方 CLI：
`boltzgen run <spec> --steps design --skip_inverse_folding --no_subprocess …`，
种子沿用 m3 生产方法（`_seed_all` + `np.random.default_rng` 补丁）。
采样随机源与官方完全一致；不同 seed 得到不同轨迹。

## 3. atom14 → sequence decode

唯一解码器是官方 `boltzgen.data.feature.featurizer.res_from_atom14`（`decode.py`
仅做调用协议 + 序列提取 + UNK 统计 + FR 校验）。writer 内的官方调用点被 hook，
路径 A（官方 writer 内解码）与路径 B（对保存下来的张量独立再解码）逐样本比对，
pool 全量 1632 samples 100% 一致（见 §14 tests / Round1 Results）。

## 4. Reward adapter

复用 round-1 的 CDR classifier worker（`vhh_rl/rewards/scorer_worker.py`），
objective `cdr_margin_guarded`，优化字段 = **`cdr_camelid_margin`**（越大越好）。
`reward.py` 只做 spec 组装与调用。UNK/FR-mismatch 样本不送 classifier、
不给分、不进入 preference（§18）。

## 5. Preference pool generation

`native_generate_pool.py`：train 24 cases × 32 samples；held-out 8 × 8；
valid100 100 × 8（base 臂已缓存）。每个 sample 保存：coords `.pt`、case 级
`feats_common.pt`、`metadata.jsonl`（seed/checkpoint哈希/commit/解码序列/reward/
UNK/FR）、`structures/*.cif`、`sequences.fasta`（§15）。

## 6. Pair construction

`preference_dataset.py`：每 case 按 reward 排序，top8 × bottom8 秩对称配对
（rank1vs rank32 … rank8 vs rank25），仅保留 `reward_w > reward_l`。
train：192 pairs（gap 中位数 10.28，p10 4.80，max 22.65）。

## 7. Paired noising

`paired_noise.py`：官方 `center_random_augmentation(..., return_second_coords=True)` 
一次抽旋转+平移作用于双方；sigma 由官方 `noise_distribution` 同值；Gaussian
noise 同张量。单元检查：x_t = x0_aug + σ·noise 对双方零误差；同输入 twin
检查通过（同一变换）。

## 8. Per-sample denoising loss

`denoise_loss.py`：复刻官方 `compute_loss` 的加权坐标 MSE（fake-atom 权重、
resolved mask、nucleotide/ligand 权重、σ loss_weight），对齐始终用 full
resolved atoms，preference mask 只作用在误差上（§34）；不包含 lddt/bond/
distogram/bfactor 辅助项。

## 9. Full-atom DPO (N2)

preference mask = `atom_design_mask & atom_pad_mask`（全部 CDR 设计原子）。
train module = `structure_module.score_model`（279.6M 参数），trunk 冻结
（conditioning 逐 case 预计算缓存）。

## 10. Fake-atom DPO (N3)

preference mask = `atom_design_mask & fake_atom_mask & atom_pad_mask`
（CDR 设计位置的 fake side-chain 槽位）。fake_atom_mask 语义有单元测试
（GLY/ALA/TRP patterns），不靠变量名假设（§42）。

## 11. Reference/backbone anchor (N4)

`anchor.py`：`L = L_DPO^fake + λ·L_anchor`，anchor 对 non-preference 原子
（CDR backbone + FR + 其它）计算 policy/reference denoised 输出的 mask-normalized
MSE，reference detach。λ 默认 0.1。

## 12. Trainable parameters

仅 `structure_module.score_model.*`；gradient audit 实测有梯度的模块为
`single_conditioner / atom_attention_encoder / token_transformer /
atom_attention_decoder / s_to_a_linear / a_norm`，reference 0 梯度。

## 13. EMA policy

released checkpoint 无 EMA 状态，官方推理用 raw 权重；本报告按 raw 口径
比较 base 与 post-trained（§51），未另维护 EMA（记录为 limitation）。

## 14. Unit tests / gates

见 `NATIVE_ROUND1_RESULTS.md` 的 Gate 表。脚本：`scripts/native_smoke_gates.py`
（pair correctness / paired-noise sync / DPO init log2 / gradient audit /
one-pair overfit）。

## 15. Known limitations

1. β 校准（§40）只做了 4 case × 50 步的数值稳定性筛查（0.1/1.0 信号过弱，
   选 β=10 但有偏大风险），未做完整 held-out 扫描。
2. λ_anchor 未做 micro-sweep（直接用 0.1）。
3. EMA 未维护（官方 checkpoint 本身无 EMA payload）。
4. 采样步数全臂统一 50 步（与 m3 生产一致），未做步数敏感性分析。
5. valid100 评测中的 geometry/refold 诊断仅覆盖 refold 子集（后续补）。
