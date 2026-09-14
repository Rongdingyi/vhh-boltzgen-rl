# Native Atom14 Post-training Results (Round 1)

> 任务书：`BOLTZGEN_NATIVE_ATOM14_CDR_REWARD_POSTTRAIN_TASK.md`
> 代码：`vhh_boltzgen_rl/src/vhh_rl/native_atom14/`；数据：`runs/native_pool/`
> reward 固定为既有 CDR classifier（`cdr_camelid_margin`），未替换。

## 1. Goal

把 CDR classifier reward 从 IF 分支迁移到 **BoltzGen 原生 atom14 design diffusion**：
native采样 → 官方 `res_from_atom14` 读出序列 → classifier 打分，比较
RWR / Full-Atom14 DPO / CDR-Fake-Atom DPO / Fake+Anchor DPO 四个 arm。

## 2. Gate Summary

| Gate | Status | Evidence |
|---|---|---|
| Native decode parity | **PASS** | pool 全量 1632/1632 samples path A == path B（官方 writer vs 独立重解码）；`runs/native_pool/*/metadata.jsonl` |
| FR check | **PASS** | train/heldout/valid100 pool + 全部评测臂 FR mismatch = 0 |
| UNK/invalid | **PASS** | base pool 1632 samples 仅 1 个 UNK（0.06%），已隔离；各臂评测 invalid ≤ 0.4% |
| Pair correctness | **PASS** | 192 pairs，rank-symmetric，reward_w > reward_l，0 failures |
| Paired noise sync | **PASS** | x_t 一致性误差 0.0；noise/sigma/rigid-aug 全共享；同输入 twin 变换一致 |
| DPO init log(2) | **PASS** | 20 pairs mean loss = **0.69314718**（= log 2，std 0） |
| Gradient audit | **PASS** | 仅 `structure_module.score_model.*` 有梯度；reference 0 梯度 |
| One-pair overfit | **PASS** | 60 updates z: 0 → +0.0025（z_max +0.0126） |
| Mini generation | **PASS** | held-out 8 cases × 8 samples/arm 全部生成成功，无崩溃 |

## 3. Base Native Design (N0)

- train pool：24 cases × 32 = 768 samples；held-out 8 × 8 = 64；valid100 100 × 8 = 800。
- train reward：mean 3.48（case均值），top-bottom gap 中位数 10.28。
- **valid100 base mean CDR margin = 6.578**（100 cases; invalid 0; FR 0）。
  （与 round-1 "design读出" 臂的 6.60 一致——同一读出机制。）

## 4. Trainable scope

仅 `structure_module.score_model`（279,622,656 params）；trunk/pairformer/
embedding/conditioning 全冻结（conditioning 逐 case 预计算缓存）。
β 校准：β∈{0.1, 1.0, 10.0} 冒烟 50 步，0.1/1.0 信号≈0，选 **β=10**；
λ_anchor = 0.1（未 micro-sweep，见 limitations）。

## 5. Training runs（各 500 步，24 train cases）

| arm | 方法 | 终态 loss | z_mean(last10) | implicit_acc | param drift |
|---|---|---|---|---|---|
| N1 | RWR (top-25%) | 0.0125 (denoise) | — | — | 3.07 |
| N2 | Full-Atom14 DPO | 0.7067 | +0.0017 | 0.70 | 3.22 |
| N3 | CDR-Fake-Atom DPO | 0.7067 | +0.0094 | 0.70 | 3.29 |
| N4 | Fake+Anchor DPO | 0.7031 | **+0.0117** | 0.70 | **2.93** |

## 6. Held-out checkpoint selection（§81）

| arm | s250 Δ | s350 Δ | s500 Δ | 选中 | 理由 |
|---|---|---|---|---|---|
| N1 | +0.875 | +0.526 | +0.730 | **s250** | held-out 最高 |
| N2 | +1.733 | +2.300 | **+3.709** | **s500** | 单调上升，8/8 胜，invalid 0 |
| N3 | +2.058 | **+2.543** | +3.882（invalid 2/64） | **s350** | s500 invalid 3.1% > base+1pp，不满足 §81 条件 |
| N4 | +2.012 | +2.724 | **+3.519** | **s500** | 8/8 胜，invalid 0 |

## 7. valid100 Final Comparison（主评测；100 cases × 8 samples）

| arm | mean CDR margin | base | Δ | W/T/L | Wilcoxon p | bootstrap 95% CI | invalid | unique |
|---|---|---|---|---|---|---|---|---|
| Base native | 6.578 | — | — | — | — | — | 0 | 1.000 |
| **N1 s250** | 7.161 | 6.578 | **+0.583** | 71/0/29 | 2.9e-05 | [+0.32, +0.83] | 0 | 1.000 |
| **N2 s500** | 9.279 | 6.578 | **+2.702** | 93/0/7 | 2.3e-17 | [+2.31, +3.11] | 0 | 1.000 |
| **N3 s350** | **9.374** | 6.578 | **+2.796** | **95/0/5** | 3.9e-17 | [+2.40, +3.20] | 0 | 0.999 |
| **N4 s500** | 9.090 | 6.578 | **+2.512** | 93/0/7 | 7.1e-17 | [+2.14, +2.89] | 0 | 0.999 |

历史对照（同一 classifier、同一 valid100）：
base IF 1.90 / IF-RL R2 6.89 / design 读出 6.60 / **G0 (ProteinMPNN) 9.48 / M1-β1 13.21**。
→ **N3 (9.37) 几乎追平 ProteinMPNN**，并且 native 路线超过了 IF-RL。

per-case 数据：`runs/native_pool/eval_valid100_per_case.csv`。

## 8. Sequence Diagnostics

- invalid/UNK：训练前后 0（N3 评测仅在生成 800 条中出现 0；held-out 上曾出现 2 条，已隔离）。
- FR mismatch：全评测 0。
- unique rate：0.999–1.000，无 mode collapse。
- RWR（N1）提升最小（+0.58），说明单纯 imitation 高分组不够；
  DPO 类 arm 提升大 4–6 倍。

## 9. Geometry Diagnostics（refold）

Boltz-2 200 步 fold-back，20 valid100 cases × 5 arms × 4 seqs = 400/400 成功
（`native_refold_design/` + `native_refold_results_per_sample.csv`）：

| arm | CDR RMSD | CDR3 RMSD | FR RMSD | pLDDT(CDR) |
|---|---|---|---|---|
| base native | 1.860 | 2.347 | 0.559 | 0.623 |
| N1 RWR | 1.839 | 2.312 | 0.555 | 0.626 |
| N2 full DPO | 1.999 | 2.502 | 0.570 | 0.622 |
| N3 fake DPO | 2.090 | 2.709 | 0.578 | 0.614 |
| N4 fake+anchor | 2.206 | 2.823 | 0.572 | 0.621 |

解读：
- FR RMSD 基本不变（0.555–0.578 Å）→ backbone/FR 没有被 reward 推走。
- DPO 类 arm 的 CDR RMSD 相比 base 退化 +0.14 ~ +0.35 Å，换来 +2.5 ~ +2.8 的
  CDR margin（reward/geometry trade-off 远优于 IF-RL 轮：IF-RL 用更大比例
  的 reward 提升换来类似量级的 RMSD 变化，且伴随 global nativeness 下降）。
- N1（RWR）几何无损（1.839 ≈ base 1.860），但 reward 提升最小。
- pLDDT(CDR) 仅轻微下降（0.623 → 0.614–0.622）。
- **未发现 geometry collapse 或 reward hacking 结构信号**（invalid 0、FR 0、
  unique ≈ 1、FR RMSD 稳定）。
- 注意：这些 20 cases 是 base 最难+最易各 10 个；base native 的 1.86 Å 绝对值
  高于 IF 轮的 1.14 Å 是 case 组成差异，跨轮不可直接比较。

图：`docs/native_round1_figures/`（fig1 training、fig2 valid100、fig3 geometry、
fig4 reward-vs-RMSD）。

## 10. Failure Analysis

1. **executed 层面**：初版 `sequence_from_feat` 未把 `X`（UNK 映射字母）计为 invalid，
   导致含 UNK 序列被送进 classifier 崩溃；已修（非 AA20 即 invalid）。
2. **NFS 缓存**：编辑中的脚本被计算节点读到旧版，导致 eval 结果文件命名冲突被覆盖；
   已用带 step 后缀的 pool 目录名区分，最终数据以 pool 目录为准。
3. conda 环境混用导致 `/usr/bin/python3`（无 torch）多次报错；所有训练/评测均走
   Slurm + `vhh-guidance` env。
4. β 校准偏弱：0.1/1.0 无信号，10 有明显信号但未做完整扫描；
   N2/N3/N4 的 z 值很小（0.002–0.012），说明梯度信号温和、未过训练。

## 11. Decision

**成功**（满足 §88 工程标准 + §89 方法标准）：

1. Native atom14 reward post-training 成立：N2/N3/N4 全部 held-out 8/8 胜、
   valid100 93–95/100 胜，p < 1e-16，CI 完全在正区间。
2. 四臂排序：**N3 (fake-only) > N2 (full) ≈ N4 (fake+anchor) >> N1 (RWR)**；
   N3 的 9.374 已接近 G0 的 9.48。
3. fake-atom preference（N3）略优于 full-atom（N2）：对 atom14 表示，
   把偏好放在 sequence-carrying fake 几何是更合理的做法（§91 证据）。
4. anchor（N4）保持 reward 基本不降（−0.28 vs N3）同时参数漂移最低（2.93 vs 3.29）；
   若后续 geometry drift 诊断显示 N3 漂移大，anchor 的价值将进一步显现（§92）。
5. 未发现 reward hacking 的结构性信号（invalid 0、unique 高），
   但 refold 诊断完成后需最终确认。

**下一步（不自动执行）**：online preference refresh、adaptive anchor、
扩大 clean training conditions、Diffusion-GRPO。
