# AG-CF-DPO Decision

任务书：`AG_CF_DPO_ADAPTIVE_GRANULARITY_TASKBOOK.md`
报告：`docs/adaptive_granularity/AG_AUDIT.md`、`AG_PRE_PILOT.md`、`AG_PILOT.md`
冻结协议：`experiments/adaptive_granularity/configs/FROZEN_PROTOCOL.yaml`
（code commit `f6c843a`；workflow-only 后续提交不改变任何协议参数）

## 1. Motivation

CF-DPO v1 的收益来自 support/credit weighting；SL-CF-DPO 表明把 `sign_flip`
强转成 signed residue label 会伤害 global CF。本轮假设：sign disagreement 应该
被当作 **attribution reliability signal**，据此决定 preference 应该分配到
residue 粒度、region 粒度，还是 abstain——而不是继续学习更多 local objective。
所有新方法仍只是 `pair_id -> {position -> weight}`，训练器完全复用
`run_weighted_dpo()`（same sigma/noise/rigid-aug/fake-atom loss/beta/lr/scope）。

## 2. Frozen Protocol

- TOL=0.05，RHO=0.70（预注册，未调参），eta_current=0.75，beta=10，lr=1e-5
- pilot：4 train cases ×100 updates，seed 20260913，checkpoint@50/100
- eval：all-heldout-8 ×8 samples，seed = case.seed_base + 800000（historical-4 为同一批样本子集）
- Current CF 权重与 `controls.py`、`weighted_dpo.py` 未改动

## 3. Phase A Audit（Gate A）

- 192 pairs / 548 changed regions / 24 cases
- residue 类别：stable+ 43.96%，stable− 16.15%，sign-flip 33.24%，weak 6.64%
- Current-CF conflict weight mass：median **0.125**，68.8% 的 pair > 0.10
- coarse-rescue regions：**299/548 = 54.6%**；含 coarse-rescue 的 pair：**94.8%**
- **Gate A 三项全部 PASS** → 允许 GPU pilot

## 4. Weights（A2）

| variant | eligible pairs | active frac | 说明 |
|---|---:|---:|---|
| no_floor | 192/192 | 0.471 | 无 floor，无 fallback |
| strict_consensus | 192/192 | 0.444 | 仅双向 stable+ |
| strict_region | 191/192 | 0.944 | region-only uniform |
| adaptive | 192/192 | 0.923 | residue 137 / region 299 / abstain 112 |
| adaptive_shuffle | 192/192 | 0.923 | 与 adaptive 同 histogram |

`validate_weights`：全部通过（weights 仅在 differing positions、非负、sum=1；
ineligible 一律空 dict，无 uniform fallback）。

## 5. Phase B（held-out 8 ×8）

| arm | reward8 | ΔCF | wins vs CF |
|---|---:|---:|---:|
| Base | -1.110 | — | — |
| Current CF | +1.883 | 0 | — |
| No-floor | +1.651 | -0.232 | 3/8 |
| Strict consensus | +1.740 | -0.143 | 4/8 |
| Strict region | +1.262 | -0.621 | 3/8 |

Gate B：B1 fail，B2 fail（region gain = 79.2% < 80%，差 0.024 reward），
**B3 pass**（rescue 54.6% ≥ 25% 且 region-only −0.621 ≥ −0.75）
→ Gate B 通过（B3 路径），按 §46 best simple 固定为 **Strict Region-only**。

## 6. Phase C（held-out 8 ×8）

| arm | reward8 | ΔCF | vs shuffle | vs eligible-CF | invalid |
|---|---:|---:|---:|---:|---:|
| Current CF | +1.883 | 0 | — | — | 0 |
| Adaptive | +1.614 | **-0.270** | +1.479 | **-0.332** | 0 |
| Adaptive shuffle | +0.135 | -1.748 | 0 | -1.811 | 0 |
| Eligible-CF | **+1.946** | **+0.062** | +1.811 | 0 | 0 |

- 全部 arm：invalid 0%，FR mismatch 0
- 3-seed micro-confirmation（§49/§50）未触发：Adaptive−CF = −0.270 < −0.2，
  不满足 borderline 区间；`gate_c_confirm.json` 未运行（§50 rule 已实现并测试）

## 7. Mechanism：gradient conflict（§52-§55）

64 个 sign-flip sites（SL edge dataset），global CF 与 local signed DPO 在**同一**
sigma/noise/rigid-aug realization 下（`sigma_mismatch_sites = 0`）：

- median cosine **−0.026**，P25 −0.142，P75 +0.150
- negative fraction **53.1%**，strong negative (cos < −0.1) **31.3%**

结论：signed local supervision 与 global CF objective 存在真实但中等强度的
梯度冲突——与上一轮 SL-CF-DPO 的伤害方向一致，但不足以解释全部损失。

## 8. Failure Analysis（§89-§91/§96）

1. **收益来自 pair filtering，而非 granularity**：Eligible-CF（同一 eligibility、
   原 CF 权重）达到 +1.946，比 Current CF 高 +0.062（噪声级，3/3/2），
   同时比 Adaptive 高 +0.332。按 §51 第一条即 Adaptive ≤ Eligible-CF → NO-GO。
   §96 要求的对照正好证明：若把收益归因给 adaptive granularity 是过度声明。
2. **granularity 分配本身无正收益**：No-floor（−0.232）、Strict-consensus
   （−0.143）、Region-only（−0.621）、Adaptive（−0.270）全部低于 Current CF；
   唯一显著正向的是"跳过不可信 pair 后继续用 CF 权重"。
3. **floor 没有明显泄漏危害**：尽管 68.8% pair 的 conflict mass > 0.10，
   Current CF 仍是最稳的加权方案 → 与 §91 一致：eta=0.75 uniform floor
   很可能起正则化作用，sign disagreement 不能直接解释为"不可信 credit"。
4. **mapping 有信号但不足以取胜**：Adaptive 大幅优于 shuffle（+1.479），
   说明 weight↔position 对应关系不是噪声；但相对 CF 仍为负，说明
   "正确但更细/更粗的 attribution"不能带来 net reward。
5. **Region-only 达到 79.2% of CF gain**：接近但未过 B2 的 80% 门槛，
   说明 region 层信息不是完全无效——但单靠它无法超过 CF。

## 9. Decision

### NO-GO

依据 §51：`Adaptive <= EligibleCF`（−0.332），且 `Adaptive < CF`（−0.270）。
不进入 24-case full、不进入 3-seed full、不进入 valid100。
本轮结论：在现有 counterfactual credit 数据上，**提升 attribution granularity
或 abstention 都不能超过 CF-DPO 的 credit weighting**；唯一有效成分是
pair filtering，但其增益 +0.062 在 8-case pilot 分辨率下不具可判定性。

## 10. Next Step（§91/§105）

1. STOP；不调 rho/TOL，不再堆 granularity 复杂度。
2. 保留 CF-DPO（eta=0.75 + c_cons 加权）为当前算法主线与论文基线。
3. 若要继续此方向，必须先证明 pair-filtering 增益（+0.06）在 3-seed /
   valid100 上可复现且大于噪声，再讨论 attribution 结构；
   当前证据不支持对 adaptive granularity 的算法声明。
