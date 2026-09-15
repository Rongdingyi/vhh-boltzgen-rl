# SL-CF-DPO Decision

Task book: `SL_CF_DPO_NEXT_ALGORITHM_TASKBOOK.md`（§48/§49 判定；§104 模板）。
Pilot report: `docs/signed_local/SL_CF_DPO_PILOT.md`。

## 1. Motivation

从 credit **weight** 转向 signed **local correction**：CF-DPO v1 的收益只来自
support/credit weighting，无法表达 counterfactual query 已经指出、但 global winner
仍然保留的 `both_negative` / `sign_flip` 局部偏好；CF-DPO v2 尝试把单残基偏好映射到
whole-structure energy proxy，已被判定无效（noise ≫ signal）。SL-CF-DPO 只做一件事：
把这些局部偏好变成 context-specific local pair，并在目标 residue 的 fake atoms 上执行
标准 Diffusion-DPO；reward 只决定 preferred/rejected orientation，不参与 magnitude。

## 2. Frozen Protocol

`experiments/signed_local/configs/FROZEN_PROTOCOL.yaml`（pilot 运行前冻结）：

- repo commit `a0484a7`，boltzgen commit `a3149cf`
- base ckpt sha256 `360af8bd…`，pairs sha256 `fd9776fd…`
- residue_credit sha256 `c8a1b01a…`，cf_scores sha256 `e805a59c…`，weights sha256 `b5d1cbe9…`
- train edges sha256 `0373b1b1…`（2405 边）
- heldout edges 在 pilot 之后按 §87 顺序构建，sha256 `b69e4e17…`（37 边）
- TOL=0.05，beta=10.0（global/local），eta=0.75，lr=1e-5，deterministic 3:1 G-G-G-L
- eval：4 held-out cases × 8 samples，seed = case.seed_base + 800000

实现全程复用已验证的 CF-DPO 机器（paired noise / rigid aug / diffusion_dpo_loss；
`native_atom14.global_cf_step`），未构造任何 global energy proxy，未使用 kappa 校准，
未把 dR 幅度写入 loss。

## 3. Local Edge Dataset

Train（192 pairs / 1480 sites）：

- both_negative 801 边，sign_flip 1604 边（2405 边，id 唯一，无重复）
- lift 接受率：winner_drop 80.5%，loser_gain 82.0%；按 site 89.7% (neg) / 89.5% (flip)
- preferred 分布 cf 49.9% / anchor 50.1%，median |dR| 0.82，median moved RMS 2.71 Å
- hard-decode OK 4810/4810，FR mismatch 0，invalid 0，dR = R(cf)-R(anchor) 不变式 2405/2405

Held-out（4 pairs / 30 sites → 37 边，10 neg / 20 flip）：接受率 70%/70%，preferred cf 56.8%。

Geometry lift 复用 `cf_dpo_v2.geometry_lift.build_local_lift`（exact decode + FR check）。

## 4. Correctness Gates

| gate | 结果 |
|---|---|
| dR 不变式 / 唯一 id | 2405/2405，0 duplicates |
| hard decode + FR | 4810/4810 ok，FR=0 |
| target-residue fake mask 不变式 | unit tests 通过 |
| init：policy==reference ⇒ z=0, loss=log2 | 通过（GPU unit test，8 edges） |
| paired sigma/noise（相同坐标 ⇒ z=0，且 policy≠ref） | 通过 |
| grad scope：仅 policy score_model，reference 无梯度 | 通过 |
| smoke pytest（7 文件） | 15 passed |
| manifold local-loss audit | **PASS**，p99 离群 1.54%（2405 边） |

## 5. Local Learnability

| audit | 结果 | gate |
|---|---|---|
| one-edge overfit（12 边 × 40 updates） | 12/12 final z>0，median Δz +0.346，nonfinite 0 | PASS |
| gradient direction（lr=1e-6 单步） | median Δz +9.75e-4 | PASS |
| sigma robustness（16 draws） | median positive fraction 0.9375 | PASS |

## 6. Pilot（held-out 4 cases × 8 samples）

| arm | global | local | reward | Δbase | ΔCF | invalid | FR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | 0 | -1.207 | 0 | -4.861 | 0 | 0 |
| CF-mini (A1) | 100 | 0 | +3.654 | +4.861 | 0 | 0 | 0 |
| Local-only (A2) | 0 | 100 | +0.362 | +1.568 | -3.292 | 0 | 0 |
| CF+Neg | 75 | 25 | +3.618 | +4.824 | -0.036 | 0 | 0 |
| **SL-CF-DPO (A4)** | 75 | 25 | **+2.446** | **+3.653** | **-1.208** | 0 | 0 |
| DirectionShuffle (A5) | 75 | 25 | +1.850 | +3.056 | -1.804 | 0 | 0 |
| CF+Local add-on (A6) | 100 | 25 | +3.420 | +4.627 | -0.234 | 0 | 0 |
| CF-125 (A7) | 125 | 0 | +4.502 | +5.709 | +0.848 | 0 | 0 |

每臂 100/125 updates ≈ 10 min（1×GPU）。A4 对 A1 的 case-level wins = 2/4。

## 7. Held-out Local Preference（37 边，3 个 sigma seed）

| arm | acc (mean ± sd) | both-neg | sign-flip | winner-drop | loser-gain |
|---|---:|---:|---:|---:|---:|
| Base | 0.500 ± 0.000 | 0.500 | 0.500 | 0.500 | 0.500 |
| CF-mini | 0.432 ± 0.058 | 0.364 | 0.346 | 0.263 | 0.444 |
| Local-only | 0.441 ± 0.025 | 0.455 | 0.385 | 0.316 | 0.500 |
| CF+Neg | 0.550 ± 0.067 | 0.545 | 0.423 | 0.368 | 0.556 |
| **SL-CF-DPO** | 0.532 ± 0.092 | 0.545 | 0.346 | 0.316 | 0.500 |
| DirectionShuffle | 0.541 ± 0.076 | 0.455 | 0.500 | 0.421 | 0.556 |

所有臂都在 chance（0.5）± 噪声内；shuffle 控制（0.541）不低于主方法（0.532）
⇒ 在 held-out 局部 pair 上看不到方向特异信号。

## 8. Final Reward（§49 判定）

```text
A2 > base        (+0.362 > -1.206)        → 不触发
A4 > A5          (+2.446 > +1.850)        → 不触发
A4 < A1          (+2.446 < +3.654)        → 触发
A6 <= A7         (+3.420 <= +4.502)       → 触发
```

GO gate（§48，主方法 A4）：

```text
ΔR_A4 > ΔR_A1 + 0.5 ?   +3.653 > +5.361   → NO
≥3/4 cases A4 ≥ A1 ?    2/4               → NO
invalid ≤ 1% ?          0%                → YES
FR mismatch = 0 ?       0                 → YES
conditional: A6 > A7 ≥ +0.5 ?  A6-A7 = -1.082 → NO
```

## 9. Failure Analysis

1. **方向信号存在但太弱**：local-only（A2）确实比 base 好（+1.568），
   说明 signed local DPO 可学习；但每步收益远低于 global CF 步（A7 比 A1 再多 +0.848）。
2. **local 步在 equal-compute 下亏**：A4 用 25 个 local 步替换 25 个 global 步，
   相对 A1 掉 -1.208；add-on A6 相对 A7 掉 -1.082 ⇒ 一个 local 步约等于 -1.0 reward，
   在当前 4-case pilot 分辨率下没有净增益。
3. **伤害集中在 sign_flip**：只加 both_negative（CF+Neg，+3.618）与纯 CF（+3.654）
   几乎持平；把 sign_flip 也加进来（A4，+2.446）才显著掉分
   ⇒ both-negative 修正中性，**signed flip 修正主动有害**。
4. **方向 shuffle 控制**：A5（+1.850）比 A4 差 0.596，说明 direction 有微小作用，
   但两者都远差于纯 CF，无法转化为优势。
5. **held-out 局部偏好**：3 seed 下全部臂落在 0.5 ± 0.09，shuffle ≥ main
   ⇒ 局部方向学习没有可复现的 held-out 泛化。
6. **manifold 不是瓶颈**：lift 接受率高、离群率 1.5%，几何 lift 不是失败原因。
7. **评估分辨率有限**：4 cases × 8 samples 的 case-level 差异很大
   （例如 A4−A1 在 4 个 case 上为 +0.36/−0.14/−1.46/+0.88），
   ±1 reward 的差异不排除噪声；但 pilot 的目的就是在噪声量级上做方向性判定，
   当前证据一致指向 NO-GO（reward 与 preference 诊断同向）。

## 10. Decision

### NO-GO

依据 §49 第三条件（A4 < A1 且 A6 ≤ A7）。
不进入 24-case / full / valid100 训练；不把 signed local preference 继续塞进
native BoltzGen。

## 11. Next Step

1. STOP（按 task book §87/§103）。
2. 保留 CF-DPO（weight/support 版本）作为当前算法主线；其收益来源是 credit weighting，
   而不是 signed local preference。
3. 若要继续 local 方向，需先解决 §9.5 的可复现性问题（held-out 局部偏好不可测），
   并证明单个 local 步的 equal-compute 收益 ≥ global CF 步；当前证据不支持。
