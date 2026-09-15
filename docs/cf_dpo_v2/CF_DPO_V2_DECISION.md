> **状态更新（审阅后，2026-09-15）**：本文档原先的 **NO-GO（算法形态）判定已撤回**。
> 审阅发现 4 个会实质改变训练方向的代码错误（drop 边 dR 方向、node ID 覆盖、
> same-seq 双重采样、Exp2 基线方向），并污染了 proxy 校验的 ground truth；
> 详见 `CF_DPO_V2_ERRATA.md`。当前正确状态为：
>
> **INVALID EXPERIMENT / NEEDS CORRECTNESS RERUN**
>
> - 有效：Exp1 数据复核；Exp3 几何 lift 接受率（86.5%、FR 0、same-seq 覆盖 100%）；
>   CF-DPO-mini matched baseline（+4.39，不受 v2 bug 影响）。
> - 无效待重跑：Exp2 的 weighted-vs-signed 数字；Exp4 signed/v2 pilot 表；
>   proxy sign agreement = 0.475（ground truth 被污染）。
> - 下文 §5–§9 的旧结论保留为历史记录，**不再作为判定依据**；修正后的结果将
>   写入重跑版文档。
>
# CF-DPO v2 Feasibility Decision

任务书：`CF_DPO_V2_RESEARCH_PROPOSAL.md`（references/cf_dpo_v2/）。
执行范围：Exp1（已有数据复核）、Exp2（精确小模型）、Exp3（原生几何 lift +
代理校验）、Exp4 pilot（4 train / 4 held-out，matched 50/100 updates）。
本阶段不做 24-case 全量扩展（原因见 §6）。

## 1. Motivation

CF-DPO 的既有增益主要来自"反事实信息的使用"，但多 seed 后 CF−N3 平均只有
+1.087，且现有实现把每个位点的 drop/gain 压缩成非负权重。v2 的问题定义是：
**全局优选标签应被修正为背景依赖的有向局部偏好，同时不扭曲同一序列内部的
几何实现分布。**

## 2. Reference Difference

DiffusionOPSD 依赖可微 reward gradient 构造显式 target；本方案不改变
generator/reward 接口，而是在已解码的**结果空间**上做比较：

```text
节点 = 合法 atom14 几何（解码序列确定）
边   = global pair / 单点有向反事实 / 同序列不同构象（ΔR=0）
损失 = BCEWithLogits((H(X_b)-H(X_a))/tau, sigmoid(ΔR/tau))
H    = -kappa (ell_theta - ell_0)   # denoising-energy proxy
```

## 3. Query Audit（Exp1，复现方案包数据）

- 2,996 个残基事件：45.9% 双方正、18.0% 双方负、**36.1% 背景符号翻转**
  （0.05 容差后 33.2%）。
- 1,020 条（190/192 对）反事实查询的 reward **高于原 winner**；每对最佳单点
  回退提升中位 1.349 分。
- "top30 占 94% credit" 是 `c_cons` 截断估计器的产物；改用 abs(c_avg/drop/gain)
  时 top30 质量约 66%。

复核与数学检查（含两残基反例、三阶残差抵消、KL 链式分解、有限图势函数恢复）
全部通过；脚本与输入哈希随方案包提供。

## 4. Target / Geometry Construction（Exp3）

合法几何 lift（residue-local frame transplant + 官方硬解码接受）：

| class | attempts | exact | rate |
|---|---|---|---|
| sign_flip | 68 | 55 | 80.9% |
| both_negative | 60 | 55 | 91.7% |
| both_positive | 64 | 56 | 87.5% |
| **total** | **192** | **167** | **86.5%** |

FR mismatch **0**；同序列 M=2 实现覆盖 **100%**（decode-preserving 扰动，
σ=0.08，非刚体凑数）。失败全部记录（decode_invalid 18、third_sequence 7）。
**Gate PASS**：合法几何补全在这批数据上稳定可行。

## 5. Same-query Realization / Static Pilot（Exp4 pilot）

比较图：346 节点 / 366 边（global 32、drop 86、gain 81、same_seq 167），
4 个固定 train case；held-out 4 cases × 8 samples；与 CF-DPO-mini 同
optimizer updates（50/100）同 base checkpoint。

| arm | held-out reward | Δ vs base | wins |
|---|---|---|---|
| base | −1.206 | 0 | — |
| **CF-DPO-mini u50** | **+1.380** | **+2.586** | 4/4 |
| **CF-DPO-mini u100** | **+3.186** | **+4.392** | 4/4 |
| signed u50 | −2.410 | −1.203 | 0/4 |
| signed u100 | −1.396 | −0.190 | 1/4 |
| v2 u50 | −1.728 | −0.521 | 1/4 |
| v2 u100 | −0.430 | +0.776 | 4/4 |

第一次运行（未校准 κ）中 |z|~1e-4、loss 恒为 log2，模型基本没有更新；
保留在 `runs/cf_dpo_v2/{signed,v2}_u100_uncalibrated/` 作为对照，不作为判定依据。
重跑加入训练集固定尺度 κ 校准（κ_eff≈6.4–8.2e3）与 nσ=3 平均后仍远低于同预算
基线；signed 在 50→100 updates 间还出现性能退化（z 可达 −2.9、pre-clip grad 2154）。

## 6. 失败机制诊断：代理失配（决定性）

在 v2-u100 checkpoint 上对 80 条非 same-seq 边做 proxy 校验（nσ=8）：

| 指标 | 值 |
|---|---|
| 代理符号一致率 vs reward | **0.475**（≈随机） |
| median \|Δh\| | 6.1e-05 |
| median σ-噪声（Δh 的 std） | **2.3e-04（≈4× 信号）** |

即：denoising-loss 差值代理**不能按 reward 正确排序边的两端**，且其自身的
σ 采样噪声高于信号。这正是 proposal §11.2 标注的"最大数学风险"在原生模型上
的实际发生：`ℓ_θ` 是变分/能量代理，不是精确 log density。

在该代理下，signed 与 v2 的 BCE 目标无法产生正确的偏好更新；CF-DPO 之所以
有效，是因为它只使用 ℓ 的**差分符号结构**（winner<loser 的 DPO 目标），而不
依赖代理的绝对尺度/排序校准。

## 7. Efficiency

| arm | optimizer updates | scorer queries（构图） | 备注 |
|---|---|---|---|
| CF-DPO-mini | 50 / 100 | 0（复用既有 192-pair 数据） | 强基线 |
| signed / v2 | 50 / 100 | 同一图：约 192（CF 查询）+ 几何 lift 约 192 次解码 | 额外 GPU 解码成本 |

两者 updates 相同；v2 额外支付几何 lift 与代理 forward（每步 nσ×4 次），
成本更高但收益更低。

## 8. Failure Modes

1. **代理失配（决定性）**：Δh 信号 < σ 噪声，符号一致率≈0.5。
2. κ 尺度问题（已修）：未校准时 |z|~1e-4，学习停滞；说明该目标对尺度极敏感。
3. signed 目标不稳：z 尾部过大（−2.9），训练后期 drifts。
4. 数据侧无问题：Exp3 接受率 86.5%、FR 0、同序列实现 100%。

## 9. Final Decision

### NO-GO（算法形态）

依据 proposal §14 的停止条件：

```text
同池普通 DPO（CF-DPO-mini, +4.39）已远超完整 v2（+0.78）；
且代理校验显示 energy proxy 无法提供正确的局部排序。
→ 收缩贡献到"反事实偏好纠错 / 数据构造"，不宣称新的优化范式。
```

**保留并继续使用 CF-DPO 主线。**

## 10. 保留的可复用贡献（诚实范围）

1. **数据审计修正**：36% 局部偏好方向翻转；94% 稀疏性是估计器截断产物。
2. **合法几何 lift pipeline**（`src/vhh_rl/cf_dpo_v2/geometry_lift.py`）：
   86.5% 接受率、0 FR mismatch、100% 同序列 M=2 覆盖；可作为独立数据构造
   工具供后续方法（含普通 DPO 的局部 pair 训练）使用。
3. **精确小模型证据**（Exp2）：加权类方法符号正确率 ~0.2，signed ~0.62，
   same-seq 边把条件几何 KL 从 0.129 压到 0.0005——理论层面的结论成立。
4. **负结果本身**：原生 denoising proxy 的排序不可用（sign acc 0.475，
   noise/signal≈4）——这是 proposal 自己列为最大风险的项目的实测结论。

## 11. Recommendation

- **keep CF-DPO main line**（继续 paper-stage 的 claim 收缩路线）。
- 若未来继续此方向，前置条件应为：找到比 denoising-loss 差分更可靠的
  reference-relative score（例如显式条件采样估计或更好的变分目标），
  否则不应重新启用 signed/v2 训练。
- 不建议：加正则系数、加 σ 平均、加更多 lift 数据来"调漂亮"（proposal §14）。

## 产物索引

- 文档：`docs/cf_dpo_v2/{EXP2_SMALL_MODEL.md,EXP2_COUNTEREXAMPLE.md,EXP3_GEOMETRY_LIFT.md,EXP3_PROXY_VALIDATION.md,EXP4_PILOT.md,CF_DPO_V2_DECISION.md}`
- 代码：`src/vhh_rl/cf_dpo_v2/{small_model,geometry_lift,graph_dataset,signed_trainer}.py`
- 脚本：`experiments/cf_dpo_v2/scripts/*`
- 数据/日志：`runs/cf_dpo_v2/{small_model,geometry_lift,graph,signed_u100,v2_u100,proxy_validation}`
