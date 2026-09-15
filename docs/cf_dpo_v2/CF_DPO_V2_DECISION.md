# CF-DPO v2 Feasibility Decision（最终，基于修正后有效实验）

## 0. 状态链

1. `de8f5c9` 初版：审阅发现 4 个实质错误（E1 drop 方向 / E2 node ID 覆盖 /
   E3 same-seq 双重采样 / E4 Exp2 方向）+ E5 proxy ground truth 污染 →
   初版 NO-GO 撤回为 INVALID（`CF_DPO_V2_ERRATA.md`）。
2. `bec73ef` 修复 E1–E5 后重跑：EXP4 v2 +0.69 / signed +0.54 vs CF-DPO-mini +4.39。
3. 第二轮审阅再发现 5 项（signed 采样 81/19、Exp2 的 mean(w) 退化对照、
   rerun 目录混 log、graph invariant 仅在测试中、κ 边训练边切换），
   修复见 `32c705c` / `3dcc3bf`。
4. κ 预校准第一版在 policy=reference 处测尺度（h≡0 → 钳到 1e8）无效，
   改为 **warmup 10 步 → 恢复 base → 固定 κ** 后重跑（本节所有数字）。

## 1. 最终有效结果

### Exp4 pilot（4 train / 4 held-out，matched 50/100 updates）

| arm | held-out reward | Δ vs base | wins |
|---|---|---|---|
| base | −1.206 | — | — |
| **CF-DPO-mini u100** | **+3.186** | **+4.392** | **4/4** |
| CF-DPO-mini u50 | +1.380 | +2.586 | 4/4 |
| v2 u100 | +0.044 | +1.250 | 3/4 |
| v2 u50 | −0.378 | +0.829 | 2/4 |
| signed u100 | −1.187 | +0.020 | 2/4 |
| signed u50 | −0.900 | +0.307 | 3/4 |

κ_eff（warmup-restart）：signed 1.52e4，v2 9.34e3；objective scale 全程固定。

### Proxy 校验（nσ=32，分层抽样，ground truth 由节点 reward 重算）

| 指标 | 值 |
|---|---|
| label mismatch | 0 |
| 代理符号一致率 | **0.615** |
| 噪声 / 信号 | ≈ 6.7× |

### Exp2 精确小模型（忠实 per-site 加权对照）

signed/v2 序列 KL 0.0005–0.0007（符号正确率 0.994）vs 加权类 6.2–7.2（0.64–0.65）；
v2 的同结果边把条件几何 KL 压到 0.0000。

## 2. Final Decision

**NO-GO（针对 "global denoising-energy potential + BCE local graph fitting" 的 v2 形态）**

依据 proposal §14：

- 同池普通 DPO（CF-DPO-mini +4.39，4/4）显著优于完整 v2（+1.25，3/4）；
- 代理校验显示该载体的局部排序不可靠（0.615，噪声 6.7× 信号）；
- Exp2 表明"有向局部比较"在精确模型下有效，但在原生能量代理上无法兑现。

## 3. 失败机制（与 CF-DPO 的对比）

```text
CF-DPO : 局部 residue credit → 局部 fake-atom denoising error   （信号集中）
v2     : 局部 residue preference → 全结构 denoising energy proxy（信号被淹没）
```

一个残基的偏好差异被全蛋白坐标误差 + σ 采样噪声淹没（SNR < 1），
因此这不是"contextual signed preference 这个想法错了"，而是**载体选错了**。

## 4. 保留资产

- Exp1 数据复核（36.1% 翻转；94% 稀疏性为估计器截断产物）；
- Exp3 合法几何 lift（86.5% 接受率、FR mismatch 0、同序列 M=2 覆盖 100%）；
- CF-DPO-mini 匹配基线；修正后的图/裁剪/校准工具与回归测试。

## 5. Recommendation

- **keep CF-DPO main line**；不继续扩展当前 global-energy v2 形态。
- 若未来重启 signed/context-dependent preference，必须让它作用于
  **local conditional diffusion signal**（例如 per-residue/per-atom 的条件项），
  而不是整结构的 energy proxy；并先验证该信号自身的 SNR（本轮的 proxy 校验
  可作为最低门槛模板）。
- 不通过增加正则/σ 平均/更多 lift 数据来"调漂亮"。

## 产物索引

- 文档：`CF_DPO_V2_ERRATA.md`、`EXP2_SMALL_MODEL.md`、`EXP2_COUNTEREXAMPLE.md`、
  `EXP3_GEOMETRY_LIFT.md`、`EXP3_PROXY_VALIDATION.md`、`EXP4_PILOT.md`、本文档
- 代码：`src/vhh_rl/cf_dpo_v2/{types,sampling,small_model,geometry_lift,graph_dataset,signed_trainer}.py`
- 脚本：`experiments/cf_dpo_v2/scripts/*`；`run.sh cfd2-*`
- 数据：`runs/cf_dpo_v2/{small_model,geometry_lift,graph,signed_u100,v2_u100,proxy_validation}`
