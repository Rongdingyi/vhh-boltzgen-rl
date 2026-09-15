# CF-DPO v2 proxy validation

- checkpoint: /share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/cf_dpo_v2/v2_u100/checkpoint_0100.pt
- edges: 78 (stratified random sample)
- sign-evaluated edges (dR != 0): 78
- stored-vs-node label mismatches: 0
- **sign agreement vs node reward: 0.526**
- median across-sigma noise (std of dh): 1.419e-04
- median |dh|: 4.791e-05

## 说明（修正版）

- ground truth 改为从节点 reward 重算 `R(b)-R(a)`：stored-vs-node **label mismatch = 0**
  （确认 E1/E2 已修复）；
- 分层随机抽样 78 条边（非顺序截断）；
- 训练后 v2 checkpoint 的代理符号一致率 **0.526**（≈随机），
  median |Δh| = 4.8e-5，σ 噪声 = 1.4e-4（**噪声 ≈ 3× 信号**）。
- 结论方向不变且现在是干净测量：**denoising-energy proxy 无法提供可靠的局部排序**。
