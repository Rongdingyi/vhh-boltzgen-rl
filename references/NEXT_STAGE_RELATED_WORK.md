# Next-Stage Related Work: 边界与差异化（task book §3）

本文件记录与现有工作的边界，避免把已有贡献当成本轮创新（任务书
`NATIVE_ATOM14_NEXT_STAGE_CREDIT_RESEARCH_TASK.md` §3）。

## 1. ProteinDPO / experimental-fitness DPO（2026 Nature Methods）

已有：sequence-level DPO、ranked DPO、weighted DPO、experimental scalar
fitness 对齐。

**不能宣称**：用 scalar reward 做 protein DPO。

## 2. ResiDPO / EnhancedMPNN

已有：residue-level structural reward、residue-level preference
optimization、preserve already-good residues。

**不能宣称**：首次做 residue-level protein DPO。

**我们的区别**：只有一个 sequence-level black-box reward，没有任何
residue-level annotation；通过 counterfactual interventions 从
sequence-level reward 自动恢复 residue credit，再映射到 continuous
atom14 geometry（`c_i^drop` / `c_i^gain` / `c_i^cons`）。

## 3. ProtAlign / ProteinOPD（2026）

已有：multi-objective preference alignment、designability preservation、
semi-online DPO、catastrophic forgetting control。

**不能宣称**：classifier + structure score 加权作为主创新。

## 4. Generic timestep-aware Diffusion-DPO

image diffusion 已有 timestep-aware / curriculum DPO。

**不能宣称**：低噪声 timestep 加大权重本身。

**我们的问题**：在 geometry-encoded sequence–structure co-design
diffusion 中，sequence identity 的空间与时间 credit 如何同时被错误分配
（本轮 H1–H4）。

## 5. 本轮假设与结论映射

| 假设 | 内容 | 本轮结论 |
|---|---|---|
| H1 | spatial credit 稀疏 | **支持（STRONG）**：top30% mass=0.940, n_eff/n_diff=0.270 |
| H2 | uniform fake-atom DPO 含 off-target updates | **方向支持**：CF 加权显著优于 N3 与 B1/B2 控制 |
| H3 | sequence identity 在 diffusion 时间上晚于 backbone 收敛 | **不支持（WEAK）**：sigma_seq=0.31，无分离窗口 |
| H4 | spatiotemporal credit 改善 reward–geometry Pareto | **部分支持**：CF-only 达到 Strong（reward +5.57、CDR RMSD 1.949 Å ≤ 2.00）；temporal 本身无效 |

## 6. 论文主线的边界声明（本轮结果对应 §66 情况 B）

- 有效：**从黑盒 sequence-level reward 通过双向 counterfactual
  intervention 恢复 residue credit，并将其映射到 atom14 fake-atom 几何的
  preference 加权**（spatial credit）。
- 无效/暂不主张：temporal credit（sequence emergence 与 backbone 收敛同步，
  gate 后训练信号稀薄）。
- 因此主线暂定为 spatial counterfactual credit（不是
  "spatiotemporal credit assignment"）；若后续引入 structural proxy
  reward 再重估 temporal 假设。
