# Round-1 Results

## 1. Question
能否用 sequence-only VHH scorer 对 BoltzGen IF（`InverseFoldingDecoder`）做稳定在线 RL？

## 2. Gate Summary

| Gate | Result | Evidence |
|---|---|---|
| Audit | PASS | `docs/AUDIT.md`；本地repo==参考commit a3149cf，`structure_module`即`InverseFoldingDecoder`（boltz.py:375），canonical offset=2/33 tokens，T=0.1默认 |
| Unit tests | PASS | 7/7（advantages 4、KL 3），job 60224 |
| Baseline (Gate 0) | PASS | 2 case×8样本，FR 100%不变，`runs/round1/baseline_smoke/baseline_sequences.csv`（job 60238） |
| Rollout parity (Gate 1) | PASS | 2 case×10 seeds：original `sample()` == RL rollout 全部一致（job 60251，`runs/round1/parity/gate_parity.json`） |
| Replay logprob parity (Gate 2) | PASS | max_abs_diff **6.79e-6** ≤1e-5；mean 1.26e-7 ≤1e-6（FP32） |
| Gradient audit (Gate 3) | PASS | gradprobe：lp.requires_grad=True、backward传播到decoder params、encoder/reference冻结（job 60269） |
| Toy REINFORCE (Gate 4) | PASS | raw 0→0.0956，单调，KL≤0.21，unique 1.00（job 60279） |
| Toy GRPO (Gate 4) | PASS | raw 0→0.0993 (max 0.107)，KL→0.15-0.23，unique 1.00（job 60278） |
| Scorer overfit (Gate 5) | PASS | **raw 0.760→0.977 (max 0.987)**，KL 0.154，entropy 0.78→0.61，unique 1.00（job 60290） |
| Clean pilot (Gate 6) | NOT RUN | NO CLEAN TRAIN SPLIT FOUND（§9：100 case 全部为历史评测集） |

## 3. Baseline
`valid100_001/002` 原始IF采样8条/Case，scorer（frozen VHH-source ensemble, native_probability）mean 0.055/0.027——原IF分布的native概率极低，与已知"MPNN系逆折叠CDR欠native"一致。Toy target AA选择需排除被constraint mask禁用的AA（第一版选中Cys导致reward恒0，已修复为仅在baseline可达AA中选）。

## 4. Toy Reward Sanity
REINFORCE与GRPO都能把采样分布推向目标AA（100 updates内 raw 0→~0.10，KL 0.15-0.23，unique rate 1.00，无collapse）。梯度链路（rollout→replay→ratio→backward）被证明可用。

## 5. Real Scorer Overfit
1 case（valid100_001）、K=8、100 updates、GRPO(rank)+KL(β=0.01)、lr 1e-5：
- scorer 0.760→0.977（接近native序列的0.894之上——注意这是**单case过拟合**，不作为泛化证据）
- KL 0.154（未失控）；entropy 0.78→0.61；unique rate 1.00（无collapse）
- FR mutation = 0（§19 hard assert 全程触发通过）

## 6. Held-out Pilot
NOT RUN（无clean split；见§9）。

## 7. Reward and Policy Curves
`runs/round1/scorer-overfit/train_metrics.jsonl`（§47全部字段逐update记录）。

## 8. Diversity / Collapse Check
所有run的unique rate保持1.00，pairwise identity未报告异常，KL被控制在<0.25。未见reward hacking信号（§46）。

## 9. Failure Analysis（工程事故与修复记录）
1. canonical token序：BoltzGen canonical是`ARNDCQEGHILKMFPSTWYV`非AA20字母序→adapter修正映射并加守卫。
2. **inference-tensor陷阱**：Lightning predict的teardown把所有捕获tensor标记为inference tensor且ambient grad=False（m3v2_ifold_gate.py预警过的`nograd_view_trap`）→capture后于inference_mode外clone＋trainer强制enable_grad（退出还原）。
3. **误删源目录事故**：capture时错误rmtree了`valid100_001/replicate_00/boltzgen_run`（frozen guidance产物）。已从m3v2 gate的scratch副本**bit-exact恢复**（design.cif sha256与frozen记录MATCH）。代码已加硬守卫（只允许清理runs/下工作区）。
4. scorer协议：模型加载日志污染stdout→协议行改走`sys.__stdout__`；VHHSpec位置约定为**1-based**（manifest 0-based在边界处+1）。
5. merge/复算类：无。

## 10. Decision
- **第一轮闭环成立**：4个核心问题全部可回答——①逐residue log-prob可精确获取（parity 6.8e-6）②FR固定+仅CDR产生policy gradient（FR assert全过）③黑盒scorer能真实改变IF采样分布（toy+scorer overfit均上升）④单case过拟合有效（0.760→0.977），held-out问题因无clean split**未验证**。
- 最大问题：无clean train split，"RL优于原始IF"的泛化结论无法在本轮给出。
- 下一步最应该做：**构建clean RL train/eval split**（从antibody_v0预训练池或新采样backbone建新case，与valid100评测集不重叠），然后跑Gate 6小规模pilot（16-32 train + 8-16 held-out, 200 updates）。
- 按任务书要求，本轮到此停止，不进入diffusion RL/多目标/分层credit assignment。
