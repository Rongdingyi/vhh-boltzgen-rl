# Implementation Report (vhh_boltzgen_rl round 1)

## 1. Environment
- BoltzGen root: `/share/home/rongdingyi/programs/proteingen/boltzgen`
- commit: `a3149cf18eeb58648d1abbb27539bd73f746cdda`（与任务书参考版本一致，无本地改动）
- IF checkpoint: `ckpts/boltzgen1_ifold.ckpt` sha256 `dd4cf108…d56578`
- scorer root: `/share/home/rongdingyi/programs/proteingen/vhh_guidance`（read-only，frozen VHH-source ensemble）
- policy env: `vhh-guidance`（torch 2.x cu126, python 3.11）；scorer env: `vhh-guidance-esmc`（ESM-C）
- 全部计算经Slurm；policy/scorer跨env经JSONL subprocess worker（§7.2）

## 2. Local BoltzGen IF anatomy（代码证据见 docs/AUDIT.md）
- encoder：`InverseFoldingEncoder.forward(feats) -> (edge_idx, valid_mask, s, z)`（boltz.py:528）
- decoder：`model.structure_module` == `InverseFoldingDecoder`（inverse_fold=True, boltz.py:375）
- sample()：`@torch.no_grad()`，全局RNG（randperm→逐位multinomial），canonical切片 `[2:22]`，per-residue约束mask（Cys禁用），symmetric tying，T=0.1
- forward()：随机edge visibility（训练增强）——不可用作trajectory evaluator（§3.2）
- 训练态与采样态是**两个随机过程**：本实现完全不使用forward() logits

## 3. Added files（全部新增于本仓库，零改动boltzgen/vhh_guidance）
- `src/vhh_rl/adapters/boltzgen_if.py`：官方CLI in-process捕获（encoder hook抓s/z/edge_idx/valid_mask/feats，decoder.sample入口即断）＋ `rollout_with_logprobs`（逐行复刻sample()＋事件记录）＋ `replay_actions`（同order同history同约束teacher-force，可选decoder模块，policy带梯度/reference无梯度）＋ FR/序列还原
- `src/vhh_rl/rl/types|advantages|losses|trainer.py`：Trajectory/PolicyEvent；rank与group-z（case内分组、tie均值、零方差置零）；REINFORCE与clipped-GRPO＋exact 20-AA categorical KL；每trajectory长度归一；update_epochs=2
- `src/vhh_rl/rewards/scoring.py`＋`scorer_worker.py`：ScorerAdapter（sha1缓存sqlite、persistent JSONL worker、协议与日志分流）、ToySequenceReward（可达AA内选target）
- `src/vhh_rl/data/case.py`、`cli/{main,common,baseline,train,evaluate}.py`：manifest校验、官方spec复用（backbone旁design.yaml，绝不重造CDR编号）、baseline/best-of-N、eval框架
- `scripts/{audit_repo.sh,make_manifest.py,parity_gate.py,submit_gate.sh}`、`run.sh`（Gate顺序入口）
- `configs/{round1_debug,round1_grpo,round1_smoke,scorer}.yaml`

## 4. RL trajectory definition
- state：结构条件（冻结encoder的s/z/edges）＋已decode history＋decode order位置
- action：canonical 20-AA之一（offset=2映射回33-token空间）
- decode order：与原始sample()同源（全局RNG randperm过滤非design）
- fixed FR：来自项目自身CDR mask（manifest design/fr_positions，§8不重造编号）
- reward：sequence-only黑盒（VHH-source ensemble native_probability，direction=higher_is_better，AUDIT依据：它是一个概率值）

## 5-7. Rollout / Replay / KL
rollout与sample()逐行对应（parity gate证明等价）；replay严格teacher-force、无重采样、policy带梯度/reference无梯度且同一history；KL为每event的完整20维categorical KL（constraint-masked项p→0无NaN）。

## 8. Parameter freezing
capture后全冻结；trainer只解冻`adapter.decoder.parameters()`并断言`isinstance(decoder, InverseFoldingDecoder)`；reference=deepcopy.eval()+requires_grad False；optimizer只收decoder参数。参考hash/base checkpoint sha256写入checkpoint metadata（§13/§48）。

## 9. Scorer adapter
黑盒、eval+no_grad、persistent worker（vhh-guidance-esmc python）、sqlite缓存（sha1→raw）、方向变换在adapter内统一（higher=better）。1-based/0-based位置换算在worker边界完成。

## 10. Tests
| test | result |
|---|---|
| test_advantage (4) | PASS |
| test_kl (3) | PASS |
| parity gate (10 seeds×2 case) | PASS |
| replay parity (20 trajs) | PASS (6.8e-6) |
| scorer determinism/batch | PASS |
| toy REINFORCE / GRPO | PASS |
| scorer overfit | PASS |
（pytest全量在GPU作业内运行；其余gate走Slurm job，均留审计日志）

## 11. Known limitations
1. NO CLEAN TRAIN SPLIT——本轮只有debug/overfit，无held-out结论。
2. 每case需跑一次官方featurisation（~秒级），未做encoder缓存加速（§59有意为之）。
3. 事故记录：开发中误删并已bit-exact恢复1个`boltzgen_run`目录（详见ROUND1_RESULTS §9.3），代码层已加防护。
4. toy reward的target选择依赖baseline可达性（constraint-aware），不具生物意义。
5. merge/多GPU/LoRA/bf16均未启用（第一轮优先正确性）。
