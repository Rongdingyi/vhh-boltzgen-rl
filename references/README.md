# References
- HannesStark/boltzgen @ a3149cf18eeb58648d1abbb27539bd73f746cdda（本地 boltzgen/）
  参考：inverse_fold.py（sample/preconditioning）、boltz.py（structure_module/encoder接线）、
  task/predict/data_from_generated.py（feats来源）。未复制任何代码；通过CLI in-process捕获复用。
- LLNL/protein_tune_rl @ a74485c（未拉取，仅按任务书§36定位为概念参考）
  借鉴思想：antibody infilling + external scalar reward + KL 正则 + JSONL 日志。
  未复用其 IgLM/PPO trainer 结构。
- vhh_esmc_guidance/scripts/cdr_all/m3v2_ifold_gate.py（本项目自有）
  捕获模式与 nograd_view_trap/inference-tensor 教训的直接来源。

## Native atom14 post-training (round 1)
- Diffusion Model Alignment Using Direct Preference Optimization, Wallace et al., arXiv:2311.12908
  （Diffusion-DPO：winner/loser 共享 timestep 与 noise；current vs frozen reference 的
  denoising-loss 差；-logsigmoid(-beta/2 * (...))）。第一轮只实现 Diffusion-DPO。
- Training Diffusion Models with Reinforcement Learning, Black et al., ICLR 2024
  （DDPO）：仅作为后续方向记录，本轮未编码。
