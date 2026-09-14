# VHH-BoltzGen RL: round-1 task board

> Source of truth: `../VHH_BOLTZGEN_RL_ROUND1_OPENCODE_TASK.md` (referenced as "the task book").
> This run does **sequence-reward RL on the frozen BoltzGen inverse-folding decoder only**.

## Commands

```bash
export BOLTZGEN_ROOT=/share/home/rongdingyi/programs/proteingen/boltzgen
export BOLTZGEN_CKPT=$BOLTZGEN_ROOT/ckpts/boltzgen1_ifold.ckpt
export VHH_SCORER_ROOT=/share/home/rongdingyi/programs/proteingen/vhh_guidance
export VHH_RL_DATA=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/round1/rl_manifest.jsonl
pip install -e .
bash run.sh audit          # Phase 0 audit -> docs/AUDIT.md
bash run.sh baseline       # Gate 0
bash run.sh toy-reinforce  # Gate 4 (REINFORCE)
bash run.sh toy-grpo       # Gate 4 (GRPO)
bash run.sh scorer-overfit # Gate 5
bash run.sh evaluate       # Gate 6 (only with a clean split)
bash run.sh all-round1     # everything in gate order, stop on first hard failure
```

All compute goes through Slurm (`sbatch scripts/*.sh`); the login node only edits
files and submits. The policy runs in the `vhh-guidance` env (BoltzGen), the scorer
worker in `vhh-guidance-esmc` (ESM-C), connected over a JSONL pipe per task book §7.2.

## Files of record

- `docs/AUDIT.md` - Phase 0 findings (this is the only place interface facts may be stated)
- `docs/IMPLEMENTATION_REPORT.md` - what was built, gate-by-gate
- `docs/ROUND1_RESULTS.md` - final answers to the four task-book questions, then STOP
