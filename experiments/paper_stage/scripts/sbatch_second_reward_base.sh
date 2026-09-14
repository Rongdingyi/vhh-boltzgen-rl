#!/usr/bin/env bash
#SBATCH --job-name=paper-sr-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-base-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-base-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
export HF_HUB_OFFLINE=1
SEC=$ROOT/runs/paper_stage/second_reward
/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python -u \
  "$ROOT/experiments/paper_stage/scripts/esmc_reward.py" \
  --in "$SEC/base_heldout.jsonl" --out "$SEC/base_heldout_scored.jsonl" \
  --cache "$SEC/esmc_cache.sqlite"
echo BASE_SCORED
