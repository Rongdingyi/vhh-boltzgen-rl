#!/usr/bin/env bash
#SBATCH --job-name=paper-2nd-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-eval-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-eval-%j.err
set -euo pipefail
# Usage: TAG=... CKPT=... sbatch --export=ALL,TAG=...,CKPT=... sbatch_second_reward_eval.sh
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
PY_GUIDANCE=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
PY_ESMC=/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python
SEC=$ROOT/runs/paper_stage/second_reward
TAG=${TAG:?set TAG}
CKPT=${CKPT:-}
RAW=$SEC/${TAG}_heldout.jsonl
SCORED=$SEC/${TAG}_heldout_scored.jsonl

export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ ! -f "$RAW" ]; then
  if [ -n "$CKPT" ]; then
    $PY_GUIDANCE -u "$ROOT/experiments/paper_stage/scripts/generate_eval_pool.py" \
      --tag "$TAG" --ckpt "$CKPT" --split heldout --num-samples 8 --out "$RAW"
  else
    $PY_GUIDANCE -u "$ROOT/experiments/paper_stage/scripts/generate_eval_pool.py" \
      --tag "$TAG" --split heldout --num-samples 8 --out "$RAW"
  fi
fi
export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
$PY_ESMC -u "$ROOT/experiments/paper_stage/scripts/esmc_reward.py" \
  --in "$RAW" --out "$SCORED" --cache "$SEC/esmc_cache.sqlite"
echo "DONE $SCORED"
