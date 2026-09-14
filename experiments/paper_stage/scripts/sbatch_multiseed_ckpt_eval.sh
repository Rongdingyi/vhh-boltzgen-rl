#!/usr/bin/env bash
#SBATCH --job-name=paper-cksel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-59%4
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-cksel-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-cksel-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
i=${SLURM_ARRAY_TASK_ID}
RUNS=(n3_s43 n3_s44 shuffle_s43 shuffle_s44 cf_s43 cf_s44)
run=${RUNS[$((i % 6))]}
step=$(printf "%04d" $((50 * (i / 6 + 1))))
CKPT=$ROOT/runs/paper_stage/$run/checkpoint_$step.pt
if [ ! -f "$CKPT" ]; then echo "missing $CKPT"; exit 1; fi
$PY -u "$ROOT/scripts/native_eval.py" --arm "paper_${run}_s${step}" --ckpt "$CKPT" \
    --split heldout --num-samples 8
