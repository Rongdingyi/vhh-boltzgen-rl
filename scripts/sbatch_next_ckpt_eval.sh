#!/usr/bin/env bash
#SBATCH --job-name=next-ckpt-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-29%3
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-ckpt-eval-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-ckpt-eval-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
R=$ROOT/runs/next_stage
i=${SLURM_ARRAY_TASK_ID}
ARMS=(full_random_sparse full_shuffle full_cf)
TAGS=(full_b1 full_b2 full_b3)
arm=${ARMS[$((i % 3))]}
tag=${TAGS[$((i % 3))]}
step=$(printf "%04d" $((50 * (i / 3 + 1))))
CKPT=$R/$arm/checkpoint_$step.pt
if [ ! -f "$CKPT" ]; then echo "missing $CKPT"; exit 1; fi
echo "[ckpt-eval] arm=$tag step=$step ckpt=$CKPT"
$PY -u "$ROOT/scripts/native_eval.py" --arm "${tag}_s${step}" --ckpt "$CKPT" \
    --split heldout --num-samples 8
