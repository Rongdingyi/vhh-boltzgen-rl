#!/usr/bin/env bash
#SBATCH --job-name=next-v100-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%3
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-v100-eval-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-v100-eval-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
R=$ROOT/runs/next_stage
i=${SLURM_ARRAY_TASK_ID}
TAGS=(full_b1_s0450 full_b2_s0450 full_b3_s0500)
CKPTS=($R/full_random_sparse/checkpoint_0450.pt $R/full_shuffle/checkpoint_0450.pt $R/full_cf/checkpoint_0500.pt)
arm=${TAGS[$((i % 3))]}
ckpt=${CKPTS[$((i % 3))]}
shard=$((i / 3))
echo "[v100] arm=$arm shard=$shard/4"
$PY -u "$ROOT/scripts/native_eval.py" --arm "$arm" --ckpt "$ckpt" \
    --split valid100 --num-samples 8 --shard "$shard" --nshards 4
