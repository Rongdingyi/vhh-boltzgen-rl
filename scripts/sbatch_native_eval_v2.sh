#!/usr/bin/env bash
#SBATCH --job-name=native-eval2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/native-eval2-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/native-eval2-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
ARM=${ARM:?set ARM}
SPLIT=${SPLIT:-heldout}
CKPT=${CKPT:?set CKPT}
NUM=${NUM:-8}
NSHARDS=${NSHARDS:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
$PY -u "$ROOT/scripts/native_eval_v2.py" --arm "$ARM" --ckpt "$CKPT" --split "$SPLIT" --num-samples "$NUM" --shard "$SHARD" --nshards "$NSHARDS"
