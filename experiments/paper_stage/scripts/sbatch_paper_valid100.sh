#!/usr/bin/env bash
#SBATCH --job-name=paper-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-v100-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-v100-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
TASKS=${PAPER_V100_TASKS:-$ROOT/runs/paper_stage/valid100_tasks.tsv}
i=${SLURM_ARRAY_TASK_ID}
line=$(sed -n "$((i + 1))p" "$TASKS")
arm=$(echo "$line" | cut -f1)
ckpt=$(echo "$line" | cut -f2)
shard=$(echo "$line" | cut -f3)
echo "[paper-v100] task $i arm=$arm shard=$shard"
$PY -u "$ROOT/scripts/native_eval.py" --arm "$arm" --ckpt "$ckpt" \
    --split valid100 --num-samples 8 --shard "$shard" --nshards 4
