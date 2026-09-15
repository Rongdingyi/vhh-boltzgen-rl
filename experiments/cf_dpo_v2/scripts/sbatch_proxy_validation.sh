#!/usr/bin/env bash
#SBATCH --job-name=cfd2-proxy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-proxy-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-proxy-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
CKPT=${CKPT:-}
NSIGMA=${NSIGMA:-32}
if [ -n "$CKPT" ]; then EXTRA="--ckpt $CKPT"; else EXTRA=""; fi
$PY -u "$ROOT/experiments/cf_dpo_v2/scripts/proxy_validation.py" $EXTRA --n-sigma "$NSIGMA"
