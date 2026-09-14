#!/usr/bin/env bash
#SBATCH --job-name=cfopsd-dbg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-dbg-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-dbg-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u "$ROOT/scripts/cf_opsd_debug_replay3.py"
