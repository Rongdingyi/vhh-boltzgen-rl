#!/usr/bin/env bash
#SBATCH --job-name=paper-2nd-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-audit-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-audit-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
export HF_HUB_OFFLINE=1
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python
$PY -u "$ROOT/experiments/paper_stage/scripts/audit_second_reward.py"
