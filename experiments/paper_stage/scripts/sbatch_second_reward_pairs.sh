#!/usr/bin/env bash
#SBATCH --job-name=paper-2nd-pairs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-pairs-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-2nd-pairs-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
export HF_HUB_OFFLINE=1
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python
$PY -u "$ROOT/experiments/paper_stage/scripts/build_second_reward_pairs.py"
