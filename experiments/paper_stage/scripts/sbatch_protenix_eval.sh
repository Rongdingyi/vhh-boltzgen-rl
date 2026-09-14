#!/usr/bin/env bash
#SBATCH --job-name=paper-pxeval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-pxeval-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-pxeval-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
export PYTHONPATH="$ROOT/src:$GUID/src"
/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python -u \
  "$ROOT/experiments/paper_stage/scripts/protenix_structure_eval.py"
