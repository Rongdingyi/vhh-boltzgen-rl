#!/usr/bin/env bash
#SBATCH --job-name=cfd2-report
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-report-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-report-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src"
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u "$ROOT/experiments/cf_dpo_v2/scripts/exp4_report.py"
