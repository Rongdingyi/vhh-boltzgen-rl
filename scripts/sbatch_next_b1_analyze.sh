#!/usr/bin/env bash
#SBATCH --job-name=next-b1-analyze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-b1-analyze-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-b1-analyze-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src"
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
$PY -u "$ROOT/scripts/next_analyze_trajectory.py"
