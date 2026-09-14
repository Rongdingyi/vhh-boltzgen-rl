#!/usr/bin/env bash
#SBATCH --job-name=next-refold-build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-refold-build-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-refold-build-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src"
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u "$ROOT/scripts/native_refold_build_next.py"
