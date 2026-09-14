#!/usr/bin/env bash
#SBATCH --job-name=cfopsd-query
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-query-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-query-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
$PY -u "$ROOT/scripts/cf_opsd_probe_queries.py"
