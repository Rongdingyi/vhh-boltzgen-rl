#!/usr/bin/env bash
#SBATCH --job-name=cfd2-graph
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-graph-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-graph-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src"
/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u "$ROOT/experiments/cf_dpo_v2/scripts/build_graph.py"
