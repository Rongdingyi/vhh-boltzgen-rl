#!/usr/bin/env bash
#SBATCH --job-name=cfd2-exp2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-exp2-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfd2-exp2-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src"
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
$PY -u "$ROOT/experiments/cf_dpo_v2/scripts/exp2_counterexample.py"
$PY -u "$ROOT/experiments/cf_dpo_v2/scripts/run_small_model.py" --sites 8 --cases 24 --pool 16 --seeds 5 --steps 2000
