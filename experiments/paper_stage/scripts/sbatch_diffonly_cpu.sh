#!/usr/bin/env bash
#SBATCH --job-name=paper-diffonly-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-diffonly-cpu-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-diffonly-cpu-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=audit|select|report}
cd "$ROOT/experiments/paper_stage/scripts"
case "$STAGE" in
  audit)  $PY -u audit_diffonly_weights.py ;;
  select) $PY -u select_diffonly_ckpt.py ;;
  report) $PY -u make_diffonly_report.py ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
