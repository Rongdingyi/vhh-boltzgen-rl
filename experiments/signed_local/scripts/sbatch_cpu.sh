#!/usr/bin/env bash
#SBATCH --job-name=slcf-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/slcf-cpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/slcf-cpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=build-edges|audit-edges|protocol|report}
case "$STAGE" in
  build-edges)
    $PY -u "$ROOT/experiments/signed_local/scripts/build_local_edges.py"
    $PY -u "$ROOT/experiments/signed_local/scripts/audit_local_edges.py" ;;
  audit-edges)
    $PY -u "$ROOT/experiments/signed_local/scripts/audit_local_edges.py" ;;
  protocol)
    $PY -u "$ROOT/experiments/signed_local/scripts/write_frozen_protocol.py" ;;
  report)
    $PY -u "$ROOT/experiments/signed_local/scripts/make_report.py" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
