#!/usr/bin/env bash
#SBATCH --job-name=ag-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/ag-cpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/ag-cpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=freeze|audit|build-weights|validate-weights|pre-report|full-report|smoke}
cd "$ROOT/experiments/adaptive_granularity/scripts"
case "$STAGE" in
  freeze)           $PY -u freeze_protocol.py ;;
  audit)            $PY -u audit_conflict.py ;;
  build-weights)    $PY -u build_weights.py && $PY -u materialize_pairs.py ;;
  validate-weights) $PY -u validate_weights.py ;;
  pre-report|report) $PY -u make_pilot_report.py ;;
  full-report)      $PY -u make_full_report.py ;;
  smoke)
    $PY -m pytest -q "$ROOT/tests/test_ag_classification.py" \
      "$ROOT/tests/test_ag_no_floor.py" "$ROOT/tests/test_ag_strict_consensus.py" \
      "$ROOT/tests/test_ag_region_weights.py" "$ROOT/tests/test_ag_adaptive_weights.py" \
      "$ROOT/tests/test_ag_abstention.py" "$ROOT/tests/test_ag_shuffle.py" \
      "$ROOT/tests/test_ag_weight_invariants.py" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
