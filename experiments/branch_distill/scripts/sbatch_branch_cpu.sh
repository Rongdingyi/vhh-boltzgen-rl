#!/usr/bin/env bash
#SBATCH --job-name=branch-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/branch-cpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/branch-cpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=freeze|smoke|gate1-report|gate2-report|gate3-report}
cd "$ROOT/experiments/branch_distill/scripts"
case "$STAGE" in
  freeze)       $PY -u freeze_protocol.py ;;
  smoke)
    $PY -m pytest -q "$ROOT/tests/test_branch_schedule.py" \
      "$ROOT/tests/test_branch_state_shapes.py" "$ROOT/tests/test_branch_decode.py" \
      "$ROOT/tests/test_branch_teacher_select.py" \
      "$ROOT/tests/test_branch_changed_positions.py" \
      "$ROOT/tests/test_branch_local_target.py" \
      "$ROOT/tests/test_branch_query_replay.py" "$ROOT/tests/test_branch_local_loss.py" \
      "$ROOT/tests/test_branch_no_credit_dependency.py" \
      "$ROOT/tests/test_branch_online_dpo_weights.py" \
      "$ROOT/tests/test_branch_gate_logic.py" ;;
  gate1-report) $PY -u gate1_report.py ;;
  gate2-report) $PY -u gate2_report.py ;;
  gate3-report) $PY -u gate3_report.py ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
