#!/usr/bin/env bash
#SBATCH --job-name=online-pref-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/online-pref-cpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/online-pref-cpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=freeze|smoke|p0-check|phase0-report|sigma-audit|phase1-report}
cd "$ROOT/experiments/online_pref/scripts"
case "$STAGE" in
  freeze)  $PY -u freeze_protocol.py ;;
  p0-check) $PY -u p0_check.py ;;
  phase0-report) $PY -u phase0_report.py ;;
  sigma-audit) $PY -u sigma_domain_audit.py ;;
  phase1-report) $PY -u phase1_report.py ;;
  smoke)
    $PY -m pytest -q "$ROOT/tests/test_online_pair_all_changed.py" \
      "$ROOT/tests/test_online_pair_no_geometry_dependency.py" \
      "$ROOT/tests/test_online_pair_bank.py" \
      "$ROOT/tests/test_online_query_accounting.py" \
      "$ROOT/tests/test_online_update_schedule.py" \
      "$ROOT/tests/test_online_pref_gates.py" \
      "$ROOT/tests/test_conditional_sigma.py" \
      "$ROOT/tests/test_conditional_sigma_mass.py" \
      "$ROOT/tests/test_prefix_suffix_disjoint.py" \
      "$ROOT/tests/test_temporal_dpo_shared_randomness.py" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
