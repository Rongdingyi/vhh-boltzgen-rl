#!/usr/bin/env bash
#SBATCH --job-name=cfopsd-reports
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-reports-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-reports-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:$ROOT/scripts:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
$PY -u "$ROOT/scripts/cf_opsd_target_report.py"
$PY -u "$ROOT/scripts/cf_opsd_same_query_probe.py"
$PY -u "$ROOT/scripts/cf_opsd_static_pilot.py"
$PY -u "$ROOT/scripts/cf_opsd_static_eval.py"
$PY -u "$ROOT/scripts/cf_opsd_onpolicy_pilot.py" || echo "onpolicy refused as expected"
$PY -u "$ROOT/scripts/cf_opsd_report.py"
echo REPORTS_DONE
