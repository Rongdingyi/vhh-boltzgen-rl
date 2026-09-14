#!/usr/bin/env bash
# CF-OPSD feasibility probe driver (task book §83). Gate-ordered; stops on failure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMD="${1:-help}"

case "$CMD" in
  audit)
    /usr/bin/python3 "$ROOT/scripts/cf_opsd_audit.py" ;;
  rollout)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_rollout.sh" ;;
  query-probe)
    PYTHONPATH="$ROOT/src" /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python \
      "$ROOT/scripts/cf_opsd_probe_queries.py" ;;
  target-probe)
    PYTHONPATH="$ROOT/src" /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python \
      "$ROOT/scripts/cf_opsd_build_targets.py" ;;
  realization-probe)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_realization.sh" ;;
  static)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_static.sh" ;;
  cfdpo-mini)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_cfdpo_mini.sh" ;;
  static-eval)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_static_eval.sh" ;;
  onpolicy)
    sbatch "$ROOT/scripts/sbatch_cf_opsd_onpolicy.sh" ;;
  *)
    echo "usage: bash run_cf_opsd_probe.sh {audit|rollout|query-probe|target-probe|realization-probe|static|cfdpo-mini|static-eval|onpolicy}" >&2
    exit 2 ;;
esac
