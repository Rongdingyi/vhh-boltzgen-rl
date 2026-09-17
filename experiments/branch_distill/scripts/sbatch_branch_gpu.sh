#!/usr/bin/env bash
#SBATCH --job-name=branch-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/branch-gpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/branch-gpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=gate1-prefix|gate1-probe|gate2-build|gate2-overfit|gate3|gate3-all|gate3-eval}
cd "$ROOT/experiments/branch_distill/scripts"
case "$STAGE" in
  gate1-prefix)  $PY -u capture_prefix_bank.py ;;
  gate1-probe)   $PY -u gate1_branch_probe.py ;;
  gate2-build)   $PY -u gate2_build_records.py ;;
  gate2-overfit) $PY -u gate2_overfit.py ;;
  gate3)
    ARM=${ARM:?set ARM=a|b|c|d}
    $PY -u gate3_train.py --arm "$ARM" ;;
  gate3-all)
    for arm in a b c d; do $PY -u gate3_train.py --arm "$arm"; done ;;
  gate3-eval)    $PY -u gate3_eval.py ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
