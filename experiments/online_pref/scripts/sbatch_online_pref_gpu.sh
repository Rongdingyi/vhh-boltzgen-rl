#!/usr/bin/env bash
#SBATCH --job-name=online-pref-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/online-pref-gpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/online-pref-gpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=phase0|phase0-arm|phase0-eval}
cd "$ROOT/experiments/online_pref/scripts"
case "$STAGE" in
  phase0-arm)
    ARM=${ARM:?set ARM=a|b|v}; SEED=${SEED:?set SEED}
    $PY -u phase0_train.py --arm "$ARM" --seed "$SEED" ;;
  phase0)
    for seed in 20260915 43 44; do
      for arm in a b v; do
        $PY -u phase0_train.py --arm "$arm" --seed "$seed"
      done
    done ;;
  phase0-eval) $PY -u phase0_eval.py ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
