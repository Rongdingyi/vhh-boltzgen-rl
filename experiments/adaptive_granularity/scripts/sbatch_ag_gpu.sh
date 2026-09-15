#!/usr/bin/env bash
#SBATCH --job-name=ag-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/ag-gpu-%A.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/ag-gpu-%A.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=pilot|pilot-eval|pilot-3seed|pilot-3seed-eval|gradient-conflict|full|full-eval|full-seeds}
cd "$ROOT/experiments/adaptive_granularity/scripts"
case "$STAGE" in
  pilot)
    ARM=${ARM:?set ARM=current|nofloor|strict|region|adaptive|shuffle|eligible-cf}
    $PY -u train_pilot.py --arm "$ARM" ;;
  pilot-eval)
    ARMS_ARG=""; [ -n "${ARMS:-}" ] && ARMS_ARG="--arms $ARMS"
    $PY -u eval_pilot.py $ARMS_ARG ;;
  pilot-3seed)
    SEEDS=${SEEDS:-42 43 44}
    for arm in current adaptive; do
      for seed in $SEEDS; do
        $PY -u train_pilot.py --arm "$arm" --seed "$seed" --steps 100
      done
    done ;;
  pilot-3seed-eval)
    $PY -u eval_3seed.py ;;
  gradient-conflict)
    $PY -u gradient_conflict_audit.py ;;
  full)
    ARM=${ARM:?set ARM=f0|f1|f2|f3|f4}
    VARIANT_ARG=""; [ -n "${VARIANT:-}" ] && VARIANT_ARG="--variant $VARIANT"
    $PY -u train_full.py --arm "$ARM" $VARIANT_ARG ;;
  full-eval)
    $PY -u eval_full.py ;;
  full-seeds)
    $PY -u train_full.py --arm f0 --seed 42 && $PY -u train_full.py --arm f2 --seed 42 \
      && $PY -u train_full.py --arm f0 --seed 43 && $PY -u train_full.py --arm f2 --seed 43 \
      && $PY -u train_full.py --arm f0 --seed 44 && $PY -u train_full.py --arm f2 --seed 44 \
      && $PY -u eval_full.py --arms f0 f2 --seeds 42 43 44 ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
