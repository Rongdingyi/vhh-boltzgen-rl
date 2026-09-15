#!/usr/bin/env bash
#SBATCH --job-name=slcf-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/slcf-gpu-%A_%a.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/slcf-gpu-%A_%a.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
STAGE=${STAGE:?set STAGE=manifold|one-edge|diagnostics|smoke|pilot|pilot-eval|heldout|full}
case "$STAGE" in
  manifold)
    $PY -u "$ROOT/experiments/signed_local/scripts/audit_lift_manifold.py" ;;
  one-edge)
    $PY -u "$ROOT/experiments/signed_local/scripts/local_one_pair_overfit.py" ;;
  diagnostics)
    $PY -u "$ROOT/experiments/signed_local/scripts/run_diagnostics.py" ;;
  smoke)
    $PY -m pytest -x -q "$ROOT/tests/test_signed_local_scheduler.py" \
      "$ROOT/tests/test_signed_local_edge_orientation.py" \
      "$ROOT/tests/test_signed_local_mask.py" \
      "$ROOT/tests/test_signed_local_dataset_invariants.py" \
      "$ROOT/tests/test_signed_local_dpo_init.py" \
      "$ROOT/tests/test_signed_local_pair_noise.py" \
      "$ROOT/tests/test_signed_local_grad_scope.py" ;;
  pilot)
    ARM=${ARM:?set ARM=cf|local|neg|main|shuffle}
    $PY -u "$ROOT/experiments/signed_local/scripts/train_pilot.py" --arm "$ARM" ;;
  pilot-eval)
    $PY -u "$ROOT/experiments/signed_local/scripts/eval_pilot.py" --with-base ;;
  heldout)
    $PY -u "$ROOT/experiments/signed_local/scripts/build_heldout_edges.py"
    $PY -u "$ROOT/experiments/signed_local/scripts/eval_local_preference.py" ;;
  full)
    ARM=${ARM:?set ARM=f0|f1|f2|f3}
    $PY -u "$ROOT/experiments/signed_local/scripts/train_full.py" --arm "$ARM" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
