#!/usr/bin/env bash
#SBATCH --job-name=next-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-smoke-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/next-smoke-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python

echo "== unit tests =="
$PY -m pytest -q \
  "$ROOT/tests/test_credit_math.py" \
  "$ROOT/tests/test_counterfactual_sequence.py" \
  "$ROOT/tests/test_credit_weights.py" \
  "$ROOT/tests/test_weight_shuffle.py" \
  "$ROOT/tests/test_temporal_schedule.py" \
  "$ROOT/tests/test_residue_atom_mapping.py" \
  "$ROOT/tests/test_weighted_loss_uniform_equivalence.py"

echo "== weighted DPO smoke (cf, 5 steps, 2 cases) =="
rm -rf "$ROOT/runs/next_stage/smoke_cf"
$PY -u "$ROOT/scripts/native_train_weighted.py" \
  --variant cf --beta 10.0 --max-steps 5 --smoke-steps 5 --max-cases 2 \
  --checkpoint-every 5 --output-dir "$ROOT/runs/next_stage/smoke_cf"

echo "== init gate check (raw DPO loss ~ log2) =="
$PY - <<'EOF'
import json, math
root = "/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl/runs/next_stage/smoke_cf"
with open(f"{root}/train_metrics.jsonl") as fh:
    first = json.loads(fh.readline())
raw = first["raw_dpo_loss"]
gap = abs(raw - math.log(2))
print(f"step1 raw_dpo_loss={raw:.8f} (log2={math.log(2):.8f}, gap={gap:.2e})")
assert gap < 5e-3, f"DPO init raw loss {raw} != log2"
import os
assert os.path.exists(f"{root}/checkpoint_0005.pt")
print("SMOKE GATES PASSED")
EOF
