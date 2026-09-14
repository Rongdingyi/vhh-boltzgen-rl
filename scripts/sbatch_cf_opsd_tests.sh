#!/usr/bin/env bash
#SBATCH --job-name=cfopsd-tests
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-tests-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/cfopsd-tests-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
export PYTHONPATH="$ROOT/src:$ROOT/scripts:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
export HF_HUB_OFFLINE=1
export LAYERNORM_TYPE=torch
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
cd "$ROOT"
$PY -m pytest -q tests/test_query_replay.py tests/test_student_grad.py tests/test_target_decode.py \
  tests/test_onpolicy_wiring.py tests/test_gate_checks.py tests/test_static_eval_rows.py
