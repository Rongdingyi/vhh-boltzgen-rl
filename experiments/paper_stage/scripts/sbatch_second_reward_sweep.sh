#!/usr/bin/env bash
#SBATCH --job-name=paper-sr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-sweep-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-sweep-%j.err
set -euo pipefail
# Held-out second-reward (ESM-C CDR PLL) checkpoint sweep: generation + scoring.
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
PY_GUIDANCE=/share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python
PY_ESMC=/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python
SEC=$ROOT/runs/paper_stage/second_reward
ARMS=(uniform cf)
for arm in "${ARMS[@]}"; do
  for step in 0100 0200 0300 0400 0500; do
    TAG=srsweep_${arm}_s${step}
    RAW=$SEC/sweep/${TAG}.jsonl
    SCORED=$SEC/sweep/${TAG}_scored.jsonl
    if [ -f "$SCORED" ]; then echo "skip $TAG (scored)"; continue; fi
    if [ ! -f "$RAW" ]; then
      export PYTHONPATH="$ROOT/src:/share/home/rongdingyi/programs/proteingen/boltzgen/src"
      export HF_HUB_OFFLINE=1 LAYERNORM_TYPE=torch CUBLAS_WORKSPACE_CONFIG=:4096:8
      $PY_GUIDANCE -u "$ROOT/experiments/paper_stage/scripts/generate_eval_pool.py" \
        --tag "$TAG" --ckpt "$SEC/$arm/checkpoint_${step}.pt" --split heldout \
        --num-samples 8 --out "$RAW"
    fi
    export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
    $PY_ESMC -u "$ROOT/experiments/paper_stage/scripts/esmc_reward.py" \
      --in "$RAW" --out "$SCORED" --cache "$SEC/esmc_cache.sqlite"
  done
done
echo "SWEEP DONE"
