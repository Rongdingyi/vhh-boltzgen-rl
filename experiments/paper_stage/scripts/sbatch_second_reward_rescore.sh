#!/usr/bin/env bash
#SBATCH --job-name=paper-sr-rescore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-rescore-%j.out
#SBATCH --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/paper-sr-rescore-%j.err
set -euo pipefail
ROOT=/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl
GUID=/share/home/rongdingyi/programs/proteingen/vhh_esmc_guidance
PY_ESMC=/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python
SEC=$ROOT/runs/paper_stage/second_reward
export PYTHONPATH="$ROOT/src:$GUID/src:/share/home/rongdingyi/programs/proteingen/esm"
export HF_HUB_OFFLINE=1
for raw in $SEC/sweep/srsweep_*_scored.jsonl; do :; done
for raw in $SEC/sweep/srsweep_*.jsonl; do
  case "$raw" in *_scored.jsonl) continue;; esac
  out="${raw%.jsonl}_scored.jsonl"
  $PY_ESMC -u "$ROOT/experiments/paper_stage/scripts/esmc_reward.py" \
    --in "$raw" --out "$out" --cache "$SEC/esmc_cache.sqlite"
done
echo RESCORE_DONE
