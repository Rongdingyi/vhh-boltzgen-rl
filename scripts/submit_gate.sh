#!/usr/bin/env bash
# Submit one round-1 gate as a Slurm GPU job. Usage: submit_gate.sh <gate>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${1:?usage: submit_gate.sh <baseline|toy-reinforce|toy-grpo|scorer-overfit|evaluate>}"
case "$GATE" in
  baseline)       MODE=baseline; CONFIG=configs/round1_grpo.yaml; MAXUP=0 ;;
  toy-reinforce)  MODE=train; CONFIG=configs/round1_debug.yaml; MAXUP=100; CONFIG2=configs/round1_debug_toy_rf.yaml ;;
  toy-grpo)       MODE=train; CONFIG=configs/round1_debug_toy.yaml; MAXUP=100 ;;
  scorer-overfit) MODE=train; CONFIG=configs/round1_grpo.yaml; MAXUP=100 ;;
  evaluate)       MODE=evaluate; CONFIG=configs/round1_grpo.yaml; MAXUP=0 ;;
  *) echo "unknown gate $GATE" >&2; exit 2 ;;
esac
export BOLTZGEN_ROOT="${BOLTZGEN_ROOT:-/share/home/rongdingyi/programs/proteingen/boltzgen}"
export BOLTZGEN_CKPT="${BOLTZGEN_CKPT:-$BOLTZGEN_ROOT/ckpts/boltzgen1_ifold.ckpt}"
export VHH_SCORER_ROOT="${VHH_SCORER_ROOT:-/share/home/rongdingyi/programs/proteingen/vhh_guidance}"
export VHH_RL_DATA="${VHH_RL_DATA:-$ROOT/runs/round1/rl_manifest.jsonl}"
mkdir -p "$ROOT/runs/round1"
[ -f "$VHH_RL_DATA" ] || { echo "manifest missing: run scripts/make_manifest.py first" >&2; exit 2; }
EXTRA=""
if [ "$MODE" != "evaluate" ]; then EXTRA="--max-updates $MAXUP"; fi
# the trainer reads its mode from the config; map the gate name onto it
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --job-name=vhhrl-$GATE \
  --output=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/vhhrl-$GATE-%j.out \
  --error=/share/home/rongdingyi/programs/proteingen/VHHdata/audit_logs/vhhrl-$GATE-%j.err \
  --wrap="cd $ROOT && export BOLTZGEN_ROOT=$BOLTZGEN_ROOT BOLTZGEN_CKPT=$BOLTZGEN_CKPT VHH_SCORER_ROOT=$VHH_SCORER_ROOT VHH_RL_DATA=$VHH_RL_DATA PYTHONPATH=$ROOT/src:/share/home/rongdingyi/programs/proteingen/esm LAYERNORM_TYPE=torch HF_HUB_OFFLINE=1 && /share/home/rongdingyi/.conda/envs/vhh-guidance/bin/python -u -m vhh_rl.cli.main $MODE --config $ROOT/$CONFIG --output-dir $ROOT/runs/round1/$GATE $EXTRA"
