#!/usr/bin/env bash
# Phase 0 audit (task book §6). Writes docs/AUDIT.md. Read-only w.r.t. boltzgen.
set -euo pipefail
OUT="${1:-docs/AUDIT.md}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOLTZ="${BOLTZGEN_ROOT:-/share/home/rongdingyi/programs/proteingen/boltzgen}"
CKPT="${BOLTZGEN_CKPT:-$BOLTZ/ckpts/boltzgen1_ifold.ckpt}"
SCORER="${VHH_SCORER_ROOT:-/share/home/rongdingyi/programs/proteingen/vhh_guidance}"

mkdir -p "$(dirname "$OUT")"
{
echo "# Phase 0 audit"
echo
echo "## BoltzGen"
echo '```text'
echo "BOLTZGEN_ROOT=$BOLTZ"
(cd "$BOLTZ" && git remote get-url origin 2>/dev/null; git rev-parse HEAD; git status --short | head -5) || true
echo "checkpoint=$CKPT"
echo "checkpoint_sha256=$(sha256sum "$CKPT" | cut -d' ' -f1)"
echo "python=$(/usr/bin/python3 --version 2>&1)"
echo '```'
echo
echo "### IF anatomy (grep evidence)"
echo '```text'
grep -n "class InverseFoldingDecoder" "$BOLTZ/src/boltzgen/model/modules/inverse_fold.py" || true
grep -n "def sample" "$BOLTZ/src/boltzgen/model/modules/inverse_fold.py" || true
grep -n "inverse_fold_design_mask\|design_mask" "$BOLTZ/src/boltzgen/model/modules/inverse_fold.py" | head -6 || true
grep -n "torch.multinomial" "$BOLTZ/src/boltzgen/model/modules/inverse_fold.py" || true
grep -n "sampling_temperature" "$BOLTZ/src/boltzgen/model/modules/inverse_fold.py" | head -4 || true
grep -n "structure_module = InverseFoldingDecoder" "$BOLTZ/src/boltzgen/model/models/boltz.py" || true
grep -n "inverse_folding_encoder(feats)" "$BOLTZ/src/boltzgen/model/models/boltz.py" || true
echo '```'
echo
echo "## VHH scorer"
echo '```text'
echo "VHH_SCORER_ROOT=$SCORER"
grep -n "camelid_native_likeness_score\|nativeness_margin" "$SCORER/src/vhh_guidance/adapters/vhh_source_adapter.py" | head -4 || true
grep -n "def score_sequences" "$SCORER/src/vhh_guidance/adapters/vhh_source_adapter.py" || true
echo '```'
} > "$OUT"
echo "wrote $OUT"
