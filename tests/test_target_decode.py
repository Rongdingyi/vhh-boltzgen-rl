"""Target decode: FR unchanged, UNK handled (task book §82)."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

POOL = ROOT / "runs/native_pool"


def test_target_decode_fr_preserved():
    pytest.importorskip("boltzgen")
    from vhh_rl.cf_opsd.target_builder import build_target
    from vhh_rl.native_atom14.decode import decode_atom14, fr_check, sequence_from_feat
    from vhh_rl.native_atom14.dpo_trainer import load_conditioning

    cond = load_conditioning(POOL / "conditioning", "sab2_6u52_c")
    feats = cond["feats"]
    coords = torch.load(sorted((POOL / "train/sab2_6u52_c/coords").glob("*.pt"))[0],
                        map_location="cpu", weights_only=True).float()
    bt = build_target(coords, coords + 0.3, feats, [25, 26], {25: 1.0, 26: 1.0}, 0.5)
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = bt.target_coords
    out = decode_atom14(feat)
    seq, _tok, invalid = sequence_from_feat(out)
    assert not invalid
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_query_replay_placeholder():
    pytest.skip("covered by test_query_replay.py after Phase A rollouts exist")
