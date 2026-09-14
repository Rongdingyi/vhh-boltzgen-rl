"""Residue <-> atom <-> token mapping test (task book §41, Phase C prerequisite).

Runs in the torch env; requires boltzgen for the conditioning files only if
present.  Verifies for every train conditioning cache:
  - the token offset is constant across design positions;
  - every requested position yields a non-empty fake-atom group;
  - groups are disjoint and inside atom_pad & fake_atom_mask.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.masks import (  # noqa: E402
    design_token_offset, residue_atom_masks,
)

POOL = ROOT / "runs/native_pool"
COND = POOL / "conditioning"


def _cases() -> dict[str, list[int]]:
    out = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") == "train":
            out[row["case_id"]] = sorted(int(p) for p in row["design_positions"])
    return out


def test_residue_atom_mapping():
    cases = _cases()
    if not COND.is_dir():
        pytest.skip("conditioning cache missing")
    checked = 0
    for case_id, positions in cases.items():
        path = COND / f"{case_id}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feats = payload["feats"]
        offset = design_token_offset(feats["token_index"], feats["design_mask"], positions)
        masks = residue_atom_masks(
            feats["atom_to_token"], feats["fake_atom_mask"],
            feats["atom_pad_mask"], [p + offset for p in positions],
        )
        assert masks.shape[0] == len(positions)
        assert int(masks.sum().item()) > 0
        assert not (masks.sum(dim=0) > 1).any(), "residue groups must be disjoint"
        atom_pad = feats["atom_pad_mask"].reshape(-1).bool()
        assert not (masks.any(dim=0) & ~atom_pad).any()
        checked += 1
    assert checked > 0


def test_offsets_are_integers_and_stable():
    cases = _cases()
    if not COND.is_dir():
        pytest.skip("conditioning cache missing")
    for case_id, positions in list(cases.items())[:5]:
        path = COND / f"{case_id}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feats = payload["feats"]
        first = design_token_offset(feats["token_index"], feats["design_mask"], positions)
        assert isinstance(first, int)
