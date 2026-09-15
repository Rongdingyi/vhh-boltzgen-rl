"""Dataset-level invariants: unique ids + dR = R(cf)-R(anchor) (task book §15/§16)."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.signed_local.edge_validator import load_edges, validate_dataset  # noqa: E402

EDGES = ROOT / "runs/signed_local/edges/train_edges.jsonl"


def test_unique_edge_ids_and_dR_invariant():
    if not EDGES.is_file():
        pytest.skip("signed_local edge dataset not built in this checkout")
    report = validate_dataset(EDGES, split="train", decode=False)
    assert report["duplicate_edge_ids"] == 0
    assert report["dR_invariant_ok"] == report["n_edges"]
    assert report["fr_mismatch_total"] == 0


def test_validate_raises_on_broken_edge():
    if not EDGES.is_file():
        pytest.skip("signed_local edge dataset not built in this checkout")
    edges = load_edges(EDGES)
    e = edges[0]
    broken = type(e)(**{**e.__dict__, "dR": e.dR + 1.0})
    with pytest.raises(ValueError):
        broken.validate()
