"""Counterfactual sequence hard checks (task book §12)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.counterfactual import (  # noqa: E402
    build_region_cfs, build_single_residue_cfs,
)

PAIR = {"case_id": "c1", "winner_sample_id": "w", "loser_sample_id": "l"}
SEQ_W = "ACDEFGHIKL"
SEQ_L = "ACDQFGRIKL"  # differs at 3 (E->Q) and 6 (H->R)
SEQS = {"w": SEQ_W, "l": SEQ_L}
DESIGN = {"c1": (3, 6)}
FIXED = {"c1": tuple(i for i in range(10) if i not in DESIGN["c1"])}


def _valid(seq, base):
    assert len(seq) == len(base)
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_single_residue_cfs():
    rows = build_single_residue_cfs(PAIR, SEQS, DESIGN, FIXED)
    assert len(rows) == 4  # 2 positions x {drop,gain}
    for row in rows:
        base = SEQ_W if row["kind"] == "winner_drop" else SEQ_L
        _valid(row["sequence"], base)
        diff = [i for i in range(10) if row["sequence"][i] != base[i]]
        assert diff == [row["position"]]
        # FR unchanged
        for i in FIXED["c1"]:
            assert row["sequence"][i] == base[i]


def test_region_cfs():
    regions = {"cdr1": [3], "cdr2": [], "cdr3": [6]}
    rows = build_region_cfs(PAIR, SEQS, regions, FIXED)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"region_drop", "region_gain"}
    drop = next(r for r in rows if r["kind"] == "region_drop" and r["region"] == "cdr3")
    assert drop["sequence"][6] == SEQ_L[6]
    assert drop["sequence"][3] == SEQ_W[3]
    for i in FIXED["c1"]:
        assert drop["sequence"][i] == SEQ_W[i]


def test_identical_sequences_produce_no_cfs():
    pair = {"case_id": "c1", "winner_sample_id": "w", "loser_sample_id": "w2"}
    seqs = {"w": "ACDEF", "w2": "ACDEF"}
    rows = build_single_residue_cfs(pair, seqs, {"c1": (3,)}, {"c1": (0, 1, 2, 4)})
    assert rows == []


def test_fr_guard_fires_on_change():
    from vhh_rl.credit.counterfactual import _check

    with pytest.raises(ValueError):
        _check("ACDEF", "ACDQF", changed=[3], fixed=[3])
    with pytest.raises(ValueError):
        _check("ACDEF", "ACDEFG", changed=[3], fixed=[3])
