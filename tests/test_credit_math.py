"""Credit math + sparsity unit tests (task book §14-§15)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.credit_metrics import (  # noqa: E402
    effective_n, residue_credits, topk_mass,
)
from vhh_rl.credit.pair_analysis import hamming_positions  # noqa: E402


def test_residue_credits():
    both_positive = residue_credits(0.0, 0.0, 2.0, 1.0)
    assert both_positive["c_avg"] == 1.5
    assert both_positive["c_cons"] == 1.0
    assert both_positive["disagreement"] == 1.0
    assert both_positive["sign_agree"]

    mixed = residue_credits(0.0, 0.0, 2.0, -1.0)
    assert mixed["c_avg"] == 0.5
    assert mixed["c_cons"] == 0.0

    negative = residue_credits(0.0, 0.0, -2.0, -1.0)
    assert negative["c_cons"] == 0.0
    assert negative["sign_agree"]


def test_topk_mass_and_neff():
    credits = [4.0, 3.0, 2.0, 1.0, 0.0, 0.0]
    mass = topk_mass(credits, n_diff=6, fractions=(0.5,))
    assert abs(mass["top50"] - (4 + 3 + 2) / 10) < 1e-9
    concentrated = effective_n([10.0, 0.0, 0.0, 0.0])
    assert abs(concentrated - 1.0) < 1e-6
    uniform = effective_n([1.0, 1.0, 1.0, 1.0])
    assert abs(uniform - 4.0) < 1e-5


def test_hamming_positions():
    seqs = "ACDEF", "ACDQF"
    assert hamming_positions(*seqs, positions=(0, 1, 2, 3, 4)) == [3]
    assert hamming_positions("AAAA", "AAAA", positions=(0, 1)) == []
