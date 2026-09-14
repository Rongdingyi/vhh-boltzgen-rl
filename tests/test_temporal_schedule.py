"""Temporal schedule tests (task book §29)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.temporal_schedule import g_hard, g_smooth  # noqa: E402

SIGMA_SEQ = 2.0


def test_smooth_monotone_decreasing_in_sigma():
    sigmas = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 10.0, 40.0]
    gs = [g_smooth(s, SIGMA_SEQ) for s in sigmas]
    assert all(a > b for a, b in zip(gs, gs[1:])), gs
    assert gs[-1] < 0.01 and gs[0] > 0.99


def test_smooth_midpoint_at_sigma_seq():
    assert abs(g_smooth(SIGMA_SEQ, SIGMA_SEQ) - 0.5) < 1e-9


def test_tau_controls_sharpness():
    x = SIGMA_SEQ * math.exp(0.5)
    assert abs(g_smooth(x, SIGMA_SEQ, tau=0.5) - 1 / (1 + math.e)) < 1e-9
    wide = g_smooth(x, SIGMA_SEQ, tau=2.0)
    narrow = g_smooth(x, SIGMA_SEQ, tau=0.1)
    assert 0.0 < narrow < wide < 1.0


def test_hard_gate():
    assert g_hard(1.0, SIGMA_SEQ) == 1.0
    assert g_hard(3.0, SIGMA_SEQ) == 0.0
    assert g_hard(SIGMA_SEQ, SIGMA_SEQ) == 1.0
