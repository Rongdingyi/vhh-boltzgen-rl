"""CF-DPO v2 edge-sampling regression tests (reviewer-found bug #3).

same-seq edges must live in their own branch: with ``same_seq_ratio=0.25`` the
empirical same-seq frequency must be ~25%, not ~53%.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_dpo_v2.sampling import choose_edge  # noqa: E402

EDGES = (
    [{"kind": "global"}] * 32
    + [{"kind": "drop"}] * 86
    + [{"kind": "gain"}] * 81
    + [{"kind": "same_seq"}] * 167
)


def _freq(variant: str, ratio: float, n: int = 20000) -> dict:
    rng = random.Random(0)
    counts = {"global": 0, "drop": 0, "gain": 0, "same_seq": 0}
    for _ in range(n):
        counts[choose_edge(EDGES, variant, ratio, rng)["kind"]] += 1
    return {k: v / n for k, v in counts.items()}


def test_v2_same_seq_frequency_matches_config():
    freq = _freq("v2", 0.25)
    assert abs(freq["same_seq"] - 0.25) < 0.02, freq
    pref = freq["drop"] + freq["gain"]
    assert abs(pref - 0.75 * 0.75) < 0.02, freq
    assert abs(freq["global"] - 0.75 * 0.25) < 0.02, freq


def test_signed_never_samples_same_seq():
    freq = _freq("signed", 0.25)
    assert freq["same_seq"] == 0.0, freq


def test_branches_are_disjoint():
    """same-seq must not be reachable through the preference-local branch."""
    same_seq = [e for e in EDGES if e["kind"] == "same_seq"]
    pref_local = [e for e in EDGES if e["kind"] in ("drop", "gain")]
    assert not set(map(id, same_seq)) & set(map(id, pref_local))
