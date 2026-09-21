"""Main runtime path must not depend on geometry verification (task book §5)."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src/vhh_rl/online_pref"
FORBIDDEN_NAMES = {"carrier_audit", "verified_changed_positions", "c_drop",
                   "c_gain", "c_cons", "residue_credit"}
FORBIDDEN_MODULES = ("vhh_rl.branch_distill.local_target", "vhh_rl.cf_opsd.local_frame")


def test_no_geometry_filter_dependency():
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        names, modules = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        offenders += [f"{path.name}: {n}" for n in sorted(names & FORBIDDEN_NAMES)]
        offenders += [f"{path.name}: {m}" for m in sorted(modules)
                      if m in FORBIDDEN_MODULES]
    assert not offenders, offenders


def test_verified_support_requires_explicit_attachment():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import torch
    from vhh_rl.online_pref.dpo_step import support_positions
    from vhh_rl.online_pref.types import OnlinePreferencePair

    pair = OnlinePreferencePair(
        pair_id="p", case_id="c", round_index=0, group_id="g", branch_progress=0.6,
        branch_step=29, branch_sigma=5.0, winner_index=0, loser_index=1,
        winner_reward=1.0, loser_reward=0.0, reward_gap=1.0,
        winner_sequence="AAAA", loser_sequence="AAAB",
        changed_positions_all=(3,), changed_positions_verified=None,
        winner_coords=torch.zeros(5, 3), loser_coords=torch.zeros(5, 3),
        conditioning={}, meta={"design_positions": (3,)})
    assert support_positions(pair, "all") == (3,)
    try:
        support_positions(pair, "verified")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
