"""No-silent-fallback rule across every builder (task book §17/§78)."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.credit.adaptive_granularity import (  # noqa: E402
    adaptive_weights, no_floor_weights, strict_consensus_weights,
    strict_region_weights,
)

UNSTABLE = [{"position": 1, "region": "cdr1", "c_drop": 1.0, "c_gain": -1.0,
             "c_cons": 0.0},
            {"position": 2, "region": "cdr1", "c_drop": -1.0, "c_gain": 1.0,
             "c_cons": 0.0}]
REGIONS = {"cdr1": {"G_drop": 1.0, "G_gain": -1.0}}


def test_every_builder_returns_ineligible():
    for builder, args in ((no_floor_weights, ()),
                          (strict_consensus_weights, ()),
                          (strict_region_weights, (REGIONS,)),
                          (adaptive_weights, (REGIONS,))):
        out = builder(UNSTABLE, *args)
        assert out.eligible is False, builder.__name__
        assert out.weights == {}, builder.__name__


def test_ineligible_result_drops_any_weight_field():
    from vhh_rl.credit.adaptive_granularity import WeightBuildResult
    res = WeightBuildResult(weights={1: 0.5, 2: 0.5}, eligible=False)
    assert res.weights == {}
