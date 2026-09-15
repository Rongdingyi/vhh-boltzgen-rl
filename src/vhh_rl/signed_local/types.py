"""SL-CF-DPO shared types and sign utilities (task book §7/§14/§16).

Hard conventions:
  * every edge stores ``dR = R(cf) - R(anchor)``            (§16)
  * ``winner_drop``: dR = -c_drop ; ``loser_gain``: dR = +c_gain
  * reward is used ONLY for orientation + threshold + logging (never magnitude)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOL = 0.05


def robust_sign(x: float, tol: float = TOL) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def classify_event(s_drop: int, s_gain: int) -> str:
    if s_drop == 1 and s_gain == 1:
        return "both_positive"
    if s_drop == -1 and s_gain == -1:
        return "both_negative"
    if s_drop * s_gain == -1:
        return "sign_flip"
    return "weak"


def preferred_side(dR: float) -> str:
    """'cf' if the counterfactual endpoint is preferred, 'anchor' otherwise."""
    if dR > TOL:
        return "cf"
    if dR < -TOL:
        return "anchor"
    return "skip"


@dataclass
class LocalPreferenceEdge:
    edge_id: str

    case_id: str
    pair_id: str
    position: int

    context: str            # "winner_drop" | "loser_gain"
    event_class: str        # "both_negative" | "sign_flip" | ...

    anchor_sample_id: str
    donor_sample_id: str

    anchor_sequence: str
    cf_sequence: str

    anchor_reward: float
    cf_reward: float
    dR: float

    preferred_side: str     # "anchor" | "cf"

    anchor_coords_path: str
    cf_coords_path: str

    c_drop: float
    c_gain: float

    touched_atoms: int
    moved_rms: float
    invalid: bool
    fr_mismatch: int

    lift_ref_percentile: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # -- invariants (task book §15/§16): production asserts ------------------
    def validate(self) -> None:
        if abs(self.dR - (self.cf_reward - self.anchor_reward)) > 1e-6:
            raise ValueError(f"{self.edge_id}: dR != R(cf)-R(anchor)")
        if self.context == "winner_drop" and abs(self.dR + self.c_drop) > 1e-6:
            raise ValueError(f"{self.edge_id}: winner_drop requires dR == -c_drop")
        if self.context == "loser_gain" and abs(self.dR - self.c_gain) > 1e-6:
            raise ValueError(f"{self.edge_id}: loser_gain requires dR == +c_gain")
        if self.invalid:
            raise ValueError(f"{self.edge_id}: invalid endpoint")
        if self.fr_mismatch != 0:
            raise ValueError(f"{self.edge_id}: FR mismatch {self.fr_mismatch}")
        expected = "cf" if self.dR > TOL else ("anchor" if self.dR < -TOL else "skip")
        if self.preferred_side != expected:
            raise ValueError(f"{self.edge_id}: preferred_side != orientation(dR)")


def edge_id_of(case_id: str, winner_id: str, loser_id: str, context: str,
               position: int) -> str:
    return f"{case_id}:{winner_id}:{loser_id}:{context}:{position}"


def coords_key_for(edge_id: str) -> str:
    import hashlib

    return hashlib.sha1(edge_id.encode()).hexdigest()[:16]


def coords_key(edge: LocalPreferenceEdge) -> str:
    return coords_key_for(edge.edge_id)
