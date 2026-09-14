"""Phase B: target decoding + metrics + Gate B aggregation (task book §29-§32)."""
from __future__ import annotations

import statistics as st

import torch

from ..native_atom14.decode import decode_atom14, fr_check, sequence_from_feat


def decode_and_score(target_coords: torch.Tensor, feats: dict, reference_sequence: str,
                     fr_positions, scorer, case, endpoint_reward: float,
                     anchor_reward: float, design_positions) -> dict:
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = target_coords.clone()
    out = decode_atom14(feat)
    seq, _tokens, invalid = sequence_from_feat(out)
    fr = fr_check(seq, reference_sequence, tuple(fr_positions))
    reward = None
    if not invalid and fr["fr_mismatch_count"] == 0:
        from ..cli.common import case_spec_for
        batch = scorer.score_sequences([seq], spec=case_spec_for(case))
        reward = float(batch.raw_scores[0])
    return {
        "target_sequence": seq,
        "target_invalid": bool(invalid),
        "target_fr_mismatch": fr["fr_mismatch_count"],
        "target_reward": reward,
        "target_minus_anchor": (reward - anchor_reward) if reward is not None else None,
        "endpoint_gap": endpoint_reward,  # winner - loser for reference
    }


def credit_match_rate(target_sequence: str, winner_sequence: str,
                      credited_positions: list[int]) -> float | None:
    if not credited_positions:
        return None
    return sum(1 for p in credited_positions
               if target_sequence[p] == winner_sequence[p]) / len(credited_positions)


def third_aa_rate(target_sequence: str, winner_sequence: str, loser_sequence: str,
                  anchor_sequence: str, differing_positions: list[int]) -> float | None:
    if not differing_positions:
        return None
    third = 0
    for p in differing_positions:
        aa = target_sequence[p]
        if aa not in {winner_sequence[p], loser_sequence[p], anchor_sequence[p]}:
            third += 1
    return third / len(differing_positions)


def gate_b(summary_rows: list[dict]) -> dict:
    """Rows: one per radius (aggregated over pairs)."""
    out = {}
    for row in summary_rows:
        ok = (row["invalid_rate"] <= 0.05
              and row["fr_mismatch_total"] == 0
              and (row["median_target_minus_anchor"] or 0) > 0
              and (row["median_credit_match"] or 0) >= 0.50
              and row["positive_cases"] >= 3)
        out[row["radius"]] = ok
    return out
