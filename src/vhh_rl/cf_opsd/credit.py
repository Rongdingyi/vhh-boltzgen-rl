"""Phase B: counterfactual residue credit for a winner/loser sequence pair.

Reuses the validated CF-DPO implementation (task book §19: no re-write with a
different convention).
"""
from __future__ import annotations

from pathlib import Path

from ..credit.counterfactual import build_single_residue_cfs


def pair_credit(scorer, case, winner_sample_id: str, loser_sample_id: str,
                winner_sequence: str, loser_sequence: str,
                design_positions, fr_positions) -> dict[int, float]:
    """Bidirectional single-residue CF credit (c_cons) for one pair."""
    from ..cli.common import case_spec_for

    pair = {"case_id": case.case_id, "winner_sample_id": winner_sample_id,
            "loser_sample_id": loser_sample_id}
    seqs = {winner_sample_id: winner_sequence, loser_sample_id: loser_sequence}
    design = {case.case_id: tuple(design_positions)}
    fixed = {case.case_id: tuple(fr_positions)}
    rows = build_single_residue_cfs(pair, seqs, design, fixed)
    pid = f"{case.case_id}:{winner_sample_id}:{loser_sample_id}"

    originals = [winner_sequence, loser_sequence]
    cf_seqs = [r["sequence"] for r in rows]
    batch = scorer.score_sequences(originals + cf_seqs, spec=case_spec_for(case))
    scores = batch.raw_scores.tolist()
    r_win, r_lose = float(scores[0]), float(scores[1])
    score_of = {}
    for row, s in zip(rows, scores[2:]):
        score_of[(row["kind"], row["position"])] = float(s)

    credits: dict[int, float] = {}
    detail: list[dict] = []
    for pos in sorted({r["position"] for r in rows}):
        c_drop = r_win - score_of[("winner_drop", pos)]
        c_gain = score_of[("loser_gain", pos)] - r_lose
        c_cons = min(c_drop, c_gain) if (c_drop > 0 and c_gain > 0) else 0.0
        credits[pos] = c_cons
        detail.append({"position": pos, "c_drop": c_drop, "c_gain": c_gain,
                       "c_cons": c_cons})
    return {
        "pair_id": pid,
        "winner_reward": r_win,
        "loser_reward": r_lose,
        "gap": r_win - r_lose,
        "credits": credits,
        "detail": detail,
        "n_queries": 2 + 2 * len(credits),
    }
