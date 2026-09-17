"""Teacher / peer selection from one BranchGroup (task book §28, §43)."""
from __future__ import annotations


def valid_siblings(siblings) -> list:
    """Valid = decodable, FR-clean, scored (§28)."""
    return [s for s in siblings
            if not s.contains_invalid and s.fr_mismatch == 0 and s.reward is not None]


def rank_valid(siblings) -> list:
    """Valid siblings sorted by reward descending (teacher = rank 1)."""
    return sorted(valid_siblings(siblings), key=lambda s: s.reward, reverse=True)


def peer_of(ranked: list, peer_type: str):
    if peer_type not in ("median", "worst"):
        raise ValueError(f"peer_type must be median|worst, got {peer_type}")
    if peer_type == "worst":
        return ranked[-1]
    rewards = sorted(s.reward for s in ranked)
    import statistics as st
    target = st.median(rewards)
    return min(ranked, key=lambda s: abs(s.reward - target))


def select_teacher_peer(siblings, peer_type: str,
                        changed_positions_fn) -> dict | None:
    """Return the §28 selection for one group, or None when ineligible."""
    ranked = rank_valid(siblings)
    if len(ranked) < 2:
        return None
    teacher = ranked[0]
    peer = peer_of(ranked, peer_type)
    if teacher.branch_index == peer.branch_index:
        return None
    changed = tuple(changed_positions_fn(teacher.endpoint_sequence,
                                         peer.endpoint_sequence))
    gap = float(teacher.reward - peer.reward)
    return {
        "teacher_index": teacher.branch_index,
        "peer_index": peer.branch_index,
        "teacher_reward": float(teacher.reward),
        "peer_reward": float(peer.reward),
        "reward_gap": gap,
        "teacher_sequence": teacher.endpoint_sequence,
        "peer_sequence": peer.endpoint_sequence,
        "changed_positions": changed,
        "peer_type": peer_type,
    }


def eligible(selection: dict | None, min_reward_gap: float = 0.30) -> bool:
    """§43 eligibility: reward gap and >=1 changed position (no credit input)."""
    if selection is None:
        return False
    return (selection["reward_gap"] >= min_reward_gap
            and len(selection["changed_positions"]) >= 1)
