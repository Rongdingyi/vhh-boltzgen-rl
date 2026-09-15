"""Edge sampling for the v2 comparison graph (disjoint branches)."""
from __future__ import annotations

import random


def choose_edge(edges: list[dict], variant: str, same_seq_ratio: float,
                rng: random.Random) -> dict:
    """Sample one edge.

    Branch probabilities (v2): same-seq ``same_seq_ratio``; the remaining mass
    splits 75/25 between preference-local (drop/gain) and global edges.
    ``signed`` never samples same-seq edges.  The local pool used for the
    preference branch contains *only* drop/gain edges, so same-seq edges can
    never be drawn twice.
    """
    same_seq = [e for e in edges if e["kind"] == "same_seq"]
    pref_local = [e for e in edges if e["kind"] in ("drop", "gain")]
    globals_ = [e for e in edges if e["kind"] == "global"]
    # signed never samples same-seq: its mass must be renormalised away so the
    # local/global split stays 75/25 (reviewer fix)
    same_mass = same_seq_ratio if variant == "v2" else 0.0
    r = rng.random()
    if same_mass > 0 and same_seq and r < same_mass:
        return rng.choice(same_seq)
    if pref_local and r < same_mass + (1.0 - same_mass) * 0.75:
        return rng.choice(pref_local)
    return rng.choice(globals_ or edges)
