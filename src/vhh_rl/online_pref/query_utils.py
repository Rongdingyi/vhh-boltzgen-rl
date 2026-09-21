"""Conditioning helpers for online preference steps (multiplicity 1: single pair)."""
from __future__ import annotations

import torch

DEVICE = "cuda"


def pair_feats(pair):
    feats = pair.conditioning["feats"]
    return feats


def network_kwargs_for_pair(pair, device=DEVICE) -> dict:
    from ..branch_distill.query_fit import network_kwargs as _network_kwargs

    return _network_kwargs(pair.conditioning, 1, device=device)
