"""Counterfactual sequence construction (task book §9, §12, §17).

Single-residue:
  winner-drop: winner with loser's residue at one differing position
  loser-gain:  loser with winner's residue at one differing position
Region:
  winner with a whole CDRk replaced by loser's CDRk (drop)
  loser  with a whole CDRk replaced by winner's CDRk (gain)

Every CF sequence is hard-checked: same length, FR unchanged, only the target
position(s) changed, all AA canonical.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def _check(base: str, cf: str, changed: Sequence[int], fixed: Sequence[int]) -> None:
    if len(cf) != len(base):
        raise ValueError("counterfactual length changed")
    if set(cf) - AA20:
        raise ValueError("counterfactual contains non-canonical residue")
    changed_set = set(changed)
    for i in fixed:
        if cf[i] != base[i]:
            raise ValueError(f"counterfactual changed frozen position {i}")
    diff = [i for i in range(len(base)) if cf[i] != base[i]]
    if sorted(diff) != sorted(changed_set):
        raise ValueError(f"counterfactual changed unexpected positions: {diff[:8]}")
    if len(changed) == 1 and base[changed[0]] == cf[changed[0]]:
        raise ValueError("counterfactual is a no-op")


def build_single_residue_cfs(pair: dict, sequences: dict[str, str],
                             design_positions: dict[str, tuple[int, ...]],
                             fixed_positions: dict[str, tuple[int, ...]]) -> list[dict]:
    case = pair["case_id"]
    w = sequences[pair["winner_sample_id"]]
    l = sequences[pair["loser_sample_id"]]
    design = design_positions[case]
    fixed = fixed_positions[case]
    out: list[dict] = []
    for i in design:
        if w[i] == l[i]:
            continue
        w_drop = w[:i] + l[i] + w[i + 1:]
        l_gain = l[:i] + w[i] + l[i + 1:]
        _check(w, w_drop, [i], fixed)
        _check(l, l_gain, [i], fixed)
        out.append({
            "pair_id": _pair_id(pair), "case_id": case, "kind": "winner_drop",
            "position": i, "region": None, "sequence": w_drop, "sequence_sha1": sha1(w_drop),
        })
        out.append({
            "pair_id": _pair_id(pair), "case_id": case, "kind": "loser_gain",
            "position": i, "region": None, "sequence": l_gain, "sequence_sha1": sha1(l_gain),
        })
    return out


def build_region_cfs(pair: dict, sequences: dict[str, str], regions: dict[str, list[int]],
                     fixed_positions: dict[str, tuple[int, ...]]) -> list[dict]:
    case = pair["case_id"]
    w = sequences[pair["winner_sample_id"]]
    l = sequences[pair["loser_sample_id"]]
    fixed = fixed_positions[case]
    out: list[dict] = []
    for region, positions in regions.items():
        pos = list(positions)
        changed = [i for i in pos if w[i] != l[i]]
        if not changed:
            continue
        w_chars = list(w)
        l_chars = list(l)
        for i in pos:
            w_chars[i] = l[i]
            l_chars[i] = w[i]
        w_drop = "".join(w_chars)
        l_gain = "".join(l_chars)
        _check(w, w_drop, changed, fixed)
        _check(l, l_gain, changed, fixed)
        out.append({
            "pair_id": _pair_id(pair), "case_id": case, "kind": "region_drop",
            "position": None, "region": region, "sequence": w_drop, "sequence_sha1": sha1(w_drop),
        })
        out.append({
            "pair_id": _pair_id(pair), "case_id": case, "kind": "region_gain",
            "position": None, "region": region, "sequence": l_gain, "sequence_sha1": sha1(l_gain),
        })
    return out


def _pair_id(pair: dict) -> str:
    return f"{pair['case_id']}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
