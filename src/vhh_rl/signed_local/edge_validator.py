"""Local edge dataset validation (task book §15/§16/§83)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..data.case import RLCase
from ..native_atom14.decode import decode_atom14, fr_check, sequence_from_feat
from .types import LocalPreferenceEdge

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"


def load_edges(path: str | Path) -> list[LocalPreferenceEdge]:
    out = []
    for line in Path(path).open():
        row = json.loads(line)
        row.pop("meta", None)
        out.append(LocalPreferenceEdge(**row))
    return out


def validate_dataset(edges_path: str | Path, *, split: str = "train",
                     decode: bool = True) -> dict:
    edges = load_edges(edges_path)
    seen: set[str] = set()
    n_dR = n_dup = n_decode = n_fr = 0
    for e in edges:
        e.validate()  # raises on any invariant violation
        if e.edge_id in seen:
            n_dup += 1
            raise ValueError(f"duplicate edge id {e.edge_id}")
        seen.add(e.edge_id)
        if abs(e.dR - (e.cf_reward - e.anchor_reward)) < 1e-6:
            n_dR += 1
        if decode:
            feats = torch.load(POOL / split / e.case_id / "feats_common.pt",
                               map_location="cpu", weights_only=False)
            ref = _case_sequence(e.case_id)
            fr_pos = _case_fr(e.case_id)
            for coords_path, expected in ((e.cf_coords_path, e.cf_sequence),
                                          (e.anchor_coords_path, e.anchor_sequence)):
                seq = _decode(coords_path, feats)
                if seq == expected:
                    n_decode += 1
                fr = fr_check(seq, ref, fr_pos)
                if fr["fr_mismatch_count"] == 0:
                    n_fr += 1
    return {
        "n_edges": len(edges),
        "unique_edge_ids": len(seen),
        "duplicate_edge_ids": n_dup,
        "dR_invariant_ok": n_dR,
        "decode_ok": n_decode,
        "decode_checked": 2 * len(edges) if decode else 0,
        "fr_mismatch_total": (2 * len(edges) - n_fr) if decode else 0,
        "classes": _counts(edges, "event_class"),
        "contexts": _counts(edges, "context"),
        "preferred": _counts(edges, "preferred_side"),
    }


def _counts(edges, attr: str) -> dict:
    out: dict[str, int] = {}
    for e in edges:
        key = str(getattr(e, attr))
        out[key] = out.get(key, 0) + 1
    return out


def _decode(coords_path: str, feats: dict) -> str:
    feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats.items()}
    feat["coords"] = torch.load(coords_path, map_location="cpu", weights_only=True).float()
    out = decode_atom14(feat)
    return sequence_from_feat(out)[0]


_CASE_CACHE: dict[str, tuple[str, tuple[int, ...]]] = {}


def _case_meta(case_id: str) -> tuple[str, tuple[int, ...]]:
    if case_id not in _CASE_CACHE:
        for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
            row = json.loads(line)
            if row["case_id"] == case_id:
                _CASE_CACHE[case_id] = (row["full_sequence"], tuple(row["fr_positions"]))
                break
    return _CASE_CACHE[case_id]


def _case_sequence(case_id: str) -> str:
    return _case_meta(case_id)[0]


def _case_fr(case_id: str) -> tuple[int, ...]:
    return _case_meta(case_id)[1]
