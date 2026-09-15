"""Local correction edge dataset builder (task book §17-§19).

One record = one context-specific local preference pair:

    winner_drop : anchor = winner pool structure, cf = winner with loser residue@i
    loser_gain  : anchor = loser  pool structure, cf = loser  with winner residue@i

Only ``both_negative`` and ``sign_flip`` sites are kept for the correction set
(§8); all lift attempts are written to ``lift_attempts.csv`` (§18).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from ..cf_dpo_v2.geometry_lift import build_local_lift
from ..data.case import RLCase
from .types import (
    TOL, LocalPreferenceEdge, classify_event, coords_key_for, edge_id_of,
    preferred_side, robust_sign,
)

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
CF_DIR = ROOT / "runs/next_stage/counterfactual"


def load_credit(credit_csv: Path | None = None) -> dict[str, dict[int, tuple[float, float, float]]]:
    out: dict[str, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    path = credit_csv or (CF_DIR / "residue_credit.csv")
    for row in csv.DictReader(path.open()):
        out[row["pair_id"]][int(row["position"])] = (
            float(row["c_drop"]), float(row["c_gain"]), float(row["c_cons"]))
    return dict(out)


def load_cf_rewards(scores_jsonl: Path | None = None) -> dict[str, float]:
    out = {}
    path = scores_jsonl or (CF_DIR / "counterfactual_scores.jsonl")
    for line in path.open():
        r = json.loads(line)
        if r.get("score") is not None:
            out[r["cf_id"]] = float(r["score"])
    return out


def load_case_pool(split: str, case_id: str) -> dict[str, dict]:
    out = {}
    for line in (POOL / split / case_id / "metadata.jsonl").open():
        r = json.loads(line)
        if r["reward_raw"] is None or r["contains_UNK"]:
            continue
        out[r["sample_id"]] = {
            "sequence": r["decoded_sequence"],
            "coords": torch.load(r["coords_path"], map_location="cpu",
                                 weights_only=True).float(),
            "reward": float(r["reward_raw"]),
        }
    return out


def load_feats(split: str, case_id: str) -> dict:
    return torch.load(POOL / split / case_id / "feats_common.pt",
                      map_location="cpu", weights_only=False)


def build_edges(
    cases: dict[str, RLCase],
    pairs: list[dict],
    credit: dict[str, dict[int, tuple[float, float, float]]],
    cf_rewards: dict[str, float],
    *,
    out_dir: Path,
    split: str = "train",
    tol: float = TOL,
    classes: set[str] | None = None,
    contexts: tuple[str, ...] = ("winner_drop", "loser_gain"),
    max_pairs: int | None = None,
) -> dict:
    """Build and persist the local correction dataset."""
    classes = classes if classes is not None else {"both_negative", "sign_flip"}
    out_dir = Path(out_dir)
    coords_dir = out_dir / "coords"
    if coords_dir.exists():
        shutil.rmtree(coords_dir)
    coords_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = out_dir / "lift_attempts.csv"
    edges_path = out_dir / f"{split}_edges.jsonl"
    summary_path = out_dir / f"{split}_edge_summary.json"

    attempt_fields = ["edge_id", "case_id", "pair_id", "position", "context",
                      "event_class", "c_drop", "c_gain", "attempted", "accepted",
                      "reason", "decoded_sequence", "expected_sequence",
                      "invalid", "fr_mismatch", "touched_atoms", "moved_rms"]
    edges: list[LocalPreferenceEdge] = []
    seen: set[str] = set()
    n_sites = 0
    n_class = defaultdict(int)
    n_att = defaultdict(int)
    n_acc = defaultdict(int)
    with attempts_path.open("w", newline="") as af:
        aw = csv.DictWriter(af, fieldnames=attempt_fields)
        aw.writeheader()
        pool_cache: dict[str, dict] = {}
        feats_cache: dict[str, dict] = {}
        for pair in pairs[: max_pairs] if max_pairs else pairs:
            cid = pair["case_id"]
            case = cases.get(cid)
            if case is None:
                continue
            pid = f"{cid}:{pair['winner_sample_id']}:{pair['loser_sample_id']}"
            if pid not in credit:
                continue
            if cid not in pool_cache:
                pool_cache[cid] = load_case_pool(split, cid)
                feats_cache[cid] = load_feats(split, cid)
            pool, feats = pool_cache[cid], feats_cache[cid]
            win = pool.get(pair["winner_sample_id"])
            lose = pool.get(pair["loser_sample_id"])
            if win is None or lose is None:
                continue
            for pos, (c_drop, c_gain, _cons) in sorted(credit[pid].items()):
                s_drop, s_gain = robust_sign(c_drop, tol), robust_sign(c_gain, tol)
                cls = classify_event(s_drop, s_gain)
                if cls not in classes:
                    continue
                n_sites += 1
                n_class[cls] += 1
                for context in contexts:
                    if context == "winner_drop":
                        acceptor, donor = win, lose
                        expected = (win["sequence"][:pos] + lose["sequence"][pos]
                                    + win["sequence"][pos + 1:])
                        other = lose["sequence"]
                        dR = -c_drop
                        anchor_reward = win["reward"]
                    else:
                        acceptor, donor = lose, win
                        expected = (lose["sequence"][:pos] + win["sequence"][pos]
                                    + lose["sequence"][pos + 1:])
                        other = win["sequence"]
                        dR = c_gain
                        anchor_reward = lose["reward"]
                    eid = edge_id_of(cid, pair["winner_sample_id"],
                                     pair["loser_sample_id"], context, pos)
                    cf_reward = anchor_reward + dR
                    # §16 hard invariant: the analytic lift value must agree with
                    # the scorer's logged counterfactual score when available.
                    cf_key = f"{pid}:{context}:{pos}"
                    if cf_key in cf_rewards:
                        drift = abs(cf_rewards[cf_key] - cf_reward)
                        if drift > 1e-4:
                            raise ValueError(
                                f"{eid}: cf_reward disagrees with logged CF score "
                                f"by {drift:.3e}")
                    att = build_local_lift(acceptor["coords"], donor["coords"], feats,
                                           pos, case.full_sequence, case.fr_positions,
                                           expected, other, acceptor["sequence"])
                    n_att[context] += 1
                    row = {
                        "edge_id": eid, "case_id": cid, "pair_id": pid,
                        "position": pos, "context": context, "event_class": cls,
                        "c_drop": c_drop, "c_gain": c_gain, "attempted": 1,
                        "accepted": int(att.ok), "reason": att.reason,
                        "decoded_sequence": att.sequence or "",
                        "expected_sequence": expected,
                        "invalid": int(att.invalid),
                        "fr_mismatch": att.fr_mismatch,
                        "touched_atoms": att.touched_atoms,
                        "moved_rms": att.moved_rms,
                    }
                    aw.writerow(row)
                    if not att.ok or att.coords is None:
                        continue
                    assert eid not in seen, f"duplicate edge id {eid}"
                    seen.add(eid)
                    edge = LocalPreferenceEdge(
                        edge_id=eid, case_id=cid, pair_id=pid, position=pos,
                        context=context, event_class=cls,
                        anchor_sample_id=(pair["winner_sample_id"] if context == "winner_drop"
                                          else pair["loser_sample_id"]),
                        donor_sample_id=(pair["loser_sample_id"] if context == "winner_drop"
                                         else pair["winner_sample_id"]),
                        anchor_sequence=acceptor["sequence"], cf_sequence=expected,
                        anchor_reward=float(anchor_reward), cf_reward=float(cf_reward),
                        dR=float(dR), preferred_side=preferred_side(dR),
                        anchor_coords_path=str(coords_dir / f"{coords_key_for(eid)}_anchor.pt"),
                        cf_coords_path=str(coords_dir / f"{coords_key_for(eid)}_cf.pt"),
                        c_drop=float(c_drop), c_gain=float(c_gain),
                        touched_atoms=int(att.touched_atoms), moved_rms=float(att.moved_rms),
                        invalid=False, fr_mismatch=0,
                    )
                    try:
                        edge.validate()
                    except ValueError:
                        continue
                    torch.save(acceptor["coords"], Path(edge.anchor_coords_path))
                    torch.save(att.coords, Path(edge.cf_coords_path))
                    n_acc[context] += 1
                    edges.append(edge)

    with edges_path.open("w") as fh:
        for e in edges:
            fh.write(json.dumps(e.__dict__) + "\n")
    pref_counts = defaultdict(int)
    for e in edges:
        pref_counts[e.preferred_side] += 1
    summary = {
        "split": split,
        "n_pairs_used": len(pairs[: max_pairs] if max_pairs else pairs),
        "n_sites_considered": n_sites,
        "n_both_negative": n_class.get("both_negative", 0),
        "n_sign_flip": n_class.get("sign_flip", 0),
        "n_drop_attempted": n_att.get("winner_drop", 0),
        "n_drop_accepted": n_acc.get("winner_drop", 0),
        "n_gain_attempted": n_att.get("loser_gain", 0),
        "n_gain_accepted": n_acc.get("loser_gain", 0),
        "acceptance_by_context": {
            k: (n_acc.get(k, 0) / n_att[k]) if n_att.get(k) else None
            for k in ("winner_drop", "loser_gain")},
        "preferred_anchor_fraction": (pref_counts.get("anchor", 0) / len(edges)) if edges else None,
        "preferred_cf_fraction": (pref_counts.get("cf", 0) / len(edges)) if edges else None,
        "median_abs_dR": _median([abs(e.dR) for e in edges]),
        "median_moved_rms": _median([e.moved_rms for e in edges]),
        "fr_mismatch": 0,
        "invalid": 0,
        "n_edges": len(edges),
    }
    summary["acceptance_by_class"] = {
        cls: _acceptance_by_class(edges, cls, n_class) for cls in sorted(classes)}
    summary_path.write_text(json.dumps(summary, indent=1))
    return summary


def _median(xs: list[float]) -> float | None:
    import statistics as st

    return st.median(xs) if xs else None


def _acceptance_by_class(edges, cls: str, n_class) -> float | None:
    accepted = sum(1 for e in edges if e.event_class == cls)
    total = n_class.get(cls, 0)
    return (accepted / total) if total else None
