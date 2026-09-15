"""CF-DPO v2 experiment 4 data: comparison graph over legal geometries.

Nodes are legal atom14 structures (pool samples, lifted counterfactual
structures, decode-preserving same-sequence realizations).  Edges carry exact
classifier reward differences:

  global    : (loser, winner)               dR = R(w) - R(l)
  drop      : (winner, winner+loser@i)      dR = R(w) - R(cf)
  gain      : (loser, loser+winner@i)       dR = R(cf) - R(l)
  same_seq  : (X_S^g1, X_S^g2)              dR = 0

Everything is detached data; no model is trained here.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..data.case import RLCase
from .geometry_lift import build_local_lift, build_same_sequence_realizations

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
CF = ROOT / "runs/next_stage/counterfactual"


@dataclass
class Node:
    node_id: str
    case_id: str
    coords: torch.Tensor
    sequence: str
    reward: float | None
    role: str  # pool | lifted | realization


@dataclass
class Edge:
    a: str
    b: str
    dR: float
    kind: str


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate comparison-graph node id: {node.node_id}")
        self.nodes[node.node_id] = node


def classify(drop: float, gain: float, tol: float = 0.05) -> str:
    if drop > tol and gain > tol:
        return "both_positive"
    if drop < -tol and gain < -tol:
        return "both_negative"
    if drop * gain < 0:
        return "sign_flip"
    return "small"


def load_cases() -> dict[str, RLCase]:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split="train",
            seed_base=int(row.get("seed_base", 0)),
        )
    return cases


def load_pool(case_id: str) -> dict[str, dict]:
    out = {}
    for line in (POOL / "train" / case_id / "metadata.jsonl").open():
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


def load_cf_rewards() -> dict[str, float]:
    out = {}
    path = CF / "counterfactual_scores.jsonl"
    for line in path.open():
        r = json.loads(line)
        out[r["cf_id"]] = float(r["score"])
    return out


def build_graph(case_ids: list[str], *, sites_per_pair: int = 3,
                same_seq_m: int = 2, seed: int = 0) -> tuple[Graph, dict]:
    rng = random.Random(seed)
    cases = load_cases()
    credit: dict[str, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    for row in csv.DictReader((CF / "residue_credit.csv").open()):
        credit[row["pair_id"]][int(row["position"])] = (
            float(row["c_drop"]), float(row["c_gain"]), float(row["c_cons"]))
    cf_rewards = load_cf_rewards()
    pairs = [json.loads(l) for l in (POOL / "pairs_train.jsonl").open()]
    graph = Graph()
    stats = {"n_lift_attempts": 0, "n_lift_ok": 0,
             "n_same_seq_attempts": 0, "n_same_seq_ok": 0,
             "classes": defaultdict(int)}
    feats_cache: dict[str, dict] = {}
    for pair in pairs:
        cid = pair["case_id"]
        if cid not in case_ids:
            continue
        case = cases[cid]
        if cid not in feats_cache:
            feats_cache[cid] = torch.load(POOL / "train" / cid / "feats_common.pt",
                                          map_location="cpu", weights_only=False)
        feats = feats_cache[cid]
        pool = load_pool(cid)
        w_id, l_id = pair["winner_sample_id"], pair["loser_sample_id"]
        if w_id not in pool or l_id not in pool:
            continue
        win, lose = pool[w_id], pool[l_id]
        graph.add_node(Node(f"{cid}:{w_id}", cid, win["coords"], win["sequence"],
                            win["reward"], "pool"))
        graph.add_node(Node(f"{cid}:{l_id}", cid, lose["coords"], lose["sequence"],
                            lose["reward"], "pool"))
        graph.edges.append(Edge(f"{cid}:{l_id}", f"{cid}:{w_id}",
                                win["reward"] - lose["reward"], "global"))
        pid = f"{cid}:{w_id}:{l_id}"
        by_class: dict[str, list[int]] = defaultdict(list)
        for pos, (d, g, _c) in credit.get(pid, {}).items():
            by_class[classify(d, g)].append(pos)
        sites: list[int] = []
        for cls in ("sign_flip", "both_negative", "both_positive"):
            if by_class[cls]:
                sites.append(by_class[cls][0])
        for cls in ("sign_flip", "both_negative", "both_positive"):
            for pos in by_class[cls][1:]:
                if len(sites) >= sites_per_pair:
                    break
                sites.append(pos)
        for pos in sites[:sites_per_pair]:
            cls = classify(*credit[pid][pos][:2])
            for direction, acceptor, donor, base_node in (
                ("winner_drop", win, lose, f"{cid}:{w_id}"),
                ("loser_gain", lose, win, f"{cid}:{l_id}"),
            ):
                expected = (win["sequence"][:pos] + lose["sequence"][pos]
                            + win["sequence"][pos + 1:]
                            if direction == "winner_drop" else
                            lose["sequence"][:pos] + win["sequence"][pos]
                            + lose["sequence"][pos + 1:])
                other = lose["sequence"] if direction == "winner_drop" else win["sequence"]
                att = build_local_lift(acceptor["coords"], donor["coords"], feats, pos,
                                       case.full_sequence, case.fr_positions,
                                       expected, other, acceptor["sequence"])
                stats["n_lift_attempts"] += 1
                if not att.ok or att.coords is None:
                    continue
                stats["n_lift_ok"] += 1
                stats["classes"][cls] += 1
                # node ids must be unique across pairs of the same case
                node_id = f"{cid}:{w_id}:{l_id}:{direction}:{pos}"
                cf_id = (f"{pid}:{'winner_drop' if direction == 'winner_drop' else 'loser_gain'}"
                         f":{pos}")
                reward = cf_rewards.get(cf_id, cf_rewards.get(f"{pid}:{direction}:{pos}"))
                graph.add_node(Node(node_id, cid, att.coords, expected, reward, "lifted"))
                # edge convention everywhere: a -> b, dR = R(b) - R(a)
                if direction == "winner_drop":      # a=winner, b=CF
                    dR = (reward - win["reward"]) if reward is not None else None
                else:                               # a=loser, b=CF
                    dR = (reward - lose["reward"]) if reward is not None else None
                if dR is not None:
                    graph.edges.append(Edge(base_node, node_id, dR, "drop"
                                            if direction == "winner_drop" else "gain"))
                # same-sequence realizations for this lifted node
                stats["n_same_seq_attempts"] += 1
                reals = build_same_sequence_realizations(
                    att.coords, feats, expected, case.fr_positions,
                    case.full_sequence, n_realizations=same_seq_m, sigma=0.08,
                    max_tries=40, seed=rng.randrange(10_000))
                if len(reals) >= 2:
                    stats["n_same_seq_ok"] += 1
                    for k, coords in enumerate(reals[:2]):
                        rid = f"{node_id}:real{k}"
                        graph.add_node(Node(rid, cid, coords, expected, reward, "realization"))
                    graph.edges.append(Edge(f"{node_id}:real0", f"{node_id}:real1", 0.0,
                                            "same_seq"))
    stats["classes"] = dict(stats["classes"])
    return graph, stats


def save_graph(graph: Graph, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": {k: {"node_id": n.node_id, "case_id": n.case_id,
                      "coords": n.coords, "sequence": n.sequence,
                      "reward": n.reward, "role": n.role}
                  for k, n in graph.nodes.items()},
        "edges": [{"a": e.a, "b": e.b, "dR": e.dR, "kind": e.kind}
                  for e in graph.edges],
    }
    torch.save(payload, path)
