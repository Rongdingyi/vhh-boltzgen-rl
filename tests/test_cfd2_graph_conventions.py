"""CF-DPO v2 graph-convention regression tests (reviewer-found bugs #1/#2).

Every edge must satisfy dR == R(b) - R(a) (the trainer computes
z = (H(b)-H(a))/tau), and node ids must be unique (no silent overwrite).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_dpo_v2.types import (  # noqa: E402
    Edge, Graph, Node, check_edge_consistency,
)

GRAPH_FILE = ROOT / "runs/cf_dpo_v2/graph/pilot_graph.pt"


def _node(nid, reward):
    return Node(nid, "case", torch.zeros(2, 3), "AA", reward, "pool")


def test_add_node_rejects_duplicates():
    g = Graph()
    g.add_node(_node("n1", 0.0))
    with pytest.raises(ValueError, match="duplicate"):
        g.add_node(_node("n1", 1.0))


def test_built_graph_edge_convention():
    if not GRAPH_FILE.is_file():
        pytest.skip("comparison graph not built in this checkout")
    payload = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    nodes, edges = payload["nodes"], payload["edges"]
    assert nodes, "empty graph"
    bad = []
    for e in edges:
        ra = nodes[e["a"]]["reward"]
        rb = nodes[e["b"]]["reward"]
        if ra is None or rb is None:
            continue
        if abs((rb - ra) - e["dR"]) > 1e-6:
            bad.append((e["kind"], e["a"], e["b"], e["dR"], rb - ra))
    assert not bad, f"{len(bad)} edges violate dR = R(b)-R(a): {bad[:3]}"


def test_built_graph_node_ids_unique_by_construction():
    if not GRAPH_FILE.is_file():
        pytest.skip("comparison graph not built in this checkout")
    payload = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    ids = list(payload["nodes"])
    assert len(ids) == len(set(ids))
    # counterfactual node ids must embed both partner sample ids
    lifted = [k for k, v in payload["nodes"].items() if v["role"] == "lifted"]
    assert lifted, "no lifted nodes"
    for k in lifted:
        parts = k.split(":")
        assert len(parts) >= 5, k


def test_edge_consistency_checker_rejects_inconsistent_edge():
    g = Graph()
    g.add_node(_node("a", 0.0))
    g.add_node(_node("b", 1.0))
    g.edges.append(Edge("a", "b", -5.0, "drop"))   # wrong sign
    with pytest.raises(ValueError, match="violates dR"):
        check_edge_consistency(g)
    g.edges[0] = Edge("a", "b", 1.0, "drop")
    check_edge_consistency(g)                       # ok
