"""Light dataclasses for the v2 comparison graph (no heavy imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    node_id: str
    case_id: str
    coords: Any
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
