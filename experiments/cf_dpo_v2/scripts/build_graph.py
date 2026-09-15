#!/usr/bin/env python
"""Build the v2 comparison graph for the fixed pilot cases."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.cf_dpo_v2.graph_dataset import build_graph, save_graph  # noqa: E402

CASES = ["sab2_6u52_c", "sab2_7sl5_d", "sab2_7nqk_b", "sab2_6mqe_h"]
OUT = ROOT / "runs/cf_dpo_v2/graph"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites-per-pair", type=int, default=3)
    parser.add_argument("--same-seq-m", type=int, default=2)
    args = parser.parse_args()
    graph, stats = build_graph(CASES, sites_per_pair=args.sites_per_pair,
                               same_seq_m=args.same_seq_m, seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    save_graph(graph, OUT / "pilot_graph.pt")
    kinds = {}
    for e in graph.edges:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    summary = {"n_nodes": len(graph.nodes), "n_edges": len(graph.edges),
               "edge_kinds": kinds, "build_stats": stats}
    (OUT / "graph_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
