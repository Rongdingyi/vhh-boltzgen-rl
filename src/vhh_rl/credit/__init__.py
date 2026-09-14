"""Counterfactual credit assignment for native atom14 DPO (next-stage task book).

Modules:
  pair_analysis     - pair Hamming / reward-gap statistics (Phase A0)
  cdr_regions       - CDR region mapping reused from the project scorer convention
  counterfactual    - single-residue / region counterfactual sequence build (A1/A2)
  credit_metrics    - c_drop/c_gain/c_avg/c_cons, sparsity, n_eff (A1)
  credit_cache      - scorer-side helpers for counterfactual scoring
"""
from .pair_analysis import hamming_stats
from .cdr_regions import region_map
from .counterfactual import build_single_residue_cfs, build_region_cfs
from .credit_metrics import (
    residue_credits, pair_summary, sparsity_metrics, topk_mass, effective_n,
)

__all__ = [
    "hamming_stats", "region_map",
    "build_single_residue_cfs", "build_region_cfs",
    "residue_credits", "pair_summary", "sparsity_metrics", "topk_mass", "effective_n",
]
