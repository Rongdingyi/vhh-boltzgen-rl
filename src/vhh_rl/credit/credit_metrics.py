"""Credit math + sparsity metrics (task book §14-§15)."""
from __future__ import annotations

import math
import statistics as st
from typing import Sequence


def residue_credits(r_win: float, r_lose: float, c_drop: float, c_gain: float) -> dict:
    """Aggregate the two interventions for one differing residue."""
    c_avg = 0.5 * (c_drop + c_gain)
    c_cons = min(c_drop, c_gain) if (c_drop > 0 and c_gain > 0) else 0.0
    return {
        "c_drop": c_drop,
        "c_gain": c_gain,
        "c_avg": c_avg,
        "c_cons": c_cons,
        "disagreement": abs(c_drop - c_gain),
        "sign_agree": (c_drop > 0 and c_gain > 0) or (c_drop < 0 and c_gain < 0)
                     or (c_drop == 0 and c_gain == 0),
    }


def topk_mass(credits: Sequence[float], n_diff: int,
              fractions=(0.10, 0.20, 0.30, 0.50)) -> dict:
    """Fraction of total positive credit captured by the top-k of n_diff residues."""
    pos_total = sum(c for c in credits if c > 0)
    out = {}
    if pos_total <= 0 or n_diff <= 0:
        return {f"top{int(f*100)}": 0.0 for f in fractions}
    ordered = sorted(credits, reverse=True)
    for f in fractions:
        k = max(1, int(math.ceil(f * n_diff)))
        out[f"top{int(f*100)}"] = sum(ordered[:k]) / pos_total
    return out


def effective_n(credits: Sequence[float]) -> float:
    num = sum(credits) ** 2
    den = sum(c * c for c in credits) + 1e-12
    return num / den


def sparsity_metrics(per_residue: Sequence[Sequence[float]], n_diff: Sequence[int]) -> dict:
    """per_residue: per-pair list of c_cons values (all differing residues)."""
    top30, neff_ratio, pos_frac = [], [], []
    for credits, nd in zip(per_residue, n_diff):
        mass = topk_mass(credits, nd)
        top30.append(mass.get("top30", 0.0))
        neff_ratio.append(effective_n(credits) / max(nd, 1))
        if nd:
            pos_frac.append(sum(1 for c in credits if c > 0) / nd)
    return {
        "top30_mass_mean": st.mean(top30) if top30 else None,
        "top30_mass_median": st.median(top30) if top30 else None,
        "neff_ratio_mean": st.mean(neff_ratio) if neff_ratio else None,
        "neff_ratio_median": st.median(neff_ratio) if neff_ratio else None,
        "positive_fraction_mean": st.mean(pos_frac) if pos_frac else None,
    }


def pair_summary(entries: Sequence[dict]) -> dict:
    """Aggregate a pair's per-residue credit entries."""
    n_diff = len(entries)
    pos = [e for e in entries if e["c_cons"] > 0]
    sign_agree = sum(1 for e in entries if e.get("sign_agree"))
    return {
        "n_diff": n_diff,
        "n_positive_consistent": len(pos),
        "positive_fraction": len(pos) / n_diff if n_diff else 0.0,
        "sign_agreement_rate": sign_agree / n_diff if n_diff else 0.0,
        "credit_abs_sum": sum(e["disagreement"] for e in entries),
        "credit_positive_sum": sum(e["c_cons"] for e in entries),
        "c_avg_sum": sum(e["c_avg"] for e in entries),
        "c_cons_sum": sum(e["c_cons"] for e in entries),
    }
