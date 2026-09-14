"""CDR region mapping: reuse the project's own scorer convention.

The scorer (and the RL manifest) derive cdr1/2/3 as the contiguous blocks of
the design-position set; this module re-exports that mapping so the credit
analysis can never drift from the classifier's segmentation (task book §13).
"""
from __future__ import annotations


def region_map(case) -> dict[str, list[int]]:
    """case: RLCase. Returns {"cdr1": [...], "cdr2": [...], "cdr3": [...]} 0-based.

    Uses the project's own segmentation (vhh_rl.data.cdr, re-exported by
    cli.common) so credit regions can never drift from the scorer convention.
    """
    from vhh_rl.data.cdr import cdr_range

    return {
        "cdr1": list(cdr_range(case, "cdr1")),
        "cdr2": list(cdr_range(case, "cdr2")),
        "cdr3": list(cdr_range(case, "cdr3")),
    }
