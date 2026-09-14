"""CDR segmentation shared by the scorer, RL manifests and credit analysis.

This module must stay dependency-free: the Phase A credit scripts run without
torch.  `vhh_rl.cli.common` re-exports these names for backwards compatibility.
"""
from __future__ import annotations


def case_spec_for(case) -> dict:
    return {
        "sequence": case.full_sequence,
        "design_positions": list(case.design_positions),
        "cdr1": list(cdr_range(case, "cdr1")),
        "cdr2": list(cdr_range(case, "cdr2")),
        "cdr3": list(cdr_range(case, "cdr3")),
        "chain_id": case.chain_id,
    }


def cdr_range(case, name: str) -> tuple[int, ...]:
    """Contiguous CDR ranges derived from design positions (no re-numbering).

    The manifest's design positions come from the project's own CDR mask; the
    three regions are recovered as the contiguous blocks of the design set, in
    the same convention the guidance project used (design == cdr1+cdr2+cdr3).
    """
    design = sorted(case.design_positions)
    if not design:
        return ()
    blocks: list[list[int]] = [[design[0]]]
    for i in design[1:]:
        if i == blocks[-1][-1] + 1:
            blocks[-1].append(i)
        else:
            blocks.append([i])
    mapping = {"cdr1": 0, "cdr2": 1, "cdr3": 2}
    if len(blocks) != 3:
        # fall back to equal thirds rather than inventing a numbering scheme;
        # recorded in the manifest's design_mask_source
        k, m = divmod(len(design), 3)
        sizes = [k + (1 if i < m else 0) for i in range(3)]
        blocks = []
        start = 0
        for size in sizes:
            blocks.append(design[start : start + size])
            start += size
    return tuple(blocks[mapping[name]])
