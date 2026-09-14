"""Official atom14 -> sequence readout wrapper (task book §4).

The ONLY decoder is the official ``boltzgen.data.feature.featurizer.res_from_atom14``.
This module adds: call protocol, sequence extraction, invalid (UNK) accounting,
and FR hard-check helpers.  It never reimplements the geometric decoding.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_BOLTZGEN_SRC = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src")
if str(_BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(_BOLTZGEN_SRC))

from boltzgen.data import const  # noqa: E402
from boltzgen.data.feature.featurizer import res_from_atom14  # noqa: E402


def decode_atom14(sample: dict[str, Any], invalid_token: str = "UNK") -> dict[str, Any]:
    """Call the official decoder on a single-sample feat dict.

    ``sample`` must contain (batch-dim removed): coords, atom_pad_mask,
    atom_to_token, token_index, design_mask, atom_resolved_mask, ref_* atom
    features (the writer assembles exactly this dict; see writer.py:266).
    Returns the updated feat dict from the official function.
    """
    return res_from_atom14(sample, invalid_token=invalid_token)


def sequence_from_feat(feat: dict[str, Any]) -> tuple[str, list[str], bool]:
    """Extract the protein one-letter sequence from a (decoded) feat dict.

    Returns (sequence, token_names, contains_invalid).
    """
    res_type = feat["res_type"]
    if res_type.dim() == 2:  # [N_tok, num_tokens]
        ids = res_type.argmax(dim=-1)
    else:
        ids = res_type
    mol_type = feat["mol_type"].reshape(-1)
    is_protein = mol_type == const.chain_type_ids["PROTEIN"]
    names = [const.tokens[int(i)] for i in ids.tolist()]
    letters = []
    invalid = False
    for name, prot in zip(names, is_protein.tolist()):
        if not prot:
            continue
        letter = const.prot_token_to_letter.get(name)
        if letter is None or letter not in "ACDEFGHIKLMNPQRSTVWY":
            invalid = True
            letters.append(letter if letter else "X")
        else:
            letters.append(letter)
    return "".join(letters), names, invalid


def fr_check(sequence: str, reference: str, fr_positions: tuple[int, ...]) -> dict[str, Any]:
    """Hard FR check (task book §17).  Never silently patched."""
    mismatches = [i for i in fr_positions if sequence[i] != reference[i]]
    return {
        "fr_mismatch_count": len(mismatches),
        "fr_positions": len(fr_positions),
        "fr_mismatch_indices": mismatches[:8],
    }


def decoded_cdr(sequence: str, design_positions: tuple[int, ...]) -> str:
    return "".join(sequence[i] for i in sorted(design_positions))
