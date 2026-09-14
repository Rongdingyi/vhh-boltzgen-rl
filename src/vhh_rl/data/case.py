from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

AA20 = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class RLCase:
    case_id: str
    structure_path: Path
    chain_id: str
    full_sequence: str
    design_positions: tuple[int, ...]
    fr_positions: tuple[int, ...]
    split: str
    seed_base: int
    design_mask_source: str = "taskbook_project_mask"

    def validate(self) -> None:
        if set(self.full_sequence) - set(AA20):
            raise ValueError(f"{self.case_id}: non-AA20 symbols in full_sequence")
        n = len(self.full_sequence)
        design = tuple(sorted(self.design_positions))
        if any(i < 0 or i >= n for i in design):
            raise ValueError(f"{self.case_id}: design position out of range")
        fr = tuple(i for i in range(n) if i not in set(design))
        if self.fr_positions != fr:
            raise ValueError(f"{self.case_id}: fr_positions disagree with design_positions")


def read_manifest(path: str | Path, splits: Sequence[str] = ("train",)) -> list[RLCase]:
    cases: list[RLCase] = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] not in splits:
                continue
            cases.append(
                RLCase(
                    case_id=row["case_id"],
                    structure_path=Path(row["structure_path"]),
                    chain_id=row.get("chain_id", "A"),
                    full_sequence=row["full_sequence"].strip().upper(),
                    design_positions=tuple(sorted(int(i) for i in row["design_positions"])),
                    fr_positions=tuple(sorted(int(i) for i in row["fr_positions"])),
                    split=row["split"],
                    seed_base=int(row.get("seed_base", 0)),
                    design_mask_source=row.get("design_mask_source", ""),
                )
            )
    for case in cases:
        case.validate()
    return cases
