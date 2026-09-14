"""Common runtime assembly shared by the CLI commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..adapters.boltzgen_if import BoltzGenIFAdapter
from ..config import RLConfig
from ..data.case import read_manifest
from ..data.cdr import case_spec_for, cdr_range  # noqa: F401  (re-export)


def build_adapter(cfg: RLConfig) -> BoltzGenIFAdapter:
    return BoltzGenIFAdapter(cfg.boltzgen_root, cfg.boltzgen_checkpoint, device=cfg.device)


def ensure_slurm() -> None:
    if not (__import__("os").environ.get("SLURM_JOB_ID") or __import__("os").environ.get("VHH_RL_ALLOW_LOGIN")):
        raise SystemExit(
            "refusing to run compute on the login node; submit through Slurm "
            "(or set VHH_RL_ALLOW_LOGIN=1 for a short smoke)"
        )


def manifest_cases(cfg: RLConfig, splits=("train",)):
    splits = tuple(set(splits) | {"train_debug"})
    return read_manifest(cfg.manifest, splits=splits)


def capture_cases(adapter: BoltzGenIFAdapter, cfg: RLConfig, cases, workdir: Path):
    captured = []
    for case in cases:
        spec_path = _ensure_spec(cfg, case, workdir)
        run_dir = case.structure_path.parent / "boltzgen_run"
        if not run_dir.is_dir():
            # some design runs wrote their outputs directly into the replicate dir
            run_dir = case.structure_path.parent
        captured.append(
            adapter.capture_case(spec_path, case.case_id, workdir / case.case_id, run_dir=run_dir)
        )
    return captured


def _ensure_spec(cfg: RLConfig, case, workdir: Path) -> Path:
    """The inverse-fold step consumes the case's OWN design.yaml (which sits
    next to the canonical backbone in the design-run replicate directory, see
    vhh_esmc_guidance m3v2_ifold_gate.py:524).  Reusing it verbatim keeps
    featurisation identical to every previous inference on this backbone; we
    never regenerate the spec here (the CDR grammar must not be reinvented).
    """
    spec_path = case.structure_path.parent / "design.yaml"
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"{case.case_id}: no design.yaml next to {case.structure_path}; "
            "the canonical backbone must come from a design run directory"
        )
    return spec_path


def log_factory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    return log
