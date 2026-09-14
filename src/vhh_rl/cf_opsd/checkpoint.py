"""Checkpoint helpers for CF-OPSD (delegates to the validated native writer)."""
from __future__ import annotations

from pathlib import Path

from ..native_atom14.checkpoint import (  # noqa: F401
    load_base_model,
    make_policy_reference,
    parameter_drift,
    save_native_checkpoint,
    trainable_score_params,
)


def save_student(base_checkpoint: str | Path, student, path: str | Path,
                 metadata: dict) -> str:
    return save_native_checkpoint(base_checkpoint, student, Path(path), metadata)
