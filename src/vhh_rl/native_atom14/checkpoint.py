"""Policy/reference checkpoint handling for native atom14 post-training.

The released design checkpoint stores raw weights (no EMA payload); the official
inference path loads ``state_dict`` directly, so policy and reference are both
initialised from the same raw weights (task book §51, documented in
NATIVE_AUDIT.md).
"""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

_BOLTZGEN_SRC = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src")
if str(_BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(_BOLTZGEN_SRC))

from boltzgen.model.models.boltz import Boltz  # noqa: E402
from boltzgen.data import const  # noqa: E402


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_override() -> dict[str, Any]:
    """The `override:` block of the official design config (design.yaml)."""
    return {
        "masker_args": {"mask": True, "mask_backbone": False},
        "validators": None,
        "step_scale_schedule": [
            {"step_scale": 1.8, "period": 0.25},
            {"step_scale": 2.0, "period": 0.25},
            {"step_scale": 1.8, "period": 0.25},
            {"step_scale": 2.0, "period": 0.25},
        ],
        "noise_scale_schedule": [
            {"noise_scale": 0.95, "period": 0.25},
            {"noise_scale": 0.88, "period": 0.25},
            {"noise_scale": 0.95, "period": 0.25},
            {"noise_scale": 0.88, "period": 0.25},
        ],
        "diffusion_process_args": {
            "sigma_min": 0.0004,
            "sigma_max": 160.0,
            "sigma_data": 16.0,
            "rho": 7,
            "P_mean": -1.2,
            "P_std": 1.5,
            "gamma_0": 0.8,
            "gamma_min": 1.0,
            "noise_scale": None,
            "step_scale": None,
            "mse_rotational_alignment": True,
            "coordinate_augmentation": True,
            "alignment_reverse_diff": True,
            "synchronize_sigmas": False,
            "sampling_schedule": "dilated",
            "time_dilation": 2.667,
            "time_dilation_start": 0.6,
            "time_dilation_end": 0.8,
        },
    }


def load_base_model(
    checkpoint: str | Path,
    device: str | torch.device = "cuda",
    predict_args: dict[str, Any] | None = None,
) -> Boltz:
    model = Boltz.load_from_checkpoint(
        str(checkpoint),
        strict=True,
        map_location="cpu",
        weights_only=False,
        predict_args=predict_args or {
            "recycling_steps": 3,
            "sampling_steps": 50,
            "diffusion_samples": 8,
        },
        **design_override(),
    )
    return model.to(device)


def make_policy_reference(base: Boltz, device: str | torch.device = "cuda") -> tuple[Boltz, Boltz]:
    """Policy = base (score_model trainable), reference = frozen deep copy."""
    policy = base
    reference = copy.deepcopy(base)
    reference.eval()
    reference.to(device)
    for param in reference.parameters():
        param.requires_grad_(False)
    # freeze everything on policy except structure_module.score_model
    for name, param in policy.named_parameters():
        param.requires_grad_(name.startswith("structure_module.score_model."))
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith("structure_module.score_model.") for n in trainable):
        raise RuntimeError("policy trainable set is not exactly structure_module.score_model")
    return policy, reference


def trainable_score_params(policy: Boltz) -> list[torch.nn.Parameter]:
    return [p for n, p in policy.named_parameters()
            if n.startswith("structure_module.score_model.") and p.requires_grad]


def parameter_drift(policy: Boltz, reference: Boltz) -> dict[str, float]:
    drift: dict[str, float] = {}
    total_sq = 0.0
    for name, param in policy.named_parameters():
        ref = dict(reference.named_parameters()).get(name)
        if ref is None:
            continue
        diff = (param.detach() - ref.detach()).float()
        sq = float(diff.pow(2).sum().item())
        total_sq += sq
        if name.startswith("structure_module.score_model.atom_attention_encoder"):
            key = "atom_attention_encoder"
        elif name.startswith("structure_module.score_model.token_transformer"):
            key = "token_transformer"
        elif name.startswith("structure_module.score_model.atom_attention_decoder"):
            key = "atom_attention_decoder"
        elif name.startswith("structure_module.score_model."):
            key = "score_model_other"
        else:
            key = "outside_score_model"
        drift[key] = drift.get(key, 0.0) + sq
    drift = {k: v ** 0.5 for k, v in drift.items()}
    drift["total"] = total_sq ** 0.5
    return drift


def save_native_checkpoint(
    base_checkpoint: str | Path,
    policy: Boltz,
    out_path: str | Path,
    metadata: dict[str, Any],
) -> str:
    """Write a loadable checkpoint: original dict with score_model weights replaced."""
    base_checkpoint = Path(base_checkpoint)
    payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    replaced = 0
    for name, tensor in policy.state_dict().items():
        if name.startswith("structure_module.score_model.") and name in state:
            state[name] = tensor.detach().cpu()
            replaced += 1
    if replaced == 0:
        raise RuntimeError("no score_model weights were written into the checkpoint")
    payload["native_posttrain"] = {
        **metadata,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "n_replaced_tensors": replaced,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return sha256_file(out_path)
