"""Uniform-weight regression gate (task book §56).

With uniform weights over ALL CDR fake residues, the residue-weighted training
loss must reproduce the N3 per-sample loss (the only difference is the
per-residue denominator epsilon, ~1e-8 relative).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.native_atom14.checkpoint import load_base_model  # noqa: E402
from vhh_rl.native_atom14.denoise_loss import (  # noqa: E402
    per_residue_denoising_loss, per_sample_denoising_loss,
)
from vhh_rl.native_atom14.dpo_trainer import (  # noqa: E402
    load_conditioning, move_conditioning,
)
from vhh_rl.native_atom14.masks import (  # noqa: E402
    cdr_fake_atom_mask, design_token_offset, residue_atom_masks,
)
from vhh_rl.native_atom14.paired_noise import paired_noising  # noqa: E402

POOL = ROOT / "runs/native_pool"
COND = POOL / "conditioning"
BASE_CKPT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")


def _first_case() -> tuple[str, list[int], Path]:
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") == "train":
            path = COND / f"{row['case_id']}.pt"
            if path.is_file():
                return row["case_id"], sorted(int(p) for p in row["design_positions"]), path
    pytest.skip("no conditioning cache")


def test_uniform_weights_equal_n3_loss():
    pytest.importorskip("boltzgen")
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    case_id, positions, cond_path = _first_case()
    cond = move_conditioning(load_conditioning(COND, case_id))
    base = load_base_model(BASE_CKPT, device="cuda")
    sm = base.structure_module
    feats = cond["feats"]
    kwargs = {
        "s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
        "feats": feats, "multiplicity": 1,
        "diffusion_conditioning": cond["diffusion_conditioning"],
    }
    coords_file = sorted(glob.glob(str(POOL / "train" / case_id / "coords" / "*.pt")))[0]
    x0 = torch.load(coords_file, map_location="cuda", weights_only=True).float()
    sigma = sm.noise_distribution(1)
    noise = torch.randn_like(x0.unsqueeze(0))
    from vhh_rl.native_atom14.paired_noise import paired_noising as pn
    atom_mask = feats["atom_pad_mask"].reshape(-1)
    paired = pn(x0, x0, atom_mask, sigma,
                augmentation=sm.coordinate_augmentation, noise=noise)

    pref = cdr_fake_atom_mask(
        feats["atom_to_token"], feats["design_mask"],
        feats["fake_atom_mask"], feats["atom_pad_mask"]).cuda()
    out_n3 = per_sample_denoising_loss(
        sm, feats, paired["x0_w_aug"], paired["x_t_w"], sigma, kwargs,
        preference_mask=pref, require_grad=False,
    )
    offset = design_token_offset(feats["token_index"], feats["design_mask"], positions)
    masks = residue_atom_masks(
        feats["atom_to_token"], feats["fake_atom_mask"],
        feats["atom_pad_mask"], [p + offset for p in positions]).cuda()
    losses, _ = per_residue_denoising_loss(
        sm, feats, paired["x0_w_aug"], paired["x_t_w"], sigma, kwargs, masks,
        require_grad=False,
    )
    losses = losses.reshape(-1)
    w = torch.full_like(losses, 1.0 / losses.numel())
    weighted = (losses * w).sum() * sm.loss_weight(sigma.reshape(-1))
    n3 = out_n3.per_sample_loss.reshape(-1)[0]
    rel = float((weighted - n3).abs() / n3.abs().clamp_min(1e-6))
    assert rel < 1e-4, f"uniform weighted loss {weighted.item()} vs N3 {n3.item()} (rel {rel})"
