"""Saved query replayed through behavior reproduces the stored anchor (§82).

The design pipeline calls the sampler with ``multiplicity = diffusion_samples``
(= diffusion_batch_size), so the exact same-query forward must be replayed with
the full sampling batch and the recorded multiplicity; the per-design anchor is
then the corresponding slice of the batch output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ROLL = ROOT / "runs/cf_opsd/rollouts/train"


def test_query_replay_matches_anchor():
    pytest.importorskip("boltzgen")
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    if not ROLL.is_dir() or not any(ROLL.glob("*/*.pt")):
        pytest.skip("Phase A rollouts not collected yet")
    torch.set_float32_matmul_precision("high")  # design CLI runs TF32
    from vhh_rl.cf_opsd.checkpoint import load_base_model
    from vhh_rl.cf_opsd.rollout import load_cond_kwargs, load_contexts, load_trajectories
    from vhh_rl.native_atom14.dpo_trainer import move_conditioning

    case_dir = sorted(ROLL.iterdir())[0]
    traj_file = sorted(case_dir.glob("seed*.pt"))[0]
    trajs = sorted(load_trajectories(traj_file), key=lambda t: t.sample_index)
    contexts = load_contexts(traj_file)
    stored = load_cond_kwargs(traj_file)
    assert contexts and stored, "rollout file has no sampling contexts/conditioning"
    steps = sorted(contexts)
    step = steps[len(steps) // 2]  # a captured candidate step
    ctx = contexts[step]
    mult = int(ctx["multiplicity"])
    stored_step = load_cond_kwargs(traj_file, step=step)

    kwargs = move_conditioning(stored_step)
    kwargs["multiplicity"] = int(stored_step.get("multiplicity", mult))
    base = load_base_model(
        Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"),
        device="cuda")
    full_query = ctx["query"].cuda().float()
    # the sampler expands its scalar sigma to the batch; shape matters for TF32
    sigma = torch.full((full_query.shape[0],), float(trajs[0].sigmas[step]), device="cuda")
    with torch.no_grad():
        denoised, _ = base.structure_module.preconditioned_network_forward(
            full_query, sigma, training=False, network_condition_kwargs=kwargs)
    denoised = denoised.float().cpu()
    worst = float((denoised - ctx["anchors"]).abs().max())
    assert worst < 1e-5, worst
