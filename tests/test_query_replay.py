"""Saved query replayed through behavior reproduces the stored anchor (§82)."""
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
    from vhh_rl.cf_opsd.checkpoint import load_base_model
    from vhh_rl.cf_opsd.rollout import load_trajectories
    from vhh_rl.native_atom14.dpo_trainer import load_conditioning, move_conditioning

    case_dir = sorted(ROLL.iterdir())[0]
    traj_file = sorted(case_dir.glob("seed*.pt"))[0]
    traj = load_trajectories(traj_file)[0]
    cond = move_conditioning(load_conditioning(ROOT / "runs/native_pool/conditioning",
                                               case_dir.name))
    kwargs = {"s_inputs": cond["s_inputs"], "s_trunk": cond["s_trunk"],
              "feats": cond["feats"], "multiplicity": 1,
              "diffusion_conditioning": cond["diffusion_conditioning"]}
    base = load_base_model(
        Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt"),
        device="cuda")
    step = len(traj.query_states) // 2
    query = traj.query_states[step].cuda().float().unsqueeze(0)
    sigma = torch.tensor([traj.sigmas[step]], device="cuda")
    with torch.no_grad():
        denoised, _ = base.structure_module.preconditioned_network_forward(
            query, sigma, training=False, network_condition_kwargs=kwargs)
    diff = (denoised.cpu().float() - traj.anchors[step]).abs().max().item()
    assert diff < 1e-5, diff
