"""Phase A: on-policy rollout capture with exact query states and anchors.

Hooks the official sampler (wrapping ``AtomDiffusion.sample`` for the schedule
and ``AtomDiffusion.preconditioned_network_forward`` for the exact network
input per step) plus the writer's official ``res_from_atom14`` decode.  No
sampler equation is modified.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch

from ..native_atom14.adapter import (
    BOLTZGEN_ROOT, DESIGN_CKPT, FOLDING_CKPT, IFOLD_CKPT, MOLDIR,
    NativeDesignAdapter, seed_all, _default_rng_patch,
)
from .types import OPSDTrajectory

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
POOL = ROOT / "runs/native_pool"
OUT = ROOT / "runs/cf_opsd/rollouts"


def collect_case_rollouts(
    adapter: NativeDesignAdapter,
    spec_path: str | Path,
    case_id: str,
    run_root: str | Path,
    num_designs: int,
    seed: int,
) -> tuple[list[OPSDTrajectory], dict]:
    """Run the official design pipeline once and capture exact OPSD data.

    Returns (trajectories, info) where trajectories are ordered by writer
    sample index and carry exact query states / anchors / schedule.
    """
    from boltzgen.model.modules.diffusion import AtomDiffusion
    from boltzgen.task.predict import writer as writer_module

    spec_path = Path(spec_path)
    run_root = Path(run_root)
    adapter.num_designs = int(num_designs)
    raw_writer: list[dict] = []
    net_calls: list[dict] = []
    schedule: dict = {}

    original_decode = writer_module.res_from_atom14
    original_sample = AtomDiffusion.sample
    original_fwd = AtomDiffusion.preconditioned_network_forward

    def decode_hook(sample, *args, **kwargs):
        out = original_decode(sample, *args, **kwargs)
        from ..native_atom14.decode import sequence_from_feat
        sequence, tokens, invalid = sequence_from_feat(out)
        raw_writer.append({
            "coords": sample["coords"].detach().float().cpu().clone(),
            "common": {k: v.detach().cpu().clone() for k, v in sample.items()
                       if k != "coords" and torch.is_tensor(v)},
            "decoded_sequence": sequence,
            "decoded_tokens": tokens,
            "contains_invalid": invalid,
        })
        return out

    def sample_hook(sample_self, *args, **kwargs):
        result = original_sample(sample_self, *args, **kwargs)
        num_steps = kwargs.get("num_sampling_steps") or sample_self.num_sampling_steps
        if sample_self.sampling_schedule == "af3":
            sigmas = sample_self.sample_schedule_af3(num_steps)
        else:
            sigmas = sample_self.sample_schedule_dilated(num_steps)
        gammas = torch.where(sigmas > sample_self.gamma_min, sample_self.gamma_0, 0.0)
        schedule.update({
            "sigmas": [float(s) for s in sigmas.tolist()],
            "t_hats": [float(sigmas[i]) * (1.0 + float(gammas[i + 1]))
                       for i in range(int(num_steps))],
            "num_steps": int(num_steps),
            "coords_traj": [t.detach().float().cpu().clone()
                            for t in result["coords_traj"]],
            "sample_atom_coords": result["sample_atom_coords"].detach().float().cpu().clone(),
            "x0_coords_traj": [t.detach().float().cpu().clone()
                               for t in result["x0_coords_traj"]],
        })
        return result

    def fwd_hook(diff_self, noised_atom_coords, sigma, network_condition_kwargs,
                 training=False, **kw):
        out = original_fwd(diff_self, noised_atom_coords, sigma,
                           network_condition_kwargs, training=training, **kw)
        denoised = out[0] if isinstance(out, tuple) else out
        net_calls.append({
            "noisy": noised_atom_coords.detach().float().cpu().clone(),
            "sigma": sigma.detach().float().cpu().clone()
            if torch.is_tensor(sigma) else torch.tensor([float(sigma)]),
            "denoised": denoised.detach().float().cpu().clone(),
        })
        return out

    writer_module.res_from_atom14 = decode_hook
    AtomDiffusion.sample = sample_hook
    AtomDiffusion.preconditioned_network_forward = fwd_hook
    started = time.time()
    try:
        if run_root.exists():
            if "cf_opsd" not in str(run_root):
                raise RuntimeError(f"refusing to clear unexpected run dir {run_root}")
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        import boltzgen.cli.boltzgen as bgcli

        argv = adapter._argv(spec_path, run_root)
        args = bgcli.build_parser().parse_args(argv)
        seed_all(int(seed))
        import numpy as np

        original_rng = _default_rng_patch(int(seed))
        try:
            bgcli.run_command(args)
        finally:
            np.random.default_rng = original_rng
    finally:
        writer_module.res_from_atom14 = original_decode
        AtomDiffusion.sample = original_sample
        AtomDiffusion.preconditioned_network_forward = original_fwd

    # match writer samples to sampler batch indices via final coords
    finals = schedule["sample_atom_coords"]
    matches = []
    for w_idx, raw in enumerate(raw_writer):
        best = None
        for b_idx in range(finals.shape[0]):
            diff = (finals[b_idx] - raw["coords"]).abs().max().item()
            if best is None or diff < best[0]:
                best = (diff, b_idx)
        if best is None or best[0] > 1e-3:
            raise RuntimeError(f"{case_id}: writer sample {w_idx} unmatched ({best})")
        matches.append(best[1])

    trajectories = []
    for i, raw in enumerate(raw_writer):
        b = matches[i]
        trajectories.append(OPSDTrajectory(
            case_id=case_id,
            sample_index=i,
            seed=int(seed),
            endpoint_coords=raw["coords"],
            endpoint_sequence=raw["decoded_sequence"],
            endpoint_reward=None,
            sigmas=[float(c["sigma"].reshape(-1)[0]) for c in net_calls],
            coords_traj=schedule["coords_traj"],
            query_states=[c["noisy"][b] for c in net_calls],
            anchors=[c["denoised"][b] for c in net_calls],
            feats_common=raw["common"],
            contains_invalid=raw["contains_invalid"],
        ))
    info = {
        "case_id": case_id,
        "seed": int(seed),
        "n_designs": len(raw_writer),
        "elapsed_seconds": time.time() - started,
        "n_net_calls": len(net_calls),
        "schedule": {k: v for k, v in schedule.items() if k not in ("coords_traj", "x0_coords_traj", "sample_atom_coords")},
        "matches": matches,
    }
    return trajectories, info


def save_trajectories(trajectories: list[OPSDTrajectory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([t.__dict__ for t in trajectories], path)


def load_trajectories(path: Path) -> list[OPSDTrajectory]:
    rows = torch.load(path, map_location="cpu", weights_only=False)
    return [OPSDTrajectory(**r) for r in rows]
