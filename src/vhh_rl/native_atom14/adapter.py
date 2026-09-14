"""Native-design adapter: run the official BoltzGen atom14 design pipeline
in-process and capture per-sample atom14 coordinates + official decoded
sequences (task book §14-16).

No sampler/decoder logic is reimplemented:
  - sampling: official CLI (`boltzgen run ... --steps design --skip_inverse_folding`)
  - decode:   official `res_from_atom14`, called by the official DesignWriter;
              we wrap that module-level symbol to record its input/output.

Capture semantics
-----------------
The official writer builds one dict per sample with keys from
`const.token_features` / `const.atom_features` (coords is sample-specific,
everything else is shared across the batch).  We capture exactly that dict
(pre-decode) plus the decoded one-hot `res_type`, so decode parity can be
re-checked later from the saved tensors alone.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_BOLTZGEN_SRC = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/src")
if str(_BOLTZGEN_SRC) not in sys.path:
    sys.path.insert(0, str(_BOLTZGEN_SRC))

from .decode import sequence_from_feat  # noqa: E402

# official config/hard-coded paths for this checkout
BOLTZGEN_ROOT = Path("/share/home/rongdingyi/programs/proteingen/boltzgen")
CKPTS = BOLTZGEN_ROOT / "ckpts"
DESIGN_CKPT = CKPTS / "boltzgen1_diverse.ckpt"
IFOLD_CKPT = CKPTS / "boltzgen1_ifold.ckpt"
FOLDING_CKPT = CKPTS / "boltz2_conf_final.ckpt"
MOLDIR = CKPTS / "moldir"


def seed_all(seed: int) -> None:
    """Same seeding method as the m3 production runner (generate_backbones.py)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_rng_patch(seed: int):
    import numpy as np

    original = np.random.default_rng
    calls = 0

    def seeded(value=None):
        nonlocal calls
        if value is not None:
            return original(value)
        result = original(np.random.SeedSequence([int(seed), calls]))
        calls += 1
        return result

    np.random.default_rng = seeded
    return original


@dataclass
class CapturedSample:
    sample_index: int
    coords: torch.Tensor  # [N_atom, 3] fp32 cpu (sampled X0)
    feats_common: dict[str, torch.Tensor]  # everything shared across the batch
    decoded_sequence: str  # official writer decode (path A)
    decoded_tokens: list[str]
    contains_invalid: bool
    cif_path: str | None = None


@dataclass
class Capture:
    case_id: str
    spec_path: str
    seed: int
    num_designs: int
    sampling_steps: int
    diffusion_batch_size: int
    samples: list[CapturedSample] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeDesignAdapter:
    """Runs the official CLI in-process with an official-decode capture hook."""

    def __init__(
        self,
        device: int = 0,
        design_ckpt: Path = DESIGN_CKPT,
        sampling_steps: int = 50,
        diffusion_batch_size: int = 8,
        num_workers: int = 0,
    ) -> None:
        self.device = device
        self.design_ckpt = Path(design_ckpt)
        self.sampling_steps = sampling_steps
        self.diffusion_batch_size = diffusion_batch_size
        self.num_workers = num_workers
        self.checkpoint_sha256 = _sha256_file(self.design_ckpt)

    # ------------------------------------------------------------------ run
    def _argv(self, spec: Path, run_root: Path) -> list[str]:
        return [
            "run", str(spec),
            "--output", str(run_root),
            "--protocol", "nanobody-anything",
            "--num_designs", str(self.num_designs),
            "--steps", "design",
            "--skip_inverse_folding",
            "--no_subprocess",
            "--devices", "1",
            "--num_workers", str(self.num_workers),
            "--use_kernels", "false",
            "--diffusion_batch_size", str(self.diffusion_batch_size),
            "--design_checkpoints", str(self.design_ckpt),
            "--inverse_fold_checkpoint", str(IFOLD_CKPT),
            "--folding_checkpoint", str(FOLDING_CKPT),
            "--moldir", str(MOLDIR),
            "--config", "design",
            f"sampling_steps={self.sampling_steps}",
        ]

    def generate(
        self,
        spec_path: str | Path,
        case_id: str,
        run_root: str | Path,
        num_designs: int,
        seed: int,
        reuse: bool = False,
    ) -> Capture:
        spec_path = Path(spec_path)
        run_root = Path(run_root)
        if not spec_path.is_file():
            raise FileNotFoundError(spec_path)
        self.num_designs = int(num_designs)
        capture = Capture(
            case_id=case_id,
            spec_path=str(spec_path),
            seed=int(seed),
            num_designs=self.num_designs,
            sampling_steps=self.sampling_steps,
            diffusion_batch_size=self.diffusion_batch_size,
        )
        captured_raw: list[dict[str, Any]] = []

        from boltzgen.task.predict import writer as writer_module

        original_decode = writer_module.res_from_atom14

        def hook(sample, *args, **kwargs):
            out = original_decode(sample, *args, **kwargs)
            sequence, tokens, invalid = sequence_from_feat(out)
            captured_raw.append(
                {
                    "coords": sample["coords"].detach().float().cpu().clone(),
                    "common": {
                        key: value.detach().cpu().clone()
                        for key, value in sample.items()
                        if key != "coords" and torch.is_tensor(value)
                    },
                    "decoded_sequence": sequence,
                    "decoded_tokens": tokens,
                    "contains_invalid": invalid,
                }
            )
            return out

        writer_module.res_from_atom14 = hook
        started = time.time()
        try:
            if reuse and (run_root / "intermediate_designs").is_dir():
                # read back previously written results is not implemented; always rerun
                pass
            if run_root.exists():
                # only ever clear our own run dir
                if "native_pool" not in str(run_root) and "native_smoke" not in str(run_root):
                    raise RuntimeError(f"refusing to clear unexpected run dir {run_root}")
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=True)
            import boltzgen.cli.boltzgen as bgcli

            argv = self._argv(spec_path, run_root)
            args = bgcli.build_parser().parse_args(argv)
            seed_all(seed)
            import numpy as np

            original_rng = _default_rng_patch(seed)
            try:
                bgcli.run_command(args)
            finally:
                np.random.default_rng = original_rng
        finally:
            writer_module.res_from_atom14 = original_decode
        capture.elapsed_seconds = time.time() - started

        # collect official CIFs in writer order for cross-checking
        cif_paths = sorted((run_root / "intermediate_designs").rglob("*.cif"))
        for idx, raw in enumerate(captured_raw):
            capture.samples.append(
                CapturedSample(
                    sample_index=idx,
                    coords=raw["coords"],
                    feats_common=raw["common"],
                    decoded_sequence=raw["decoded_sequence"],
                    decoded_tokens=raw["decoded_tokens"],
                    contains_invalid=raw["contains_invalid"],
                    cif_path=str(cif_paths[idx]) if idx < len(cif_paths) else None,
                )
            )
        return capture

    def generate_capture_trajectory(
        self,
        spec_path: str | Path,
        case_id: str,
        run_root: str | Path,
        num_designs: int,
        seed: int,
    ) -> tuple[Capture, dict[str, Any]]:
        """Like generate(), but also capture the full denoising trajectory.

        Hooks ``AtomDiffusion.sample`` (wraps the official call, never changes
        the sampler equations) and records, per sample call:
          - x0_coords_traj : list[T] of [B, N_atom, 3] fp16 CPU tensors
          - sigmas         : the official schedule used for this call
          - t_hats         : sigma_tm * (1 + gamma) per step (audit x-axis)
        The writer's official decode hook still records the final unbatched
        feat dicts so every trajectory can be re-decoded with res_from_atom14.
        """
        spec_path = Path(spec_path)
        run_root = Path(run_root)
        if not spec_path.is_file():
            raise FileNotFoundError(spec_path)
        self.num_designs = int(num_designs)
        capture = Capture(
            case_id=case_id,
            spec_path=str(spec_path),
            seed=int(seed),
            num_designs=self.num_designs,
            sampling_steps=self.sampling_steps,
            diffusion_batch_size=self.diffusion_batch_size,
        )
        raw_writer: list[dict[str, Any]] = []
        sample_calls: list[dict[str, Any]] = []

        from boltzgen.model.modules.diffusion import AtomDiffusion
        from boltzgen.task.predict import writer as writer_module

        original_decode = writer_module.res_from_atom14
        original_sample = AtomDiffusion.sample

        def decode_hook(sample, *args, **kwargs):
            out = original_decode(sample, *args, **kwargs)
            sequence, tokens, invalid = sequence_from_feat(out)
            raw_writer.append(
                {
                    "coords": sample["coords"].detach().float().cpu().clone(),
                    "common": {
                        key: value.detach().cpu().clone()
                        for key, value in sample.items()
                        if key != "coords" and torch.is_tensor(value)
                    },
                    "decoded_sequence": sequence,
                    "decoded_tokens": tokens,
                    "contains_invalid": invalid,
                }
            )
            return out

        def sample_hook(sample_self, *args, **kwargs):
            result = original_sample(sample_self, *args, **kwargs)
            num_steps = kwargs.get("num_sampling_steps") or sample_self.num_sampling_steps
            if sample_self.sampling_schedule == "af3":
                sigmas = sample_self.sample_schedule_af3(num_steps)
            else:
                sigmas = sample_self.sample_schedule_dilated(num_steps)
            gammas = torch.where(sigmas > sample_self.gamma_min, sample_self.gamma_0, 0.0)
            feats = kwargs.get("feats")
            feats_cpu = (
                {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                 for k, v in feats.items()}
                if isinstance(feats, dict) else None
            )
            sample_calls.append(
                {
                    "sample_atom_coords": result["sample_atom_coords"].detach().float().cpu().clone(),
                    "x0_coords_traj": [
                        t.detach().float().cpu().half().clone() for t in result["x0_coords_traj"]
                    ],
                    "feats": feats_cpu,
                    "batch_size": int(result["sample_atom_coords"].shape[0]),
                    "sigmas": [float(s) for s in sigmas.tolist()],
                    "t_hats": [
                        float(sigmas[i]) * (1.0 + float(gammas[i + 1]))
                        for i in range(int(num_steps))
                    ],
                    "num_sampling_steps": int(num_steps),
                }
            )
            return result

        writer_module.res_from_atom14 = decode_hook
        AtomDiffusion.sample = sample_hook
        started = time.time()
        try:
            if run_root.exists():
                if "native_pool" not in str(run_root) and "next_stage" not in str(run_root):
                    raise RuntimeError(f"refusing to clear unexpected run dir {run_root}")
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=True)
            import boltzgen.cli.boltzgen as bgcli

            argv = self._argv(spec_path, run_root)
            args = bgcli.build_parser().parse_args(argv)
            seed_all(seed)
            import numpy as np

            original_rng = _default_rng_patch(seed)
            try:
                bgcli.run_command(args)
            finally:
                np.random.default_rng = original_rng
        finally:
            writer_module.res_from_atom14 = original_decode
            AtomDiffusion.sample = original_sample
        capture.elapsed_seconds = time.time() - started

        cif_paths = sorted((run_root / "intermediate_designs").rglob("*.cif"))
        for idx, raw in enumerate(raw_writer):
            capture.samples.append(
                CapturedSample(
                    sample_index=idx,
                    coords=raw["coords"],
                    feats_common=raw["common"],
                    decoded_sequence=raw["decoded_sequence"],
                    decoded_tokens=raw["decoded_tokens"],
                    contains_invalid=raw["contains_invalid"],
                    cif_path=str(cif_paths[idx]) if idx < len(cif_paths) else None,
                )
            )

        # match each writer sample to its (sample_call, batch_index) by final coords
        matches = []
        for w_idx, raw in enumerate(raw_writer):
            best = None
            for c_idx, call in enumerate(sample_calls):
                finals = call["sample_atom_coords"]
                for b_idx in range(finals.shape[0]):
                    diff = (finals[b_idx] - raw["coords"]).abs().max().item()
                    if best is None or diff < best[0]:
                        best = (diff, c_idx, b_idx)
            if best is None or best[0] > 1e-3:
                raise RuntimeError(
                    f"{case_id}: writer sample {w_idx} could not be matched to a "
                    f"sampler batch (best max-abs diff {None if best is None else best[0]})"
                )
            matches.append({"design_index": w_idx, "sample_call": best[1], "batch_index": best[2]})
        traj = {
            "case_id": case_id,
            "seed": int(seed),
            "sample_calls": sample_calls,
            "matches": matches,
            "elapsed_seconds": capture.elapsed_seconds,
        }
        return capture, traj

    # ------------------------------------------------------- conditioning
    def capture_conditioning(
        self,
        spec_path: str | Path,
        case_id: str,
        run_root: str | Path,
        seed: int,
    ) -> dict[str, Any]:
        """Capture the frozen trunk conditioning for one case.

        Hooks ``AtomDiffusion.sample`` to grab its ``network_condition_kwargs``
        (s_inputs, s_trunk, feats, diffusion_conditioning) and aborts the run
        with a sentinel before any denoising step happens.  The captured
        tensors are exactly what the official forward pass feeds to the score
        model; DPO training reuses them (trunk stays frozen).
        """
        spec_path = Path(spec_path)
        run_root = Path(run_root)
        self.num_designs = 1
        captured: dict[str, Any] = {}

        from boltzgen.model.modules.diffusion import AtomDiffusion

        original_sample = AtomDiffusion.sample

        class _ConditioningDone(Exception):
            pass

        def hook(sample_self, *args, **kwargs):
            for key in ("s_inputs", "s_trunk", "feats", "diffusion_conditioning"):
                value = kwargs.get(key)
                if isinstance(value, dict):
                    converted = {}
                    for k, v in value.items():
                        if torch.is_tensor(v):
                            converted[k] = v.detach().cpu().clone()
                        elif callable(v) and hasattr(v, "keywords") and hasattr(v, "func"):
                            # functools.partial (to_keys): serialize its tensors
                            converted[k] = {
                                "__partial__": True,
                                "func": v.func.__name__,
                                "keywords": {
                                    kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                                    for kk, vv in v.keywords.items()
                                },
                            }
                        else:
                            converted[k] = v
                    captured[key] = converted
                elif torch.is_tensor(value):
                    captured[key] = value.detach().cpu().clone()
            raise _ConditioningDone()

        AtomDiffusion.sample = hook
        try:
            if run_root.exists():
                if "native_cache" not in str(run_root):
                    raise RuntimeError(f"refusing to clear unexpected run dir {run_root}")
                shutil.rmtree(run_root)
            run_root.mkdir(parents=True, exist_ok=True)
            import boltzgen.cli.boltzgen as bgcli

            argv = self._argv(spec_path, run_root)
            args = bgcli.build_parser().parse_args(argv)
            seed_all(seed)
            import numpy as np

            original_rng = _default_rng_patch(seed)
            try:
                try:
                    bgcli.run_command(args)
                except _ConditioningDone:
                    pass
            finally:
                np.random.default_rng = original_rng
        finally:
            AtomDiffusion.sample = original_sample
        missing = [k for k in ("s_inputs", "s_trunk", "feats", "diffusion_conditioning")
                   if k not in captured]
        if missing:
            raise RuntimeError(f"conditioning capture failed for {case_id}: missing {missing}")
        captured["case_id"] = case_id
        captured["spec_path"] = str(spec_path)
        captured["seed"] = int(seed)
        captured["checkpoint_sha256"] = self.checkpoint_sha256
        return captured
