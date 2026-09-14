"""Gates 4/5: toy REINFORCE, toy GRPO, and real-scorer single-case overfit."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import torch

from ..adapters.boltzgen_if import BoltzGenIFAdapter
from ..config import RLConfig
from ..rewards.scoring import ToySequenceReward
from ..rl.trainer import run_updates
from .baseline import _PerCaseScorer
from .common import build_adapter, case_spec_for, capture_cases, ensure_slurm, log_factory, manifest_cases


def run_train(cfg: RLConfig, output_dir: Path) -> int:
    ensure_slurm()
    log = log_factory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

    cases = manifest_cases(cfg, splits=("train",))
    if cfg.debug_cases:
        cases = [c for c in cases if c.case_id in set(cfg.debug_cases)]
    if cfg.max_cases:
        cases = cases[: cfg.max_cases]
    if not cases:
        raise SystemExit("no training cases")
    adapter = build_adapter(cfg)
    captured = capture_cases(adapter, cfg, cases, output_dir / "captures")
    seed_base_of = {c.case_id: c.seed_base for c in cases}

    # §11: freeze everything, unfreeze exactly the decoder; assert identity
    for name, module in (("encoder", adapter.encoder),):
        for p in module.parameters():
            assert not p.requires_grad, f"{name} must stay frozen"
    for p in adapter.decoder.parameters():
        p.requires_grad_(True)
    from boltzgen.model.models.boltz import InverseFoldingDecoder  # type: ignore

    assert isinstance(adapter.decoder, InverseFoldingDecoder), (
        "structure_module is not InverseFoldingDecoder; re-run the audit"
    )
    trainable_ids = {id(p) for p in adapter.trainable_parameters()}

    # §13: immutable reference
    reference_decoder = copy.deepcopy(adapter.decoder)
    reference_decoder.eval()
    for p in reference_decoder.parameters():
        p.requires_grad_(False)
    reference_hash = json.dumps(
        {k: float(v.sum()) for k, v in reference_decoder.state_dict().items()}, sort_keys=True
    )

    optimizer = torch.optim.AdamW(
        adapter.trainable_parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    if cfg.mode in ("toy-reinforce", "toy-grpo"):
        # §25: target AA chosen from the baseline rollouts of case 0 (allowed-AA
        # intersection comes from the per-residue constraint mask implicitly)
        case0 = captured[0]
        seeds = [seed_base_of[case0.case_id] * 1000 + i for i in range(8)]
        baseline = [adapter.rollout_with_logprobs(case0, 0.5, seed=s).final_sequence for s in seeds]
        target = ToySequenceReward.pick_target(baseline, case0.design_positions)
        log(f"[toy] target AA = {target}")
        reward = ToySequenceReward(target)
        algorithm = "reinforce" if cfg.mode == "toy-reinforce" else "grpo"
    else:
        from .baseline import _build_scorer  # real scorer (may be toy in tests)

        reward = _build_scorer(cfg, cases)
        algorithm = cfg.algorithm.name

    temperature = float(cfg.sampling.temperature) if cfg.sampling.temperature != "auto" else 0.5
    result = run_updates(
        adapter=adapter,
        cases=captured,
        sequences_by_case={c.case_id: c.full_sequence for c in cases},
        seed_base_by_case=seed_base_of,
        reward=reward,
        advantage_type=cfg.advantage.type,
        algorithm=algorithm,
        clip_eps=cfg.algorithm.clip_eps,
        beta_kl=cfg.algorithm.beta_kl,
        update_epochs=cfg.algorithm.update_epochs,
        group_size=cfg.sampling.group_size,
        temperature=temperature,
        max_updates=cfg.training.max_updates,
        output_dir=output_dir,
        checkpoint_every=cfg.training.checkpoint_every,
        eval_every=cfg.training.eval_every,
        optimizer=optimizer,
        reference_decoder=reference_decoder,
        design_meta={},
        log=log,
    )
    (output_dir / "reference_state.json").write_text(
        json.dumps({"reference_sum_hash": reference_hash}, indent=2)
    )
    return 0
