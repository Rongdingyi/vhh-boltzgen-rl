"""Round-based online preference DPO trainer (task book §9/§12/§58).

Phase 0 arms:
  B  fresh sibling pairs, ``all changed`` support
  V  *exact same* pair bank + schedule as B, ``verified`` support only (§7)

Later phases reuse the same loop with a different sigma sampler; the DPO math
and update schedule format stay unchanged.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .metrics import pair_stats, reward_query_count
from .pair_bank import load_pair_bank, pair_bank_path, save_pair_bank, schedule_path
from .pair_builder import build_pairs_for_groups
from .step_randomness import (build_update_schedule, read_schedule,
                              sample_full_sigma, sample_standard_noise_like,
                              seed_augmentation, write_schedule)

DEVICE = "cuda"


@dataclass
class OnlineArmConfig:
    arm: str
    seed: int
    rounds: int = 4
    updates_per_round: int = 25
    siblings: int = 8
    sampling_steps: int = 50
    branch_progress: float = 0.60
    min_reward_gap: float = 0.30
    peer_types: tuple = ("worst", "median")
    support: str = "all"
    sigma_region: str = "full"          # full | suffix | prefix (Phase 1+)
    ema_decay: float = 0.99
    beta: float = 10.0
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    data_source: str = "fresh"          # fresh | bank
    bank_source_root: str | None = None
    output_dir: str = ""
    bank_root: str = ""
    log_tag: str | None = None

    @property
    def total_updates(self) -> int:
        return self.rounds * self.updates_per_round


def _default_deps():
    from ..branch_distill.branch_rollout import run_branch_group, slice_conditioning
    from ..cf_opsd.rollout import collect_case_rollouts
    from ..native_atom14.checkpoint import (
        load_base_model, make_policy_reference, parameter_drift,
        save_native_checkpoint, trainable_score_params,
    )
    from ..cf_opsd.behavior_ema import ema_update

    def _sha256(path) -> str:
        import hashlib
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _sample_sigma(pair, spec, cfg, student):
        if cfg.sigma_region == "full":
            return sample_full_sigma(student.structure_module,
                                     seed=spec.sigma_seed, device=DEVICE)
        raise NotImplementedError("conditional sigma sampling arrives with Phase 1")

    return {
        "sample_sigma": _sample_sigma,
        "collect_case_rollouts": collect_case_rollouts,
        "run_branch_group": run_branch_group,
        "slice_conditioning": slice_conditioning,
        "load_base_model": load_base_model,
        "make_policy_reference": make_policy_reference,
        "save_native_checkpoint": save_native_checkpoint,
        "trainable_score_params": trainable_score_params,
        "parameter_drift": parameter_drift,
        "ema_update": ema_update,
        "sha256": _sha256,
    }


def run_online_arm(cfg: OnlineArmConfig, deps: dict | None = None) -> dict:
    """Train one online arm and return the round summary."""
    deps = {**_default_deps(), **(deps or {})}
    from .dpo_step import changed_position_dpo_step
    from .query_utils import network_kwargs_for_pair

    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"

    base = deps["load_base_model"](deps["base_checkpoint"], device=DEVICE)
    student, reference = deps["make_policy_reference"](base, device=DEVICE)
    behavior = deps["load_base_model"](deps["base_checkpoint"], device=DEVICE)
    params = deps["trainable_score_params"](student)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def log(row: dict) -> None:
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    rounds_log = []
    reward_queries_total = 0
    pair_records_total = 0
    t0 = time.time()
    for round_index in range(1, cfg.rounds + 1):
        if cfg.data_source == "fresh":
            behavior_ckpt = (deps["base_checkpoint"] if round_index == 1
                             else output_dir / f"behavior_r{round_index - 1}.pt")
            groups = deps["build_groups"](round_index, behavior_ckpt, cfg)
            pairs = build_pairs_for_groups(
                groups, deps["design_positions"], round_index=round_index,
                peer_types=cfg.peer_types, min_reward_gap=cfg.min_reward_gap)
            reward_queries = reward_query_count(groups)
            behavior_sha = deps["sha256"](behavior_ckpt) if Path(behavior_ckpt).is_file() else None
            save_pair_bank(pair_bank_path(cfg.bank_root, cfg.seed, round_index),
                           round_index=round_index, behavior_checkpoint_sha256=behavior_sha,
                           branch_progress=cfg.branch_progress, groups=groups,
                           pairs=pairs, reward_queries=reward_queries)
            schedule = build_update_schedule(pairs, updates=cfg.updates_per_round,
                                             seed=cfg.seed, round_index=round_index)
            write_schedule(schedule_path(cfg.bank_root, cfg.seed, round_index), schedule)
            groups_meta = None
        else:
            source = Path(cfg.bank_source_root or cfg.bank_root)
            bank = load_pair_bank(pair_bank_path(source, cfg.seed, round_index))
            pairs = bank["pairs"]
            reward_queries = int(bank["reward_queries"])
            schedule = read_schedule(schedule_path(source, cfg.seed, round_index))
            groups_meta = {"n_groups": len(bank["groups"])}

        if cfg.support == "verified":
            deps["attach_verified"](pairs, cfg)   # Phase 0 control only
        if not pairs:
            raise SystemExit(f"arm {cfg.arm} round {round_index}: no eligible pairs")

        for spec in schedule:
            pair = next(p for p in pairs if p.pair_id == spec.pair_id)
            kwargs = network_kwargs_for_pair(pair, device=DEVICE)
            feats = kwargs["feats"]
            sigma = deps["sample_sigma"](pair, spec, cfg, student)
            noise = sample_standard_noise_like(pair.winner_coords.to(DEVICE),
                                               spec.noise_seed)
            seed_augmentation(spec.augmentation_seed)
            out = changed_position_dpo_step(
                student.structure_module, reference.structure_module, feats, pair,
                kwargs, support=cfg.support, sigma=sigma.to(DEVICE), noise=noise,
                beta=cfg.beta)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            log({
                "phase": "phase0", "arm": cfg.arm, "seed": cfg.seed,
                "round": round_index, "update": spec.update_index,
                "pair_id": pair.pair_id, "case_id": pair.case_id,
                "branch_progress": pair.branch_progress,
                "branch_step": pair.branch_step, "branch_sigma": pair.branch_sigma,
                "sigma_region": cfg.sigma_region,
                "train_sigma": float(sigma.reshape(-1)[0]),
                "winner_reward": pair.winner_reward, "loser_reward": pair.loser_reward,
                "reward_gap": pair.reward_gap,
                "n_changed_positions": len(pair.changed_positions_all),
                "support_mode": cfg.support,
                "n_support_positions": len(
                    pair.changed_positions_verified if cfg.support == "verified"
                    else pair.changed_positions_all),
                "dpo_z": float(out.dpo.z.mean()),
                "implicit_acc": float(out.dpo.implicit_acc),
                "loss": float(out.loss.detach()),
                "grad_norm": float(grad_norm),
                "noise_seed": spec.noise_seed,
                "augmentation_seed": spec.augmentation_seed,
                "sigma_seed": spec.sigma_seed,
            })

        deps["ema_update"](behavior.structure_module, student.structure_module,
                           decay=cfg.ema_decay)
        deps["save_native_checkpoint"](deps["base_checkpoint"], student,
                                       output_dir / f"student_r{round_index}.pt",
                                       {"arm": cfg.arm, "round": round_index,
                                        "seed": cfg.seed})
        deps["save_native_checkpoint"](deps["base_checkpoint"], behavior,
                                       output_dir / f"behavior_r{round_index}.pt",
                                       {"arm": cfg.arm, "round": round_index})
        stats = (pair_stats(pairs, deps["design_positions"]))
        reward_queries_total += reward_queries
        pair_records_total += len(pairs)
        rounds_log.append({
            "round": round_index,
            "n_groups": groups_meta["n_groups"] if groups_meta else None,
            "n_valid_siblings": reward_queries,
            "reward_queries_round": reward_queries,
            "reward_queries_cumulative": reward_queries_total,
            "n_pair_records": len(pairs),
            "pair_records_cumulative": pair_records_total,
            **stats,
            "behavior_sha": None if cfg.data_source == "bank" else "see bank",
            "param_drift": deps["parameter_drift"](student, reference)["total"],
            "elapsed_seconds": time.time() - t0,
        })
        print(f"[online-{cfg.arm}] seed {cfg.seed} round {round_index}: "
              f"pairs={len(pairs)} queries={reward_queries} "
              f"hamming={stats.get('mean_pair_hamming'):.1f}", flush=True)
    final = output_dir / f"student_r{cfg.rounds}.pt"
    (output_dir / "student_r4.pt").write_bytes(final.read_bytes())
    summary = {"arm": cfg.arm, "seed": cfg.seed, "rounds": rounds_log,
               "reward_queries_total": reward_queries_total,
               "pair_records_total": pair_records_total,
               "updates": cfg.updates_per_round * cfg.rounds}
    (output_dir / "rounds.json").write_text(json.dumps(summary, indent=1))
    return summary
