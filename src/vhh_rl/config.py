from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScorerConfig:
    name: str = "vhh_source_ensemble"
    direction: str = "higher_is_better"  # AUDIT-confirmed; see docs/AUDIT.md §scorer
    objective: str = "native_probability"
    worker_python: str = "/share/home/rongdingyi/.conda/envs/vhh-guidance-esmc/bin/python"
    worker_script: str = ""
    cache: bool = True
    # Round-2 guardrail: opt = main_reward - guard_lambda * max(0, floor - nativeness).
    # floor per case comes from guard_floors_path (JSON {case_id: floor});
    # if empty, floors default to -inf (guardrail disabled).
    guard_lambda: float = 0.0
    guard_floors_path: str = ""


@dataclass
class SamplingConfig:
    temperature: str = "auto"  # "auto" | float string
    candidate_temperatures: tuple[float, ...] = (0.1, 0.3, 0.5, 0.8)
    min_unique_rate: float = 0.60
    temperature_cases: int = 4
    temperature_samples_per_case: int = 16
    group_size: int = 8


@dataclass
class AdvantageConfig:
    type: str = "rank"  # "rank" | "group_z"
    eps: float = 1.0e-6


@dataclass
class AlgorithmConfig:
    name: str = "grpo"  # "grpo" | "reinforce"
    clip_eps: float = 0.20
    update_epochs: int = 2
    beta_kl: float = 0.01


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class TrainingConfig:
    cases_per_rollout_batch: int = 4
    max_updates: int = 200
    checkpoint_every: int = 25
    eval_every: int = 25


@dataclass
class EvalConfig:
    num_samples_per_case: int = 8
    compare_base_best_of_n: bool = True


@dataclass
class RLConfig:
    project: str = "vhh_boltzgen_rl_round1"
    seed: int = 42
    mode: str = "grpo"  # baseline | toy-reinforce | toy-grpo | scorer-overfit | evaluate
    boltzgen_root: str = ""
    boltzgen_checkpoint: str = ""
    scorer_root: str = ""
    manifest: str = ""
    output_root: str = "runs"
    device: str = "cuda"
    precision: str = "fp32"
    train_scope: str = "inverse_folding_decoder"
    keep_eval_mode: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reward: ScorerConfig = field(default_factory=ScorerConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    max_cases: int = 0  # 0 = all manifest cases
    debug_cases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_output_dir(self) -> Path:
        return Path(self.output_root) / f"{self.mode}_{Path(self.manifest).stem}"


def _apply(d: dict[str, Any], obj: Any) -> Any:
    for key, value in d.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key {key!r} for {type(obj).__name__}")
        current = getattr(obj, key)
        if isinstance(current, (SamplingConfig, AdvantageConfig, AlgorithmConfig, OptimizerConfig, TrainingConfig, EvalConfig, ScorerConfig)):
            _apply(value, current)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> RLConfig:
    raw = yaml.safe_load(Path(path).read_text())
    env_map = {
        "${BOLTZGEN_ROOT}": "BOLTZGEN_ROOT",
        "${BOLTZGEN_CKPT}": "BOLTZGEN_CKPT",
        "${VHH_SCORER_ROOT}": "VHH_SCORER_ROOT",
        "${VHH_RL_DATA}": "VHH_RL_DATA",
    }
    cfg = _apply(raw or {}, RLConfig())
    for key in ("boltzgen_root", "boltzgen_checkpoint", "scorer_root", "manifest"):
        value = getattr(cfg, key)
        if value in env_map:
            env = os.environ.get(env_map[value], "")
            if not env:
                raise SystemExit(f"environment variable {env_map[value]} is required (config {key})")
            setattr(cfg, key, env)
    return cfg


def config_sha256(cfg: RLConfig) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
