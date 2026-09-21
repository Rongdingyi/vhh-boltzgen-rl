"""Deterministic per-update randomness (task book §9/§10).

Every arm of the same seed/round/update must see the same pair, the same
standard Gaussian noise and the same rigid augmentation draw; only the sigma
domain may differ.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .types import UpdateSpec


def make_generator(seed: int, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def sample_standard_noise_like(coords: torch.Tensor, seed: int) -> torch.Tensor:
    g = make_generator(seed, coords.device)
    return torch.randn((1, *coords.shape), generator=g, device=coords.device,
                       dtype=coords.dtype)


def seed_augmentation(seed: int) -> None:
    """Seed the global RNG right before the augmented DPO step (§10)."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample_full_sigma(structure_module, *, seed: int, device) -> torch.Tensor:
    """Original training noise distribution (independent generator, §18)."""
    g = make_generator(seed, device)
    sigma = structure_module.noise_distribution(1, generator=g)
    return sigma.reshape(-1)


def build_update_schedule(pairs, *, updates: int = 25, seed: int,
                          round_index: int) -> list[UpdateSpec]:
    """Frozen per-round update schedule (§9): record choice + all RNG seeds."""
    if not pairs:
        raise ValueError("empty pair list")
    rng = random.Random(f"{int(seed)}:{int(round_index)}")
    order = list(range(len(pairs)))
    rng.shuffle(order)
    specs = []
    for i in range(int(updates)):
        pair = pairs[order[i % len(order)]]
        specs.append(UpdateSpec(
            update_index=i,
            round_index=int(round_index),
            pair_id=pair.pair_id,
            sigma_seed=rng.randrange(0, 2**31 - 1),
            noise_seed=rng.randrange(0, 2**31 - 1),
            augmentation_seed=rng.randrange(0, 2**31 - 1),
        ))
    return specs


def write_schedule(path, specs: list[UpdateSpec]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.__dict__ for s in specs], indent=1))


def read_schedule(path) -> list[UpdateSpec]:
    rows = json.loads(Path(path).read_text())
    return [UpdateSpec(**row) for row in rows]
