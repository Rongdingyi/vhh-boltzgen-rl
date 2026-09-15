"""Exact finite-state test-bed for CF-DPO v2 (proposal §12 experiment 2).

Binary sites z in {0,1}^n, one geometry latent per sequence from a small set
of realizations with reference probabilities p0(X|S).  The reward R(S) may
contain arbitrary-order interactions.  Everything is enumerated, so the
Gibbs target q*(X) ∝ p0(X) exp(R(S)/beta) and the conditional p0(X|S) are
exact.

Training signals compared (all fit the same linear potential family
H_theta(X) = theta . phi(X), so the comparison isolates the *data/loss*, not
model capacity):

  a) current_cf : static nonnegative pair weights, c_cons + uniform floor
                  (the deployed CF-DPO recipe)
  b) sign_only  : weights from sign(drop)/sign(gain) of the counterfactual
                  queries (positive-only, no magnitude/direction)
  c) signed     : directed local comparison edges (BCE on t=sigmoid(dR/tau))
  d) v2         : signed edges + same-sequence (decoder-equivalent) edges
  e) oracle     : direct fit to the exact Gibbs target

Metrics: sequence-marginal KL, conditional-geometry KL, full-distribution KL,
and local preference sign accuracy on held-out local edges.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import torch


@dataclass
class SmallWorld:
    n_sites: int
    beta: float = 1.0
    tau: float = 1.0
    n_geom: int = 3
    seed: int = 20260914

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.sequences = ["".join(bits) for bits in itertools.product("01", repeat=self.n_sites)]
        self.seq_index = {s: i for i, s in enumerate(self.sequences)}
        # arbitrary-order reward coefficients (up to third order)
        self.coeffs: dict[tuple[int, ...], float] = {}
        for i in range(self.n_sites):
            self.coeffs[(i,)] = rng.uniform(-1.5, 1.5)
        for i, j in itertools.combinations(range(self.n_sites), 2):
            self.coeffs[(i, j)] = rng.uniform(-1.5, 1.5) * (1.0 if rng.random() < 0.6 else -1.0)
        if self.n_sites >= 3:
            for combo in itertools.combinations(range(self.n_sites), 3):
                self.coeffs[combo] = rng.uniform(-1.0, 1.0)
        # geometry realization probabilities per sequence (reference p0(X|S))
        self.geom_probs: dict[str, list[float]] = {}
        for s in self.sequences:
            vals = [rng.uniform(0.5, 2.0) for _ in range(self.n_geom)]
            z = sum(vals)
            self.geom_probs[s] = [v / z for v in vals]
        # reference sequence marginal: uniform over sequences
        self.p0_seq = {s: 1.0 / len(self.sequences) for s in self.sequences}
        # reward
        self.reward: dict[str, float] = {s: self._reward(s) for s in self.sequences}
        # exact Gibbs target over (sequence, geometry) and marginals
        logits = {}
        for s in self.sequences:
            for g in range(self.n_geom):
                logits[(s, g)] = math.log(self.p0_seq[s]) + math.log(self.geom_probs[s][g]) \
                    + self.reward[s] / self.beta
        m = max(logits.values())
        exp = {k: math.exp(v - m) for k, v in logits.items()}
        Z = sum(exp.values())
        self.q_star = {k: v / Z for k, v in exp.items()}
        self.q_star_seq = {s: sum(self.q_star[(s, g)] for g in range(self.n_geom))
                           for s in self.sequences}
        # feature map for the trainable family: per-site indicator + pair interactions
        self.feature_names: list[tuple[int, ...]] = []
        for r in range(1, min(self.n_sites, 3) + 1):
            self.feature_names.extend(itertools.combinations(range(self.n_sites), r))
        # deepcopy helper for torch conversion is done in the trainer

    def _reward(self, s: str) -> float:
        z = [int(c) for c in s]
        value = 0.0
        for combo, c in self.coeffs.items():
            prod = 1.0
            for i in combo:
                prod *= z[i]
            value += c * prod
        return value

    # ------------------------------------------------------------- utilities
    def winner_loser(self) -> tuple[str, str]:
        order = sorted(self.sequences, key=lambda s: self.reward[s])
        return order[-1], order[0]

    def local_move(self, s: str, i: int) -> str:
        chars = list(s)
        chars[i] = "1" if chars[i] == "0" else "0"
        return "".join(chars)

    def local_signs(self, base_plus: str, base_minus: str, i: int) -> tuple[float, float]:
        """c_drop / c_gain for site i between a winner and loser sequence."""
        drop = self.reward[base_plus] - self.reward[self.local_move(base_plus, i)]
        gain = self.reward[self.local_move(base_minus, i)] - self.reward[base_minus]
        return drop, gain

    def conditional_geometry_kl(self, q_geom: dict[str, list[float]]) -> float:
        out = 0.0
        for s in self.sequences:
            for g in range(self.n_geom):
                p = q_geom[s][g]
                r = self.geom_probs[s][g]
                if p > 0:
                    out += self.q_star_seq[s] * p * math.log(p / r)
        return out

    def seq_kl(self, q_seq: dict[str, float]) -> float:
        out = 0.0
        for s in self.sequences:
            if q_seq[s] > 0:
                out += q_seq[s] * math.log(q_seq[s] / self.q_star_seq[s])
        return out

    def full_kl(self, q: dict[tuple[str, int], float]) -> float:
        out = 0.0
        for k, p in q.items():
            if p > 0:
                out += p * math.log(p / self.q_star[k])
        return out


def linear_logits(theta: torch.Tensor, world: SmallWorld) -> dict[tuple[str, int], float]:
    """H_theta(X) for every (sequence, geometry) node (geometry-independent)."""
    out = {}
    for s in world.sequences:
        z = [int(c) for c in s]
        h = 0.0
        for k, combo in enumerate(world.feature_names):
            prod = 1.0
            for i in combo:
                prod *= z[i]
            h += float(theta[k]) * prod
        for g in range(world.n_geom):
            out[(s, g)] = h
    return out


def q_from_H(H: dict[tuple[str, int], float], world: SmallWorld) -> dict[tuple[str, int], float]:
    logits = {}
    for (s, g), h in H.items():
        logits[(s, g)] = math.log(world.p0_seq[s]) + math.log(world.geom_probs[s][g]) + h
    m = max(logits.values())
    exp = {k: math.exp(v - m) for k, v in logits.items()}
    Z = sum(exp.values())
    return {k: v / Z for k, v in exp.items()}
