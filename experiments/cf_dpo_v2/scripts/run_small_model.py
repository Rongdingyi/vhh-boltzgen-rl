#!/usr/bin/env python
"""CF-DPO v2 experiment 2: exact small-model comparison (proposal §12).

All methods fit the same linear potential H_theta(X)=theta.phi(X) and see the
same queried data (pool, pairs, single-site counterfactual edges and
same-sequence geometry edges).  Differences are purely in the loss/data use.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_dpo_v2.small_model import SmallWorld  # noqa: E402

OUT = ROOT / "runs/cf_dpo_v2/small_model"


def build_features(world: SmallWorld, device="cpu") -> torch.Tensor:
    """[n_nodes, n_feat]: sequence interactions (<=3) + geometry indicators."""
    rows = []
    seq_feats = []
    for s in world.sequences:
        z = [int(c) for c in s]
        feats = []
        for combo in world.feature_names:
            prod = 1.0
            for i in combo:
                prod *= z[i]
            feats.append(prod)
        seq_feats.append(feats)
    n_seq_feat = len(world.feature_names)
    n_geom_feat = max(1, world.n_geom - 1)
    for si, s in enumerate(world.sequences):
        for g in range(world.n_geom):
            geo = [1.0 if k == g - 1 else 0.0 for k in range(n_geom_feat)]
            rows.append(seq_feats[si] + geo)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def nodes_index(world: SmallWorld) -> dict[tuple[str, int], int]:
    out = {}
    idx = 0
    for s in world.sequences:
        for g in range(world.n_geom):
            out[(s, g)] = idx
            idx += 1
    return out


def q_from_theta(theta: torch.Tensor, feats: torch.Tensor, log_p0: torch.Tensor) -> torch.Tensor:
    logits = log_p0 + feats @ theta
    return torch.softmax(logits, dim=0)


def exact_target(world: SmallWorld, feats: torch.Tensor, log_p0: torch.Tensor) -> torch.Tensor:
    # q* ∝ p0 exp(R(S)/beta): recover R/beta from the world's q_star logits
    logits = []
    for s in world.sequences:
        for g in range(world.n_geom):
            logits.append(math.log(world.p0_seq[s]) + math.log(world.geom_probs[s][g])
                          + world.reward[s] / world.beta)
    t = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(t, dim=0)


def build_data(world: SmallWorld, n_cases: int, pool_size: int, seed: int):
    """Pool + pairs + queried edges, shared by every method."""
    rng = random.Random(seed)
    edges = []            # (a_seq, a_geom, b_seq, b_geom, dR, kind)
    pairs = []            # (winner, loser, drop/gain per site)
    for _ in range(n_cases):
        pool = [rng.choice(world.sequences) for _ in range(pool_size)]
        ranked = sorted(set(pool), key=lambda s: world.reward[s])
        if len(ranked) < 2:
            continue
        winner, loser = ranked[-1], ranked[0]
        pairs.append((winner, loser))
        g_w, g_l = rng.randrange(world.n_geom), rng.randrange(world.n_geom)
        if world.n_geom >= 2:
            while g_l == g_w:
                g_l = rng.randrange(world.n_geom)
        edges.append((loser, g_l, winner, g_w,
                      world.reward[winner] - world.reward[loser], "global"))
        for i in range(world.n_sites):
            if winner[i] == loser[i]:
                continue
            drop_seq = world.local_move(winner, i)
            gain_seq = world.local_move(loser, i)
            d_drop = world.reward[winner] - world.reward[drop_seq]
            d_gain = world.reward[gain_seq] - world.reward[loser]
            edges.append((winner, g_w, drop_seq, g_w, d_drop, "drop"))
            edges.append((loser, g_l, gain_seq, g_l, d_gain, "gain"))
        # same-sequence geometry edges for the winner/loser (decoder equivalence)
        for s in {winner, loser}:
            if world.n_geom >= 2:
                g1, g2 = rng.sample(range(world.n_geom), 2)
                edges.append((s, g1, s, g2, 0.0, "same_seq"))
    return pairs, edges


def build_eval_edges(world: SmallWorld, n_cases: int, seed: int):
    """Held-out local move edges for sign-accuracy evaluation."""
    rng = random.Random(seed)
    edges = []
    for _ in range(n_cases):
        s = rng.choice(world.sequences)
        i = rng.randrange(world.n_sites)
        moved = world.local_move(s, i)
        edges.append((s, 0, moved, 0, world.reward[moved] - world.reward[s], "eval"))
    pairs = []
    for a, b in itertools.combinations(world.sequences, 2):
        for i in range(world.n_sites):
            if a[i] == b[i]:
                continue
            d, g = world.local_signs(a, b, i)
            pairs.append((a, b, i, d, g))
    return edges, pairs


def credit_weights(world: SmallWorld, pairs) -> dict[tuple[str, str], float]:
    """Deployed CF-DPO analogue: c_cons + uniform floor per pair (mean weight)."""
    eta = 0.75
    out = {}
    for winner, loser in pairs:
        diffs = [i for i in range(world.n_sites) if winner[i] != loser[i]]
        cons = []
        for i in diffs:
            d, g = world.local_signs(winner, loser, i)
            cons.append(min(d, g) if (d > 0 and g > 0) else 0.0)
        total = sum(c for c in cons if c > 0)
        n = max(1, len(diffs))
        if total > 0:
            w = st.mean([(1 - eta) / n + eta * max(c, 0.0) / total for c in cons])
        else:
            w = 1.0
        out[(winner, loser)] = w
    return out


def sign_only_weights(world: SmallWorld, pairs) -> dict[tuple[str, str], float]:
    out = {}
    for winner, loser in pairs:
        diffs = [i for i in range(world.n_sites) if winner[i] != loser[i]]
        c = 0
        for i in diffs:
            d, g = world.local_signs(winner, loser, i)
            c += int(d > 0) + int(g > 0)
        out[(winner, loser)] = c / max(1, 2 * len(diffs))
    return out


def edge_sign_accuracy(world, theta, feats, edges) -> float:
    idx = nodes_index(world)
    ok = n = 0
    for (sa, ga, sb, gb, dR, kind) in edges:
        if dR == 0:
            continue
        z = float((feats[idx[(sb, gb)]] - feats[idx[(sa, ga)]]) @ theta.detach())
        ok += int((z > 0) == (dR > 0))
        n += 1
    return ok / max(1, n)


def train_method(name: str, world: SmallWorld, feats, log_p0, target, pairs, edges,
                 n_steps: int = 1500, lr: float = 0.05, seed: int = 0,
                 eval_edges=None, eval_pairs=None):
    torch.manual_seed(seed)
    n_feat = feats.shape[1]
    theta = torch.zeros(n_feat, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    idx = nodes_index(world)
    tau = world.tau

    def H(seq, g=0):
        return float((feats[idx[(seq, g)]] @ theta).detach())

    w_cf = credit_weights(world, pairs)
    w_sign = sign_only_weights(world, pairs)
    for step in range(n_steps):
        opt.zero_grad()
        if name == "oracle":
            q = q_from_theta(theta, feats, log_p0)
            loss = (q * (q.log() - target.log())).sum()
        else:
            pieces = []
            if name in ("current_cf", "sign_only"):
                for (win, lose) in pairs:
                    a = idx[(win, 0)]
                    b = idx[(lose, 0)]
                    z = (feats[b] @ theta - feats[a] @ theta) / tau
                    t = torch.sigmoid(torch.tensor(
                        (world.reward[win] - world.reward[lose]) / tau))
                    weight = w_cf[(win, lose)] if name == "current_cf" else w_sign[(win, lose)]
                    pieces.append(weight * torch.nn.functional.binary_cross_entropy_with_logits(
                        z.reshape(1), t.reshape(1)))
            else:  # signed / v2
                for (sa, ga, sb, gb, dR, kind) in edges:
                    if kind == "same_seq" and name != "v2":
                        continue
                    a = idx[(sa, ga)]
                    b = idx[(sb, gb)]
                    z = (feats[b] @ theta - feats[a] @ theta) / tau
                    t = torch.sigmoid(torch.tensor(dR / tau))
                    pieces.append(torch.nn.functional.binary_cross_entropy_with_logits(
                        z.reshape(1), t.reshape(1)))
            loss = torch.stack(pieces).mean()
        loss.backward()
        opt.step()
    # evaluate
    with torch.no_grad():
        q = q_from_theta(theta, feats, log_p0)
        seq_kl = 0.0
        cond_kl = 0.0
        for s in world.sequences:
            qg = [float(q[idx[(s, g)]]) for g in range(world.n_geom)]
            qs = sum(qg)
            if qs > 1e-12:
                for g in range(world.n_geom):
                    p = qg[g] / qs
                    r = world.geom_probs[s][g]
                    cond_kl += qs * p * math.log(p / r)
            if qs > 0:
                seq_kl += qs * math.log(qs / world.q_star_seq[s])
        full_kl = float((q * (q.log() - target.log())).sum())
    out = {"seq_kl": seq_kl, "cond_geom_kl": cond_kl, "full_kl": full_kl,
           "theta_norm": float(theta.norm())}
    if eval_edges is not None:
        out["sign_acc"] = edge_sign_accuracy(world, theta, feats, eval_edges)
    if eval_pairs is not None:
        idx = nodes_index(world)
        flip_ok = flip_n = pos_ok = pos_n = 0
        for (a, b, i, d, g) in eval_pairs:
            # signed method prediction at site i: H(loser->winner move) - H(loser)
            moved = world.local_move(b, i)
            z = float((feats[idx[(moved, 0)]] - feats[idx[(b, 0)]]) @ theta.detach())
            correct = int((z > 0) == (g > 0))
            if d * g < 0:
                flip_n += 1
                flip_ok += correct
            else:
                pos_n += 1
                pos_ok += correct
        out["sign_acc_flip"] = flip_ok / max(1, flip_n)
        out["sign_acc_consistent"] = pos_ok / max(1, pos_n)
    return out


def sign_accuracy(world: SmallWorld, theta: torch.Tensor, feats, edges) -> float:
    idx = nodes_index(world)
    ok = n = 0
    for (sa, ga, sb, gb, dR, kind) in edges:
        if kind in ("global", "same_seq") or dR == 0:
            continue
        z = float((feats[idx[(sb, gb)]] - feats[idx[(sa, ga)]]) @ theta.detach())
        ok += int((z > 0) == (dR > 0))
        n += 1
    return ok / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--pool", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for world_seed in range(args.seeds):
        world = SmallWorld(n_sites=args.sites, seed=20260914 + world_seed)
        feats = build_features(world)
        log_p0 = torch.log(torch.tensor(
            [world.p0_seq[s] * world.geom_probs[s][g]
             for s in world.sequences for g in range(world.n_geom)], dtype=torch.float32))
        target = exact_target(world, feats, log_p0)
        pairs, edges = build_data(world, args.cases, args.pool, seed=1234 + world_seed)
        eval_edges, eval_pairs = build_eval_edges(world, 64, seed=777 + world_seed)
        for method in ("current_cf", "sign_only", "signed", "v2", "oracle"):
            res = train_method(method, world, feats, log_p0, target, pairs, edges,
                              n_steps=args.steps, seed=world_seed,
                              eval_edges=eval_edges, eval_pairs=eval_pairs)
            results.setdefault(method, []).append(res)
        print(f"[world {world_seed}] done", flush=True)

    summary = {}
    for method, rows in results.items():
        summary[method] = {
            k: {"mean": st.mean(r[k] for r in rows),
                "std": st.stdev(r[k] for r in rows) if len(rows) > 1 else 0.0}
            for k in ("seq_kl", "cond_geom_kl", "full_kl",
                      "sign_acc", "sign_acc_flip", "sign_acc_consistent")
        }
    (OUT / "summary.json").write_text(json.dumps({"summary": summary, "n_seeds": args.seeds,
                                                  "args": vars(args)}, indent=1))
    lines = ["| method | seq KL | cond-geom KL | full KL | sign acc | sign acc (flip) | sign acc (consistent) |",
             "|---|---|---|---|---|---|---|"]
    for method, s in summary.items():
        lines.append(
            f"| {method} | {s['seq_kl']['mean']:.4f} ± {s['seq_kl']['std']:.4f} | "
            f"{s['cond_geom_kl']['mean']:.4f} ± {s['cond_geom_kl']['std']:.4f} | "
            f"{s['full_kl']['mean']:.4f} ± {s['full_kl']['std']:.4f} | "
            f"{s['sign_acc']['mean']:.3f} | "
            f"{s['sign_acc_flip']['mean']:.3f} | "
            f"{s['sign_acc_consistent']['mean']:.3f} |")
    doc = "# CF-DPO v2 — Experiment 2: exact small model\n\n" + "\n".join(lines) + "\n"
    (OUT / "EXP2_SMALL_MODEL.md").write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
