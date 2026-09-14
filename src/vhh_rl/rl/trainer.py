"""Round-1 online RL trainer (REINFORCE / GRPO-style) on the IF decoder.

Implements task book §26-§35 exactly: per-case grouped rewards, rank/group-z
advantages, two update epochs over stored trajectories, clipped ratio with
frozen stored old_logprobs, exact categorical KL to a frozen reference, and
the §19 FR hard check before every update.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from ..adapters.boltzgen_if import CapturedCase, BoltzGenIFAdapter
from ..rewards.scoring import ScorerAdapter, ToySequenceReward, ScoreBatch
from .advantages import group_advantages
from .losses import exact_categorical_kl, grpo_policy_loss, reinforce_loss


def _diversity_stats(sequences: list[str]) -> dict[str, float]:
    unique = len(set(sequences)) / max(1, len(sequences))
    if len(sequences) < 2:
        return {"unique_rate": unique, "mean_pairwise_identity": 1.0}
    total, pairs = 0.0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            same = sum(a == b for a, b in zip(sequences[i], sequences[j]))
            total += same / len(sequences[i])
            pairs += 1
    return {"unique_rate": unique, "mean_pairwise_identity": total / max(1, pairs)}


def run_updates(
    *,
    adapter: BoltzGenIFAdapter,
    cases: list[CapturedCase],
    sequences_by_case: dict[str, str],
    seed_base_by_case: dict[str, int] | None = None,
    reward,  # ScorerAdapter | ToySequenceReward
    advantage_type: str,
    algorithm: str,
    clip_eps: float,
    beta_kl: float,
    update_epochs: int,
    group_size: int,
    temperature: float,
    max_updates: int,
    output_dir: Path,
    checkpoint_every: int,
    eval_every: int,
    optimizer: torch.optim.Optimizer,
    reference_decoder,
    design_meta: dict[str, dict],
    log,
) -> dict:
    """The only place gradients are taken.  Returns the metrics log."""
    device = adapter.device
    # Lightning's predict teardown leaves torch.set_grad_enabled(False) ambient
    # (documented as the nograd_view_trap in m3v2_ifold_gate.py).  Everything
    # below builds gradients, so force grad mode on and restore on exit.
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    trainable = [p for p in adapter.decoder.parameters() if p.requires_grad]
    # reference is a deep-copied adapter whose decoder must never update
    reference_decoder.eval()
    metrics_path = output_dir / "train_metrics.jsonl"
    history: list[dict] = []

    def make_reward_fn():
        if isinstance(reward, ToySequenceReward):
            return lambda seqs, case: reward.score_sequences(seqs, case.design_positions)
        return lambda seqs, case: reward.score_sequences(seqs, case)

    start_update = 0
    ckpt_dir = output_dir / "checkpoints"
    if ckpt_dir.is_dir():
        existing = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if existing:
            latest = existing[-1]
            payload = torch.load(latest, map_location=device, weights_only=False)
            adapter.decoder.load_state_dict(payload["decoder_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            start_update = int(payload["global_update"])
            log(f"[resume] continuing from {latest.name} (update {start_update})")
    for update in range(start_update + 1, max_updates + 1):
        started = time.time()
        trajectories = []
        for case in cases:
            for k in range(group_size):
                base = (seed_base_by_case or {}).get(case.case_id, 0)
                traj = adapter.rollout_with_logprobs(
                    case, temperature, seed=base * 1000 + update * 100 + k
                )
                traj.case_id = case.case_id
                native = sequences_by_case[case.case_id]
                if not adapter.fr_unchanged(case, traj, native):
                    raise RuntimeError(
                        f"FR mutation detected in case {case.case_id} update {update}: "
                        "training aborted (task book §19)"
                    )
                trajectories.append((case, traj))
        # rewards, grouped per case
        groups: dict[str, list] = {}
        for case, traj in trajectories:
            groups.setdefault(case.case_id, []).append(traj)
        reward_by_case: dict[str, list[float]] = {}
        for case_id, trajs in groups.items():
            case_for_group = next(c for c in cases if c.case_id == case_id)
            score = make_reward_fn()([t.final_sequence for t in trajs], case_for_group)
            for t, raw in zip(trajs, score.raw_scores.tolist()):
                t.raw_reward = float(raw)
            opt = score.optimization_scores.tolist()
            for t, o in zip(trajs, opt):
                t.optimization_reward = float(o)
            reward_by_case[case_id] = [t.optimization_reward for t in trajs]
            if score.info:
                for t in trajs:
                    t.score_info = dict(score.info)
        flat_adv, zero_var = group_advantages(reward_by_case, method=advantage_type)
        flat_index = 0
        for case, traj in trajectories:
            traj.advantage = float(flat_adv[flat_index])
            flat_index += 1
        for p in trainable:
            p.grad = None

        update_stats = {"clip_fraction": 0.0, "ratio_mean": 1.0}
        losses = []
        kls = []
        entropies = []
        for _epoch in range(update_epochs):
            optimizer.zero_grad(set_to_none=True)
            policy_losses, kl_losses, ent_list, ratio_stats = [], [], [], []
            for case, traj in trajectories:
                current = adapter.replay_actions(case, traj, temperature, with_grad=True)
                with torch.no_grad():
                    reference = adapter.replay_actions(
                        case, traj, temperature, with_grad=False,
                        decoder=reference_decoder,
                    )
                traj.new_logprobs = [current["action_logprobs"]]
                traj.new_full_logprobs = [current["full_logprobs"]]
                traj.ref_full_logprobs = [reference["full_logprobs"]]
                new_lp = current["action_logprobs"]
                old_lp = torch.tensor([e.old_logprob for e in traj.events], device=new_lp.device)
                if algorithm == "reinforce":
                    loss = reinforce_loss(new_lp, traj.advantage)
                    ratio_stats.append({"ratio_mean": 1.0, "clip_fraction": 0.0})
                else:
                    loss, stats = grpo_policy_loss(new_lp, old_lp, traj.advantage, clip_eps)
                    ratio_stats.append(stats)
                policy_losses.append(loss)
                kl_losses.append(
                    exact_categorical_kl(
                        current["full_logprobs"], reference["full_logprobs"]
                    )
                )
                with torch.no_grad():
                    ent_list.append(
                        sum(e.old_entropy for e in traj.events) / max(1, traj.num_policy_events)
                    )
            loss_policy = torch.stack(policy_losses).mean()
            loss_kl = torch.stack(kl_losses).mean()
            loss = loss_policy if algorithm == "reinforce" else loss_policy + beta_kl * loss_kl
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0e9)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss_policy))
            kls.append(float(loss_kl))
            entropies.append(sum(ent_list) / len(ent_list))
            update_stats = {
                "ratio_mean": sum(r["ratio_mean"] for r in ratio_stats) / len(ratio_stats),
                "clip_fraction": sum(r["clip_fraction"] for r in ratio_stats) / len(ratio_stats),
            }
        raw_all = [t.raw_reward for _, t in trajectories]
        seqs = [t.final_sequence for _, t in trajectories]
        div = _diversity_stats(seqs)
        infos = [getattr(t, "score_info", {}) or {} for _, t in trajectories]
        guard_nat = [i["guard_nat_mean"] for i in infos if "guard_nat_mean" in i]
        guard_viol = [i["guard_viol_rate"] for i in infos if "guard_viol_rate" in i]
        record = {
            "update": update,
            "num_cases": len(cases),
            "num_trajectories": len(trajectories),
            "reward/raw_mean": sum(raw_all) / len(raw_all),
            "reward/raw_min": min(raw_all),
            "reward/raw_max": max(raw_all),
            "policy/loss": sum(losses) / len(losses),
            "policy/kl": sum(kls) / len(kls),
            "policy/entropy": sum(entropies) / len(entropies),
            "policy/ratio_mean": update_stats["ratio_mean"],
            "policy/clip_fraction": update_stats["clip_fraction"],
            "train/grad_norm": float(grad_norm),
            "train/lr": optimizer.param_groups[0]["lr"],
            "sequence/unique_rate": div["unique_rate"],
            "sequence/mean_pairwise_identity": div["mean_pairwise_identity"],
            "debug/zero_variance_groups": zero_var,
            "seconds": time.time() - started,
        }
        if guard_nat:
            record["reward/guard_nat_mean"] = sum(guard_nat) / len(guard_nat)
            record["reward/guard_viol_rate"] = sum(guard_viol) / len(guard_viol)
        history.append(record)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        log(
            f"[update {update}] raw_mean={record['reward/raw_mean']:.4f} "
            f"kl={record['policy/kl']:.4f} unique={div['unique_rate']:.2f}"
        )
        if checkpoint_every and update % checkpoint_every == 0:
            _save_checkpoint(adapter, optimizer, update, temperature, output_dir)
    torch.set_grad_enabled(prev_grad)
    return {"history": history, "metrics_path": str(metrics_path)}


def _save_checkpoint(
    adapter: BoltzGenIFAdapter,
    optimizer: torch.optim.Optimizer,
    update: int,
    temperature: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoints/checkpoint_{update:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder_state_dict": adapter.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_update": update,
            "temperature": temperature,
            "base_checkpoint_sha256": adapter.checkpoint_sha256,
        },
        path,
    )
    return path
