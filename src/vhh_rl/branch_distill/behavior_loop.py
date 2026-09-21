"""Gate 3 outer loop: offline A / online B / distill C,D (task book §41-§52).

All heavy dependencies are injectable so the orchestration can be unit-tested
without GPUs; the default dependency set wires the real adapter, tail sampler,
reward scorer and trainers.
"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .online_diff_dpo import online_dpo_step
from .online_distill import MASK_MODES, online_distill_step
from .query_fit import record_target_mask
from .teacher_select import eligible, select_teacher_peer
from .decode import decode_coords_with_fr
from .local_target import (build_target, carrier_audit, carrier_positions_match,
                           changed_design_positions, verified_changed_positions)

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
DEVICE = "cuda"


@dataclass
class Gate3Config:
    train_cases: list[str]
    rounds: int = 4
    updates_per_round: int = 25
    sampling_steps: int = 50
    siblings_per_case: int = 8
    selected_progress: float = 0.7
    source_states_per_case: int = 1
    ema_decay: float = 0.99
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    min_reward_gap: float = 0.30
    teacher_peer_pairs: tuple = ("best_vs_worst", "best_vs_median")
    seed: int = 20260915
    log_every: int = 1

    @property
    def total_updates(self) -> int:
        return self.rounds * self.updates_per_round


def progress_to_step(progress: float, n_steps: int) -> int:
    """§22 (shared with the Gate 1 scripts)."""
    return max(0, min(n_steps - 1, int(round(progress * n_steps)) - 1))


def load_gate3_config(path: Path) -> Gate3Config:
    import yaml
    payload = yaml.safe_load(Path(path).read_text())
    known = {f for f in Gate3Config.__dataclass_fields__}
    kwargs = {k: v for k, v in payload.items() if k in known}
    if "teacher_peer_pairs" in kwargs:
        kwargs["teacher_peer_pairs"] = tuple(kwargs["teacher_peer_pairs"])
    return Gate3Config(**kwargs)


# ------------------------------------------------------------------ records --

def build_online_records(groups, cfg: Gate3Config, design_positions,
                         reference_sequence: str, fr_positions,
                         *, peer_types=None) -> tuple[list, list[dict]]:
    """Teacher/peer selection + target construction + carrier audit (§28-§30)."""
    peer_types = peer_types or cfg.teacher_peer_pairs
    records, audits = [], []
    for group in groups:
        feats = group.conditioning["feats"]
        for peer_type in peer_types:
            selection = select_teacher_peer(
                group.siblings, peer_type.replace("best_vs_", ""),
                changed_positions_fn=lambda t, p: changed_design_positions(
                    t, p, design_positions))
            audit = {"case_id": group.case_id, "progress": group.progress,
                     "peer_type": peer_type,
                     "eligible": eligible(selection, cfg.min_reward_gap)}
            if not audit["eligible"]:
                audits.append(audit)
                continue
            teacher = group.siblings[selection["teacher_index"]]
            peer = group.siblings[selection["peer_index"]]
            # Amendment 1: audit the FULL changed set, then supervise only the
            # carrier-verified subset (target geometry is rebuilt on it).
            carrier = carrier_audit(peer.endpoint_coords, teacher.endpoint_coords,
                                    feats, selection["changed_positions"],
                                    reference_sequence, fr_positions)
            matches = carrier_positions_match(carrier["sequence"],
                                              selection["teacher_sequence"],
                                              selection["changed_positions"])
            verified = verified_changed_positions(matches,
                                                  selection["changed_positions"])
            audit.update({
                "teacher_index": selection["teacher_index"],
                "peer_index": selection["peer_index"],
                "reward_gap": selection["reward_gap"],
                "n_changed": len(selection["changed_positions"]),
                "carrier_invalid": carrier["contains_invalid"],
                "carrier_fr": carrier["fr_mismatch"],
                "carrier_all_match": bool(all(matches.values())),
                "carrier_matches": matches,
                "amendment_id": 1,
                "carrier_verified_positions": len(verified),
                "carrier_original_rate": (len(verified) / len(matches)) if matches else 0.0,
            })
            if not verified:
                audit["eligible"] = False
                audits.append(audit)
                continue
            target, touched = build_target(peer.anchor_coords, teacher.endpoint_coords,
                                           feats, verified)
            # truth check for the target actually used (§39 B): the target must
            # decode to the teacher AA on the verified positions, stay valid and
            # FR-clean
            target_audit = decode_coords_with_fr(target, feats, reference_sequence,
                                                 fr_positions)
            target_match = carrier_positions_match(target_audit["sequence"],
                                                   selection["teacher_sequence"],
                                                   verified)
            audit.update({
                "target_invalid": target_audit["contains_invalid"],
                "target_fr": target_audit["fr_mismatch"],
                "target_all_match": bool(all(target_match.values())),
                "target_matches": target_match,
            })
            audits.append(audit)
            record = _make_record(group, {**selection,
                                          "changed_positions": tuple(verified)},
                                  target, touched, carrier, verified, target_audit,
                                  target_match)
            records.append(record)
    return records, audits


def _make_record(group, selection, target, touched, carrier, verified=None,
                 target_audit=None, target_match=None):
    from .types import DistillRecord
    teacher = group.siblings[selection["teacher_index"]]
    peer = group.siblings[selection["peer_index"]]
    return DistillRecord(
        record_id=f"{group.case_id}:{group.progress:.2f}:{selection['peer_type']}",
        case_id=group.case_id,
        progress=group.progress,
        start_step=group.start_step,
        group_seed=group.group_seed,
        branch_count=len(group.siblings),
        teacher_index=selection["teacher_index"],
        peer_index=selection["peer_index"],
        teacher_reward=selection["teacher_reward"],
        peer_reward=selection["peer_reward"],
        reward_gap=selection["reward_gap"],
        teacher_sequence=selection["teacher_sequence"],
        peer_sequence=selection["peer_sequence"],
        changed_positions=selection["changed_positions"],
        full_query_batch=group.full_query_batch,
        full_anchor_batch=group.full_anchor_batch,
        peer_anchor=group.siblings[selection["peer_index"]].anchor_coords,
        target_coords=target.detach().cpu(),
        touched_mask=touched.detach().cpu(),
        pre_state=group.pre_state,
        conditioning=group.conditioning,
        meta={"sigma": group.sigma, "peer_type": selection["peer_type"],
              "carrier_sequence": carrier["sequence"],
              "teacher_endpoint": teacher.endpoint_coords,
              "peer_endpoint": peer.endpoint_coords,
              "design_positions": tuple(group.meta.get("design_positions", ())),
              "sampling_scales": dict(group.meta.get("sampling_scales", {})),
              "amendment_id": 1,
              "carrier_verified_positions": tuple(verified or selection["changed_positions"]),
              "carrier_invalid": bool(carrier["contains_invalid"]),
              "carrier_fr": int(carrier["fr_mismatch"]),
              "target_invalid": bool((target_audit or {}).get("contains_invalid", False)),
              "target_fr": int((target_audit or {}).get("fr_mismatch", 0)),
              "target_all_match": bool(all((target_match or {}).values()))
              if target_match else False,
              }, 
    )


def target_mask_for(group, touched) -> torch.Tensor:
    feats = group.conditioning["feats"]
    pad = feats["atom_pad_mask"].reshape(-1).bool()
    fake = feats["fake_atom_mask"].reshape(-1).bool()
    return touched.reshape(-1).bool() & fake & pad


# ------------------------------------------------------------------- arms ----

def run_online_update(arm: str, student, reference, records, cfg: Gate3Config,
                      rng: random.Random, step: int, optimizer, params,
                      log_fn) -> dict:
    """One optimizer update for arm B/C/D over a sampled online record."""
    if not records:
        raise RuntimeError(f"arm {arm}: no eligible online records")
    record = rng.choice(records)
    from .query_fit import network_kwargs
    # Arm B optimizes a single winner/loser pair: the DPO forward is batch 1 even
    # though the pair was discovered inside a K-sibling group.
    network = network_kwargs(record.conditioning, 1, device=DEVICE)
    design_positions = record.meta.get("design_positions") or ()
    if arm == "B":
        teacher = record.meta["teacher_endpoint"].to(DEVICE)
        peer = record.meta["peer_endpoint"].to(DEVICE)
        out = online_dpo_step(student.structure_module, reference.structure_module,
                              network["feats"], teacher, peer,
                              record.changed_positions, network,
                              design_positions=design_positions, beta=10.0)
        loss = out.loss
    else:
        mask_mode = "all_design" if arm == "C" else "changed_only"
        loss, _metrics = online_distill_step(student, record, mask_mode,
                                             design_positions)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
    optimizer.step()
    row = {"arm": arm, "step": step, "record_id": record.record_id,
           "case_id": record.case_id, "teacher_index": record.teacher_index,
           "peer_index": record.peer_index, "teacher_reward": record.teacher_reward,
           "peer_reward": record.peer_reward, "reward_gap": record.reward_gap,
           "n_changed_positions": len(record.changed_positions),
           "mask_mode": "changed_only" if arm == "D" else
                        ("all_design" if arm == "C" else "uniform_changed"),
           "loss": float(loss.detach()), "grad_norm": float(grad_norm)}
    log_fn(row)
    return row


def ema_refresh(behavior, student, decay: float) -> None:
    from ..cf_opsd.behavior_ema import ema_update
    ema_update(behavior.structure_module, student.structure_module, decay=decay)


def clear_round_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
