"""K-sibling branching from one captured state (task book §21-§25)."""
from __future__ import annotations

import torch

from .decode import decode_coords_with_fr
from .metrics import group_metrics
from .tail_sampler import continue_from_state
from .types import BranchGroup, SiblingEndpoint


def slice_conditioning(conditioning: dict, index: int) -> dict:
    """Slice a batch-shaped conditioning dict down to one source sample."""
    def _slice(value):
        if torch.is_tensor(value) and value.dim() > 0:
            if value.shape[0] > index:
                return value[index: index + 1]
            return value
        return value
    return {k: _slice(v) for k, v in conditioning.items()}


def run_branch_group(*, diffusion, case_id: str, source_sample_index: int,
                     progress: float, start_step: int,
                     pre_state: torch.Tensor, conditioning: dict,
                     multiplicity: int, num_sampling_steps: int, group_seed: int,
                     reference_sequence: str, fr_positions,
                     design_positions, scorer=None, spec=None,
                     decode_endpoints: bool = True,
                     step_scale: float | None = None,
                     noise_scale: float | None = None) -> BranchGroup:
    """Sample K siblings from the captured state and score their endpoints."""
    feats = conditioning["feats"]
    atom_mask = feats["atom_pad_mask"]
    if atom_mask.dim() == 2 and atom_mask.shape[0] != 1:
        raise ValueError("conditioning must be sliced to one sample before branching")
    network_kwargs = {
        "s_inputs": conditioning["s_inputs"],
        "s_trunk": conditioning["s_trunk"],
        "feats": feats,
        "diffusion_conditioning": conditioning["diffusion_conditioning"],
    }
    out = continue_from_state(
        diffusion,
        pre_state=pre_state,
        start_step=start_step,
        num_sampling_steps=num_sampling_steps,
        multiplicity=multiplicity,
        atom_mask=atom_mask,
        network_condition_kwargs=network_kwargs,
        seed=group_seed,
        step_scale=step_scale,
        noise_scale=noise_scale,
    )

    endpoints = out["endpoint_coords"].detach().float().cpu()
    queries = out["first_query"].detach().float().cpu()
    anchors = out["first_anchor"].detach().float().cpu()

    decoded = []
    for k in range(multiplicity):
        if decode_endpoints:
            audit = decode_coords_with_fr(endpoints[k], feats, reference_sequence,
                                          fr_positions)
        else:
            audit = {"sequence": "", "contains_invalid": False, "fr_mismatch": 0}
        decoded.append(audit)

    rewards: list[float | None] = [None] * multiplicity
    if scorer is not None:
        scorable = [k for k, d in enumerate(decoded)
                    if not d["contains_invalid"] and d["fr_mismatch"] == 0]
        if scorable:
            batch = scorer.score_sequences(
                [decoded[k]["sequence"] for k in scorable], spec=spec)
            for k, value in zip(scorable, batch.raw_scores.tolist()):
                rewards[k] = float(value)

    siblings = [
        SiblingEndpoint(
            branch_index=k,
            endpoint_coords=endpoints[k],
            endpoint_sequence=decoded[k]["sequence"],
            reward=rewards[k],
            contains_invalid=bool(decoded[k]["contains_invalid"]),
            fr_mismatch=int(decoded[k]["fr_mismatch"]),
            query_coords=queries[k],
            anchor_coords=anchors[k],
        )
        for k in range(multiplicity)
    ]
    group = BranchGroup(
        case_id=case_id,
        source_sample_index=source_sample_index,
        progress=float(progress),
        start_step=int(start_step),
        sigma=float(out["first_sigma"]),
        group_seed=int(group_seed),
        pre_state=pre_state.detach().float().cpu().clone(),
        full_query_batch=queries,
        full_anchor_batch=anchors,
        siblings=siblings,
        conditioning={"s_inputs": conditioning["s_inputs"].detach().cpu().clone(),
                      "s_trunk": conditioning["s_trunk"].detach().cpu().clone(),
                      "feats": feats.detach().cpu().clone() if torch.is_tensor(feats)
                      else feats,
                      "diffusion_conditioning":
                          conditioning["diffusion_conditioning"].detach().cpu().clone()
                          if torch.is_tensor(conditioning["diffusion_conditioning"])
                          else conditioning["diffusion_conditioning"]},
        meta={"n_sampling_steps": int(num_sampling_steps),
              "multiplicity": int(multiplicity),
              "coords_traj_tail": out["coords_traj_tail"],
              "design_positions": tuple(design_positions),
              "sampling_scales": {"step_scale": step_scale,
                                  "noise_scale": noise_scale}},
    )
    group.meta["group_metrics"] = group_metrics(siblings, design_positions)
    return group
