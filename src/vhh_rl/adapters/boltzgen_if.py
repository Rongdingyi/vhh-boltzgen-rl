"""BoltzGen inverse-folding adapter: capture, rollout, replay.

AUDIT summary (docs/AUDIT.md is authoritative):
- local repo == reference commit a3149cf; ``structure_module`` IS the
  ``InverseFoldingDecoder`` when ``inverse_fold=True`` (boltz.py:375).
- encoding: ``edge_idx, valid_mask, s, z = inverse_folding_encoder(feats)``
  (boltz.py:528); the encoder is frozen in round 1, so per-case encoding is
  cached and reused across the K rollouts of a case (task book §60).
- ``sample()`` is ``@torch.no_grad()`` and uses the GLOBAL torch RNG
  (``torch.randperm`` then ``torch.multinomial`` per design position); parity
  therefore requires seeding the global RNG identically and consuming RNG in
  the identical order. ``rollout_with_logprobs`` below replicates sample()
  line-for-line and additionally records per-event log-probs; it never
  changes original sample().
- canonical slice: logits[:, canonicals_offset : canonicals_offset+20]
  (canonicals_offset=2, num_tokens=33 in this build; asserted at load).
- per-residue constraint mask comes from ``build_constraint_logit_mask`` with
  ``feats["aa_constraint_mask"]`` and ``inverse_fold_restriction`` (Cys
  avoidance in the nanobody protocol) -- replay applies the identical mask.
- symmetry: tied positions share one policy decision (logits averaged);
  ``PolicyEvent`` records the group so replay writes the same action into all
  member positions exactly once for the loss.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

AA20 = "ACDEFGHIKLMNPQRSTVWY"
# BoltzGen canonical token order (alphabetical three-letter -> one-letter).
AA_CANONICAL = "ARNDCQEGHILKMFPSTWYV"


def canonical_permutation() -> tuple[list[str], int, int]:
    from boltzgen.data import const

    tokens = list(const.canonical_tokens)
    if const.num_tokens != 33 or const.canonicals_offset != 2:
        # asserted, not assumed: a vocab change silently breaks token offsets
        raise RuntimeError(
            f"unexpected vocab: num_tokens={const.num_tokens} "
            f"canonicals_offset={const.canonicals_offset}"
        )
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    aa_order = tuple(three_to_one[t] for t in tokens)
    return list(aa_order), int(const.canonicals_offset), int(const.num_tokens)


@dataclass
class PolicyEvent:
    position: int
    symmetric_positions: tuple[int, ...]
    action_canonical_id: int
    old_logprob: float
    old_entropy: float


@dataclass
class Trajectory:
    case_id: str
    decode_order: tuple[int, ...]
    events: list[PolicyEvent]
    final_sequence: str
    temperature: float
    seed: int
    design_positions: tuple[int, ...]
    raw_reward: float | None = None
    optimization_reward: float | None = None
    advantage: float | None = None
    new_logprobs: list[torch.Tensor] | None = None
    new_full_logprobs: list[torch.Tensor] | None = None
    ref_full_logprobs: list[torch.Tensor] | None = None

    @property
    def num_policy_events(self) -> int:
        return len(self.events)


@dataclass
class CapturedCase:
    """Frozen per-case encoder outputs plus the feature dict needed by the decoder."""

    case_id: str
    feats: dict[str, Any]
    s: torch.Tensor
    z: torch.Tensor
    edge_idx: torch.Tensor
    valid_mask: torch.Tensor
    design_positions: tuple[int, ...]
    full_sequence: str
    checkpoint_sha256: str = ""

    def to(self, device: str) -> "CapturedCase":
        return CapturedCase(
            case_id=self.case_id,
            feats=self.feats,  # tensors inside moved by the caller where needed
            s=self.s.to(device),
            z=self.z.to(device),
            edge_idx=self.edge_idx.to(device),
            valid_mask=self.valid_mask.to(device),
            design_positions=self.design_positions,
            full_sequence=self.full_sequence,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def _sequence_from_decoded(decoded_seq: torch.Tensor, offset: int) -> str:
    ids = decoded_seq.argmax(dim=-1) - offset
    return "".join(AA_CANONICAL[int(i)] for i in ids.tolist())


class BoltzGenIFAdapter:
    """Loads the official inverse-folding pipeline once per case and exposes
    rollout / replay on top of the captured tensors.

    The official path is used for featurisation + encoding exactly as the
    nanobody-anything ``inverse_folding`` step does (same CLI entry, same
    features); we hook ``InverseFoldingEncoder.forward`` to capture its outputs
    and stop the official sampler from running.  Nothing in the boltzgen tree
    is modified.
    """

    def __init__(self, boltzgen_root: str | Path, checkpoint: str | Path, device: str = "cuda") -> None:
        self.root = Path(boltzgen_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.encoder: Any = None
        self.decoder: Any = None
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.aa_order, self.offset, self.num_tokens = canonical_permutation()
        if "".join(self.aa_order) != AA_CANONICAL:
            raise RuntimeError(
                "canonical one-letter order changed upstream: "
                f"{''.join(self.aa_order)} != {AA_CANONICAL}"
            )

    # ---------------------------------------------------------------- loading

    def capture_case(self, spec_path: str | Path, case_id: str, workdir: str | Path, run_dir: str | Path | None = None) -> CapturedCase:
        """Run the official inverse-folding step with hooks; capture one case.

        The hooks record the encoder module, the decoder module and the tensors
        the sampler would have consumed, then raise a sentinel so the CLI run
        stops before the official ``sample()`` call.  This mirrors the proven
        capture pattern in vhh_esmc_guidance/scripts/cdr_all/m3v2_ifold_gate.py.
        """
        from boltzgen.model.modules.inverse_fold import (
            InverseFoldingDecoder,
            InverseFoldingEncoder,
        )

        captured: dict[str, Any] = {}

        class _CaptureDone(Exception):
            pass

        def make_encoder_hook(orig):
            def hook(self, feats):
                out = orig(self, feats)
                captured["encoder"] = self
                captured["edge_idx"], captured["valid_mask"], captured["s"], captured["z"] = out
                captured["feats"] = copy.deepcopy(_clone_feats(feats))
                return out  # let the pipeline reach decoder.sample, which we
                # capture at entry and stop there (never runs the sampler)
            return hook

        def make_decoder_hook(orig):
            def hook(self, *args, **kwargs):
                captured["decoder"] = self
                raise _CaptureDone()
            return hook

        orig_encoder_forward = InverseFoldingEncoder.forward
        orig_decoder_sample = InverseFoldingDecoder.sample
        InverseFoldingEncoder.forward = make_encoder_hook(orig_encoder_forward)
        InverseFoldingDecoder.sample = make_decoder_hook(orig_decoder_sample)
        workdir = Path(workdir)
        # HARD GUARD: only our own workdir may ever be removed.  The run_dir
        # is a read-only copy SOURCE from the frozen guidance outputs.
        if workdir.exists() and ("captures" not in str(workdir)):
            raise RuntimeError(f"refusing to clear unexpected workdir {workdir}")
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if run_dir is not None:
            run_dir = Path(run_dir)
            if not run_dir.is_dir():
                raise FileNotFoundError(f"design run dir missing: {run_dir}")
            shutil.copytree(run_dir, workdir, dirs_exist_ok=True)
        try:
            try:
                self._run_official(str(spec_path), workdir)
            except _CaptureDone:
                pass
        finally:
            InverseFoldingEncoder.forward = orig_encoder_forward
            InverseFoldingDecoder.sample = orig_decoder_sample
        missing = [k for k in ("encoder", "decoder", "s", "z", "edge_idx", "valid_mask", "feats") if k not in captured]
        if missing:
            raise RuntimeError(f"capture failed for {case_id}: missing {missing}")
        self.encoder = captured["encoder"]
        self.decoder = captured["decoder"]
        # freeze everything at load; the trainer unfreezes exactly the decoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        self.encoder = self.encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()
        # Lightning's predict ran under torch.inference_mode(); every tensor it
        # produced is an inference tensor and silently carries no grad_fn.  A
        # clone taken OUTSIDE inference mode re-materialises them as normal
        # tensors, which is what the replay path's autograd needs.
        with torch.inference_mode(False):
            captured["s"] = captured["s"].clone()
            captured["z"] = captured["z"].clone()
            captured["edge_idx"] = captured["edge_idx"].clone()
            captured["valid_mask"] = captured["valid_mask"].clone()
            captured["feats"] = {
                k: (v.clone() if isinstance(v, torch.Tensor) and torch.is_inference(v) else v)
                for k, v in _feats_to_device(captured["feats"], self.device).items()
            }
        feats = captured["feats"]
        valid_mask = captured["valid_mask"]
        design = self._design_positions_from_feats(feats, valid_mask)
        return CapturedCase(
            case_id=case_id,
            feats=feats,
            s=captured["s"].to(self.device),
            z=captured["z"].to(self.device),
            edge_idx=captured["edge_idx"].to(self.device),
            valid_mask=valid_mask,
            design_positions=design,
            full_sequence="",
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def _run_official(self, spec: str, workdir: Path) -> None:
        import boltzgen.cli.boltzgen as bgcli

        argv = [
            "run", spec,
            "--output", str(workdir),
            "--protocol", "nanobody-anything",
            "--num_designs", "1",
            "--steps", "inverse_folding",
            "--no_subprocess",
            "--devices", "1",
            "--num_workers", "0",
            "--use_kernels", "false",
            "--inverse_fold_checkpoint", str(self.checkpoint),
            "--design_checkpoints", str(self.root / "ckpts/boltzgen1_diverse.ckpt"),
            "--folding_checkpoint", str(self.root / "ckpts/boltzgen1_ifold.ckpt"),
            "--moldir", str(self.root / "ckpts/moldir"),
        ]
        args = bgcli.build_parser().parse_args(argv)
        bgcli.run_command(args)

    def _design_positions_from_feats(self, feats: dict[str, Any], valid_mask: torch.Tensor) -> tuple[int, ...]:
        key = "inverse_fold_design_mask" if "inverse_fold_design_mask" in feats else "design_mask"
        design = feats[key].bool()[valid_mask]
        return tuple(int(i) for i in torch.where(design)[0].tolist())

    # ---------------------------------------------------------------- rollout

    @torch.no_grad()
    def rollout_with_logprobs(
        self,
        case: CapturedCase,
        temperature: float,
        *,
        seed: int | None = None,
        record_events: bool = True,
    ) -> Trajectory:
        """Line-for-line replication of ``InverseFoldingDecoder.sample`` that
        also records per-event log-probs/entropy.  Global RNG only (parity).

        ``feats`` is restored from a pristine copy each call because sample()
        writes ``feats["res_type"]``.
        """
        decoder = self.decoder
        s, z, edge_idx, valid_mask = case.s, case.z, case.edge_idx, case.valid_mask
        feats = copy.deepcopy(case.feats)
        if seed is not None:
            torch.manual_seed(int(seed))

        num_nodes = s.shape[0]
        design_key = "inverse_fold_design_mask" if "inverse_fold_design_mask" in feats else "design_mask"
        design_mask = feats[design_key].bool()[valid_mask]
        num_not_design = (~design_mask).sum().item()
        num_design = design_mask.sum().item()
        assert num_design == num_nodes - num_not_design

        constraint_mask = feats.get("aa_constraint_mask")
        constraint_mask = constraint_mask[valid_mask] if constraint_mask is not None else None
        from boltzgen.model.modules.inverse_fold import build_constraint_logit_mask
        from boltzgen.data import const

        per_residue_mask = build_constraint_logit_mask(
            num_nodes=num_nodes,
            aa_constraint_mask=constraint_mask,
            inverse_fold_restriction=decoder.inverse_fold_restriction,
            canonical_tokens=const.canonical_tokens,
            inf=decoder.inf,
            device=s.device,
        )

        order = torch.randperm(num_nodes, device=s.device).cpu().numpy().tolist()
        if num_not_design > 0:
            id_not_design = torch.where(~design_mask)[0].cpu().numpy().tolist()
            for i in id_not_design:
                order.remove(i)
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            decoded_seq[~design_mask] = logits[~design_mask] = feats["res_type_clone"][valid_mask][~design_mask].float()
        else:
            order = [i for i in order if design_mask[i]]
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)

        if decoder.tie_symmetric_sequences and "symmetric_group" in feats:
            sym_groups, position_to_group = decoder._build_symmetric_groups(feats, valid_mask, design_mask)
            sampled = set(torch.where(~design_mask)[0].cpu().numpy().tolist()) if num_not_design > 0 else set()
        else:
            sym_groups, position_to_group = {}, {}
            sampled = set()

        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        events: list[PolicyEvent] = []
        executed_order: list[int] = []

        for i in order:
            if decoder.tie_symmetric_sequences and i in sampled:
                continue
            if decoder.tie_symmetric_sequences and i in position_to_group:
                positions = tuple(sym_groups[position_to_group[i]])
            else:
                positions = (i,)

            aggregated_logits = None
            for pos in positions:
                s_pos = s[pos : pos + 1]
                edge_mask_pos = dst_idx == pos
                z_pos = z[edge_mask_pos]
                src_idx_pos = src_idx[edge_mask_pos]
                res_type_pos = decoded_seq[src_idx_pos]
                res_rep = decoder.seq_to_s(res_type_pos)
                neighbors_rep_pos = torch.concat([z_pos, s[src_idx_pos] + res_rep], dim=-1)
                s_temp = s_pos
                for layer in decoder.decoder_layers:
                    s_temp = layer.sample(s_temp, neighbors_rep_pos)
                logits_pos = decoder.predictor(s_temp)
                aggregated_logits = logits_pos if aggregated_logits is None else aggregated_logits + logits_pos
            aggregated_logits = aggregated_logits / len(positions)

            pred_canonical = (
                aggregated_logits[:, const.canonicals_offset : len(const.canonical_tokens) + const.canonicals_offset]
                + per_residue_mask[i : i + 1]
            )
            if temperature is None:
                ids_canonical = torch.argmax(pred_canonical, dim=-1)
                log_probs_row = None
            else:
                log_probs_row = F.log_softmax(pred_canonical / temperature, dim=-1)
                probs = log_probs_row.exp()
                ids_canonical = torch.multinomial(probs, num_samples=1).squeeze(-1)

            ids = ids_canonical + const.canonicals_offset
            pred_one_hot = F.one_hot(ids, num_classes=const.num_tokens)
            for pos in positions:
                decoded_seq[pos] = pred_one_hot
                logits[pos] = aggregated_logits
                if decoder.tie_symmetric_sequences:
                    sampled.add(pos)
            executed_order.append(i)
            if record_events and log_probs_row is not None:
                action = int(ids_canonical.item())
                events.append(
                    PolicyEvent(
                        position=i,
                        symmetric_positions=positions,
                        action_canonical_id=action,
                        old_logprob=float(log_probs_row[0, action].item()),
                        old_entropy=float(-(probs * log_probs_row).sum().item()),
                    )
                )

        sequence = _sequence_from_decoded(decoded_seq, const.canonicals_offset)
        n_tokens = valid_mask.shape[1]
        res_type = torch.zeros(1, n_tokens, decoder.num_res_type, device=s.device)
        res_type[valid_mask] = decoded_seq
        feats["res_type"] = res_type.long()
        return Trajectory(
            case_id=case.case_id,
            decode_order=tuple(executed_order),
            events=events,
            final_sequence=sequence,
            temperature=float(temperature),
            seed=int(seed) if seed is not None else -1,
            design_positions=case.design_positions,
        )

    # ----------------------------------------------------------------- replay

    def replay_actions(
        self,
        case: CapturedCase,
        trajectory: Trajectory,
        temperature: float,
        *,
        with_grad: bool,
        decoder: Any | None = None,
    ) -> dict[str, Any]:
        """Teacher-force the recorded (decode_order, actions) through the same
        autoregressive path.  No sampling.  Grad enabled iff ``with_grad``.
        ``decoder`` defaults to the policy decoder; the frozen reference
        decoder is passed explicitly by the trainer (task book §23)."""
        decoder = decoder if decoder is not None else self.decoder
        s, z, edge_idx, valid_mask = case.s, case.z, case.edge_idx, case.valid_mask
        feats = copy.deepcopy(case.feats)
        grad_ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with grad_ctx:
            num_nodes = s.shape[0]
            from boltzgen.data import const
            from boltzgen.model.modules.inverse_fold import build_constraint_logit_mask

            design_key = "inverse_fold_design_mask" if "inverse_fold_design_mask" in feats else "design_mask"
            design_mask = feats[design_key].bool()[valid_mask]
            num_not_design = (~design_mask).sum().item()
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            if num_not_design > 0:
                decoded_seq[~design_mask] = feats["res_type_clone"][valid_mask][~design_mask].float()

            constraint_mask = feats.get("aa_constraint_mask")
            constraint_mask = constraint_mask[valid_mask] if constraint_mask is not None else None
            per_residue_mask = build_constraint_logit_mask(
                num_nodes=num_nodes,
                aa_constraint_mask=constraint_mask,
                inverse_fold_restriction=decoder.inverse_fold_restriction,
                canonical_tokens=const.canonical_tokens,
                inf=decoder.inf,
                device=s.device,
            )
            src_idx, dst_idx = edge_idx[0], edge_idx[1]
            new_logprobs: list[torch.Tensor] = []
            full_logprobs: list[torch.Tensor] = []
            for event in trajectory.events:
                positions = event.symmetric_positions
                aggregated_logits = None
                for pos in positions:
                    s_pos = s[pos : pos + 1]
                    edge_mask_pos = dst_idx == pos
                    z_pos = z[edge_mask_pos]
                    src_idx_pos = src_idx[edge_mask_pos]
                    res_type_pos = decoded_seq[src_idx_pos]
                    res_rep = decoder.seq_to_s(res_type_pos)
                    neighbors_rep_pos = torch.concat([z_pos, s[src_idx_pos] + res_rep], dim=-1)
                    s_temp = s_pos
                    for layer in decoder.decoder_layers:
                        s_temp = layer.sample(s_temp, neighbors_rep_pos)
                    logits_pos = decoder.predictor(s_temp)
                    aggregated_logits = logits_pos if aggregated_logits is None else aggregated_logits + logits_pos
                aggregated_logits = aggregated_logits / len(positions)
                pred_canonical = (
                    aggregated_logits[:, const.canonicals_offset : len(const.canonical_tokens) + const.canonicals_offset]
                    + per_residue_mask[event.position : event.position + 1]
                )
                log_probs_row = F.log_softmax(pred_canonical / temperature, dim=-1)
                action = torch.tensor([[event.action_canonical_id]], device=log_probs_row.device)
                new_logprobs.append(log_probs_row[0, action].squeeze())
                full_logprobs.append(log_probs_row)
                ids = event.action_canonical_id + const.canonicals_offset
                pred_one_hot = F.one_hot(torch.tensor([ids], device=s.device), num_classes=const.num_tokens)
                for pos in positions:
                    decoded_seq[pos] = pred_one_hot
            return {
                "action_logprobs": torch.stack(new_logprobs),
                "full_logprobs": full_logprobs,
                "sequence": _sequence_from_decoded(decoded_seq, const.canonicals_offset),
            }

    # ----------------------------------------------------------------- checks

    def fr_unchanged(self, case: CapturedCase, trajectory: Trajectory, native: str) -> bool:
        design = set(case.design_positions)
        if len(native) != len(trajectory.final_sequence):
            return False
        return all(trajectory.final_sequence[i] == native[i] for i in range(len(native)) if i not in design)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.decoder.parameters() if p.requires_grad]


def _clone_feats(feats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in feats.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def _feats_to_device(feats: dict[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in feats.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out
