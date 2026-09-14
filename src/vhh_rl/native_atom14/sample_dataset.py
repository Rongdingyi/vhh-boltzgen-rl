"""Native sample pool: storage + loading (task book §15).

Layout per case:

    pool/<split>/<case_id>/
      metadata.jsonl          # one row per sample (see fields below)
      coords/<sample_id>.pt   # sampled X0 atom14 coords [N_atom, 3] fp32
      feats_common.pt         # writer-shared feats (masks etc.), one per case
      structures/<sample_id>.cif
      sequences.fasta
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# keys needed for decode re-check + DPO masks (subset of const.atom/token features)
COMMON_FEAT_KEYS = (
    "atom_pad_mask",
    "atom_resolved_mask",
    "atom_to_token",
    "token_index",
    "design_mask",
    "fake_atom_mask",
    "mol_type",
    "token_pad_mask",
    "backbone_mask",
    "res_type",
    "ccd",
    "ref_atom_name_chars",
    "ref_element",
    "ref_pos",
    "ref_charge",
    "ref_chirality",
    "ref_space_uid",
    "new_to_old_atomidx",
    "atom_backbone_feat",
    "is_standard",
    "asym_id",
    "entity_id",
    "sym_id",
)


@dataclass
class NativeSample:
    case_id: str
    sample_id: str
    split: str
    seed: int
    decoded_sequence: str
    contains_invalid: bool
    fr_mismatch: int
    reward_raw: float | None
    coords_path: str
    cif_path: str | None

    def load_coords(self) -> torch.Tensor:
        return torch.load(self.coords_path, map_location="cpu", weights_only=True)


def save_case_pool(
    pool_root: Path,
    split: str,
    case_id: str,
    capture: Any,
    sequence_rows: list[dict[str, Any]],
    reference_sequence: str,
    fr_positions: tuple[int, ...],
    design_positions: tuple[int, ...],
    checkpoint_sha256: str,
    boltzgen_commit: str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write coords/feats/metadata/cif/fasta for one case capture.

    ``sequence_rows`` must align with ``capture.samples`` and carry
    decoded_sequence/contains_invalid/fr_mismatch/reward_raw.
    """
    case_dir = Path(pool_root) / split / case_id
    coords_dir = case_dir / "coords"
    struct_dir = case_dir / "structures"
    coords_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)

    assert len(sequence_rows) == len(capture.samples), (len(sequence_rows), len(capture.samples))
    # shared feats = intersection of captured common feats of the first sample
    common = capture.samples[0].feats_common
    common_keep = {k: v for k, v in common.items() if k in COMMON_FEAT_KEYS}
    torch.save(common_keep, case_dir / "feats_common.pt")

    fasta_lines = []
    with (case_dir / "metadata.jsonl").open("w") as handle:
        for sample, row in zip(capture.samples, sequence_rows):
            sample_id = f"{case_id}_s{sample.sample_index:02d}"
            coords_path = coords_dir / f"{sample_id}.pt"
            torch.save(sample.coords.float(), coords_path)
            cif_target = struct_dir / f"{sample_id}.cif"
            cif_path = sample.cif_path
            if cif_path and Path(cif_path).is_file():
                cif_target.write_bytes(Path(cif_path).read_bytes())
            else:
                cif_target = None
            record = {
                "case_id": case_id,
                "split": split,
                "sample_id": sample_id,
                "sample_index": sample.sample_index,
                "seed": capture.seed,
                "checkpoint_sha256": checkpoint_sha256,
                "boltzgen_commit": boltzgen_commit,
                "sampling_steps": capture.sampling_steps,
                "diffusion_batch_size": capture.diffusion_batch_size,
                "design_positions": list(design_positions),
                "n_design_positions": len(design_positions),
                "decoded_sequence": row["decoded_sequence"],
                "decoded_cdr_sequence": "".join(
                    row["decoded_sequence"][i] for i in sorted(design_positions)
                ),
                "reward_raw": row.get("reward_raw"),
                "reward_optimization": row.get("reward_raw"),
                "contains_UNK": row["contains_invalid"],
                "FR_sequence_match": row["fr_mismatch"] == 0,
                "FR_mismatch_count": row["fr_mismatch"],
                "decode_parity": row.get("decode_parity"),
                "coords_path": str(coords_path),
                "feats_common_path": str(case_dir / "feats_common.pt"),
                "cif_path": str(cif_target) if cif_target else None,
            }
            if extra_meta:
                record.update(extra_meta)
            handle.write(json.dumps(record) + "\n")
            fasta_lines.append(f">{sample_id}\n{row['decoded_sequence']}\n")
    (case_dir / "sequences.fasta").write_text("".join(fasta_lines))
    return case_dir


def load_case_samples(case_dir: Path) -> list[NativeSample]:
    out = []
    for line in (Path(case_dir) / "metadata.jsonl").open():
        r = json.loads(line)
        out.append(
            NativeSample(
                case_id=r["case_id"],
                sample_id=r["sample_id"],
                split=r["split"],
                seed=int(r["seed"]),
                decoded_sequence=r["decoded_sequence"],
                contains_invalid=bool(r["contains_UNK"]),
                fr_mismatch=int(r["FR_mismatch_count"]),
                reward_raw=r["reward_raw"],
                coords_path=r["coords_path"],
                cif_path=r.get("cif_path"),
            )
        )
    return out


def pool_summary(pool_root: Path, split: str) -> dict[str, Any]:
    """Reward pool statistics (task book §19)."""
    import statistics as st

    per_case = {}
    for case_dir in sorted((Path(pool_root) / split).iterdir()):
        if not case_dir.is_dir():
            continue
        samples = load_case_samples(case_dir)
        rewards = [s.reward_raw for s in samples if s.reward_raw is not None]
        n_total = len(samples)
        n_invalid = sum(1 for s in samples if s.contains_invalid)
        n_fr_mismatch = sum(1 for s in samples if s.fr_mismatch > 0)
        uniq = len({s.decoded_sequence for s in samples})
        row = {
            "n_total": n_total,
            "n_scored": len(rewards),
            "invalid_rate": n_invalid / max(n_total, 1),
            "fr_mismatch_rate": n_fr_mismatch / max(n_total, 1),
            "unique_rate": uniq / max(n_total, 1),
        }
        if rewards:
            srt = sorted(rewards, reverse=True)
            q = max(1, len(srt) // 4)
            row.update(
                reward_mean=st.mean(rewards),
                reward_std=st.stdev(rewards) if len(rewards) > 1 else 0.0,
                reward_min=min(rewards),
                reward_max=max(rewards),
                top_quartile_mean=st.mean(srt[:q]),
                bottom_quartile_mean=st.mean(srt[-q:]),
                top_bottom_gap=st.mean(srt[:q]) - st.mean(srt[-q:]),
            )
        per_case[case_id_name(case_dir)] = row
    all_rewards = [v["reward_mean"] for v in per_case.values() if "reward_mean" in v]
    stds = [v["reward_std"] for v in per_case.values() if "reward_std" in v]
    return {
        "per_case": per_case,
        "n_cases": len(per_case),
        "reward_mean_over_cases": st.mean(all_rewards) if all_rewards else None,
        "reward_std_over_cases": st.mean(stds) if stds else None,
        "zero_variance_cases": [c for c, v in per_case.items() if v.get("reward_std", 1.0) < 1e-4],
        "invalid_rate": st.mean([v["invalid_rate"] for v in per_case.values()]) if per_case else 0.0,
        "fr_mismatch_rate": st.mean([v["fr_mismatch_rate"] for v in per_case.values()]) if per_case else 0.0,
    }


def case_id_name(case_dir: Path) -> str:
    return case_dir.name
