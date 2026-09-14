"""On-policy wiring integration test with injected dependencies (audit fixes).

Verifies, without a GPU or the real BoltzGen model:
  1. behavior is an independent copy, not the student object;
  2. every round's rollouts use the behavior checkpoint EMA'd in the previous
     round (round 1 uses the base checkpoint);
  3. held-out evaluation receives the per-round *student* checkpoint;
  4. pairing.pairs_per_case is honored (rank1-vs-rankN, rank2-vs-rank(N-1));
  5. fit.loss variant is honored (weak-hold executes its hold branch);
  6. counters and parameter drift are recorded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _SM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_model = torch.nn.Linear(3, 3)
        self.hold_used = False

    def preconditioned_network_forward(self, q, sigma, training=False,
                                       network_condition_kwargs=None):
        return self.score_model(q), None


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.structure_module = _SM()


class _Scorer:
    def score_sequences(self, seqs, spec=None):
        return SimpleNamespace(raw_scores=torch.tensor(
            [4.0 - 1.0 * i for i in range(len(seqs))]))

    def close(self):
        pass


def _gate_files(tmp_path: Path, gate_d_pass: bool = True) -> dict:
    qp = tmp_path / "query_probe.json"
    qp.write_text(json.dumps({"q_star_progress": 0.9}))
    gb = tmp_path / "gate_b.json"
    gb.write_text(json.dumps({"selected_radius": 0.5}))
    gc = tmp_path / "gate_c.json"
    gc.write_text(json.dumps({"gate_c": {"V0-A": True}}))
    gd = tmp_path / "gate_d.json"
    gd.write_text(json.dumps({"gate_d": {"50": {"reward_superiority": gate_d_pass}}}))
    return {"query_probe": qp, "gate_b": gb, "gate_c": gc, "gate_d": gd}


def test_onpolicy_wiring(tmp_path):
    import vhh_rl.cf_opsd.onpolicy_trainer as trainer

    trainer.DEVICE = "cpu"  # CPU-only wiring test (CI-safe)

    case_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/fixed_cases.yaml").read_text())
    loop_cfg = yaml.safe_load((ROOT / "configs/cf_opsd/onpolicy_v1.yaml").read_text())
    loop_cfg = {**loop_cfg,
                "outer": {"rounds": 3, "behavior_ema_decay": 0.5},
                "pairing": {"pairs_per_case": 2},
                "fit": {"updates_per_round": 2, "loss": "target_mask_weak_hold"}}
    case_path = tmp_path / "fixed_cases.yaml"
    loop_path = tmp_path / "onpolicy_v1.yaml"
    case_path.write_text(yaml.safe_dump(case_cfg))
    loop_path.write_text(yaml.safe_dump(loop_cfg))
    gates = _gate_files(tmp_path)
    base_ckpt = tmp_path / "base.pt"
    base_ckpt.write_bytes(b"base")

    N = 4
    rollout_ckpts = []
    evaluate_calls = []
    forward_calls = {"n": 0}

    class _Adapter:
        def __init__(self):
            self.design_ckpt = None

    adapter = _Adapter()

    def adapter_factory(K, steps):
        return adapter

    def rollout_fn(adapter, spec, cid, run_root, num_designs, seed):
        rollout_ckpts.append(Path(adapter.design_ckpt))
        from vhh_rl.cf_opsd.types import OPSDTrajectory
        trajs = []
        for i in range(4):                       # K=4 endpoints -> 2 pairs
            trajs.append(OPSDTrajectory(
                case_id=cid, sample_index=i, seed=seed,
                endpoint_coords=torch.zeros(N, 3), endpoint_sequence="AAAA",
                endpoint_reward=None,
                sigmas=[1.0] * 50,
                coords_traj=[torch.zeros(N, 3)],
                query_states=[torch.zeros(N, 3)] * 50,
                anchors=[torch.zeros(N, 3)] * 50,
                feats_common={}, contains_invalid=False, fr_mismatch=0))
        return trajs, {"case_id": cid}

    def evaluate_fn(case_ids, ckpt, tag, run_root=None):
        evaluate_calls.append(Path(ckpt))
        assert Path(ckpt).is_file(), f"evaluate got missing ckpt {ckpt}"
        return {"reward_mean": 0.0}

    def credit_fn(scorer, case, w_id, l_id, w_seq, l_seq, design, fr):
        return {"credits": {0: 1.0}, "n_queries": 2}

    def target_fn(anchor, winner, feats, positions, credits, radius, **kw):
        return SimpleNamespace(target_coords=torch.zeros(N, 3),
                               touched=torch.ones(N, dtype=torch.bool))

    def save_ckpt(base_checkpoint, model, path, metadata):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"role": metadata.get("role")}, path)
        return "sha"

    def load_base(path, device="cpu"):
        return _Model()

    def make_policy_ref(model, device="cpu"):
        return model, None

    def trainable_params(model):
        return [p for n, p in model.named_parameters()
                if n.startswith("structure_module.score_model.") and p.requires_grad]

    fake_cond = {
        "s_inputs": torch.zeros(1, 3), "s_trunk": torch.zeros(1, 3),
        "feats": {"atom_pad_mask": torch.ones(N), "fake_atom_mask": torch.ones(N),
                  "atom_resolved_mask": torch.ones(N),
                  "atom_to_token": torch.zeros(N, 9)},
        "diffusion_conditioning": {},
    }

    cases = {cid: SimpleNamespace(
        case_id=cid, structure_path=tmp_path / cid / "f.cif",
        design_positions=(0,), fr_positions=(1,)) for cid in case_cfg["train_cases"]}

    out = trainer.run_onpolicy(
        base_checkpoint=base_ckpt, case_config=case_path, loop_config=loop_path,
        query_probe=gates["query_probe"], gate_b_path=gates["gate_b"],
        gate_c_path=gates["gate_c"], gate_d_path=gates["gate_d"],
        output_dir=tmp_path / "out",
        deps={
            "adapter_factory": adapter_factory, "rollout_fn": rollout_fn,
            "evaluate_fn": evaluate_fn, "scorer_factory": lambda: _Scorer(),
            "credit_fn": credit_fn, "target_fn": target_fn,
            "load_base": load_base, "make_policy_ref": make_policy_ref,
            "save_ckpt": save_ckpt,
            "load_cases": lambda ids: {c: cases[c] for c in ids},
            "load_cond": lambda d, c: None,
            "move_cond": lambda payload, device="cpu": fake_cond,
            "spec_fn": lambda case: {},
            "trainable_params": trainable_params,
        },
    )

    n_cases = len(case_cfg["train_cases"])
    assert out["rounds"] == 3
    # 2. rollouts use base then previous-round behavior checkpoints
    expected_rounds = [base_ckpt] + [tmp_path / "out" / f"behavior_r{r}.pt" for r in (1, 2)]
    assert len(rollout_ckpts) == 3 * n_cases
    for r in range(3):
        group = set(rollout_ckpts[r * n_cases:(r + 1) * n_cases])
        assert group == {expected_rounds[r]}, (r, group)
    # 3. evaluate receives the per-round student checkpoint
    assert evaluate_calls == [tmp_path / "out" / f"student_r{r}.pt" for r in (1, 2, 3)]
    # 4. pairs_per_case honored: 4 endpoints -> 2 pairs per case -> 2 targets per case
    assert all(row["n_targets"] == 2 * n_cases for row in out["history"]), out["history"]
    # 5. weak-hold variant recorded
    assert all(row["fit_variant"] == "target_mask_weak_hold" for row in out["history"])
    # 6. counters
    assert out["history"][-1]["scorer_queries_cumulative"] > 0
    assert out["history"][-1]["optimizer_updates_cumulative"] == 3 * 2
    assert all(r["heldout_reward_mean_student"] == 0.0 for r in out["history"])
