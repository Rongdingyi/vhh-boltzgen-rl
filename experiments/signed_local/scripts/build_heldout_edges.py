#!/usr/bin/env python
"""Held-out local preference edges (task book §51): top1-vs-bottom1 + scorer queries."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))
from vhh_rl.cli.common import case_spec_for  # noqa: E402
from vhh_rl.data.case import RLCase  # noqa: E402
from vhh_rl.signed_local.edge_builder import build_edges  # noqa: E402
from vhh_rl.cf_opsd.credit import pair_credit  # noqa: E402
from vhh_rl.native_atom14.reward import make_reward_adapter  # noqa: E402

HELDOUT = ["sab2_4hf5_h", "sab2_4mwf_h", "sab2_5mp6_h", "sab2_6cvk_b2"]
OUT = ROOT / "runs/signed_local/edges"


def main() -> None:
    cases = {}
    for line in (ROOT / "runs/round1_rl_split/rl_manifest_split.jsonl").open():
        row = json.loads(line)
        if row.get("split") != "test" or row["case_id"] not in HELDOUT:
            continue
        cases[row["case_id"]] = RLCase(
            case_id=row["case_id"], structure_path=Path(row["structure_path"]),
            chain_id=row.get("chain_id", "A"), full_sequence=row["full_sequence"],
            design_positions=tuple(row["design_positions"]),
            fr_positions=tuple(row["fr_positions"]), split="test",
            seed_base=int(row.get("seed_base", 0)))
    scorer = make_reward_adapter(cache_path=ROOT / "runs/signed_local/reward_cache.sqlite")
    pairs, credit, cf_rewards = [], {}, {}
    for cid, case in sorted(cases.items()):
        rows = [json.loads(l) for l in (ROOT / "runs/native_pool/heldout" / cid
                                        / "metadata.jsonl").open()]
        rows = [r for r in rows if r["reward_raw"] is not None
                and not r["contains_UNK"] and r["FR_mismatch_count"] == 0]
        if len(rows) < 2:
            continue
        ranked = sorted(rows, key=lambda r: r["reward_raw"], reverse=True)
        w, l = ranked[0], ranked[-1]
        pid = f"{cid}:{w['sample_id']}:{l['sample_id']}"
        pairs.append({"case_id": cid, "winner_sample_id": w["sample_id"],
                      "loser_sample_id": l["sample_id"]})
        pc = pair_credit(scorer, case, w["sample_id"], l["sample_id"],
                         w["decoded_sequence"], l["decoded_sequence"],
                         case.design_positions, case.fr_positions)
        credit[pid] = {r["position"]: (r["c_drop"], r["c_gain"], r["c_cons"])
                       for r in pc["detail"]}
        for pos, (c_drop, c_gain, _c) in credit[pid].items():
            cf_rewards[f"{pid}:winner_drop:{pos}"] = pc["winner_reward"] - c_drop
            cf_rewards[f"{pid}:loser_gain:{pos}"] = pc["loser_reward"] + c_gain
    scorer.close()
    summary = build_edges(cases, pairs, credit, cf_rewards, out_dir=OUT, split="heldout")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
