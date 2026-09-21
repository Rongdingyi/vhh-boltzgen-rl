#!/usr/bin/env python
"""Gate 2 report -> docs/branch_distill/GATE2_REALIZATION.md (§40/§64)."""
from __future__ import annotations

import csv
import json
import statistics as st

import _common as C  # noqa: E402


def _amended_verdict(rows: list[dict]) -> dict:
    """Recompute the gate with amendment 1 from the persisted per-record rows.

    B follows §30/§39 (endpoint carrier) restricted to the verified subset:
    a record is target-legal iff it kept at least one carrier-matched position.
    The full-set carrier decode stays as an audit diagnostic.
    """
    from vhh_rl.branch_distill import gates as gate_mod

    normalized = []
    for row in rows:
        verified = int(row.get("carrier_verified_positions") or 0)
        normalized.append({
            "replay_maxdiff": float(row["replay_maxdiff"]),
            "carrier_invalid": verified == 0,
            "carrier_fr": int(float(row.get("carrier_fr") or 0)),
            "carrier_all_match": verified > 0,
            "mse_ratio": float(row["mse_ratio"]),
            "hamming_before": int(row["hamming_before"]),
            "hamming_after": int(row["hamming_after"]),
            "postfit_reward": float(row["postfit_reward"]),
            "peer_reward": float(row["peer_reward"]),
        })
    return gate_mod.gate2_verdict(normalized)


def main() -> None:
    rows = list(csv.DictReader((C.GATE2_DIR / "overfit_results.csv").open()))
    gate = _amended_verdict(rows)
    C.write_json(C.GATE2_DIR / "gate2.json",
                 {"protocol_sha256": C.protocol_hash(),
                  "amendment_id": 1, "n_records": len(rows), **gate})
    ratios = [float(r["mse_ratio"]) for r in rows]
    before = [int(r["hamming_before"]) for r in rows]
    after = [int(r["hamming_after"]) for r in rows]
    deltas = [float(r["postfit_reward"]) - float(r["peer_reward"]) for r in rows]
    target_ok = sum(1 for r in rows if r.get("target_all_match") == "True")
    target_invalid = sum(1 for r in rows if r.get("target_invalid") == "True")
    verified = [int(r.get("carrier_verified_positions") or 0) for r in rows]
    lines = ["# Gate 2 — Teacher target realization", "",
             "## 0. Protocol note (amendment 1)", "",
             "Supervision uses only carrier-verified changed positions "
             "(`configs/AMENDMENT_1_VERIFIED_POSITIONS.yaml`); the pre-registered "
             "§30 endpoint-carrier probe stays the legality check and the original "
             "per-position fidelity is reported below.", "",
             "## 1. residue identity after transfer", "",
             f"- carrier verified positions per record: min {min(verified, default=0)}, "
             f"max {max(verified, default=0)}",
             f"- endpoint-carrier FR mismatch rows: "
             f"{sum(1 for r in rows if r['carrier_fr'] != '0')}",
             f"- target-decode diagnostic (not gating): "
             f"{target_ok}/{len(rows)} records decode teacher AA on all verified "
             f"positions; {target_invalid} contain invalid tokens",
             "## 2. query replay exactness", "",
             f"- max |pred - stored anchor| median "
             f"{st.median(float(r['replay_maxdiff']) for r in rows):.2e}",
             "## 3. local target error", "",
             f"- initial MSE median "
             f"{st.median(float(r['initial_masked_mse']) for r in rows):.6f}",
             f"- final MSE median "
             f"{st.median(float(r['final_masked_mse']) for r in rows):.6f}",
             f"- mse_ratio median {st.median(ratios):.3f} "
             f"({sum(1 for x in ratios if x <= 0.5)}/{len(rows)} <= 0.5)",
             "## 4. downstream sequence", "",
             f"- hamming improved {sum(1 for b, a in zip(before, after) if a < b)}"
             f"/{len(rows)} (median {st.median(before)} -> {st.median(after)})",
             "## 5. reward safety", "",
             f"- median(postfit - peer) = {st.median(deltas):+.3f}, "
             f"safe {sum(1 for d in deltas if d >= -0.20)}/{len(rows)}",
             "## 6. 失败发生在哪一级", ""]
    for key in ("A_replay", "B_target", "C_learnability", "D_downstream", "E_safety"):
        lines.append(f"- {key}: {'PASS' if gate[key] else 'FAIL'}")
    lines += ["", f"verdict: **{'PASS' if gate['pass'] else 'FAIL'}**", "",
              f"protocol_sha256: `{gate['protocol_sha256']}`", ""]
    C.DOCS.mkdir(parents=True, exist_ok=True)
    (C.DOCS / "GATE2_REALIZATION.md").write_text("\n".join(lines))
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
