#!/usr/bin/env python
"""P0 regression check: query accounting fields and geometry-free main path."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import _common as C  # noqa: E402

FORBIDDEN_NAMES = {"carrier_audit", "verified_changed_positions", "c_drop",
                   "c_gain", "c_cons", "residue_credit"}
FORBIDDEN_MODULES = ("vhh_rl.branch_distill.local_target", "vhh_rl.cf_opsd.local_frame")


def main() -> None:
    src = C.ROOT / "src/vhh_rl/online_pref"
    offenders = []
    for path in sorted(src.glob("*.py")):
        tree = ast.parse(path.read_text())
        names, modules = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        offenders += [f"{path.name}: {n}" for n in sorted(names & FORBIDDEN_NAMES)]
        offenders += [f"{path.name}: {m}" for m in sorted(modules)
                      if m in FORBIDDEN_MODULES]
    train_src = (C.ROOT / "experiments/branch_distill/scripts/gate3_train.py").read_text()
    report_src = (C.ROOT / "experiments/branch_distill/scripts/gate3_report.py").read_text()
    accounting_ok = ("reward_queries_round" in train_src
                     and "pair_records_round" in train_src
                     and "reward_queries_cumulative" in report_src)
    payload = {"geometry_free_main_path": not offenders, "offenders": offenders,
               "query_accounting_fixed": accounting_ok}
    C.write_json(C.PHASE0 / "p0_check.json", payload)
    print(json.dumps(payload, indent=1))
    if offenders or not accounting_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
