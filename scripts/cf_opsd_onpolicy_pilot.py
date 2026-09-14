#!/usr/bin/env python
"""Phase E: on-policy EMA refresh pilot (only if Gate D passes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.onpolicy_trainer import run_onpolicy  # noqa: E402

BASE = Path("/share/home/rongdingyi/programs/proteingen/boltzgen/ckpts/boltzgen1_diverse.ckpt")


def main() -> None:
    result = run_onpolicy(
        base_checkpoint=BASE,
        config_path=ROOT / "configs/cf_opsd/fixed_cases.yaml",
        query_probe=ROOT / "runs/cf_opsd/query_probe.json",
        target_probe=ROOT / "runs/cf_opsd/target_probe/gate_b.json",
        output_dir=ROOT / "runs/cf_opsd/onpolicy_v1",
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
