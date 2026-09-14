"""CLI entry: vhh-rl {audit,baseline,train,evaluate} (thin wrappers)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vhh-rl")
    parser.add_argument("command", choices=("audit", "baseline", "train", "evaluate"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args, extra = parser.parse_known_args(argv)

    from vhh_rl.config import load_config

    cfg = load_config(args.config)
    if args.command == "audit":
        sys.exit(subprocess_audit())
    if args.command == "baseline":
        from vhh_rl.cli.baseline import run_baseline

        sys.exit(run_baseline(cfg, Path(args.output_dir or cfg.resolved_output_dir())))
    if args.command == "train":
        from vhh_rl.cli.train import run_train

        sys.exit(run_train(cfg, Path(args.output_dir or cfg.resolved_output_dir())))
    if args.command == "evaluate":
        from vhh_rl.cli.evaluate import run_evaluate

        sys.exit(run_evaluate(cfg, Path(args.output_dir or cfg.resolved_output_dir())))
    return 0


def subprocess_audit() -> int:
    import subprocess

    return subprocess.call(["bash", str(ROOT / "scripts" / "audit_repo.sh"), str(ROOT / "docs/AUDIT.md")])


if __name__ == "__main__":
    main()
