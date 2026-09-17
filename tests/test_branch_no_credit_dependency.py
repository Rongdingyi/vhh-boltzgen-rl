"""Runtime branch_distill code must not read counterfactual credit (task book §58)."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src/vhh_rl/branch_distill"
FORBIDDEN_NAMES = {"c_drop", "c_gain", "c_cons", "credit_weights",
                   "target_builder", "residue_credit"}
FORBIDDEN_MODULES = ("vhh_rl.cf_opsd.credit", "vhh_rl.cf_opsd.target_builder",
                     "vhh_rl.cf_opsd.onpolicy_trainer")


def _code_identifiers(path: Path):
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
    return names, modules


def test_no_forbidden_code_dependencies():
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        names, modules = _code_identifiers(path)
        offenders += [f"{path.name}: {n}" for n in sorted(names & FORBIDDEN_NAMES)]
        offenders += [f"{path.name}: {m}" for m in sorted(modules)
                      if any(f in m for f in FORBIDDEN_MODULES)]
    assert not offenders, offenders


def test_no_cf_opsd_target_or_credit_imports():
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text()
        assert "from ..cf_opsd.target_builder" not in text
        assert "from ..cf_opsd.credit" not in text
        assert "cf_opsd.onpolicy_trainer" not in text
