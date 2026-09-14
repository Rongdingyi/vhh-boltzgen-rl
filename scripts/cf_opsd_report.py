#!/usr/bin/env python
"""Assemble docs/cf_opsd/CF_OPSD_DECISION.md from the gate artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
RUNS = ROOT / "runs/cf_opsd"
DOC = ROOT / "docs/cf_opsd/CF_OPSD_DECISION.md"


def jload(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> None:
    query = jload(RUNS / "query_probe.json")
    gate_b = jload(RUNS / "target_probe/gate_b.json")
    gate_c = jload(RUNS / "realization_probe/gate_c.json")
    gate_d = jload(RUNS / "pilot_eval/gate_d.json")
    pilot = jload(RUNS / "pilot_eval/pilot_eval.json")

    lines = ["# CF-OPSD Feasibility Decision\n"]
    lines.append("## 1. Motivation")
    lines.append("把 counterfactual residue credit 转成显式 atom14 几何 target，"
                 "以 on-policy self-distillation 替代/补充 CF-DPO。\n")
    lines.append("## 2. Reference Difference")
    lines.append("DiffusionOPSD 依赖可微 reward gradient；BoltzGen 的 reward 经由"
                 "硬 `res_from_atom14` 得到离散序列，不可微。本探索用 black-box"
                 " counterfactual credit 构造 bounded positive target。\n")
    lines.append("## 3. Query Audit")
    if query:
        lines.append(f"- q* = {query['q_star_progress']:.0%}，status = {query['status']}")
    else:
        lines.append("- (未运行)")
    lines.append("\n## 4. Target Construction")
    if gate_b:
        for r in gate_b["radii"]:
            lines.append(f"- rho={r['radius']}: gain median "
                         f"{r['median_target_minus_anchor']}, match {r['median_credit_match']}, "
                         f"invalid {r['invalid_rate']:.2f}, gate={gate_b['gate_b'].get(str(r['radius']))}")
        lines.append(f"- selected radius: {gate_b['selected_radius']}")
    else:
        lines.append("- (未运行)")
    lines.append("\n## 5. Same-query Realization")
    if gate_c:
        for v, ok in gate_c["gate_c"].items():
            lines.append(f"- {v}: Gate C = {'PASS' if ok else 'FAIL'}")
    else:
        lines.append("- (未运行)")
    lines.append("\n## 6. Static Pilot")
    if pilot:
        for tag, res in pilot.items():
            lines.append(f"- {tag}: held-out reward {res.get('reward_mean')}")
    if gate_d:
        lines.append(f"- gate_d: {json.dumps(gate_d, default=str)}")
    if not pilot:
        lines.append("- (未运行)")
    lines.append("\n## 7. On-policy Pilot")
    lines.append("- 仅在 Gate D PASS 后运行；见 runs/cf_opsd/onpolicy_v1/。\n")
    lines.append("## 8. Efficiency")
    lines.append("- optimizer updates / scorer queries 统计见各 probe 产物。\n")
    lines.append("## 9. Failure Modes")
    lines.append("- 详见各阶段文档与 gate JSON。\n")

    b_ok = bool(gate_b and gate_b.get("selected_radius") is not None)
    c_ok = bool(gate_c and any(gate_c["gate_c"].values()))
    d_ok = bool(gate_d and any((v or {}).get("reward_superiority") for v in gate_d.values()))
    if b_ok and c_ok and d_ok:
        decision = "### GO\nCF-OPSD 在静态阶段达到 Gate D 的 reward superiority 条件。"
    elif b_ok and c_ok:
        decision = ("### NO-GO（当前规模）\nGate B/C 通过但 Gate D 未达到 ≥1.10× reward "
                    "或 ≥30% updates 优势；按 §54/§68 停止 OPSD，回到 CF-DPO 主线。")
    elif b_ok and not c_ok:
        decision = "### NO-GO\nTarget 可诊断但 student 同-query 学不进去（Gate C FAIL，§44）。"
    elif not b_ok and query:
        decision = "### NO-GO\nTarget 无法从 counterfactual credit 稳定构造（Gate B FAIL，§32）。"
    else:
        decision = "### 待定\n部分阶段未运行。"
    lines.append(f"## 10. Final Decision\n\n{decision}\n")
    lines.append("## 11. Recommendation")
    lines.append("- keep CF-DPO main line（除非上表 GO）")
    lines.append("- 不实现 soft decoder / negative target / structural critic（§5/§65/§95）")
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC}")
    print(decision)


if __name__ == "__main__":
    main()
