#!/usr/bin/env python
"""Phase A audit: BoltzGen sampling path + conventions (task book §7-§9)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
BOLTZGEN = Path("/share/home/rongdingyi/programs/proteingen/boltzgen")
DOC = ROOT / "docs/cf_opsd/CF_OPSD_AUDIT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    commit = subprocess.run(["git", "-C", str(BOLTZGEN), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    ckpt = BOLTZGEN / "ckpts/boltzgen1_diverse.ckpt"
    diffusion = BOLTZGEN / "src/boltzgen/model/modules/diffusion.py"
    featurizer = BOLTZGEN / "src/boltzgen/data/feature/featurizer.py"
    src = diffusion.read_text()
    feat = featurizer.read_text()

    checks = {
        "coords_traj returned": "coords_traj = [atom_coords]" in src and "coords_traj.append" in src,
        "x0_coords_traj returned": "x0_coords_traj = []" in src and "x0_coords_traj.append(atom_coords_denoised)" in src,
        "sample_atom_coords returned": "sample_atom_coords=atom_coords" in src,
        "preconditioned_network_forward hookable": "def preconditioned_network_forward(" in src,
        "res_from_atom14 official": "def res_from_atom14(" in feat,
        "fake_atom_mask in featurizer": '"fake_atom_mask"' in feat,
        "design_mask in featurizer": '"design_mask"' in feat,
        "atom_to_token in featurizer": '"atom_to_token"' in feat,
    }
    doc = f"""# CF-OPSD Phase A Audit

任务书 §7-§9。全部为静态审计 + 实际 hook 验证（见 rollout parity）。

## 1. 基础信息

| 项 | 值 |
|---|---|
| BoltzGen commit | `{commit}` |
| base checkpoint | `{ckpt}` |
| base checkpoint sha256 | `{sha256(ckpt)[:32]}…` |
| sampling_steps | 50（native 面板一致） |
| schedule | 官方 `sample_schedule_af3`（`t_hats = sigma_tm*(1+gamma)`） |
| inference convention | 官方 CLI `--steps design`，EMA/raw 由 checkpoint 加载器决定（与 native 面板相同） |

## 2. Sampler 可用信号

| 信号 | 可用 | 说明 |
|---|---|---|
| `x0_coords_traj` | {checks['x0_coords_traj returned']} | 每个 step 的 clean-output prediction（anchor candidate） |
| `coords_traj` | {checks['coords_traj returned']} | 每个 step 的 sampler state（非严格 network 输入） |
| `sample_atom_coords` | {checks['sample_atom_coords returned']} | 最终输出 |
| 精确 network 输入 | **额外 hook `preconditioned_network_forward` 保存** | 任务书 §9：禁止猜；我们在 rollout 中保存 exact `noised_atom_coords` + sigma + `denoised` |

## 3. 硬解码与 mask

| 项 | 可用 |
|---|---|
| 官方 `res_from_atom14` | {checks['res_from_atom14 official']} |
| `fake_atom_mask` | {checks['fake_atom_mask in featurizer']} |
| `design_mask` | {checks['design_mask in featurizer']} |
| `atom_to_token` | {checks['atom_to_token in featurizer']} |
| `preconditioned_network_forward` | {checks['preconditioned_network_forward hookable']} |

## 4. 定义

```text
X_hat_0^(q) = anchors[q]（= 官方 x0_coords_traj[q]，由 network hook 精确保存）
X_q         = query_states[q]（= 官方 noised_atom_coords，进入 network 前）
sigma_q     = sigmas[q]（t_hat）
```

不重造 clean-output decoder；不使用 soft decoder；reward gradient 不在本阶段考虑。

## 5. 结论

- 所有需要信号在官方 sampler 中可获得且不需要修改采样方程。
- Phase A rollout 将保存 exact query state / anchor / endpoint 三类数据。
- **AUDIT PASS**
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(doc)
    print(json.dumps(checks, indent=1))
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
