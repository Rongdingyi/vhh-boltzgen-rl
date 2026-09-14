"""Local frame rotation/translation invariance (task book §23)."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path("/share/home/rongdingyi/programs/proteingen/vhh_boltzgen_rl")
sys.path.insert(0, str(ROOT / "src"))

from vhh_rl.cf_opsd.local_frame import frame_from_backbone, to_local, to_world


def _random_rotation(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_frame_transfer_invariance():
    g = torch.Generator().manual_seed(0)
    coords = torch.randn(12, 3, generator=g)
    n, ca, c = 0, 1, 2
    fake = torch.tensor([6, 7, 8])
    F, o = frame_from_backbone(coords, n, ca, c)
    local_ref = to_local(coords[fake], F, o)
    worst = 0.0
    for seed in range(5):
        R = _random_rotation(seed)
        t = torch.randn(3, generator=torch.Generator().manual_seed(100 + seed)) * 5
        moved = coords @ R.T + t
        F2, o2 = frame_from_backbone(moved, n, ca, c)
        local2 = to_local(moved[fake], F2, o2)
        worst = max(worst, float((local2 - local_ref).abs().max()))
    assert worst < 1e-4, worst
    restored = to_world(local_ref, F, o)
    assert float((restored - coords[fake]).abs().max()) < 1e-5
