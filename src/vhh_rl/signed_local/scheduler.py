"""Deterministic G-G-G-L branch scheduler (task book §34/§69)."""
from __future__ import annotations


class ThreeToOneScheduler:
    """step starts from 1; step 4, 8, 12, ... are local (G G G L)."""

    def __init__(self, global_steps: int = 3, local_steps: int = 1) -> None:
        if global_steps < 1 or local_steps < 1:
            raise ValueError("cycle sizes must be >= 1")
        self.global_steps = int(global_steps)
        self.local_steps = int(local_steps)
        self.cycle = self.global_steps + self.local_steps

    def branch(self, step: int) -> str:
        if step < 1:
            raise ValueError("step starts from 1")
        return "local" if step % self.cycle == 0 else "global"

    def counts(self, total_steps: int) -> dict[str, int]:
        local = sum(1 for s in range(1, total_steps + 1) if self.branch(s) == "local")
        return {"global": total_steps - local, "local": local}
