"""SL-CF-DPO: Signed Local Counterfactual Diffusion-DPO (next algorithm task book).

Global branch = frozen CF-DPO (native_atom14.global_cf_step).
Local branch  = context-specific signed local preference pairs, target-residue
fake-atom Diffusion-DPO.  Reward is used for orientation/threshold only.
"""
from .types import (  # noqa: F401
    TOL, LocalPreferenceEdge, classify_event, preferred_side, robust_sign,
)
from .scheduler import ThreeToOneScheduler  # noqa: F401
from .local_dpo import signed_local_dpo_loss  # noqa: F401
from .local_mask import target_residue_mask  # noqa: F401
