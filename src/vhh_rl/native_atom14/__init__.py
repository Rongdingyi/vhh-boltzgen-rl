"""Native atom14 design post-training (task book:
BOLTZGEN_NATIVE_ATOM14_CDR_REWARD_POSTTRAIN_TASK.md).

Isolated from the IF-GRPO branch (`vhh_rl.rl`) which is kept untouched.
"""
from .adapter import NativeDesignAdapter, CapturedSample, Capture
from .decode import decode_atom14, sequence_from_feat, fr_check, decoded_cdr
from .masks import atom_design_mask, cdr_fake_atom_mask, anchor_mask
from .sample_dataset import NativeSample, save_case_pool, load_case_samples, pool_summary

__all__ = [
    "NativeDesignAdapter",
    "CapturedSample",
    "Capture",
    "decode_atom14",
    "sequence_from_feat",
    "fr_check",
    "decoded_cdr",
    "atom_design_mask",
    "cdr_fake_atom_mask",
    "anchor_mask",
    "NativeSample",
    "save_case_pool",
    "load_case_samples",
    "pool_summary",
]
