import torch
from vhh_rl.rl.losses import exact_categorical_kl

def test_same_distribution_kl_zero():
    lp = [torch.log_softmax(torch.randn(1, 20), dim=-1)]
    assert exact_categorical_kl(lp, [l.clone() for l in lp]).item() < 1e-6

def test_shifted_distribution_kl_positive():
    a = [torch.log_softmax(torch.randn(1, 20), dim=-1)]
    b = [torch.log_softmax(torch.randn(1, 20), dim=-1)]
    assert exact_categorical_kl(a, b).item() > 0

def test_masked_entries_no_nan():
    logits = torch.full((1, 20), -1e9)
    logits[0, 5] = 0.0
    lp = [torch.log_softmax(logits, dim=-1)]
    kl = exact_categorical_kl(lp, [torch.log_softmax(torch.randn(1, 20), dim=-1)])
    assert torch.isfinite(kl)
