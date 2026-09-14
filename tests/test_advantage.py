import torch
from vhh_rl.rl.advantages import group_advantages

def test_rank_sorted_mean_zero():
    adv, _ = group_advantages({"c1": [1.0, 2.0, 3.0, 4.0]}, method="rank")
    assert adv.tolist() == sorted(adv.tolist())
    assert abs(adv.mean().item()) < 1e-6
    assert adv[-1] > adv[0]

def test_rank_ties_average():
    adv, _ = group_advantages({"c": [1.0, 1.0, 2.0]}, method="rank")
    assert adv[0] == adv[1]
    assert abs(adv.mean().item()) < 1e-6

def test_group_z_zero_variance():
    adv, zero = group_advantages({"c": [2.0, 2.0, 2.0]}, method="group_z")
    assert zero == 1 and torch.all(adv == 0)

def test_groups_are_independent():
    adv, _ = group_advantages({"a": [0.0, 1.0], "b": [100.0, 101.0]}, method="group_z")
    assert abs(adv[0].item() - (-1.0)) < 1e-5 and abs(adv[1].item() - 1.0) < 1e-5
    assert abs(adv[2].item() - (-1.0)) < 1e-5
