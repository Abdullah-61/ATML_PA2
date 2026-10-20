import math, torch
from task3_grpo.grpo import (group_relative_advantages, grpo_loss_parts, grpo_policy_loss,
                             mask_truncated_sequences, group_reward_stats, ADV_EPS)


def test_group_advantages_are_per_group():
    r = torch.tensor([1., 2., 3., 10., 20., 30.]); g = torch.tensor([0, 0, 0, 1, 1, 1])
    a = group_relative_advantages(r, g)
    assert torch.allclose(a[:3], a[3:], atol=1e-4)                      # same standardised shape in each group
    assert torch.allclose(a[:3], torch.tensor([-1.2247, 0., 1.2247]), atol=1e-3)
    assert abs(float(a[:3].mean())) < 1e-6 and abs(float(a[3:].mean())) < 1e-6


def test_constant_group_is_uninformative():
    r = torch.tensor([1., 1., 1., 1., 0., 2., 0., 2.]); g = torch.tensor([0]*4 + [1]*4)
    a = group_relative_advantages(r, g)
    assert torch.all(a[:4] == 0)
    _, inf = group_reward_stats(r, g); assert inf.tolist() == [False, True]


def test_clip_is_pessimistic_min():
    new = torch.tensor([[math.log(2.0)]], requires_grad=True); old = torch.zeros(1, 1)
    pt, kl, d = grpo_loss_parts(new, old, torch.tensor([1.0]), torch.ones(1, 1), old, 0.2)
    assert math.isclose(float(pt), -1.2, rel_tol=1e-5)
    pt.backward(); assert float(new.grad) == 0.0
    pt, _, _ = grpo_loss_parts(torch.tensor([[math.log(2.0)]]), old, torch.tensor([-1.0]), torch.ones(1, 1), old, 0.2)
    assert math.isclose(float(pt), 2.0, rel_tol=1e-5)


def test_on_policy_ratio_one_no_clipping():
    lp = torch.randn(2, 5)
    _, d = grpo_policy_loss(lp, lp.clone(), torch.tensor([1., -1.]), torch.ones(2, 5), lp.clone(), 0.2, 0.1)
    assert float(d["clip_fraction"]) == 0.0 and math.isclose(float(d["sampled_kl"]), 0.0, abs_tol=1e-6)


def test_length_normalisation():
    # same per-token objective (A=1, ratio=1) on a 2-token and a 4-token completion
    lp = torch.zeros(2, 4); m = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]]); adv = torch.ones(2)
    pt_c, _, _ = grpo_loss_parts(lp, lp, adv, m, lp, 0.2, "grpo")
    pt_d, _, _ = grpo_loss_parts(lp, lp, adv, m, lp, 0.2, "dr_grpo", 8)
    assert math.isclose(float(pt_c), -1.0, rel_tol=1e-6)               # 1/T_k: both sequences count equally
    assert math.isclose(float(pt_d), -(2 / 8 + 4 / 8) / 2, rel_tol=1e-6)  # constant: long sequence counts 2x


def test_truncated_completion_is_masked():
    m = mask_truncated_sequences(torch.ones(2, 3), [False, True])
    assert m.sum(-1).tolist() == [3.0, 0.0]


def test_microbatch_accumulation_equals_batch_loss():
    torch.manual_seed(0)
    N, T, beta, eps = 4, 6, 0.1, 0.2
    new, old, ref = torch.randn(N, T) * .1, torch.randn(N, T) * .1, torch.randn(N, T) * .1
    adv = torch.randn(N)
    m = (torch.rand(N, T) > 0.3).float(); m[2] = 0.0
    for lt in ("grpo", "dr_grpo"):
        full, _ = grpo_policy_loss(new, old, adv, m, ref, eps, beta, lt, 10)
        total, n_tok = 0.0, m.sum()
        for i in range(N):
            pt, kl, _ = grpo_loss_parts(new[i:i+1], old[i:i+1], adv[i:i+1], m[i:i+1], ref[i:i+1], eps, lt, 10)
            total += float(pt) / N + beta * float(kl) * float(m[i].sum() / n_tok)
        assert math.isclose(float(full), total, rel_tol=1e-5, abs_tol=1e-7)

