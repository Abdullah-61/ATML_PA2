import math, torch
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards, normalize_advantages


def test_clip_is_pessimistic_min():
    eps = 0.2
    new = torch.tensor([[math.log(2.0)]], requires_grad=True)
    loss, ratio, cf = ppo_policy_loss(new, torch.zeros(1, 1), torch.tensor([[1.0]]), torch.ones(1, 1), eps)
    assert math.isclose(float(loss.detach()), -1.2, rel_tol=1e-5)
    loss.backward(); assert float(new.grad) == 0.0 and float(cf) == 1.0
    loss, _, _ = ppo_policy_loss(torch.tensor([[math.log(2.0)]]), torch.zeros(1, 1), torch.tensor([[-1.0]]), torch.ones(1, 1), eps)
    assert math.isclose(float(loss.detach()), 2.0, rel_tol=1e-5)


def test_on_policy_ratio_one():
    lp = torch.randn(2, 5)
    loss, ratio, cf = ppo_policy_loss(lp, lp.clone(), torch.randn(2, 5), torch.ones(2, 5), 0.2)
    assert torch.allclose(ratio, torch.ones(2, 5)) and float(cf) == 0.0


def test_gae_terminal_reward():
    r = torch.tensor([[0., 0., 1.]]); v = torch.zeros(1, 3); m = torch.ones(1, 3)
    adv, ret = compute_gae(r, v, m, gamma=1.0, lam=0.95)
    assert torch.allclose(adv, torch.tensor([[0.9025, 0.95, 1.0]]), atol=1e-4)
    adv, _ = compute_gae(r[:, :3], v, torch.tensor([[1., 1., 0.]]), gamma=1.0, lam=0.95)
    assert float(adv[0, 2]) == 0.0


def test_shaped_reward_places_terminal_and_kl():
    lp, ref = torch.tensor([[-1., -1., -1.]]), torch.tensor([[-2., -2., -2.]])
    rw = shaped_rewards(torch.tensor([2.0]), lp, ref, torch.tensor([[1., 1., 0.]]), 0.1)
    assert torch.allclose(rw, torch.tensor([[-0.1, 1.9, 0.0]]), atol=1e-6)


def test_normalize_raises_on_nan():
    import pytest
    with pytest.raises(FloatingPointError):
        normalize_advantages(torch.tensor([[float("nan"), 1.0]]), torch.ones(1, 2))
