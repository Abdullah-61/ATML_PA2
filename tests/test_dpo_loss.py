import math, torch
from task1_dpo.dpo import dpo_loss

def test_matches_equation():
    pc, pr, rc, rr = (torch.tensor([x]) for x in (-10., -14., -12., -13.))
    beta = 0.1
    loss, d = dpo_loss(pc, pr, rc, rr, beta)
    m = (-10 + 12) - (-14 + 13)          # = 3
    assert math.isclose(float(loss), -math.log(1 / (1 + math.exp(-beta * m))), rel_tol=1e-5)
    assert float(d["preference_accuracy"]) == 1.0

def test_zero_at_init():  # policy == reference -> loss = log 2
    x = torch.tensor([-5., -7.])
    loss, _ = dpo_loss(x, x - 1, x, x - 1, 0.3)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-5)
