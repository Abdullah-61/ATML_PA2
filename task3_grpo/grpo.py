from __future__ import annotations

import torch

from common.metrics import masked_mean, sample_entropy

# Epsilon of the group-relative advantage A = (r - mu) / (sigma + eps)  (same value as the starter's default).
# A group is "uninformative" when its reward std is <= this value (advantages are then exactly 0).
ADV_EPS = 1e-6


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = ADV_EPS):
    """One scalar advantage per sampled completion, standardised *within its own prompt group*.

    A_k = (r_k - mu_g) / (sigma_g + eps),  mu_g, sigma_g = mean / (population) std of the rewards of group g.
    (The starter standardised over the whole batch, ignoring `group_ids`; that is not GRPO.)
    """
    rewards = rewards.float()
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        m = group_ids == g
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = ADV_EPS):
    """Per-group reward std and the informative flag (std > eps)."""
    stds = []
    for g in torch.unique(group_ids):
        stds.append(rewards[group_ids == g].float().std(unbiased=False))
    stds = torch.stack(stds)
    return stds, stds > eps


def grpo_loss_parts(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps,
                    loss_type="grpo", max_completion_length: int | None = None):
    """Differentiable (policy_term, kl) of the GRPO loss plus detached diagnostics.

    policy_term = -mean_k [ sum_t min(rho A, clip(rho) A) * mask / norm_k ]
      grpo   : norm_k = T_k (realised, unmasked length)          -> 1/T_k sequence normalisation
      dr_grpo: norm_k = max_completion_length (a constant)       -> no length dependence
    kl = masked mean over all unmasked tokens of the k3 estimator  exp(ref-new) - (ref-new) - 1.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        per_sequence = token_sum / token_mask.sum(-1).clamp_min(1.0)
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        per_sequence = token_sum / float(max_completion_length)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
    policy_term = -per_sequence.mean()

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)

    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    diag = {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }
    return policy_term, kl, diag


def grpo_policy_loss(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps, beta,
                     loss_type="grpo", max_completion_length: int | None = None):
    """PPO-style clipped GRPO loss for already-sampled completions: policy_term + beta * KL.

    `token_mask` is all-zero for a completion deliberately masked because it hit the maximum length.
    """
    policy_term, kl, diag = grpo_loss_parts(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps,
                                            loss_type, max_completion_length)
    return policy_term + float(beta) * kl, diag


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
