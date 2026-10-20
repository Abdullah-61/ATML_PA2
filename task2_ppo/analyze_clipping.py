from __future__ import annotations

import argparse
import json

import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task2_ppo.continue_train import ensure_fork
from task2_ppo.evaluate import ensure_eval
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards
from task2_ppo.utils import optimizer_step, response_logprobs, set_ppo_mode, trainable_to_fp32


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")
    return rows


def build_items(cfg, tok, rows):
    """Use exactly what the cache stores: old/ref log-probs, critic values, effective terminal reward."""
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    from common.data import prompt_messages
    eos = tok.eos_token_id
    items = []
    for r in rows:
        pm = prompt_messages(prompt_rows[int(r["source_index"])])
        p_ids = tok.apply_chat_template(pm, tokenize=True, add_generation_prompt=True)
        if not isinstance(p_ids, list):
            p_ids = p_ids["input_ids"]
        old = torch.as_tensor(r["old_logprobs"]).float().flatten()
        ref = torch.as_tensor(r["ref_logprobs"]).float().flatten()
        val = torch.as_tensor(r["values"]).float().flatten()
        n = len(old)
        assert len(ref) == n and len(val) == n, "cached old/ref/value lengths disagree"
        r_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if bool(r["terminated_with_eos"]) and len(r_ids) + 1 == n:
            r_ids = r_ids + [eos]  # the decoded text drops the EOS token the cache scored
        if len(r_ids) != n:
            raise ValueError(f"source_index {r['source_index']}: re-tokenised response has {len(r_ids)} tokens "
                             f"but the cache has {n} log-probs; cannot align tokens to cached log-probs")
        items.append({"source_index": int(r["source_index"]), "p_ids": p_ids, "r_ids": r_ids, "n": n,
                      "old": old, "ref": ref, "values": val,
                      "reward": float(r["effective_terminal_reward"])})
    return items


def row_logp(policy, item, dev, grad=False):
    ids = torch.tensor([item["p_ids"] + item["r_ids"]], device=dev)
    pw = len(item["p_ids"])
    lp, _ = response_logprobs(policy, ids, torch.ones_like(ids), pw, ids[:, pw:], grad=grad)
    return lp[0]


def surrogate_metrics(ratios, advs, eps):
    r, a = torch.cat(ratios), torch.cat(advs)
    unc, clp = r * a, r.clamp(1 - eps, 1 + eps) * a
    return {"eps": eps, "surrogate_clipped": float(torch.minimum(unc, clp).mean()),
            "surrogate_unclipped": float(unc.mean()),
            "clip_fraction": float(((r < 1 - eps) | (r > 1 + eps)).float().mean()),
            "affected_fraction": float((clp < unc - 1e-12).float().mean()),
            "ratio_mean_abs_dev": float((r - 1).abs().mean()), "ratio_max": float(r.max()),
            "ratio_min": float(r.min()), "n_tokens": int(r.numel())}


def probe_updates(cfg, items, advs, eps, lr, steps, dev):
    """Take `steps` gradient steps on the fixed cached batch and track how the ratios spread."""
    policy = trainable_to_fp32(load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True))
    params = trainable_parameters(policy)
    opt = AdamW(params, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    total = sum(it["n"] for it in items)
    curve = []
    for step in range(steps + 1):
        set_ppo_mode(policy)
        ratios, adv_l = [], []
        for it, adv in zip(items, advs):
            train = step < steps
            with torch.set_grad_enabled(train):
                lp = row_logp(policy, it, dev, grad=train)
                loss, ratio, _ = ppo_policy_loss(lp[None], it["old"].to(dev)[None], adv.to(dev)[None],
                                                 torch.ones_like(lp)[None], eps)
                if train:
                    scaler.scale(loss * it["n"] / total).backward()
            ratios.append(ratio[0]); adv_l.append(adv.to(dev))
        curve.append({"step": step, **surrogate_metrics(ratios, adv_l, eps)})
        if step < steps:
            optimizer_step(scaler, opt, params, float(cfg["max_grad_norm"]))
    clear_gpu(policy, opt)
    return curve


def run_cached_study(config_path):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    tok = load_tokenizer(cfg["base_model"])
    items = build_items(cfg, tok, rows)

    # current midpoint policy / reference vs the cached log-probs (consistency check + ratios)
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    new_l, ref_chk = [], []
    with torch.no_grad():
        for it in items:
            new_l.append(row_logp(policy, it, dev).float())
            with reference_mode(policy):
                ref_chk.append(row_logp(policy, it, dev).float())
    old = [it["old"].to(dev) for it in items]
    ref = [it["ref"].to(dev) for it in items]
    check = {"mean_abs_diff_policy_vs_cached_old": float(torch.cat([(a - b).abs() for a, b in zip(new_l, old)]).mean()),
             "mean_abs_diff_reference_vs_cached_ref": float(torch.cat([(a - b).abs() for a, b in zip(ref_chk, ref)]).mean()),
             "n_rollouts": len(items), "n_tokens": sum(it["n"] for it in items),
             "reward_source": "cache.effective_terminal_reward",
             "reward_mean": sum(it["reward"] for it in items) / len(items),
             "source_indices": [it["source_index"] for it in items]}
    print(json.dumps(check, indent=1))
    clear_gpu(policy)

    # advantages from the cached critic values, rewards, and old/ref log-probs
    olds, refs = pad_sequence(old, batch_first=True), pad_sequence(ref, batch_first=True)
    values = pad_sequence([it["values"].to(dev) for it in items], batch_first=True)
    mask = pad_sequence([torch.ones(it["n"], device=dev) for it in items], batch_first=True)
    rew = torch.tensor([it["reward"] for it in items], device=dev)
    rewards = shaped_rewards(rew, olds, refs, mask, float(cfg["kl_beta"]))
    adv, _ = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)
    advs = [adv[i, : it["n"]] for i, it in enumerate(items)]

    ratios = [torch.exp(a - b) for a, b in zip(new_l, old)]
    static = [surrogate_metrics(ratios, advs, float(e)) for e in cfg["clip_values"]]

    steps = int(cfg["clip_probe_steps"])
    lr0 = float(cfg["policy_learning_rate"])
    probes = []
    for mult in cfg["clip_probe_lr_mult"]:
        for e in cfg["clip_values"]:
            probes.append({"eps": float(e), "lr": lr0 * mult, "lr_mult": mult,
                           "curve": probe_updates(cfg, items, advs, float(e), lr0 * mult, steps, dev)})

    res = {"check": check, "static": static, "probes": probes, "kl_beta": cfg["kl_beta"],
           "gamma": cfg["gamma"], "gae_lambda": cfg["gae_lambda"]}
    out = repo_path(cfg["results_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "cached_clipping_study.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(static, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if not args.skip_cached:
        run_cached_study(args.config)
        clear_gpu()
    if not args.skip_forks:
        kl = float(cfg["kl_beta"])
        for e in cfg["clip_values"]:
            name, out = ensure_fork(args.config, float(e), kl)
            ensure_eval(args.config, out, name)
            clear_gpu()


if __name__ == "__main__":
    main()
