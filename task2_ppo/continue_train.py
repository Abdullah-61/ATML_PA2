from __future__ import annotations

import argparse
import json
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, set_seed
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss
from task2_ppo.utils import (
    filter_prompts,
    generate_rollout,
    optimizer_step,
    response_logprobs,
    response_values,
    set_ppo_mode,
    trainable_to_fp32,
)


def fork_name(eps: float, kl: float) -> str:
    return f"eps{eps:g}_kl{kl:g}"


def _check_finite(name, t):
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"non-finite values in {name}")


def run_ppo(config_path: str, output_dir: str, updates: int, clip_epsilon: float, kl_beta: float, run_name: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = repo_path(output_dir).joinpath(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- models: policy + critic start from the supplied PPO midpoint -------------------
    tok = load_tokenizer(cfg["base_model"])
    policy = trainable_to_fp32(load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True))
    val_model = trainable_to_fp32(
        load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode=cfg["value_train_mode"]))
    ref_model = load_policy(cfg, adapter_path=None, trainable=False)
    rm, rtok = load_reward_model(cfg)

    prompt_pool = read_jsonl(cfg["paths"]["rl_prompt_train"])
    kept_indices, skipped_indices = filter_prompts(tok, prompt_pool, int(cfg["max_prompt_length"]))
    print(f"Kept prompts: {len(kept_indices)}, Skipped prompts: {len(skipped_indices)}")

    # ---- optimisers / hyper-parameters, all from the release config ----------------------
    policy_params = trainable_parameters(policy)
    policy_opt = AdamW(policy_params, lr=float(cfg["policy_learning_rate"]))
    val_groups = value_parameter_groups(val_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                                        head_lr=float(cfg["value_head_learning_rate"]))
    val_opt = AdamW(val_groups)
    value_params = [p for g in val_groups for p in g["params"]]
    policy_scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    value_scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    prompts_per_update = int(cfg["prompts_per_update"])
    ppo_epochs = int(cfg["ppo_epochs"])
    max_new_tokens = int(cfg["max_response_length"])
    gamma, gae_lambda = float(cfg["gamma"]), float(cfg["gae_lambda"])
    max_grad_norm = float(cfg["max_grad_norm"])
    missing_eos_penalty = float(cfg["missing_eos_penalty"])

    log_path = repo_path(cfg["results_dir"]).joinpath(f"train_log_{run_name}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start_time = time.time()
    prompt_idx = 0

    for u in range(1, updates + 1):
        t0 = time.time()

        # 1. prompts for this update (deterministic order, identical across forks) ----------
        batch_msgs = []
        for _ in range(prompts_per_update):
            idx = kept_indices[prompt_idx % len(kept_indices)]
            prompt_idx += 1
            batch_msgs.append(prompt_messages(prompt_pool[idx]))

        # 2. on-policy rollout from the current policy (= pi_old) ---------------------------
        set_ppo_mode(policy)
        with torch.no_grad():
            rollout = generate_rollout(policy, tok, batch_msgs, cfg, max_new_tokens)
        seqs = rollout["sequences"].to(dev)
        attn_mask = rollout["attention_mask"].to(dev)
        response_mask = rollout["response_mask"].to(dev).float()
        pw = rollout["prompt_width"]
        resp_ids = rollout["response_ids"].to(dev)
        responses = rollout["responses"]
        lengths = torch.as_tensor(rollout["response_lengths"], dtype=torch.float32)
        eos_flags = [bool(x) for x in rollout["terminated_with_eos"]]
        W = resp_ids.shape[1]

        # 3. learned reward (+ penalty for not terminating) ---------------------------------
        with torch.no_grad():
            raw_rewards = score_reward_pairs(rm, rtok, batch_msgs, responses,
                                             max_length=int(cfg["reward_max_length"])).to(dev).float()
            penalties = torch.tensor([0.0 if f else missing_eos_penalty for f in eos_flags],
                                     device=dev, dtype=raw_rewards.dtype)
            task_rewards = raw_rewards - penalties

        # 4. old / reference log-probs, entropy, critic values, GAE ---------------------------
        set_ppo_mode(val_model)
        with torch.no_grad():
            old_logprobs, token_entropy = response_logprobs(policy, seqs, attn_mask, pw, resp_ids, entropy=True)
            with reference_mode(ref_model):
                ref_logprobs, _ = response_logprobs(ref_model, seqs, attn_mask, pw, resp_ids)
            entropy = float(masked_mean(token_entropy, response_mask))
            del token_entropy

            values = response_values(val_model, seqs, attn_mask, pw, W)
            _check_finite("critic values", values)
            rewards = shaped_rewards(task_rewards, old_logprobs, ref_logprobs, response_mask, kl_beta)
            advantages, returns = compute_gae(rewards, values, response_mask, gamma, gae_lambda)
            norm_adv = normalize_advantages(advantages, response_mask)

        kl_token = float(masked_mean(old_logprobs - ref_logprobs, response_mask))  # sampled-token KL
        n_tokens = response_mask.sum(-1)
        total_tokens = n_tokens.sum()

        # 5. PPO epochs on this rollout (ratio is recomputed every epoch) --------------------
        ep_pl, ep_vl, ep_gp, ep_gv, ep_clip, ep_rmax, ep_rmean, skipped_steps = [], [], [], [], [], [], [], 0
        for _ in range(ppo_epochs):
            set_ppo_mode(policy)
            set_ppo_mode(val_model)

            # policy: token-weighted mean over the whole batch, one sequence at a time
            pl_total, clip_num, ratios = 0.0, 0.0, []
            for i in range(seqs.shape[0]):
                sl = slice(i, i + 1)
                new_lp, _ = response_logprobs(policy, seqs[sl], attn_mask[sl], pw, resp_ids[sl], grad=True)
                loss_i, ratio_i, clip_i = ppo_policy_loss(new_lp, old_logprobs[sl], norm_adv[sl],
                                                          response_mask[sl], clip_epsilon)
                if not torch.isfinite(loss_i):
                    raise FloatingPointError(f"non-finite policy loss at update {u}")
                w = n_tokens[i] / total_tokens
                policy_scaler.scale(loss_i * w).backward()
                pl_total += float(loss_i.detach() * w)
                clip_num += float(clip_i * n_tokens[i])
                ratios.append(ratio_i[response_mask[sl].bool()])
            g_p, stepped = optimizer_step(policy_scaler, policy_opt, policy_params, max_grad_norm)
            skipped_steps += int(not stepped)
            ratios = torch.cat(ratios)

            # critic
            vl_total = 0.0
            for i in range(seqs.shape[0]):
                sl = slice(i, i + 1)
                pred = response_values(val_model, seqs[sl], attn_mask[sl], pw, W)
                loss_v = value_mse_loss(pred, returns[sl], response_mask[sl])
                value_scaler.scale(loss_v * (n_tokens[i] / total_tokens)).backward()
                vl_total += float(loss_v.detach() * n_tokens[i] / total_tokens)
            g_v, stepped = optimizer_step(value_scaler, val_opt, value_params, max_grad_norm)
            skipped_steps += int(not stepped)

            ep_pl.append(pl_total); ep_vl.append(vl_total); ep_gp.append(g_p); ep_gv.append(g_v)
            ep_clip.append(clip_num / float(total_tokens))
            ep_rmax.append(float(ratios.max())); ep_rmean.append(float(ratios.mean()))

        # 6. diagnostics -----------------------------------------------------------------------
        mval, mret = values[response_mask.bool()], returns[response_mask.bool()]
        ret_var = mret.var(unbiased=False)
        explained_var = float(1.0 - ((mret - mval) ** 2).mean() / ret_var) if ret_var > 0 else float("nan")
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        peak_vram = torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == "cuda" else 0.0

        row = {
            "update": u,
            "reward_raw": float(raw_rewards.mean()),
            "reward_task": float(task_rewards.mean()),
            "kl_token": kl_token,
            "policy_loss": mean(ep_pl),
            "value_loss": mean(ep_vl),
            "entropy": entropy,
            "clip_fraction": mean(ep_clip),
            "clip_fraction_last_epoch": ep_clip[-1],
            "affected_fraction": mean(ep_clip),
            "ratio_max": max(ep_rmax),
            "ratio_mean": mean(ep_rmean),
            "grad_norm_policy": mean(ep_gp),
            "grad_norm_value": mean(ep_gv),
            "resp_len": float(lengths.mean()),
            "eos_rate": sum(eos_flags) / len(eos_flags),
            "value_mean": float(mval.mean()),
            "return_mean": float(mret.mean()),
            "explained_var": explained_var,
            "skipped_optimizer_steps": skipped_steps,
            "peak_vram_gb": peak_vram,
            "wall_clock_s": time.time() - t0,
        }
        append_jsonl(log_path, row)
        print(f"Update {u:02d} | R {row['reward_raw']:.3f} | KL {row['kl_token']:.4f} | "
              f"P {row['policy_loss']:.4f} | V {row['value_loss']:.3f} | clip {row['clip_fraction']:.3f} | "
              f"gP {row['grad_norm_policy']:.3f} | len {row['resp_len']:.0f} | skipped {skipped_steps}")

    # ---- save adapters + run metadata --------------------------------------------------------
    policy.save_pretrained(out_dir)
    val_model.save_pretrained(out_dir.joinpath("value_model"))
    meta = {"run_name": run_name, "updates": updates, "clip_epsilon": clip_epsilon, "kl_beta": kl_beta,
            "ppo_epochs": ppo_epochs, "prompts_per_update": prompts_per_update,
            "max_response_length": max_new_tokens,
            "wall_clock_s": time.time() - start_time,
            "peak_vram_gb": torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == "cuda" else 0.0,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    text = json.dumps(meta, indent=2)
    repo_path(cfg["results_dir"]).joinpath(f"run_meta_{run_name}.json").write_text(text)
    out_dir.joinpath("run_meta.json").write_text(text)
    clear_gpu(policy, ref_model, val_model, rm)


def ensure_fork(config_path: str, eps: float, kl: float) -> tuple[str, str]:
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl)
    output = f"outputs/task2_ppo/{name}"
    if repo_path(output).joinpath("adapter_config.json").exists():
        print(f"Fork {name} exists — skipping")
    else:
        run_ppo(config_path, "outputs/task2_ppo", int(cfg["fork_updates"]), eps, kl, name)
    return name, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output", default="outputs/task2_ppo")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    up = args.updates if args.updates is not None else int(cfg["updates"])
    eps = args.clip_epsilon if args.clip_epsilon is not None else float(cfg["clip_epsilon"])
    kl = args.kl_beta if args.kl_beta is not None else float(cfg["kl_beta"])
    run_ppo(args.config, args.output, up, eps, kl, args.run_name)


if __name__ == "__main__":
    main()
