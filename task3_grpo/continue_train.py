from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, set_seed
from common.metrics import masked_mean
from common.models import (clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode,
                           trainable_parameters)
from task3_grpo.grpo import (ADV_EPS, group_relative_advantages, group_reward_stats, grpo_loss_parts,
                             mask_truncated_sequences)
from task3_grpo.utils import (filter_prompts, generate_rollout, optimizer_step, response_logprobs,
                              set_train_no_dropout, trainable_to_fp32)

MAX_SKIPPED_STEP_RETRIES = 10  # GradScaler overflow retries of the *same* batch before giving up


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = trainable_to_fp32(load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True))
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {"cfg": cfg, "tokenizer": tokenizer, "policy": policy, "reward_model": reward_model,
            "reward_tokenizer": reward_tokenizer, "prompt_rows": prompts, "optimizer": optimizer}


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None,
             loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg, tok, policy, opt = bundle["cfg"], bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rtok, prompt_pool = bundle["reward_model"], bundle["reward_tokenizer"], bundle["prompt_rows"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    updates = int(updates) if updates is not None else int(cfg["updates"])
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    K = int(cfg["num_generations"])
    prompts_per_update = int(cfg["prompts_per_update"])
    policy_epochs = int(cfg["policy_epochs"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    mask_trunc = bool(cfg["mask_truncated_completions"])
    max_grad_norm = float(cfg["max_grad_norm"])
    params = trainable_parameters(policy)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    kept, skipped = filter_prompts(tok, prompt_pool, int(cfg["max_prompt_length"]))
    print(f"[{run_name}] loss_type={loss_type} | K={K} | prompts/update={prompts_per_update} | "
          f"epochs={policy_epochs} | kept prompts {len(kept)}, skipped {len(skipped)}")

    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    log_path, seq_path = results / f"train_log_{run_name}.jsonl", results / f"seqstats_{run_name}.jsonl"
    for p in (log_path, seq_path):
        if p.exists():
            p.unlink()

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    start, prompt_idx, total_gen_tokens = time.time(), 0, 0

    for u in range(1, updates + 1):
        t0 = time.time()

        # 1. prompts -> K completions each (group_ids identify the prompt of every completion) -----------
        batch_msgs, group_ids = [], []
        for g in range(prompts_per_update):
            msgs = prompt_messages(prompt_pool[kept[prompt_idx % len(kept)]])
            prompt_idx += 1
            batch_msgs += [msgs] * K
            group_ids += [g] * K
        group_ids = torch.tensor(group_ids, device=dev)

        with torch.no_grad():
            ro = generate_rollout(policy, tok, batch_msgs, cfg, max_len)
        seqs, attn = ro["sequences"].to(dev), ro["attention_mask"].to(dev)
        resp_ids, pw = ro["response_ids"].to(dev), ro["prompt_width"]
        lengths = torch.tensor(ro["response_lengths"], dtype=torch.float32, device=dev)
        truncated = list(ro["truncated"])
        token_mask = ro["response_mask"].to(dev).float()
        if mask_trunc:
            token_mask = mask_truncated_sequences(token_mask, truncated)
        N = seqs.shape[0]
        total_gen_tokens += int(lengths.sum())

        # 2. rewards, group-relative advantages, group statistics ---------------------------------------------
        with torch.no_grad():
            rewards = score_reward_pairs(rm, rtok, batch_msgs, ro["responses"]).to(dev).float()
        adv = group_relative_advantages(rewards, group_ids, ADV_EPS)
        g_std, g_inf = group_reward_stats(rewards, group_ids, ADV_EPS)

        # 3. old / reference log-probs (reference = base model, adapter disabled), entropy -----------------------
        with torch.no_grad():
            old_lp, ent = response_logprobs(policy, seqs, attn, pw, resp_ids, entropy=True)
            with reference_mode(policy):
                ref_lp, _ = response_logprobs(policy, seqs, attn, pw, resp_ids)
        entropy_full = float(masked_mean(ent, token_mask)); del ent
        kl_token = float(masked_mean(old_lp - ref_lp, token_mask))

        n_tok = token_mask.sum(-1)
        total_tok = n_tok.sum()
        norm_len = n_tok.clamp_min(1.0) if loss_type == "grpo" else torch.full_like(n_tok, float(max_len))
        tok_weight = adv.abs() / (N * norm_len)       # |per-token weight| of each completion in the policy term
        mass = tok_weight * n_tok                      # total policy-term weight allocated to the completion

        # 4. PPO-style epochs on this rollout --------------------------------------------------------------------
        ep = {"policy_term": [], "kl": [], "loss": [], "clip": [], "gn": [], "ratio_mean": [], "sample_ent": []}
        retries, seq_rows, did_update = 0, [], total_tok > 0
        for epoch in range(policy_epochs if did_update else 0):
            for attempt in range(MAX_SKIPPED_STEP_RETRIES + 1):
                set_train_no_dropout(policy)
                acc = {"policy_term": 0.0, "kl": 0.0, "loss": 0.0, "clip": 0.0, "ratio_mean": 0.0, "sample_ent": 0.0}
                seq_rows = []
                for i in range(N):
                    if n_tok[i] == 0:
                        seq_rows.append({"grad_norm": 0.0})
                        continue
                    sl = slice(i, i + 1)
                    new_lp, _ = response_logprobs(policy, seqs[sl], attn[sl], pw, resp_ids[sl], grad=True)
                    pt, kl, d = grpo_loss_parts(new_lp, old_lp[sl], adv[sl], token_mask[sl], ref_lp[sl], eps,
                                                loss_type, max_len)
                    w = n_tok[i] / total_tok
                    loss_i = pt / N + beta * kl * w
                    if not torch.isfinite(loss_i):
                        raise FloatingPointError(f"non-finite loss at update {u}, completion {i}")
                    before = [None if p.grad is None else p.grad.detach().clone() for p in params]
                    scaler.scale(loss_i).backward()
                    sq = 0.0
                    for p, b in zip(params, before):
                        if p.grad is not None:
                            sq += float((p.grad - b if b is not None else p.grad).float().pow(2).sum())
                    seq_rows.append({"grad_norm": math.sqrt(sq) / scaler.get_scale()})
                    acc["policy_term"] += float(pt.detach()) / N
                    acc["kl"] += float(kl.detach()) * float(w)
                    acc["loss"] += float(loss_i.detach())
                    acc["clip"] += float(d["clip_fraction"]) * float(w)
                    acc["ratio_mean"] += float(d["ratio_mean"]) * float(w)
                    acc["sample_ent"] += float(d["sample_entropy"]) * float(w)
                gn, stepped = optimizer_step(scaler, opt, params, max_grad_norm)
                if stepped:
                    break
                retries += 1
            else:
                raise FloatingPointError(f"optimizer step skipped {MAX_SKIPPED_STEP_RETRIES + 1} times in a row")
            for k, v in (("policy_term", acc["policy_term"]), ("kl", acc["kl"]), ("loss", acc["loss"]),
                         ("clip", acc["clip"]), ("gn", gn), ("ratio_mean", acc["ratio_mean"]),
                         ("sample_ent", acc["sample_ent"])):
                ep[k].append(v)
            for i, r in enumerate(seq_rows):
                append_jsonl(seq_path, {"update": u, "epoch": epoch, "idx": i, "group": int(group_ids[i]),
                                        "length": int(lengths[i]), "n_train_tokens": int(n_tok[i]),
                                        "masked": bool(n_tok[i] == 0), "reward": float(rewards[i]),
                                        "adv": float(adv[i]), "grad_norm": r["grad_norm"],
                                        "tok_weight": float(tok_weight[i]), "mass": float(mass[i])})

        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
        row = {
            "update": u, "loss_type": loss_type,
            "reward": float(rewards.mean()),
            "kl_k3": mean(ep["kl"]), "kl_token": kl_token,
            "group_reward_std": float(g_std.mean()),
            "frac_uninformative_groups": float((~g_inf).float().mean()),
            "policy_loss": mean(ep["policy_term"]), "total_loss": mean(ep["loss"]),
            "grad_norm": mean(ep["gn"]),
            "entropy": entropy_full, "sample_entropy": mean(ep["sample_ent"]),
            "resp_len": float(lengths.mean()), "gen_tokens": int(lengths.sum()),
            "masked_frac": float((n_tok == 0).float().mean()), "eos_rate": sum(ro["terminated_with_eos"]) / N,
            "clip_fraction": mean(ep["clip"]), "ratio_mean": mean(ep["ratio_mean"]),
            "scaler_retries": retries,
            "peak_vram_gb": torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == "cuda" else 0.0,
            "wall_clock_s": time.time() - t0,
        }
        append_jsonl(log_path, row)
        print(f"Update {u:02d} | R {row['reward']:.3f} | KL(k3) {row['kl_k3']:.5f} | gStd {row['group_reward_std']:.3f} | "
              f"uninf {row['frac_uninformative_groups']:.2f} | pLoss {row['policy_loss']:.4f} | "
              f"gN {row['grad_norm']:.3f} | ent {row['entropy']:.3f} | len {row['resp_len']:.0f} | "
              f"masked {row['masked_frac']:.2f} | retries {retries}")

    policy.save_pretrained(out)
    meta = {"run_name": run_name, "loss_type": loss_type, "updates": updates, "num_generations": K,
            "prompts_per_update": prompts_per_update, "policy_epochs": policy_epochs, "clip_epsilon": eps,
            "kl_beta": beta, "max_completion_length": max_len, "total_generated_tokens": total_gen_tokens,
            "wall_clock_s": time.time() - start,
            "peak_vram_gb": torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == "cuda" else 0.0,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}
    text = json.dumps(meta, indent=2)
    (results / f"run_meta_{run_name}.json").write_text(text)
    (out / "run_meta.json").write_text(text)
    clear_gpu(policy, rm, opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
