from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.utils import filter_prompts, generate_rollout, response_logprobs


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


@torch.no_grad()
def evaluate_policy(config_path: str, adapter: str | None, name: str):
    b = load_evaluation_bundle(config_path, adapter)
    cfg, tok, policy = b["cfg"], b["tokenizer"], b["policy"]
    rm, rtok = b["reward"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kept, skipped = filter_prompts(tok, b["rows"], int(cfg["max_prompt_length"]))
    sel = kept[: int(cfg["eval_prompts"])]           # the same fixed held-out prompts for every condition
    prompts = [prompt_messages(b["rows"][i]) for i in sel]
    max_new, bs = int(cfg["eval_max_completion_length"]), int(cfg["eval_gen_batch"])
    set_seed(int(cfg["seed"]))

    recs = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s:s + bs]
        gen = generate_rollout(policy, tok, chunk, cfg, max_new)
        seq, attn = gen["sequences"].to(dev), gen["attention_mask"].to(dev)
        resp, mask = gen["response_ids"].to(dev), gen["response_mask"].to(dev).float()
        pw = gen["prompt_width"]
        rew = score_reward_pairs(rm, rtok, chunk, gen["responses"]).cpu().tolist()
        step = 2
        for k in range(0, len(chunk), step):
            sl = slice(k, k + step)
            plp, ent = response_logprobs(policy, seq[sl], attn[sl], pw, resp[sl], entropy=True)
            with reference_mode(policy):
                rlp, _ = response_logprobs(policy, seq[sl], attn[sl], pw, resp[sl])
            m = mask[sl]
            d = (plp - rlp) * m
            cnt = m.sum(-1).clamp_min(1.0)
            ent_mean = ((ent * m).sum(-1) / cnt).cpu().tolist()
            for j in range(plp.shape[0]):
                i = k + j
                recs.append({"source_index": sel[s + i], "prompt": chunk[i], "response": gen["responses"][i],
                             "reward": rew[i], "length": int(gen["response_lengths"][i]),
                             "kl_token_mean": float(d[j].sum() / cnt[j]), "kl_seq": float(d[j].sum()),
                             "entropy": ent_mean[j], "terminated": bool(gen["terminated_with_eos"][i]),
                             "truncated": bool(gen["truncated"][i])})
            del plp, rlp, ent

    L = np.array([r["length"] for r in recs], dtype=float)
    R = np.array([r["reward"] for r in recs], dtype=float)
    q1, q3 = np.percentile(L, [25, 75])
    res = {"name": name, "adapter": adapter, "seed": cfg["seed"], "n_prompts": len(recs), "source_indices": sel,
           "n_prompts_skipped_too_long": len(skipped),
           "reward_mean": float(R.mean()), "reward_std": float(R.std()),
           "kl_token_mean": float(np.mean([r["kl_token_mean"] for r in recs])),
           "kl_seq": float(np.mean([r["kl_seq"] for r in recs])),
           "entropy_mean": float(np.mean([r["entropy"] for r in recs])),
           "len_mean": float(L.mean()), "len_std": float(L.std()), "len_iqr": float(q3 - q1),
           "corr_reward_len": safe_corr(R, L),
           "eos_rate": float(np.mean([r["terminated"] for r in recs])),
           "truncated_frac": float(np.mean([r["truncated"] for r in recs])),
           "max_new_tokens": max_new, "temperature": cfg["generation"]["temperature"],
           "top_p": cfg["generation"]["top_p"]}
    out_dir = repo_path(cfg["results_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{name}.json").write_text(json.dumps(res, indent=1))
    with open(out_dir / f"generations_{name}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1))
    clear_gpu(b)
    return res


def ensure_eval(config_path: str, adapter: str | None, name: str):
    cfg = load_yaml(config_path)
    path = repo_path(cfg["results_dir"]).joinpath(f"eval_{name}.json")
    if path.exists():
        print(f"[eval {name}] exists — skipping")
        return json.loads(path.read_text())
    return evaluate_policy(config_path, adapter, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", help="omit (or 'none') to evaluate the frozen reference model")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_policy(args.config, None if args.adapter in (None, "none") else args.adapter, args.name)


if __name__ == "__main__":
    main()
