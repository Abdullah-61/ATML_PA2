from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common.data import (encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages,
                         prompt_messages_from_preference, read_jsonl, repo_path)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import mean_response_length, sampled_kl, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_fitting

STRATUM_KEYS = ("stratum", "length_stratum", "length_bucket", "bucket", "group", "category")


def load_evaluation_bundle(config_path: str, adapter: str, rows_key: str = "dpo_standard_eval", with_reward: bool = True):
    cfg = load_yaml(config_path)
    out = {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"][rows_key]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
    }
    out["reward"] = load_reward_model(cfg) if with_reward else None
    return out


@torch.no_grad()
def preference_stats(policy, tokenizer, rows, cfg, beta, bs=4):
    """Per-pair DPO loss / margin m_theta (sequence logprobs summed over response tokens)."""
    dev = next(policy.parameters()).device
    recs = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        encs = {"c": [], "r": []}
        for row in chunk:
            pm = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            encs["c"].append(encode_prompt_response(tokenizer, pm, yc, int(cfg["max_sequence_length"])))
            encs["r"].append(encode_prompt_response(tokenizer, pm, yr, int(cfg["max_sequence_length"])))
        bc = {k: v.to(dev) for k, v in pad_batch(tokenizer, encs["c"]).items()}
        br = {k: v.to(dev) for k, v in pad_batch(tokenizer, encs["r"]).items()}
        pc, _, _ = response_sequence_logprobs(policy, bc)
        pr, _, _ = response_sequence_logprobs(policy, br)
        with reference_mode(policy):
            rc, _, _ = response_sequence_logprobs(policy, bc)
            rr, _, _ = response_sequence_logprobs(policy, br)
        m = (pc - rc) - (pr - rr)
        for j in range(len(chunk)):
            loss_j, _ = dpo_loss(pc[j:j+1], pr[j:j+1], rc[j:j+1], rr[j:j+1], beta)
            recs.append({"margin": float(m[j]), "loss": float(loss_j),
                         "len_chosen": int(bc["response_mask"][j].sum()), "len_rejected": int(br["response_mask"][j].sum())})
    return recs


def summarize_pref(recs):
    m = np.array([r["margin"] for r in recs])
    return {"n": len(recs), "dpo_loss": float(np.mean([r["loss"] for r in recs])),
            "pref_acc": float((m > 0).mean()), "margin_mean": float(m.mean())}


@torch.no_grad()
def generate_and_score(bundle, prompts, max_new, gen_bs=16, seed=0):
    """Sample responses; return per-sample response, KL estimate (policy-ref), reward, length."""
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    g = cfg["generation"]
    set_seed(seed)
    out = []
    for i in range(0, len(prompts), gen_bs):
        chunk = prompts[i:i + gen_bs]
        gen = batch_generate(policy, tok, chunk, 1024, max_new, g["temperature"], g["top_p"], g["do_sample"])
        pw, resp, mask = gen["prompt_width"], gen["response_ids"], gen["response_mask"]
        base = len(out)  # index of this chunk's first sample in `out`
        # full-vocab logits are huge: score the KL pass two sequences at a time and free them
        for k in range(0, len(chunk), 2):
            sl = slice(k, k + 2)
            pol_lp, lg = response_token_logprobs(policy, gen["sequences"][sl], gen["attention_mask"][sl], pw, resp[sl])
            del lg
            with reference_mode(policy):
                ref_lp, lg = response_token_logprobs(policy, gen["sequences"][sl], gen["attention_mask"][sl], pw, resp[sl])
            del lg
            for j in range(pol_lp.shape[0]):
                r_ = k + j
                n = int(mask[r_].sum())
                # token-mean (course helper) and sequence-sum of the log-ratio are both stored
                kl_tok = float(sampled_kl(pol_lp[j:j+1], ref_lp[j:j+1], mask[r_:r_+1]))
                kl_seq = float(((pol_lp[j] - ref_lp[j]) * mask[r_]).sum())
                out.append({"response": gen["responses"][r_], "length": n, "kl_token_mean": kl_tok, "kl_seq_sum": kl_seq,
                            "truncated": gen["truncated"][r_]})
            del pol_lp, ref_lp
        torch.cuda.empty_cache()
        if bundle.get("reward") is not None:
            rm, rtok = bundle["reward"]
            sc = score_reward_pairs(rm, rtok, chunk, gen["responses"]).cpu().tolist()
            for k, s_ in enumerate(sc):
                out[base + k]["reward"] = float(s_)
    return out


def length_stats(lengths):
    a = np.array(lengths, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"len_mean": float(a.mean()), "len_std": float(a.std()), "len_iqr": float(q3 - q1)}


def full_eval(config_path, adapter, name, beta=None, n_gen_prompts=None, max_pref=None):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, tok = bundle["cfg"], bundle["tokenizer"]
    beta = float(beta if beta is not None else cfg["beta"])  # beta used for held-out DPO loss
    all_rows = bundle["rows"][: max_pref] if max_pref else bundle["rows"]
    rows, kept_idx, skipped_idx = filter_fitting(tok, all_rows, int(cfg["max_sequence_length"]))
    res = {"name": name, "adapter": adapter, "beta_eval": beta, "seed": cfg["seed"],
           "n_heldout_total": len(all_rows), "n_skipped_overlong_prompt": len(skipped_idx),
           "skipped_source_indices": skipped_idx}
    res.update(summarize_pref(preference_stats(bundle["policy"], tok, rows, cfg, beta)))
    n_gen = int(n_gen_prompts or cfg.get("eval_gen_prompts", 100))
    prompts = [prompt_messages_from_preference(r) for r in rows[:n_gen]]
    gens = generate_and_score(bundle, prompts, int(cfg["max_generation_tokens"]), seed=int(cfg["seed"]))
    res["gen_prompt_indices"] = kept_idx[:len(prompts)]  # source row indices in the held-out file
    res["kl_token_mean"] = float(np.mean([g["kl_token_mean"] for g in gens]))
    res["kl_seq_sum"] = float(np.mean([g["kl_seq_sum"] for g in gens]))
    res["reward_mean"] = float(np.mean([g["reward"] for g in gens]))
    res["truncated_frac"] = float(np.mean([g["truncated"] for g in gens]))
    res.update(length_stats([g["length"] for g in gens]))
    # explicit word-limit compliance on the 10 common prompts
    wl = read_jsonl(cfg["paths"]["word_limit_prompts"])
    wl_gens = generate_and_score({**bundle, "reward": None}, [prompt_messages(r) for r in wl],
                                 int(cfg["max_generation_tokens"]), seed=int(cfg["seed"]))
    comp = []
    for r, g in zip(wl, wl_gens):
        c = word_limit_compliance(prompt_messages(r)[-1]["content"], g["response"])
        g.update({"prompt_id": r["prompt_id"], "words": word_count(g["response"]), "compliant": c})
        comp.append(c)
    res["wordlimit_compliance"] = float(np.mean([c for c in comp if c is not None]))
    res["wordlimit_n"] = sum(c is not None for c in comp)
    res["wordlimit_mean_words"] = float(np.mean([g["words"] for g in wl_gens]))
    out_dir = repo_path(cfg["results_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / f"eval_{name}.json", "w"), indent=1)
    with open(out_dir / f"generations_{name}.jsonl", "w") as f:
        for idx, (p, g) in enumerate(zip(prompts, gens)):
            f.write(json.dumps({"row_index": idx, "prompt": p, **g}, ensure_ascii=False) + "\n")
    with open(out_dir / f"wordlimit_{name}.jsonl", "w") as f:
        for g in wl_gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1))
    clear_gpu(bundle)
    return res


def find_stratum_key(rows):
    """Known names first; otherwise any scalar field with exactly 3 distinct values (the 3 length strata)."""
    for k in STRATUM_KEYS:
        if k in rows[0]:
            return k
    skip = {"prompt", "chosen", "rejected", "messages"}
    cands = []
    for k, v in rows[0].items():
        if k in skip or isinstance(v, (list, dict, float)):
            continue
        vals = {str(r.get(k)) for r in rows}
        if len(vals) == 3:
            cands.append(k)
    if len(cands) >= 1:
        print(f"[stratified_eval] auto-detected stratum field {cands[0]!r} (candidates: {cands}); "
              f"values: {sorted({str(r[cands[0]]) for r in rows})}")
        return cands[0]
    raise KeyError(f"No 3-valued stratum field found; row keys = {list(rows[0])}")


def stratified_eval(config_path, adapter, name, beta=None):
    """Held-out preference accuracy per length stratum (preferred-longer / matched / rejected-longer)."""
    bundle = load_evaluation_bundle(config_path, adapter, rows_key="dpo_length_eval", with_reward=False)
    cfg = bundle["cfg"]
    rows, kept_idx, skipped_idx = filter_fitting(bundle["tokenizer"], bundle["rows"], int(cfg["max_sequence_length"]))
    print(f"[stratified_eval] kept {len(rows)}, skipped {len(skipped_idx)} over-long prompts")
    key = find_stratum_key(rows)
    beta = float(beta if beta is not None else cfg["beta"])
    recs = preference_stats(bundle["policy"], bundle["tokenizer"], rows, cfg, beta)
    res = {"name": name, "stratum_key": key, "overall": summarize_pref(recs), "strata": {},
           "n_skipped_overlong_prompt": len(skipped_idx), "skipped_source_indices": skipped_idx}
    for s in sorted({str(r[key]) for r in rows}):
        sub = [rec for rec, r in zip(recs, rows) if str(r[key]) == s]
        res["strata"][s] = summarize_pref(sub)
    out_dir = repo_path(cfg["results_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / f"stratified_{name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    clear_gpu(bundle)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta for held-out DPO loss (default: config beta)")
    ap.add_argument("--n-gen-prompts", type=int)
    args = ap.parse_args()
    full_eval(args.config, args.adapter, args.name, args.beta, args.n_gen_prompts)


if __name__ == "__main__":
    main()
