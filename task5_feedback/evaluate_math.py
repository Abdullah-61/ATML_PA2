from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from tqdm import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer
from task5_feedback.metrics_t5 import (failure_type, format_check, gold_final, judge_record, problem_text,
                                       verifier_pref)
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
# (A, B) for every judged pair. A-vs-SFT gives the "RLAIF pairwise win rate against SFT"; rlvr-vs-rlaif is the
# head-to-head. All three pairs together give a K=3 direct-RLAIF group reward per prompt.
PAIRS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]
DATASETS = ["gsm", "transfer"]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


# ------------------------------------------------------------------------------------------------------------ paths
def out_dir(cfg):
    p = repo_path(cfg["results_dir"]) / "task5_feedback"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gen_path(cfg, dataset, policy):
    return out_dir(cfg) / f"gen_{dataset}_{policy}.jsonl"


def judge_path(cfg, dataset):
    return out_dir(cfg) / f"judge_{dataset}.jsonl"


def judge_cache_path(cfg):
    return out_dir(cfg) / "judge_cache.json"


def load_judge(cfg):
    return PairwiseAIJudge(cfg, judge_cache_path(cfg))


# ------------------------------------------------------------------------------------------------------- generation
def generate_for_policy(cfg, rows, tokenizer, policy: str, batch_size: int = 16):
    """Greedy, identical decoding for every policy; fixed row order."""
    model = load_frozen_policy(cfg, policy)
    max_new = int(cfg["math_max_new_tokens"])
    recs, t0 = [], time.time()
    for s in tqdm(range(0, len(rows), batch_size), desc=f"gen {policy}"):
        chunk = rows[s:s + batch_size]
        gen = batch_generate(model, tokenizer, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                             max_new_tokens=max_new, temperature=0.0, top_p=1.0, do_sample=False)
        for i, r in enumerate(chunk):
            resp, gold = gen["responses"][i], gold_final(r)
            trunc = bool(gen["truncated"][i])
            recs.append({
                "row_index": s + i,
                "source_index": str(r.get("source_index", r.get("prompt_id", s + i))),
                "policy": policy,
                "question": problem_text(r),
                "gold_final": gold,
                "response": resp,
                "response_tokens": int(gen["response_lengths"][i]),
                "response_words": len(resp.split()),
                "truncated": trunc,
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "pred_final": extract_designated_final(resp),
                "exact": exact_reward(resp, gold),
                **format_check(resp),
                "failure_type": failure_type(resp, gold, trunc),
                **({"svamp_type": r["svamp_type"]} if "svamp_type" in r else {}),
            })
    secs = time.time() - t0
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recs, secs


def run_generation(cfg, dataset, policies, batch_size):
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    timing_path = out_dir(cfg) / "generation_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for p in policies:
        adapter = policy_specs(cfg)[p]
        if adapter is not None and not repo_path(adapter).joinpath("adapter_config.json").exists():
            raise FileNotFoundError(f"{p}: no adapter at {adapter} (run scripts.download_assets)")
        path = gen_path(cfg, dataset, p)
        if path.exists() and len(read_jsonl(path)) == len(rows):
            print(f"[{dataset}/{p}] {path.name} already complete — skipping")
            continue
        recs, secs = generate_for_policy(cfg, rows, tokenizer, p, batch_size)
        write_jsonl(path, recs)
        timing[f"{dataset}/{p}"] = {"seconds": secs, "n": len(recs)}
        timing_path.write_text(json.dumps(timing, indent=1))
        acc = sum(r["exact"] for r in recs) / len(recs)
        print(f"[{dataset}/{p}] saved {len(recs)} responses -> {path.name} | exact acc {acc:.3f} | {secs:.0f}s")


# ------------------------------------------------------------------------------------------------------------ judging
def run_judging(cfg, judge, dataset):
    gens = {p: {r["source_index"]: r for r in read_jsonl(gen_path(cfg, dataset, p))} for p in POLICIES}
    ids = [r["source_index"] for r in read_jsonl(gen_path(cfg, dataset, "sft"))]
    for p in POLICIES:
        assert set(gens[p]) == set(ids), f"{dataset}: {p} was generated on a different prompt set"
    jp = judge_path(cfg, dataset)
    done = {(r["source_index"], r["a_policy"], r["b_policy"]) for r in read_jsonl(jp)} if jp.exists() else set()
    todo = [(sid, a, b) for sid in ids for a, b in PAIRS if (sid, a, b) not in done]
    print(f"[{dataset}] {len(ids)} prompts x {len(PAIRS)} pairs: {len(done)} judged, {len(todo)} to judge")
    with jp.open("a", encoding="utf-8") as f:
        for sid, a, b in tqdm(todo, desc=f"judge {dataset}"):
            ra, rb = gens[a][sid], gens[b][sid]
            rec = judge_record(judge, ra["question"], ra["response"], rb["response"])
            rec.update({"source_index": sid, "a_policy": a, "b_policy": b, "a_exact": ra["exact"], "b_exact": rb["exact"],
                        "verifier_pref": verifier_pref(ra["exact"], rb["exact"])})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer", "both"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    ap.add_argument("--policies", nargs="*", default=POLICIES)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    datasets = DATASETS if args.dataset == "both" else [args.dataset]
    print("Datasets:", datasets, "| policies:", args.policies, "| greedy | max_new_tokens:", cfg["math_max_new_tokens"])
    if args.stage in ("generate", "all"):
        for d in datasets:
            run_generation(cfg, d, args.policies, args.batch_size)
    if args.stage in ("judge", "all"):
        judge = load_judge(cfg)
        print("Fixed pairwise judge loaded:", cfg["ai_judge_model"])
        for d in datasets:
            run_judging(cfg, judge, d)
    print("Done. Aggregate with: python -m task5_feedback.compare_feedback")


if __name__ == "__main__":
    main()
