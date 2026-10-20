from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path, read_jsonl, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def generated_path(cfg, policy_name: str):
    return repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy_name}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", help="subset of sft dpo ppo grpo (default: all four)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    specs = policy_specs(cfg)
    names = args.policies or list(specs)
    n_expected = len(load_xstest(cfg))
    print("Policies:", names, "| XSTest rows:", n_expected, "| max_new_tokens:", cfg["safety_max_new_tokens"],
          "| greedy decoding")

    for name in names:
        adapter = specs[name]
        if adapter is not None and not repo_path(adapter).joinpath("adapter_config.json").exists():
            raise FileNotFoundError(f"{name}: no adapter at {adapter} (copy the frozen standard adapter there first)")
        path = generated_path(cfg, name)
        if path.exists() and len(read_jsonl(path)) == n_expected:
            print(f"[{name}] {path.name} already complete — skipping")
            continue
        print(f"[{name}] generating with adapter: {adapter}")
        records = generate_for_policy(cfg, name)
        write_jsonl(path, records)
        print(f"[{name}] saved {len(records)} responses -> {path}")

    # sanity: identical prompt order across policies
    orders = {n: [r["xstest_id"] for r in read_jsonl(generated_path(cfg, n))] for n in names
              if generated_path(cfg, n).exists()}
    ref = next(iter(orders.values()))
    assert all(o == ref for o in orders.values()), "prompt order differs between policies"
    print("OK — identical fixed prompt order for", list(orders))


if __name__ == "__main__":
    main()
