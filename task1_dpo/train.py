from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.generation import response_sequence_logprobs
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_token_len(tokenizer, row) -> int:
    ids = tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list):  # BatchEncoding in some transformers versions
        ids = ids["input_ids"]
    return len(ids)


def filter_fitting(tokenizer, rows, max_length):
    """The starter encoder raises when the prompt alone >= max_length. Course data must not be edited, so
    over-long-prompt pairs are skipped at load time (identically for every condition) and reported.
    Returns (kept_rows, kept_source_indices, skipped_source_indices)."""
    kept, skipped = [], []
    for i, row in enumerate(rows):
        (kept if prompt_token_len(tokenizer, row) < max_length else skipped).append(i)
    return [rows[i] for i in kept], kept, skipped


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    tokenizer = load_tokenizer(cfg["base_model"])
    rows, kept_idx, skipped_idx = filter_fitting(tokenizer, read_jsonl(path), int(cfg["max_sequence_length"]))
    if max_examples is not None:  # first n pairs that fit -> same subset for every beta
        rows, kept_idx = rows[: int(max_examples)], kept_idx[: int(max_examples)]
    print(f"[data] {path}: kept {len(rows)} pairs, skipped {len(skipped_idx)} with prompt >= "
          f"{cfg['max_sequence_length']} tokens")
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "kept_indices": kept_idx,
        "skipped_indices": skipped_idx,
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, loader, opt = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    dev = next(model.parameters()).device
    log_path = repo_path(cfg["results_dir"]) / f"train_log_{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    json.dump({"dataset": dataset_path or cfg["paths"]["dpo_standard_train"], "n_kept": len(bundle["kept_indices"]),
               "n_skipped_overlong_prompt": len(bundle["skipped_indices"]), "kept_source_indices": bundle["kept_indices"],
               "skipped_source_indices": bundle["skipped_indices"]},
              open(log_path.parent / f"data_filter_{run_name}.json", "w"))

    def to_dev(b):
        return {k: v.to(dev) for k, v in b.items()}

    window, step, n_batches = [], 0, len(loader)
    model.train()
    for it, (chosen, rejected) in enumerate(loader):
        chosen, rejected = to_dev(chosen), to_dev(rejected)
        # frozen reference = same weights with the LoRA adapter disabled
        with torch.no_grad(), reference_mode(model):
            ref_c, _, _ = response_sequence_logprobs(model, chosen)
            ref_r, _, _ = response_sequence_logprobs(model, rejected)
        pol_c, _, _ = response_sequence_logprobs(model, chosen)
        pol_r, _, _ = response_sequence_logprobs(model, rejected)
        loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        (loss / accum).backward()
        window.append((float(loss.detach()), float(diag["preference_accuracy"]), float(diag["logit_mean"])))
        if (it + 1) % accum == 0 or it + 1 == n_batches:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {"step": step, "examples_seen": (it + 1) * int(cfg["batch_size"]), "beta": beta,
                   "loss": sum(w[0] for w in window) / len(window),
                   "train_pref_acc": sum(w[1] for w in window) / len(window),
                   "logit_mean": sum(w[2] for w in window) / len(window)}
            window = []
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if step % 5 == 0:
                print(rec, flush=True)
    log_f.close()
    model.save_pretrained(str(output))
    json.dump({"run_name": run_name, "beta": beta, "n_examples": len(bundle["rows"]), "seed": cfg["seed"],
               "dataset": dataset_path or cfg["paths"]["dpo_standard_train"], "lr": cfg["learning_rate"],
               "batch_size": cfg["batch_size"], "grad_accum_steps": accum, "optimizer_steps": step},
              open(output / "run_meta.json", "w"), indent=1)
    print("saved adapter to", output)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
