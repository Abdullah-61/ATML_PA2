from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import full_eval
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    n = int(cfg["short_ablation_examples"])
    rows = []
    for beta in cfg["betas"]:
        name = f"beta{beta}"
        out = f"outputs/task1_dpo/{name}"
        # identical data subset (first n rows, fixed order), seed, optimizer, LoRA; only beta changes
        if not repo_path(out).joinpath("adapter_config.json").exists():  # resume after disconnect
            run_training(args.config, name, output_path=out, beta=beta, max_examples=n)
            clear_gpu()
        rows.append(full_eval(args.config, out, name, beta=beta))  # held-out DPO loss uses the run's own beta
        clear_gpu()
    json.dump(rows, open(repo_path(cfg["results_dir"]) / "beta_ablation_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
