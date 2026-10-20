from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import full_eval, stratified_eval
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--standard-adapter", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    std = args.standard_adapter or cfg["standard_output"]
    bal = cfg["length_output"]
    if not args.skip_train and not repo_path(bal).joinpath("adapter_config.json").exists():
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=bal)
        clear_gpu()
    summary = {}
    for name, ad in (("standard", std), ("length_balanced", bal)):
        summary[name] = {"stratified": stratified_eval(args.config, ad, name),
                         "generation": full_eval(args.config, ad, f"{name}_lenstudy")}
        clear_gpu()
    json.dump(summary, open(repo_path(cfg["results_dir"]) / "length_study_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
