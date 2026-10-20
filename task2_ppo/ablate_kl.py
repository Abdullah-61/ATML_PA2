from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import ensure_fork
from task2_ppo.evaluate import ensure_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps = float(cfg["clip_epsilon"])
    print("KL beta conditions:", cfg["kl_values"], "| fork updates:", cfg["fork_updates"], "| eps fixed at", eps)
    rows = []
    for kl in cfg["kl_values"]:
        name, out = ensure_fork(args.config, eps, float(kl))
        rows.append(ensure_eval(args.config, out, name))
    (repo_path(cfg["results_dir"]) / "kl_ablation_summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
