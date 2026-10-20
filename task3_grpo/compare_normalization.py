from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.metrics import safe_corr
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import ensure_eval

LOSS_TYPES = {"grpo": "canonical_grpo", "dr_grpo": "dr_grpo"}


def ensure_fork(config_path: str, loss_type: str):
    cfg = load_yaml(config_path)
    name = LOSS_TYPES[loss_type]
    out = f"outputs/task3_grpo/{name}"
    if repo_path(out).joinpath("adapter_config.json").exists():
        print(f"Fork {name} exists — skipping")
    else:
        run_grpo(config_path, out, int(cfg["fork_updates"]), loss_type, name)
    return name, out


def summarize_normalization(cfg):
    """Length-conditioned statistics from the per-completion logs of the two forks.

    Rule (defined once): 'short' / 'long' = below / above the pooled median completion length of all
    trained (unmasked) completions of BOTH forks. grad_norm is the gradient-norm contribution of one completion
    to the update; grad_norm_per_adv divides it by |A| to remove the advantage magnitude.
    """
    res_dir = repo_path(cfg["results_dir"])
    frames = {}
    for lt, name in LOSS_TYPES.items():
        df = pd.DataFrame(read_jsonl(res_dir / f"seqstats_{name}.jsonl"))
        frames[name] = df[~df["masked"]].copy()
    median = float(np.median(np.concatenate([f["length"].values for f in frames.values()])))
    out = {"pooled_median_length": median}
    for name, df in frames.items():
        df["long"] = df["length"] > median
        inf = df[df["adv"].abs() > 0].copy()
        inf["gn_per_adv"] = inf["grad_norm"] / inf["adv"].abs()
        sh, lo = inf[~inf["long"]], inf[inf["long"]]
        total_mass = df["mass"].sum()
        out[name] = {
            "n_completions": int(len(df)), "n_informative": int(len(inf)),
            "mean_length": float(df["length"].mean()),
            "corr_length_gradnorm": safe_corr(inf["length"], inf["grad_norm"]),
            "corr_length_gradnorm_per_adv": safe_corr(inf["length"], inf["gn_per_adv"]),
            "gradnorm_short": float(sh["grad_norm"].mean()) if len(sh) else float("nan"),
            "gradnorm_long": float(lo["grad_norm"].mean()) if len(lo) else float("nan"),
            "gradnorm_long_over_short": float(lo["grad_norm"].mean() / sh["grad_norm"].mean())
            if len(sh) and len(lo) else float("nan"),
            "gn_per_adv_long_over_short": float(lo["gn_per_adv"].mean() / sh["gn_per_adv"].mean())
            if len(sh) and len(lo) else float("nan"),
            "per_token_weight_short": float(df[~df["long"]]["tok_weight"].mean()),
            "per_token_weight_long": float(df[df["long"]]["tok_weight"].mean()),
            "weight_mass_share_long": float(df[df["long"]]["mass"].sum() / total_mass) if total_mass > 0 else float("nan"),
        }
    (res_dir / "normalization_summary.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame({n: out[n] for n in LOSS_TYPES.values()}).round(5).to_string())
    print("pooled median length:", median)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--analysis-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"], "| K:", cfg["num_generations"], "| eps:", cfg["clip_epsilon"],
          "| beta:", cfg["kl_beta"], "| same midpoint, seed, prompts, generation settings for both")
    if not args.analysis_only:
        for lt in LOSS_TYPES:
            name, out = ensure_fork(args.config, lt)
            ensure_eval(args.config, out, name)
    summarize_normalization(cfg)


if __name__ == "__main__":
    main()
