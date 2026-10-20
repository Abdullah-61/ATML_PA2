from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task3_grpo.grpo import ADV_EPS


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng: np.random.Generator | None = None):
    """Partition every prompt's cached completions into disjoint groups of size K.

    The total number of generations is identical for every K (all cached completions are used exactly once):
        K=2 -> 4 groups / prompt,  K=4 -> 2 groups / prompt,  K=8 -> 1 group / prompt.
    rng=None  : contiguous split by generation_index (deterministic).
    rng given : the prompt's completions are randomly permuted first (a random partition).
    Returns a list of (prompt_id, [rows]).
    """
    groups = []
    for pid, rows in by_prompt.items():
        if len(rows) % k:
            raise ValueError(f"K={k} does not divide the {len(rows)} cached completions of prompt {pid}")
        order = np.arange(len(rows)) if rng is None else rng.permutation(len(rows))
        for s in range(0, len(rows), k):
            groups.append((pid, [rows[j] for j in order[s:s + k]]))
    return groups


def difficulty_bins(by_prompt, n_bins: int):
    """Binning rule (defined once): prompt difficulty = mean reward of ALL its cached completions
    (lower mean reward = harder). Prompts are sorted by that mean and cut into n_bins equal-size
    bins; D1 is the hardest bin. Every group inherits the bin of its prompt, for every K."""
    means = {pid: float(np.mean([r["reward"] for r in rows])) for pid, rows in by_prompt.items()}
    order = sorted(means, key=means.get)
    labels = {3: ["hard", "medium", "easy"]}.get(n_bins, [f"D{b + 1}" for b in range(n_bins)])
    bin_of = {}
    for b, chunk in enumerate(np.array_split(np.array(order), n_bins)):
        for pid in chunk:
            bin_of[str(pid)] = labels[b]
    return means, bin_of, labels


def _metrics(groups, thresholds):
    if not groups:
        return {}
    stds, cen, advs = [], [], []
    for _, rows in groups:
        r = np.array([x["reward"] for x in rows], dtype=float)
        sd = r.std()
        stds.append(sd)
        cen.append(r - r.mean())
        advs.append((r - r.mean()) / (sd + ADV_EPS))
    stds = np.array(stds)
    cen, advs = np.concatenate(cen), np.concatenate(advs)
    out = {"n_groups": len(groups), "n_generations": int(sum(len(g) for _, g in groups)),
           "informative_rate": float((stds > ADV_EPS).mean()),
           "mean_group_reward_std": float(stds.mean()),
           "std_of_group_reward_std": float(stds.std()),
           "signal_variance": float((cen ** 2).mean()),        # variance of the un-normalised relative signal r - mu_g
           "advantage_variance": float((advs ** 2).mean())}     # variance of the normalised advantages
    for t in thresholds:
        out[f"informative_rate_std>{t:g}"] = float((stds > t).mean())
    return out


def analyze(cfg, by_prompt):
    ks = [int(k) for k in cfg["group_sizes"]]
    thresholds = [float(t) for t in cfg["informative_std_thresholds"]]
    n_bins, n_part, seed = int(cfg["difficulty_bins"]), int(cfg["regroup_partitions"]), int(cfg["seed"])
    means, bin_of, labels = difficulty_bins(by_prompt, n_bins)

    def summarize(groups):
        res = {"overall": _metrics(groups, thresholds)}
        for lab in labels:
            res[lab] = _metrics([g for g in groups if bin_of[g[0]] == lab], thresholds)
        return res

    results = {"rule": {"informative": f"group reward std > {ADV_EPS:g}",
                        "difficulty": "prompt mean reward over all cached completions; sorted ascending; "
                                      f"{n_bins} equal-size bins; D1/hard = lowest mean reward",
                        "n_random_partitions": n_part, "prompt_difficulty": means, "prompt_bin": bin_of,
                        "total_generations": sum(len(g) for g in by_prompt.values())},
               "contiguous": {}, "random_mean": {}, "random_sd": {}}
    for k in ks:
        results["contiguous"][k] = summarize(regroup_equal_generation_budget(by_prompt, k))
        reps = [summarize(regroup_equal_generation_budget(by_prompt, k, np.random.default_rng(seed + r)))
                for r in range(n_part)]
        mean, sd = {}, {}
        for part in reps[0]:
            keys = [m for m in reps[0][part] if isinstance(reps[0][part][m], (int, float))]
            mean[part] = {m: float(np.mean([r[part][m] for r in reps])) for m in keys}
            sd[part] = {m: float(np.std([r[part][m] for r in reps])) for m in keys}
        results["random_mean"][k], results["random_sd"][k] = mean, sd

    # sensitivity of the group reward spread to prompt difficulty (Spearman rho over prompts, per K)
    sens = {}
    pids = list(by_prompt)
    diff = pd.Series([means[p] for p in pids], index=pids).rank()
    for k in ks:
        spread = {p: [] for p in pids}
        for r in range(n_part):
            for pid, rows in regroup_equal_generation_budget(by_prompt, k, np.random.default_rng(seed + r)):
                spread[pid].append(np.std([x["reward"] for x in rows]))
        s = pd.Series([np.mean(spread[p]) for p in pids], index=pids).rank()
        sens[k] = float(np.corrcoef(diff, s)[0, 1])
    results["spearman_difficulty_vs_group_std"] = sens
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    first = next(iter(by_prompt.values()))
    print("Cached prompts:", len(by_prompt), "| completions/prompt:", len(first),
          "| total generations:", sum(len(g) for g in by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))
    res = analyze(cfg, by_prompt)
    out = repo_path(cfg["results_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "group_size_study.json").write_text(json.dumps(res, indent=1))
    rows = []
    for k, parts in res["random_mean"].items():
        for part, m in parts.items():
            rows.append({"K": k, "bin": part, **{a: round(b, 4) for a, b in m.items()}})
    print(pd.DataFrame(rows).to_string(index=False))
    print("Spearman(difficulty, group std):", res["spearman_difficulty_vs_group_std"])


if __name__ == "__main__":
    main()
