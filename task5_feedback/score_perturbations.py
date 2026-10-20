from __future__ import annotations

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from task5_feedback.metrics_t5 import (group_win_rates, judge_record, pair_outcome, problem_text, gold_final,
                                       response_text, verifier_pref)
from task5_feedback.rlvr import exact_reward, extract_designated_final

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
# Fixed order: pairs below are (i, j) with i < j in this order, so the 10 judgments are exactly the ones
# PairwiseAIJudge.group_rewards would make for a K=5 group (shared cache) -> per-variant direct-RLAIF reward.
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                 "persuasive_filler_correct", "gold_distractor_wrong_final"]
SHORT = {"clean_correct": "clean", "corrupt_reasoning_correct_final": "reason_corrupt",
         "good_reasoning_wrong_final": "wrong_final", "persuasive_filler_correct": "filler",
         "gold_distractor_wrong_final": "gold_distractor"}
# (first, second, diagnostically better side or None, pair name, family)
CONTROLLED_PAIRS = [
    ("clean_correct", "corrupt_reasoning_correct_final", "A", "clean>reason_corrupt", "reasoning sensitivity"),
    ("clean_correct", "good_reasoning_wrong_final", "A", "clean>wrong_final", "outcome sensitivity"),
    ("clean_correct", "persuasive_filler_correct", "A", "clean>filler", "style / verbosity"),
    ("clean_correct", "gold_distractor_wrong_final", "A", "clean>gold_distractor", "outcome sensitivity (distractor)"),
    ("corrupt_reasoning_correct_final", "good_reasoning_wrong_final", None, "reason_corrupt vs wrong_final", "process vs outcome"),
    ("corrupt_reasoning_correct_final", "persuasive_filler_correct", "B", "filler>reason_corrupt", "cross"),
    ("corrupt_reasoning_correct_final", "gold_distractor_wrong_final", None, "reason_corrupt vs gold_distractor", "process vs outcome"),
    ("good_reasoning_wrong_final", "persuasive_filler_correct", "B", "filler>wrong_final", "style vs outcome"),
    ("good_reasoning_wrong_final", "gold_distractor_wrong_final", None, "wrong_final vs gold_distractor", "distractor (no gold)"),
    ("persuasive_filler_correct", "gold_distractor_wrong_final", "A", "filler>gold_distractor", "style vs outcome"),
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def _problem_and_gold(variants: dict):
    """Problem text / gold are shared by the 5 variants; take them from any variant that carries them."""
    prob = gold = None
    for v in VARIANT_ORDER:
        r = variants[v]
        if prob is None:
            try:
                prob = problem_text(r)
            except KeyError:
                pass
        if gold is None:
            try:
                gold = gold_final(r)
            except KeyError:
                pass
    if prob is None or gold is None:
        raise KeyError(f"diagnostic rows need a problem text and gold answer; keys: {sorted(variants[VARIANT_ORDER[0]])}")
    return prob, gold


def score_diagnostics(groups, judge):
    """Returns (pair_records, variant_records)."""
    pairs, variants_out = [], []
    idx = {v: i for i, v in enumerate(VARIANT_ORDER)}
    for pid in tqdm(sorted(groups, key=lambda x: (len(x), x)), desc="diagnostic problems"):
        g = groups[pid]
        problem, gold = _problem_and_gold(g)
        resp = {v: response_text(g[v]) for v in VARIANT_ORDER}
        rew = {v: exact_reward(resp[v], gold) for v in VARIANT_ORDER}
        prefs = {}
        for first, second, better, name, family in CONTROLLED_PAIRS:
            j = judge_record(judge, problem, resp[first], resp[second])
            vp = verifier_pref(rew[first], rew[second])
            prefs[(idx[first], idx[second])] = j["judge_pref"]
            pairs.append({"problem_id": pid, "pair": name, "family": family, "first": SHORT[first], "second": SHORT[second],
                          "better": better, "verifier_first": rew[first], "verifier_second": rew[second],
                          "verifier_pref": vp, "verifier_outcome": pair_outcome(vp, better),
                          "judge_outcome": pair_outcome(j["judge_pref"], better), **j})
        gw = group_win_rates(len(VARIANT_ORDER), prefs)
        for v in VARIANT_ORDER:
            variants_out.append({"problem_id": pid, "variant": SHORT[v], "variant_type": v, "gold_final": gold,
                                 "pred_final": extract_designated_final(resp[v]), "verifier_reward": rew[v],
                                 "rlaif_group_reward": gw[idx[v]], "response_words": len(resp[v].split()),
                                 "response": resp[v], "problem": problem})
    return pairs, variants_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups), "| variants/problem:", len(VARIANT_ORDER), "| judged pairs/problem:", len(CONTROLLED_PAIRS))
    from task5_feedback.evaluate_math import load_judge, out_dir   # lazy: keeps this module importable without a GPU
    judge = load_judge(cfg)
    pairs, variants = score_diagnostics(groups, judge)
    od = out_dir(cfg)
    write_jsonl(od / "diag_pairs.jsonl", pairs)
    write_jsonl(od / "diag_variants.jsonl", variants)
    vr = defaultdict(list)
    for r in variants:
        vr[r["variant"]].append(r["verifier_reward"])
    print("Verifier reward by variant:", {k: sum(v) / len(v) for k, v in vr.items()})
    print("Saved", od / "diag_pairs.jsonl", "and", od / "diag_variants.jsonl")


if __name__ == "__main__":
    main()
