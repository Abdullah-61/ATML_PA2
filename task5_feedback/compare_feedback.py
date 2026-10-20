"""Aggregate Task 5: in-domain (GSM8K), controlled reward diagnostics, out-of-domain (SVAMP), cost.  Torch-free."""
from __future__ import annotations

import argparse
import json
import timeit

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.metrics_t5 import (bootstrap_ci, cohen_kappa, group_win_rates, mcnemar_exact, point_biserial,
                                       wilson)
from task5_feedback.rlvr import exact_reward

POLICIES = ["sft", "rlvr", "rlaif"]
PAIRS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]
FAIL_TYPES = ["correct", "wrong_final", "gold_in_reasoning_wrong_final", "missing_final", "truncated_no_final"]


def out_dir(cfg):
    p = repo_path(cfg["results_dir"]) / "task5_feedback"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_gens(cfg, dataset) -> pd.DataFrame:
    frames = []
    for p in POLICIES:
        path = out_dir(cfg) / f"gen_{dataset}_{p}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run task5_feedback.evaluate_math --dataset {dataset}")
        frames.append(pd.DataFrame(read_jsonl(path)))
    return pd.concat(frames, ignore_index=True)


def load_judgments(cfg, dataset) -> pd.DataFrame:
    path = out_dir(cfg) / f"judge_{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run task5_feedback.evaluate_math --dataset {dataset} --stage judge")
    return pd.DataFrame(read_jsonl(path))


# ------------------------------------------------------------------------------------------------- policy-level table
def policy_table(gens: pd.DataFrame, judg: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rows = []
    for p in POLICIES:
        d = gens[gens.policy == p]
        n, k = len(d), int(d.exact.sum())
        lo, hi = wilson(k, n)
        row = {"policy": p, "n": n, "exact_acc": k / n, "exact_acc_lo": lo, "exact_acc_hi": hi,
               "has_final_rate": d.has_final.mean(), "strict_format_rate": d.strict_format.mean(),
               "truncated_rate": d.truncated.mean(), "mean_tokens": d.response_tokens.mean(),
               "median_tokens": d.response_tokens.median(), "mean_words": d.response_words.mean()}
        if p == "sft":
            row.update({"win_vs_sft": 0.5, "win_vs_sft_lo": np.nan, "win_vs_sft_hi": np.nan,
                        "W": np.nan, "T": np.nan, "L": np.nan})
        else:
            j = judg[(judg.a_policy == p) & (judg.b_policy == "sft")]
            s = j.judge_pref.map({"A": 1.0, "TIE": 0.5, "B": 0.0}).values
            lo, hi = bootstrap_ci(s, seed=seed)
            row.update({"win_vs_sft": float(s.mean()) if len(s) else np.nan, "win_vs_sft_lo": lo, "win_vs_sft_hi": hi,
                        "W": int((j.judge_pref == "A").sum()), "T": int((j.judge_pref == "TIE").sum()),
                        "L": int((j.judge_pref == "B").sum())})
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


def head_to_head(judg: pd.DataFrame, seed: int = 0) -> dict:
    j = judg[(judg.a_policy == "rlvr") & (judg.b_policy == "rlaif")]
    s = j.judge_pref.map({"A": 1.0, "TIE": 0.5, "B": 0.0}).values
    lo, hi = bootstrap_ci(s, seed=seed)
    return {"rlvr_win_rate_vs_rlaif": float(s.mean()) if len(s) else float("nan"), "lo": lo, "hi": hi,
            "rlvr_wins": int((j.judge_pref == "A").sum()), "ties": int((j.judge_pref == "TIE").sum()),
            "rlaif_wins": int((j.judge_pref == "B").sum()), "n": len(j)}


def paired_accuracy_tests(gens: pd.DataFrame) -> pd.DataFrame:
    wide = gens.pivot(index="source_index", columns="policy", values="exact").astype(bool)
    rows = []
    for a, b in PAIRS:
        only_a, only_b = int((wide[a] & ~wide[b]).sum()), int((~wide[a] & wide[b]).sum())
        rows.append({"A": a, "B": b, "acc_A": wide[a].mean(), "acc_B": wide[b].mean(), "only_A_correct": only_a,
                     "only_B_correct": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ verifier vs judge agreement
def agreement_table(judg: pd.DataFrame) -> pd.DataFrame:
    """Per comparison: 3-way agreement / kappa between verifier preference (exact_A vs exact_B) and judge preference.
    'decisive' = exactly one response correct (verifier has an opinion); 'verifier_tie' = both right or both wrong."""
    rows = []
    groups = [(f"{a} vs {b}", judg[(judg.a_policy == a) & (judg.b_policy == b)]) for a, b in PAIRS] + [("pooled", judg)]
    for name, j in groups:
        dec, tie = j[j.verifier_pref != "TIE"], j[j.verifier_pref == "TIE"]
        both_right = tie[tie.a_exact == 1]
        both_wrong = tie[tie.a_exact == 0]
        rows.append({
            "comparison": name, "n": len(j),
            "agreement_3way": float((j.verifier_pref == j.judge_pref).mean()) if len(j) else np.nan,
            "cohen_kappa": cohen_kappa(j.verifier_pref, j.judge_pref),
            "n_decisive": len(dec),
            "judge_picks_correct": float((dec.judge_pref == dec.verifier_pref).mean()) if len(dec) else np.nan,
            "judge_tie_when_decisive": float((dec.judge_pref == "TIE").mean()) if len(dec) else np.nan,
            "judge_picks_incorrect": float(((dec.judge_pref != "TIE") & (dec.judge_pref != dec.verifier_pref)).mean()) if len(dec) else np.nan,
            "n_verifier_tie": len(tie),
            "judge_decisive_when_verifier_tie": float((tie.judge_pref != "TIE").mean()) if len(tie) else np.nan,
            "n_both_correct": len(both_right),
            "judge_decisive_both_correct": float((both_right.judge_pref != "TIE").mean()) if len(both_right) else np.nan,
            "n_both_wrong": len(both_wrong),
            "judge_decisive_both_wrong": float((both_wrong.judge_pref != "TIE").mean()) if len(both_wrong) else np.nan,
        })
    return pd.DataFrame(rows).set_index("comparison")


def group_reward_table(gens: pd.DataFrame, judg: pd.DataFrame):
    """Per prompt, the three pairwise judgments define a K=3 direct-RLAIF group reward for each policy's response.
    Compare it with the binary verifier reward on the same responses."""
    idx = {p: i for i, p in enumerate(POLICIES)}
    recs = []
    for sid, j in judg.groupby("source_index"):
        if len(j) < len(PAIRS):
            continue
        prefs = {}
        for _, r in j.iterrows():
            i, k = idx[r.a_policy], idx[r.b_policy]
            p = r.judge_pref
            if i > k:
                i, k, p = k, i, {"A": "B", "B": "A", "TIE": "TIE"}[p]
            prefs[(i, k)] = p
        gw = group_win_rates(3, prefs)
        for p in POLICIES:
            recs.append({"source_index": sid, "policy": p, "rlaif_group_reward": gw[idx[p]]})
    g = pd.DataFrame(recs).merge(gens[["source_index", "policy", "exact", "response_words"]], on=["source_index", "policy"])
    summary = {"n_responses": len(g),
               "corr_group_reward_vs_exact": point_biserial(g.rlaif_group_reward, g.exact),
               "corr_group_reward_vs_words": point_biserial(g.rlaif_group_reward, g.response_words),
               "mean_group_reward_correct": float(g.loc[g.exact == 1, "rlaif_group_reward"].mean()),
               "mean_group_reward_incorrect": float(g.loc[g.exact == 0, "rlaif_group_reward"].mean()),
               "mean_group_reward_by_policy": g.groupby("policy").rlaif_group_reward.mean().round(4).to_dict()}
    return g, summary


def failure_table(gens: pd.DataFrame) -> pd.DataFrame:
    return (pd.crosstab(gens.policy, gens.failure_type, normalize="index")
            .reindex(index=POLICIES, columns=FAIL_TYPES, fill_value=0.0))


def judge_position_bias(judg: pd.DataFrame) -> dict:
    j = judg[~judg.identical.astype(bool)]
    nt = j[j.picked_slot != "TIE"]
    return {"n_nonidentical": len(j), "tie_rate": float((j.judge_pref == "TIE").mean()) if len(j) else np.nan,
            "picked_first_shown_rate_non_tie": float((nt.picked_slot == "first").mean()) if len(nt) else np.nan,
            "identical_pairs": int(judg.identical.astype(bool).sum())}


# ------------------------------------------------------------------------------------------- controlled diagnostics
def diagnostic_tables(cfg):
    od = out_dir(cfg)
    if not (od / "diag_pairs.jsonl").exists():
        raise FileNotFoundError("diag_pairs.jsonl missing — run task5_feedback.score_perturbations")
    pairs = pd.DataFrame(read_jsonl(od / "diag_pairs.jsonl"))
    var = pd.DataFrame(read_jsonl(od / "diag_variants.jsonl"))
    rows = []
    for (name, fam), d in pairs.groupby(["pair", "family"], sort=False):
        row = {"pair": name, "family": fam, "n": len(d), "has_ground_truth": d.better.notna().all()}
        for mech, col, pcol in [("verifier", "verifier_outcome", "verifier_pref"), ("judge", "judge_outcome", "judge_pref")]:
            if row["has_ground_truth"]:
                for o in ("better", "tie", "wrong"):
                    row[f"{mech}_{o}"] = float((d[col] == o).mean())
            else:   # no diagnostic ground truth: report which side the mechanism favours
                row[f"{mech}_prefers_first"] = float((d[pcol] == "A").mean())
                row[f"{mech}_tie"] = float((d[pcol] == "TIE").mean())
                row[f"{mech}_prefers_second"] = float((d[pcol] == "B").mean())
        rows.append(row)
    pair_tab = pd.DataFrame(rows).set_index("pair")

    order = ["clean", "reason_corrupt", "wrong_final", "filler", "gold_distractor"]
    var_tab = (var.groupby("variant").agg(n=("verifier_reward", "size"), verifier_reward=("verifier_reward", "mean"),
                                          rlaif_group_reward=("rlaif_group_reward", "mean"),
                                          rlaif_group_reward_sd=("rlaif_group_reward", "std"),
                                          mean_words=("response_words", "mean")).reindex(order))
    # rank of each variant within its problem under the RLAIF group reward (1 = best)
    var["rlaif_rank"] = var.groupby("problem_id").rlaif_group_reward.rank(ascending=False, method="average")
    var_tab["mean_rlaif_rank"] = var.groupby("variant").rlaif_rank.mean().reindex(order)
    # how often each variant gets the (possibly shared) highest RLAIF group reward in its problem
    mx = var.groupby("problem_id").rlaif_group_reward.transform("max")
    n_at_max = var.assign(m=var.rlaif_group_reward == mx).groupby("problem_id").m.transform("sum")
    var_tab["top_or_tied_rate"] = var.assign(t=var.rlaif_group_reward == mx).groupby("variant").t.mean().reindex(order)
    var_tab["strict_top_rate"] = var.assign(t=(var.rlaif_group_reward == mx) & (n_at_max == 1)).groupby("variant").t.mean().reindex(order)

    def S(pair, mech):
        d = pairs[pairs.pair == pair]
        col = f"{mech}_outcome"
        return {"S": float((d[col] == "better").mean()), "tie": float((d[col] == "tie").mean()),
                "wrong": float((d[col] == "wrong").mean()), "n": len(d)}

    sens = {}
    for mech in ("verifier", "judge"):
        sens[mech] = {
            "S_reason (clean > reason_corrupt)": S("clean>reason_corrupt", mech),
            "S_outcome (clean > wrong_final)": S("clean>wrong_final", mech),
            "S_outcome_distractor (clean > gold_distractor)": S("clean>gold_distractor", mech),
            "filler (clean > filler)": S("clean>filler", mech),
        }
        d = pairs[pairs.pair.isin(["clean>wrong_final", "clean>gold_distractor"])]
        sens[mech]["S_outcome_pooled"] = {"S": float((d[f"{mech}_outcome"] == "better").mean()),
                                          "tie": float((d[f"{mech}_outcome"] == "tie").mean()),
                                          "wrong": float((d[f"{mech}_outcome"] == "wrong").mean()), "n": len(d)}
    sens_tab = pd.DataFrame({(m, k): v for m in sens for k, v in sens[m].items()}).T
    sens_tab.index.names = ["mechanism", "quantity"]
    return pairs, var, pair_tab, var_tab, sens_tab


# --------------------------------------------------------------------------------------------------------- transfer
def transfer_drop(in_tab: pd.DataFrame, ood_tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in POLICIES:
        a, b = in_tab.loc[p], ood_tab.loc[p]
        rows.append({"policy": p, "acc_gsm": a.exact_acc, "acc_svamp": b.exact_acc,
                     "acc_drop": a.exact_acc - b.exact_acc,
                     "acc_rel_drop": (a.exact_acc - b.exact_acc) / a.exact_acc if a.exact_acc else np.nan,
                     "acc_gain_vs_sft_gsm": a.exact_acc - in_tab.loc["sft"].exact_acc,
                     "acc_gain_vs_sft_svamp": b.exact_acc - ood_tab.loc["sft"].exact_acc,
                     "win_vs_sft_gsm": a.win_vs_sft, "win_vs_sft_svamp": b.win_vs_sft,
                     "win_drop": a.win_vs_sft - b.win_vs_sft if p != "sft" else np.nan,
                     "strict_format_gsm": a.strict_format_rate, "strict_format_svamp": b.strict_format_rate,
                     "mean_tokens_gsm": a.mean_tokens, "mean_tokens_svamp": b.mean_tokens})
    return pd.DataFrame(rows).set_index("policy")


# ------------------------------------------------------------------------------------------------------------- cost
def cost_table(cfg, gens_all: list[pd.DataFrame], judg_all: list[pd.DataFrame], diag_pairs: pd.DataFrame | None):
    j = pd.concat([*judg_all] + ([diag_pairs] if diag_pairs is not None else []), ignore_index=True)
    fresh = j[(~j.cached.astype(bool)) & (~j.identical.astype(bool))]
    g = pd.concat(gens_all, ignore_index=True)
    sample = g[["response", "gold_final"]].head(500).values.tolist()
    n_rep = 20
    t = timeit.timeit(lambda: [exact_reward(r, gld) for r, gld in sample], number=n_rep)
    ver_us = 1e6 * t / (n_rep * max(len(sample), 1))
    judge_s = float(fresh.secs.mean()) if len(fresh) else float("nan")
    timing_path = out_dir(cfg) / "generation_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    return {"verifier_us_per_check": ver_us, "judge_s_per_pairwise_call": judge_s,
            "judge_calls_timed": int(len(fresh)), "judge_over_verifier_ratio": judge_s * 1e6 / ver_us if ver_us else np.nan,
            "judge_calls_per_rlaif_group(K)": {K: K * (K - 1) // 2 for K in (4, 8, 16)},
            "est_judge_s_per_group(K=8)": 28 * judge_s, "verifier_checks_per_group(K=8)": 8,
            "generation_timing_s": timing}


# ------------------------------------------------------------------------------------------------------------ driver
def run_all(cfg):
    od = out_dir(cfg)
    res, out = {}, {}
    seed = int(cfg.get("seed", 0))
    gens, judg = {}, {}
    for d in ("gsm", "transfer"):
        try:
            gens[d], judg[d] = load_gens(cfg, d), load_judgments(cfg, d)
        except FileNotFoundError as e:
            print("skip", d, "->", e)
            continue
        tab = policy_table(gens[d], judg[d], seed)
        agr = agreement_table(judg[d])
        g, gsum = group_reward_table(gens[d], judg[d])
        tests = paired_accuracy_tests(gens[d])
        fail = failure_table(gens[d])
        h2h = head_to_head(judg[d], seed)
        pos = judge_position_bias(judg[d])
        tab.to_csv(od / f"{d}_policy_table.csv"); agr.to_csv(od / f"{d}_agreement.csv")
        tests.to_csv(od / f"{d}_paired_accuracy_tests.csv", index=False); fail.to_csv(od / f"{d}_failure_types.csv")
        g.to_csv(od / f"{d}_group_rewards.csv", index=False)
        out[d] = {"policy": tab, "agreement": agr, "group": g, "group_summary": gsum, "tests": tests, "failures": fail,
                  "h2h": h2h, "position": pos}
        res[d] = {"policy": tab.round(4).to_dict("index"), "agreement": agr.round(4).to_dict("index"),
                  "group_reward": gsum, "paired_accuracy_tests": tests.round(4).to_dict("records"),
                  "failure_types": fail.round(4).to_dict("index"), "head_to_head": h2h, "judge_position": pos}
    if "gsm" in out and "transfer" in out:
        drop = transfer_drop(out["gsm"]["policy"], out["transfer"]["policy"])
        drop.to_csv(od / "transfer_drop.csv")
        out["drop"] = drop
        res["transfer_drop"] = drop.round(4).to_dict("index")
    diag_pairs = None
    try:
        diag_pairs, diag_var, pair_tab, var_tab, sens_tab = diagnostic_tables(cfg)
        pair_tab.to_csv(od / "diag_pair_table.csv"); var_tab.to_csv(od / "diag_variant_table.csv")
        sens_tab.to_csv(od / "diag_sensitivity.csv")
        out["diag"] = {"pairs": diag_pairs, "variants": diag_var, "pair_table": pair_tab, "variant_table": var_tab,
                       "sensitivity": sens_tab, "position": judge_position_bias(diag_pairs)}
        res["diagnostics"] = {"pairs": pair_tab.round(4).to_dict("index"), "variants": var_tab.round(4).to_dict("index"),
                              "sensitivity": {f"{a} | {b}": v for (a, b), v in sens_tab.round(4).to_dict("index").items()},
                              "judge_position": out["diag"]["position"]}
    except FileNotFoundError as e:
        print("skip diagnostics ->", e)
    if gens:
        out["cost"] = cost_table(cfg, list(gens.values()), list(judg.values()), diag_pairs)
        res["cost"] = out["cost"]
    (od / "task5_summary.json").write_text(json.dumps(res, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = run_all(cfg)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for d in ("gsm", "transfer"):
        if d in out:
            print(f"\n===== {d}: policy table =====\n", out[d]["policy"].round(3).to_string())
            print(f"\n===== {d}: verifier-judge agreement =====\n", out[d]["agreement"].round(3).to_string())
    if "drop" in out:
        print("\n===== transfer drop =====\n", out["drop"].round(3).to_string())
    if "diag" in out:
        print("\n===== diagnostic pairs =====\n", out["diag"]["pair_table"].round(3).to_string())
        print("\n===== sensitivities =====\n", out["diag"]["sensitivity"].round(3).to_string())
    if "cost" in out:
        print("\n===== cost =====\n", json.dumps(out["cost"], indent=1, default=float))
    print("\nSaved:", out_dir(cfg) / "task5_summary.json")


if __name__ == "__main__":
    main()
