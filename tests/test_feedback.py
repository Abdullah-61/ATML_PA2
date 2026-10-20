"""CPU-only tests for the Task 5 analysis code (fake judge, synthetic generations)."""
import hashlib
import json
import math

import pandas as pd

from task5_feedback.metrics_t5 import (cohen_kappa, failure_type, format_check, group_win_rates, judge_record,
                                       mcnemar_exact, pair_outcome, verifier_pref, wilson)
from task5_feedback.rlvr import exact_reward


class FakeJudge:
    """Mimics PairwiseAIJudge's interface: prefers the response containing the gold '#### 42', else TIE."""
    def __init__(self):
        self.cache, self.cfg = {}, {"ai_judge_model": "fake"}

    def _key(self, problem, a, b):
        return hashlib.sha256(json.dumps({"p": problem, "a": a, "b": b}, sort_keys=True).encode()).hexdigest()

    def compare(self, problem, a, b):
        k = self._key(problem, a, b)
        if k not in self.cache:
            ga, gb = "#### 42" in a, "#### 42" in b
            self.cache[k] = "A" if ga and not gb else ("B" if gb and not ga else ("A" if len(a) > len(b) else "TIE"))
        return self.cache[k]


def test_verifier_and_format():
    assert exact_reward("so 40+2 = 42\n#### 42", "42") == 1.0
    assert exact_reward("maybe 42? no.\n#### 41", "42") == 0.0          # gold only as discarded intermediate
    assert format_check("x\n#### 42")["strict_format"] and not format_check("#### 4\n#### 42 then more")["strict_format"]
    assert failure_type("we get 42 ... #### 41", "42") == "gold_in_reasoning_wrong_final"
    assert failure_type("so 7", "42") == "missing_final" and failure_type("so 7", "42", truncated=True) == "truncated_no_final"
    assert failure_type("#### 1,000", "1000") == "correct"


def test_pair_helpers_and_stats():
    assert verifier_pref(1, 0) == "A" and verifier_pref(0, 0) == "TIE"
    assert pair_outcome("A", "A") == "better" and pair_outcome("TIE", "A") == "tie" and pair_outcome("B", "A") == "wrong"
    assert pair_outcome("A", None) == "n/a"
    assert group_win_rates(3, {(0, 1): "A", (0, 2): "TIE", (1, 2): "B"}) == [0.75, 0.0, 0.75]
    assert wilson(0, 50)[0] == 0.0 and math.isclose(mcnemar_exact(0, 5), 2 / 32)
    assert cohen_kappa(["A", "B", "TIE"], ["A", "B", "TIE"]) == 1.0
    j = FakeJudge()
    r = judge_record(j, "p", "#### 42", "#### 7")
    assert r["judge_pref"] == "A" and r["picked_slot"] in ("first", "second")
    assert judge_record(j, "p", "same", "same")["identical"]


def _fake_results(tmp_path):
    cfg = {"results_dir": str(tmp_path), "seed": 1, "paths": {}}
    od = tmp_path / "task5_feedback"
    od.mkdir(parents=True)
    j = FakeJudge()
    from task5_feedback.metrics_t5 import failure_type as ft
    for d in ("gsm", "transfer"):
        resp = {"sft": ["#### 42", "#### 3", "x"], "rlvr": ["#### 42", "#### 42", "#### 5"], "rlaif": ["long #### 42", "#### 3", "#### 42"]}
        for p, rs in resp.items():
            with open(od / f"gen_{d}_{p}.jsonl", "w") as f:
                for i, r in enumerate(rs):
                    f.write(json.dumps({"source_index": str(i), "policy": p, "question": "q", "gold_final": "42", "response": r,
                                        "response_tokens": len(r), "response_words": len(r.split()), "truncated": False,
                                        "exact": exact_reward(r, "42"), **format_check(r), "failure_type": ft(r, "42")}) + "\n")
        with open(od / f"judge_{d}.jsonl", "w") as f:
            for i in range(3):
                for a, b in [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]:
                    ra, rb = resp[a][i], resp[b][i]
                    rec = judge_record(j, "q", ra, rb)
                    rec.update({"source_index": str(i), "a_policy": a, "b_policy": b, "a_exact": exact_reward(ra, "42"),
                                "b_exact": exact_reward(rb, "42"), "verifier_pref": verifier_pref(exact_reward(ra, "42"), exact_reward(rb, "42"))})
                    f.write(json.dumps(rec) + "\n")
    # diagnostics: 2 problems x 5 variants
    from task5_feedback.score_perturbations import score_diagnostics
    from common.data import write_jsonl
    V = {"clean_correct": "6*7=42\n#### 42", "corrupt_reasoning_correct_final": "6*7=40 so\n#### 42",
         "good_reasoning_wrong_final": "6*7=42\n#### 41", "persuasive_filler_correct": "Trust me, expert. 6*7=42\n#### 42",
         "gold_distractor_wrong_final": "not 42; it is\n#### 40"}
    groups = {str(p): {v: {"problem_id": p, "variant_type": v, "question": "6*7?", "gold_final": "42", "response": t}
                       for v, t in V.items()} for p in (1, 2)}
    pairs, var = score_diagnostics(groups, j)
    write_jsonl(od / "diag_pairs.jsonl", pairs)
    write_jsonl(od / "diag_variants.jsonl", var)
    return cfg


def test_run_all(tmp_path):
    from task5_feedback.compare_feedback import run_all
    out = run_all(_fake_results(tmp_path))
    t = out["gsm"]["policy"]
    assert math.isclose(t.loc["rlvr", "exact_acc"], 2 / 3) and t.loc["sft", "win_vs_sft"] == 0.5
    agr = out["gsm"]["agreement"]
    assert agr.loc["pooled", "n"] == 9
    sens = out["diag"]["sensitivity"]
    assert sens.loc[("verifier", "S_reason (clean > reason_corrupt)"), "tie"] == 1.0     # binary verifier ties
    assert sens.loc[("verifier", "S_outcome (clean > wrong_final)"), "S"] == 1.0
    vt = out["diag"]["variant_table"]
    assert vt.loc["clean", "verifier_reward"] == 1.0 and vt.loc["gold_distractor", "verifier_reward"] == 0.0
    assert "drop" in out and "cost" in out
    assert (tmp_path / "task5_feedback" / "task5_summary.json").exists()
