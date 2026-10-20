import math
import pandas as pd
from task4_safety.evaluate_safety import (calibration_table, cohen_kappa, mcnemar_exact, wilson, audit_agreement)


def _df():
    rows = []
    for p, labs in {"sft": ["SAFE_ANSWER", "OVER_REFUSAL", "UNSAFE_COMPLIANCE", "JUSTIFIED_REFUSAL"],
                    "dpo": ["SAFE_ANSWER", "SAFE_ANSWER", "JUSTIFIED_REFUSAL", "JUSTIFIED_REFUSAL"],
                    "ppo": ["OVER_REFUSAL"] * 2 + ["AMBIGUOUS", "JUSTIFIED_REFUSAL"],
                    "grpo": ["SAFE_ANSWER", "AMBIGUOUS", "UNSAFE_COMPLIANCE", "UNSAFE_COMPLIANCE"]}.items():
        for i, l in enumerate(labs):
            rows.append({"policy": p, "xstest_id": i, "benchmark_class": "SAFE" if i < 2 else "UNSAFE",
                         "label": l, "response_tokens": 10 * (i + 1), "type": "t", "confidence": .9, "rationale_tag": "x"})
    return pd.DataFrame(rows)


def test_calibration_rates():
    c = calibration_table(_df(), 256)
    assert c.loc["sft", "safe_answer_rate"] == 0.5 and c.loc["sft", "safe_over_refusal_rate"] == 0.5
    assert c.loc["sft", "unsafe_compliance_rate"] == 0.5 and c.loc["dpo", "unsafe_justified_refusal_rate"] == 1.0
    assert c.loc["ppo", "safe_over_refusal_rate"] == 1.0 and c.loc["ppo", "ambiguous_rate"] == 0.25
    assert c.loc["grpo", "ambiguous_rate_safe"] == 0.5 and c.loc["sft", "mean_response_tokens"] == 25.0


def test_kappa_and_wilson_and_mcnemar():
    a = ["SAFE_ANSWER", "OVER_REFUSAL", "SAFE_ANSWER", "OVER_REFUSAL"]
    assert cohen_kappa(a, a) == 1.0
    assert math.isclose(cohen_kappa(a, ["OVER_REFUSAL", "SAFE_ANSWER", "OVER_REFUSAL", "SAFE_ANSWER"]), -1.0)
    lo, hi = wilson(50, 100); assert lo < .5 < hi and 0.39 < lo < 0.41
    assert mcnemar_exact(0, 0) == 1.0 and math.isclose(mcnemar_exact(0, 5), 2 / 32)


def test_audit_agreement():
    m = _df().assign(manual_label=lambda x: x["label"])
    m.loc[0, "manual_label"] = "OVER_REFUSAL"
    r, conf = audit_agreement(m)
    assert r["n_labeled"] == 16 and math.isclose(r["agreement"], 15 / 16) and conf.values.sum() == 16

