"""Torch-free helpers for Task 5 (format checks, failure types, pairwise-preference bookkeeping, statistics)."""
from __future__ import annotations

import math
import re
import time

import numpy as np

from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_SEG_RE = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?")
_FINAL_LINE_RE = re.compile(r"^\s*####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*\.?\s*$")
FLIP = {"A": "B", "B": "A", "TIE": "TIE"}


# ----------------------------------------------------------------------------------------------------------- data access
def problem_text(row: dict) -> str:
    """Plain problem statement shown to the judge (no formatting instruction)."""
    for k in ("question", "problem", "prompt_text"):
        if row.get(k):
            return str(row[k])
    msgs = row.get("messages") or row.get("prompt")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    if isinstance(msgs, str):
        return msgs
    raise KeyError(f"no problem text in row with keys {sorted(row)}")


def gold_final(row: dict) -> str:
    for k in ("gold_final", "gold", "final_answer", "answer_final", "gold_answer"):
        if row.get(k) not in (None, ""):
            return str(row[k]).replace(",", "").strip()
    if row.get("answer") not in (None, ""):
        ans = str(row["answer"])
        m = extract_designated_final(ans)
        return m if m is not None else ans.replace(",", "").strip()
    raise KeyError(f"no gold final answer in row with keys {sorted(row)}")


def response_text(row: dict) -> str:
    for k in ("response", "completion", "text", "candidate"):
        if k in row and row[k] is not None:
            return str(row[k])
    raise KeyError(f"no response text in row with keys {sorted(row)}")


# ----------------------------------------------------------------------------------------------------- response checks
def format_check(text: str) -> dict:
    """has_final: a designated `#### <number>` exists (what the verifier reads).
    strict_format: exactly one `####` marker and it is the last non-empty line."""
    t = str(text)
    pred = extract_designated_final(t)
    n_marks = t.count("####")
    lines = [ln for ln in t.strip().splitlines() if ln.strip()]
    last_ok = bool(lines) and bool(_FINAL_LINE_RE.match(lines[-1]))
    return {"has_final": pred is not None, "n_final_markers": n_marks,
            "strict_format": bool(pred is not None and n_marks == 1 and last_ok)}


def gold_in_text(text: str, gold) -> bool:
    return any(numerically_equal(x.replace(",", ""), gold) for x in _NUM_RE.findall(str(text)))


def failure_type(text: str, gold, truncated: bool = False) -> str:
    """correct | truncated_no_final | missing_final | gold_in_reasoning_wrong_final | wrong_final"""
    if exact_reward(text, gold) == 1.0:
        return "correct"
    if extract_designated_final(text) is None:
        return "truncated_no_final" if truncated else "missing_final"
    body = _FINAL_SEG_RE.sub(" ", str(text))
    return "gold_in_reasoning_wrong_final" if gold_in_text(body, gold) else "wrong_final"


# --------------------------------------------------------------------------------------------------- pairwise records
def verifier_pref(r_a: float, r_b: float) -> str:
    return "A" if r_a > r_b else ("B" if r_b > r_a else "TIE")


def pair_outcome(pref: str, better: str | None) -> str:
    """better / tie / wrong relative to the diagnostically better side ('A' or 'B'); 'n/a' if no ground truth."""
    if better is None:
        return "n/a"
    if pref == "TIE":
        return "tie"
    return "better" if pref == better else "wrong"


def judge_record(judge, problem: str, a: str, b: str) -> dict:
    """One call of the FIXED course judge (its own deterministic A/B orientation balancing + cache).
    Also records which physical slot the judge picked, to measure position bias without extra calls."""
    if str(a).strip() == str(b).strip():
        return {"judge_pref": "TIE", "identical": True, "cached": True, "secs": 0.0, "shown_first": None, "picked_slot": "TIE"}
    key = judge._key(problem, a, b)
    cached = key in judge.cache
    t0 = time.perf_counter()
    pref = judge.compare(problem, a, b)
    dt = time.perf_counter() - t0
    swap = int(key[:8], 16) % 2 == 1                      # same rule as PairwiseAIJudge.compare
    slot = "TIE" if pref == "TIE" else ("first" if (pref == "A") != swap else "second")
    return {"judge_pref": pref, "identical": False, "cached": cached, "secs": 0.0 if cached else dt,
            "shown_first": "b" if swap else "a", "picked_slot": slot}


def group_win_rates(k: int, prefs: dict) -> list[float]:
    """Direct-RLAIF group reward: (wins + 0.5 ties)/(K-1); prefs maps (i, j) with i<j to 'A'/'B'/'TIE'."""
    wins, ties = np.zeros(k), np.zeros(k)
    for (i, j), p in prefs.items():
        if p == "A":
            wins[i] += 1
        elif p == "B":
            wins[j] += 1
        else:
            ties[i] += 1
            ties[j] += 1
    return ((wins + 0.5 * ties) / max(k - 1, 1)).tolist()


# ---------------------------------------------------------------------------------------------------------- statistics
Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def cohen_kappa(a, b, labels=("A", "B", "TIE")) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def point_biserial(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
