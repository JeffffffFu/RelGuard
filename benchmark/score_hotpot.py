"""HotpotQA's own two metrics, transcribed from its official scorer.

    python benchmark/score_hotpot.py runs/<run>/predictions_hotpotqa_*.json

    EM   the normalised answer matches the normalised gold exactly
    F1   bag-of-words overlap between them, token level

Both come from hotpot_evaluate_v1.py and nothing here is a re-interpretation:
the normalisation, the tokenisation and the yes/no rule are the published ones,
so a number produced here is comparable with a number from the leaderboard.

Deterministic, and that is the point. Every metric we have reported so far
asks a model to grade an answer, which puts a one-to-two point floor under
every comparison; these two put none, so a difference between two runs is a
difference between two runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import string
from collections import Counter, defaultdict

SPECIAL = ("yes", "no", "noanswer")


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str) -> float:
    """Token-level F1 over bags of words.

    The yes/no rule is not decoration. Without it "Yes, because the director
    was born first" scores partial credit against a gold of "yes"; the official
    scorer gives it zero, and 63 of the 1,000 questions are yes/no.
    """
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if (p in SPECIAL or g in SPECIAL) and p != g:
        return 0.0

    p_tokens, g_tokens = p.split(), g.split()
    same = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if not same:
        return 0.0
    precision, recall = same / len(p_tokens), same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def score(path: str) -> dict:
    records = [r for r in json.load(open(path, encoding="utf-8")) if not r.get("error")]
    rows = defaultdict(list)
    for r in records:
        said = r.get("generated_answer") or ""
        # Every spelling the benchmark accepts, scored at the best of them --
        # which is what MuSiQue's own scorer does. HotpotQA and 2Wiki ship no
        # aliases, so for them this is the single gold and nothing changes.
        golds = [r["ground_truth"]] + list(r.get("ground_truth_aliases") or [])
        scored = {"em": 100 * max(exact_match(said, g) for g in golds),
                  "f1": 100 * max(f1(said, g) for g in golds)}
        rows[r["question_type"]].append(scored)
        rows["overall"].append(scored)

    print(f"\n=== {os.path.basename(path)} ({len(records)} questions) ===")
    print(f"{'question type':<16}{'EM':>8}{'F1':>8}{'n':>7}")
    print("-" * 39)
    summary = {}
    for kind in sorted(rows, key=lambda k: (k == "overall", k)):
        em = statistics.mean(r["em"] for r in rows[kind])
        tf1 = statistics.mean(r["f1"] for r in rows[kind])
        summary[kind] = {"em": round(em, 2), "f1": round(tf1, 2), "n": len(rows[kind])}
        print(f"{kind:<16}{em:>8.2f}{tf1:>8.2f}{len(rows[kind]):>7}")

    empty = sum(1 for r in records if not (r.get("generated_answer") or "").strip())
    if empty:
        print(f"  {empty} empty answers, scored 0")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", nargs="+")
    p.add_argument("--output")
    args = p.parse_args()
    all_summaries = {os.path.basename(path): score(path) for path in args.predictions}
    if args.output:
        json.dump(all_summaries, open(args.output, "w", encoding="utf-8"), indent=2)
        print(f"\nwritten to {args.output}")
