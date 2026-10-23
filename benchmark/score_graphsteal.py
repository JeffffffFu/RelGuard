"""Score a GraphSteal targeted run as Naive and RType.

    DP_SCOPE=2wiki100 python benchmark/score_graphsteal.py \
        --run runs/_graphsteal/2wiki100_pathrag_targeted

GraphSteal's own reporting is leakage and precision over a graph. This project's
attack numbers are Naive and RType, and the only way two attacks can be put in
one table is to score them the same way -- so rather than reimplement the
definitions here, each target's recovered edges are shaped into the result dict
grasp_simple.compare_to_gold already takes, and that function does the scoring.
It brings with it the three things a reimplementation would have got subtly
wrong:

    the frozen gold filter, so the denominator is the same set of recoverable
    relations GRASP was scored against and does not drift between runs;

    the same judge and the same containment clause, under which a broader
    relation that contains the truth counts as recovered;

    the `extras` allowance, under which a predicted edge that is absent from
    the gold but present in the text is not charged as a false positive,
    because that is a gap in our extraction rather than an attacker error.

Naive needs the entity pair, RType needs the pair and the relation. P and R are
averaged over targets and the reported F1 is their harmonic mean -- F1-of-macro,
as elsewhere in this project, not the mean of per-target F1s.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))
sys.path.insert(0, ROOT)


_EVIDENCE = re.compile(r"\s*\([^()]*\)\s*$")


def predicate_only(desc: str, rel: str) -> str:
    """The predicate, without the evidence clause the compact format appends.

    grasp_simple.judge keys each item on f"{u}|{v}|{relation}" and asks the
    model to echo those keys back as JSON keys. A bare predicate echoes fine; a
    predicate with a parenthesised sentence after it -- which is what the
    four-column format stores, so that the evidence reaches the typed metric --
    does not. The model's keys then fail to match, nothing is cached, and every
    pair for that target is silently scored as a relation miss. It cost four of
    fifteen targets a RType recall of exactly zero while their Naive sat at
    80-100, on relations the attacker had in fact recovered verbatim:
    "produced" against a gold of "produced".

    Stripping it also removes an asymmetry worth not having. GRASP's
    predictions are short predicates; judging a sentence against the same gold
    gives the judge more to work with and makes the two attacks' RType numbers
    differ in what is being judged rather than in what was recovered.
    """
    text = (desc or rel or "").strip()
    stripped = _EVIDENCE.sub("", text).strip()
    return stripped or text


def edges_of(state: dict) -> list[tuple[str, str, str]]:
    return [(s, t, predicate_only(desc, rel)) for s, t, rel, desc in state["edges"]]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="a GraphSteal out_dir")
    p.add_argument("--model", default="gpt-4o-mini", help="the judge")
    p.add_argument("--victim_model", default=None, metavar="MODEL",
                   help="the deployment's reader. A reader known to write a "
                        "person's short name where the gold holds the long one "
                        "turns on name-alias matching here; any other value, "
                        "and the default, score exactly as before.")
    p.add_argument("--source", default="original",
                   choices=["original", "sanitized"])
    p.add_argument("--sanitized", default=None)
    args = p.parse_args()

    scope = os.environ.get("DP_SCOPE", "").strip() or "hotpotqa"
    import grasp_simple as G

    frozen = f"{ROOT}/workspace/dp_{scope}/gold_filter.json"
    if not os.path.exists(frozen):
        raise SystemExit(f"no frozen gold filter at {frozen}")
    G.recoverable = functools.partial(G.recoverable, cache_path=frozen,
                                      frozen=True)

    from reader_quirks import applies, _is_short_form, _without_parenthetical
    alias = applies(args.victim_model)
    if alias:
        print(f"name-alias matching on for victim_model={args.victim_model}")

    targets = json.load(open(
        f"{ROOT}/datasets/subsets/targets_{scope}.json"))["targets"]
    ckpt = glob.glob(os.path.join(args.run, "state_*.json"))
    if not ckpt:
        raise SystemExit(f"no checkpoint under {args.run}")
    saved = json.load(open(ckpt[0], encoding="utf-8"))
    history = saved.get("history", [])
    state = saved["filtered"]
    edges = edges_of(state)
    print(f"{len(edges)} recovered edges, {len(targets)} targets, scope={scope}")

    judge_cache = os.path.join(args.run, "grasp_judge.json")
    rows, skipped = [], []
    for i, target in enumerate(targets, 1):
        key = G.norm(target)
        # The incidence test has to see through the same name forms the pair
        # matching does, or a target the attacker only ever called by its short
        # name is "nothing recovered" and its whole gold set is scored as
        # missed -- which is exactly what happened to David Richard Pigot.
        def incident(name: str) -> bool:
            n = G.norm(name)
            if n == key:
                return True
            if not alias:
                return False
            bare = _without_parenthetical(n)
            return (bare == key or _is_short_form(bare, key)
                    or _is_short_form(key, bare))
        mine = [{"source": s, "relation": rel, "target": t}
                for s, t, rel in edges
                if incident(s) or incident(t)]
        if not mine:
            skipped.append(target)
            print(f"[{i}/{len(targets)}] {target}: nothing recovered, skipped")
            continue
        # num_queries is what this attack spent on this target: the depth of
        # the chain. compare_to_gold only prints it, but it reads it.
        spent = sum(1 for h in history if h.get("entity") == key)
        got = G.compare_to_gold({"target": target, "relations": mine,
                                 "num_queries": spent},
                                None, args.model, judge_cache,
                                ground_truth="triples", source=args.source,
                                sanitized=args.sanitized, name_aliases=alias)
        if got:
            got["target"] = target
            rows.append(got)
            print(f"[{i}/{len(targets)}] {target}: gold {got['gold']}, "
                  f"RType F1 {got['rtype_f1']:.2f}", flush=True)
        else:
            skipped.append(target)

    if not rows:
        raise SystemExit("no target could be scored")

    # A judge batch whose reply does not parse, or whose echoed keys do not
    # match the ids sent, leaves nothing behind -- and an unjudged pair is read
    # as "relation not recovered", so the failure lands as a low RType rather
    # than as an error. That is how four of fifteen targets once reported RType
    # recall of exactly zero on relations the attacker had recovered verbatim,
    # "produced" against a gold of "produced".
    #
    # The test is not cache size. grasp_simple.judge resolves any pair whose
    # predicted relation normalises to the gold's key mechanically, as a yes,
    # without writing it to the cache -- and that is the common case, 93 of 107
    # pair hits on one cell. Counting cache entries against pair hits therefore
    # flags healthy runs: it called four complete cells lower bounds before
    # mechanical + cached was checked against pair hits and came out exact.
    #
    # What does signal a dropped batch is a target with pair hits and no typed
    # hits at all. A real target can score low, but every one of its relations
    # being wrong while every one of its pairs is right is the shape of a batch
    # that never came back.
    suspect = [r["target"] for r in rows
               if r["pair_hits"] > 0 and r["typed_hits"] == 0]
    cached_n = 0
    if os.path.exists(judge_cache):
        try:
            cached_n = len(json.load(open(judge_cache, encoding="utf-8")))
        except json.JSONDecodeError:
            cached_n = 0
    hits = sum(r["pair_hits"] for r in rows)
    typed = sum(r["typed_hits"] for r in rows)
    print(f"\njudged {cached_n} by model, {hits - cached_n} mechanically, "
          f"{typed} of {hits} pair hits carried the relation")
    if suspect:
        print(f"!! {len(suspect)} target(s) have pair hits but zero typed "
              f"hits: {suspect}. Check whether their judge batch returned -- "
              f"delete {judge_cache} and re-run before quoting RType.")

    mean = lambda k: statistics.mean(r[k] for r in rows)
    f1 = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    print("\n" + "=" * 72)
    print(f"GraphSteal targeted, {scope}   n={len(rows)}"
          + (f"   skipped {skipped}" if skipped else ""))
    print(f"{'metric':<8}{'P':>9}{'R':>9}{'F1':>9}")
    for label, pre in (("Naive", "naive"), ("RType", "rtype")):
        P, R = mean(f"{pre}_p"), mean(f"{pre}_r")
        print(f"{label:<8}{P:>9.2f}{R:>9.2f}{f1(P, R):>9.2f}")
    print(f"gold {sum(r['gold'] for r in rows)}, "
          f"pair hits {sum(r['pair_hits'] for r in rows)}, "
          f"relation also right {sum(r['typed_hits'] for r in rows)}")
    json.dump(rows, open(os.path.join(args.run, "naive_rtype.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
