"""The typed relations a GraphRAG index should carry, per corpus.

    python -m dp.graph_inject --original
    python -m dp.graph_inject --run eps1_seed0_inplace
    DP_SCOPE=2wiki python -m dp.graph_inject --run identity_inplace

LightRAG's graph has no relation type, so the predicate the mechanism perturbs
cannot survive its extraction: measured over ten passages, 0 of 31 edges whose
entity pair matched our extraction carried anything comparable to the extracted
predicate, because what an edge carries is an 18-word explanation of why two
entities are related. See lightrag/operate.py's injection note for why asking
the LLM for the predicate instead -- 41% exact agreement -- is not enough.

This writes the triples the index should be handed, so the predicate reaches the
graph verbatim rather than being re-derived from the text by a second extractor.
Which predicate that is depends on the corpus, and the distinction is the whole
point:

    --original          the extracted predicates, for the undefended corpus
    --run <sanitised>   what dp.inplace actually wrote into the text. A swapped
                        occurrence carries the released relation; an occurrence
                        the substitution skipped still reads as the author wrote
                        it, so it carries the true one. Those skips are the
                        protection gap and they have to be represented
                        faithfully -- injecting the released relation for a
                        passage the mechanism never managed to change would
                        claim a protection the text does not have.

The output is keyed by passage title because that is what a chunk can be matched
on: benchmarks.pack() concatenates passages up to a token budget, so a chunk
holds whole passages and each one begins with its title.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dp.paths import under  # noqa: E402

SANITIZED = under("sanitized")


def from_extraction() -> dict[str, list[dict]]:
    """Every extracted triple, keyed by the passage it was found in."""
    from dp.typemap import extraction

    out: dict[str, list[dict]] = collections.defaultdict(list)
    for record in extraction()["records"]:
        title = record["document"].split("\n", 1)[0].strip()
        for triple in record["triples"]:
            out[title].append({"subject": triple["subject"],
                               "relation": triple["relation"],
                               "object": triple["object"]})
    return dict(out)


def from_run(name: str) -> dict[str, list[dict]]:
    """What a dp.inplace run left in the text, keyed by passage.

    `detail` records one entry per substitution actually applied, with the
    relation that went in. Occurrences the run skipped are not in it, and they
    keep the author's wording, so the true relation is injected for those -- the
    text still says it, and a graph that did not would understate what an
    attacker can read.
    """
    path = os.path.join(SANITIZED, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run python -m dp.inplace first")
    run = json.load(open(path, encoding="utf-8"))

    swapped: dict[tuple, str] = {}
    for swap in run["detail"]:
        swapped[(swap["title"], swap["subject"], swap["object"],
                 swap["relation"])] = swap["released"]

    # A triple whose relation was withheld has no predicate to inject: the
    # sentences that stated it are gone, so nothing in the text asserts it and a
    # typed edge would claim a relation the corpus no longer carries. Those
    # triples are dropped from the injection rather than injected with the true
    # relation, which is what `swapped.get(key, triple["relation"])` would
    # otherwise do for them.
    gone = {(t, int(i)) for t, i in
            (s.rsplit("|", 1) for s in run.get("deleted_sentences") or [])}

    # which triples lost every one of their occurrences to a deleted sentence
    from dp.spans import occurrences

    alive: dict[tuple, int] = collections.Counter()
    total: dict[tuple, int] = collections.Counter()
    for it in occurrences():
        key = (it["title"], it["subject"], it["object"], it["relation"])
        total[key] += 1
        if (it["title"], it["sentence_id"]) not in gone:
            alive[key] += 1

    out: dict[str, list[dict]] = collections.defaultdict(list)
    seen, stats = set(), collections.Counter()
    for title, triples in from_extraction().items():
        for triple in triples:
            key = (title, triple["subject"], triple["object"],
                   triple["relation"])
            if total.get(key) and not alive.get(key):
                stats["withheld, every occurrence deleted"] += 1
                continue
            released = swapped.get(key, triple["relation"])
            if released is None:
                stats["withheld"] += 1
                continue
            stats["released" if key in swapped else "untouched"] += 1
            dedup = (title, triple["subject"], released, triple["object"])
            if dedup in seen:
                continue
            seen.add(dedup)
            out[title].append({"subject": triple["subject"],
                               "relation": released,
                               "object": triple["object"]})
    print(f"{name}: {stats['released']} substituted, {stats['untouched']} "
          f"left as the author wrote them "
          f"({100 * stats['untouched'] / max(sum(stats.values()), 1):.1f}% "
          f"unprotected)")
    return dict(out)


def write(by_title: dict[str, list[dict]], name: str) -> str:
    path = under("inject", f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"corpus": name, "passages": len(by_title),
               "triples": sum(len(v) for v in by_title.values()),
               "by_title": by_title},
              open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"{sum(len(v) for v in by_title.values())} triples over "
          f"{len(by_title)} passages -> {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--original", action="store_true",
                       help="the extracted predicates, undefended corpus")
    group.add_argument("--run", help="a dp.inplace run, e.g. eps1_seed0_inplace")
    args = p.parse_args()
    if args.original:
        write(from_extraction(), "original")
    else:
        write(from_run(args.run), args.run)
