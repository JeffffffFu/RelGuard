"""M3: a candidate relation pool per type pair, generated not harvested.

    python -m dp.relation_pool --build
    python -m dp.relation_pool

The pool an edge samples from has to be a function of its endpoints' types and
of nothing else. Reading it off the corpus would break that twice over: the
support would track what the documents happen to say, and an adversary who
learns which labels were on offer learns something about the one that was
there.

So a pool is generated from its type pair and nothing else. No corpus is
passed -- not the documents being sanitised and not a held-out sample of them
either -- which makes the pool a pure function of (subject type, object type)
and the invariant true by construction rather than by argument. The ten
relations for PERSON -> ORG would be the same ten whatever the 500 documents
turned out to say.

The pools are not drawn from KILT's 106 Wikidata properties, which was the
first attempt. Those are a harvest of what Wikidata happens to record, and the
harvest came out lopsided: eight type pairs held exactly one relation, which
is a mechanism that cannot perturb, and eleven of the pairs the extraction
actually produced had no relation at all. Generating per pair asks the
question a candidate set has to answer -- not "which relations exist" but
"which relations could hold between a thing of this type and a thing of that
type" -- and every pair comes back with ten.

The `observed_pairs` it generates for are read from the extraction, so the set
of pairs does depend on the corpus. That is a count, not a relation: which
type pairs occur is a property of the entities, which this project treats as
public, and the pool for a pair is identical whether that pair occurs once or
three hundred times.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import statistics
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from dp.paths import under  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from dp.entity_types import ONTONOTES  # noqa: E402
from dp.typemap import extraction, load_map, normalise  # noqa: E402

GENERATOR = "openai/gpt-5-mini"
"""What writes the pools, and why it is not the gpt-4o-mini everything else uses.

The prompt's hardest requirement is that no member of a set contain another,
because the mechanism substitutes members for one another and swapping a
relation for one that still entails it conceals nothing. gpt-4o-mini does not
meet it. Asked for ten relations on the same 88 type pairs, it returned
`is_located_in` beside `is_situated_in` in five separate pools -- the same
relation twice -- and `was_performed_by` beside `is_performed_by`, which differ
only in tense. On the four pairs compared by hand it produced four containing
pairs for (PERSON, PERSON) and three for (ORG, PERSON): `employs`, `hires` and
`recruits` together.

gpt-5-mini on the identical prompt returns sets that span kinds of tie rather
than wordings of one. For (PERSON, PERSON) it gives kinship, friendship,
employment in both directions, teaching in both directions, litigation and
homicide, where gpt-4o-mini gave ten labels for how two people regard each
other. Over the whole run the distinct-relation count rises from 522 to 734 of a
slightly smaller total, and relations shared by two or more pools fall from 97 to
63: it writes for the pair in front of it instead of reaching for the same
general predicates everywhere.

It also honours "return fewer where the pair admits fewer" -- 35 of 88 pools come
back with seven to nine members, where gpt-4o-mini always padded to ten.

Embedding distance does not show this and must not be used to check it. Pool
-internal pairwise distance is 0.5432 under gpt-4o-mini and 0.5474 under
gpt-5-mini, indistinguishable; what changes is the character of the closest
pairs. gpt-4o-mini's are synonyms (`is_located_in`/`is_situated_in`, 0.1483),
gpt-5-mini's are opposites (`exports_product`/`imports_product`, 0.2341;
`translated_from`/`translated_into`, 0.2338). Substituting an opposite conceals;
substituting a synonym does not. The distance is lower for the better pair,
which is why the containment requirement is enforced by the prompt and reported
against the judge the attack uses, not against a distance threshold.
"""

PATH = under("relation_pools.json")

PROMPT = """You are a semantic relation calibration assistant.

Given an ordered pair of entity types, generate a calibrated set of semantic
relations that could hold FROM a thing of the subject type TO a thing of the
object type.

You are given the two types and nothing else. There is no corpus, no document
and no example: the set must be a function of the ordered type pair alone, so
that the same pair always yields the same set whatever any particular text
happens to say. Do not ask for context and do not assume any.

Three tests, in order. A relation enters the set only if it passes all three.

1. It must be meaningful for THIS ordered pair.

   - Assertable of an arbitrary pair of instances of these two types. For
     (PERSON, GPE) `was born in` qualifies; `directed` does not, because a
     person does not direct a country.
   - Correct in this direction. The subject type acts or is described first.
     For (ORG, PERSON) `employs` qualifies and `works for` does not -- the
     latter is the same relation with the arguments reversed and belongs to
     (PERSON, ORG).
   - Dependent on both types. A relation that would read the same with the
     object replaced by anything at all is not a relation of this pair.

2. No member may stand in a containment relation to another member.

   Two relations are in containment when asserting one of a pair would let a
   reader conclude the other. A set holding both the general relation and the
   specific one inside it is not a usable set: these relations are substituted
   for one another in text, and swapping the specific for the general leaves the
   general true, so the swap conceals nothing. Keep whichever of the two is more
   specific and drop the other.

   Apply the test to every pair of members before answering: if A holds of two
   instances, must B also hold? If yes for either ordering, one of them goes.

3. The set must be semantically spread.

   Choose members so that knowing one holds says as little as possible about
   whether another holds. Relations that differ only in wording, tense, voice or
   politeness are one relation. Relations that describe the same kind of tie
   between the same kinds of participant are close even when their words share
   nothing, and a set of such relations gives the mechanism nothing to work
   with. Reach across kinds of tie: how the two came to be connected, what one
   does to the other, where or when the connection holds, what institution or
   document mediates it.

Return as many relations as pass all three tests, up to {N}. Fewer is correct
where the pair admits fewer: every member of the set is something that may be
written into a text in place of the truth, so a padded set does damage. Do not
stop early either -- if the pair genuinely admits {N} distinct, non-containing
relations, return {N}.

Other requirements:
- Express each relation as a concise lower-case snake_case predicate.
- Avoid vague predicates: related_to, associated_with, connected_to, involves,
  part_of, is_a.
- Prefer relations that would be mutually exclusive if both were asserted of the
  same ordered pair.
- Do not name any specific entity.

Input:
Subject Type: {subject_type}
Object Type: {object_type}

Output:
{{
  "relation_pool": [
    "<relation_1>",
    "<relation_2>",
    "..."
  ]
}}"""


def observed_pairs() -> collections.Counter:
    """The ordered OntoNotes type pairs the extracted triples actually use."""
    mapping = load_map()
    pairs = collections.Counter()
    for record in extraction()["records"]:
        entities = record["entities"]
        for triple in record["triples"]:
            s, o = triple["subject"], triple["object"]
            if s not in entities or o not in entities:
                continue
            a = mapping.get(normalise(entities[s]))
            b = mapping.get(normalise(entities[o]))
            if a and b:
                pairs[(a, b)] += 1
    return pairs


async def _generate(client, pair, n, model, semaphore) -> tuple:
    prompt = PROMPT.format(subject_type=pair[0], object_type=pair[1], N=n)
    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", reply.choices[0].message.content or "", re.S)
    if not match:
        return pair, []
    try:
        pool = json.loads(match.group(0)).get("relation_pool", [])
    except json.JSONDecodeError:
        return pair, []
    # the template's own placeholders come back sometimes; they are not relations
    clean, seen = [], set()
    for relation in pool:
        r = normalise(relation)
        if r and not re.fullmatch(r"relation_\d+|\.\.\.", r) and r not in seen:
            seen.add(r)
            clean.append(r)
    return pair, clean


async def build(n: int, model: str, concurrency: int) -> dict:
    from openai import AsyncOpenAI

    pairs = observed_pairs()
    print(f"{len(pairs)} type pairs, at most {n} relations each, "
          f"no corpus input")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_generate(client, pair, n, model, semaphore) for pair in pairs))

    pools = {f"{a}|{b}": members for (a, b), members in results}
    empty = [k for k, v in pools.items() if not v]
    print(f"generated for {len(pools) - len(empty)}/{len(pools)} pairs"
          + (f"; empty: {empty}" if empty else ""))

    # N is a cap, not a quota: the prompt is told to return fewer where the pair
    # admits fewer, because a padded set is worse than a short one -- every
    # member is something the mechanism may put into the text in place of the
    # truth. So the size distribution is worth printing, and a pool of one is
    # worth naming: it is a mechanism that cannot perturb.
    sizes = collections.Counter(len(v) for v in pools.values())
    print("pool sizes: " + ", ".join(f"{k}:{v}" for k, v in sorted(sizes.items())))
    thin = sorted(k for k, v in pools.items() if 0 < len(v) < 2)
    if thin:
        print(f"pools with a single candidate, which cannot perturb: {thin}")

    json.dump({"model": model, "n": n, "classes": list(ONTONOTES),
               "pools": pools},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return pools


def select(pools: dict, keep: int, model: str | None = None) -> dict:
    """Thin each pool to `keep` candidates chosen for spread, not for rank.

    Greedy max-min over the verbalised embeddings: start from the two
    furthest apart, then repeatedly add whichever remaining candidate is
    furthest from everything already chosen.

        c_{k+1} = argmax_c min_{c' in chosen} d(c, c')

    Measured, this is the wrong objective and the function is kept only to
    reproduce the comparison. Max-min does what it promises -- the ten it keeps
    are mutually far, median pairwise distance 0.4262 against the generator's
    own 0.10 -- and that is exactly the problem. It keeps the ten relations
    furthest out, and a private relation is typically not out there, so the
    quantity the mechanism actually reads -- the spread of d(r, c) across the
    pool, which is all it reads, because the softmax is invariant to shifting
    every distance by a constant -- comes out worse than if the ten had been
    picked at random:

        selection       |C|   spread   E[d(r,r~)] at eps=0 / 1 / 20
        max-min          10   0.1511     0.4415  0.4400  0.4040
        random           10   0.1616     0.4104  0.4085  0.3685
        k-center         10   0.1745     0.4371  0.4351  0.3831
        all              30   0.2417     0.4087  0.4065  0.3407

    The eps=0 column is uniform sampling, so it measures the pool and not the
    mechanism, and max-min is last in it: the subset it chooses sits further
    from the private relations before any epsilon is applied.

    Thinning is therefore not done. The pools are used at their generated size
    of 30, which costs nothing -- a pool is still a function of its type pair
    alone, so Theorem 1 is untouched, and a wider candidate set cannot weaken
    an exp(eps * d) guarantee.
    """
    import numpy as np

    from dp.mechanism import DEFAULT_EMBEDDER, distance, embeddings
    from dp.verbalize import load_sentences, text_of

    # the same encoder the mechanism scores with. Selecting under one metric
    # and sampling under another picks the subset that is spread in the wrong
    # space: the first pass chose with text-embedding-3-small and the
    # mechanism now measures with an NLI encoder, and "most spread" is not the
    # same ten relations in both.
    model = model or DEFAULT_EMBEDDER
    sentences = load_sentences()
    texts = {text_of(r, a, b, sentences)
             for (a, b), members in pools.items() for r in members}
    vectors = embeddings(texts, model)

    thinned = {}
    for (a, b), members in pools.items():
        if len(members) <= keep:
            thinned[(a, b)] = members
            continue
        said = {r: text_of(r, a, b, sentences) for r in members}
        d = {(x, y): distance(said[x], said[y], vectors)
             for x in members for y in members}
        first, second = max(((x, y) for x in members for y in members if x < y),
                            key=lambda p: d[p])
        chosen = [first, second]
        while len(chosen) < keep:
            rest = [r for r in members if r not in chosen]
            chosen.append(max(rest, key=lambda r: min(d[(r, c)] for c in chosen)))
        thinned[(a, b)] = chosen
    return thinned


def pairwise(pools: dict, model: str | None = None) -> dict:
    """Within-pool pairwise distances, which is what distinguishability is."""
    import numpy as np

    from dp.mechanism import DEFAULT_EMBEDDER, distance, embeddings
    from dp.verbalize import load_sentences, text_of

    model = model or DEFAULT_EMBEDDER
    sentences = load_sentences()
    texts = {text_of(r, a, b, sentences)
             for (a, b), members in pools.items() for r in members}
    vectors = embeddings(texts, model)
    out = {}
    for (a, b), members in pools.items():
        said = {r: text_of(r, a, b, sentences) for r in members}
        out[(a, b)] = [distance(said[x], said[y], vectors)
                       for i, x in enumerate(members) for y in members[i + 1:]]
    return out


def load_pools() -> dict[tuple[str, str], list[str]]:
    if not os.path.exists(PATH):
        raise SystemExit(f"no {PATH}; run with --build")
    raw = json.load(open(PATH, encoding="utf-8"))["pools"]
    return {tuple(k.split("|")): v for k, v in raw.items()}


def report() -> None:
    pools, pairs = load_pools(), observed_pairs()
    sizes = [len(v) for v in pools.values()]
    weighted = [len(pools.get(pair, [])) for pair, n in pairs.items()
                for _ in range(n)]
    vocabulary = collections.Counter(r for v in pools.values() for r in v)

    print(f"{len(pools)} pools, {sum(sizes)} relations, "
          f"{len(vocabulary)} distinct\n")
    print(f"pool size: min {min(sizes)}, median {statistics.median(sizes):.0f}, "
          f"max {max(sizes)}")
    print(f"usage-weighted mean |C|: {statistics.mean(weighted):.2f}")
    print(f"relations shared by two or more pools: "
          f"{sum(1 for _, c in vocabulary.items() if c > 1)}/{len(vocabulary)}")

    print(f"\n{'type pair':<28}{'triples':>8}{'|C|':>5}  pool")
    for pair, n in pairs.most_common():
        members = pools.get(pair, [])
        shown = ", ".join(members[:6]) + (" ..." if len(members) > 6 else "")
        print(f"{pair[0] + ' -> ' + pair[1]:<28}{n:>8}{len(members):>5}  {shown[:78]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--select", type=int,
                   help="thin the saved pools to this many by greedy max-min")
    p.add_argument("-n", type=int, default=10, help="relations per type pair")
    p.add_argument("--model", default=GENERATOR)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.n, args.model, args.concurrency))
    if args.select:
        pools = select(load_pools(), args.select)
        json.dump({"model": args.model, "n": args.select,
                   "classes": list(ONTONOTES),
                   "pools": {f"{a}|{b}": v for (a, b), v in pools.items()}},
                  open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"thinned to {args.select} per pool")
    report()
