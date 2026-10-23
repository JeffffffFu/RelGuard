"""Turn a relation into a full sentence, so the distance is between meanings.

    python -m dp.verbalize --build
    python -m dp.verbalize

The mechanism scores candidates by the distance between relation embeddings,
and embedding a bare predicate gives it very little to work with. Measured
over the generated pools, the distance from a private relation to the ten
candidates spans 0.10 at the median -- every candidate sits about 0.55 away
and they are all about equally far, so at epsilon=1 the nearest is 4% likelier
than the furthest and the sampling is a coin toss with ten sides.

So each relation is verbalised against its type pair:

    works_for   (PERSON, ORG)  ->  "A person works for an organization."
    founded     (PERSON, ORG)  ->  "A person founded an organization."
    member_of   (PERSON, ORG)  ->  "A person is a member of an organization."

A model writes these rather than a rule, because the grammar of a snake_case
predicate is not recoverable from its shape: `works_for` wants "works for",
`produced_by` wants "is produced by", `member_of` wants "is a member of", and
a heuristic that gets two of those right gets the third wrong.

The same verbalisation is used for the private relation and for every
candidate, so the comparison stays like-for-like. It depends on the relation
and the type pair, both of which the pool already depends on, so it adds
nothing to what the mechanism conditions on.
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

PATH = under("verbalized.json")

READABLE = {
    "PERSON": "person", "NORP": "group", "FAC": "facility",
    "ORG": "organization", "GPE": "country", "LOC": "place",
    "PRODUCT": "product", "EVENT": "event",
    "WORK_OF_ART": "work of art", "LAW": "law", "LANGUAGE": "language",
}

PROMPT = """Turn each relation into one short English sentence stating what the
relation says, using the given subject and object types as the two arguments.

Rules:
- The sentence is a template, not a fact: use the type words themselves as the
  arguments, with an article. Never name a specific entity.
- Fix the grammar the predicate needs. "works_for" becomes "works for";
  "produced_by" becomes "is produced by"; "member_of" becomes "is a member of".
- Keep it to one clause. No extra detail, no explanation.

Examples:
  works_for   (person, organization)   -> A person works for an organization.
  founded     (person, organization)   -> A person founded an organization.
  member_of   (person, organization)   -> A person is a member of an organization.
  produced_by (work of art, organization) -> A work of art is produced by an organization.

Return only a JSON object mapping each input key to its sentence:
{{"<key>": "<sentence>", ...}}

Items:
{items}"""


def key(relation: str, subject_type: str, object_type: str) -> str:
    return f"{relation}|{subject_type}|{object_type}"


def needed() -> set[tuple[str, str, str]]:
    """Every (relation, subject type, object type) the mechanism will score.

    Both the private relations and the pool candidates, each against the type
    pair of the fact it belongs to.
    """
    from dp.mechanism import facts

    out = set()
    for fact in facts():
        a, b = fact["pair"]
        out.add((fact["relation"], a, b))
        for candidate in fact["pool"]:
            out.add((candidate, a, b))
    return out


async def _say(client, items: list[tuple], model: str, semaphore) -> dict:
    listing = "\n".join(
        f"  {r}  ({READABLE.get(a, a.lower())}, {READABLE.get(b, b.lower())})"
        f"   key={key(r, a, b)}" for r, a, b in items)
    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(items=listing)}])
    match = re.search(r"\{.*\}", reply.choices[0].message.content or "", re.S)
    if not match:
        return {}
    try:
        said = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    # The model is asked to echo key=<relation>|<SUBJ>|<OBJ> and often replies
    # with the bare relation instead, reading "key=" as a label rather than an
    # instruction. Both spellings are accepted: a bare relation is resolved
    # against this batch, which is unambiguous because a batch never holds the
    # same relation under two type pairs. Rejecting them cost 204 items that
    # no batch size could recover, because the failure was in the key and not
    # in the batching.
    full = {key(r, a, b) for r, a, b in items}
    bare = {}
    for r, a, b in items:
        bare.setdefault(r, []).append(key(r, a, b))

    out = {}
    for k, v in said.items():
        if not (isinstance(v, str) and v.strip()):
            continue
        if k in full:
            out[k] = v.strip()
        elif len(bare.get(k, [])) == 1:
            out[bare[k][0]] = v.strip()
    return out


def _batch(items: list[tuple], size: int) -> list[list[tuple]]:
    """Batches in which no relation name appears twice.

    `_say` accepts a bare relation where it asked for `relation|SUBJ|OBJ`,
    because the model reads `key=` as a label rather than an instruction and
    replies with the relation about half the time. That fallback resolves the
    bare name against the batch, so it only works while the batch holds the
    relation once -- and a slice of the sorted item list holds the same relation
    under every type pair it occurs in, consecutively, which is the worst
    possible arrangement. Measured on a 907-item probe at size 20, 105 items
    (11.6%) came back ambiguous and were dropped.
    """
    buckets: list[list[tuple]] = []
    seen: list[set] = []
    for item in items:
        relation = item[0]
        for n, bucket in enumerate(buckets):
            if relation not in seen[n] and len(bucket) < size:
                bucket.append(item)
                seen[n].add(relation)
                break
        else:
            buckets.append([item])
            seen.append({relation})
    return buckets


async def build(batch: int, model: str, concurrency: int) -> dict[str, str]:
    from openai import AsyncOpenAI

    existing = (json.load(open(PATH, encoding="utf-8"))["sentences"]
                if os.path.exists(PATH) else {})
    items = [t for t in sorted(needed()) if key(*t) not in existing]
    if not items:
        print("nothing new to verbalise")
        return existing
    batches = _batch(items, batch)
    print(f"{len(items)} new items in {len(batches)} batches, model {model}")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    done = [0]

    async def one(b):
        result = await _say(client, b, model, semaphore)
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == len(batches):
            print(f"  {done[0]}/{len(batches)} batches", flush=True)
        return result

    results = await asyncio.gather(*(one(b) for b in batches))
    sentences = dict(existing)
    for result in results:
        sentences.update(result)

    missing = [key(*t) for t in items if key(*t) not in sentences]
    print(f"verbalised {len(sentences)} total; {len(missing)} missing")

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump({"model": model, "sentences": sentences, "missing": missing},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return sentences


def load_sentences() -> dict[str, str]:
    if not os.path.exists(PATH):
        raise SystemExit(f"no {PATH}; run with --build")
    return json.load(open(PATH, encoding="utf-8"))["sentences"]


def text_of(relation: str, subject_type: str, object_type: str,
            sentences: dict[str, str]) -> str:
    """The verbalisation, falling back to a bare template when one is missing.

    The fallback is deliberately poor rather than absent: a missing sentence
    should degrade one candidate's distance, not crash the sweep.
    """
    said = sentences.get(key(relation, subject_type, object_type))
    if said:
        return said
    return (f"A {READABLE.get(subject_type, subject_type.lower())} "
            f"{relation.replace('_', ' ')} a "
            f"{READABLE.get(object_type, object_type.lower())}.")


def report() -> None:
    sentences = load_sentences()
    print(f"{len(sentences)} verbalisations\n")
    shown = collections.Counter()
    for k, v in sentences.items():
        relation, a, b = k.split("|")
        if shown[(a, b)] < 2 and shown["total"] < 14:
            print(f"  {relation:<28}({a}, {b})")
            print(f"      {v}")
            shown[(a, b)] += 1
            shown["total"] += 1
    lengths = [len(v.split()) for v in sentences.values()]
    print(f"\nwords per sentence: median {statistics.median(lengths):.0f}, "
          f"max {max(lengths)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.batch, args.model, args.concurrency))
    report()
