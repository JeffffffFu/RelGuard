"""M3: the type signature of each KILT relation, and the pool it implies.

    python -m dp.signature --build
    python -m dp.signature

The pool an edge samples from must be a function of its endpoints' types and
of nothing else. A pool that depended on the relation it is meant to hide
would leak through its own support: an adversary who learns which labels were
on offer learns something about the one that was there.

Typing the entities and then collecting which relations hold between types was
the obvious way round, and T-REx will not support it. Its `instance of` reaches
only 15.3% of subjects, and `human` is not among its objects at all -- KILT
dropped facts that trivial when it built the slot-filling task -- so the single
most common class in Wikipedia prose cannot be typed that way. Six fine types
came back as Person and all six were occupations.

So the relations are typed instead. A Wikidata property carries a domain and a
range: `place of birth` goes from a Person to a Location whatever the subject
happens to be, and that is a fact about the property, not about any corpus.
Which also gives the pool directly, with no entity typing on this side at all:

    C(t_s, t_o) = { r : signature(r) = (t_s, t_o) }

An entity in a private document still has to be typed, and that asks a model
about the entity's name alone -- never about the document, and never about the
relation being protected.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import statistics
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from dp.entity_types import COARSE, COARSEN  # noqa: E402

KILT = os.path.join(ROOT, "datasets", "kilt")
PATH = os.path.join(KILT, "relation_signatures.json")

PROMPT = """For each Wikidata property below, give the coarse type of its SUBJECT
and the coarse type of its OBJECT.

Every property must be assigned. Do not skip one, do not leave one out of the
reply, and do not answer with any string outside the class names given.

Some properties take an object that is a word rather than a thing -- an
occupation, a genre, a language, a rank, a position, a taxonomic rank, a
religion. Those take Category. Reserve Product for manufactured things.

Classes:
{classes}

Rules:
- Type what the property's arguments ARE, in general, not what one example
  happens to be. "place of birth" is (Person, Location) because its subject is
  always a person.
- Several properties share a signature. That is expected; do not vary a
  signature to make them differ.
- A property whose object is a word rather than a thing has Category as its
  object type. Its subject is usually still a real thing: `occupation` is
  (Person, Category) and `genre` is (Work, Category).
- Use the given direction. "capital" is (Location, Location) and so is
  "capital of"; "founded by" is (Organization, Person) while "founded" would
  be the reverse.

Return only a JSON object, with no other text:
{{"<property>": ["<subject class>", "<object class>"], ...}}

Properties:
{relations}"""


def relations() -> dict[str, int]:
    path = os.path.join(KILT, "trex_relations.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; extract it from the T-REx parquet first")
    return json.load(open(path, encoding="utf-8"))["relations"]


async def _sign(client, names: list[str], model: str, semaphore) -> dict:
    import re

    classes = "\n".join(f"- {name}: {gloss}" for name, gloss in COARSE.items())
    prompt = PROMPT.format(classes=classes,
                           relations=json.dumps(names, ensure_ascii=False))
    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", reply.choices[0].message.content or "", re.S)
    if not match:
        return {}
    try:
        assigned = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    # a reply naming an OntoNotes class instead of a pooling class is folded
    # rather than discarded: the two vocabularies are one map apart, and
    # dropping the answer loses the relation for no reason
    def fold(cls):
        if cls in COARSE:
            return cls
        return COARSEN.get(cls)

    out = {}
    for name, pair in assigned.items():
        if name not in set(names) or not isinstance(pair, list) or len(pair) != 2:
            continue
        subject, obj = fold(pair[0]), fold(pair[1])
        if subject and obj:
            out[name] = (subject, obj)
    return out


async def build(batch: int, model: str, concurrency: int) -> dict:
    from openai import AsyncOpenAI

    names = list(relations())
    batches = [names[i:i + batch] for i in range(0, len(names), batch)]
    print(f"{len(names)} relations in {len(batches)} batches, model {model}")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_sign(client, b, model, semaphore) for b in batches))

    signatures: dict[str, tuple] = {}
    for result in results:
        signatures.update(result)
    missing = [n for n in names if n not in signatures]
    print(f"signed {len(signatures)}/{len(names)}"
          + (f"; unsigned: {missing}" if missing else ""))

    json.dump({"model": model, "classes": list(COARSE),
               "signatures": {k: list(v) for k, v in signatures.items()}},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return signatures


def load_signatures() -> dict[str, tuple[str, str]]:
    if not os.path.exists(PATH):
        raise SystemExit(f"no {PATH}; run with --build")
    raw = json.load(open(PATH, encoding="utf-8"))["signatures"]
    return {name: tuple(pair) for name, pair in raw.items()}


def pools(signatures: dict[str, tuple[str, str]]) -> dict[tuple, list[str]]:
    """(subject type, object type) -> the relations with that signature."""
    pool = collections.defaultdict(list)
    for name, pair in signatures.items():
        pool[pair].append(name)
    return dict(pool)


def report() -> None:
    signatures, counts = load_signatures(), relations()
    pool = pools(signatures)
    sizes = [len(v) for v in pool.values()]
    # weighted by how often a relation is actually used, which is what an edge
    # drawn from the corpus will meet
    weighted = [len(pool[signatures[r]]) for r, n in counts.items()
                for _ in range(max(n // 1000, 1)) if r in signatures]

    print(f"{len(signatures)} relations over {len(pool)} type pairs "
          f"(of {len(COARSE) ** 2} possible)\n")
    print(f"pool size: min {min(sizes)}, median {statistics.median(sizes):.0f}, "
          f"max {max(sizes)}")
    print(f"usage-weighted mean pool size: {statistics.mean(weighted):.2f}")
    print(f"type pairs with a pool of 1: "
          f"{sum(1 for s in sizes if s == 1)}/{len(pool)}")
    print(f"relations in a pool of 1: "
          f"{sum(len(v) for v in pool.values() if len(v) == 1)}/{len(signatures)}")

    print(f"\n{'type pair':<28}{'|C|':>5}  relations")
    for pair, members in sorted(pool.items(), key=lambda kv: -len(kv[1])):
        members = sorted(members, key=lambda r: -counts.get(r, 0))
        shown = ", ".join(members[:7]) + (" ..." if len(members) > 7 else "")
        print(f"{pair[0] + ' -> ' + pair[1]:<28}{len(members):>5}  {shown[:88]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--batch", type=int, default=35)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.batch, args.model, args.concurrency))
    report()
