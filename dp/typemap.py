"""Map the extracted entity types onto OntoNotes, then onto the pooling classes.

    python -m dp.typemap --build
    python -m dp.typemap

The extraction prompt asks for an entity type and leaves the vocabulary open,
so 500 documents came back with 391 distinct strings -- 297 after case and
punctuation are normalised, and the top fifteen of those are two thirds of the
mass. `Person` and `person` are separate strings; so are `tv_show`, `tv_series`
and `show`.

An open vocabulary cannot key a candidate pool: the pool has to be a function
of a fixed, public type inventory, and one that grows with the corpus would
also leak through its own support. So the strings are mapped onto the eleven
OntoNotes classes, and the pool is keyed on those eleven directly.

Not on the coarse seven: dp.relation_pool keys on what load_map returns, so
the pools on disk read `PERSON|WORK_OF_ART` and `ORG|GPE`, 62 pairs of the 121
the eleven admit. COARSEN survives for dp.signature and for
`typed` below, neither of which the mechanism calls. Keeping the eleven is
what makes a pool worth asking for: GPE, LOC and FAC collapse into one
Location under COARSEN, and a set of relations that has to hold for a country,
a mountain range and a bridge alike can only be the vague ones.

The mapping is over type strings alone -- never the entity, never the document,
never the relation. A type that had seen the relation it is meant to help hide
would carry it into the pool.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from dp.paths import under  # noqa: E402

from dp.entity_types import COARSEN, ONTONOTES  # noqa: E402
from dp.signature import load_signatures, pools  # noqa: E402

EXTRACTION = under("extraction_sample50.json")
PATH = under("ontonotes_type_map.json")

FORCE = """Assign each entity-type label below to exactly one OntoNotes class.

Classes:
{classes}

Every label must be assigned. Do not skip a label, do not leave one out of the
reply, and do not answer with any string other than the eleven class names. A
label that fits none of them well still takes the closest: a date or a year is
best served by whichever class names a thing rather than a time, an abstract
word like "concept" or "entity" by PRODUCT, a biological group by PRODUCT.

Classify what the label's INSTANCES are. "award" is EVENT because awards are
occasions; "work" and "media" are WORK_OF_ART; "dynasty" is ORG.

Return only a JSON object with one entry per label and no other text:
{{"<label>": "<class>", ...}}

Labels:
{labels}"""

PROMPT = """Assign each entity-type label below to exactly one OntoNotes class.

Classes:
{classes}

Rules:
- Classify what the label's INSTANCES are. "director" is PERSON because
  directors are people; "film" is WORK_OF_ART because films are works.
- A label naming a kind of place is GPE if it is a country, city or state and
  LOC otherwise; a built structure is FAC.
- Labels that none of the eleven fit -- dates, quantities, species, abstract
  concepts, occupations as such -- take the single best fit and will be folded
  into Other downstream. Prefer PRODUCT over inventing a class.

Return only a JSON object, with no other text:
{{"<label>": "<class>", ...}}

Labels:
{labels}"""


def normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_") or "unknown"


# Words that cannot carry a relation on their own, plus the handful of
# predicates that are grammatical but say nothing. A triple whose relation is
# one of these is not a relation this project perturbs: there is no semantics
# in `is_a` or `part_of` to randomise, the candidate pools do not contain
# anything a reader would call a substitute for them, and replacing one in the
# text rewrites a noun phrase rather than a claim. The extraction prompt
# already asks for specific predicates and mostly gets them -- 82 of 2,160
# triples (3.8%) came back like this, across 21 kinds.
FUNCTION = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "by", "for",
    "to", "with", "from", "as", "into", "onto", "under", "over", "near",
    "within", "is", "was", "are", "were", "be", "been", "being", "has", "have",
    "had", "its", "their", "his", "her", "s", "that", "which", "who", "whose",
}
VAGUE = {
    "is_a", "part_of", "is_part_of", "related_to", "associated_with",
    "connected_to", "includes", "contains", "linked_to", "involves",
    "pertains_to", "concerns", "relates_to",
}


def is_concrete(relation: str) -> bool:
    """Whether a relation names something specific enough to randomise."""
    if relation in VAGUE:
        return False
    return any(w not in FUNCTION
               for w in re.findall(r"[a-z']+", relation.lower()))


def extraction(concrete_only: bool = True) -> dict:
    """The extracted triples, by default only the ones naming a real relation.

    Filtering here rather than at each call site keeps one definition of what
    counts as a relation: the mechanism's facts, the spans dp.spans narrows and
    the sentences dp.inplace substitutes into all come through this function,
    so a triple dropped here is dropped everywhere and no stage can disagree
    with another about what it is protecting.
    """
    if not os.path.exists(EXTRACTION):
        raise SystemExit(f"no {EXTRACTION}; run python -m dp.extract first")
    data = json.load(open(EXTRACTION, encoding="utf-8"))
    if not concrete_only:
        return data
    records = []
    for record in data["records"]:
        records.append({**record,
                        "triples": [tr for tr in record["triples"]
                                    if is_concrete(tr["relation"])]})
    return {**data, "records": records}


async def _assign(client, labels: list[str], model: str, semaphore,
                  template: str = PROMPT) -> dict:
    classes = "\n".join(f"- {n}: {g}" for n, g in ONTONOTES.items())
    prompt = template.format(classes=classes,
                             labels=json.dumps(labels, ensure_ascii=False))
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
    return {k: v for k, v in assigned.items()
            if k in set(labels) and v in ONTONOTES}


async def build(batch: int, model: str, concurrency: int) -> dict[str, str]:
    from openai import AsyncOpenAI

    counts = collections.Counter(
        normalise(t) for r in extraction()["records"]
        for t in r["entities"].values())
    labels = [label for label, _ in counts.most_common()]
    batches = [labels[i:i + batch] for i in range(0, len(labels), batch)]
    print(f"{len(labels)} type labels in {len(batches)} batches, model {model}")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_assign(client, b, model, semaphore) for b in batches))
    mapping: dict[str, str] = {}
    for result in results:
        mapping.update(result)

    missing = [label for label in labels if label not in mapping]
    covered = sum(counts[label] for label in mapping)
    print(f"assigned {len(mapping)}/{len(labels)} labels, "
          f"{100 * covered / sum(counts.values()):.1f}% of entity mass; "
          f"{len(missing)} unassigned -> Other")

    json.dump({"model": model, "map": mapping, "unassigned": missing},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return mapping


async def retry_unmapped(batch: int, model: str, concurrency: int) -> dict[str, str]:
    """Assign the labels the first pass left out, under a prompt that forbids
    skipping.

    The first prompt said to "prefer PRODUCT over inventing a class" and the
    model read that as permission to say nothing: 92 of 297 labels came back
    absent, and absent became Other by default. That put `award`, `work`,
    `media` and `chart` -- which have obvious homes in EVENT and WORK_OF_ART --
    in the same bucket as the labels that genuinely have none, and an untyped
    entity cannot be told from one typed Other.
    """
    from openai import AsyncOpenAI

    existing = json.load(open(PATH, encoding="utf-8"))["map"] if os.path.exists(PATH) else {}
    counts = collections.Counter(
        normalise(t) for r in extraction()["records"]
        for t in r["entities"].values())
    labels = [label for label, _ in counts.most_common() if label not in existing]
    if not labels:
        print("nothing unmapped")
        return existing
    batches = [labels[i:i + batch] for i in range(0, len(labels), batch)]
    print(f"{len(labels)} unmapped labels in {len(batches)} batches")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_assign(client, b, model, semaphore, FORCE) for b in batches))
    added: dict[str, str] = {}
    for result in results:
        added.update(result)

    still = [label for label in labels if label not in added]
    print(f"assigned {len(added)}/{len(labels)}"
          + (f"; still unmapped: {still}" if still else "; none left"))

    merged = {**existing, **added}
    covered = sum(counts[label] for label in merged if label in counts)
    print(f"map now covers {len(merged)} labels, "
          f"{100 * covered / sum(counts.values()):.1f}% of entity mass")
    json.dump({"model": model, "map": merged, "unassigned": still},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return merged


# Since SCHEMA 3 the extraction emits one of the eleven OntoNotes classes
# directly, so the free-string mapping below has nothing to map: `normalise`
# turns `PERSON` into `person` and this table turns it back. A scope extracted
# before that change still has its model-built map on disk and still uses it,
# which is why the file wins where it exists rather than being deleted.
IDENTITY = {normalise(name): name for name in ONTONOTES}


def load_map() -> dict[str, str]:
    """Type string -> OntoNotes class.

    The identity over the eleven classes, overlaid with whatever a --build pass
    wrote for this scope. A scope whose extraction already emits the closed
    inventory needs no --build and no model call.
    """
    on_disk = (json.load(open(PATH, encoding="utf-8"))["map"]
               if os.path.exists(PATH) else {})
    return {**IDENTITY, **on_disk}


def coarse_of(raw_type: str, mapping: dict[str, str]) -> str:
    """A free-form type string -> OntoNotes -> the pooling class."""
    return COARSEN.get(mapping.get(normalise(raw_type), ""), "Other")


def typed_triples(mapping: dict[str, str]) -> list[dict]:
    """Every extracted triple with its endpoint types and its type pair."""
    out = []
    for record in extraction()["records"]:
        entities = record["entities"]
        for triple in record["triples"]:
            s, o = triple["subject"], triple["object"]
            if s not in entities or o not in entities:
                continue
            out.append({
                "document": record["document"].split("\n", 1)[0],
                "subject": s, "relation": triple["relation"], "object": o,
                "subject_type": entities[s], "object_type": entities[o],
                "pair": (coarse_of(entities[s], mapping),
                         coarse_of(entities[o], mapping)),
                "occurrences": triple["occurrences"],
            })
    return out


def report() -> None:
    mapping = load_map()
    triples = typed_triples(mapping)
    pool = pools(load_signatures())

    pairs = collections.Counter(t["pair"] for t in triples)
    print(f"{len(triples)} triples with both endpoints typed, "
          f"{len(pairs)} distinct type pairs\n")

    print(f"{'type pair':<30}{'triples':>9}{'share':>8}{'|C|':>6}")
    print("-" * 53)
    total = sum(pairs.values())
    for pair, n in pairs.most_common():
        size = len(pool.get(pair, []))
        print(f"{pair[0] + ' -> ' + pair[1]:<30}{n:>9}{100 * n / total:>7.1f}%"
              + (f"{size:>6}" if size else f"{'--':>6}"))

    with_pool = sum(n for pair, n in pairs.items() if len(pool.get(pair, [])) > 1)
    print(f"\ntriples whose type pair has a pool of 2 or more: "
          f"{with_pool}/{total} ({100 * with_pool / total:.1f}%)")
    weighted = [len(pool.get(t["pair"], [])) for t in triples]
    print(f"mean |C| over these triples: "
          f"{sum(weighted) / max(len(weighted), 1):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--retry", action="store_true",
                   help="assign only the labels the first pass skipped, under "
                        "a prompt that forbids skipping")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.batch, args.model, args.concurrency))
    if args.retry:
        asyncio.run(retry_unmapped(args.batch, args.model, args.concurrency))
    report()
