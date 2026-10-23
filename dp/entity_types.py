"""M2: a coarse entity type for anything the mechanism has to type.

    python -m dp.entity_types --build      # fine -> coarse, once, from KILT
    python -m dp.entity_types              # report what was built

The candidate pool (M3) is keyed on the types of an edge's two endpoints, so
the type inventory decides how much room the mechanism has. Too fine and every
type pair holds one relation, which leaves nothing to sample; the old medical
schema learned this the hard way, where ten clinical types gave candidate sets
of at most eleven and a weighted mean of 5.89.

T-REx supplies types through `instance of`, and in raw form they are unusable
for this:

    14,746 distinct strings, of which most are aliases of each other --
    album / audio album / record album / music album, or railway station /
    rail station / railway stop / rail stop / train station / railroad
    station -- because KILT's slot filling lists every valid surface form as
    a separate answer.

    the grain is wild. `commune of France` and `Divisional Secretariat of
    Sri Lanka` sit beside `natural number` and `CD block`.

    and it reaches only 213,401 of 1,390,860 subjects, 15.3%.

So the fine strings are mapped once onto COARSE, and an entity takes the
coarse type its fine types agree on. The map is built over the most common
fine types rather than all of them: the top 1,000 cover 89.9% of the mass and
the tail is where the single-use oddities are.

The typing of a private entity must come from public knowledge about that
entity and never from the relation being protected -- a type derived from the
relation would leak through the pool's support, which is the one thing the
pool's construction exists to prevent. That is why this module is built from
KILT and asked only ever about an entity, not about a document.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

KILT = os.path.join(ROOT, "datasets", "kilt")
MAP_PATH = os.path.join(KILT, "coarse_type_map.json")

ONTONOTES: dict[str, str] = {
    "PERSON": "people, including fictional",
    "NORP": "nationalities or religious or political groups",
    "FAC": "buildings, airports, highways, bridges and other facilities",
    "ORG": "companies, agencies, institutions and other organizations",
    "GPE": "countries, cities, states",
    "LOC": ("non-GPE locations: continents, regions, mountain ranges, "
            "bodies of water"),
    "PRODUCT": "objects, vehicles, foods and the like; not services",
    "EVENT": "named hurricanes, battles, wars, sports events",
    "WORK_OF_ART": "titles of books, songs and other works",
    "LAW": "named documents made into laws",
    "LANGUAGE": "any named language",
}
"""The eleven entity classes of OntoNotes 5.0 (LDC2013T19), with LOC widened.

Upstream's LOC gloss reads "non-GPE locations: mountain ranges, bodies of
water", and since SCHEMA 3 this text is the extraction prompt's definition of
the class rather than a note to a human annotator. Read literally it excludes a
continent and a region, and the extractor obeyed it: on ten passages
`South America`, `tropical America`, `southeastern Asia` and `Europe` were left
out as belonging to no class, taking four relations with them. Continents and
regions are named here so that they land in LOC, which is where a reader of the
original gloss would have put them anyway.

Its other seven -- DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL --
are quantities and timestamps rather than entities and are left out.

Taken as given rather than designed, because it is spaCy's English NER
inventory: an entity can be typed locally, for nothing, and identically every
time, which matters because a type that moves between runs moves the candidate
pool with it. Every other typed thing in this project has come from a model
behind an API that does not repeat itself -- three identical calls through
OpenRouter came back three different ways.

Where the type actually comes from, and the hole in that
------------------------------------------------------

Not from spaCy. Since SCHEMA 3 the extraction prompt carries this table and the
extractor returns one of the eleven names directly, in the same call that
returns the relations. The reasons were practical: the open vocabulary it
replaced cost a second model pass to fold 428 free strings onto eleven classes,
left 141 of them unassigned, and let in 803 entities the pool has no use for --
`date`, `year`, `title`, `position`, `profession`, biological taxa. Closing the
vocabulary at the point of extraction removed the pass, removed those, and cut
endpoint mismatch by more than half on a 12-passage probe (34 to 15). Because
the extraction is cached by document hash, the type is now reproducible too,
which is what the spaCy argument above was after.

What it gives up is an independence the privacy argument leans on. A candidate
pool is a function of (subject type, object type) and of nothing else, and that
is what makes Theorem 1's support independent of the private relation. A type
assigned in the same call that extracted the relation has seen that relation, so
the pool depends on it -- weakly, through the type -- and the invariant holds by
argument rather than by construction.

The exposure is probably small: `Seth Rogen` is PERSON whatever the sentence
says about him, and the eleven classes are coarse enough that a relation would
have to change a type's class to leak through it. But "probably small" is not
measured, and the honest options are to type with spaCy -- local, free,
deterministic, and blind to the relation, with the eleven classes as its native
output -- or to measure how often a type changes when the relation around it
does, and state the bound. Neither is done yet.
"""

COARSEN: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "WORK_OF_ART": "Work",
    "LAW": "Work",
    "EVENT": "Event",
    "PRODUCT": "Product",
    "NORP": "Category",
    "LANGUAGE": "Category",
}
"""OntoNotes' eleven to the seven the candidate pool is keyed on.

Two inventories because they answer different questions. Typing wants to be as
specific as the tagger can be; pooling wants to be as coarse as the semantics
allow, because a pool is what the mechanism samples from and eleven classes
fragment 106 relations across up to 121 cells.

Category is the class that makes the rest work. A third of KILT's relations
take an object that is a word rather than a thing -- `occupation` takes
"actor", `genre` takes "documentary film", `native language` takes "French",
`taxon rank` takes "species", `position held` takes "mayor" -- and OntoNotes
has no class for those because they are not entities. Without somewhere to put
them the signatures could not be assigned at all: 32 of 106 came back blank.
Sending them to Product instead put `country of citizenship` in the same pool
as `occupation`, so the mechanism would offer "this person's occupation is the
United States" as a substitution -- not a false statement but an ungrammatical
one, and an ungrammatical one is what the polishing step repairs back toward
the truth.

Naming the class for what those objects are keeps a pool coherent: the things
that can stand in for an occupation are a religion, a language, a rank, a
position. NORP and LANGUAGE join it from the entity side for the same reason --
a nationality or a language is a category of thing, not a thing.

The other merges: GPE, LOC and FAC because a city, a mountain range and a
railway station are all somewhere; LAW joins Work because a named statute is a
written document.

Regrouping is a change to this map alone, which is the point of having it.
"""

COARSE: dict[str, str] = {
    "Person": "an individual human, real or fictional",
    "Organization": "a company, party, team, agency, label, institution or "
                    "other body of people",
    "Location": "somewhere: a country, city, region, body of water, landform, "
                "building, station or other place",
    "Work": "a created work: film, series, album, song, book, journal, game, "
            "painting, statute",
    "Event": "something that happened or recurs: a war, battle, competition, "
             "election, ceremony",
    "Product": "a manufactured thing: a vehicle, device, vessel, aircraft, "
               "material, foodstuff",
    "Category": "a word that classifies rather than names: an occupation, "
                "genre, language, religion, rank, position, taxonomic rank",
}
"""The pooling classes and what each one means.

Glossed in prose rather than by the OntoNotes types they absorb, which was a
mistake worth recording: the gloss is pasted into the prompt that assigns a
relation its signature, and a gloss reading "Location: FAC, GPE, LOC" taught
the model to answer in OntoNotes names. Replies came back as ["ORG", "GPE"],
the filter dropped everything outside the seven, and 65 of 106 relations went
unsigned for a reason that had nothing to do with the relations.

Which OntoNotes type lands where is COARSEN's business, above.
"""


PROMPT = """Assign each Wikidata type label below to exactly one coarse class.

Classes:
{classes}

Rules:
- Choose the class the label's INSTANCES belong to, not the class of the word.
  "album" is a Work because albums are works; "occupation" is Other because an
  occupation is not a person.
- Aliases of the same concept must get the same class.
- Use Other when none of the seven fits, not when you are unsure between two.

Return only a JSON object mapping each label to its class, with no other text:
{{"<label>": "<class>", ...}}

Labels:
{labels}"""


def fine_types() -> dict[str, int]:
    path = os.path.join(KILT, "trex_fine_types.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; extract it from the T-REx parquet first")
    return json.load(open(path, encoding="utf-8"))["fine_types"]


async def _classify(client, labels: list[str], model: str, semaphore) -> dict:
    classes = "\n".join(f"- {name}: {gloss}" for name, gloss in COARSE.items())
    prompt = PROMPT.format(classes=classes, labels=json.dumps(labels, ensure_ascii=False))
    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
    import re

    text = reply.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        assigned = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    # a class outside the inventory is a failed assignment, not a new class
    return {k: v for k, v in assigned.items()
            if k in set(labels) and v in COARSE}


async def build(top: int, batch: int, model: str, concurrency: int) -> dict[str, str]:
    from openai import AsyncOpenAI

    counts = fine_types()
    labels = [label for label, _ in
              sorted(counts.items(), key=lambda kv: -kv[1])[:top]]
    batches = [labels[i:i + batch] for i in range(0, len(labels), batch)]
    print(f"{len(labels)} fine types in {len(batches)} batches, model {model}")

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_classify(client, b, model, semaphore) for b in batches))

    mapping: dict[str, str] = {}
    for result in results:
        mapping.update(result)
    unassigned = [label for label in labels if label not in mapping]
    print(f"assigned {len(mapping)}/{len(labels)}; "
          f"{len(unassigned)} unassigned, defaulting to Other")

    os.makedirs(KILT, exist_ok=True)
    json.dump({"model": model, "classes": list(COARSE),
               "map": mapping, "unassigned": unassigned},
              open(MAP_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return mapping


def load_map() -> dict[str, str]:
    if not os.path.exists(MAP_PATH):
        raise SystemExit(f"no {MAP_PATH}; run with --build")
    return json.load(open(MAP_PATH, encoding="utf-8"))["map"]


def coarse_of(fine: list[str], mapping: dict[str, str]) -> str:
    """The coarse type an entity's fine types agree on.

    An entity carries a median of three fine types because they are aliases,
    so they normally agree. When they do not, the most common wins and a tie
    goes to whichever the mapping saw first -- an entity typed both Facility
    and Location is not worth a special case.
    """
    votes = collections.Counter(mapping[f] for f in fine if f in mapping)
    return votes.most_common(1)[0][0] if votes else "Other"


def report() -> None:
    mapping = load_map()
    counts = fine_types()
    entity_fine = json.load(
        open(os.path.join(KILT, "trex_entity_fine_types.json"), encoding="utf-8"))

    by_mass = collections.Counter()
    for fine, n in counts.items():
        by_mass[mapping.get(fine, "Other")] += n
    total_mass = sum(by_mass.values())

    entities = collections.Counter(
        coarse_of(fine, mapping) for fine in entity_fine.values())
    total_entities = sum(entities.values())

    print(f"{len(mapping)} fine types mapped, "
          f"{len(entity_fine):,} entities typed\n")
    print(f"{'coarse type':<16}{'fine-type mass':>16}{'entities':>14}")
    print("-" * 46)
    for name in COARSE:
        print(f"{name:<16}{100 * by_mass[name] / total_mass:>15.1f}%"
              f"{100 * entities[name] / total_entities:>13.1f}%")

    print("\nexamples per class:")
    for name in COARSE:
        members = [f for f, c in mapping.items() if c == name]
        members.sort(key=lambda f: -counts.get(f, 0))
        print(f"   {name:<14}{', '.join(members[:6])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--top", type=int, default=1000,
                   help="most common fine types to map; 1000 covers 89.9%%")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.top, args.batch, args.model, args.concurrency))
    report()
