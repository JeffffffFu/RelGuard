"""M1+M2: relation facts out of a document, with where in the text they were said.

    python -m dp.extract --benchmark hotpotqa:dev --limit 10
    python -m dp.extract --benchmark hotpotqa:dev            # the whole corpus

One LLM call per document returns entities with types and triples with
occurrences. Both halves are needed and neither is the interesting part:

    entities + types   feed the type-conditioned candidate pool (M3), which
                       must be a function of the endpoint types and nothing
                       else -- a pool that depended on the relation it is
                       meant to hide would leak through its own support.
    occurrences        are what makes write-back possible. A sanitised
                       relation has to replace the words that said the
                       original, so the mechanism needs the span, not just the
                       fact.

Nothing here depends on epsilon, so it runs once and is cached by a hash of
the document. Every epsilon and every seed reads the same extraction, which
is also what makes two of them comparable: a difference between two sweeps is
then a difference in the mechanism and not in what was extracted.

The occurrence contract is the part that bites. `relation_mention` must be a
literal span of the document -- the prompt says so twice -- because write-back
locates it by string search. A mention the model paraphrased cannot be found,
and a fact whose mentions cannot be found cannot be sanitised; those are
counted and reported rather than quietly dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from dp.paths import under  # noqa: E402

REASONING = {"extra_body": {"reasoning": {"effort": "minimal"}}}
"""Turn the reasoning tokens off, which is what makes a reasoning model usable here.

gpt-5-mini generates reasoning tokens by default, and they are billed as output
at eight times the input rate. Over 908 passages with this prompt -- eleven type
definitions, four blocks of rules and a line-numbered document -- it ran at a
quarter of gpt-4o-mini's throughput. With effort=minimal it is *faster* than
gpt-4o-mini on the same twelve passages, 25.5s against 39.1s, and still returns
31% more triples (228 against 174). The task is to find entities, find relations
and copy a substring; none of that needs several steps of thought, and the probe
that chose this model was measuring instruction-following, not reasoning.

gpt-5-nano was tried at the same setting and is not usable: 45 triples where
gpt-5-mini found 228, and localisation fell from 99.1% to 86.5%.
"""

EXTRACTOR = "openai/gpt-5-mini"
"""What reads the documents, and why it is not the gpt-4o-mini this started on.

The same reason dp.relation_pool moved: the hard requirement here is a predicate
narrow enough to substitute, and gpt-4o-mini does not meet it. Its recorded
mentions ran to a median of eight words and 86.1% contained the object entity,
which is why dp/spans.py existed -- a second model pass whose whole job was to
narrow what the first one had returned too wide. Asked directly for the shortest
substring that states the relation, gpt-5-mini returns a median of two words,
with every predicate a literal substring of the line it names (213 of 213 on a
twelve-passage probe) and no bare prepositions.
"""

CACHE = under("extraction")

PROMPT = """You are a knowledge graph extraction assistant.

Identify the entities and the semantic relations explicitly expressed between
them, and for each relation mark exactly where in the document it is stated.

1. Entity Extraction
- Identify the entities belonging to one of these eleven types, and give each
  that type verbatim:

{classes}

- Assign a consistent canonical name to each entity across all mentions, and
  resolve pronouns, aliases and abbreviations to it wherever the reference is
  clear. Do not create two entities for one thing.
- A mention that is not one of the eleven types is NOT an entity here. Leave it
  out entirely rather than forcing it into the closest type. In particular leave
  out: dates and years ("2009", "7 March 2007"); titles, positions, offices and
  professions ("President of the United States", "Duke of Albany",
  "veterinarian"); abstract concepts and bare nouns ("policy", "wildlife",
  "art"); quantities, percentages and sums of money; biological taxa
  ("Araceae", "Colocasia esculenta").
- `type` must be one of the eleven strings exactly as spelled above.

2. Relation Extraction
- Extract only relations the document states directly, as
  (subject, relation, object), with the direction preserved.
- Normalise the relation to a concise lower-case snake_case predicate:
  works_for, founded, located_in, directed_by.
- Prefer the specific relation the text supports over a general one.
- Do not infer from external knowledge, commonsense, negation, hypotheticals or
  speculation.
- Output each distinct triple once. If one subject-object pair carries several
  distinct relations, keep them all.

3. Marking the predicate -- the part that has to be exact

The document is printed below one numbered line at a time. For every occurrence
of every triple, report the line it is on and the words on that line that state
it:

- "line": the line number as printed in the margin.
- "predicate": the SHORTEST substring of that line that states the relation and
  nothing else, copied character for character.
    * usually a verb, or a verb with its particle and preposition:
      "was directed by", "grew up in", "premiered on", "founded"
    * it must contain NO part of the subject name and NO part of the object name
    * it must contain no date, number, place, title or any modifier that would
      still be true if the relation were different
    * two or three words is the usual answer; more than five is almost always
      wrong -- look again for the verb inside what you were about to write
    * NEVER a bare preposition, article or copula. "as", "of", "in", "by", "is"
      alone are all wrong answers. If the relation is carried by juxtaposition
      and no verb states it, give "" and the occurrence is dropped.

Do not report a character offset. The position is found by searching the line
for the substring, which is why it has to be copied exactly.

4. Output Format
Return only a valid JSON object:

{{"entities": [
    {{"name": "<entity_name>", "type": "<ENTITY_TYPE>"}}],
  "triples": [{{
      "subject": "<subject_entity>",
      "relation": "<canonical_relation>",
      "object": "<object_entity>",
      "occurrences": [{{"line": <line_number>, "predicate": "<exact substring>"}}]}}]}}

Every subject and object in "triples" must exactly match the "name" of an
entity listed in "entities". A relation whose subject or object is not one of
the eleven types is therefore not extracted at all. Do not include explanations
or any text outside the JSON object.

Document:
{ruled}"""


def ruled(document: str) -> str:
    """The document with a line number in the margin of every line.

    The model is asked for a line and a substring, not for a character offset.
    Both halves of that were measured on twelve passages. Asked for a sentence
    index and an offset within the sentence, half the predicates (118 of 233)
    were not substrings of the sentence it named -- it has to split sentences
    the same way split_document does to count them, and on `A&R man`, `Jr.` and
    `U.S.` it does not. Asked for a line and a column, every predicate (213 of
    213) was a substring of the line it named, and the column was right 35% of
    the time: it reported `featured` at column 291 three times over, self
    -consistently and wrongly. A model works in tokens; counting characters is
    not something it can do, and nothing is lost by not asking -- a substring
    that is exact can be located by searching for it.
    """
    return "\n".join(f"{i + 1}| {line}"
                      for i, line in enumerate(document.split("\n")))


SCHEMA = 5
"""Bumped whenever the shape `validate` returns changes, or whenever the prompt
changes in a way that makes an old answer the wrong answer. It is part of the
cache key, so the change invalidates the cache instead of being read back as
though it were current -- which is how `entities` went from a list of names to a
name->type map and broke a reader that had no way to tell.

2 -> 3 closes the type vocabulary. The prompt used to leave `entity_type` open
and dp/typemap.py mapped the strings onto OntoNotes afterwards, which cost a
model pass and lost what the eleven classes cannot hold: of 6,986 entities over
908 HotpotQA passages, 803 carried one of 141 labels the mapping left
unassigned, and the biggest of them were things this project does not want --
`concept` (97 entities: `policy`, `wildlife`), `date` (41) and `year` (28),
`title` (35: `President of the United States`), `position` (18), `profession`
(9), and biological taxa (61 across genus/family/species/breed). Those are not
entities a relation between people and organisations needs, and OntoNotes
excludes dates and quantities by design, so there was never a class for them.

Asking for the eleven directly, and asking for mentions outside them to be left
out rather than forced into the nearest fit, makes the type a closed public
inventory at the point of extraction -- which is what the candidate pool needed
all along."""


# The eleven classes, rendered for the prompt. Built from dp.entity_types so the
# inventory has one definition: the prompt, the pooling table and the type map
# cannot drift apart.
def _class_block() -> str:
    from dp.entity_types import ONTONOTES

    return "\n".join(f"    {name}: {gloss}" for name, gloss in ONTONOTES.items())


CLASS_BLOCK = _class_block()


def fingerprint(document: str, model: str) -> str:
    return hashlib.sha256(
        f"v{SCHEMA}|{model}|{document}".encode("utf-8")).hexdigest()[:24]


def _json_object(reply: str) -> dict:
    """The outermost {...}, fences and preamble discarded.

    gpt-4o-mini honours "only a valid JSON object" most of the time and wraps
    it in ```json the rest, which is the one failure mode worth absorbing here
    rather than retrying for.
    """
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {(reply or '')[:200]}")
    return json.loads(match.group(0))


# An entity the document does not contain is the one hallucination this pipeline
# cannot absorb. The relation checks already require the mention to be a literal
# span, but nothing required the endpoints to be in the text, and the extractor
# fills gaps from what it knows: `Ireland` appears as an entity in three passages
# that never use the word, inferred from `Viscount Barnewall` and from an Irish
# surname, and so do `Brazil`, `Achaemenid Empire`, `First Congo War`. The prompt
# forbids inferring from external knowledge, but that clause is about relations.
#
# The test is word-level containment rather than a literal substring, because a
# literal test rejects the canonical names the prompt asks for. Measured over
# 6,986 entities, a literal test rejects 73 and a word-level one 32: the
# difference is 41 legitimate normalisations -- `Christopher 'Chris' Robin
# Duesterdiek` where the text writes `Chris Duesterdiek`, `2018 FIFA World Cup`
# where it writes `2018 World Cup`, `James II of England` for `James II` -- every
# one of which is the same entity under a fuller name. The 32 it does reject are
# all places, empires and wars the text never names.
NAME_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "at", "s"}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in NAME_STOP]


def in_document(name: str, document_words: set) -> bool:
    """Whether every content word of a name occurs somewhere in the document."""
    words = _content_words(name)
    return bool(words) and all(w in document_words for w in words)


# Words that cannot state a relation on their own. A predicate made only of
# these is rejected; the point is to require a verb or participle, not to
# enumerate English. Kept in step with dp.spans.FUNCTION, which applies the same
# bar to the spans that file produces.
FUNCTION = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "by", "for",
    "to", "with", "from", "as", "into", "onto", "under", "over", "near",
    "within", "is", "was", "are", "were", "be", "been", "being", "has", "have",
    "had", "its", "their", "his", "her", "s", "that", "which", "who", "whose",
    "also", "both", "then", "there", "here",
}


# Copulas and auxiliaries. They are verbs and they state nothing on their own:
# `is a` is tagged VBZ+DT and passes any "contains a verb" test, and it is what
# `is a Canadian` becomes once the object is cut out of it. A predicate needs a
# verb that carries meaning.
LIGHT_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "having", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "get", "gets", "got", "gotten", "become", "becomes", "became",
}


def _tagged(text: str):
    """NLTK's POS tags for a predicate, cached per string.

    nltk and its perceptron tagger are already present, so this is local, free
    and the same every time -- which a predicate filter has to be, because it
    decides what the mechanism is allowed to perturb.
    """
    if text not in _TAGS:
        import nltk

        _TAGS[text] = nltk.pos_tag(nltk.word_tokenize(text))
    return _TAGS[text]


_TAGS: dict = {}


def states_a_relation(predicate: str) -> bool:
    """Whether a predicate is a predicate: a content verb and nothing broken.

    Deliberately strict. A span that is not really a predicate does damage out of
    all proportion to its size, because dp.inplace writes the released relation
    over exactly these characters: `is a Canadian` cut back to `is a` and then
    substituted gives `Faith Julia Goldy is married to right-wing writer`, which
    has lost the nationality the question was about. Losing the occurrence costs
    one relation's protection and is visible in the protection gap; admitting a
    bad span costs an entity and is visible nowhere.

    Four tests, each of which rejected something real in the extraction:

        a content word, so the span says something. `is a` -- what `is a
        Canadian` becomes once the object is cut out -- is nothing but function
        words.

        balanced quotes and brackets. `told "` and `Taken from their debut
        album, "` are phrases cut in the middle; the unmatched quote is the
        evidence.

        no leading or trailing hyphen: `-based`, from `New York-based`, is a
        suffix rather than a predicate.

        at most six words, which the prompt already asks for.

    A fifth test was tried and removed: that the span contain a verb which is not
    a copula or auxiliary, by NLTK's tagger. The premise is wrong. English states
    a great many relations with a nominal predicate, and over the occurrences that
    had already passed the endpoint test this rejected a further 40.8% of them --
    `wife of`, `younger sister of`, `is the home venue of`, `became part of`,
    `was a daughter of`. It also mistagged bare verbs, because the tagger is
    trained on sentences and these are isolated phrases: `stars` and `uses` came
    back as plural nouns. What it was meant to catch -- `is a`, `-based`,
    `told "` -- the other three tests already catch.
    """
    text = (predicate or "").strip()
    if not text or len(text.split()) > 6:
        return False
    if text.startswith("-") or text.endswith("-"):
        return False
    for left, right in (('"', '"'), ("'", "'"), ("(", ")"), ("[", "]")):
        if text.count(left) != text.count(right) if left != right \
                else text.count(left) % 2:
            return False
    return any(w not in FUNCTION
               for w in re.findall(r"[a-z']+", text.lower()))


def free_of_endpoints(predicate: str, subject: str, object_: str) -> bool:
    """Whether the predicate holds no word of either entity name.

    Word level, not substring: the text refers to `Etan Boritzer` as `Boritzer`
    after the first sentence, so a substring test on the canonical name misses
    the mention a predicate is most likely to contain. Capitalisation is required
    so this cannot fire on `of` inside `University of Colorado`.
    """
    parts = {w for name in (subject, object_)
             for w in re.findall(r"[A-Za-z][\w'-]*", name or "")
             if len(w) > 2 and w[:1].isupper()}
    words = set(re.findall(r"[A-Za-z][\w'-]*", predicate))
    if words & parts:
        return False
    for name in (subject, object_):
        if name and name in predicate:
            return False
    return True


def place(occ: dict, subject: str, object_: str, lines: list) -> dict | None:
    """Turn a (line, predicate) pair into a located occurrence, or reject it.

    The model gives a line and an exact substring; the position on that line is
    found here. Three things are settled by code rather than asked for:

    the offset, by searching the line. A model counts tokens, not characters:
    asked for a column it was right 35% of the time and self-consistently wrong
    the rest, reporting one predicate at the same false column three times.

    which occurrence, when the substring appears on the line more than once. The
    predicate of a triple lies between its two endpoints, so the occurrence
    bracketed by the subject and the object mentions is the one meant. Where
    neither endpoint is on the line -- a pronoun subject, an object named in the
    heading -- the first occurrence is taken and that is recorded, because it is
    a guess rather than a location.

    the sentence index, from the character offset. It was asked for before and
    the model could not produce it: it has to split sentences exactly as
    split_document does to number them, and on `A&R man`, `Jr.` and `U.S.` it
    does not. Counting the sentence boundaries before an offset we already hold
    cannot disagree with itself.
    """
    predicate = (occ.get("predicate") or "").strip()
    index = occ.get("line")
    if not predicate or not isinstance(index, int):
        return None
    if not states_a_relation(predicate):
        return None
    if not free_of_endpoints(predicate, subject, object_):
        return None
    if not 1 <= index <= len(lines):
        # a line number past the end is usually off by one; the substring still
        # has to be found, so every line is tried before the occurrence is lost
        index = next((n + 1 for n, line in enumerate(lines)
                      if predicate in line), None)
        if index is None:
            return None
    line = lines[index - 1]
    if predicate not in line:
        index = next((n + 1 for n, other in enumerate(lines)
                      if predicate in other), None)
        if index is None:
            return None
        line = lines[index - 1]

    starts = [i for i in range(len(line))
              if line.startswith(predicate, i)]
    bracketed = []
    for s in starts:
        left, right = line[:s], line[s + len(predicate):]
        if (subject in left and object_ in right) or \
                (object_ in left and subject in right):
            bracketed.append(s)
    start = bracketed[0] if bracketed else starts[0]

    # the document offset, so a span can be addressed without searching again
    before = sum(len(l) + 1 for l in lines[:index - 1])
    return {"line": index, "predicate": predicate,
            "line_start": start, "char_start": before + start,
            "ambiguous": len(starts) > 1 and not bracketed,
            # what dp.spans and dp.inplace read. The predicate is already the
            # narrow span those two existed to produce, so it is handed over
            # under the name they expect rather than being re-derived.
            "relation_mention": predicate,
            "sentence_id": _sentence_of(lines, index, start)}


def _sentence_of(lines: list, index: int, start: int) -> int:
    """Which sentence of the body an offset falls in, counting as spans.py does.

    Line 1 is the title, which split_document holds out, so the body is lines 2
    onward and sentence numbering starts there.
    """
    body_before = "".join(l + " " for l in lines[1:index - 1])
    text = body_before + lines[index - 1][:start] if index > 1 else ""
    return 1 + len(re.findall(r"(?<=[.!?])\s+", text))


def validate(record: dict, document: str) -> dict:
    """Check the invariants the write-back depends on, and record what failed.

    Nothing is dropped and nothing is repaired. A fact that cannot be located
    in the document cannot be sanitised later, and knowing how often that
    happens is the difference between a mechanism with a known coverage and
    one with an unknown one.
    """
    # name -> type, because the candidate pool is keyed on the endpoint types
    # and dropping them here would mean asking for them again later.
    #
    # A type outside the eleven is rejected rather than kept as `Other`. The
    # prompt asks for the closed inventory and mostly obeys it, but asking is
    # not enforcing, and an entity typed `concept` or `date` has no candidate
    # pool and no business in a relation this project perturbs. Rejecting it
    # here also drops every triple that used it, which is the point -- the
    # alternative is a triple whose endpoint type is `Other`, which the
    # mechanism skips anyway, counted as extracted.
    from dp.entity_types import ONTONOTES

    lines = document.split("\n")
    entities, off_inventory, not_in_text = {}, [], []
    document_words = set(_content_words(document))
    for e in record.get("entities", []):
        name, kind = e.get("name"), (e.get("type") or "").strip()
        if not name:
            continue
        if kind.upper() not in ONTONOTES:
            off_inventory.append(f"{name} ({kind or 'no type'})")
        elif not in_document(name, document_words):
            not_in_text.append(f"{name} ({kind.upper()})")
        else:
            entities[name] = kind.upper()

    triples, problems = [], []
    if off_inventory:
        problems.append({"why": f"{len(off_inventory)} entities outside the "
                                f"eleven types", "entities": off_inventory[:20]})
    if not_in_text:
        problems.append({"why": f"{len(not_in_text)} entities not in the "
                                f"document", "entities": not_in_text[:20]})

    for triple in record.get("triples", []):
        subject, relation = triple.get("subject"), triple.get("relation")
        obj = triple.get("object")
        if not (subject and relation and obj):
            problems.append({"triple": triple, "why": "incomplete"})
            continue
        if subject not in entities or obj not in entities:
            # the prompt requires both endpoints to be listed as entities; a
            # triple that breaks it has no type, so no candidate pool
            problems.append({"triple": [subject, relation, obj],
                             "why": "endpoint not in entities"})
            continue

        located, missing = [], 0
        for occ in triple.get("occurrences", []):
            placed = place(occ, subject, obj, lines)
            if placed is None:
                missing += 1
            else:
                located.append(placed)
        if missing:
            problems.append({"triple": [subject, relation, obj],
                             "why": f"{missing} predicate(s) not locatable"})
        if located:
            triples.append({**triple, "occurrences": located})

    return {"entities": entities, "triples": triples, "problems": problems}


async def extract_one(client, document: str, model: str, semaphore) -> dict:
    """One document's record, cached by (schema, model, document).

    The raw reply is cached beside the validated record, and `validate` is
    re-applied to it on every read. That is not how this started: the cache held
    only the validated record, so tightening a check meant paying the model again
    for every document to re-derive something the reply already contained. The
    predicate filter was tightened three times.
    """
    cache_path = os.path.join(CACHE, f"{fingerprint(document, model)}.json")
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path, encoding="utf-8"))
        raw = cached.get("raw")
        if raw is None:
            return cached
        try:
            record = validate(_json_object(raw), document)
        except Exception as exc:
            record = {"entities": {}, "triples": [],
                      "problems": [{"why": f"{type(exc).__name__}: {exc}"}]}
        record["document"] = document
        record["raw"] = raw
        return record

    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": PROMPT.format(
                classes=CLASS_BLOCK, ruled=ruled(document))}],
            **REASONING)
    try:
        record = validate(_json_object(reply.choices[0].message.content), document)
    except Exception as exc:
        record = {"entities": {}, "triples": [],
                  "problems": [{"why": f"{type(exc).__name__}: {exc}"}]}

    record["document"] = document
    record["raw"] = reply.choices[0].message.content or ""
    os.makedirs(CACHE, exist_ok=True)
    json.dump(record, open(cache_path, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    return record


async def extract(documents: list[str], model: str, concurrency: int) -> list[dict]:
    from openai import AsyncOpenAI

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    done = [0]

    async def one(document):
        record = await extract_one(client, document, model, semaphore)
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == len(documents):
            print(f"  extracted {done[0]}/{len(documents)}", flush=True)
        return record

    return await asyncio.gather(*(one(d) for d in documents))


def report(records: list[dict]) -> None:
    import collections
    import statistics

    triples = [t for r in records for t in r["triples"]]
    predicates = collections.Counter(t["relation"] for t in triples)
    types = collections.Counter(ty for r in records
                                for ty in r["entities"].values())
    print(f"\n{len(records)} documents, "
          f"{sum(len(r['entities']) for r in records)} entities, "
          f"{len(triples)} triples, {len(predicates)} distinct predicates, "
          f"{len(types)} distinct entity types")
    if triples:
        print(f"triples per document: median "
              f"{statistics.median(len(r['triples']) for r in records):.0f}, "
              f"max {max(len(r['triples']) for r in records)}")
        print(f"occurrences per triple: median "
              f"{statistics.median(len(t['occurrences']) for t in triples):.0f}")
    problems = collections.Counter(p["why"].split(" mention")[0]
                                   for r in records for p in r["problems"])
    if problems:
        print("\nproblems:")
        for why, n in problems.most_common():
            print(f"   {why:<40}{n:>6}")
    print("\nmost common entity types:")
    for ty, n in types.most_common(12):
        print(f"   {ty:<34}{n:>5}")
    print("\nmost common predicates:")
    for predicate, n in predicates.most_common(20):
        print(f"   {predicate:<34}{n:>5}")
    singles = sum(1 for _, n in predicates.items() if n == 1)
    print(f"\npredicates seen exactly once: {singles}/{len(predicates)} "
          f"({100 * singles / max(len(predicates), 1):.0f}%)")


def context_of(sample: int, seed: int, benchmark: str,
               subset_name: str | None = None) -> list[str]:
    """The passages the sampled questions name, gold and distractor alike.

    A question's `context` is the ten paragraphs the dataset puts in front of
    it: the ones that support the answer and the ones that do not. Sanitising
    those is what a pilot needs -- the rest of the corpus is not read while
    answering these questions, so extracting it would be paying for documents
    no measurement touches.

    HotpotQA and 2WikiMultiHopQA both spell `context` as [[title, sentences]],
    so the directory is the only thing that differs between them. It is read
    off the benchmark name rather than passed, because the sample has to be
    the same one run_bench.py answers and that is keyed on the name too.
    """
    from benchmarks import fixed_subset, load, stratified

    _, questions = load(benchmark)
    chosen = (fixed_subset(questions, subset_name) if subset_name
              else stratified(questions, sample, seed))
    wanted = {q.id for q in chosen}

    name, _, subset = benchmark.partition(":")
    directory = {"hotpotqa": "hotpotqa", "hotpotqa-dp": "hotpotqa",
                 "2wiki": "2wiki", "musique": "musique"}.get(name)
    if directory is None:
        raise SystemExit(f"--sample needs a dataset whose questions carry a "
                         f"context; {name!r} does not")
    root = os.path.join(ROOT, "datasets", directory)

    rows = json.load(open(os.path.join(root, f"{subset}_questions.json"),
                          encoding="utf-8"))
    titles = {title for row in rows if row["_id"] in wanted
              for title, _ in row["context"]}

    passages = json.load(open(os.path.join(root, f"{subset}.json"),
                              encoding="utf-8"))
    return [f"{p['title']}\n{p['text']}" for p in passages
            if p["title"] in titles]


def main(args) -> None:
    from benchmarks import load

    if args.sample or args.subset:
        documents = context_of(args.sample, args.seed, args.benchmark,
                               args.subset)
        what = (f"the context of the {args.subset} subset" if args.subset
                else f"the context of {args.sample} questions (seed {args.seed})")
    else:
        corpora, _ = load(args.benchmark)
        documents = [d for c in corpora for d in (c.parts or (c.text,))]
        what = "the whole corpus"
    if args.limit:
        documents = documents[: args.limit]
    print(f"{len(documents)} documents from {what}, model {args.model}")

    records = asyncio.run(extract(documents, args.model, args.concurrency))
    report(records)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"benchmark": args.benchmark, "model": args.model,
               "schema": SCHEMA, "sample": args.sample, "seed": args.seed,
               "documents": len(documents), "records": records},
              open(args.output, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"\nwritten to {args.output}")
    print(f"per-document cache under {CACHE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="hotpotqa:dev")
    p.add_argument("--sample", type=int,
                   help="extract the context of this many questions, the same "
                        "subset run_bench.py answers under --sample")
    p.add_argument("--seed", type=int, default=0, help="fixes the sampled subset")
    p.add_argument("--subset", metavar="NAME",
                   help="extract the context of datasets/subsets/NAME.json "
                        "instead of a proportional sample")
    p.add_argument("--limit", type=int, help="first N documents only")
    p.add_argument("--output", default=under("extraction_sample50.json"))
    p.add_argument("--model", default=EXTRACTOR)
    p.add_argument("--concurrency", type=int, default=6)
    main(p.parse_args())
