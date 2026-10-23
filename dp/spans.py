"""The minimal substring of a sentence that states a relation.

    python -m dp.spans --build
    python -m dp.spans

The recorded `relation_mention` is too wide to cut out. Of 2,165 occurrences,
86.1% contain the object entity and 73.3% run past five words; the longest is
37. Blanking one takes the relation and also the date, the apposition and the
second entity, and measured over the run where 90.1% of relations were
released unchanged, only half of a sentence's words survived into the scaffold
-- against 100% surviving the rebuild once they were in it. The loss is the
cut, not the writer.

So the cut is narrowed to the predicate. A model is asked for the exact
substring that states the relation and nothing else: no entity name, no date,
no modifier that would still be true if the relation were different.

    recorded   based on the Welsh CGI television series "Fireman Sam"
    minimal    based on

    recorded   The film premiered on February 22, 2009 at the closing Gala of
               the Jameson Dublin International Film Festival.
    minimal    premiered on

A rule was tried first -- strip the two entity strings and anything that looks
like a date -- and it cut the blanked words by only 27.9% while leaving
wreckage like "that premiered on at the 59th", because what is left of a span
after deleting substrings is not a phrase.

This pass is a function of the original sentence and the true triple, so it
does not depend on epsilon: it runs once and every run reads the cache. That
is what makes narrow spans cheap enough to use, and it is why substitution can
then be done without a model at all.
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

from dp.typemap import extraction  # noqa: E402


def split_document(document: str) -> tuple[str, list[str]]:
    """Title, then the body split into sentences.

    A passage is stored as "title\ntext" and the title has no full stop, so
    splitting the whole document on sentence punctuation glues the title to the
    first sentence. Every occurrence's sentence index would then be off by one,
    and anything that rewrote that sentence would rewrite the title with it --
    one passage came back as "Fireman Sam: dedicated to the Welsh CGI television
    series: The Great Fire of Pontypandy endorses 2009". The title is what the
    gold supporting facts are keyed on and what every retriever indexes, so it
    is held out here and never modified.

    This lived in dp/reconstruct.py, the scaffold-and-rebuild route that
    dp/inplace.py replaced. It is the only thing two live modules still wanted
    from that file, so it moved here -- sentence splitting is what this module
    does -- and the file was deleted.
    """
    title, _, body = document.partition("\n")
    parts = re.split(r"(?<=[.!?])\s+", body.strip())
    return title.strip(), [p for p in parts if p]

PATH = under("spans.json")

PROMPT = """For each item, copy out the shortest substring of the sentence that
states the given relation between the given subject and object.

Rules:
- Return a verbatim substring of the sentence, copied exactly, including case
  and punctuation. Do not paraphrase and do not add words.
- Include only what states the relation: usually a verb or a verb plus its
  preposition.
- Exclude the subject and object names, and any name at all.
- Exclude anything that would still be true if the relation were different:
  dates, numbers, places, titles, appositions, descriptive modifiers.
- Prefer the shortest span that a reader would still read as the relation.
  "was later amended as" over "amended"; "premiered on" over "The film
  premiered on February 22, 2009 at".
- The span must contain a verb or a participle. A bare preposition or article
  is not a relation: "in", "of", "by", "at", "is" alone are all wrong answers.
  If the sentence states the relation only by juxtaposition and no verb carries
  it, return the empty string "" instead of guessing a connective.
- Never return the whole sentence.
- The span must contain NO part of the subject name and NO part of the object
  name. If the only verb sits between them, return just that verb. If a
  preposition is glued to the object name, return the preposition alone.
- Two or three words is the usual answer. More than five is almost always
  wrong: look again for the verb inside what you were about to return.

More examples, all of which earlier attempts got wrong by returning too much:
  sentence: Hamilton Island Airport is located 8 NM north of the coastal city of Newcastle in the local government area of Port Stephens.
  subject: Hamilton Island Airport   object: Port Stephens   relation: located_in
  -> "is located"

  sentence: Head of a Woman is an early Cubist sculpture created circa 1911 by the Hungarian avant-garde sculptor Joseph Csaky.
  subject: Head of a Woman   object: Joseph Csaky   relation: created_by
  -> "created"

  sentence: Romance of the Three Kingdoms is a 2009 animated series of Japan.
  subject: Romance of the Three Kingdoms   object: Japan   relation: located_in
  -> ""      (nothing but the preposition states it, and "of" is not a relation;
              replacing it would also hit the "of" inside the title)

Examples:
  sentence: Fireman Sam: The Great Fire of Pontypandy is based on the Welsh CGI television series "Fireman Sam".
  subject: Fireman Sam   object: Welsh CGI television series   relation: based_on
  -> "is based on"

  sentence: The film premiered on February 22, 2009 at the closing Gala of the Jameson Dublin International Film Festival.
  subject: The film   object: Jameson Dublin International Film Festival   relation: premiered_at
  -> "premiered on"

  sentence: He worked for public service broadcaster Raidio Teilifis Eireann (RTE) and the "Sunday Tribune" newspaper.
  subject: He   object: Raidio Teilifis Eireann   relation: worked_for
  -> "worked for"

Return only a JSON object mapping each item's key to its span:
{{"<key>": "<span>", ...}}

Items:
{items}"""


# Words that cannot by themselves state a relation. A span made only of these
# is rejected; the point is to require a verb or participle, not to enumerate
# English.
FUNCTION = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "by", "for",
    "to", "with", "from", "as", "into", "onto", "under", "over", "near",
    "within", "is", "was", "are", "were", "be", "been", "being", "has", "have",
    "had", "its", "their", "his", "her", "s", "that", "which", "who", "whose",
}


def content_word(span: str) -> bool:
    return any(w not in FUNCTION
               for w in re.findall(r"[A-Za-z']+", span.lower()))


def occurrences() -> list[dict]:
    """Every (document, sentence, triple) the scaffold would cut.

    The sentence is resolved here rather than carried, so a span can be
    validated against the text it has to be a substring of.
    """
    out = []
    for record in extraction()["records"]:
        title, lines = split_document(record["document"])
        for triple in record["triples"]:
            for i, occ in enumerate(triple.get("occurrences") or []):
                mention = occ.get("relation_mention") or ""
                if not mention:
                    continue
                index = occ.get("sentence_id")
                if not (isinstance(index, int) and 1 <= index <= len(lines)
                        and mention in lines[index - 1]):
                    index = next((n + 1 for n, line in enumerate(lines)
                                  if mention in line), None)
                if index is None:
                    continue
                out.append({
                    "key": f"{title}|{index}|{triple['subject']}|"
                           f"{triple['object']}|{triple['relation']}|{i}",
                    "title": title, "sentence_id": index,
                    "sentence": lines[index - 1], "mention": mention,
                    "subject": triple["subject"], "object": triple["object"],
                    "relation": triple["relation"]})
    return out


async def _say(client, items, model, semaphore) -> dict:
    listing = "\n".join(
        f"  key={it['key']}\n    sentence: {it['sentence']}\n"
        f"    subject: {it['subject']}   object: {it['object']}   "
        f"relation: {it['relation']}" for it in items)
    async with semaphore:
        reply = await client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user",
                       "content": PROMPT.format(items=listing)}])
    match = re.search(r"\{.*\}", reply.choices[0].message.content or "", re.S)
    if not match:
        return {}
    try:
        said = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    by_key = {it["key"]: it for it in items}
    out = {}
    for key, span in said.items():
        it = by_key.get(key)
        if it is None or not isinstance(span, str):
            continue
        span = span.strip().strip('"')
        if not span or span not in it["sentence"]:
            # a span that is not literally in the sentence cannot be cut out
            # of it, so there is nowhere to substitute
            continue
        if len(span) >= len(it["mention"]):
            continue  # narrowed nothing
        # a span holding an entity makes substitution delete that entity:
        # `of Japan` replaced by `located in` loses Japan, and the entities
        # are the part of a sentence this project is trying hardest to keep.
        # Rejecting here puts the item back in the queue for a retry instead
        # of letting it through to do damage -- 61 got through before this
        # check existed, and 494 more arrived by falling back to the mention.
        if (it["subject"] and it["subject"] in span) or \
                (it["object"] and it["object"] in span):
            continue
        if not content_word(span):
            # A span of nothing but function words does not state a relation,
            # and substituting into one does damage out of proportion to its
            # size: `in` appeared 61 times, `of` 14, and the `of` inside
            # "Romance of the Three Kingdoms" is part of a title, not a
            # predicate. 193 of 1,732 spans were like this.
            continue
        out[key] = span
    return out


async def build(batch: int, model: str, concurrency: int) -> dict:
    from openai import AsyncOpenAI

    existing = (json.load(open(PATH, encoding="utf-8"))["spans"]
                if os.path.exists(PATH) else {})
    items = [it for it in occurrences() if it["key"] not in existing]
    if not items:
        print(f"nothing new; {len(existing)} spans already narrowed")
        return existing

    batches = [items[i:i + batch] for i in range(0, len(items), batch)]
    print(f"{len(items)} occurrences to narrow, {len(batches)} batches, {model}")
    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    done = [0]

    async def one(b):
        got = await _say(client, b, model, semaphore)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(batches):
            print(f"  {done[0]}/{len(batches)} batches", flush=True)
        return got

    results = await asyncio.gather(*(one(b) for b in batches))
    spans = dict(existing)
    for got in results:
        spans.update(got)

    missing = [it["key"] for it in items if it["key"] not in spans]
    print(f"{len(spans)} narrowed; {len(missing)} fell back to the mention")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump({"model": model, "spans": spans, "missing": missing},
              open(PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return spans


def load_spans() -> dict[str, str]:
    if not os.path.exists(PATH):
        raise SystemExit(f"no {PATH}; run with --build")
    return json.load(open(PATH, encoding="utf-8"))["spans"]


def report() -> None:
    spans = load_spans()
    items = {it["key"]: it for it in occurrences()}
    pairs = [(items[k]["mention"], v) for k, v in spans.items() if k in items]
    before = [len(m.split()) for m, _ in pairs]
    after = [len(s.split()) for _, s in pairs]
    print(f"{len(spans)}/{len(items)} occurrences narrowed "
          f"({100 * len(spans) / max(len(items), 1):.1f}%)\n")
    print(f"words cut: mention median {statistics.median(before):.0f} "
          f"mean {statistics.mean(before):.1f} max {max(before)}")
    print(f"           span    median {statistics.median(after):.0f} "
          f"mean {statistics.mean(after):.1f} max {max(after)}")
    print(f"           total blanked words down "
          f"{100 * (1 - sum(after) / sum(before)):.1f}%")
    holds = sum(1 for k, v in spans.items()
                if k in items and items[k]["object"] in v)
    print(f"spans still containing the object entity: {holds}/{len(pairs)} "
          f"({100 * holds / max(len(pairs), 1):.1f}%)")
    print("\nexamples:")
    for m, s in sorted(pairs, key=lambda p: -len(p[0]))[:6]:
        print(f"  mention: {m[:100]}")
        print(f"  span   : {s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args()
    if args.build:
        asyncio.run(build(args.batch, args.model, args.concurrency))
    report()
