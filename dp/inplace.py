"""M5, the other way: substitute the relation where it stands.

    python -m dp.inplace --identity
    python -m dp.inplace --epsilon 1 --seed 0

The scaffold-and-rebuild route blanks a span and has a model write the
sentence again. Measured on the run where 90.1% of relations came back
unchanged, that still cost 8 EM against the untouched corpus, and the loss was
located: of a sentence's words, half died at the cut and essentially none died
in the rebuild. Rebuilding was never the problem. Cutting that wide was.

So nothing is rebuilt. The narrow span from dp.spans is replaced by the
released relation's surface form and every other character of the sentence
stays as it was:

    He worked for public service broadcaster Raidio Teilifis Eireann (RTE)
       ^^^^^^^^^^
    He is affiliated with public service broadcaster Raidio Teilifis Eireann (RTE)

The date, the apposition, the second entity, the clause that follows -- all of
it survives, because none of it is touched. And since substitution is a string
operation, a run costs no model calls at all: the 1,232 rebuild calls per
epsilon become zero, and only dp.spans pays a model, once, for every epsilon.

What this gives up is stated plainly. The surviving words are the author's own,
so wording next to the span that implies the true relation survives with them,
where a wide cut would have taken it. This is a weaker exposure than
scaffolding, deliberately: the relation that is released is still the
mechanism's, but the sentence around it is no longer laundered.

`--identity` substitutes each relation with itself. The corpus that comes back
is what this pipeline does when the mechanism does nothing, so the gap to the
untouched corpus is the pipeline's own cost, with no privacy in it -- the
number the scaffold route never had.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import difflib
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

from dp.spans import load_spans, occurrences, split_document  # noqa: E402

OUT_DIR = under("sanitized")
PERTURBED = under("perturbed")


DATE = re.compile(
    r"^(\d{1,4}(st|nd|rd|th)?|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|circa|c\.)$", re.I)

# a predicate head begins at one of these or at a verb; the list is what
# survived inspecting the spans this rule actually has to handle
HEAD = re.compile(
    r"\b((?:is|was|are|were|has|have|had|been|being)\s+)?"
    r"(?:[a-z]+(?:ed|es|s|ing)|[a-z]+)\s*"
    r"(?:in|on|at|by|of|for|to|with|from|as|into|under|near|within)?\b")


_PHRASES: dict = {}


def surface(relation: str, pair: tuple | None = None) -> str:
    """The relation as a predicate phrase, grammatical where that is known.

    `relation.replace("_", " ")` is what this was, and it puts ungrammatical
    text into the corpus: `located_in` becomes `located in`, so a sentence reads
    `X located in Y` with no copula, and `reigned_over` lands as `was reigned
    over` where the span already carried `was`.

    dp.verbalize has already written a grammatical sentence for every relation
    and type pair, for the mechanism's distances -- "A person is a member of an
    organization." -- and the predicate is what sits between the two type words.
    Taking it from there costs nothing, the file is already built and cached, and
    the phrase inserted into the text is one a reader wrote rather than one a
    `replace` produced.
    """
    plain = relation.replace("_", " ").strip()
    if pair is None:
        return plain
    phrase = _predicate_phrase(relation, pair)
    return phrase or plain


def _predicate_phrase(relation: str, pair: tuple) -> str | None:
    """The middle of a verbalised sentence: everything but the two type words."""
    if not _PHRASES:
        try:
            from dp.verbalize import READABLE, load_sentences
            _PHRASES["sentences"] = load_sentences()
            _PHRASES["readable"] = READABLE
        except SystemExit:
            _PHRASES["sentences"] = {}
            _PHRASES["readable"] = {}
    sentences = _PHRASES.get("sentences") or {}
    readable = _PHRASES.get("readable") or {}
    a, b = pair
    said = sentences.get(f"{relation}|{a}|{b}")
    if not said:
        return None
    subject = readable.get(a, a.lower())
    object_ = readable.get(b, b.lower())
    text = said.strip().rstrip(".")
    # "A person is a member of an organization" -> "is a member of"
    head = re.match(rf"^(?:an?|the)\s+{re.escape(subject)}\s+", text, re.I)
    if not head:
        return None
    text = text[head.end():]
    tail = re.search(rf"\s+(?:an?|the)?\s*{re.escape(object_)}$", text, re.I)
    if tail:
        text = text[:tail.start()]
    text = text.strip()
    return text if 0 < len(text.split()) <= 6 else None


def fit(span: str, replacement: str) -> str:
    """Carry the span's leading auxiliary and trailing comma into the new text.

    A span is a predicate phrase in a sentence and a relation name is a
    predicate phrase out of one, so the two do not always join. Three cases are
    handled and the rest is left alone, because a surface repair that guesses is
    worse than a rough join: a reader and an extractor both recover
    `is affiliated with` from `is is affiliated with`, but neither recovers a
    clause that was rewritten away.
    """
    lead = re.match(r"^(is|was|are|were|has|have|had|who|which|that)\s+", span,
                    re.I)
    tail = re.search(r"([,;:]\s*)$", span)
    out = replacement
    if lead and not re.match(r"^(is|was|are|were|has|have|had)\b", out, re.I):
        head = lead.group(1)
        # "who"/"which"/"that" introduce a clause and need a verb after them; a
        # bare relation name like `rival of` would leave "who rival of"
        out = (f"{head} is {out}" if head.lower() in ("who", "which", "that")
               else f"{head} {out}")
    if tail:
        out += tail.group(1)
    return out


def name_parts(*entities: str) -> list[str]:
    """The words of an entity name that can stand for it on their own.

    A canonical name is not how the text refers to the entity throughout.
    `Etan Boritzer` is `Boritzer` after the first sentence, `Alex Cox` is `Cox`,
    and `General Karl Freiherr Mack von Leiberich` is `General Mack`. Matching
    only the full string therefore misses the mentions a span is most likely to
    contain, which is how `Boritzer was first` came to be replaced wholesale.

    Capitalisation is required, so this cannot fire on an ordinary word that
    happens to sit in a name -- `of` in `University of Colorado`, `is` in
    `What is God?`. Short tokens are dropped for the same reason: a two-letter
    capital is as likely to be an initial or a roman numeral as a surname, and
    cutting a span at one gains nothing.
    """
    out = set()
    for entity in entities:
        for word in re.findall(r"[A-Za-z][\w'-]*", entity or ""):
            if len(word) > 2 and word[:1].isupper():
                out.add(word)
    return sorted(out, key=len, reverse=True)


def without_entities(span: str, subject: str, object_: str) -> str | None:
    """Shrink a span so it holds no part of either entity name.

    Replacing a span that contains an entity deletes that entity: `of Japan`
    overwritten by `located in` loses Japan, and 30.5% of substitutions were
    doing exactly that. The span is cut back to its longest run that touches
    neither name, so the entity stays where the author put it.

    The cut is made on the single words of a name as well as on the whole
    string, because the text uses both. See name_parts.
    """
    pieces = [span]
    for entity in sorted({subject, object_}, key=len, reverse=True):
        if not entity:
            continue
        pieces = [p for piece in pieces for p in piece.split(entity)]
    for part in name_parts(subject, object_):
        pieces = [p for piece in pieces
                  for p in re.split(rf"\b{re.escape(part)}\b", piece)]
    pieces = [p for p in (piece.strip() for piece in pieces) if p]
    if not pieces:
        return None
    best = max(pieces, key=len)
    return best if best in span else None


# A capitalised word inside a span, once both entity names have been cut out of
# it, is a third thing the author wrote and the relation does not need: a proper
# noun, or -- far more often in this corpus -- a demonym or a language.
#
#     is an American            ->  is writer of
#     is an Irish-              ->  is born in
#     holding Argentine nationality  ->  has nationality
#     with Swiss nationality    ->  has nationality
#
# Every one of those substitutions deletes the only word that answers the
# question. 2WikiMultiHopQA asks for a nationality or a language outright and
# its own convention is to answer with the adjective, so `American` is not
# decoration around the relation, it is the gold answer sitting inside the span.
# Replacing it protects nothing -- the relation `is_writer_of` is still plainly
# there in the rest of the sentence -- and costs the question.
#
# Only inner words are tested. A span may well start at a sentence boundary, and
# the capital there is the sentence's, not a name's.
INNER_CAPITAL = re.compile(r"\S+\s+(.*)$", re.S)


def holds_a_name(span: str, subject: str, object_: str) -> bool:
    """Whether a capitalised word survives in the span that is not an endpoint."""
    rest = INNER_CAPITAL.match(span.strip())
    if not rest:
        return False
    known = set(name_parts(subject, object_))
    for word in re.findall(r"[A-Za-z][\w'-]*", rest.group(1)):
        if len(word) > 2 and word[:1].isupper() and word not in known:
            return True
    return False


# A span dp.spans accepted is not always only a predicate. It passes every test
# that file applies -- a literal substring, shorter than the mention, free of
# both entity names, carrying a content word -- and can still be most of a
# clause: `was the tenth and longest-reigning sultan of`, `are the only two
# women in Ottoman history to be regarded as official regents`. Replacing those
# wholesale deletes "the tenth and longest-reigning" along with the relation.
# Measured on eps=1, 62 of 1,679 substitutions (3.7%) dropped three or more
# words and every one came from a span this check now shortens.
#
# So the span is narrowed once more at substitution time, keeping the modifiers
# where the author put them and swapping only the verb that carries the
# relation.
# The predicate is found rather than carved out by position. A first version cut
# at a run of leading modifiers and got the direction wrong -- it kept `the only
# two` and replaced `women in Ottoman history to be regarded as official
# regents` -- because the modifiers in these spans are not all at the front.
#
# The relation in a span like `was the tenth and longest-reigning sultan of` is
# carried by the auxiliary and the final preposition; `sultan`, `tenth`,
# `longest-reigning` describe an entity. So the predicate is taken to be the
# leading auxiliary plus the trailing preposition, and everything between them
# is kept where the author put it.
AUX = r"(?:is|are|was|were|be|been|being|has|have|had|who|which|that)"
PREP = r"(?:of|in|on|at|by|for|to|with|from|as|into|under|over|near|within)"
SPLIT = re.compile(rf"^((?:{AUX})\s+)?(.*?)(\s+{PREP})$", re.I)


# Taking a long span apart and reassembling it was tried here and abandoned.
# The idea was to keep the modifiers where the author put them and swap only the
# verb, so that `was the tenth and longest-reigning sultan of` kept "the tenth
# and longest-reigning". It does reduce deletion -- substitutions losing three
# or more words fell from 62 to 22 -- and it does not survive reading the output:
#
#   is the official radio broadcaster of -> is broadcaster of official radio
#   holds a large stake in               -> holds stake in large
#   was the tenth ... sultan of          -> was reigned over tenth and
#                                           longest-reigning sultan
#
# An auxiliary, a description and a preposition are one construction in English;
# detaching the description and appending it after the new predicate puts it on
# the wrong side of the preposition. A first version also duplicated whatever the
# relation and the description had in common (43 of 1,554 substitutions), and
# removing the echo only exposed the word-order problem underneath.
#
# The span is therefore replaced whole, as it was. What that costs is in the
# report: 1.3% of substitutions drop three or more words. Repairing the join is
# left to --polish, which reads the result rather than guessing at its syntax.


# Whether a span plausibly expresses the relation it was located for. The three
# checks dp.spans applies -- a literal substring, shorter than the mention, free
# of both entity names -- do not test this, so spans that name a thing rather
# than assert a relation get through and are then overwritten:
#
#   the county seat   -> located in      Massachusetts,  -> located in
#   home              -> located in      Puccini's opera -> composed by
#   originating       -> voiced          owner of        -> located in
#
# That is how two of the four questions identity answered wrong were lost: the
# passage about Worcester had `the county seat` and `Massachusetts,` replaced,
# and the question asks which Massachusetts county the county seat belongs to.
# Both words needed to answer it were gone.
#
# A span passes if it shares a word stem with the relation, or if it contains a
# verb. Either alone rejects too much: stems alone would refuse `is
# headquartered in` for `located_in`, which is the extractor doing its job, and
# verbs alone would refuse `pastor of` for `pastor_of`. Together they reject
# 4.8% of spans, and the ones they reject include every span traced to a wrong
# answer.
# Case-sensitive on the inflection branch: `United` ends in `ed` and is not a
# verb, and treating it as one let `Massachusetts, United States` pass as a
# predicate and be overwritten with `located in`, deleting both the state and
# the country. Auxiliaries and the listed verbs stay case-insensitive, since a
# sentence can open with one.
VERBISH = re.compile(
    r"\b(?:[a-z]+(?:ed|ing|es)\b"
    r"|is|are|was|were|be|been|being|has|have|had|co-\w+|became|becomes"
    r"|make|made|take|took|give|gave|hold|holds|held|win|won|run|runs|ran"
    r"|lead|leads|led|own|owns|found|build|built|write|wrote|play|plays"
    r"|serve|serves|act|acts|star|stars|connect|connects|offer|offers"
    r"|threaten|threatens|follow|follows|cover|covers|border|borders)\b")


CASED = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|became|becomes|made|took"
    r"|gave|held|won|ran|led|found|built|wrote|played|served)\b", re.I)


def _stems(text: str) -> set:
    from dp.spans import FUNCTION

    out = set()
    for word in re.findall(r"[A-Za-z]+", (text or "").lower()):
        if word in FUNCTION or len(word) < 3:
            continue
        out.add(re.sub(r"(?:ing|ed|es|s|ion|er|ment)$", "", word)[:6])
    return out


def expresses(span: str, relation: str) -> bool:
    return (bool(_stems(span) & _stems(relation))
            or bool(VERBISH.search(span)) or bool(CASED.search(span)))


def predicate_head(span: str) -> str | None:
    """The verb-ish head of a span the model would not narrow.

    Only reached when dp.spans has no span for an occurrence -- 21% of them.
    Replacing the recorded mention instead was measured at a median of 7 words
    and a maximum of 32, so a rule takes one more attempt before the occurrence
    is skipped. Dates and numbers are dropped first, since a mention like
    "located 8 NM north of the coastal city of Newcastle" is mostly geography.

    A stricter version of this was tried and reverted. It required the candidate
    to contain a verb, to run two to four words, and to occur exactly once in
    its document, which does stop the rule returning `of`, `and` or `won` --
    spans that locate nothing and protect nothing when replaced. But it also
    took the occurrences it could no longer serve from 17 to 190, so 11.3% of
    relations would have been left in the text unprotected, and the utility it
    bought back was 52.00 EM against 50.00: inside the +/-2 EM that two runs of
    the same corpus differ by. A privacy cost that large for a difference that
    small is not a trade worth making.
    """
    from dp.spans import content_word

    words = [w for w in span.split() if not DATE.match(w.strip(".,;:()\"'"))]
    if not words:
        return None
    for size in (3, 2, 1):
        for start in range(len(words) - size + 1):
            candidate = " ".join(words[start:start + size])
            # the same bar the model's spans are held to: a run of function
            # words is not a relation, so the rule may not produce one either
            if candidate in span and content_word(candidate):
                return candidate
    return None


def released_of(epsilon: float | None, seed: int, variant: str) -> dict:
    """(subject, object, relation) -> released relation, or identity."""
    if epsilon is None:
        return {}
    path = os.path.join(PERTURBED, f"eps{epsilon:g}_seed{seed}{variant}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run python -m dp.mechanism first")
    trace = json.load(open(path, encoding="utf-8"))["trace"]
    return {(t["document"], t["subject"], t["object"], t["relation"]):
            (t["released"], tuple(t["pair"])) for t in trace}


POLISH = """Repair the grammar of the passage below. Some words were replaced \
in place, so the joins can be ungrammatical.

Change as little as possible. Only the joins are wrong; the rest of the passage
is correct and must come back identical, word for word.

Allowed: fix agreement, a doubled or missing auxiliary, an article, word order
within the replaced phrase, punctuation at the join.

Not allowed:
- rewording, shortening, expanding or reordering anything that reads correctly
- changing any entity name, number, date or quantity
- changing any relation; if a relation reads oddly or looks factually wrong,
  leave it exactly as it is -- it is not a mistake to repair
- adding or removing information, including explanatory clauses
- touching the first line, which is the title

Most passages need one or two words changed. Returning the passage unchanged is
the right answer when the joins already read.

Return only the passage.

Passage:
{document}"""

# How far a polished passage may drift from the one handed to it. The pass is
# meant to repair joins, and a reader asked to repair will sometimes rewrite
# instead; a rewrite would both lose the author's wording and give the model a
# chance to restore a relation the mechanism replaced. A passage that comes back
# too different is discarded in favour of the rough one, which is at worst
# ungrammatical and at best still carries what the mechanism released.
POLISH_FLOOR = 0.90


async def polish_all(records: list, model: str, concurrency: int) -> None:
    """Make the substituted passages read as English, without reading the original.

    The polisher sees only text the mechanism has already rewritten, so this is
    post-processing of private output and adds no privacy loss. What it must not
    do is undo the substitution: a reader that can guess the original relation
    from the surrounding sentence might "correct" the released one back, and that
    would quietly return the corpus to its undefended state. The instruction
    therefore tells it that an odd-looking relation is not an error, and
    `report` measures how often the released relation survives the pass.
    """
    from openai import AsyncOpenAI

    todo = [r for r in records if r["touched"]]
    if not todo:
        return
    print(f"polishing {len(todo)} changed passages with {model}")
    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    done = [0]

    async def one(record):
        async with semaphore:
            try:
                reply = await client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": POLISH.format(
                        document=record["sanitized"])}])
                text = (reply.choices[0].message.content or "").strip()
            except Exception:
                text = ""
        done[0] += 1
        if done[0] % 100 == 0 or done[0] == len(todo):
            print(f"  polished {done[0]}/{len(todo)}", flush=True)
        if not text:
            return
        # the heading is held out: it is what retrieval and the gold supporting
        # facts are keyed on
        head = record["sanitized"].split("\n", 1)[0]
        body = text.split("\n", 1)[1] if "\n" in text else text
        candidate = head + "\n" + body.strip()
        similarity = difflib.SequenceMatcher(
            None, record["sanitized"], candidate).ratio()
        if similarity < POLISH_FLOOR:
            record["polish_rejected"] = round(similarity, 3)
            return
        record["rough"] = record["sanitized"]
        record["sanitized"] = candidate
        record["polish_similarity"] = round(similarity, 3)

    await asyncio.gather(*(one(r) for r in todo))


def run(epsilon: float | None, seed: int, variant: str,
        polish: str | None = None, concurrency: int = 10) -> dict:
    spans = load_spans()
    released = released_of(epsilon, seed, variant)

    # group by sentence: several relations can be stated in one, and each
    # substitution has to see the offsets the previous ones left behind
    from dp.typemap import extraction
    documents = {r["document"].split("\n", 1)[0]: r["document"]
                 for r in extraction()["records"]}

    # Pass one: which sentences go. A triple whose released relation is null is
    # not rewritten -- every sentence holding one of its recorded occurrences is
    # removed, and with it whatever else that sentence said. Deletion precedes
    # replacement, so a substitution that would have landed in a deleted
    # sentence never happens; the same relation's other occurrences, in
    # sentences that survive, are substituted as usual.
    #
    # Deleting the sentence rather than the predicate span is what null means.
    # Blanking the span alone leaves `Faith Julia Goldy right-wing writer`: the
    # gap says a relation stood there and the surrounding words still suggest
    # which, so withholding would be announced and only half done.
    doomed: set = set()
    if released:
        for it in occurrences():
            got = released.get((it["title"], it["subject"], it["object"],
                                it["relation"]))
            if got and got[0] is None:
                doomed.add((it["title"], it["sentence_id"]))

    per_sentence = collections.defaultdict(list)
    skipped: list = []
    unchanged: list = []
    withheld: list = []
    for it in occurrences():
        if (it["title"], it["sentence_id"]) in doomed:
            withheld.append(it["key"])
            continue
        span, how = spans.get(it["key"]), "narrowed"
        if span is None:
            span, how = predicate_head(it["mention"]), "rule"
        if span is not None:
            trimmed = without_entities(span, it["subject"], it["object"])
            if trimmed != span:
                how += "+trimmed"
            span = trimmed
        if span is not None and holds_a_name(span, it["subject"], it["object"]):
            # a proper noun or a demonym survived the entity cut, so replacing
            # this span would delete a word the author chose and the relation
            # does not carry. See holds_a_name.
            span = None
        if span is not None and not expresses(span, it["relation"]):
            # The span names something rather than asserting this relation, so
            # overwriting it would delete evidence and protect nothing. Before
            # giving up on the occurrence, the rule gets a turn on the full
            # mention: a narrowed span can fail this test while the mention
            # still contains a phrase that passes it, and skipping outright
            # took the unprotected share from 1.0% to 18.8%.
            fallback = predicate_head(it["mention"])
            if fallback:
                fallback = without_entities(fallback, it["subject"],
                                            it["object"])
            if fallback and not holds_a_name(fallback, it["subject"],
                                             it["object"]) \
                    and expresses(fallback, it["relation"]):
                span, how = fallback, "rule-after-reject"
            else:
                span = None
        if span is None:
            # nothing can be replaced here without deleting an entity or a
            # clause. The occurrence is left alone, which leaves the author's
            # own wording -- and so the true relation -- in the text, so it is
            # counted and reported rather than quietly dropped.
            skipped.append(it["key"])
            continue
        key = (it["title"], it["subject"], it["object"], it["relation"])
        got = released.get(key) if released else None
        new, pair = got if got else (it["relation"], None)
        if new is None:
            # its sentence is in `doomed` and was skipped above; reaching here
            # means the occurrence sits in a sentence no other record named, so
            # there is nothing to delete and nothing to write
            withheld.append(it["key"])
            continue
        if released and new == it["relation"]:
            # The mechanism released this fact's own relation, so there is
            # nothing to sanitise and the author's wording stands. Substituting
            # anyway rewrote it into the extractor's normalised predicate and
            # lost whatever else the span carried -- `was the tenth and
            # longest-reigning sultan of` became `was reigned over` -- for no
            # privacy at all.
            #
            # Only when a mechanism ran. `--identity` exists to measure what the
            # mapping costs on its own, so it substitutes every span with the
            # true relation and the gap to the untouched corpus is that cost.
            # Skipping there would make the corpus identical and the diagnostic
            # vacuous.
            unchanged.append(it["key"])
            continue
        per_sentence[(it["title"], it["sentence_id"])].append(
            {**it, "span": span, "released": new, "how": how, "pair": pair,
             "narrowed": it["key"] in spans})

    records, swaps = [], []
    for title, document in documents.items():
        heading, lines = split_document(document)
        out = list(lines)
        touched = False
        gone = sorted(i for t, i in doomed if t == title)
        for index in range(1, len(lines) + 1):
            if (title, index) in doomed:
                continue
            jobs = per_sentence.get((title, index))
            if not jobs:
                continue
            text = lines[index - 1]
            # apply from the end so an earlier offset stays valid, and skip a
            # span that overlaps one already replaced
            placed = []
            for job in sorted(jobs, key=lambda j: -text.find(j["span"])):
                start = text.find(job["span"])
                if start < 0:
                    continue
                end = start + len(job["span"])
                if any(start < e and end > s for s, e in placed):
                    continue
                placed.append((start, end))
                replacement = fit(job["span"],
                                  surface(job["released"], job.get("pair")))
                text = text[:start] + replacement + text[end:]
                swaps.append({**{k: job[k] for k in
                                 ("title", "sentence_id", "subject", "object",
                                  "relation", "released", "span", "narrowed")},
                              "how": job["how"],
                              "replacement": replacement})
            if text != lines[index - 1]:
                touched = True
            out[index - 1] = re.sub(r"\s+", " ", text).strip()
        if gone:
            # retained sentences keep their order; the title is never a
            # candidate for deletion, because split_document holds it out and
            # the gold supporting facts and every retriever are keyed on it
            out = [line for i, line in enumerate(out, 1)
                   if (title, i) not in doomed]
            touched = True
        # A document nothing was substituted in is returned as it was, not
        # reassembled. Rejoining its sentences with single spaces changed five
        # of five hundred passages under --identity -- whitespace only, but a
        # corpus that is meant to be identical should be identical.
        records.append({"title": title,
                        "sanitized": (heading + "\n" + " ".join(out)
                                      if touched else document),
                        "original": document, "touched": touched})

    if polish:
        asyncio.run(polish_all(records, polish, concurrency))

    name = ("identity" if epsilon is None
            else f"eps{epsilon:g}_seed{seed}{variant}") + "_inplace" \
        + ("_polished" if polish else "")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.json")
    json.dump({"mode": "inplace", "epsilon": epsilon, "seed": seed,
               "variant": variant, "documents": len(records),
               "swaps": len(swaps), "skipped": skipped,
               "unchanged": len(unchanged), "withheld": withheld,
               "deleted_sentences": sorted(f"{t}|{i}" for t, i in doomed),
               "polished": bool(polish),
               "records": records, "detail": swaps},
              open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return {"path": path, "records": records, "swaps": swaps,
            "skipped": skipped, "unchanged": unchanged,
            "withheld": withheld, "doomed": doomed}


def report(result: dict) -> None:
    swaps, records = result["swaps"], result["records"]
    if result.get("doomed"):
        gone = len(result["doomed"])
        lost = len(result.get("withheld") or [])
        docs = len({t for t, _ in result["doomed"]})
        print(f"\n{gone} sentences deleted across {docs} documents, taking "
              f"{lost} occurrences with them: a triple whose relation was "
              f"withheld removes every sentence that stated it, and whatever "
              f"else those sentences said.")
    if result.get("unchanged"):
        print(f"\n{len(result['unchanged'])} occurrences left untouched: the "
              f"mechanism released their own relation")
    print(f"\n{len(swaps)} substitutions across {len(records)} documents, "
          f"{len(result['skipped'])} occurrences skipped")
    for how, n in collections.Counter(s["how"] for s in swaps).most_common():
        print(f"  {how:<18}{n:>6}  ({100 * n / max(len(swaps), 1):.1f}%)")
    held = sum(1 for s in swaps
               if s["object"] in s["span"] or s["subject"] in s["span"])
    print(f"  spans still holding an entity: {held}")
    if not swaps:
        print("  nothing was substituted, so the corpus is the original")
        return
    cut = [len(s["span"].split()) for s in swaps]
    print(f"  words replaced: median {statistics.median(cut):.0f}, "
          f"mean {statistics.mean(cut):.1f}, max {max(cut)}")
    same = sum(1 for s in swaps if s["released"] == s["relation"])
    print(f"  relation unchanged:    {same}/{len(swaps)} "
          f"({100 * same / max(len(swaps), 1):.1f}%)")

    def words(text):
        return set(re.findall(r"[A-Za-z0-9À-ɏ]+", text.lower()))

    kept, ratios = [], []
    for r in records:
        o, n = words(r["original"]), words(r["sanitized"])
        if o:
            kept.append(len(o & n) / len(o))
        ratios.append(difflib.SequenceMatcher(
            None, r["original"], r["sanitized"]).ratio())
    print(f"  original words surviving: median {statistics.median(kept):.3f}, "
          f"mean {statistics.mean(kept):.3f}")
    print(f"  character similarity:     median {statistics.median(ratios):.3f}")
    print(f"  documents left identical: "
          f"{sum(1 for r in records if not r['touched'])}/{len(records)}")
    polished = [r for r in records if "rough" in r]
    rejected = [r for r in records if "polish_rejected" in r]
    if polished or rejected:
        sims = [r["polish_similarity"] for r in polished]
        print(f"\n  polished {len(polished)} passages, "
              f"rejected {len(rejected)} for drifting too far")
        if sims:
            print(f"  similarity to the rough text: median "
                  f"{statistics.median(sims):.3f}, min {min(sims):.3f}")
        kept_orig = [difflib.SequenceMatcher(None, r["original"],
                                             r["sanitized"]).ratio()
                     for r in records if r["touched"]]
        if kept_orig:
            print(f"  similarity to the ORIGINAL after polishing: median "
                  f"{statistics.median(kept_orig):.3f}")
    if any("rough" in r for r in records):
        kept = sum(1 for s in swaps
                   if surface(s["released"]).split()[0].lower()
                   in next((r["sanitized"].lower() for r in records
                            if r["title"] == s["title"]), ""))
        print(f"  released relation still present after polishing: "
              f"{kept}/{len(swaps)} ({100 * kept / max(len(swaps), 1):.1f}%)"
              f"  <- a drop here would be the polisher undoing the mechanism")
    print("\nexamples:")
    for s in swaps[:5]:
        print(f"  [{s['title']}] {s['relation']} -> {s['released']}")
        print(f"      {s['span']!r} -> {s['replacement']!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epsilon", type=float,
                   help="a mechanism run; omit with --identity")
    p.add_argument("--identity", action="store_true",
                   help="substitute every relation with itself")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="")
    p.add_argument("--polish", nargs="?", const="gpt-4o-mini", default=None,
                   metavar="MODEL",
                   help="repair the grammar of the changed passages; the "
                        "polisher never sees the original text")
    p.add_argument("--concurrency", type=int, default=10)
    args = p.parse_args()
    if not args.identity and args.epsilon is None:
        raise SystemExit("pass --epsilon or --identity")
    result = run(None if args.identity else args.epsilon, args.seed,
                 args.variant, args.polish, args.concurrency)
    report(result)
    print(f"\n-> {result['path']}")
