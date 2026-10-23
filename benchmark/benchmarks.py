"""Benchmark loading: two corpora formats, one schema.

Pure data. Nothing here imports LightRAG or talks to a model, so the loaders
stay usable from a runner, a scorer, or a notebook.

    corpora, questions = load("hotpotqa:dev")
    corpora, questions = load("2wiki100-dp-delta:original")

A Question names the corpus it belongs to via `source`. What a runner does with
that is its own decision -- GraphRAG-Bench indexes each corpus separately and
answers only its own questions, while the LightRAG paper indexes an entire
domain together -- so `source` records provenance and never implies a topology.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")


@dataclass(frozen=True)
class Corpus:
    name: str
    text: str
    parts: tuple[str, ...] = ()
    """The documents to index separately, when the corpus has a unit of its own
    worth preserving. A retriever is given `parts` if they are there and `text`
    otherwise; `text` stays populated either way, because a scorer wants the
    corpus as one body of text regardless of how it was indexed."""


@dataclass(frozen=True)
class Question:
    id: str
    source: str  # name of the Corpus this question is answered against
    question: str
    answer: str
    question_type: str
    evidence: str = ""  # supporting passage, when the benchmark ships one
    # Other spellings the benchmark accepts for the same answer. MuSiQue ships
    # them and its own scorer maximises over them, so ignoring them scores a
    # right answer wrong: 28 of our 100 questions have one, and the gold for
    # `Rockland County` lists `Rockland County, New York` beside it. Empty for a
    # benchmark that ships none, which leaves those unaffected.
    aliases: tuple = ()


def pack(documents, budget=1100, model="gpt-4o"):
    """Group documents into windows of at most `budget` tokens, splitting none.

    The two things this is between. A corpus handed over as one body of text is
    cut into 1,200-token chunks wherever the count lands, which straddles the
    passage boundaries HotpotQA's gold supporting facts are stated at. Handing
    over 9,811 passages instead aligns them exactly -- and costs six times as
    much to index, because the extraction prompt carries 3,059 tokens of
    instruction and examples that are re-sent for every chunk, so a 130-token
    passage pays 23 times its own weight in overhead. It also takes the
    co-occurrence away: LightRAG finds a relation between two entities by
    seeing them in one window, and one passage per window leaves it only the
    relations stated inside a single passage, which is not where a multi-hop
    answer lives.

    Packing keeps whole passages and keeps several per window, so no boundary
    is cut and the indexing cost is unchanged. The count uses LightRAG's
    tokeniser so that a window it is given is a window it keeps.

    `budget` is its chunk STEP, not its chunk size. chunking_by_token_size
    walks `range(0, len(tokens), max_token_size - overlap_token_size)` and
    takes max_token_size from each start, so with its defaults it steps 1100
    and takes 1200: anything above 1100 tokens comes back as a full chunk plus
    a sliver. Budgeting at 1200 turned 1,154 windows into 1,890 chunks, and
    736 of those were offcuts of a few dozen tokens -- each paying the full
    3,059-token extraction prompt to extract entities from a fragment.
    """
    import tiktoken

    encoder = tiktoken.encoding_for_model(model)
    separator = len(encoder.encode("\n\n"))
    packed, window, used = [], [], 0
    for document in documents:
        size = len(encoder.encode(document))
        if window and used + separator + size > budget:
            packed.append("\n\n".join(window))
            window, used = [], 0
        window.append(document)
        used += size + (separator if len(window) > 1 else 0)
    if window:
        packed.append("\n\n".join(window))
    return tuple(packed)


def fixed_subset(questions, name):
    """The questions a saved subset names, in the file's order.

    Some subsets are chosen rather than sampled. datasets/subsets/*.json holds
    one: the 100 questions whose context covers the fewest documents, which is
    what makes re-indexing a sanitised corpus affordable -- on MuSiQue the same
    100 questions span 629 passages where a random 100 span 1,596.

    The selection is stratified by question type to the whole set's proportions.
    Minimising coverage alone collapses onto one type, because the passages that
    many questions share are the ones a single kind of question reuses: 2Wiki
    came back 100/100 compositional and MuSiQue 37% 4-hop against 17% in the
    full set, and either would have measured the hardest subtype rather than the
    dataset. Stratifying costs little -- 2Wiki 275 passages to 334, MuSiQue 596
    to 629 -- and keeps the number comparable with the published ones.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "subsets", f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no subset {name!r} at {path}")
    wanted = set(_read_json(path)["question_ids"])
    picked = [q for q in questions if q.id in wanted]
    if len(picked) != len(wanted):
        missing = wanted - {q.id for q in picked}
        raise SystemExit(f"subset {name!r}: {len(missing)} ids not in this "
                         f"benchmark, e.g. {sorted(missing)[:3]}")
    return sorted(picked, key=lambda q: q.id)


def stratified(questions, size, seed):
    """Sample proportionally across question types.

    The files are ordered by type, so a plain head-of-file sample is all Fact
    Retrieval -- the easiest level, and the one least sensitive to retrieval
    quality. Fixing the seed keeps the subset identical across runs, which is
    what makes two configurations comparable.
    """
    if size >= len(questions):
        return questions
    buckets = defaultdict(list)
    for q in questions:
        buckets[q.question_type].append(q)

    rng = random.Random(seed)
    picked = []
    for qtype in sorted(buckets):
        group = buckets[qtype]
        take = round(size * len(group) / len(questions))
        picked += rng.sample(group, min(max(take, 1), len(group)))
    return sorted(picked, key=lambda q: q.id)


def load(spec: str, data_root: str = DATA_ROOT) -> tuple[list[Corpus], list[Question]]:
    """`spec` is "<benchmark>:<subset>", e.g. "hotpotqa:dev"."""
    benchmark, _, subset = spec.partition(":")
    loaders = {"hotpotqa": _load_hotpotqa,
               "2wiki": _load_2wiki,
               "hotpotqa-dp": _load_hotpotqa_dp,
               "hotpotqa-dp-base": _load_hotpotqa_dp_base,
               "hotpotqa-dp-delta": _load_hotpotqa_dp_delta,
               "hotpotqa100-dp-base": _load_hotpotqa100_dp_base,
               "hotpotqa100-dp-delta": _load_hotpotqa100_dp_delta,
               "2wiki-dp-base": _load_2wiki_dp_base,
               "2wiki-dp-delta": _load_2wiki_dp_delta,
               "2wiki100-dp-base": _load_2wiki100_dp_base,
               "2wiki100-dp-delta": _load_2wiki100_dp_delta}
    if benchmark not in loaders:
        raise ValueError(f"unknown benchmark {benchmark!r}; try {sorted(loaders)}")
    if not subset:
        raise ValueError(f"spec {spec!r} is missing a subset, e.g. '{benchmark}:dev'")
    return loaders[benchmark](subset, data_root)


# ----------------------------------------------------------------- HotpotQA
# The 1,000-question sample of the distractor dev split that HippoRAG published
# and the graph-RAG papers after it reuse, so the numbers line up with theirs
# rather than with a subset of our own choosing. Its corpus ships deduplicated:
# the raw distractor format repeats a passage in every question that retrieves
# it, and the 9,811 here are the distinct ones.
#
# One index over all of them, not one per passage -- a question names no
# document, and answering it against its own two gold paragraphs would measure
# nothing. Pass --topology single.

def _load_hotpotqa(subset, data_root):
    root = os.path.join(data_root, "hotpotqa")
    passages = _read_json(os.path.join(root, f"{subset}.json"))
    documents = [f"{p['title']}\n{p['text']}" for p in passages]
    corpora = [Corpus(name="HotpotQA", text="\n\n".join(documents),
                      parts=pack(documents))]

    questions = []
    for q in _read_json(os.path.join(root, f"{subset}_questions.json")):
        # supporting_facts are (title, sentence index) into this question's own
        # context; the sentences themselves are what a scorer can use
        sentences = dict(q["context"])
        evidence = [sentences[title][i] for title, i in q["supporting_facts"]
                    if title in sentences and i < len(sentences[title])]
        questions.append(Question(
            id=q["_id"], source="HotpotQA", question=q["question"],
            answer=q["answer"], question_type=q["type"],
            evidence=" ".join(s.strip() for s in evidence)))

    _check_sources(corpora, questions, spec=f"hotpotqa:{subset}")
    return corpora, questions


# ---------------------------------------------------------- 2WikiMultiHopQA
# Welbl et al., and distributed with HippoRAG's reproduction set. 6,119
# passages and 1,000 questions, structured like HotpotQA's -- a title and a
# body per passage, supporting facts keyed on titles -- with one addition that
# matters here.
#
# Each question carries `evidences`: the Wikidata triples the answer rests on,
# with property names and QIDs.
#
#     [["Lothair II", "mother", "Ermengarde of Tours"],
#      ["Ermengarde of Tours", "date of death", "20 March 851"]]
#
# So the relations do not have to be extracted. They are annotated, they come
# from a closed vocabulary of 27 properties, and the ground truth for an attack
# is the dataset's own rather than one model's reading of the text. Every
# complaint this project accumulated about extracted gold on HotpotQA -- a
# predicate like `evolved on`, an object named nowhere in its sentence, a target
# that is only a point of comparison -- is answered by construction.
#
# What the annotation does not give is position: a triple says `mother` holds
# between two entities, not where in the passage it is stated. In-place
# substitution needs that, so locating the relation in the text is the one step
# this dataset still requires.

def _2wiki_questions(root, subset):
    questions = []
    for q in _read_json(os.path.join(root, f"{subset}_questions.json")):
        sentences = dict(q["context"])
        evidence = [sentences[title][i] for title, i in q["supporting_facts"]
                    if title in sentences and i < len(sentences[title])]
        questions.append(Question(
            id=q["_id"], source="2WikiMultiHopQA", question=q["question"],
            answer=q["answer"], question_type=q["type"],
            evidence=" ".join(s.strip() for s in evidence)))
    return questions


def _load_2wiki(subset, data_root):
    root = os.path.join(data_root, "2wiki")
    passages = _read_json(os.path.join(root, f"{subset}.json"))
    documents = [f"{p['title']}\n{p['text']}" for p in passages]
    corpora = [Corpus(name="2WikiMultiHopQA", text="\n\n".join(documents),
                      parts=pack(documents))]
    questions = _2wiki_questions(root, subset)
    _check_sources(corpora, questions, spec=f"2wiki:{subset}")
    return corpora, questions


def _load_2wiki100_dp_base(subset, data_root):
    return _dp_base(f"2wiki100-dp-base:{subset}", "2wiki", "2WikiMultiHopQA",
                    _2wiki_questions(os.path.join(data_root, "2wiki"), "dev"),
                    data_root, scope="_2wiki100")


def _load_2wiki100_dp_delta(subset, data_root):
    return _dp_delta(f"2wiki100-dp-delta:{subset}", subset, "2wiki",
                     "2WikiMultiHopQA",
                     _2wiki_questions(os.path.join(data_root, "2wiki"), "dev"),
                     data_root, scope="_2wiki100")


# ------------------------------------------------------- HotpotQA, sanitised
# The same 9,811 passages and the same 1,000 questions, with every document the
# mechanism touched replaced by its sanitised version. The subset is
# `<epsilon>_<seed>`, e.g. hotpotqa-dp:eps1_seed0.
#
# Titles are carried over verbatim and only bodies are rewritten, because the
# gold supporting facts are keyed on titles and so is retrieval: a sanitised
# corpus whose titles had drifted would score differently for a reason that
# has nothing to do with the mechanism.
#
# Which documents are sanitised is the pilot's scope, and it is worth stating
# plainly rather than hiding in a ratio. 500 passages were extracted -- the
# context of the 50 questions, ten each -- so every gold supporting passage
# for those questions is covered. The remaining 9,311 are indexed untouched;
# they are corpus-level distractors, and a question whose answer lives in one
# of them is not protected by this run.

def _load_hotpotqa_dp(subset, data_root):
    sanitised = os.path.join(data_root, "..", "workspace", "dp", "sanitized",
                             f"{subset}.json")
    if not os.path.exists(sanitised):
        raise SystemExit(f"no {sanitised}; run python -m dp.inplace first")
    rewritten = {r["title"]: r["sanitized"]
                 for r in _read_json(sanitised)["records"]}

    root = os.path.join(data_root, "hotpotqa")
    passages = _read_json(os.path.join(root, "dev.json"))
    documents = [rewritten.get(p["title"], f"{p['title']}\n{p['text']}")
                 for p in passages]
    corpora = [Corpus(name="HotpotQA", text="\n\n".join(documents),
                      parts=pack(documents))]

    questions = []
    for q in _read_json(os.path.join(root, "dev_questions.json")):
        sentences = dict(q["context"])
        evidence = [sentences[title][i] for title, i in q["supporting_facts"]
                    if title in sentences and i < len(sentences[title])]
        questions.append(Question(
            id=q["_id"], source="HotpotQA", question=q["question"],
            answer=q["answer"], question_type=q["type"],
            evidence=" ".join(s.strip() for s in evidence)))

    _check_sources(corpora, questions, spec=f"hotpotqa-dp:{subset}")
    return corpora, questions


# ------------------------------------------- HotpotQA, sanitised, split in two
# The same corpus as hotpotqa-dp, cut along the one line that matters: the 500
# passages the mechanism extracted from, which are the only ones any epsilon
# can change, and the 9,311 it never touched, which are byte-identical in every
# run.
#
# Indexing the whole corpus per epsilon re-paid for the identical 96% of it
# every time. Measured: 1,259 windows for one epsilon, of which the sanitised
# passages occupy 62. Building the untouched 1,200 once and inserting the 62
# on top cuts the per-epsilon index from about forty minutes to about three,
# which is what makes a real sweep affordable.
#
# That the two halves compose was checked rather than assumed, on a corpus
# small enough to count by hand: copying a two-document index and inserting a
# third took entities from 10 to 14 and relations from 3 to 7 -- merged, not
# appended, since an entity named in both halves stayed one node -- a repeated
# insert changed nothing, and a question whose answer bridges the two halves
# was answered from both.
#
# The split is by extraction, not by whether a document ended up rewritten.
# 500 were extracted and 482 yielded a triple; the other 18 are carried in the
# delta group even though they come through unchanged, because membership then
# does not depend on epsilon and the base index stays one artefact.

def _dp_touched(data_root, scope=""):
    """Titles of the passages the mechanism extracted from, in any run."""
    path = os.path.join(data_root, "..", "workspace", "dp" + scope,
                        "extraction_sample50.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run python -m dp.extract first")
    return {r["document"].split("\n", 1)[0]
            for r in _read_json(path)["records"]}


def _hotpot_questions(root):
    questions = []
    for q in _read_json(os.path.join(root, "dev_questions.json")):
        sentences = dict(q["context"])
        evidence = [sentences[title][i] for title, i in q["supporting_facts"]
                    if title in sentences and i < len(sentences[title])]
        questions.append(Question(
            id=q["_id"], source="HotpotQA", question=q["question"],
            answer=q["answer"], question_type=q["type"],
            evidence=" ".join(s.strip() for s in evidence)))
    return questions


# The base/delta split is identical for the two multi-hop datasets -- the only
# differences are the directory, the corpus name and how the questions are
# read -- so the two of them are one pair of functions parameterised by a
# dataset, not two pairs that have to be kept in step by hand.

def _dp_base(spec, directory, name, questions, data_root, scope):
    """The untouched passages only. Build once, reuse for every epsilon."""
    root = os.path.join(data_root, directory)
    touched = _dp_touched(data_root, scope)
    documents = [f"{p['title']}\n{p['text']}"
                 for p in _read_json(os.path.join(root, "dev.json"))
                 if p["title"] not in touched]
    corpora = [Corpus(name=name, text="\n\n".join(documents),
                      parts=pack(documents))]
    _check_sources(corpora, questions, spec=spec)
    return corpora, questions


def _dp_delta(spec, subset, directory, name, questions, data_root, scope):
    """Only the extracted passages, to insert on top of a base index.

    `subset` is a mechanism run -- eps1_seed0_inplace, identity_inplace -- or
    the word `original`, which carries the same passages unmodified and so
    measures the no-DP baseline under exactly this packing. Without it the
    baseline would be packed differently from the runs it anchors, and the gap
    between them would include that difference.
    """
    root = os.path.join(data_root, directory)
    touched = _dp_touched(data_root, scope)
    rewritten = {}
    if subset != "original":
        path = os.path.join(data_root, "..", "workspace", "dp" + scope,
                            "sanitized", f"{subset}.json")
        if not os.path.exists(path):
            raise SystemExit(f"no {path}; run python -m dp.inplace first")
        rewritten = {r["title"]: r["sanitized"]
                     for r in _read_json(path)["records"]}

    documents = [rewritten.get(p["title"], f"{p['title']}\n{p['text']}")
                 for p in _read_json(os.path.join(root, "dev.json"))
                 if p["title"] in touched]
    corpora = [Corpus(name=name, text="\n\n".join(documents),
                      parts=pack(documents))]
    _check_sources(corpora, questions, spec=spec)
    return corpora, questions


def _load_hotpotqa_dp_base(subset, data_root):
    return _dp_base(f"hotpotqa-dp-base:{subset}", "hotpotqa", "HotpotQA",
                    _hotpot_questions(os.path.join(data_root, "hotpotqa")),
                    data_root, scope="")


def _load_hotpotqa_dp_delta(subset, data_root):
    return _dp_delta(f"hotpotqa-dp-delta:{subset}", subset, "hotpotqa",
                     "HotpotQA",
                     _hotpot_questions(os.path.join(data_root, "hotpotqa")),
                     data_root, scope="")


# The same HotpotQA corpus under a separate scope, for the 100 questions chosen
# to cover the fewest passages. It is a second scope rather than a rebuild of the
# first because the 50-question pilot's extraction, pools, spans, mechanism
# traces, sanitised corpora and attack ground truth all live in workspace/dp/
# and a second --subset run would overwrite every one of them.

def _load_hotpotqa100_dp_base(subset, data_root):
    return _dp_base(f"hotpotqa100-dp-base:{subset}", "hotpotqa", "HotpotQA",
                    _hotpot_questions(os.path.join(data_root, "hotpotqa")),
                    data_root, scope="_hotpotqa100")


def _load_hotpotqa100_dp_delta(subset, data_root):
    return _dp_delta(f"hotpotqa100-dp-delta:{subset}", subset, "hotpotqa",
                     "HotpotQA",
                     _hotpot_questions(os.path.join(data_root, "hotpotqa")),
                     data_root, scope="_hotpotqa100")


def _load_2wiki_dp_base(subset, data_root):
    return _dp_base(f"2wiki-dp-base:{subset}", "2wiki", "2WikiMultiHopQA",
                    _2wiki_questions(os.path.join(data_root, "2wiki"), "dev"),
                    data_root, scope="_2wiki")


def _load_2wiki_dp_delta(subset, data_root):
    return _dp_delta(f"2wiki-dp-delta:{subset}", subset, "2wiki",
                     "2WikiMultiHopQA",
                     _2wiki_questions(os.path.join(data_root, "2wiki"), "dev"),
                     data_root, scope="_2wiki")


# ------------------------------------------------------------------ helpers

def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fingerprint(text: str) -> str:
    """Identify a document by content. LightRAG strips on insert, so do the
    same here -- otherwise documents with surrounding whitespace get a
    different identity inside the index than they have in the eval set."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _first(answers):
    if isinstance(answers, list):
        return str(answers[0]) if answers else ""
    try:
        parsed = ast.literal_eval(answers)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
    except (ValueError, SyntaxError):
        pass
    return str(answers)


def _check_sources(corpora, questions, spec):
    known = {c.name for c in corpora}
    orphans = {q.source for q in questions} - known
    if orphans:
        raise ValueError(
            f"{spec}: {len(orphans)} question source(s) have no corpus, "
            f"e.g. {sorted(orphans)[:3]}"
        )


if __name__ == "__main__":  # quick inventory
    import collections
    import sys

    for spec in sys.argv[1:] or ["hotpotqa:dev", "2wiki:dev"]:
        corpora, questions = load(spec)
        chars = sum(len(c.text) for c in corpora)
        types = collections.Counter(q.question_type for q in questions)
        print(f"{spec:<26} {len(corpora):>3} corpora, {chars / 1e6:5.2f}M chars, "
              f"{len(questions):>5} questions")
        for name, n in types.most_common():
            print(f"{'':28} {name:<22} {n:>5}")
