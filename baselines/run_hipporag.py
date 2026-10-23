"""Answer a benchmark with HippoRAG 2, writing predictions our scorers can read.

    python baselines/run_hipporag.py --benchmark hotpotqa:dev --sample 20 \
        --passages_from_questions --out_dir runs/_hippo
    python benchmark/score_hotpot.py runs/_hippo/predictions_*.json

HippoRAG's own main.py reads its copy of the data and reports its own EM and
F1. Neither is what we want: the comparison with LightRAG is only worth making
if both sides answer the same question list and are graded by the same code.
So the questions come from benchmark/benchmarks.py and the scoring happens
afterwards, in benchmark/score_hotpot.py.

The baseline itself is unmodified -- baselines/HippoRAG is the upstream clone,
and everything here does is build its BaseConfig and reshape its output.

Its defaults are kept where they are its method (retrieval_top_k=200 feeding
linking_top_k=5 and qa_top_k=5, passage_node_weight=0.05) and overridden only
to match what LightRAG runs on: gpt-4o-mini reading, text-embedding-3-small
retrieving, both through whatever OPENAI_BASE_URL points at. Upstream runs
every method on Llama-3.3-70B with NV-Embed-v2, which by its own table is
worth 12.5 F1 on HotpotQA, so its published numbers are not these numbers.

One mismatch cannot be configured away: rerank_dspy_file_path is a DSPy
program optimised against Llama-3.3-70B, and the triple filter runs it on
gpt-4o-mini. It is upstream's only such file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIPPO = os.path.join(ROOT, "baselines", "HippoRAG")
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))
sys.path.insert(0, HIPPO)

from benchmarks import fixed_subset, load, stratified  # noqa: E402


def dataset_dir(benchmark: str) -> str:
    """Which datasets/ directory this benchmark's passages and questions live in.

    It was hard-coded to hotpotqa in both readers. Running 2wiki through them
    therefore loaded HotpotQA's 9,811 passages and filtered them by 2Wiki's
    titles, which matched 6 of 334: the index held six passages, gold recall was
    0/100, and HippoRAG 2 answered "Insufficient information" to 72 questions.
    It reads as a retrieval failure and is a wrong file.
    """
    name = benchmark.partition(":")[0]
    directory = {"hotpotqa": "hotpotqa", "2wiki": "2wiki",
                 "musique": "musique"}.get(name)
    if directory is None:
        raise SystemExit(f"no dataset directory known for {name!r}")
    return os.path.join(ROOT, "datasets", directory)


def passages(benchmark: str) -> list[str]:
    """The corpus as HippoRAG wants it: one "title\\ntext" string per passage.

    Our loader concatenates a corpus into a single document, which is right for
    LightRAG -- it chunks what it is given -- and wrong here, because a passage
    is a node in this graph and the unit retrieval returns.
    """
    raw = json.load(open(os.path.join(dataset_dir(benchmark),
                                      f"{benchmark.partition(':')[2]}.json"),
                         encoding="utf-8"))
    return [f"{p['title']}\n{p['text']}" for p in raw]


SANITIZED = [None]


def sanitized_passages(path: str) -> dict:
    """title -> rewritten body, from a dp.inplace run.

    Both of these baselines read the corpus out of datasets/, which is the
    original text. Without this they would index the undefended passages while
    the configuration under test names a sanitised corpus, and the utility
    reported for every epsilon would be the undefended utility. LightRAG avoids
    it only because its loader (benchmarks._dp_delta) does the substitution.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f"no sanitised corpus at {path}")
    with open(path, encoding="utf-8") as handle:
        return {r["title"]: r["sanitized"]
                for r in json.load(handle)["records"]}


def needed_by(questions, benchmark: str) -> list[str]:
    """Only the passages these questions name, gold and distractor alike.

    A smoke test over 9,811 passages pays for OpenIE on all of them. This cuts
    that to the couple of hundred the sample touches -- which also makes
    retrieval much easier, so a score from it says the plumbing works and
    nothing about how well the method does.
    """
    subset = benchmark.partition(":")[2]
    rows = json.load(open(os.path.join(dataset_dir(benchmark),
                                       f"{subset}_questions.json"),
                          encoding="utf-8"))
    wanted = {q.id for q in questions}
    titles = {title for row in rows if row["_id"] in wanted
              for title, _ in row["context"]}
    rewritten = sanitized_passages(SANITIZED[0])
    return [rewritten.get(p.split("\n", 1)[0], p)
            for p in passages(benchmark)
            if p.split("\n", 1)[0] in titles]


ANSWER_FORMAT_NOTE = (
    "\n\nFormat your reply exactly as the examples above: your reasoning after "
    "\"Thought:\", then a final line beginning with \"Answer:\" followed by the "
    "answer alone -- a name, a date, a yes/no, or a short phrase -- and nothing "
    "after it."
)


class _FormatPinned:
    """A qa_llm wrapper that spells out the `Answer:` requirement.

    HippoRAG extracts the prediction with
    `response_content.split('Answer:')[1]`, and falls back to the whole reply
    when that raises. gpt-4o-mini follows the few-shot scaffold and emits the
    marker every time; gemini-2.5-flash skipped it on 55 of 200 questions, so a
    third of the answers came back as reasoning prose -- mean answer length 250
    characters against gpt-4o-mini's 19 -- and scored EM 0 whatever they said.
    That is a format mismatch, not a weaker reader, and leaving it in would have
    put a 16-point drop in the table that the model did not earn.

    The note is appended per request rather than added to the template on
    purpose: it reaches only a run that passes --qa_model --pin_answer_format,
    so every gpt-4o-mini result on disk stays reproducible byte for byte and
    none of them need re-running. The cost is that the two columns are then
    prompted differently, which belongs in the caption: the pinned column is
    "this reader, told how to format", not "this reader".
    """

    def __init__(self, inner):
        self.inner = inner

    def infer(self, messages, **kwargs):
        patched = [dict(m) if isinstance(m, dict) else m for m in messages]
        for message in reversed(patched):
            if isinstance(message, dict) and message.get("role") == "user":
                message["content"] = str(message.get("content", "")) \
                    + ANSWER_FORMAT_NOTE
                break
        return self.inner.infer(patched, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _Defended:
    """A qa_llm wrapper carrying a prompt-level defense, for HippoRAG.

    HippoRAG has no relation table and no response template to patch: its
    reader is handed ranked passages under its own QA prompt. What a
    prompt-level defense is, though, is an instruction the deployment adds to
    its reader, and the reader call is reachable the same way `_FormatPinned`
    reaches it -- by wrapping `rag.qa_llm` after construction, which touches
    nothing under baselines/HippoRAG/. The text comes from
    `defenses.reader_prompt`, so it is the same wording LightRAG and PathRAG
    get, and it goes in as a leading system message rather than being appended
    to the user turn, because that is where the other two carry it.

    ID Alignment and Decoy cannot be expressed here: there is no ID column to
    align and no endpoint columns to decoy. `reader_prompt` returns "" for
    them, and the runner refuses rather than writing a file named after a
    defense that did nothing.
    """

    def __init__(self, inner, text: str):
        self.inner, self.text = inner, text

    def infer(self, messages, **kwargs):
        patched = [dict(m) if isinstance(m, dict) else m for m in messages]
        if patched and isinstance(patched[0], dict) \
                and patched[0].get("role") == "system":
            patched[0]["content"] = self.text + "\n\n" \
                + str(patched[0].get("content", ""))
        else:
            patched.insert(0, {"role": "system", "content": self.text})
        return self.inner.infer(patched, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main(args) -> None:
    from src.hipporag.HippoRAG import HippoRAG
    from src.hipporag.utils.config_utils import BaseConfig

    _, questions = load(args.benchmark)
    if args.subset:
        questions = fixed_subset(questions, args.subset)
    elif args.sample:
        questions = stratified(questions, args.sample, args.seed)
    docs = (needed_by(questions, args.benchmark) if args.passages_from_questions
            else passages(args.benchmark))
    print(f"{len(questions)} questions, {len(docs)} passages")

    base_url = os.getenv("OPENAI_BASE_URL")
    config = BaseConfig(
        save_dir=args.save_dir,
        llm_base_url=base_url,
        llm_name=args.llm_model,
        embedding_base_url=base_url,
        embedding_model_name=args.embedding_model,
        embedding_provider="openai",
        dataset=args.benchmark.replace(":", "_"),
        # upstream's own retrieval settings: a wide candidate pool, narrowed by
        # the triple filter, and only the last five passages reach the reader
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=args.qa_top_k,
        embedding_batch_size=args.embedding_batch_size,
        openie_mode="online",
        # Upstream raises if any one chunk fails extraction, which at 9,811
        # chunks means the index does not get built: one failure left the
        # embedding stores written and the graph absent, and the next attempt
        # refused to continue on that inconsistency. This authorises the graph
        # rebuild; per its own docstring the embedding and OpenIE stores are
        # reused, and the LLM cache makes the successful calls free to repeat.
        force_index_from_scratch=args.force_index,
        rerank_dspy_file_path=os.path.join(
            HIPPO, "src", "hipporag", "prompts", "dspy_prompts",
            "filter_llama3.3-70B-Instruct.json"),
    )

    with HippoRAG(global_config=config) as rag:
        rag.index(docs)
        if args.qa_model:
            # Swap only the reader, after construction.
            #
            # The constructor's own hooks cannot do this. `qa_llm=` alone also
            # becomes the extraction LLM (`self.llm_model = extraction_llm or
            # qa_llm`), and passing both plus the `index_identity` that then
            # becomes mandatory lands in the manifest validator, which compares
            # the index's recorded producer against the configured one. That
            # validator is right to refuse: the OpenIE results in this save_dir
            # were produced by `llm_name`, and renaming the files to claim
            # otherwise is how the first attempt at this failed
            # (StateConsistencyError: "OpenIE provenance is missing or
            # incompatible").
            #
            # Replacing `rag.qa_llm` after `index()` leaves construction,
            # validation and retrieval exactly as they were -- the graph, the
            # embeddings and the OpenIE results all stay the ones `llm_name`
            # built -- and changes only which model writes the answer. That is
            # the variable this experiment wants isolated.
            from dataclasses import replace
            from src.hipporag.llm import _get_llm_class
            qa = _get_llm_class(replace(config, llm_name=args.qa_model))
            if args.pin_answer_format:
                qa = _FormatPinned(qa)
            rag.qa_llm = qa
            print(f"QA reader overridden: {config.llm_name} builds and "
                  f"retrieves, {args.qa_model} answers"
                  + ("  [Answer: format pinned]" if args.pin_answer_format
                     else ""))
        if args.defense != "none" or args.safe_prompt:
            sys.path.insert(0, os.path.join(ROOT, "benchmark"))
            import defenses
            defenses.install(args.defense, seed=args.seed,
                             safe_prompt=args.safe_prompt, victim="hipporag")
            text = defenses.reader_prompt()
            if text:
                rag.qa_llm = _Defended(rag.qa_llm, text)
            # The context side, for the two analogues. HippoRAG's own QA
            # prompt lists passages as `Wikipedia Title: ...` with no number,
            # so ID Alignment has nothing to align on this side and the
            # utility cell is the safe prompt alone -- which is the honest
            # cost of a defense that, here, changes nothing the reader sees.
            # Decoy's trailing lines are added by wrapping `retrieve` on the
            # instance, so `qa` builds its prompt from the decoyed passages
            # and nothing under baselines/HippoRAG/ is touched.
            if args.defense == "decoy":
                triples = defenses.load_openie_triples(os.path.join(
                    args.save_dir, f"openie_results_ner_{config.llm_name}.json"))
                undecoyed = rag.retrieve

                def retrieve(*a, **k):
                    result = undecoyed(*a, **k)
                    solutions = result[0] if isinstance(result, tuple) else result
                    for sol in solutions:
                        blocks = defenses.passage_blocks(
                            list(sol.docs or []), triples, label="")
                        sol.docs = [b.lstrip("\n") for b in blocks]
                    return result

                rag.retrieve = retrieve
            elif args.defense == "id_align":
                print("hipporag: ID Alignment has no passage numbers to align "
                      "in HippoRAG's own QA prompt; this utility cell carries "
                      "the safe prompt only")
            if not text and args.defense != "decoy":
                raise SystemExit(f"--defense {args.defense}: nothing reaches "
                                 f"HippoRAG's reader; refusing to write a "
                                 f"file named after it")
            print(f"QA reader defended: {args.defense}"
                  + (" +safe_prompt" if args.safe_prompt else "")
                  + (f" ({len(text)} chars prepended as a system message)"
                     if text else ""))
        if args.qa_cache_tag:
            # Point the reader's cache at a file of its own, after index().
            #
            # HippoRAG caches every LLM call in save_dir/llm_cache/<model>.sqlite
            # on a key covering the messages and the generation parameters, so
            # repeating a configuration already measured returns the first run's
            # answers and the "repeat" measures nothing. Deleting that file is
            # not the fix: it also holds the OpenIE calls that built this index,
            # and losing them forces a re-extraction that would change the graph
            # the run is supposed to share.
            #
            # Redirecting only `cache_file_name`, and only after index() has
            # returned, leaves the OpenIE entries where they are and sends the
            # QA calls to an empty database.
            old = rag.qa_llm.cache_file_name
            base, ext = os.path.splitext(old)
            rag.qa_llm.cache_file_name = f"{base}_{args.qa_cache_tag}{ext}"
            print(f"QA cache redirected to "
                  f"{os.path.basename(rag.qa_llm.cache_file_name)} "
                  f"(index cache {os.path.basename(old)} untouched)")
        solutions, _, _ = rag.rag_qa(queries=[q.question for q in questions])

    records = [{
        "benchmark": args.benchmark,
        "id": q.id,
        "question": q.question,
        "source": q.source,
        "evidence": q.evidence,
        "question_type": q.question_type,
        "ground_truth": q.answer,
        # what the reader was actually given: HippoRAG.py truncates the ranked
        # list to qa_top_k before building the prompt, so recording all of
        # s.docs would record the candidate pool instead of the context
        "context": "\n\n".join((s.docs or [])[:args.qa_top_k]),
        "generated_answer": s.answer or "",
    } for q, s in zip(questions, solutions)]

    os.makedirs(args.out_dir, exist_ok=True)
    slug = args.benchmark.replace(":", "_")
    d = ("" if args.defense == "none" and not args.safe_prompt
         else f"_def-{args.defense}" + ("_safe" if args.safe_prompt else ""))
    out = os.path.join(args.out_dir,
                       f"predictions_{slug}_hipporag2_top{args.qa_top_k}{d}.json")
    json.dump(records, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {len(records)} predictions to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="hotpotqa:dev")
    p.add_argument("--sample", type=int)
    p.add_argument("--seed", type=int, default=0, help="fixes the sampled subset")
    p.add_argument("--subset", metavar="NAME",
                   help="answer the questions datasets/subsets/NAME.json names, "
                        "so every method answers the same list")
    p.add_argument("--passages_from_questions", action="store_true",
                   help="index only the passages the sample names; for a smoke "
                        "test, and not comparable with a full-corpus run")
    p.add_argument("--sanitized", metavar="PATH",
                   help="a dp.inplace corpus; its passages replace the "
                        "originals, which is what makes an epsilon run an "
                        "epsilon run")
    p.add_argument("--llm_model", default="gpt-4o-mini")
    p.add_argument("--pin_answer_format", action="store_true",
                   help="append the `Answer:` format requirement to each QA "
                        "prompt; only applies together with --qa_model, so no "
                        "existing single-model result is affected")
    p.add_argument("--qa_model", default=None, metavar="MODEL",
                   help="answer with this model while --llm_model keeps "
                        "building and retrieving; leaves the index's OpenIE "
                        "provenance intact, so an existing save_dir is reused "
                        "rather than re-extracted. Unset means one model does "
                        "both, as before.")
    p.add_argument("--embedding_model", default="text-embedding-3-small")
    p.add_argument("--qa_top_k", type=int, default=5)
    p.add_argument("--force_index", action="store_true",
                   help="rebuild the graph, reusing the embedding and OpenIE "
                        "stores; needed after a failed index leaves the two "
                        "out of step")
    p.add_argument("--embedding_batch_size", type=int, default=64)
    p.add_argument("--save_dir", default=os.path.join(ROOT, "workspace", "hipporag"))
    p.add_argument("--qa_cache_tag", default=None, metavar="TAG",
                   help="answer with a reader cache of its own, named with "
                        "TAG. Needed to repeat a configuration already on "
                        "disk: its QA prompts are byte-identical, so a cached "
                        "rerun returns the first run's answers. The index's "
                        "own OpenIE cache entries are left alone")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "_hippo"))
    p.add_argument("--defense", default="none",
                   help="a prompt-level defenses.MODES entry (abstractive, "
                        "extractive, protective), prepended to the reader's "
                        "system message; id_align and decoy are refused, "
                        "HippoRAG building no relation table")
    p.add_argument("--safe_prompt", action="store_true",
                   help="carry the safe prompt as well, as the attack driver "
                        "does by default")
    _args = p.parse_args()
    SANITIZED[0] = _args.sanitized
    main(_args)
