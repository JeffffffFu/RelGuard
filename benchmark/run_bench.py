"""Index a benchmark with LightRAG and answer its questions.

Writes predictions in GraphRAG-Bench's schema, so the scorer -- theirs or
score_hotpot.py -- can read the output of any runner, not just this one.

Two indexing topologies, because the benchmarks disagree about it:

    per-source   one index per corpus, each answering only its own questions.
                 GraphRAG-Bench's protocol, and the only honest one when a
                 question names the document it came from.
    single       every corpus in one index. What the LightRAG paper does, and
                 what makes retrieval a real search rather than a lookup in a
                 document already known to be the right one.

The retrieval context is captured with a second, context-only query rather than
by patching kg_query to return it -- the two calls send an identical keyword
extraction prompt, so the second hits LightRAG's response cache and the source
tree stays untouched.

    python run_bench.py --benchmark 2wiki100-dp-delta:original --subset 2wiki_min100 \
        --topology single --mode hybrid --top_k 10
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
import re
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)  # importable whether run from here or from the root
sys.path.insert(0, os.path.join(ROOT, "LightRAG-v1.0.3"))  # the vendored, patched LightRAG

from benchmarks import fixed_subset, load, stratified
from openai import APIStatusError

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache

# Answer shape, per GraphRAG-Bench difficulty level. v1.0.3's aquery takes no
# system_prompt, but response_type lands in the answer prompt's
# "---Target response length and format---" slot, so no source change is needed.
#
# Shape has to be set, or LightRAG writes multi-paragraph essays against
# reference answers of a dozen words and precision-based metrics collapse. It
# also has to vary: one "direct and concise" instruction for every level told
# the model to refuse the role-play questions at the top of the taxonomy, whose
# references run to 140 words -- 87.5% of Creative Generation came back "I don't
# know" while the retrieved context was there all along.
#
# Nothing here tells the model how to decline; LightRAG's own answer prompt
# already says "If you don't know the answer, just say so. Do not make anything
# up." Repeating that in the format slot turned refusal into a format option.
ANSWER_SHAPE = {
    "Relation 1-hop": (
        "Exactly one label from the list offered, and nothing else: no "
        "punctuation, no explanation, no restatement of the question."),
    "Relation 2-hop": (
        "The entity names alone, separated by semicolons, and nothing else. "
        "If the knowledge base shows none, reply exactly: none."),
    # HotpotQA. Gold answers run to two words at the median and eighteen at the
    # longest, and both metrics are computed over the whole reply: a sentence
    # around the right span scores 0 EM and a diluted F1. Nothing here tells
    # the model how to decline -- LightRAG's answer prompt already does.
    "bridge": (
        "The answer alone -- a name, phrase, date or number, copied from the "
        "context. No sentence, no explanation, no restatement of the question. "
        "Give your single best answer even when the context is thin: a guess "
        "can be right, and declining is scored as wrong."),
    "comparison": (
        "The answer alone, with no sentence and no explanation. Usually that is "
        "one of the alternatives the question names, copied verbatim; reply with "
        "exactly yes or no only when the question can be answered with nothing "
        "else. Give your single best answer rather than declining."),
    # 2WikiMultiHopQA. Its question types invite the reasoning chain into the
    # reply, and both metrics read the whole reply: on 50 questions without
    # these entries, `compositional`, `bridge_comparison` and `inference` all
    # scored 0 EM while answering correctly -- "Benjamin Pierce Cheney Jr.'s
    # wife, Julia Arthur, is from Canada." against a gold of `Canadian`, and
    # "The film 'Je Fais Le Mort' (2013) was directed by..." against a gold
    # that is the film's name. Overall EM was 8.00 where HotpotQA's is 56.00.
    # The reasoning still happens; it just must not reach the reply.
    # The four conventions below are this dataset's, not this method's, and the
    # same instruction is given to every run so that the comparison between
    # them is unaffected. Each was read off the 21 `compositional` questions in
    # the sample: a date written `February 24, 1761` against a gold of
    # `24 February 1761`, `New York City` against `Manhattan`, `Canada` against
    # `Canadian`, and three replies that were still sentences.
    "_2wiki_shape": (
        "Give only the answer, with nothing before or after it -- not a "
        "sentence, not a restatement of the question, not the reasoning. "
        "Four conventions this corpus follows:\n"
        "- dates read day month year, as in 24 February 1761\n"
        "- a place is named as narrowly as the context allows: Manhattan, not "
        "New York City; Richmond, not Richmond, Virginia\n"
        "- a nationality or a language is given as the adjective: Canadian, "
        "not Canada; French, not France\n"
        "- a title is copied as the context spells it\n"
        "Answer even when the context is thin: a guess can be right, and "
        "declining is scored as wrong."),
    # MuSiQue. Its hops are labelled 2hop/3hop/4hop and nothing in its answers
    # is type-specific, so the three share one block. Two things set it apart
    # from 2Wiki, both read off the 100 questions in the subset:
    #
    #   the answer is not always a short span. 17% run five words or more, and
    #   some are whole clauses -- `The dynasty regrouped and defeated the
    #   Portuguese`, `a long period of prosperity for the British people`. An
    #   instruction to answer in a word or two would score those wrong for
    #   obeying it, so the rule is "as the context puts it", not "short".
    #
    #   the chain is 3 or 4 hops for 48% of them, against 2 for all of HotpotQA,
    #   so naming the intermediate entity is the likeliest way to lose a point.
    "_musique_shape": (
        "Give only the answer, with nothing before or after it -- not a "
        "sentence of your own, not a restatement of the question, not the "
        "reasoning.\n"
        "- copy the answer as the context words it. It may be a name, a date, a "
        "number, a phrase or a whole clause; match the context's length rather "
        "than shortening it\n"
        "- the question chains two to four facts. Work the chain out silently "
        "and reply with its endpoint only, never an intermediate entity\n"
        "- a date is copied in the context's own format\n"
        "Answer even when the context is thin: a guess can be right, and "
        "declining is scored as wrong."),
    "compositional": (
        "Work out the chain of facts silently and reply with only its endpoint, "
        "never the intermediate entity. "),
    "bridge_comparison": (
        "Normally one of the alternatives the question names, copied verbatim. "
        "Compare the two silently; do not state the dates you compared. "),
    "inference": (
        "Reply with the entity or value the question asks for. "),
    "Fact Retrieval": (
        "A direct answer in plain text -- a phrase or one short sentence, "
        "with no explanation, headings or bullet points."
    ),
    "Complex Reasoning": (
        "A concise answer in plain text, at most two or three sentences, "
        "stating the conclusion and the reasoning that connects it."
    ),
    "Contextual Summarize": (
        "A coherent summary in plain prose, roughly 100 words, "
        "without headings or bullet points."
    ),
    "Creative Generation": (
        "A considered answer in plain prose, roughly 150 words. Reason beyond "
        "the retrieved passages where the question calls for it, grounding the "
        "response in them without restricting it to what they state verbatim."
    ),
}
DEFAULT_SHAPE = "A direct and concise answer in plain text."


# A provider that caps concurrent spend rather than request rate answers 402,
# and LightRAG's own retry wrapper does not cover it: tenacity is configured for
# RateLimitError, APIConnectionError and APITimeoutError, so an APIStatusError
# goes straight up through the gather and takes the indexing run with it. That
# failure is not loud -- the chunks are already embedded by then, so the working
# directory is left with a full vdb_chunks.json and an empty graph, which reads
# as a finished index until it is queried.
#
# OpenRouter sends Retry-After with the 402 and it is honoured; the fallback is
# for a provider that does not.
BUDGET_RETRIES = 8
BUDGET_WAIT = 120.0


def _json_body(text: str) -> str:
    """The JSON object inside a keyword-extraction reply, or the reply as it came.

    LightRAG parses that reply with a bare `json.loads(result)`, above a comment
    that says the extraction is "handled in use_model_func" -- upstream's own
    model function strips the prose around the object, and ours never did,
    because `keyword_extraction` arrives as a named parameter while the body it
    guards pops it from kwargs (see `reader`). With gpt-4o-mini that went
    unnoticed: it answers with a bare object. gemini-2.5-flash wraps the same
    object in a ```json fence, `json.loads` raises, LightRAG returns
    `fail_response`, and retrieval never runs -- 100 questions came back at
    EM 0.00 and F1 0.59 before this.

    A direct parse is tried first, so a reply that already worked is returned
    byte-identical and no existing result moves.
    """
    if not isinstance(text, str):
        return text
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def reader(model: str):
    """LightRAG's llm_model_func for an arbitrary OpenAI-compatible model.

    Its own gpt_4o_mini_complete hard-codes the name, and `keyword_extraction`
    reaches it as a named parameter while the body pops it from **kwargs -- so
    the structured-output path it guards has never run. Matching that here
    rather than fixing it keeps a change of reader from also being a change of
    how keywords are parsed.
    """
    async def llm(prompt, system_prompt=None, history_messages=[],
                  keyword_extraction=False, **kwargs):
        from tenacity import RetryError
        for attempt in range(BUDGET_RETRIES):
            try:
                out = await openai_complete_if_cache(
                    model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages, **kwargs)
                return _json_body(out) if keyword_extraction else out
            except (APIStatusError, RetryError) as exc:
                # LightRAG's own tenacity gives a 429 three quick tries and
                # then raises RetryError; a provider that has rate-limited a
                # model upstream ("temporarily rate-limited upstream") needs
                # minutes, not seconds, so the wait happens here. The 402
                # in-flight budget case keeps its Retry-After.
                cause = exc.last_attempt.exception() if isinstance(exc, RetryError) else exc
                status = getattr(cause, "status_code", None)
                if status not in (402, 429, 500, 502, 503, 529) \
                        or attempt == BUDGET_RETRIES - 1:
                    raise
                headers = getattr(getattr(cause, "response", None), "headers", None) or {}
                wait = (float(headers.get("Retry-After") or BUDGET_WAIT)
                        if status == 402 else 20.0 * (attempt + 1))
                print(f"  {status} from the provider, waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{BUDGET_RETRIES})", flush=True)
                await asyncio.sleep(wait)
    return llm


async def answer_questions(rag, questions, benchmark, param_of, concurrency, progress):
    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        shape = ANSWER_SHAPE.get(q.question_type, DEFAULT_SHAPE)
        # 2WikiMultiHopQA's four types share a block of corpus conventions;
        # the per-type entry says only what is specific to that type
        if q.question_type in ("compositional", "bridge_comparison",
                               "inference", "comparison") and \
                q.source == "2WikiMultiHopQA":
            shape = shape + "\n" + ANSWER_SHAPE["_2wiki_shape"]
        elif q.source == "MuSiQue":
            shape = ANSWER_SHAPE["_musique_shape"]
        async with sem:
            record = {
                "benchmark": benchmark,  # provenance: lets a scorer reload the corpora
                "id": q.id,
                "question": q.question,
                "source": q.source,
                "evidence": q.evidence,
                "question_type": q.question_type,
                "ground_truth": q.answer,
                "ground_truth_aliases": list(q.aliases),
            }
            try:
                record["context"] = await rag.aquery(
                    q.question, param=param_of(shape, only_need_context=True)
                )
                record["generated_answer"] = await rag.aquery(
                    q.question, param=param_of(shape)
                )
            except Exception as exc:
                record.update(
                    context=record.get("context", ""),
                    generated_answer="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            progress()
            return record

    return await asyncio.gather(*(one(q) for q in questions))


async def main(args):
    # A defense is the deployment's, not the attack's, so benign utility has to
    # be measured through the same patched context builder the attack faced.
    # Installed before the first query and left in place for the run.
    #
    # The safe prompt is opt-in here and on by default in attack_defended,
    # deliberately: every utility number already on disk was produced without
    # it, and a flag that silently added it would make those numbers and the
    # new ones the same filename with different meanings.
    if args.defense != "none" or args.safe_prompt:
        import defenses
        defenses.install(args.defense, safe_prompt=args.safe_prompt)

    corpora, questions = load(args.benchmark)
    if args.subset:
        questions = fixed_subset(questions, args.subset)
    elif args.sample:
        questions = stratified(questions, args.sample, args.seed)

    by_source = defaultdict(list)
    for q in questions:
        by_source[q.source].append(q)

    if args.topology == "single":
        groups = [("all", corpora, questions)]
    else:
        groups = [(c.name, [c], by_source[c.name]) for c in corpora if by_source[c.name]]

    def param_of(shape, only_need_context=False):
        return QueryParam(
            mode=args.mode,
            top_k=args.top_k,
            only_need_context=only_need_context,
            response_type=shape,
        )

    slug = args.benchmark.replace(":", "_")
    done, total = [0], len(questions)

    def progress():
        done[0] += 1
        if done[0] % 50 == 0 or done[0] == total:
            print(f"  answered {done[0]}/{total}", flush=True)

    results = []
    for name, group_corpora, group_questions in groups:
        # --index answers against an index that already exists, whatever built
        # it. A probe set has no corpus of its own: the questions are the
        # variable and the graph under them is what changes between runs.
        working_dir = args.index or os.path.join(
            args.workspace, slug, args.topology, name)
        fresh = not os.path.exists(working_dir)
        if fresh and args.base_index:
            # Start from a prebuilt index and add this run's documents to it.
            # The corpus split by <dataset>-dp-base / -delta is what makes
            # this worth doing: the untouched bulk of the corpus is identical
            # under every epsilon -- 95% of HotpotQA, 93% of 2Wiki -- so it is
            # indexed once and copied, and only the sanitised passages are
            # extracted per run. LightRAG merges on
            # insert -- an entity named in both halves stays one node -- and
            # skips documents it has already seen, so copy-then-insert gives
            # the same graph as indexing both halves together.
            import shutil
            shutil.copytree(args.base_index, working_dir)
            print(f"copied base index from {args.base_index}")
        os.makedirs(working_dir, exist_ok=True)
        # The response cache keys on compute_args_hash(model, messages), so a
        # configuration that changes the prompt gets its own entries and the
        # twelve defense cells did not collide. Re-running one of them does:
        # the prompt is byte-identical, every answer is served from the cache,
        # and the "repeat" measures nothing. --no_llm_cache is for that case.
        rag = LightRAG(working_dir=working_dir,
                       llm_model_func=reader(args.llm_model),
                       llm_model_max_async=args.index_concurrency,
                       **({"enable_llm_cache": False} if args.no_llm_cache
                          else {}))

        empty = not any(c.text.strip() for c in group_corpora)
        if fresh and empty:
            raise SystemExit(
                f"{working_dir} has no index and {args.benchmark} carries no "
                "corpus to build one from; point --workspace at a run's index")
        if fresh:
            # a corpus with `parts` is indexed as those parts, so the chunker
            # cannot straddle a boundary the benchmark treats as a unit
            documents = [d for c in group_corpora for d in (c.parts or (c.text,))]
            print(f"indexing {name}: {len(group_corpora)} corpora, "
                  f"{len(documents)} documents "
                  f"({sum(len(c.text) for c in group_corpora) / 1e6:.2f}M chars)")
            await rag.ainsert(documents)
        else:
            print(f"reusing index at {working_dir}")

        print(f"answering {len(group_questions)} questions for {name} "
              f"(mode={args.mode}, top_k={args.top_k})")
        results += await answer_questions(
            rag, group_questions, args.benchmark, param_of, args.concurrency, progress
        )

    os.makedirs(args.out_dir, exist_ok=True)
    # The defense belongs in the name for the same reason the retrieval budget
    # does: it changes the answers, and without it the six defense cells of one
    # dataset would overwrite one file. An undefended run with no safe prompt
    # keeps the original name, so every utility file already on disk stays
    # addressable.
    d = ("" if args.defense == "none" and not args.safe_prompt
         else f"_def-{args.defense}" + ("_safe" if args.safe_prompt else ""))
    out = os.path.join(
        args.out_dir,
        f"predictions_{slug}_{args.mode}_top{args.top_k}{d}.json",
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    errors = sum(1 for r in results if "error" in r)
    print(f"wrote {len(results)} predictions to {out} (errors: {errors})")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True,
                   help="e.g. hotpotqa:dev, 2wiki100-dp-delta:original")
    p.add_argument("--mode", default="hybrid",
                   choices=["naive", "local", "global", "hybrid"])
    p.add_argument("--topology", default="per-source", choices=["per-source", "single"])
    p.add_argument("--top_k", type=int, default=30,
                   help="the paper's appendix H.2 configures 30; their runner "
                        "script defaults to 5 and LightRAG itself to 60")
    p.add_argument("--sample", type=int, help="N questions, sampled proportionally across question types")
    p.add_argument("--seed", type=int, default=0, help="fixes the sampled subset")
    p.add_argument("--llm_model", default="gpt-4o-mini",
                   help="the reader, and the extractor when indexing. "
                        "meta-llama/llama-3.3-70b-instruct is what the "
                        "graph-RAG papers compare against, and on OpenRouter "
                        "it is cheaper than gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=6,
                   help="parallel questions while answering")
    p.add_argument("--subset", metavar="NAME",
                   help="answer the questions datasets/subsets/NAME.json names "
                        "instead of sampling; see benchmarks.fixed_subset")
    p.add_argument("--index_concurrency", type=int, default=16, metavar="N",
                   help="parallel LLM calls while indexing, LightRAG's "
                        "llm_model_max_async. The default is LightRAG's own. "
                        "Lower it when the provider caps in-flight spend "
                        "rather than request rate: OpenRouter answers 402 "
                        "in_flight_budget_exhausted, which tenacity retries "
                        "into the same wall, and the index is left with "
                        "embedded chunks and an empty graph")
    p.add_argument("--workspace", default=os.path.join(ROOT, "workspace"))
    p.add_argument("--base_index", metavar="DIR",
                   help="copy this prebuilt index first, then insert this "
                        "run's documents on top of it; use with "
                        "<dataset>-dp-base to build DIR and "
                        "<dataset>-dp-delta to add a run")
    p.add_argument("--index",
                   help="answer against this LightRAG working directory "
                        "instead of deriving one from --workspace")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--no_llm_cache", action="store_true",
                   help="answer every question with a real call. Needed to "
                        "repeat a configuration already on disk: its prompts "
                        "are byte-identical, so a cached rerun returns the "
                        "first run's answers and measures no variance")
    p.add_argument("--defense", default="none",
                   help="a mode from defenses.MODES, patching LightRAG's "
                        "context construction the way the attack runs saw it")
    p.add_argument("--safe_prompt", action="store_true",
                   help="stack the safe prompt, as Section VII-C-1 of the "
                        "GRASP paper does for its two proposed defenses. Off "
                        "by default, unlike attack_defended.py -- see main()")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
