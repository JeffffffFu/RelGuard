"""Answer a benchmark with PathRAG, writing predictions our scorers can read.

    python baselines/run_pathrag.py --benchmark hotpotqa:dev \
        --subset hotpotqa_min100 --passages_from_questions \
        --workspace workspace/pathrag_hp100 --out_dir runs/_hp100
    python benchmark/score_hotpot.py runs/_hp100/predictions_*pathrag*.json

PathRAG (arXiv, BUPT-GAMMA) is a fork of LightRAG: same module layout, same
chunking, same entity extraction prompt, and `llm.py` still refers to
`LightRAG(`. What it replaces is retrieval. LightRAG hands the reader a flat
table of the top-k edges; PathRAG enumerates every path of up to three hops
between the retrieved entity nodes (`find_paths_and_edges_with_stats`), gives
each a flow-decayed weight (`bfs_weighted_paths`, with a threshold and an alpha),
and hands over the highest-weighted paths. The reader therefore sees reasoning
chains rather than disconnected edges, which is the method's claim.

Two upstream facts this works around rather than fixes:

    QueryParam.mode is annotated Literal["hybrid"] but defaults to "global",
    and aquery raises on anything but "hybrid". So mode is set explicitly.

    PathRAG.query is `await loop.run_until_complete(await self.aquery(...))`,
    which awaits a coroutine twice and awaits a non-awaitable. Only aquery
    works, so only aquery is used -- the same call run_bench.py makes of
    LightRAG.

The questions come from benchmark/benchmarks.py and the grading happens in
benchmark/score_hotpot.py, so PathRAG, LightRAG and HippoRAG 2 answer one list
and are graded by one piece of code. The answer-shape prompt is the same one
LightRAG gets, passed through `response_type` exactly as run_bench.py passes it:
PathRAG inherits that parameter from LightRAG, so withholding it would compare
a constrained reader against an unconstrained one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, HERE)                              # baselines/PathRAG
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from openai import (APIConnectionError, APIStatusError, APITimeoutError,  # noqa: E402
                    RateLimitError)
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,  # noqa: E402
                      wait_exponential)

from benchmarks import fixed_subset, load, stratified  # noqa: E402
from run_bench import ANSWER_SHAPE, DEFAULT_SHAPE, reader  # noqa: E402


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


def needed_by(questions, benchmark):
    """Only the passages these questions name, gold and distractor alike.

    Read from the dataset directory rather than filtered out of Corpus.parts.
    `pack()` concatenates passages into ~1,100-token windows, so a window's
    first line is the first passage's title and every other passage in it is
    invisible to a title test. Filtering parts that way took 908 passages down
    to 119 windows -- 13% of the corpus indexed, and gold recall 56/100 against
    LightRAG's 95/100, which read as PathRAG retrieving badly when it had never
    been given the text.
    """
    name, _, subset = benchmark.partition(":")
    directory = {"hotpotqa": "hotpotqa", "2wiki": "2wiki",
                 "musique": "musique"}.get(name)
    if directory is None:
        raise SystemExit(f"--passages_from_questions does not know {name!r}")
    root = os.path.join(ROOT, "datasets", directory)
    rows = json.load(open(os.path.join(root, f"{subset}_questions.json"),
                          encoding="utf-8"))
    wanted = {q.id for q in questions}
    titles = {title for row in rows if row["_id"] in wanted
              for title, _ in row["context"]}
    passages = json.load(open(os.path.join(root, f"{subset}.json"),
                              encoding="utf-8"))
    rewritten = sanitized_passages(SANITIZED[0])
    return [rewritten.get(p["title"], f"{p['title']}\n{p['text']}")
            for p in passages if p["title"] in titles]


def embedder(model: str):
    """The same embedding endpoint LightRAG reaches, wrapped as PathRAG wants it.

    PathRAG's own `openai_embedding` hard-codes base_url="https://api.openai.com/v1",
    where LightRAG's defaults it to None and so lets AsyncOpenAI read
    OPENAI_BASE_URL. With the project's key that is a 401 before a single
    document is indexed. Passing the function in leaves the vendored file alone
    and keeps both methods on one endpoint and one embedding model, which is what
    makes their retrieval comparable.

    The retry is not optional, and it was missing at first. Both vendored
    libraries wrap their own embedding call in tenacity; a hand-written
    replacement that does not is a single timeout away from losing an index. One
    did: 2Wiki's PathRAG run died on `APITimeoutError` after the whole corpus had
    been extracted, and because the driver piped stdout through grep and ended
    the line with `|| true`, the traceback was filtered out and the exit code
    discarded -- the run looked like it had simply produced no predictions.
    """
    import numpy as np
    from PathRAG.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
    @retry(stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=4, max=120),
           retry=retry_if_exception_type(
               (APITimeoutError, APIConnectionError, RateLimitError,
                APIStatusError)))
    async def embed(texts: list[str]) -> "np.ndarray":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(timeout=120.0)   # reads OPENAI_BASE_URL and the key
        reply = await client.embeddings.create(model=model, input=texts,
                                               encoding_format="float")
        return np.array([d.embedding for d in reply.data])

    return embed


def shape_for(q):
    shape = ANSWER_SHAPE.get(q.question_type, DEFAULT_SHAPE)
    if q.question_type in ("compositional", "bridge_comparison", "inference",
                           "comparison") and q.source == "2WikiMultiHopQA":
        shape = shape + "\n" + ANSWER_SHAPE["_2wiki_shape"]
    elif q.source == "MuSiQue":
        shape = ANSWER_SHAPE["_musique_shape"]
    return shape


class _NoCache:
    """PathRAG's response cache, with the memory removed. See `run`."""

    def __init__(self, inner):
        self.inner = inner

    async def get_by_id(self, _id):
        return None

    async def upsert(self, _data):
        return None

    async def index_done_callback(self):
        return None

    def __getattr__(self, name):
        return getattr(self.inner, name)


async def run(args) -> None:
    from PathRAG import PathRAG, QueryParam

    # The defense is installed after PathRAG is imported and before the first
    # query, the same order attack_defended.py uses: it patches two names
    # inside the loaded `PathRAG` package (operate.list_of_list_to_csv and
    # prompt.PROMPTS["rag_response"]) and nothing under baselines/PathRAG/ is
    # edited. The safe prompt is opt-in here and on by default in the attack
    # driver, for the reason run_bench.py gives: every utility file already on
    # disk was produced without it.
    if args.defense != "none" or args.safe_prompt:
        sys.path.insert(0, os.path.join(ROOT, "benchmark"))
        import defenses
        defenses.install(args.defense, seed=args.seed,
                         safe_prompt=args.safe_prompt, victim="pathrag")

    corpora, questions = load(args.benchmark)
    if args.subset:
        questions = fixed_subset(questions, args.subset)
    elif args.sample:
        questions = stratified(questions, args.sample, args.seed)
    documents = (needed_by(questions, args.benchmark)
                 if args.passages_from_questions
                 else [d for c in corpora for d in (c.parts or (c.text,))])
    print(f"{len(questions)} questions, {len(documents)} documents")

    fresh = not os.path.exists(args.workspace)
    os.makedirs(args.workspace, exist_ok=True)
    # The response cache is off because PathRAG's key is unsound for this use.
    # kg_query computes `compute_args_hash(query_param.mode, query)` and returns
    # the cached value before it looks at `only_need_context`, so the key covers
    # neither top_k, nor response_type, nor whether the caller asked for context
    # or for an answer. The first consequence is a silent one: a second run at a
    # different top_k returned byte-identical answers for all 100 questions and
    # a "context" field holding the previous run's answer -- `Kirk Humphreys`,
    # `I don't know.` -- so the configuration under test never reached retrieval.
    # `enable_llm_cache=False` cannot be used, and neither can a None cache:
    # both send PathRAG into a storage constructor it calls with two arguments
    # missing (PathRAG.py:245, and again per query in aquery), so the run
    # raises before or at the first question. The cache object is kept and
    # wrapped instead: every lookup misses and every write is discarded, while
    # `global_config` and the rest pass through untouched, which is all
    # handle_cache and save_to_cache read. Nothing under baselines/PathRAG/ is
    # edited, and the cache file on disk is neither read nor grown.
    rag = PathRAG(working_dir=args.workspace,
                  llm_model_func=reader(args.llm_model),
                  embedding_func=embedder(args.embedding_model),
                  llm_model_max_async=args.index_concurrency)
    if args.no_cache:
        rag.llm_response_cache = _NoCache(rag.llm_response_cache)
        print("response cache off")
    if fresh:
        print("indexing")
        await rag.ainsert(documents)

    sem = asyncio.Semaphore(args.concurrency)
    done = [0]

    async def one(q):
        shape = shape_for(q)
        record = {"benchmark": args.benchmark, "id": q.id,
                  "question": q.question, "source": q.source,
                  "evidence": q.evidence, "question_type": q.question_type,
                  "ground_truth": q.answer,
                  "ground_truth_aliases": list(q.aliases)}
        async with sem:
            try:
                record["context"] = await rag.aquery(
                    q.question, param=QueryParam(
                        mode="hybrid", top_k=args.top_k, response_type=shape,
                        only_need_context=True))
                record["generated_answer"] = await rag.aquery(
                    q.question, param=QueryParam(
                        mode="hybrid", top_k=args.top_k, response_type=shape))
            except Exception as exc:
                record.update(context=record.get("context", ""),
                              generated_answer="",
                              error=f"{type(exc).__name__}: {exc}")
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(questions):
                print(f"  answered {done[0]}/{len(questions)}", flush=True)
            return record

    records = await asyncio.gather(*(one(q) for q in questions))

    os.makedirs(args.out_dir, exist_ok=True)
    slug = args.benchmark.replace(":", "_")
    # an undefended run keeps its original name, so every file on disk stays
    # addressable; a defended one carries the setting, as run_bench.py does
    d = ("" if args.defense == "none" and not args.safe_prompt
         else f"_def-{args.defense}" + ("_safe" if args.safe_prompt else ""))
    out = os.path.join(args.out_dir,
                       f"predictions_{slug}_pathrag_top{args.top_k}{d}.json")
    if args.defense != "none" or args.safe_prompt:
        import defenses
        fired = defenses.applied()
        print(f"defense patches fired: {dict(fired) or 'NONE'}")
        if args.defense not in ("none",) and not any(
                k.startswith("PathRAG:") for k in fired):
            raise SystemExit("the defense never reached PathRAG's context: "
                             "refusing to write a file named after it")
    json.dump(records, open(out, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    errors = sum(1 for r in records if r.get("error"))
    print(f"wrote {len(records)} predictions to {out} (errors: {errors})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="hotpotqa:dev")
    p.add_argument("--subset", metavar="NAME",
                   help="answer the questions datasets/subsets/NAME.json names, "
                        "so every method answers the same list")
    p.add_argument("--sample", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--passages_from_questions", action="store_true",
                   help="index only the passages the questions name")
    p.add_argument("--sanitized", metavar="PATH",
                   help="a dp.inplace corpus; its passages replace the "
                        "originals, which is what makes an epsilon run an "
                        "epsilon run")
    p.add_argument("--llm_model", default="gpt-4o-mini")
    p.add_argument("--embedding_model", default="text-embedding-3-small",
                   help="the same model LightRAG and HippoRAG 2 retrieve with")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--no_cache", action="store_true",
                   help="disable PathRAG's response cache, whose key covers "
                        "neither top_k nor only_need_context")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--index_concurrency", type=int, default=12)
    p.add_argument("--workspace",
                   default=os.path.join(ROOT, "workspace", "pathrag"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "_pathrag"))
    p.add_argument("--defense", default="none",
                   help="a defenses.MODES entry, applied to PathRAG's own "
                        "context tables and response prompt at runtime; see "
                        "benchmark/defenses.py")
    p.add_argument("--safe_prompt", action="store_true",
                   help="append the safe prompt as well, as the attack driver "
                        "does by default")
    _args = p.parse_args()
    SANITIZED[0] = _args.sanitized
    asyncio.run(run(_args))
