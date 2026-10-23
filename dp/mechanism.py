"""M4: the metric exponential mechanism over each fact's candidate pool.

    python -m dp.mechanism --epsilon 1 3 5 --seed 0
    python -m dp.mechanism --epsilon 1 --report_only

Each extracted fact (s, r, o) carries a type pair, the type pair carries a
pool, and the relation is resampled from that pool:

    d(r, r') = sqrt( (1 - cos(Emb(r), Emb(r'))) / 2 )
    u(r, c)  = -d(r, c)
    Pr[r~ = c | r] = exp(-eps/2 * d(r, c)) / sum_{c' in P} exp(-eps/2 * d(r, c'))

Two things in that are worth saying out loud, because the previous attempt at
this project got both wrong.

The square root is what makes d a metric. Without it, (1 - cos)/2 is the
squared chord distance and violates the triangle inequality, so the
reverse-triangle step that gives |u(r,c) - u(r',c)| <= d(r,r') does not hold
and the mechanism needs a global sensitivity bound instead. With it, u is
1-Lipschitz in d and the metric calibrates the mechanism by itself -- no
Delta U, which last time was set to a conservative 1.0 against a utility that
only ever spanned 0.379, and threw away a factor of 2.6 in epsilon for
nothing.

The private relation need not be in the pool. The pool is generated from the
type pair alone and the extracted predicate comes from the document, so they
share a vocabulary only by accident. It does not matter for calibration: r
enters only through d(r, c), and `located_in` has a distance to `is_within`
whether or not anyone wrote both into the same list.

It does matter for accounting. The mechanism can return r itself whenever r
is in its pool, and the vocabularies overlap: on 2Wiki, 106 of 969 extracted
triples (10.9%) have their own relation sitting in their candidate pool, at
distance zero. Being the nearest
candidate, r is then also the most likely one -- so 42 of those 969 were
released unchanged at epsilon=10, falling to 24, 15 and 9 at epsilon 5, 1 and
0.1 as the preference for near candidates weakens.

Nothing about the guarantee breaks: sampling proportional to exp(-eps*d/2) is
exactly what Theorem 1 covers, and d(r, r) = 0 is a legitimate draw. But those
triples reach the rewritten text with their true relation intact, so an attack
recovers them without inferring anything, and reporting that as a shortfall in
protection would be wrong -- they were never perturbed. Retention is therefore
worth measuring after all, alongside d(r, r~), which the trace records.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import statistics
import sys

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

from dp.paths import under  # noqa: E402

from dp.relation_pool import load_pools  # noqa: E402
from dp.verbalize import load_sentences, text_of  # noqa: E402
from dp.typemap import extraction, load_map, normalise  # noqa: E402

EMBEDDINGS = under("relation_embeddings.json")
DEFAULT_EMBEDDER = "sentence-transformers/nli-roberta-base-v2"
"""What the distance is measured in.

Not text-embedding-3-small, which is what this started with and what the
numbers below are against. A retrieval embedding asks whether two texts are
about the same thing, and "A person is expelled from an organization" and
"A person merges with an organization" are both about a person and an
organization, so it places them close together -- which is the one judgement
the mechanism needs it not to make. Measured over the pools, the distance
from a private relation to its ten candidates spanned 0.102, so at epsilon=1
the nearest was 5% likelier than the furthest and sampling was a coin toss
with ten sides.

An NLI-trained encoder is optimised on entailment against contradiction, so
separating those two sentences is its job. The same measurement gives 0.176,
a 72% wider spread, and the nearest candidate moves in from 0.377 to 0.327 --
closer, not merely rescaled, which rules out its being a change of units.
Truncating the OpenAI vectors helped in the same direction and far less
(0.120 at 128 dimensions), which is consistent with most of those dimensions
carrying topic rather than relation.

It is also local, free and deterministic, where every embedding in this
project so far has come from an API that does not repeat itself.
"""
OUT_DIR = under("perturbed")


def facts(include_true: bool = False) -> list[dict]:
    """Every extracted fact with its type pair and the pool that pair implies.

    This is the "map the type pair back to the entity pair" step: the pool was
    generated per type pair, and here each entity pair collects the pool its
    own types point at.

    `include_true` appends the fact's own relation to its pool, which changes
    the mechanism and changes what it guarantees. It is worth having because
    it changes the arithmetic completely -- d(r, r) = 0 against candidates at
    0.45, a gap 4.4 times the 0.10 that separates the candidates from each
    other, so epsilon finally has something to act on -- and worth defaulting
    to off, because Theorem 1 requires the pool to be independent of the
    private relation and this makes it the one thing it must not be. An
    adversary who sees the support learns that r was in it.
    """
    mapping, pools = load_map(), load_pools()
    out = []
    for record in extraction()["records"]:
        entities = record["entities"]
        title = record["document"].split("\n", 1)[0]
        for triple in record["triples"]:
            s, o = triple["subject"], triple["object"]
            if s not in entities or o not in entities:
                continue
            a = mapping.get(normalise(entities[s]))
            b = mapping.get(normalise(entities[o]))
            if not (a and b):
                continue
            pool = list(pools.get((a, b), []))
            if not pool:
                continue
            if include_true and triple["relation"] not in pool:
                pool.append(triple["relation"])
            out.append({"document": title, "subject": s, "object": o,
                        "relation": triple["relation"], "pair": [a, b],
                        "pool": pool, "occurrences": triple["occurrences"]})
    return out


async def _embed_openai(texts: list[str], model: str) -> dict[str, list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    vectors: dict[str, list[float]] = {}
    for i in range(0, len(texts), 512):
        batch = texts[i:i + 512]
        reply = await client.embeddings.create(model=model, input=batch,
                                               encoding_format="float")
        vectors.update({t: d.embedding for t, d in zip(batch, reply.data)})
    return vectors


def _embed_local(texts: list[str], model: str) -> dict[str, list[float]]:
    from sentence_transformers import SentenceTransformer

    # DP_EMBEDDER_REVISION pins the Hub revision, so the vectors -- and with
    # them every draw the mechanism makes -- are the ones the paper's traces
    # were made with. config.py carries the value; main.py exports it.
    revision = os.environ.get("DP_EMBEDDER_REVISION", "").strip() or None
    encoder = SentenceTransformer(model, revision=revision)
    matrix = encoder.encode(texts, batch_size=64, show_progress_bar=False)
    return {t: matrix[i].tolist() for i, t in enumerate(texts)}


def _embed(texts: list[str], model: str) -> dict[str, list[float]]:
    if "/" in model:
        return _embed_local(texts, model)
    return asyncio.run(_embed_openai(texts, model))


def embeddings(texts: set[str], model: str) -> dict[str, np.ndarray]:
    """Unit-norm embedding per relation string, cached by model and text set.

    The cache key covers every string, so adding a pool or re-extracting
    invalidates it rather than silently scoring new relations against an old
    space.
    """
    ordered = sorted(texts)
    key = hashlib.sha256(
        (model + "|" + "|".join(ordered)).encode("utf-8")).hexdigest()[:16]
    if os.path.exists(EMBEDDINGS):
        cached = json.load(open(EMBEDDINGS, encoding="utf-8"))
        if cached.get("key") == key:
            return {t: np.array(v) for t, v in cached["vectors"].items()}

    print(f"embedding {len(ordered)} relation strings with {model}")
    raw = _embed(ordered, model)
    vectors = {}
    for text, vector in raw.items():
        v = np.array(vector, dtype=np.float64)
        vectors[text] = v / np.linalg.norm(v)
    os.makedirs(os.path.dirname(EMBEDDINGS), exist_ok=True)
    json.dump({"model": model, "key": key,
               "vectors": {t: v.tolist() for t, v in vectors.items()}},
              open(EMBEDDINGS, "w", encoding="utf-8"))
    return vectors


def whiten(vectors: dict[str, np.ndarray], fit_on: set[str],
            k: int) -> dict[str, np.ndarray]:
    """Re-express the space so its directions carry comparable variance.

    The encoder does not spread relation sentences over the sphere; it packs
    them into a thin slab offset from the origin. Measured over the 620
    candidate sentences, the mean vector has norm 0.6056, so 36.7% of every
    vector is a component all of them share -- and for unit vectors the mean
    pairwise cosine IS that shared norm squared, which is why no pair of
    relation sentences ever came out negative while unrelated sentences did
    17.9% of the time. The slab is thin as well as offset: the top ten of 768
    directions hold 63.6% of the remaining variance.

        v' = (v - mu) Sigma^{-1/2},   then renormalised

    Both mu and Sigma are fitted on `fit_on` -- the candidate sentences and
    nothing else. A pool therefore still depends on (subject type, object
    type) alone, and no private relation enters the transform, so Theorem 1 is
    untouched. The result is a linear map followed by projection to the unit
    sphere, and chord distance there is a metric, so u = -d stays 1-Lipschitz.
    Verified rather than assumed: 0 of 44640 triples violate the triangle
    inequality at k=64.

    `k` truncates to the leading k directions, and it is not optional.
    Dividing by sqrt(lambda) in every direction amplifies the smallest by
    1/sqrt(lambda_min), and here lambda_768 is 1/42026 of lambda_1 -- those
    directions carry no relation signal, and inflating them drowns the ones
    that do. Whitening all 768 made candidates mutually orthogonal, a median
    pairwise distance of 0.7077, while collapsing the only quantity the
    mechanism reads -- the spread of d(r, .) across a pool -- from 0.1651 to
    0.0084. k=64 keeps 90.8% of the variance and takes the spread to 0.2308.
    """
    names = sorted(fit_on)
    X = np.stack([vectors[n] for n in names])
    mu = X.mean(axis=0)
    w, U = np.linalg.eigh(np.cov((X - mu).T))
    order = np.argsort(w)[::-1]
    w, U = w[order][:k], U[:, order][:, :k]
    W = U / np.sqrt(np.maximum(w, 1e-8))

    out = {}
    for text, vector in vectors.items():
        v = (vector - mu) @ W
        norm = np.linalg.norm(v)
        # a vector landing on the new origin has no direction left to compare;
        # keeping it unnormalised would make its cosine meaningless, so it is
        # left at zero and distance() reads it as maximally far
        out[text] = v / norm if norm > 0 else v
    return out


def distance(a: str, b: str, vectors: dict[str, np.ndarray]) -> float:
    """sqrt((1 - cos)/2), in [0, 1]. A metric, unlike the same thing unrooted.

    The cosine is clamped before the subtraction: floating point returns
    1 + 1e-16 for a vector against itself, and a negative radicand is a
    NaN rather than a zero.
    """
    if a == b:
        return 0.0
    cos = float(np.dot(vectors[a], vectors[b]))
    return float(np.sqrt((1.0 - min(1.0, max(-1.0, cos))) / 2.0))


NULL = None
"""Withholding the relation, the output the pool does not contain.

The sampling space is the pool plus this one extra outcome, and its weight is
exp(-eps/2) -- exactly the weight a candidate at distance 1 would receive. That
is the same as extending the metric with d(r, null) = 1 for every r, and the
extension is still a metric, which is the whole reason it can be done this way:

    d(r, null) <= d(r, c) + d(c, null)   is   1 <= d(r,c) + 1      true
    d(r1, r2)  <= d(r1, null) + d(null, r2) = 2   true since d <= 1

So u = -d stays 1-Lipschitz, no global sensitivity bound is needed, and Theorem
1's argument carries over unchanged. The weight also does not depend on r, so the
support is still a function of the type pair alone.

Because every measured d(r, c) is well under 1 -- the median candidate sits near
0.5 -- null is always the least likely outcome, and its share falls as epsilon
rises: about 9% at epsilon=0 with a pool of ten, 7% at 1, 3% at 5, under 1% at
10. Small epsilon therefore means the relation is more often withheld outright
rather than exchanged for a near neighbour, which is the behaviour a stronger
privacy setting should have.
"""


NULL_DISTANCE = 1.0
"""Where null sits, as a parameter rather than the metric's upper bound.

d = sqrt((1-cos)/2) has range [0, 1] and 1 is attained only at cos = -1, so the
default puts null at the furthest point the metric admits -- further than any
real candidate, which over 9638 measured (r, c) pairs never exceeds 0.7116. It
is also the end of the range that makes null *least* likely, and null is the
only outcome that removes a sentence, so the default minimises the one
operation that protects entity pairs.

Lowering it raises the deletion rate, but only within a bound: the triangle
inequality d(r1, r2) <= d(r1, null) + d(null, r2) = 2*lambda needs
lambda >= D_max/2, and D_max measured over these pools is 0.7140, so
lambda >= 0.357. Below that the extension is no longer a metric, u = -d is no
longer 1-Lipschitz, and Theorem 1's argument does not carry over -- the run
still executes and still reports, but what it reports is not
epsilon-metric-DP. Such a run prints a warning and tags its filename.
"""


def probabilities(relation: str, pool: list[str], epsilon: float,
                  vectors: dict[str, np.ndarray],
                  with_null: bool = True,
                  null_distance: float = NULL_DISTANCE) -> np.ndarray:
    """exp(-eps/2 * d) over the pool and, unless ablated, over null.

    The last entry of the returned array is null's; the first len(pool) are the
    candidates', in order. With with_null=False the null entry is absent and the
    array has len(pool) entries -- the ablation that asks how much of the
    structural protection comes from deleting sentences rather than from
    replacing predicates.

    Dropping null renormalises the candidates but does not reshape their
    distribution: at epsilon=1 null carries exp(-0.5)=0.607 against about ten
    candidates at roughly exp(-0.5*0.4)=0.82 each, so it holds about 7% of the
    mass and removing it spreads that 7% proportionally. The replacement a fact
    receives is therefore drawn from the same shape either way, which is what
    makes the ablation attribute cleanly to deletion.

    Exponents are shifted by their maximum before exponentiating: the standard
    guard against overflow, and because the shift cancels in the normalisation it
    leaves the distribution exactly unchanged -- which matters here, where a
    quietly altered distribution would be an unnoticed privacy claim.
    """
    d = np.array([distance(relation, c, vectors) for c in pool]
                 + ([null_distance] if with_null else []))
    exponents = -epsilon * d / 2.0
    weights = np.exp(exponents - exponents.max())
    return weights / weights.sum()


def perturb(epsilon: float, seed: int, model: str,
            include_true: bool = False, whiten_k: int = 0,
            with_null: bool = True,
            null_distance: float = NULL_DISTANCE) -> dict:
    """Sample a released relation per fact.

    Distances are taken between verbalised relations -- "A person works for an
    organization." rather than `works_for` -- because the bare predicates are
    all about equally far from each other and the mechanism had nothing to
    distinguish. See dp.verbalize.
    """
    rows = facts(include_true)
    sentences = load_sentences()
    say = lambda f, r: text_of(r, f["pair"][0], f["pair"][1], sentences)
    strings = {say(f, r) for f in rows for r in [f["relation"]] + f["pool"]}
    vectors = embeddings(strings, model)
    if whiten_k:
        # fitted on the candidates only, never on the private relations
        candidates = {say(f, c) for f in rows for c in f["pool"]}
        vectors = whiten(vectors, candidates, whiten_k)
        print(f"whitened to {whiten_k} dims, fitted on "
              f"{len(candidates)} candidate sentences")
    rng = random.Random(seed)

    trace = []
    for fact in rows:
        said = [say(fact, c) for c in fact["pool"]]
        p = probabilities(say(fact, fact["relation"]), said, epsilon, vectors,
                          with_null, null_distance)
        # one draw per triple, not per occurrence: every mention of a triple
        # carries the same outcome, so a reader cannot average several
        # independent draws of the same fact back towards the truth
        index = rng.choices(range(len(p)), weights=p)[0]
        withheld = with_null and index == len(fact["pool"])
        released = NULL if withheld else fact["pool"][index]
        d = [distance(say(fact, fact["relation"]), s, vectors) for s in said]
        trace.append({
            **{k: fact[k] for k in ("document", "subject", "object",
                                    "relation", "pair", "occurrences")},
            "released": released,
            "distance": null_distance if withheld else round(d[index], 4),
            "probability": round(float(p[index]), 6),
            "nearest": fact["pool"][int(np.argmin(d))],
            "spread": round(max(d) - min(d), 4),
            "pool_size": len(fact["pool"]),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"eps{epsilon:g}_seed{seed}"
                                + (f"_wh{whiten_k}" if whiten_k else "")
                                + ("_withtrue" if include_true else "")
                                + ("" if with_null else "_nonull")
                                + ("" if null_distance == 1.0
                                   else f"_lam{null_distance:g}") + ".json")
    json.dump({"epsilon": epsilon, "seed": seed, "model": model,
               "include_true": include_true, "whiten_k": whiten_k,
               "with_null": with_null,
               "null_distance": null_distance,
               "metric_valid": null_distance >= 0.357,
               "facts": len(trace), "trace": trace},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return {"path": out, "trace": trace}


def report(trace: list[dict], epsilon: float) -> None:
    withheld = [t for t in trace if t["released"] is NULL]
    kept = [t for t in trace if t["released"] is not NULL]
    print(f"\n  withheld (null): {len(withheld)}/{len(trace)} "
          f"({100 * len(withheld) / len(trace):.1f}%)  -- every sentence holding "
          f"one of these is deleted, see dp.inplace")
    if not kept:
        return
    trace = kept
    d = [t["distance"] for t in trace]
    nearest = sum(1 for t in trace if t["released"] == t["nearest"])
    mass = [t["probability"] for t in trace]
    print(f"\neps={epsilon:g}  {len(trace)} facts")
    print(f"  d(r, r~):            mean {statistics.mean(d):.4f}  "
          f"median {statistics.median(d):.4f}  max {max(d):.4f}")
    print(f"  released the nearest candidate: {nearest}/{len(trace)} "
          f"({100 * nearest / len(trace):.1f}%)")
    print(f"  probability of what was released: mean "
          f"{statistics.mean(mass):.4f}  (uniform would be "
          f"{statistics.mean(1 / t['pool_size'] for t in trace):.4f})")
    spread = [t["spread"] for t in trace]
    print(f"  pool spread max(d)-min(d): mean {statistics.mean(spread):.4f}  "
          f"median {statistics.median(spread):.4f}")
    print(f"  implied best/worst odds at this eps: "
          f"{statistics.median([pow(2.718281828, epsilon * s / 2) for s in spread]):.3f}")
    retained = sum(1 for t in trace if t["released"] == t["relation"])
    print(f"  released the true relation: {retained}/{len(trace)} "
          f"({100 * retained / len(trace):.1f}%)")
    common = collections.Counter(t["released"] for t in trace)
    print(f"  distinct released relations: {len(common)}; "
          f"most common {common.most_common(5)}")


def main(args) -> None:
    if args.null_distance < 0.357:
        print(f"!! null_distance={args.null_distance:g} < 0.357: the metric "
              f"extension's triangle inequality fails (D_max=0.714), so this "
              f"run is NOT epsilon-metric-DP. Measuring only.")
    print(f"{len(facts(args.include_true))} facts have a pool"
          + (" (true relation added)" if args.include_true else ""))
    for epsilon in args.epsilon:
        result = perturb(epsilon, args.seed, args.model, args.include_true,
                         args.whiten, not args.no_null, args.null_distance)
        report(result["trace"], epsilon)
        print(f"  -> {result['path']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epsilon", type=float, nargs="+", default=[1.0],
                   help="one run per value")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_null", action="store_true",
                   help="ablation: drop null from the output space, so every "
                        "fact is replaced and no sentence is deleted. Writes "
                        "to ..._nonull.json so it cannot overwrite a main run.")
    p.add_argument("--model", default=DEFAULT_EMBEDDER,
                   help="a sentence-transformers id (local) or an OpenAI "
                        "embedding model name")
    p.add_argument("--whiten", type=int, default=0, metavar="K",
                   help="whiten the embedding space to K dimensions, fitted "
                        "on the candidate sentences; 0 disables. 64 keeps "
                        "90.8%% of the variance, see whiten()")
    p.add_argument("--include_true", action="store_true",
                   help="add each fact's own relation to its pool; breaks the "
                        "independence Theorem 1 assumes, see facts()")
    p.add_argument("--null_distance", type=float, default=NULL_DISTANCE,
                   metavar="LAMBDA",
                   help="where null sits in the metric; 1.0 (the default, and "
                        "the metric's upper bound) makes deletion least "
                        "likely, lower raises it. Below 0.357 the extension "
                        "is not a metric and the DP argument lapses -- see "
                        "NULL_DISTANCE. Anything but 1.0 writes to "
                        "..._lam<LAMBDA>.json")
    main(p.parse_args())
