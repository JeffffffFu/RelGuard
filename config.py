"""Every knob of a run, in one place. main.py reads this and nothing else does.

Change a value here, or override it for one invocation:

    python main.py perturb --set EPSILONS=[1,5] --set SEED=1

Each stage's own module keeps its CLI and its defaults; this file is the
experiment's view of them, so a sweep is described once rather than retyped
per command. Values that are a property of the corpus (scope, subset, target
list) and values that are a property of the method (epsilon, the embedder, the
pool size) are kept apart below because they are changed for different reasons.
"""

from __future__ import annotations

import sys

# ------------------------------------------------------------------ corpus

SCOPE = "2wiki100"
"""Which corpus every artefact is keyed on: workspace/dp_<SCOPE>/. The two the
tables report are 2wiki100 and hotpotqa100 -- the hundred-question subsets
every method answers."""

CORPORA = {
    # scope: (benchmark spec for the baselines, --subset name, dp-delta prefix)
    "2wiki100": ("2wiki:dev", "2wiki_min100", "2wiki100-dp-delta"),
    "hotpotqa100": ("hotpotqa:dev", "hotpotqa_min100", "hotpotqa100-dp-delta"),
}

# ------------------------------------------------------------------ models

EXTRACTOR = "gpt-4o-mini"            # dp.extract, dp.typemap, dp.spans, dp.verbalize
POOL_GENERATOR = "openai/gpt-5-mini"  # dp.relation_pool; see its note on why not the extractor
EMBEDDER = "sentence-transformers/nli-roberta-base-v2"   # dp.mechanism's distance
EMBEDDER_REVISION = "f83f701a097e6fa75b956a3e601dde649ea818cc"
"""The Hub revision of EMBEDDER the reported traces were made with. The
mechanism's draws are a deterministic function of the vectors, so a different
revision is a different mechanism."""
READER = "gpt-4o-mini"               # the victim's reader, attack side and utility side
ATTACKER = "gpt-4o-mini"             # the attacker's own LLM and the typed judge
INDEX_MODEL = "gpt-4o-mini"          # what every index was built with; HippoRAG keys its store on it

# ------------------------------------------------------------------ mechanism

EPSILONS = [0.1, 1.0, 5.0, 10.0]
SEED = 0
POOL_SIZE = 10          # candidate relations per ordered type pair
NULL_DISTANCE = 1.0     # where null sits; >= 0.357 keeps the metric a metric
WITH_NULL = True        # False is the ablation that never deletes a sentence
WHITEN = 0              # PCA components removed from the embeddings; 0 is off
POLISH = None           # a model to smooth the rewritten sentence, or None

# ------------------------------------------------------------------ victims

VICTIMS = ["lightrag", "pathrag", "hipporag"]
SHORT = {"lightrag": "lr", "pathrag": "pr", "hipporag": "hp"}
TOP_K = {"lightrag": 10, "pathrag": 10, "hipporag": 5}
MODE = "hybrid"         # LightRAG's query mode; PathRAG only has hybrid

# ------------------------------------------------------------------ attacks

GRASP_QUERIES = 5       # per target
GRAPHSTEAL = dict(turns=75, command="predicate", order="degree", top_k=40,
                  gold="trace")
TARGETS = "datasets/subsets/targets_{scope}.json"   # the fifteen GRASP targets

# ------------------------------------------------------------------ defenses

DEFENSES = ["none", "abstractive", "extractive", "id_align", "decoy", "protective"]
SAFE_PROMPT_WITH = {"abstractive", "extractive", "id_align", "decoy"}
"""Which defenses stack the safe prompt, per Section VII-C-1 of Song et al.;
`none` and `protective` run without it."""
ROUNDS = 3

# ------------------------------------------------------------------ plumbing

CONCURRENCY = 8
PYTHON = sys.executable     # the interpreter main.py itself runs under
