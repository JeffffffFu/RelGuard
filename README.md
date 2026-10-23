# RelGuard

Differential privacy for the relation semantics of a GraphRAG knowledge graph.
Each extracted fact (subject, relation, object) carries a type pair; the type
pair carries a candidate pool generated from the types alone; the relation is
resampled from that pool by a metric exponential mechanism, or withheld, which
deletes the sentence. The released relation is written back in place of the
predicate's own words, the rewritten corpus is indexed by an unmodified GraphRAG
system, and the result is measured from both sides: what reconstruction attacks
recover from the deployment (privacy), and what a QA benchmark loses (utility).
The GRASP paper's defenses are included as baselines on the same victims.

## Installation

```bash
git clone --recursive https://github.com/JeffffffFu/RelGuard.git
cd RelGuard
conda env create -f environment.yml     # creates `relguard`: Python 3.10, every version pinned
conda activate relguard
cp .env.example .env                    # fill in OPENAI_API_KEY; OPENAI_BASE_URL for an OpenAI-compatible gateway
```

`environment.yml` is a full `conda env export`, so the versions are the ones
the reported numbers were produced with. Two things worth knowing:

- **No GPU is needed.** The only local model is the relation encoder
  (`sentence-transformers/nli-roberta-base-v2`), run with the CPU build of
  torch, which the file pulls from PyTorch's own index. Everything else is an
  API call through the `openai` client.
- **The three victims are not installed, they are imported from the
  repository**: the patched LightRAG from `LightRAG-v1.0.3/`, PathRAG from
  `baselines/PathRAG/`, HippoRAG 2 from the `baselines/HippoRAG/` submodule.
  If you cloned without `--recursive`, run `git submodule update --init`.

**Note:** run every command from the repository root with the `relguard`
environment active. `main.py` launches each stage with the interpreter it
runs under.

## Datasets

### Fetched automatically

```bash
python scripts/fetch_data.py
```

| What | From | Lands in |
|---|---|---|
| HotpotQA, 1,000-question sample and its 9,811 passages | HippoRAG 2's reproduction set (the submodule) | `datasets/hotpotqa/` |
| 2WikiMultiHopQA, 1,000 questions and 6,119 passages | the same | `datasets/2wiki/` |
| the hundred-question subsets and the fifteen attack targets | `assets/subsets/` | `datasets/subsets/` |
| the per-corpus artefacts of the method: extraction, candidate pools, verbalisations, predicate spans, frozen gold filter | `assets/<scope>/` | `workspace/dp_<scope>/` |

The artefacts under `assets/` were produced by model calls and would not come
back identical if regenerated; shipping them is what makes a run here
comparable with the reported numbers. `extract` and `pools` (below) rebuild
them from scratch for a new corpus.

### Optional

```bash
python scripts/fetch_data.py --trex     # KILT T-REx parquet from the Hugging Face hub, ~550 MB
```

Only needed to rebuild the entity-type inventory (`dp.entity_types --build`,
`dp.signature --build`); the files those produce are already in `assets/kilt/`.

## Code structure

- `main.py` -- the entry point: `ours`, `baselines`, `results`, and the single stages.
- `config.py` -- every parameter of a run; `--set KEY=VALUE` overrides one.
- `scripts/fetch_data.py` -- puts the datasets and the shipped artefacts in place.
- `assets/` -- the artefacts a run must share with the reported numbers (see Datasets).
- `dp/` -- the mechanism, in pipeline order:
  `extract.py` (entities with types, triples with where they are said),
  `entity_types.py` / `typemap.py` / `signature.py` (the eleven OntoNotes
  classes the pools are keyed on), `relation_pool.py` (a candidate pool per
  ordered type pair), `verbalize.py` (a relation as a sentence, so the
  distance is between meanings), `mechanism.py` (the metric exponential
  mechanism over pool + null), `spans.py` / `inplace.py` (the released relation
  substituted where the predicate stood), `graph_inject.py` (the typed
  relations an index should carry).
- `benchmark/` -- the loaders (`benchmarks.py`), the QA scorer
  (`score_hotpot.py`), the LightRAG runner (`run_bench.py`), the attacks
  (`grasp_simple.py`, `agea.py`, `graphsteal.py`, `score_graphsteal.py`), the
  defenses (`defenses.py`, `defenses_hipporag.py`, `attack_defended.py`), the
  sweeps (`sweep_defenses.sh`, `sweep_graphsteal.sh`) and the defense table
  (`defense_table.py`). See `benchmark/README.md`.
- `baselines/` -- `run_pathrag.py` and `run_hipporag.py`, the other two
  victims' runners; `PathRAG/` vendored with its changes recorded;
  `HippoRAG/` as a submodule, unmodified.
- `LightRAG-v1.0.3/` -- the third victim: upstream v1.0.3 plus
  `dp-graphrag.patch` (`UPSTREAM_COMMIT.txt` names the base).
- `datasets/`, `workspace/`, `runs/`, `results/`, `logs/` -- generated; gitignored.

## Demo experiments

Two methods, each needing the corpus (`--scope`: `2wiki100` or `hotpotqa100`)
and the target model (`--model`: the victim's reader, on both the attack side
and the QA side). Everything else defaults from `config.py`.

### Our method

```bash
python main.py ours --scope 2wiki100 --model gpt-4o-mini --eps 1 5
```

Runs the mechanism end to end at each epsilon: perturb, rewrite, inject,
build the three victims' indexes, answer the hundred questions, run GRASP and
GraphSteal, and print the results -- per victim, Naive and RType F1 under
both attacks for identity and each epsilon, and QA F1 for the original corpus
and each epsilon. `--eps` is the method's parameter.

```bash
python main.py results --scope 2wiki100 --model gpt-4o-mini     # print the table again, from disk
```

### Baseline defenses

```bash
python main.py baselines --scope 2wiki100 --model gpt-4o-mini
```

Runs the GRASP paper's defenses (Abstractive, Extractive, ID Alignment,
Decoy) and GraphSteal's protective prompt on the same corpus and target
model, three independent rounds, and prints the defense table.

### Single stages

```bash
python main.py stage perturb --scope 2wiki100
python main.py stage index   --scope 2wiki100 --model gpt-4o-mini --eps 1
python main.py stage attack  --scope 2wiki100 --model gpt-4o-mini --eps 1
```

Stages: `extract pools perturb rewrite inject index attack defenses table`.
Add `--dry` to any command to print what it would run without running it.
Every stage resumes over what exists -- an index is built once, a scored
target or a checkpointed attack is not repeated -- so an interrupted command
is simply rerun.

**Note:** do not run `extract` or `pools` when comparing against the reported
numbers; they call a model and replace the shipped artefacts.

### Where the results land

| File | Content |
|---|---|
| `runs/results_<scope>[_vm-<model>].json` | one entry per (victim, corpus): `grasp` and `graphsteal` as `{naive, rtype}` F1, `qa_f1`; `null` where not on disk |
| `runs/results_<scope>[_vm-<model>].xlsx` | the same table as a sheet |
| `runs/defense_3run_mean[_vm-<model>].xlsx` | the defense table: per victim and benchmark, six settings, both attacks' Naive/RType F1 and QA F1, mean of three rounds; a Runs sheet with every round |

Corpora are named `original`, `identity_inplace` and `eps<e>_seed<s>_inplace`.
Behind every number is its run:

| Path | Content |
|---|---|
| `runs/_u_<scope>[_vm-<model>]/.../predictions_*.json` | one record per question: question, gold, the reader's answer, the retrieval context; `benchmark/score_hotpot.py <file>` scores it by question type |
| `runs/_grasp/<scope>_<victim>_<corpus>[_vm-<model>]/` | `naive_rtype.json` (one row per target: P, R, F1, hits), `turns_<i>.jsonl` (every query and reply), `judge.json` |
| `runs/_graphsteal/<scope>_<victim>_<corpus>[_vm-<model>]/` | the same, plus `state_*.json` (the recovered graph and frontier) |
| `runs/_defense/sweep[_vm-<model>]/rep<r>/<cell>/`, `runs/_graphsteal/defense[_vm-<model>]/rep<r>/<cell>/` | the defense cells, laid out as above |
| `workspace/dp_<scope>/perturbed/`, `sanitized/`, `inject/` | the mechanism's trace per fact, the rewritten corpus, the typed relations handed to each index |
| `workspace/u_<scope>_<lr\|pr\|hp>_<corpus>/` | the victims' indexes |

## Notes on defaults

All in `config.py`; override any for one run with `--set KEY=VALUE`.

| Parameter | Default | Remark |
|---|---|---|
| `EPSILONS` | `[0.1, 1, 5, 10]` | `--eps` on the command line |
| `SEED` | `0` | the mechanism's draws and the decoy endpoints |
| `POOL_SIZE` | `10` | candidate relations per ordered type pair |
| `NULL_DISTANCE` | `1.0` | where null sits in the metric; `>= 0.357` keeps it a metric |
| `WITH_NULL` | `True` | `False` never deletes a sentence (ablation) |
| `EMBEDDER` / `EMBEDDER_REVISION` | `nli-roberta-base-v2`, pinned | the relation encoder and its Hub revision |
| `READER` | `gpt-4o-mini` | `--model`; the victim's reader on both sides |
| `ATTACKER` | `gpt-4o-mini` | the attacker's own LLM and the typed judge; fixed across target models |
| `INDEX_MODEL` | `gpt-4o-mini` | what every index is built with; HippoRAG keys its store on it |
| `VICTIMS` | `lightrag, pathrag, hipporag` | |
| `TOP_K` | `10, 10, 5` | retrieval budget per victim (HippoRAG's own default) |
| `GRASP_QUERIES` | `5` | per target |
| `GRAPHSTEAL` | `75` turns, `predicate`, `degree`, top_k `40` | the targeted configuration |
| `DEFENSES` / `ROUNDS` | all six settings, `3` | the baseline sweep |
| `CONCURRENCY` | `8` | parallel questions while answering |

## License

MIT. Third-party code under its own license in its directory: LightRAG
(HKUDS), PathRAG, HippoRAG (OSU-NLP-Group); see `LICENSE`.
