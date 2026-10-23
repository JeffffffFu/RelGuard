# RelGuard

Differential privacy for the relation semantics of a GraphRAG knowledge graph
(the code name of the pipeline is DP-GraphRAG; paths and scopes use it).

Each extracted fact (subject, relation, object) carries a type pair, the type
pair carries a candidate pool generated from the types alone, and the relation
is resampled from that pool by a metric exponential mechanism -- or withheld,
which deletes the sentence. The released relation is written back in place of
the predicate's own words, the rewritten corpus is indexed by an unmodified
GraphRAG system, and the result is measured from both sides: what three
reconstruction attacks recover from the deployment, and what a QA benchmark
loses.

## Setup

    git clone --recursive https://github.com/JeffffffFu/RelGuard.git
    cd RelGuard
    conda env create -f environment.yml      # creates `relguard`; see Environment below
    conda activate relguard
    cp .env.example .env                     # OPENAI_API_KEY, optionally OPENAI_BASE_URL
    python scripts/fetch_data.py             # datasets/ and workspace/ from the submodule and assets/

### Environment

One conda environment, `relguard`, from `environment.yml`: Python 3.10 and
every package pinned to the version the reported numbers were produced with
(the file is a full `conda env export`, not a hand-written list). What it
contains and why:

- `torch==2.14.1+cpu` and `sentence-transformers==6.1.0` run the relation
  encoder locally; the CPU wheel comes from PyTorch's own index, which the
  file names. A GPU is not needed anywhere in the pipeline. The encoder's Hub
  revision is pinned separately in `config.EMBEDDER_REVISION`, because the
  mechanism's draws are a function of the vectors.
- `openai` is the client for every model call; `OPENAI_BASE_URL` in `.env`
  points it at any OpenAI-compatible gateway (the reported runs went through
  one), `OPENAI_API_KEY` is read from the same file.
- The three victims are imported from the repository, not installed: the
  patched LightRAG from `LightRAG-v1.0.3/`, PathRAG from `baselines/PathRAG/`,
  HippoRAG 2 from the `baselines/HippoRAG/` submodule (`--recursive`, or
  `git submodule update --init`). Their own dependencies (`nano-vectordb`,
  `networkx`, `igraph`, `dspy`, `tiktoken`, ...) are in the file.
- `openpyxl` writes the tables; `tenacity` is what the victims retry with.

Everything is run from the repository root with that environment's
interpreter; `main.py` launches each stage with `sys.executable`, so
activating the environment is the only step. To rebuild the file after
changing a dependency: `conda env export -n relguard --no-builds`, drop the
`prefix:` line, and keep the `--extra-index-url` line under `pip:`.

`fetch_data.py` builds `datasets/` from the HotpotQA and 2WikiMultiHopQA
samples HippoRAG 2 publishes (the submodule), and copies into `workspace/` the
per-corpus artefacts the reported numbers depend on: the extraction, the
candidate pools, their verbalisations, the predicate spans and the frozen gold
filter. These were produced by model calls and would not come back identical
if regenerated.

## Running

Two methods, each needing the corpus and the target model; everything else
defaults from `config.py`.

    python main.py ours      --scope 2wiki100 --model gpt-4o-mini --eps 1 5
    python main.py baselines --scope 2wiki100 --model gpt-4o-mini

`ours` runs the mechanism end to end at each epsilon -- perturb, rewrite,
inject, build the three victims' indexes, answer the hundred questions, run
GRASP and GraphSteal -- and prints the results: per victim, privacy F1 for
identity (no DP) and each epsilon under both attacks, and QA F1 for the
original corpus and each epsilon. `results` prints that table again from
what is on disk. `baselines` runs the GRASP paper's defenses and GraphSteal's
protective prompt on the same corpus and model, three independent rounds,
and prints the defense table.

`--model` is the victim's reader on the attack side and the QA side; the
attacker and the judge stay at `config.ATTACKER`, so a delta between target
models is attributable. Any config value can be overridden for one run with
`--set KEY=VALUE`; `--dry` prints the commands without running them. Every
stage resumes over what exists, so an interrupted command is rerun as is.

Single stages, for a partial rerun:

    python main.py stage perturb --scope 2wiki100
    python main.py stage attack  --scope 2wiki100 --model gpt-4o-mini --eps 1

`extract` and `pools` regenerate the shipped artefacts from scratch, for a new
corpus; do not run them when comparing against the paper.

## Layout

    main.py                  `ours`, `baselines`, `results`, and the single stages
    config.py                every parameter of a run
    scripts/fetch_data.py    datasets/ and workspace/ from the submodule and assets/
    assets/                  the artefacts a run must share with the paper (see Setup)

    dp/                      the mechanism, in pipeline order
      extract.py             entities with types, triples with where they are said
      entity_types.py        the eleven OntoNotes classes the pools are keyed on
      typemap.py             extracted type strings onto those classes
      signature.py           type signatures and the pools they imply
      relation_pool.py       a candidate pool per ordered type pair, generated
      verbalize.py           a relation as a sentence, so the distance is between meanings
      mechanism.py           the metric exponential mechanism over pool + null
      spans.py               the shortest substring of a sentence that states a relation
      inplace.py             the released relation substituted where the predicate stood
      graph_inject.py        the typed relations an index should carry

    benchmark/               loaders, the QA scorer, three attacks, the defenses,
                             the sweeps and the table -- see benchmark/README.md
    baselines/
      run_pathrag.py         PathRAG utility runner      (baselines/PathRAG/, vendored with its changes recorded)
      run_hipporag.py        HippoRAG 2 utility runner   (baselines/HippoRAG/, submodule, unmodified)
    LightRAG-v1.0.3/         the third victim: upstream v1.0.3 plus dp-graphrag.patch

    datasets/ workspace/ results/ runs/ logs/    generated by fetch_data.py and the stages; gitignored

## Victims and attacks

Three GraphRAG systems are attacked as deployments: LightRAG, PathRAG and
HippoRAG 2. Three attacks reconstruct the graph through their query interface
-- GRASP, AGEA and GraphSteal -- and are scored on the same fifteen targets
against a frozen gold extraction. Naive counts an entity pair recovered; RType
also requires the relation.

## Baselines, fixed

A row standing for "no DP" or "no defense" in a privacy table is the identity
rewrite: the corpus passed through the write-back with every relation released
unchanged, which isolates the mechanism's noise from the rewrite's
canonicalisation. A utility table's baseline is the original corpus. The sweep
scripts and `defense_table.py` encode both.

## What reproduces exactly, and what does not

With the shipped artefacts, the pinned environment and the pinned embedder
revision (`config.EMBEDDER_REVISION`), `perturb`, `rewrite` and `inject`
reproduce the paper's traces, rewritten corpora and injected relations byte
for byte -- this was checked. Everything downstream calls a model: building
an index, answering questions, attacking, judging. Those numbers reproduce
within the run-to-run spread the paper reports (three independent runs of
one cell move RType F1 by up to 9-12 points on these victims), not exactly,
so a delta is read with its paired confidence interval and not its mean.

## Metrics worth knowing the shape of

**HotpotQA / 2Wiki: EM and token F1**, deterministic. EM reads the whole
reply and gold answers run to two words, so the answer shape matters;
`ANSWER_SHAPE` in `run_bench.py` holds one instruction per question type.

**Attack metrics** are macro over targets, padded to the full target list: a
target the attack recovered nothing for is a zero, not a target to drop.

## License

MIT; third-party code under its own license in its directory (see LICENSE).
