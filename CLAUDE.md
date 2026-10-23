# DP-GraphRAG

## Environment

One conda environment, `relguard`, built from `environment.yml` (Python 3.10, every
version pinned). Run everything from the project root with that environment's
interpreter; `main.py` uses `sys.executable` for the stages it launches.

## Baselines, fixed

A row that stands for "no DP" or "no defense" in a privacy or attack table is
the **identity** rewrite (the corpus passed through the DP write-back with
every relation released unchanged). In a utility table it is the **original**
corpus. The sweep scripts and `benchmark/defense_table.py` encode this; do not
add an original-corpus baseline to a privacy table or an identity row to a
utility table.

## Alignment with the reported numbers

`assets/` holds the extraction, the candidate pools, their verbalisations, the
predicate spans and the frozen gold filter for both corpora, and `config.py`
pins the seed, the models and the embedder revision. `scripts/fetch_data.py`
puts them in place. Regenerating any of these with a model call produces a
different artefact; do not regenerate them when comparing against the paper.
