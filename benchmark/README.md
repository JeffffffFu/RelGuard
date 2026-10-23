# Benchmarks, attacks and defenses

    benchmarks.py        HotpotQA and 2WikiMultiHopQA into one Corpus/Question
                         schema, plus the `*-dp-delta:<corpus>` specs that
                         substitute a rewritten corpus. Imports neither a
                         victim nor a model client.
    run_bench.py         index with LightRAG, answer, record the retrieval
                         context; `--defense` patches the context builder.
    score_hotpot.py      EM and token F1, transcribed from hotpot_evaluate_v1.py.
                         Calls no model.

    grasp_simple.py      GRASP: one entity's one-hop relations by repeated
                         structured queries; also the typed judge and the
                         frozen gold filter every attack is scored with.
    agea.py              AGEA, and the parsers, graph memory and metrics
                         GraphSteal shares with it.
    graphsteal.py        GraphSteal: frontier traversal, targeted or not.
    score_graphsteal.py  scores a GraphSteal run on the GRASP targets.

    defenses.py          the GRASP paper's defenses, installed on a victim at
                         runtime; defenses_hipporag.py is the branch for a
                         passage-only victim.
    attack_defended.py   GRASP against a defended victim.
    reader_quirks.py     name-alias scoring for readers that write short names.
    sweep_defenses.sh    GRASP and utility, six settings, two benchmarks,
                         three rounds; sweep_graphsteal.sh the same for
                         GraphSteal. `VM=<model>` changes the reader.
    defense_table.py     the defense table, read back from the runs.

Every number reported comes from the benchmark that defined it: HotpotQA's
two metrics call nothing, which is why they are worth having -- they put no
judge-sized floor under a comparison. The attack metrics are macro over the
fifteen targets, padded to the full list: a target the attack recovered
nothing for is a zero, not a target to drop.

## Answer shape

EM reads the whole reply and the gold answers run to two words, so an essay
containing the right span scores zero. `ANSWER_SHAPE` in run_bench.py holds
one instruction per question type; fixing two of them was worth 10 EM.

## Topology

HotpotQA and 2Wiki need `--topology single`: the passages are shared across
the questions and a question names no document.
