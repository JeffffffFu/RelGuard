"""RelGuard: our method and the baseline defenses, from the command line.

    python main.py ours      --scope 2wiki100 --model gpt-4o-mini --eps 1 5
    python main.py baselines --scope 2wiki100 --model gpt-4o-mini
    python main.py results   --scope 2wiki100 --model gpt-4o-mini

`ours` runs the mechanism end to end on one corpus and one target model:
perturb, rewrite, inject, index, answer, attack, and prints the results --
privacy F1 for identity (no DP) and each epsilon under GRASP and GraphSteal,
QA F1 for the original corpus and each epsilon, for every victim. Epsilon is
its parameter; everything else defaults from config.py.

`baselines` runs the GRASP paper's defenses (and GraphSteal's protective
prompt) on the same corpus and target model, three independent rounds, and
prints the defense table. They need no epsilon.

`--scope` (2wiki100 or hotpotqa100) and `--model` (the victim's reader, on
both the attack side and the QA side) are required for both; the attacker,
the judge, the seed, the budgets and every path are in config.py, and any of
them can be overridden for one invocation with `--set KEY=VALUE`.

Single stages, for debugging or a partial rerun:

    python main.py stage perturb --scope 2wiki100
    python main.py stage index   --scope 2wiki100 --model gpt-4o-mini --eps 1
    python main.py stage attack  --scope 2wiki100 --model gpt-4o-mini --eps 1

    stages: extract pools perturb rewrite inject index attack defenses table

Nothing is computed here. Each stage assembles the command line its module
already understands and runs it, and prints the line, so any step can be rerun
by hand; `--dry` prints without running. Every stage resumes over what exists:
an index is built once, a scored target or a checkpointed attack is not
repeated, and the commands can be rerun after an interruption.

Two baselines are fixed, and this file follows them: a privacy number's
"no DP" row is the identity rewrite, a utility number's "no DP" row is the
original corpus.
"""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config as C  # noqa: E402

DRY = False


def run(cmd: list[str], env: dict | None = None) -> None:
    extra = {k: v for k, v in (env or {}).items()}
    shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in extra.items())
    print(f"$ {shown}{' ' if shown else ''}{' '.join(shlex.quote(c) for c in cmd)}",
          flush=True)
    if DRY:
        return
    full = {**os.environ, **extra}
    subprocess.run(cmd, cwd=ROOT, env=full, check=True)


def py(module_or_path: str, *args) -> list[str]:
    head = ([C.PYTHON, "-m", module_or_path] if module_or_path.startswith("dp.")
            else [C.PYTHON, module_or_path])
    return head + [str(a) for a in args]


def scope_env() -> dict:
    return {"DP_SCOPE": C.SCOPE, "DP_EMBEDDER_REVISION": C.EMBEDDER_REVISION}


def reader_tag() -> str:
    """"" for the default reader; "_vm-<model>" otherwise, so two target models
    never share an output path."""
    return "" if C.READER == "gpt-4o-mini" else "_vm-" + C.READER.replace("/", "-")


def utility_file(victim: str, corpus: str) -> str:
    spec, _, delta = C.CORPORA[C.SCOPE]
    out = f"runs/_u_{C.SCOPE}{reader_tag()}"
    if victim == "lightrag":
        return f"{out}/predictions_{delta}_{corpus}_hybrid_top{C.TOP_K[victim]}.json"
    bm = spec.replace(":", "_")
    tag = "pathrag_top10" if victim == "pathrag" else f"hipporag2_top{C.TOP_K[victim]}"
    return f"{out}/{C.SHORT[victim]}_{corpus}/predictions_{bm}_{tag}.json"


def attack_dir(attack: str, victim: str, corpus: str) -> str:
    return f"runs/_{attack}/{C.SCOPE}_{victim}_{corpus}{reader_tag()}"


def corpus_names() -> list[str]:
    """identity first, then one per epsilon: the names dp.inplace writes."""
    return ["identity_inplace"] + [f"eps{e:g}_seed{C.SEED}_inplace" for e in C.EPSILONS]


def index_dir(victim: str, corpus: str) -> str:
    base = f"workspace/u_{C.SCOPE}_{C.SHORT[victim]}_{corpus}"
    if victim == "lightrag":
        _, _, delta = C.CORPORA[C.SCOPE]
        return f"{base}/{delta}_{corpus}/single/all"
    return base


# ------------------------------------------------------------------ stages

def extract():
    spec, subset, _ = C.CORPORA[C.SCOPE]
    run(py("dp.extract", "--benchmark", spec, "--subset", subset,
           "--model", C.EXTRACTOR, "--concurrency", C.CONCURRENCY), scope_env())


def pools():
    env = scope_env()
    run(py("dp.typemap", "--build", "--model", C.EXTRACTOR), env)
    run(py("dp.signature", "--build", "--model", C.EXTRACTOR), env)
    run(py("dp.relation_pool", "--build", "-n", C.POOL_SIZE,
           "--model", C.POOL_GENERATOR, "--concurrency", C.CONCURRENCY), env)
    run(py("dp.verbalize", "--build", "--model", C.EXTRACTOR,
           "--concurrency", C.CONCURRENCY), env)
    run(py("dp.spans", "--build", "--model", C.EXTRACTOR), env)


def perturb():
    args = ["--epsilon", *C.EPSILONS, "--seed", C.SEED, "--model", C.EMBEDDER,
            "--whiten", C.WHITEN, "--null_distance", C.NULL_DISTANCE]
    if not C.WITH_NULL:
        args.append("--no_null")
    run(py("dp.mechanism", *args), scope_env())


def rewrite():
    env = scope_env()
    polish = ["--polish", C.POLISH] if C.POLISH else []
    run(py("dp.inplace", "--identity", "--seed", C.SEED,
           "--concurrency", C.CONCURRENCY, *polish), env)
    for e in C.EPSILONS:
        run(py("dp.inplace", "--epsilon", f"{e:g}", "--seed", C.SEED,
               "--concurrency", C.CONCURRENCY, *polish), env)


def inject():
    env = scope_env()
    run(py("dp.graph_inject", "--original"), env)
    for name in corpus_names():
        run(py("dp.graph_inject", "--run", name), env)


def index(corpora: list[str] | None = None):
    """Build each victim's index over each corpus and answer the hundred
    questions. `original` is the utility baseline; identity and the epsilons
    are the privacy configurations."""
    spec, subset, delta = C.CORPORA[C.SCOPE]
    for corpus in corpora or ["original"] + corpus_names():
        typed = {"DP_TYPED_RELATIONS": f"workspace/dp_{C.SCOPE}/inject/{corpus}.json"}
        sanitized = ([] if corpus == "original" else
                     ["--sanitized", f"workspace/dp_{C.SCOPE}/sanitized/{corpus}.json"])
        out = f"runs/_u_{C.SCOPE}{reader_tag()}"
        for victim in C.VICTIMS:
            ws = f"workspace/u_{C.SCOPE}_{C.SHORT[victim]}_{corpus}"
            if victim == "lightrag":
                run(py("benchmark/run_bench.py", "--benchmark", f"{delta}:{corpus}",
                       "--subset", subset, "--topology", "single", "--mode", C.MODE,
                       "--top_k", C.TOP_K[victim], "--llm_model", C.READER,
                       "--workspace", ws, "--out_dir", out,
                       "--concurrency", C.CONCURRENCY), {**scope_env(), **typed})
            elif victim == "pathrag":
                run(py("baselines/run_pathrag.py", "--benchmark", spec, "--subset", subset,
                       "--passages_from_questions", *sanitized, "--workspace", ws,
                       "--top_k", C.TOP_K[victim], "--llm_model", C.READER,
                       "--out_dir", f"{out}/{C.SHORT[victim]}_{corpus}",
                       "--concurrency", C.CONCURRENCY), {**scope_env(), **typed})
            else:
                reader = ([] if C.READER == C.INDEX_MODEL else
                          ["--qa_model", C.READER, "--pin_answer_format"])
                run(py("baselines/run_hipporag.py", "--benchmark", spec, "--subset", subset,
                       "--passages_from_questions", *sanitized, "--save_dir", ws,
                       "--llm_model", C.INDEX_MODEL, *reader, "--qa_top_k", C.TOP_K[victim],
                       "--out_dir", f"{out}/{C.SHORT[victim]}_{corpus}"), scope_env())


def attack():
    """GRASP and GraphSteal against every victim on identity and each epsilon,
    scored against the frozen gold filter."""
    spec, subset, delta = C.CORPORA[C.SCOPE]
    targets = C.TARGETS.format(scope=C.SCOPE)
    vm = [] if C.READER == "gpt-4o-mini" else ["--victim_model", C.READER]
    for corpus in corpus_names():
        src = ["--source", "sanitized", "--sanitized",
               f"workspace/dp_{C.SCOPE}/sanitized/{corpus}.json"]
        for victim in C.VICTIMS:
            idx = index_dir(victim, corpus)
            hippo = (["--benchmark", f"{delta}:{corpus}", "--index_model", C.INDEX_MODEL]
                     if victim == "hipporag" else [])
            run(py("benchmark/attack_defended.py", "--victim", victim, "--index", idx,
                   "--defense", "none", "--no_safe_prompt", *src, *vm, *hippo,
                   "--queries", C.GRASP_QUERIES, "--judge_model", C.ATTACKER,
                   "--top_k", C.TOP_K[victim], "--seed", C.SEED,
                   "--out_dir", attack_dir("grasp", victim, corpus)), scope_env())
            g = C.GRAPHSTEAL
            out = attack_dir("graphsteal", victim, corpus)
            run(py("benchmark/graphsteal.py", "--victim", victim, "--index", idx,
                   "--targets", targets, "--turns", g["turns"], "--command", g["command"],
                   "--order", g["order"], "--top_k", g["top_k"], "--gold", g["gold"],
                   "--model", C.ATTACKER, *vm, *hippo, "--seed", C.SEED,
                   "--out_dir", out), scope_env())
            run(py("benchmark/score_graphsteal.py", "--run", out, "--model", C.ATTACKER,
                   *src, *vm), scope_env())


def defenses():
    """The two sweep drivers, one lane per victim, sequentially. Run the lanes
    in parallel by hand (see the scripts) when wall-clock matters. The
    defended cells attack the original index and the `none` cells the identity
    index, so both are built first."""
    index(["original", "identity_inplace"])
    settings = " ".join(f"{d}:{'safe' if d in C.SAFE_PROMPT_WITH else 'nosafe'}"
                        for d in C.DEFENSES)
    env = {"VM": C.READER, "ROUNDS": str(C.ROUNDS), "SETTINGS": settings,
           "DP_SCOPES": C.SCOPE}
    for victim in C.VICTIMS:
        run(["bash", "benchmark/sweep_defenses.sh", victim], env)
        run(["bash", "benchmark/sweep_graphsteal.sh", victim], env)


def table():
    run(py("benchmark/defense_table.py", "--victim_model", C.READER))


def results() -> dict:
    """Privacy and utility of the method, read back from the runs on disk.

    One block per victim. Privacy rows are identity (the no-DP baseline) and
    each epsilon: Naive and RType F1 under GRASP and GraphSteal, macro over
    the targets padded to the full list, as every table in this project.
    Utility rows are the original corpus (the no-DP baseline) and each
    epsilon: token F1 over the hundred questions. Missing cells print as "-".
    """
    import contextlib, io, json
    sys.path.insert(0, os.path.join(ROOT, "benchmark"))
    from score_hotpot import score

    def macro(path):
        p = os.path.join(ROOT, path, "naive_rtype.json")
        if not os.path.exists(p):
            return None
        rows = json.load(open(p, encoding="utf-8"))
        n = len(json.load(open(os.path.join(ROOT, C.TARGETS.format(scope=C.SCOPE))))["targets"])
        f1 = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
        avg = lambda k: sum(r[k] for r in rows) / n
        return {"naive": f1(avg("naive_p"), avg("naive_r")),
                "rtype": f1(avg("rtype_p"), avg("rtype_r"))}

    def qa(path):
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            return score(p)["overall"]["f1"]

    table = {}
    print(f"\n== results  scope {C.SCOPE}  reader {C.READER}  seed {C.SEED}")
    head = (f"{'':10}{'corpus':22}{'GRASP Naive':>12}{'GRASP RType':>12}"
            f"{'GS Naive':>10}{'GS RType':>10}{'QA F1':>8}")
    for victim in C.VICTIMS:
        print(f"\n{victim}\n{head}")
        for corpus in ["original"] + corpus_names():
            g, s_, q = macro(attack_dir("grasp", victim, corpus)), \
                       macro(attack_dir("graphsteal", victim, corpus)), \
                       qa(utility_file(victim, corpus))
            table[(victim, corpus)] = {"grasp": g, "graphsteal": s_, "qa_f1": q}
            cell = lambda v: f"{v:.2f}" if v is not None else "-"
            label = {"original": "original (QA baseline)",
                     "identity_inplace": "identity (privacy baseline)"}.get(corpus, corpus)
            print(f"{'':10}{label:22}{cell(g and g['naive']):>12}{cell(g and g['rtype']):>12}"
                  f"{cell(s_ and s_['naive']):>10}{cell(s_ and s_['rtype']):>10}{cell(q):>8}")
    out = os.path.join(ROOT, "runs", f"results_{C.SCOPE}{reader_tag()}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"scope": C.SCOPE, "reader": C.READER, "seed": C.SEED,
               "epsilons": C.EPSILONS,
               "cells": {f"{v}|{c}": r for (v, c), r in table.items()}},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n-> {out}")
    return table


def ours() -> None:
    """Our method, end to end, for config.SCOPE, config.READER, config.EPSILONS.

    The shipped artefacts (extraction, pools, verbalisations, spans) are the
    starting point, so `extract` and `pools` are not run here: rerunning them
    would call a model and produce different artefacts.
    """
    perturb()
    rewrite()
    inject()
    index(["original"] + corpus_names())
    attack()
    results()


def baselines() -> None:
    """The baseline defenses on the same corpus and target model, then the table."""
    inject_original = f"workspace/dp_{C.SCOPE}/inject/original.json"
    if not os.path.exists(os.path.join(ROOT, inject_original)):
        rewrite()
        inject()
    defenses()
    table()


STAGES = {"extract": extract, "pools": pools, "perturb": perturb, "rewrite": rewrite,
          "inject": inject, "index": index, "attack": attack, "defenses": defenses,
          "table": table}


def apply_overrides(items: list[str]) -> None:
    for item in items:
        key, _, value = item.partition("=")
        if not hasattr(C, key):
            raise SystemExit(f"config has no {key}")
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = value
        setattr(C, key, parsed)
        print(f"config: {key} = {parsed!r}")


def main() -> None:
    global DRY
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, scope_required):
        sp.add_argument("--scope", required=scope_required, metavar="CORPUS",
                        choices=sorted(C.CORPORA),
                        help="the corpus: " + " or ".join(sorted(C.CORPORA)))
        sp.add_argument("--model", required=scope_required, metavar="READER",
                        help="the target model: the victim's reader on the attack "
                             "side and the QA side; the attacker and the judge stay "
                             "fixed (config.ATTACKER)")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config.py value for this run")
        sp.add_argument("--dry", action="store_true", help="print the commands only")

    o = sub.add_parser("ours", help="our method: the mechanism at each epsilon, "
                                    "attacked and answered, then the results table")
    common(o, True)
    o.add_argument("--eps", type=float, nargs="+", metavar="E", default=None,
                   help=f"epsilons to run (default {C.EPSILONS})")
    b = sub.add_parser("baselines", help="the baseline defenses on the same corpus "
                                         "and target model, then the defense table")
    common(b, True)
    r = sub.add_parser("results", help="print our method's results again, from disk")
    common(r, True)
    r.add_argument("--eps", type=float, nargs="+", metavar="E", default=None)
    st = sub.add_parser("stage", help="one stage of the pipeline")
    st.add_argument("name", choices=list(STAGES))
    common(st, False)
    st.add_argument("--eps", type=float, nargs="+", metavar="E", default=None)

    args = p.parse_args()
    DRY = args.dry
    if args.scope:
        C.SCOPE = args.scope
    if args.model:
        C.READER = args.model
    if getattr(args, "eps", None):
        C.EPSILONS = args.eps
    apply_overrides(args.set)
    print(f"== {args.command}{' ' + args.name if args.command == 'stage' else ''}"
          f"  scope {C.SCOPE}  reader {C.READER}  eps {C.EPSILONS}  seed {C.SEED}")
    {"ours": ours, "baselines": baselines, "results": results,
     "stage": lambda: STAGES[args.name]()}[args.command]()


if __name__ == "__main__":
    main()
