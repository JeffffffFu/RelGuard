"""The pipeline, one stage per subcommand, every parameter from config.py.

    python main.py extract                 # facts out of the corpus, once
    python main.py pools                   # types, pools, verbalisations, spans
    python main.py perturb                 # the mechanism, one trace per epsilon
    python main.py rewrite                 # identity and every epsilon written back
    python main.py inject                  # typed relations for each index
    python main.py index                   # build every victim's indexes, answer QA
    python main.py attack                  # GRASP and GraphSteal on identity and every epsilon
    python main.py defenses                # the baseline defenses, three rounds
    python main.py table                   # the defense table from the runs
    python main.py all                     # the stages above, in order

    python main.py perturb --dry           # print the commands, run nothing
    python main.py index --set SCOPE=hotpotqa100 --set READER=gemini-2.5-flash

Nothing is computed here. Each stage assembles the command line its module
already understands and runs it, so a stage can always be rerun by hand from
the line this prints. Stages are resumable where their modules are: the
mechanism and the rewrite overwrite their traces, indexes are built once and
reused, and the attack and defense drivers skip cells whose result exists.

Two baselines are fixed, and this file follows them: a privacy run attacks
the identity rewrite as its "no DP" row, and a utility run reads the original
corpus as its "no DP" row.
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
        out = f"runs/_u_{C.SCOPE}"
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
            tag = f"{C.SCOPE}_{victim}_{corpus}"
            run(py("benchmark/attack_defended.py", "--victim", victim, "--index", idx,
                   "--defense", "none", "--no_safe_prompt", *src, *vm, *hippo,
                   "--queries", C.GRASP_QUERIES, "--judge_model", C.ATTACKER,
                   "--top_k", C.TOP_K[victim], "--seed", C.SEED,
                   "--out_dir", f"runs/_grasp/{tag}"), scope_env())
            g = C.GRAPHSTEAL
            out = f"runs/_graphsteal/{tag}"
            run(py("benchmark/graphsteal.py", "--victim", victim, "--index", idx,
                   "--targets", targets, "--turns", g["turns"], "--command", g["command"],
                   "--order", g["order"], "--top_k", g["top_k"], "--gold", g["gold"],
                   "--model", C.ATTACKER, *vm, *hippo, "--seed", C.SEED,
                   "--out_dir", out), scope_env())
            run(py("benchmark/score_graphsteal.py", "--run", out, "--model", C.ATTACKER,
                   *src, *vm), scope_env())


def defenses():
    """The two sweep drivers, one lane per victim, sequentially. Run the lanes
    in parallel by hand (see the scripts) when wall-clock matters."""
    env = {"VM": C.READER}
    for victim in C.VICTIMS:
        run(["bash", "benchmark/sweep_defenses.sh", victim], env)
        run(["bash", "benchmark/sweep_graphsteal.sh", victim], env)


def table():
    run(py("benchmark/defense_table.py", "--victim_model", C.READER))


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
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config.py value for this run; the value is "
                        "parsed as a Python literal when it is one")
    p.add_argument("--dry", action="store_true", help="print the commands only")
    args = p.parse_args()
    DRY = args.dry
    apply_overrides(args.set)
    if C.SCOPE not in C.CORPORA:
        raise SystemExit(f"SCOPE {C.SCOPE!r} is not in config.CORPORA")
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        print(f"\n== {name}  (scope {C.SCOPE}, reader {C.READER})")
        STAGES[name]()


if __name__ == "__main__":
    main()
