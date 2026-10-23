"""GRASP against a LightRAG deployment carrying the paper's own defenses.

    DP_SCOPE=2wiki100 python benchmark/attack_defended.py \
        --index workspace/u_2wiki100_lr_original/2wiki100-dp-delta_original/single/all \
        --defense decoy --source original

One run per defense setting; `--defense` takes any mode in defenses.MODES --
none, the paper's abstractive / extractive prompt baselines, its id_align and
decoy context defenses, or GraphSteal's protective prompt. The rest reproduces the GRASP driver this project already
uses, so a defended cell is comparable with the undefended ones already on
disk: the same fifteen targets, the same frozen gold filter, the same judge,
the same F1-of-macro.

The checkpoint key carries the defense name as well as the extraction schema
and the target list: a key that ignored the defense would read the undefended
run's per-target scores back as its own.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "LightRAG-v1.0.3"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(ROOT, ".env"))

import defenses  # noqa: E402
import grasp_simple as G  # noqa: E402
from dp.extract import SCHEMA  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", required=True, help="the victim's index directory")
    p.add_argument("--victim", default="lightrag",
                   choices=["lightrag", "pathrag", "hipporag"],
                   help="the defenses in defenses.py patch LightRAG's context "
                        "construction, so --defense other than none applies "
                        "only to lightrag; the other two are here for a plain "
                        "GRASP run against a chosen target model")
    p.add_argument("--benchmark", default=None,
                   help="hipporag only: the benchmark its retriever needs")
    p.add_argument("--defense", default="none", choices=defenses.MODES)
    p.add_argument("--no_safe_prompt", action="store_true",
                   help="the paper applies its defenses jointly with the safe "
                        "prompt (Section 7.3.1), so it is on by default")
    p.add_argument("--mode", default="hybrid",
                   choices=["hybrid", "local", "global"],
                   help="naive mode builds no relation table, so the defenses "
                        "do not apply to it")
    p.add_argument("--top_k", type=int, default=10)
    # One flag used to drive both the victim's reader and the judge, which is
    # fine while they are the same model and wrong the moment the experiment is
    # "swap the target model": the judge decides RType, so changing it with the
    # victim would move the metric and the system together and neither delta
    # would mean anything. They are separate now, and the judge's default does
    # not move.
    p.add_argument("--victim_model", default="gpt-4o-mini",
                   help="the deployment's reader -- the target model")
    p.add_argument("--index_model", default=None, metavar="MODEL",
                   help="hipporag only: the model the index was built with, "
                        "when it differs from --victim_model. HippoRAG keys "
                        "its OpenIE results and working directory on it and "
                        "refuses a mismatch, so it must stay the builder.")
    p.add_argument("--judge_model", default="gpt-4o-mini",
                   help="scores RType; keep it fixed across target models or "
                        "the numbers are not comparable")
    p.add_argument("--queries", type=int, default=5,
                   help="GRASP's per-target query budget")
    p.add_argument("--source", default="original",
                   choices=["original", "sanitized"])
    p.add_argument("--sanitized", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="attack only the first N targets -- a smoke test, so "
                        "it demands --out_dir: a partial run must never land "
                        "in the cell directory a full one would be read from")
    p.add_argument("--seed", type=int, default=0,
                   help="draws the decoy endpoints; the table is otherwise "
                        "deterministic")
    args = p.parse_args()

    scope = os.environ.get("DP_SCOPE", "").strip() or "hotpotqa"
    targets = json.load(open(
        f"{ROOT}/datasets/subsets/targets_{scope}.json"))["targets"]
    if args.limit is not None:
        if not args.out_dir:
            raise SystemExit("--limit needs --out_dir")
        targets = targets[:args.limit]
    stamp = hashlib.sha256("|".join(targets).encode()).hexdigest()[:8]
    # The safe prompt is part of the setting, so it has to be part of the key.
    # Without it `--defense none` and `--defense none --no_safe_prompt` share a
    # directory and the second run reads the first one's per-target scores back
    # as its own -- the same cache-key failure the module docstring warns about,
    # one field short.
    safe = "" if not args.no_safe_prompt else "_nosafe"
    # the target model belongs in the key for the same reason the defense does
    vm = ("" if args.victim_model == "gpt-4o-mini"
          else "_vm-" + args.victim_model.replace("/", "-"))
    # The source has to name the corpus, not just its kind. `--source
    # sanitized` is the same word for every epsilon, so four DP cells would
    # have shared one directory and each would have read the previous one's
    # per-target scores back as its own -- the cache-key failure this module's
    # docstring warns about, one field short again.
    which = (os.path.basename(args.sanitized).replace(".json", "")
             if args.sanitized else args.source)
    # The prompt is part of the setting, so it is part of the key.
    tag = (f"{scope}_s{SCHEMA}_{stamp}_{args.victim}_def-{args.defense}"
           f"{safe}{vm}_{which}")
    out_dir = args.out_dir or os.path.join(ROOT, "runs", "_defense", tag)
    os.makedirs(out_dir, exist_ok=True)

    frozen = f"{ROOT}/workspace/dp_{scope}/gold_filter.json"
    if not os.path.exists(frozen):
        raise SystemExit(f"no frozen gold filter at {frozen}")
    G.recoverable = functools.partial(G.recoverable, cache_path=frozen,
                                      frozen=True)

    # use_lightrag first, install second: the defense patches names inside the
    # `lightrag` package, and patching before the package is imported would
    # leave the possibility of a second copy being the one that serves queries.
    # Installing afterwards patches the module already in use, and the assert
    # below confirms it on the object LightRAG itself holds.
    if args.victim == "lightrag":
        G.use_lightrag(args.index, args.mode, args.top_k, args.victim_model)
    else:
        from agea import attach_victim
        attach_victim(args.victim, args.index, args.top_k, args.victim_model,
                      args.benchmark, args.index_model)
    # HippoRAG builds no relation table on the path the attack uses, so for it
    # ID Alignment and Decoy are the passage-channel analogues defenses.py
    # describes -- the former nearly the paper's defense, the latter a port.
    # The table reports them under the same rows; the caption says which.
    if args.victim == "hipporag" and args.defense in ("id_align", "decoy"):
        print(f"hipporag: {args.defense} is the passage-channel analogue, "
              f"see defenses.passage_blocks")
    defenses.install(args.defense, seed=args.seed,
                     safe_prompt=not args.no_safe_prompt, victim=args.victim)
    # The patch is checked on the object the victim itself holds, not on the
    # defenses module: what matters is which copy serves the queries.
    if args.victim in ("lightrag", "pathrag"):
        pkg = "lightrag" if args.victim == "lightrag" else "PathRAG"
        _op = __import__(pkg, fromlist=["operate"]).operate
        _pr = __import__(pkg, fromlist=["prompt"]).prompt
        if args.defense != "none":
            assert hasattr(_op.list_of_list_to_csv, "_undefended"), \
                f"{pkg}'s relation table is not going through the defense"
        if args.defense == "decoy":
            assert "---Internal fields---" in _pr.PROMPTS["rag_response"], \
                f"the decoy instruction did not reach {pkg}'s rag_response"
    elif args.defense != "none":
        if args.defense == "decoy":
            assert "---Internal fields---" in defenses.reader_prompt(), \
                "the decoy instruction did not reach hipporag's reader"
        elif args.defense != "id_align":
            assert defenses.reader_prompt(), \
                "hipporag's reader would get no defense text"
    print(f"{len(targets)} targets, victim={args.victim}, "
          f"defense={args.defense}, "
          f"victim={args.victim_model}, judge={args.judge_model}, tag={tag}\n")

    # Name-alias matching, on only for a reader that writes a person's short
    # name where the gold holds the long one. It is a scoring-side fix: the
    # prompt-side one was tried and removed, see reader_quirks.
    from reader_quirks import applies
    alias = applies(args.victim_model)
    if alias:
        print(f"name-alias matching on for victim_model={args.victim_model}")

    rows, skipped = [], []
    for i, target in enumerate(targets, 1):
        one = os.path.join(out_dir, f"target_{i}.json")
        if os.path.exists(one):
            saved = json.load(open(one))
            if saved:
                rows.append(saved)
            print(f"[{i}/{len(targets)}] {target}: cached")
            continue
        res = G.run_attack(target, args.queries,
                           os.path.join(out_dir, f"turns_{i}.jsonl"))
        got = G.compare_to_gold(res, None, args.judge_model,
                                os.path.join(out_dir, "judge.json"),
                                ground_truth="triples", source=args.source,
                                sanitized=args.sanitized,
                                name_aliases=alias)
        if got:
            got["target"] = target
            rows.append(got)
        else:
            skipped.append(target)
        json.dump(got or {}, open(one, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False)
        print(f"[{i}/{len(targets)}] {target}: gold {got.get('gold', '-') if got else '-'}, "
              f"RType F1 {got.get('rtype_f1', 0) if got else 0:.2f}", flush=True)

    if not rows:
        raise SystemExit(f"{tag}: no target could be scored")
    json.dump(rows, open(os.path.join(out_dir, "naive_rtype.json"), "w",
                         encoding="utf-8"), indent=2, ensure_ascii=False)

    # A target the attack recovered nothing for is a zero, not a target to drop:
    # skipping it lifts the macro by removing a 0, so a defense that silences a
    # target entirely would score as if it had never been attacked. Both
    # conventions are printed, and the padded one is the one to quote.
    n = len(targets)
    mean = lambda k: statistics.mean(r[k] for r in rows)
    pad = lambda k: sum(r[k] for r in rows) / n
    f1 = lambda a, b: 2 * a * b / (a + b) if a + b else 0.0
    print("\n" + "=" * 72)
    print(f"{tag}   scored {len(rows)}/{n}"
          + (f"   recovered nothing for {skipped}" if skipped else ""))
    for label, agg in (("padded to 15", pad), ("scored only", mean)):
        print(f"\n  [{label}]  {'metric':<8}{'P':>9}{'R':>9}{'F1':>9}")
        for name, pre in (("Naive", "naive"), ("RType", "rtype")):
            P, R = agg(f"{pre}_p"), agg(f"{pre}_r")
            print(f"  {'':<14}{name:<8}{P:>9.2f}{R:>9.2f}{f1(P, R):>9.2f}")
    print(f"\ngold {sum(r['gold'] for r in rows)}, "
          f"pair hits {sum(r['pair_hits'] for r in rows)}, "
          f"relation also right {sum(r['typed_hits'] for r in rows)}")
    # Which patches actually fired. A cell whose defense never reached the
    # victim still completes and still reports a number; this line is what
    # tells that cell from a real one, so it is printed and saved with it.
    fired = dict(defenses.applied())
    # `install` patches every importable victim, so the counter also shows
    # the ones that were not serving; what decides whether this cell was
    # defended is the serving victim's own keys.
    mine = {"lightrag": "lightrag:", "pathrag": "PathRAG:",
            "hipporag": "reader_"}[args.victim]
    reached = {k: v for k, v in fired.items() if k.startswith(mine)}
    print(f"defense patches fired on {args.victim}: {reached or 'NONE'}")
    json.dump({"defense": args.defense, "victim": args.victim,
               "safe_prompt": not args.no_safe_prompt,
               "reached_victim": reached, "all": fired},
              open(os.path.join(out_dir, "defense_applied.json"), "w"),
              indent=2)
    if args.defense != "none" and not reached:
        print("!! the defense never reached the victim: this cell is undefended")


if __name__ == "__main__":
    main()
