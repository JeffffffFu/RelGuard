"""Put the data a run needs where the code expects it.

    python scripts/fetch_data.py            # datasets/ from the HippoRAG submodule, assets/ into workspace/
    python scripts/fetch_data.py --trex     # also download KILT T-REx, only needed to rebuild the type inventory

Nothing here calls a model. Two things are assembled:

datasets/      HotpotQA and 2WikiMultiHopQA, the 1,000-question samples and
               their corpora that HippoRAG 2 publishes under
               baselines/HippoRAG/reproduce/dataset/ -- the same files, under
               the names benchmark/benchmarks.py reads. The hundred-question
               subsets and the fifteen attack targets come from assets/subsets.
workspace/     the per-corpus artefacts that the mechanism's privacy argument
               and the reported numbers depend on, copied from assets/<scope>/:
               the extraction (which entities and triples, and where in the
               text), the candidate pools, their verbalisations, the predicate
               spans and the frozen gold filter. These were produced by LLM
               calls and would not come back identical if regenerated; shipping
               them is what makes a run here comparable with the paper's.

With these in place every stage of main.py runs from the mechanism onwards
(`python main.py perturb` is deterministic given the seed and the pinned
embedder), and `python main.py extract` / `pools` regenerate the artefacts
from scratch for a new corpus.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIPPO = os.path.join(ROOT, "baselines", "HippoRAG", "reproduce", "dataset")
ASSETS = os.path.join(ROOT, "assets")

DATASETS = {
    # ours                         HippoRAG's reproduction files
    "hotpotqa": ("hotpotqa.json", "hotpotqa_corpus.json"),
    "2wiki": ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json"),
}

SCOPES = ("2wiki100", "hotpotqa100")


def copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    print(f"  {os.path.relpath(dst, ROOT)}")


def datasets() -> None:
    if not os.path.isdir(HIPPO):
        raise SystemExit("baselines/HippoRAG is empty: clone with --recursive, "
                         "or run `git submodule update --init`")
    for name, (questions, corpus) in DATASETS.items():
        root = os.path.join(ROOT, "datasets", name)
        copy(os.path.join(HIPPO, questions), os.path.join(root, "dev_questions.json"))
        copy(os.path.join(HIPPO, corpus), os.path.join(root, "dev.json"))
    copy(os.path.join(ASSETS, "subsets"), os.path.join(ROOT, "datasets", "subsets"))
    copy(os.path.join(ASSETS, "kilt"), os.path.join(ROOT, "datasets", "kilt"))


def workspace() -> None:
    for scope in SCOPES:
        src = os.path.join(ASSETS, scope)
        if not os.path.isdir(src):
            continue
        copy(src, os.path.join(ROOT, "workspace", f"dp_{scope}"))


def trex() -> None:
    """The KILT T-REx split, from the Hugging Face hub, as parquet.

    Only `python -m dp.entity_types --build` and `dp.signature --build` read
    it, and both have their outputs shipped under assets/kilt; this is for
    rebuilding that inventory from the source.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    out = os.path.join(ROOT, "datasets", "kilt")
    os.makedirs(out, exist_ok=True)
    for name in ("trex_train_0.parquet", "trex_train_1.parquet",
                 "trex_train_2.parquet", "trex_validation.parquet"):
        split = "validation" if "validation" in name else "train"
        path = hf_hub_download("facebook/kilt_tasks", f"trex/{split}-"
                               + ("00000-of-00001" if split == "validation"
                                  else f"{name.split('_')[-1][0]}0000-of-00003")
                               + ".parquet", repo_type="dataset")
        shutil.copy2(path, os.path.join(out, name))
        print(f"  datasets/kilt/{name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trex", action="store_true", help="also download KILT T-REx")
    args = p.parse_args()
    print("datasets/"); datasets()
    print("workspace/"); workspace()
    if args.trex:
        print("KILT T-REx"); trex()


if __name__ == "__main__":
    sys.exit(main())
