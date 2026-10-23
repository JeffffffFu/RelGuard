"""Where one corpus's artefacts live.

Everything this pipeline caches -- the extraction, the type map, the candidate
pools, the verbalisations, the spans, the mechanism traces, the sanitised
corpora -- is a function of one corpus and of nothing else. Two corpora
therefore cannot share a file: the pools are keyed on the type pairs that
corpus's entities produce, `--build` rewrites the whole file rather than
merging, and a second dataset's build would silently drop the first one's
pairs. The HotpotQA numbers already in the paper were one `--build` away from
being unreproducible.

So the directory carries the scope, read from DP_SCOPE:

    DP_SCOPE=2wiki python -m dp.mechanism --epsilon 1
      ->  workspace/dp_2wiki/perturbed/eps1_seed0.json

The default is empty, which resolves to workspace/dp/ -- exactly where the
HotpotQA pilot already is, so nothing moves and no path in it changes.

It is an environment variable and not a flag because eight modules read these
paths, three of them are imported as libraries with no CLI of their own, and a
flag would have to be threaded through all eight to reach them. A scope is a
property of the run, like a working directory, not an argument to one step.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCOPE = os.environ.get("DP_SCOPE", "").strip()
DIR = os.path.join(ROOT, "workspace", "dp" + (f"_{SCOPE}" if SCOPE else ""))


def under(*parts: str) -> str:
    """A path inside this scope's directory."""
    return os.path.join(DIR, *parts)
