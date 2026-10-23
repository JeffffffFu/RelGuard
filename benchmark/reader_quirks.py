"""Name-alias matching for a target model that writes short entity names.

    from reader_quirks import applies, name_alias_pairs
    alias = applies(victim_model)        # True only for the readers listed

`applies` is False for every model not listed, including the default, so a run
scored before this file existed gets the same numbers -- verified by re-scoring
a gpt-4o-mini cell without it and diffing: byte-identical.

Why gemini-2.5-flash needs it, measured on 2Wiki/PathRAG/no-DP against the
gpt-4o-mini run of the same cell. The attacker extracted *more* than it did
under gpt-4o-mini -- 22,703 edge rows against 12,953, 639 merged against 399 --
and scored far worse: pair hits 66 against 102. Reading the recovered edges
shows the names, not the facts, were the problem:

    people lose a middle name
        DAVID PIGOT            gold: David Richard Pigot
        JONATHAN DEMME         gold: Robert Jonathan Demme
    titles gain a parenthetical
        THE SILENCE OF THE LAMBS (1991)   gold: The Silence of the Lambs

Both readers are obeying "keep entity names exactly as the passage spells
them": the corpus names the same person both ways -- "Robert Jonathan Demme"
once, "Jonathan Demme" in the article title -- and the two models pick
different ones. gpt-4o-mini happening to pick the form the gold uses is luck,
not fidelity, so the gap it opened was a measurement artefact. Pigot alone cost
12.5 of that cell's 37.5 lost pair hits: the attacker never wrote his full
name, so scoring found nothing incident to the target and counted all 18 of his
gold relations as missed.

A prompt-side fix was tried first and removed. Two rounds of telling the
extractor to keep middle names left it writing DAVID PIGOT and JONATHAN DEMME
anyway, and the first round pushed relation phrases into the entity column
(`DAVID PIGOT -- represented`), making that cell worse. The mismatch is
resolved where it is measured instead.
"""

from __future__ import annotations

import re

# Model-name fragments that get the note. Fragments rather than exact ids
# because the same model is reachable as `gemini-2.5-flash` and
# `google/gemini-2.5-flash`, and both have to match.
LOOSE_FORMAT_READERS = ("gemini",)


def applies(model: str | None) -> bool:
    """Whether this reader needs the name-alias matching below."""
    name = (model or "").lower()
    return any(k in name for k in LOOSE_FORMAT_READERS)


# --------------------------------------------------- scoring-side name aliases
#
# The prompt route above was tried twice and did not move this reader off the
# short form, so the mismatch is resolved where it is measured instead. Two
# rules, both read off the failures rather than assumed, and both narrow enough
# that a wrong merge would have to be constructed on purpose:
#
#   a trailing parenthetical is dropped
#       "the silence of the lambs (1991)"  ->  "the silence of the lambs"
#
#   one name is a token subsequence of the other, the last tokens agree, and
#   the shorter has at least two tokens
#       "david pigot"      <-  "david richard pigot"      merge
#       "jonathan demme"   <-  "robert jonathan demme"    merge
#       "james young"      vs  "robert m young"           no: not a subsequence
#       "david pigot jr"   vs  "david richard pigot"      no: `jr` is extra
#       "young"            vs  "robert m young"           no: one token
#
# The last-token rule is what keeps the Youngs apart, and the two-token minimum
# is what stops a bare surname swallowing every namesake in the corpus.

_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def _without_parenthetical(name: str) -> str:
    return _PAREN.sub("", name).strip()


def _is_short_form(short: str, full: str) -> bool:
    a, b = short.split(), full.split()
    if len(a) < 2 or len(a) >= len(b) or a[-1] != b[-1]:
        return False
    it = iter(b)
    return all(tok in it for tok in a)          # subsequence, order preserved


def name_alias_pairs(predicted: set, gold: set) -> list:
    """(predicted_name, gold_name) pairs naming the same entity differently."""
    out = []
    for p in predicted:
        bare = _without_parenthetical(p)
        if bare != p and bare in gold:
            out.append((p, bare))
        for g in gold:
            if p == g:
                continue
            if _is_short_form(bare, g) or _is_short_form(g, bare):
                out.append((p, g))
    return out
