"""Build the defense table from the runs on disk.

    python benchmark/defense_table.py
    python benchmark/defense_table.py --out runs/defense_3run_mean.xlsx
    python benchmark/defense_table.py --victim_model gemini-2.5-flash

With --victim_model, every cell comes from the `_vm-<model>` roots the two
sweep scripts write for that reader, including the baseline row: the
sweeps' own `none` cells, which attack the identity index, and their `none`
utility runs on the original corpus -- the same rule as the default table,
read from the sweep rather than from the privacy and utility tables, which
hold gpt-4o-mini only.

Both metrics for both benchmarks, one block of rows per victim GraphRAG, means
over three independent runs. Nothing is typed in: every number is read back
from the run it came from, so the sheet can be checked against the runs it
summarises.

Two conventions, and they differ on purpose:

    Naive F1, RType F1    rebuilt from the averaged P and R (F1-of-macro),
                          not averaged over the three runs' F1 values
    QA F1                 a plain average

F1 is not linear in P and R, so averaging three F1 values is not the F1 of the
averaged P and R. The privacy numbers follow the attack's own summary, which
reports P and R and derives F1 from them; the utility number has no P and R to
rebuild from.

Per-target scores are padded to the full target list, not averaged over the
targets that scored: a target the attack recovered nothing for is a zero, and
dropping it would let a defense that silences a target entirely score as if
that target had never been attacked. This matches attack_defended.py's
`padded to 15`, which is the convention it says to quote.

Adding a victim is adding entries to CELLS. A row whose runs are not on disk
yet comes out blank and is listed on stderr, so a half-finished sweep still
builds a readable sheet.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))

from score_hotpot import score  # noqa: E402

# The project's two baselines, fixed. A row standing for "no DP" or "no
# defense" in any privacy or attack table is the identity rewrite -- the
# corpus passed through the DP write-back with every relation released
# unchanged -- and in any utility table it is the original, untouched corpus.
# Stated by the user as a rule on 2026-10-20 and not a choice this file makes:
# the sweep scripts attack the identity index for their `none` setting, and
# this table reads its baseline row from the privacy and utility tables, which
# follow the same rule. Do not add an original-corpus baseline here.
PRIVACY_BASELINE = "identity"
UTILITY_BASELINE = "original"

VM_TAG = ""   # "_vm-<model>" when a non-default reader is being tabulated

ATTACK = os.path.join(ROOT, "runs", "_defense")
PRIVACY_TABLE = os.path.join(ROOT, "runs", "privacy_3run_mean.xlsx")
GRASP_IDENTITY = os.path.join(ROOT, "runs", "_grasp")
GS_IDENTITY = os.path.join(ROOT, "runs", "_graphsteal")
GRAPHSTEAL = os.path.join(ROOT, "runs", "_graphsteal", "defense")
"""The second attack. benchmark/sweep_graphsteal.sh names every cell
rep{r}/{ds}_{victim}_{mode}_{safe}, for all three victims alike, and
score_graphsteal.py writes the same naive_rtype.json the GRASP cells carry."""
UTILITY = {"2wiki": os.path.join(ROOT, "runs", "_u_2wiki100"),
           "hotpot": os.path.join(ROOT, "runs", "_u_hotpotqa100")}

N_TARGETS = 15
"""What the per-target scores are padded to. See the module docstring."""

DATASETS = [("2wiki", "2WikiMultiHopQA"), ("hotpot", "HotpotQA")]

DEFENSES = ["no-DP (identity)", "Abstractive", "Extractive",
            "ID Alignment", "Decoy", "Protective Sys. Prompt"]
"""Row order. The first row is the baseline every delta is taken against, and
it is the privacy table's no-DP row, not a defended-sweep cell: the identity
rewrite, i.e. the corpus passed through the DP write-back with every relation
released unchanged, attacked with no defense installed. That is what the DP
rows in privacy_3run_mean.xlsx are measured against, so a baseline defense and
an epsilon can be read off the same reference. Its three runs come from

    GRASP        run 1 from runs/_grasp/atk_*_identity_inplace.json; runs 2
                 and 3 exist only as the P and R the privacy table's Runs
                 sheet records (its note: "Run 1 is the artefact on disk"),
                 so they are read from that sheet
    GraphSteal   runs/_graphsteal/{,rep2/,rep3/}*_identity_inplace
    utility      the utility table's no-DP runs, which are the original
                 corpus (the identity rewrite is the attack baseline, the
                 original corpus the utility baseline; utility_3run_mean.xlsx
                 says the same)

The defended cells themselves were run on the original index. The deltas
therefore include whatever the identity rewrite does to the attack on its
own, which on LightRAG is not small (HotpotQA GRASP Naive: original 79 vs
identity 95, run 1); the sweep's own original-corpus No defense cells remain
under runs/_defense/sweep and runs/_graphsteal/defense for the other reading.

Abstractive, Extractive, ID Alignment and Decoy all carry the project's
appended safe prompt, per Section VII-C-1 of Song et al. (arXiv:2602.06495).
Protective Sys. Prompt is run without it, being the same idea prepended, and
stacking the two would confound which one acted. So those four deltas include
the safe prompt's own effect and this table does not isolate it; measured on
its own it was -1.61 Naive F1 and -5.30 RType F1 on 2Wiki/LightRAG.
"""

VICTIMS = ["LightRAG", "PathRAG", "HippoRAG 2"]

# (victim, dataset, defense) -> three (attack cell, QA predictions) pairs, one
# per run, relative to ATTACK and UTILITY[dataset].
#
# The paths are irregular and are written out rather than derived. Run 1 came
# from attack_defended.py, which names a cell after the scope, the extraction
# schema, the victim and the defense; runs 2 and 3 came from a repeat driver
# that names it after the dataset and the defense alone. Deriving one spelling
# from the other would be a rule with two cases and no third user, and a wrong
# guess here reads another cell's scores as this one's -- which is the mistake
# the checkpoint-key comment in attack_defended.py records.
_LR2, _LRH = "2wiki100_s5_ee3d10b8_lightrag", "hotpotqa100_s5_4f0b8bd3_lightrag"
_P2 = "predictions_2wiki100-dp-delta_original_hybrid_top10"
_PH = "predictions_hotpotqa100-dp-delta_original_hybrid_top10"

CELLS: dict[tuple[str, str, str], list[tuple[str, str]]] = {}


def _lightrag(dataset: str, defense: str, attack_tag: str, rep_tag: str,
              qa_suffix: str) -> None:
    cell, pred = (_LR2, _P2) if dataset == "2wiki" else (_LRH, _PH)
    stem = "2wiki100" if dataset == "2wiki" else "hotpotqa100"
    CELLS[("LightRAG", dataset, defense)] = [
        (f"{cell}_def-{attack_tag}_original", f"{pred}{qa_suffix}.json"),
        (f"rep2/{stem}_{rep_tag}", f"defrep2/{pred}{qa_suffix}.json"),
        (f"rep3/{stem}_{rep_tag}", f"defrep3/{pred}{qa_suffix}.json"),
    ]


# PathRAG and HippoRAG 2 came later, from benchmark/sweep_defenses.sh, which
# names every round the same way; the asymmetry with LightRAG above is the
# history, not a choice.
_SETTINGS = [("no-DP (identity)", "none", "nosafe"), ("Abstractive", "abstractive", "safe"),
             ("Extractive", "extractive", "safe"), ("ID Alignment", "id_align", "safe"),
             ("Decoy", "decoy", "safe"), ("Protective Sys. Prompt", "protective", "nosafe")]


def _no_dp_utility(victim: str, ds: str) -> list[str]:
    """The utility table's three no-DP runs: original corpus, one per rep."""
    bm = "2wiki_dev" if ds == "2wiki" else "hotpotqa_dev"
    stem = "2wiki100" if ds == "2wiki" else "hotpotqa100"
    if victim == "LightRAG":
        name = f"predictions_{stem}-dp-delta_original_hybrid_top10.json"
        return [name, f"rep2/{name}", f"rep3/{name}"]
    sub, tag = (("pr_original", "pathrag_top10") if victim == "PathRAG"
                else ("hp_original", "hipporag2_top5"))
    name = f"predictions_{bm}_{tag}.json"
    return [f"{sub}/{name}", f"rep2/{sub}/{name}", f"rep3/{sub}/{name}"]


def _swept(victim: str, short: str, pred_tag: str) -> None:
    for ds, _ in DATASETS:
        stem = "2wiki100" if ds == "2wiki" else "hotpotqa100"
        bm = "2wiki_dev" if ds == "2wiki" else "hotpotqa_dev"
        for defense, mode, safe in _SETTINGS:
            if defense == DEFENSES[0] and not VM_TAG:
                CELLS[(victim, ds, defense)] = [
                    (None, pred) for pred in _no_dp_utility(victim, ds)]
                continue
            if defense == DEFENSES[0]:
                # the sweep's own baseline: identity attack cells, original
                # utility, both under the reader's tagged roots
                CELLS[(victim, ds, defense)] = [
                    (None, f"defrep{r}{VM_TAG}/{pred_tag[0]}/predictions_"
                           + (f"{stem}-dp-delta_original_hybrid_top10.json"
                              if victim == "LightRAG" else
                              f"{bm}_{pred_tag[1]}.json"))
                    for r in (1, 2, 3)]
                continue
            suffix = ("" if mode == "none" else f"_def-{mode}") \
                     + ("_safe" if safe == "safe" else "")
            pred = (f"{stem}-dp-delta_original_hybrid_top10{suffix}.json"
                    if victim == "LightRAG" else f"{bm}_{pred_tag[1]}{suffix}.json")
            CELLS[(victim, ds, defense)] = [
                (f"sweep{VM_TAG}/rep{r}/{stem}_{short}_{mode}_{safe}",
                 f"defrep{r}{VM_TAG}/{pred_tag[0]}/predictions_{pred}")
                for r in (1, 2, 3)]


def build_cells() -> None:
    CELLS.clear()
    _swept("PathRAG", "pathrag", ("pr", "pathrag_top10"))
    _swept("HippoRAG 2", "hipporag", ("hp", "hipporag2_top5"))
    if VM_TAG:
        _swept("LightRAG", "lightrag", ("lr", "hybrid_top10"))
        return
    for _ds, _ in DATASETS:
        CELLS[("LightRAG", _ds, DEFENSES[0])] = [
            (None, pred) for pred in _no_dp_utility("LightRAG", _ds)]
        _lightrag(_ds, "Abstractive", "abstractive", "abstractive_safe",
                  "_def-abstractive_safe")
        _lightrag(_ds, "Extractive", "extractive", "extractive_safe",
                  "_def-extractive_safe")
        _lightrag(_ds, "ID Alignment", "id_align", "id_align_safe",
                  "_def-id_align_safe")
        _lightrag(_ds, "Decoy", "decoy", "decoy_safe", "_def-decoy_safe")
        _lightrag(_ds, "Protective Sys. Prompt", "protective_nosafe",
                  "protective_nosafe", "_def-protective")


build_cells()



def graphsteal_cell(victim: str, dataset: str, defense: str, run: int) -> str:
    short = {"LightRAG": "lightrag", "PathRAG": "pathrag",
             "HippoRAG 2": "hipporag"}[victim]
    stem = "2wiki100" if dataset == "2wiki" else "hotpotqa100"
    mode, safe = {d: (m, sf) for d, m, sf in _SETTINGS}[defense]
    return f"rep{run}/{stem}_{short}_{mode}_{safe}"


_SHEET_CACHE: dict = {}


def privacy_no_dp(victim: str, dataset: str, run: int) -> dict | None:
    """The privacy table's no-DP row for one run: GRASP and GraphSteal P and R.

    Read from the Runs sheet, which is the only place runs 2 and 3 of GRASP
    exist. Values are as typed there, two decimals; run 1 agrees with the
    per-target artefact to that precision.
    """
    if not _SHEET_CACHE:
        import openpyxl
        ws = openpyxl.load_workbook(PRIVACY_TABLE, data_only=True)["Runs"]
        label = {"2WikiMultiHopQA": "2wiki", "HotpotQA": "hotpot"}
        current = None
        for row in ws.iter_rows(min_row=4, values_only=True):
            if row[0] in label:
                current = label[row[0]]
            if row[2] != "no-DP" or row[1] not in VICTIMS:
                continue
            _SHEET_CACHE[(row[1], current, int(row[3]))] = {
                "naive_p": float(row[4]), "naive_r": float(row[5]),
                "rtype_p": float(row[6]), "rtype_r": float(row[7]),
                "gs_naive_p": float(row[8]), "gs_naive_r": float(row[9]),
                "gs_rtype_p": float(row[10]), "gs_rtype_r": float(row[11])}
    return _SHEET_CACHE.get((victim, dataset, run))


def identity_cells(victim: str, dataset: str, run: int) -> tuple[str, str]:
    """(GRASP summary file, GraphSteal cell dir) for the identity rewrite."""
    short = {"LightRAG": "lightrag", "PathRAG": "pathrag",
             "HippoRAG 2": "hipporag"}[victim]
    stem, schema = (("2wiki100", "s5_ee3d10b8") if dataset == "2wiki"
                    else ("hotpotqa100", "s5_4f0b8bd3"))
    grasp = os.path.join(GRASP_IDENTITY,
                         f"atk_{stem}_{schema}_{short}_identity_inplace.json")
    rep = "" if run == 1 else f"rep{run}/"
    gs = os.path.join(GS_IDENTITY, f"{rep}{stem}_{short}_identity_inplace")
    return grasp, gs


def no_dp_scores(victim: str, dataset: str, run: int) -> dict | None:
    """The baseline row's attack scores for one run: the privacy table's.

    The sheet is read first and the artefacts only fill what it lacks, so
    this row *is* the privacy table's no-DP row, to the two decimals typed
    there. Checked against the artefacts: GRASP run 1 agrees exactly, and the
    GraphSteal cells agree except LightRAG/2Wiki run 3, which scored 14 of 15
    targets and which the sheet averages over the 14 (Naive P 82.02) where
    this table's own cells pad to 15 (76.55). The sheet's value is kept, so
    the row stays the privacy table's.
    """
    out = dict(privacy_no_dp(victim, dataset, run) or {})
    grasp_file, gs_dir = identity_cells(victim, dataset, run)
    if os.path.exists(grasp_file):
        rows = json.load(open(grasp_file, encoding="utf-8"))
        for k in ("naive_p", "naive_r", "rtype_p", "rtype_r"):
            out.setdefault(k, sum(r[k] for r in rows) / N_TARGETS)
    gs = attack_scores(os.path.basename(gs_dir), os.path.dirname(gs_dir))
    for k, v in (gs or {}).items():
        out.setdefault(f"gs_{k}", v)
    return out or None


def sweep_baseline(victim: str, dataset: str, run: int) -> dict | None:
    """The baseline row from the sweeps' own `none` cells (identity index)."""
    short = {"LightRAG": "lightrag", "PathRAG": "pathrag",
             "HippoRAG 2": "hipporag"}[victim]
    stem = "2wiki100" if dataset == "2wiki" else "hotpotqa100"
    cell = f"rep{run}/{stem}_{short}_none_nosafe_identity"
    grasp = attack_scores(cell, os.path.join(ATTACK, "sweep" + VM_TAG))
    gs = attack_scores(cell, GRAPHSTEAL + VM_TAG)
    if grasp is None and gs is None:
        return None
    out = dict(grasp or {})
    out.update({f"gs_{k}": v for k, v in (gs or {}).items()})
    return out


def attack_scores(cell: str, root: str = ATTACK) -> dict | None:
    """Padded Naive and RType P and R for one attack cell, or None if absent."""
    path = os.path.join(root, cell, "naive_rtype.json")
    if not os.path.exists(path):
        return None
    rows = json.load(open(path, encoding="utf-8"))
    return {k: sum(r[k] for r in rows) / N_TARGETS
            for k in ("naive_p", "naive_r", "rtype_p", "rtype_r")}


def qa_f1(dataset: str, predictions: str) -> float | None:
    """Overall token F1 for one utility run, or None if absent.

    `score` prints its own per-question-type table, which is noise when it is
    called 36 times; it is swallowed and only the overall F1 is kept.
    """
    path = os.path.join(UTILITY[dataset], predictions)
    if not os.path.exists(path):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        return score(path)["overall"]["f1"]


def collect() -> tuple[dict, list[str]]:
    """Read every run named in CELLS. Returns the per-run rows and what is missing."""
    runs, missing = {}, []
    for (victim, dataset, defense), triple in CELLS.items():
        got = []
        for i, (cell, predictions) in enumerate(triple, 1):
            utility = qa_f1(dataset, predictions)
            if cell is None:
                base = (sweep_baseline(victim, dataset, i) if VM_TAG
                        else no_dp_scores(victim, dataset, i))
                if base is None:
                    missing.append(f"{victim}/{dataset}/{defense} run{i}: "
                                   f"no identity baseline")
                if utility is None:
                    missing.append(f"{victim}/{dataset}/{defense} run{i}: "
                                   f"no predictions {predictions}")
                got.append(None if base is None else {**base, "qa_f1": utility})
                continue
            scores = attack_scores(cell)
            gs_cell = graphsteal_cell(victim, dataset, defense, i)
            gs = attack_scores(gs_cell, GRAPHSTEAL + VM_TAG)
            if scores is None:
                missing.append(f"{victim}/{dataset}/{defense} run{i}: "
                               f"no GRASP cell {cell}")
            if gs is None:
                missing.append(f"{victim}/{dataset}/{defense} run{i}: "
                               f"no GraphSteal cell {gs_cell}")
            if utility is None:
                missing.append(f"{victim}/{dataset}/{defense} run{i}: "
                               f"no predictions {predictions}")
            if scores is None and gs is None:
                got.append(None)
                continue
            row = {"qa_f1": utility}
            row.update(scores or {})
            row.update({f"gs_{k}": v for k, v in (gs or {}).items()})
            got.append(row)
        runs[(victim, dataset, defense)] = got
    return runs, missing


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def means(runs: dict, victim: str, dataset: str, defense: str) -> dict | None:
    """The three runs reduced to one row, or None if no run scored."""
    got = [r for r in runs.get((victim, dataset, defense), []) if r]
    if not got:
        return None
    def macro(prefix: str):
        have = [r for r in got if f"{prefix}naive_p" in r]
        if not have:
            return None
        avg = lambda k: statistics.mean(r[k] for r in have)
        return (f1(avg(f"{prefix}naive_p"), avg(f"{prefix}naive_r")),
                f1(avg(f"{prefix}rtype_p"), avg(f"{prefix}rtype_r")))
    grasp, gs = macro(""), macro("gs_")
    qa = [r["qa_f1"] for r in got if r["qa_f1"] is not None]
    return {"naive_f1": grasp[0] if grasp else None,
            "rtype_f1": grasp[1] if grasp else None,
            "gs_naive_f1": gs[0] if gs else None,
            "gs_rtype_f1": gs[1] if gs else None,
            "qa_f1": statistics.mean(qa) if qa else None,
            "n": len(got)}


# ----------------------------------------------------------------- the sheet

HEAD_A = "FF44546A"
HEAD_B = "FF2E75B6"
STRIPE = "FFEDF2F9"
DELTA_FMT = r"\+0.00;\-0.00;0.00"


def write(path: str, runs: dict) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    wb = openpyxl.Workbook()
    mean = wb.active
    mean.title = "Mean"
    runs_ws = wb.create_sheet("Runs")

    bold = Font(bold=True)
    white = Font(bold=True, color="FFFFFFFF")
    under = Border(bottom=Side(style="thin", color="FFBFBFBF"))
    centre = Alignment(horizontal="center")
    stripe = lambda: PatternFill("solid", fgColor=STRIPE)

    # ---- Mean: victim, defense, then Naive/RType/QA and a delta for each,
    # per benchmark. Six metric columns per benchmark, twelve in all.
    mean["A1"] = (f"Defenses on three GraphRAG victims, reader "
                  f"{VM_TAG[4:] if VM_TAG else 'gpt-4o-mini'}. Mean of "
                  "three independent runs. Naive F1 and RType F1 lower is "
                  "better, QA F1 higher is better. Every Δ is against the "
                  "same victim's own No defense row.")
    mean.merge_cells("A1:V1")
    mean["A1"].font = bold

    mean["A3"], mean["B3"] = "Victim", "Defense"
    mean.merge_cells("A3:A4")
    mean.merge_cells("B3:B4")
    METRICS = ("naive_f1", "rtype_f1", "gs_naive_f1", "gs_rtype_f1", "qa_f1")
    HEADS = ("GRASP Naive F1", "Δ", "GRASP RType F1", "Δ",
             "GraphSteal Naive F1", "Δ", "GraphSteal RType F1", "Δ",
             "QA F1", "Δ")
    for col, (_, label), fill in (("C", DATASETS[0], HEAD_A),
                                  ("M", DATASETS[1], HEAD_B)):
        start = mean[f"{col}3"].column
        mean.cell(row=3, column=start, value=label)
        mean.merge_cells(start_row=3, start_column=start,
                         end_row=3, end_column=start + 9)
        for i, name in enumerate(HEADS):
            c = mean.cell(row=4, column=start + i, value=name)
            # three words in a ten-wide column: wrapped, or Excel clips
            # "GraphSteal Naive F1" to its first word and the column reads
            # as a duplicate of the GRASP one
            c.font = white
            c.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
            c.fill = PatternFill("solid", fgColor=fill)
        head = mean.cell(row=3, column=start)
        head.font, head.alignment = white, centre
        head.fill = PatternFill("solid", fgColor=fill)
    for addr in ("A3", "B3"):
        mean[addr].font, mean[addr].alignment = white, centre
        mean[addr].fill = PatternFill("solid", fgColor=HEAD_A)

    row = 5
    runs_row = 4
    blocks = []          # (victim, first Mean row) so deltas can target it
    for v_index, victim in enumerate(VICTIMS):
        base_row = row
        blocks.append((victim, base_row))
        for d_index, defense in enumerate(DEFENSES):
            mean.cell(row=row, column=1,
                      value=victim if d_index == 0 else None)
            mean.cell(row=row, column=2, value=defense)
            for col, (key, _) in zip(("C", "M"), DATASETS):
                start = mean[f"{col}{row}"].column
                got = means(runs, victim, key, defense)
                for i, metric in enumerate(METRICS):
                    value = mean.cell(row=row, column=start + 2 * i)
                    value.value = None if not got else got[metric]
                    value.number_format = "0.00"
                    value.alignment = centre
                    delta = mean.cell(row=row, column=start + 2 * i + 1)
                    letter = value.column_letter
                    delta.value = ("—" if row == base_row else
                                   f"={letter}{row}-{letter}${base_row}")
                    delta.number_format = DELTA_FMT
                    delta.alignment = centre
            for c in range(1, 23):
                cell = mean.cell(row=row, column=c)
                cell.border = under
                if row == base_row:
                    cell.font = bold
                elif d_index % 2:
                    cell.fill = stripe()
            row += 1
        row += 1 if v_index < len(VICTIMS) - 1 else 0

    # ---- Runs: every run behind every mean, in the order they are read
    runs_ws["A1"] = ("The three independent runs behind every mean. Naive and "
                     "RType P and R are padded to the full target list; QA F1 "
                     "is the overall token F1 of that run's predictions.")
    runs_ws.merge_cells("A1:M1")
    runs_ws["A1"].font = bold
    for i, name in enumerate(("victim", "benchmark", "defense", "run",
                              "GRASP Naive P", "GRASP Naive R",
                              "GRASP RType P", "GRASP RType R",
                              "GS Naive P", "GS Naive R",
                              "GS RType P", "GS RType R",
                              "QA F1"), start=1):
        c = runs_ws.cell(row=3, column=i, value=name)
        c.font, c.alignment = white, centre
        c.fill = PatternFill("solid", fgColor=HEAD_A)

    striped = False
    for victim in VICTIMS:
        for key, label in DATASETS:
            for defense in DEFENSES:
                triple = runs.get((victim, key, defense)) or []
                if not any(triple):
                    continue
                for i, got in enumerate(triple, 1):
                    if not got:
                        continue
                    runs_ws.cell(row=runs_row, column=1,
                                 value=victim if i == 1 and defense == DEFENSES[0] else None)
                    runs_ws.cell(row=runs_row, column=2,
                                 value=label if i == 1 and defense == DEFENSES[0] else None)
                    runs_ws.cell(row=runs_row, column=3,
                                 value=defense if i == 1 else None)
                    runs_ws.cell(row=runs_row, column=4, value=i)
                    for j, k in enumerate(("naive_p", "naive_r",
                                           "rtype_p", "rtype_r",
                                           "gs_naive_p", "gs_naive_r",
                                           "gs_rtype_p", "gs_rtype_r",
                                           "qa_f1")):
                        c = runs_ws.cell(row=runs_row, column=5 + j,
                                         value=got.get(k))
                        c.number_format = "0.00"
                        c.alignment = centre
                    for c in range(1, 14):
                        cell = runs_ws.cell(row=runs_row, column=c)
                        cell.border = under
                        if striped:
                            cell.fill = stripe()
                    runs_row += 1
                striped = not striped

    # ---- the notes, which are part of the table and not commentary on it
    notes = [
        "Naive F1 and RType F1 are rebuilt from the averaged P and R "
        "(F1-of-macro), not averaged over the three runs' F1 values; QA F1 is "
        "a plain average. F1 is not linear in P and R, so the two conventions "
        "do not agree and mixing them would be the error.",
        "Two attacks, both scored on the same 15 targets against the same "
        "frozen gold: GRASP (benchmark/attack_defended.py) and GraphSteal in "
        "its targeted mode (benchmark/sweep_graphsteal.sh: 75 turns, predicate "
        "command, degree order, top_k 40), the configuration the privacy "
        "table's GraphSteal rows use. The utility column is shared.",
        "Naive scores an entity pair recovered at all; RType also requires the "
        "relation. Naive is therefore the ceiling and RType the attack this "
        "project reports, and a defense that moves Naive is removing the pair "
        "from the answer rather than blurring the predicate.",
        "Abstractive, Extractive, ID Alignment and Decoy all carry the "
        "project's appended safe prompt, per Section VII-C-1 of Song et al. "
        "(arXiv:2602.06495); Protective Sys. Prompt is run without it, being "
        "the same idea prepended, and stacking the two would confound which "
        "one acted. Every Δ is against No defense, which carries no prompt "
        "at all, so those four Δ include the safe prompt's own effect and "
        "this table no longer isolates it. On the runs behind it that effect "
        "was -1.61 Naive F1 and -5.30 RType F1 on 2Wiki/LightRAG.",
        "HippoRAG 2 builds no relation table, so its ID Alignment and Decoy "
        "rows are the HippoRAG branch in defenses_hipporag.py, written against "
        "what it does have: every passage labelled [Passage 1] with its title "
        "line replaced by a constant (the title is the handle an attacker "
        "cites across queries), and a Facts: line of two decoy triples under "
        "each passage, recombined from other retrieved passages' OpenIE "
        "entities and relation words with real pairs rejected, plus an "
        "internal-fields instruction. The reader is the same direct call every "
        "HippoRAG number on disk uses, DP rows included, so the rows are "
        "comparable with them; on the utility side ID Alignment changes "
        "nothing HippoRAG's own QA prompt shows, so that cell is the safe "
        "prompt alone.",
        "Setup: 15 targets, frozen gold filter, judge gpt-4o-mini, GRASP 5 "
        "queries per target. Utility is 100 questions per benchmark on the "
        "original corpus. Reader gpt-4o-mini, original (no-DP) text. Repeats "
        "reused run 1's index; attack caching is off by default and utility "
        "used --no_llm_cache.",
        "Generated by benchmark/defense_table.py from the runs under "
        "runs/_defense and runs/_u_*. Re-run it rather than editing cells, or "
        "the sheet stops being checkable against the runs.",
    ]
    row += 2
    for note in notes:
        mean.cell(row=row, column=1, value=note)
        mean.cell(row=row, column=1).alignment = Alignment(vertical="top")
        row += 2

    mean.column_dimensions["A"].width = 12
    mean.column_dimensions["B"].width = 23
    for col in "CDEFGHIJKLMNOPQRSTUV":
        mean.column_dimensions[col].width = 10
    mean.row_dimensions[4].height = 44
    mean.freeze_panes = "C5"
    runs_ws.column_dimensions["A"].width = 12
    runs_ws.column_dimensions["B"].width = 18
    runs_ws.column_dimensions["C"].width = 23
    runs_ws.column_dimensions["D"].width = 6
    for col in "EFGHIJKLM":
        runs_ws.column_dimensions[col].width = 10
    runs_ws.freeze_panes = "E4"

    wb.save(path)


def report(runs: dict) -> None:
    """The sheet as text, so a run can be checked without opening Excel."""
    for victim in VICTIMS:
        rows = [(d, means(runs, victim, DATASETS[0][0], d),
                 means(runs, victim, DATASETS[1][0], d)) for d in DEFENSES]
        if not any(a or b for _, a, b in rows):
            print(f"\n{victim}: no runs on disk")
            continue
        print(f"\n{victim}")
        width = 16
        header = "".join(f"{n:>{width}}" for n in
                         ("GRASP Naive", "GRASP RType", "GS Naive",
                          "GS RType", "QA F1") * 2)
        print(f"{'':24}{'2WikiMultiHopQA':^80}{'HotpotQA':^80}")
        print(f"{'':24}{header}")
        base = {k: means(runs, victim, k, DEFENSES[0]) for k, _ in DATASETS}
        for defense, a, b in rows:
            out = f"{defense:24}"
            for got, ref in ((a, base[DATASETS[0][0]]),
                             (b, base[DATASETS[1][0]])):
                for k in ("naive_f1", "rtype_f1", "gs_naive_f1",
                          "gs_rtype_f1", "qa_f1"):
                    if not got or got[k] is None:
                        out += f"{'-':>{width}}"
                        continue
                    text = f"{got[k]:.2f}"
                    if ref and ref[k] is not None and defense != DEFENSES[0]:
                        text += f" ({got[k] - ref[k]:+.2f})"
                    out += f"{text:>{width}}"
            print(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(ROOT, "runs",
                                                 "defense_3run_mean.xlsx"))
    p.add_argument("--text_only", action="store_true",
                   help="print the table and write nothing")
    p.add_argument("--victim_model", default="gpt-4o-mini",
                   help="the reader the cells were measured with; a "
                        "non-default value reads the `_vm-<model>` sweep "
                        "roots and writes to a file named after it")
    args = p.parse_args()
    global VM_TAG
    if args.victim_model != "gpt-4o-mini":
        VM_TAG = "_vm-" + args.victim_model.replace("/", "-")
        build_cells()
        if args.out == os.path.join(ROOT, "runs", "defense_3run_mean.xlsx"):
            args.out = os.path.join(ROOT, "runs",
                                    f"defense_3run_mean{VM_TAG}.xlsx")

    runs, missing = collect()
    report(runs)
    if missing:
        print(f"\n{len(missing)} run(s) not on disk:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
    if not args.text_only:
        write(args.out, runs)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
