#!/usr/bin/env bash
# The defense sweep on PathRAG and HippoRAG 2: six settings, two benchmarks,
# three independent rounds, attack side and utility side.
#
#     bash benchmark/sweep_defenses.sh pathrag     # one victim, all rounds
#     bash benchmark/sweep_defenses.sh hipporag
#     bash benchmark/sweep_defenses.sh lightrag
#     VM=gemini-2.5-flash bash benchmark/sweep_defenses.sh pathrag   # another reader
#
# VM is the deployment's reader on both sides: the attacked reader and the
# QA reader. The attacker's LLM and the typed judge stay gpt-4o-mini, and
# HippoRAG keeps gpt-4o-mini as its index model (the OpenIE store is keyed on
# the builder). A non-default VM writes under `_vm-<model>` roots of its own
# and never touches the gpt-4o-mini cells.
#
# Resumable: a cell whose result file exists is skipped, so a killed sweep is
# restarted with the same command. One victim per invocation so the two can
# run as parallel lanes; within a lane everything is sequential, which keeps
# the API load the same as one hand-launched run.
#
# Settings follow the LightRAG cells already on disk, so the three victims are
# comparable: No defense and Protective carry no safe prompt, the other four
# carry it (Section VII-C-1 of Song et al.). Utility repeats bypass the
# response caches, as the LightRAG repeats did, or a repeat measures nothing.
#
# Output layout, read by defense_table.py:
#     runs/_defense/sweep/rep{r}/{ds}_{victim}_{defense}_{safe|nosafe}/
#     runs/_u_{ds}/defrep{r}/{pr|hp}/predictions_*.json
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
VICTIM=${1:?victim: lightrag, pathrag or hipporag}
case $VICTIM in lightrag) SHORT=lr ;; pathrag) SHORT=pr ;; hipporag) SHORT=hp ;; *) echo "unknown victim $VICTIM"; exit 1 ;; esac
VM=${VM:-gpt-4o-mini}
TAG=""; VMARGS=""
if [ "$VM" != gpt-4o-mini ]; then
  TAG="_vm-${VM//\//-}"; VMARGS="--victim_model $VM"
  [ "$VICTIM" = hipporag ] && VMARGS="$VMARGS --index_model gpt-4o-mini"
fi
ROOT_OUT=runs/_defense/sweep$TAG
LOGS=$ROOT_OUT/logs; mkdir -p "$LOGS"
stamp() { date +%H:%M:%S; }

# defense:safe-or-not
SETTINGS="none:nosafe abstractive:safe extractive:safe id_align:safe decoy:safe protective:nosafe"

# Two passes: the second only reaches cells the first left without a
# result, which after a provider rate-limit is the usual way a cell fails.
for pass in 1 2; do
for r in 1 2 3; do
  for ds in 2wiki100 hotpotqa100; do
    case $ds in
      2wiki100)   BM=2wiki:dev;    SUBSET=2wiki_min100;    ATTACK_BM=2wiki100-dp-delta:original ;;
      hotpotqa100) BM=hotpotqa:dev; SUBSET=hotpotqa_min100; ATTACK_BM=hotpotqa100-dp-delta:original ;;
    esac
    for setting in $SETTINGS; do
      d=${setting%%:*}; safe=${setting##*:}
      # Project rule: a privacy baseline ("no defense", "no-DP") is the
      # identity rewrite, never the original corpus; defended cells run on
      # the original index, as a deployment would. Utility baselines are the
      # original corpus, so the utility side below never uses identity.
      if [ "$d" = none ]; then
        INDEX=workspace/u_${ds}_${SHORT}_identity_inplace
        [ "$VICTIM" = lightrag ] && INDEX=workspace/u_${ds}_lr_identity_inplace/${ds}-dp-delta_identity_inplace/single/all
        SRC="--source sanitized --sanitized workspace/dp_${ds}/sanitized/identity_inplace.json"
        cell=${ds}_${VICTIM}_${d}_${safe}_identity
      else
        INDEX=workspace/u_${ds}_${SHORT}_original
        [ "$VICTIM" = lightrag ] && INDEX=workspace/u_${ds}_lr_original/${ds}-dp-delta_original/single/all
        SRC="--source original"
        cell=${ds}_${VICTIM}_${d}_${safe}
      fi
      UINDEX=workspace/u_${ds}_${SHORT}_original   # utility: always the original corpus
      [ "$VICTIM" = lightrag ] && UINDEX=workspace/u_${ds}_lr_original/${ds}-dp-delta_original/single/all

      # ---- attack side
      out=$ROOT_OUT/rep$r/$cell
      if [ -f "$out/naive_rtype.json" ]; then
        echo "$(stamp) skip attack  rep$r $cell"
      else
        echo "$(stamp) attack       rep$r $cell"
        extra=""; [ "$safe" = nosafe ] && extra="--no_safe_prompt"
        [ "$VICTIM" = hipporag ] && extra="$extra --benchmark $ATTACK_BM"
        DP_SCOPE=$ds $PY benchmark/attack_defended.py --victim "$VICTIM" \
          --index "$INDEX" --defense "$d" $SRC --seed 0 $VMARGS \
          --out_dir "$out" $extra > "$LOGS/attack_rep${r}_$cell.log" 2>&1 \
          || echo "$(stamp) !! attack failed: rep$r $cell (see $LOGS)"
      fi

      # ---- utility side
      udir=runs/_u_$ds/defrep$r$TAG/$SHORT
      suffix=""; [ "$d" != none ] && suffix="_def-$d"
      [ "$safe" = safe ] && suffix="${suffix}_safe"
      case $VICTIM in
        lightrag) pred="$udir/predictions_${ds}-dp-delta_original_hybrid_top10${suffix}.json" ;;
        pathrag)  pred="$udir/predictions_${BM/:/_}_pathrag_top10${suffix}.json" ;;
        hipporag) pred="$udir/predictions_${BM/:/_}_hipporag2_top5${suffix}.json" ;;
      esac
      if [ -f "$pred" ]; then
        echo "$(stamp) skip utility rep$r $cell"
        continue
      fi
      echo "$(stamp) utility      rep$r $cell"
      extra="--defense $d"; [ "$safe" = safe ] && extra="$extra --safe_prompt"
      mkdir -p "$udir"
      case $VICTIM in
        lightrag)
          # the *-dp-delta:original loader holds the whole dev split; the
          # hundred questions every method answers are the --subset
          $PY benchmark/run_bench.py --benchmark "${ds}-dp-delta:original" \
            --subset "$SUBSET" \
            --topology single --mode hybrid --top_k 10 --index "$UINDEX" \
            --llm_model "$VM" --no_llm_cache --out_dir "$udir" $extra \
            > "$LOGS/utility_rep${r}_$cell.log" 2>&1 \
            || echo "$(stamp) !! utility failed: rep$r $cell" ;;
        pathrag)
          $PY baselines/run_pathrag.py --benchmark "$BM" --subset "$SUBSET" \
            --passages_from_questions --workspace "$UINDEX" --top_k 10 \
            --llm_model "$VM" --no_cache --out_dir "$udir" $extra \
            > "$LOGS/utility_rep${r}_$cell.log" 2>&1 \
            || echo "$(stamp) !! utility failed: rep$r $cell" ;;
        hipporag)
          $PY baselines/run_hipporag.py --benchmark "$BM" --subset "$SUBSET" \
            --passages_from_questions --save_dir "$UINDEX" --qa_top_k 5 \
            ${TAG:+--qa_model "$VM" --pin_answer_format} \
            --qa_cache_tag "def_${d}_${safe}_rep$r" --out_dir "$udir" $extra \
            > "$LOGS/utility_rep${r}_$cell.log" 2>&1 \
            || echo "$(stamp) !! utility failed: rep$r $cell" ;;
      esac
    done
  done
done
done
echo "$(stamp) lane $VICTIM done"
