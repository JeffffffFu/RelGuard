#!/usr/bin/env bash
# GraphSteal against every defended victim: the second attack column of the
# defense table. Same six settings, two benchmarks and three rounds as
# sweep_defenses.sh, and the same GraphSteal configuration as the privacy
# table's rows (targeted DFS over the 15 GRASP targets, 75 turns, predicate
# command, degree order, top_k 40, gold from the extraction trace), so a
# defended cell is comparable with the DP cells already on disk.
#
#     bash benchmark/sweep_graphsteal.sh lightrag
#     bash benchmark/sweep_graphsteal.sh pathrag
#     bash benchmark/sweep_graphsteal.sh hipporag
#     VM=gemini-2.5-flash bash benchmark/sweep_graphsteal.sh pathrag   # another reader
#
# VM is the deployment's reader (the target model). The attacker's own LLM and
# the typed judge stay gpt-4o-mini whatever VM is, so a delta between readers
# is attributable. A non-default VM writes under a `_vm-<model>` root of its
# own, and HippoRAG keeps gpt-4o-mini as its index model, because the OpenIE
# store and save_dir are keyed on the builder and a different name would
# trigger a re-extraction.
#
# Resumable: a cell with naive_rtype.json is skipped. The utility side is not
# run here -- it is the same utility as sweep_defenses.sh measured, and
# defense_table.py reads it from there.
#
# Output: runs/_graphsteal/defense/rep{r}/{ds}_{victim}_{defense}_{safe|nosafe}/
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
VICTIM=${1:?victim: lightrag, pathrag or hipporag}
VM=${VM:-gpt-4o-mini}
TAG=""; VMARGS=""
if [ "$VM" != gpt-4o-mini ]; then
  TAG="_vm-${VM//\//-}"; VMARGS="--victim_model $VM"
  [ "$VICTIM" = hipporag ] && VMARGS="$VMARGS --index_model gpt-4o-mini"
fi
ROOT_OUT=runs/_graphsteal/defense$TAG
LOGS=$ROOT_OUT/logs; mkdir -p "$LOGS"
stamp() { date +%H:%M:%S; }
SETTINGS="none:nosafe abstractive:safe extractive:safe id_align:safe decoy:safe protective:nosafe"

# Two passes: the second only reaches cells the first left without a
# result, which after a provider rate-limit is the usual way a cell fails.
for pass in 1 2; do
for r in 1 2 3; do
  for ds in 2wiki100 hotpotqa100; do
    for setting in $SETTINGS; do
      d=${setting%%:*}; safe=${setting##*:}
      # Project rule: the privacy baseline (`none`) is the identity rewrite,
      # never the original corpus; defended cells run on the original index.
      if [ "$d" = none ]; then corpus=identity_inplace; cell=${ds}_${VICTIM}_${d}_${safe}_identity
      else corpus=original; cell=${ds}_${VICTIM}_${d}_${safe}; fi
      case $VICTIM in
        lightrag) INDEX=workspace/u_${ds}_lr_${corpus}/${ds}-dp-delta_${corpus}/single/all; extra="" ;;
        pathrag)  INDEX=workspace/u_${ds}_pr_${corpus}; extra="" ;;
        hipporag) INDEX=workspace/u_${ds}_hp_${corpus}; extra="--benchmark ${ds}-dp-delta:${corpus}" ;;
      esac
      SRC="--source original"; [ "$d" = none ] && SRC="--source sanitized --sanitized workspace/dp_${ds}/sanitized/identity_inplace.json"
      out=$ROOT_OUT/rep$r/$cell
      if [ -f "$out/naive_rtype.json" ]; then
        echo "$(stamp) skip   rep$r $cell"; continue
      fi
      echo "$(stamp) attack rep$r $cell"
      sp=""; [ "$safe" = nosafe ] && sp="--no_safe_prompt"
      DP_SCOPE=$ds $PY benchmark/graphsteal.py --victim "$VICTIM" --index "$INDEX" \
        --targets datasets/subsets/targets_$ds.json --turns 75 --command predicate \
        --order degree --top_k 40 --gold trace --seed 0 \
        --defense "$d" $sp $extra $VMARGS --out_dir "$out" \
        > "$LOGS/attack_rep${r}_$cell.log" 2>&1 \
        || { echo "$(stamp) !! attack failed: rep$r $cell"; continue; }
      echo "$(stamp) score  rep$r $cell"
      DP_SCOPE=$ds $PY benchmark/score_graphsteal.py --run "$out" $SRC \
        ${VM:+--victim_model $VM} \
        > "$LOGS/score_rep${r}_$cell.log" 2>&1 \
        || echo "$(stamp) !! score failed: rep$r $cell"
    done
  done
done
done
echo "$(stamp) lane $VICTIM done"
