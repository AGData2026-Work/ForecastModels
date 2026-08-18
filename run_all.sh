#!/usr/bin/env bash
# Full matrix: 2 builds x 2 architectures x 2 driver conventions = 8 runs.
#
#   bash run_all.sh                    # cpu, all cores
#   DEVICE=mps bash run_all.sh         # apple silicon gpu (benchmark first, see RUNBOOK)
#   THREADS=8 bash run_all.sh          # pin cpu threads
#   BUILDS="configs/build1.yaml" bash run_all.sh    # build 1 only
#
# RESUMABLE. A run whose predictions_paired.csv already exists is skipped, so an
# interrupted session can be restarted with the same command. Delete the run's
# folder to force a redo.
set -uo pipefail
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-0}"
BUILDS="${BUILDS:-configs/build1.yaml configs/build2.yaml}"
mkdir -p logs
FAILED=()
for cfg in $BUILDS; do
  BUILD=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$cfg'))['build']['name'])")
  for kind in RNN GRU; do
    for conv in unconditional conditional; do
      DEST="outputs/$BUILD/${kind}_${conv}"
      if [ -f "$DEST/predictions_paired.csv" ]; then
        echo "SKIP  $BUILD $kind $conv  (already complete)"
        continue
      fi
      LOG="logs/${BUILD}_${kind}_${conv}.log"
      echo "RUN   $BUILD $kind $conv on $DEVICE  -> $LOG"
      START=$(date +%s)
      if python3 src/run.py --config "$cfg" --kind "$kind" --convention "$conv" \
           --device "$DEVICE" --threads "$THREADS" 2>&1 | tee "$LOG"; then
        echo "DONE  $BUILD $kind $conv in $(( ($(date +%s)-START)/60 )) min"
      else
        echo "FAIL  $BUILD $kind $conv -- see $LOG"
        FAILED+=("$BUILD/$kind/$conv")
      fi
    done
  done
done
echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED RUNS: ${FAILED[*]}"
  echo "Re-run the same command to retry only these; completed runs are skipped."
fi
echo "=== summary ==="
python3 src/summary_table.py --csv "outputs/$(date +%Y%m%d)_MAPE_summary.csv"
python3 src/summary_table.py --metric MAE
python3 src/report.py --gate
