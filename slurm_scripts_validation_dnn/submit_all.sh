#!/bin/bash
# Submit DNN reruns (12 jobs: 4 reps × 3 datasets)
# DNN baseline rerun with new [128, 64] no-BatchNorm architecture matching
# qsar_qm_models BNN/VBLL base, so DNN/BNN-Full/VBLL-Full rows are directly
# comparable on validation.
COUNT=0
for f in val_dnn_*.sh; do
  jid=$(sbatch --parsable "$f")
  echo "$f -> $jid"
  COUNT=$((COUNT+1))
done
echo "Submitted $COUNT jobs"
