#!/bin/bash
# Submit VBLL-Full validation jobs (12 total: 4 reps × 3 datasets)
COUNT=0
for f in val_vbll_full_*.sh; do
  jid=$(sbatch --parsable "$f")
  echo "$f -> $jid"
  COUNT=$((COUNT+1))
done
echo "Submitted $COUNT jobs"
