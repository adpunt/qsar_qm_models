#!/bin/bash
# Submit β-side (MLP base) validation jobs: 36 total
# - 12 × val_mlp_*           (MLP deterministic = NN-β)
# - 12 × val_mlp_bnn_*       (full BNN on MLP base = BNN-β)
# - 12 × val_mlp_vbll_*      (full VBLL on MLP base = VBLL-β)
COUNT=0
# Single glob — val_mlp_*.sh already matches all 36 unique files
# (BNN and VBLL filenames also start with val_mlp_). Listing additional
# globs would re-submit those subsets, which caused 24 duplicate jobs
# in the first submission attempt.
for f in val_mlp_*.sh; do
  jid=$(sbatch --parsable "$f")
  echo "$f -> $jid"
  COUNT=$((COUNT+1))
done
echo "Submitted $COUNT jobs (expected 36)"
