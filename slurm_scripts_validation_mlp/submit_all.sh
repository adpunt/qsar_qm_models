#!/bin/bash
# Submit β-side (MLP base) validation jobs: 36 total
# - 12 × val_mlp_* (NN-β deterministic) -> dnn → mlp
# - 12 × val_mlp_bnn_* (BNN-β, full BNN on MLP)
# - 12 × val_mlp_vbll_* (VBLL-β, full VBLL on MLP)
COUNT=0
for f in val_mlp_*.sh val_mlp_bnn_*.sh val_mlp_vbll_*.sh; do
  jid=$(sbatch --parsable "$f")
  echo "$f -> $jid"
  COUNT=$((COUNT+1))
done
echo "Submitted $COUNT jobs (expected 36)"
