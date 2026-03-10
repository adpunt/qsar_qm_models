#!/bin/bash
# Submit all 24 scripts in parallel (6 GP RBF + 18 flexible_dnn)

for f in gp_rbf_*.sh flexdnn_*.sh; do
    echo "Submitting $f"
    sbatch "$f"
done

echo "Submitted 24 jobs"
