#!/bin/bash
# Submit continuous_pdv gap-fill scripts
# 3 array scripts (11 tasks each) + 2 single scripts = 35 parallel tasks

cd "$(dirname "$0")"

for f in anova_cpdv_*.sh; do
    echo "Submitting $f ..."
    sbatch "$f"
done

echo "Done. All fixup scripts submitted."
