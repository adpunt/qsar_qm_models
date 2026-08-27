#!/bin/bash
for f in vbll_*.sh; do
    echo "Submitting $f"
    sbatch "$f"
done
