#!/bin/bash
# Re-run VBLL × value_proportional (previous runs used invalid strategy name "valprop")

echo "=== VBLL valprop re-runs ==="
sbatch dnn_vbll_valprop.sh
sbatch mlp_vbll_valprop.sh

echo ""
echo "Total: 2 jobs submitted"
echo "Each runs 5 reps × 11 sigmas × 10 bootstraps"
echo "Previous runs (jobs 11352465/11352466) used --noise-strategy valprop (invalid)"
echo "These use --noise-strategy value_proportional (correct)"
echo ""
echo "Use 'squeue -u scat9264' to monitor"
