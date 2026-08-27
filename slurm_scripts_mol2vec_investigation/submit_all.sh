#!/bin/bash
# Submit mol2vec investigation jobs
# SCP this directory first:
#   scp -r slurm_scripts_mol2vec_investigation scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_mol2vec_investigation

echo "Submitting mol2vec investigation scripts..."
JOB1=$(sbatch investigate_seeds.sh | awk '{print $4}')
echo "  Seeds investigation: Job $JOB1"

JOB2=$(sbatch investigate_archs.sh | awk '{print $4}')
echo "  Architecture investigation: Job $JOB2"

echo ""
echo "Total: 2 jobs submitted"
echo "  Seeds: 5 seeds × 3 strategies = 15 runs"
echo "  Archs: 5 architectures × 3 strategies = 15 runs"
echo ""
echo "Results will be saved as: results/mol2vec_investigate_*.csv"
echo "Monitor: squeue -u scat9264"
