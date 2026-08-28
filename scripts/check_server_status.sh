#!/bin/bash
# Run this on the server to check experiment status

echo "=========================================="
echo "SLURM JOB STATUS"
echo "=========================================="
squeue -u $USER

echo ""
echo "=========================================="
echo "ANOVA RESULT FILES (new experiments)"
echo "=========================================="
cd /data/stat-cadd/scat9264/qsar_qm_models/results
ls -la anova_*.csv 2>/dev/null | wc -l
echo "files found"
ls anova_*.csv 2>/dev/null

echo ""
echo "=========================================="
echo "SLURM SCRIPTS AVAILABLE"
echo "=========================================="
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts 2>/dev/null && ls *.sh 2>/dev/null || echo "No slurm_scripts directory"

echo ""
echo "=========================================="
echo "RECENT SLURM LOGS (last 24h)"
echo "=========================================="
find /data/stat-cadd/scat9264/qsar_qm_models -name "slurm-*.out" -mtime -1 2>/dev/null | head -20
