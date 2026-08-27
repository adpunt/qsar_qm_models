#!/bin/bash
#SBATCH --job-name=mlp_outlier_reseed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# Re-run MLP + outlier with seed 123 (replacing bad seed-42 data)
# Bad data was deleted before submitting this script.

for REP in ecfp4 pdv smiles mhggnn mol2vec; do
    echo "=== MLP outlier ${REP} (seed 123) ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m mlp \
        -r ${REP} \
        --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --noise-strategy outlier \
        -n 10000 \
        -b 10 \
        --normalize True \
        --random-seed 123 \
        -f ../results/anova_outlier_${REP}_mlp.csv
done

echo "Done: MLP outlier re-run (all reps, seed 123)"
