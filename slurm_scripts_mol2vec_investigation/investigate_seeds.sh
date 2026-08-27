#!/bin/bash
#SBATCH --job-name=mol2vec_seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Investigation: Is dnn/mol2vec/valprop positive NDS (+0.738) a seed fluke?
# Tests 5 random seeds on dnn + mol2vec with valprop, legacy, and outlier strategies.
# Original experiment used seed 42 (default).

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

SIGMAS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for SEED in 42 123 456 789 1337; do
    for STRATEGY in value_proportional legacy outlier; do
        # Map strategy name to short form for filename
        case $STRATEGY in
            value_proportional) STRAT_SHORT="valprop" ;;
            legacy) STRAT_SHORT="legacy" ;;
            outlier) STRAT_SHORT="outlier" ;;
        esac

        echo "=== Running dnn/mol2vec/${STRAT_SHORT} seed=${SEED} ==="
        python process_and_train.py -d QM9 -t homo_lumo_gap \
            -m dnn \
            -r mol2vec \
            --random-seed $SEED \
            --sigma $SIGMAS \
            --noise-strategy $STRATEGY \
            -n 10000 \
            -b 10 \
            --normalize True \
            -f ../results/mol2vec_investigate_seed${SEED}_${STRAT_SHORT}_dnn.csv
    done
done

echo "=== Seed investigation complete ==="
