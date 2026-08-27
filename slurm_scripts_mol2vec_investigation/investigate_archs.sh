#!/bin/bash
#SBATCH --job-name=mol2vec_archs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

# Investigation: Is dnn/mol2vec/valprop positive NDS architecture-specific?
# Tests 5 NN architectures via flexible_dnn on mol2vec with valprop, legacy, outlier.
# flexible_dnn with default [128,64] already showed NDS=-1.856 on valprop (opposite extreme).

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

SIGMAS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

# Architecture 1: [64, 32] (smaller)
for STRATEGY in value_proportional legacy outlier; do
    case $STRATEGY in
        value_proportional) STRAT_SHORT="valprop" ;;
        legacy) STRAT_SHORT="legacy" ;;
        outlier) STRAT_SHORT="outlier" ;;
    esac
    echo "=== Running flexible_dnn [64,32] mol2vec/${STRAT_SHORT} ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m flexible_dnn \
        -r mol2vec \
        --hidden-sizes 64 32 \
        --sigma $SIGMAS \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 10 \
        --normalize True \
        -f ../results/mol2vec_investigate_arch64_32_${STRAT_SHORT}.csv
done

# Architecture 2: [256, 128] (wider)
for STRATEGY in value_proportional legacy outlier; do
    case $STRATEGY in
        value_proportional) STRAT_SHORT="valprop" ;;
        legacy) STRAT_SHORT="legacy" ;;
        outlier) STRAT_SHORT="outlier" ;;
    esac
    echo "=== Running flexible_dnn [256,128] mol2vec/${STRAT_SHORT} ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m flexible_dnn \
        -r mol2vec \
        --hidden-sizes 256 128 \
        --sigma $SIGMAS \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 10 \
        --normalize True \
        -f ../results/mol2vec_investigate_arch256_128_${STRAT_SHORT}.csv
done

# Architecture 3: [128, 128] (uniform width)
for STRATEGY in value_proportional legacy outlier; do
    case $STRATEGY in
        value_proportional) STRAT_SHORT="valprop" ;;
        legacy) STRAT_SHORT="legacy" ;;
        outlier) STRAT_SHORT="outlier" ;;
    esac
    echo "=== Running flexible_dnn [128,128] mol2vec/${STRAT_SHORT} ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m flexible_dnn \
        -r mol2vec \
        --hidden-sizes 128 128 \
        --sigma $SIGMAS \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 10 \
        --normalize True \
        -f ../results/mol2vec_investigate_arch128_128_${STRAT_SHORT}.csv
done

# Architecture 4: [64, 64, 32] (deeper)
for STRATEGY in value_proportional legacy outlier; do
    case $STRATEGY in
        value_proportional) STRAT_SHORT="valprop" ;;
        legacy) STRAT_SHORT="legacy" ;;
        outlier) STRAT_SHORT="outlier" ;;
    esac
    echo "=== Running flexible_dnn [64,64,32] mol2vec/${STRAT_SHORT} ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m flexible_dnn \
        -r mol2vec \
        --hidden-sizes 64 64 32 \
        --sigma $SIGMAS \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 10 \
        --normalize True \
        -f ../results/mol2vec_investigate_arch64_64_32_${STRAT_SHORT}.csv
done

# Architecture 5: [128, 64] (default — re-run for comparison since original flexible_dnn showed -1.856)
for STRATEGY in value_proportional legacy outlier; do
    case $STRATEGY in
        value_proportional) STRAT_SHORT="valprop" ;;
        legacy) STRAT_SHORT="legacy" ;;
        outlier) STRAT_SHORT="outlier" ;;
    esac
    echo "=== Running flexible_dnn [128,64] mol2vec/${STRAT_SHORT} ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m flexible_dnn \
        -r mol2vec \
        --hidden-sizes 128 64 \
        --sigma $SIGMAS \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 10 \
        --normalize True \
        -f ../results/mol2vec_investigate_arch128_64_${STRAT_SHORT}.csv
done

echo "=== Architecture investigation complete ==="
