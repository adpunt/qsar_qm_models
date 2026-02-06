#!/bin/bash
#SBATCH --job-name=runtime_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=short
#SBATCH --mem=64G

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

echo "Runtime Scaling Test"
echo "===================="
echo ""

for n in 100 500 1000 2000 5000; do
    echo "=== N=$n ==="
    echo "SVM:"
    time python process_and_train.py -d QM9 -t homo_lumo_gap -m svm -r ecfp4 --sigma 0.0 -n $n -b 1 -s scaffold
    echo ""
    echo "Conformal-RF:"
    time python process_and_train.py -d QM9 -t homo_lumo_gap -m conformal -r ecfp4 --sigma 0.0 -n $n -b 1 -s scaffold --cp-base-model rf
    echo ""
    echo "Conformal-DNN:"
    time python process_and_train.py -d QM9 -t homo_lumo_gap -m conformal -r ecfp4 --sigma 0.0 -n $n -b 1 -s scaffold --cp-base-model dnn
    echo ""
done
