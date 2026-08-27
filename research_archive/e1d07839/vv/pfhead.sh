#!/bin/bash
# ============================================================================
# PREFLIGHT — run this ONCE on the server BEFORE submitting anything.
# ============================================================================
# It costs ~2 minutes and catches the failures that would otherwise waste days
# of queue time. Two of these were caught locally and are real:
#
#   * quantile_forest vs scikit-learn version clash. On this laptop
#     RandomForestQuantileRegressor.fit() raises
#     "Invalid parameter 'monotonic_cst'". If the server env has the same
#     clash, every QRF job fails on contact.
#   * The cached hERG CSV has been written with different label column names by
#     different versions. The loader now accepts pKi / pChEMBL / pchembl_value,
#     but the cache still has to exist and parse.
#
#   sbatch --account=<acct> --partition=short preflight.sh
#   # or just run it in an interactive session:
#   #   bash preflight.sh
# ============================================================================
#SBATCH --job-name=unc_preflight
#SBATCH --output=logs/preflight_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -uo pipefail

export MAMBA_EXE="/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/fakeroot/data/stat-cadd/scat9264/KIRBy
. /private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/fakeroot/data/stat-cadd/scat9264/qsar_qm_models/setup.sh
cd tests
echo "REACHED END OF HEADER"
