set -uo pipefail

export MAMBA_EXE="/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
# Guarded: under `set -u` an unset CONDA_PREFIX aborts the shell here, before
# python is ever reached. The reference scripts get away with the unguarded
# form only because they do not set -u.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

cd /private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/fakeroot/data/stat-cadd/scat9264/KIRBy
. /private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/fakeroot/data/stat-cadd/scat9264/qsar_qm_models/setup.sh
cd tests
echo "REACHED END OF HEADER"
