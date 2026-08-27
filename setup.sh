#!/usr/bin/env bash
# Activate env_test, and build it if it is not there yet.
#
# This file is SOURCED by every job script (`. setup.sh`), so it must be cheap
# and must not mutate the environment on the ordinary path. It used to run four
# `pip install` commands every single time, which on a 390-task array meant 390
# concurrent writers into one shared site-packages -- and it silently undid
# whatever the last rebuild had pinned. Installation now happens only when the
# recipe has changed, recorded by a stamp file inside the environment itself.
#
#   . setup.sh                      activate; build only if absent or stale
#   SETUP_REBUILD=1 . setup.sh      delete the environment and build it again
#   ENV_TEST_PREFIX=/path . setup.sh   use a prefix instead of a name
#
# Rebuild commands for the cluster, copy-paste: RERUN_PLAN.md section 2.8i.
# Verify afterwards with:
#   python scripts/check_environment.py --deep --validation
#
# No `set -e`: this file is sourced, and a non-zero from a probe would kill the
# caller's shell -- which is how jobs 12822693 and 12822694 ended up running in
# the system Anaconda without saying so.

ENV_NAME="env_test"
_setup_src="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd "$(dirname "$_setup_src")" && pwd)"
YML_FILE="$REPO_ROOT/env.yml"
CONSTRAINTS="$REPO_ROOT/pip-constraints.txt"

# Every pip command below, and every pip command conda runs for the pip: block
# of env.yml, is held to the versions conda installed. A pip package that wants
# a different torch/scikit-learn/lightgbm/xgboost now fails the build loudly
# instead of quietly swapping in a wheel that carries its own OpenMP runtime.
export PIP_CONSTRAINT="$CONSTRAINTS"

# --- which tool ------------------------------------------------------------
# micromamba first where it exists (the laptop's env_test lives under it); on
# ARC it has never worked and is not on PATH, so this falls through to conda.
if command -v micromamba &>/dev/null; then
    SETUP_TOOL="micromamba"
    eval "$(micromamba shell hook --shell bash)"
elif command -v mamba &>/dev/null; then
    SETUP_TOOL="mamba"
    source "$(mamba info --base)/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
    SETUP_TOOL="conda"
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Neither micromamba, mamba, nor conda was found. Aborting."
    return 1 2>/dev/null || exit 1
fi

if [ -n "${ENV_TEST_PREFIX:-}" ]; then
    ENV_SELECT=(-p "$ENV_TEST_PREFIX")
    ENV_LABEL="$ENV_TEST_PREFIX"
else
    ENV_SELECT=(-n "$ENV_NAME")
    ENV_LABEL="$ENV_NAME"
fi

setup_env_exists() {
    if [ -n "${ENV_TEST_PREFIX:-}" ]; then
        [ -x "$ENV_TEST_PREFIX/bin/python" ]
    else
        "$SETUP_TOOL" env list 2>/dev/null | awk '{print $1}' | grep -Fxq "$ENV_NAME"
    fi
}

setup_activate() {
    if [ "$SETUP_TOOL" = "micromamba" ]; then
        micromamba activate "${ENV_SELECT[@]}" 2>/dev/null || micromamba activate "$ENV_LABEL"
    else
        conda activate "$ENV_LABEL"
    fi
}

# --- rebuild on request ----------------------------------------------------
if [ "${SETUP_REBUILD:-0}" = "1" ] && setup_env_exists; then
    echo "SETUP_REBUILD=1: removing $ENV_LABEL"
    "$SETUP_TOOL" env remove "${ENV_SELECT[@]}" -y
fi

# --- create ----------------------------------------------------------------
if ! setup_env_exists; then
    echo "Creating $ENV_LABEL from $YML_FILE (one OpenMP runtime; see the header there)"
    # strict priority is set here because an environment file cannot carry it:
    # conda's valid keys are name, dependencies, prefix, channels, variables,
    # and a `channel_priority:` line in the yml is silently ignored.
    if [ "$SETUP_TOOL" = "micromamba" ]; then
        CONDA_CHANNEL_PRIORITY=strict micromamba create -y "${ENV_SELECT[@]}" -f "$YML_FILE"
    else
        CONDA_CHANNEL_PRIORITY=strict "$SETUP_TOOL" env create "${ENV_SELECT[@]}" -f "$YML_FILE"
    fi
fi

setup_activate

# --- verify activation -----------------------------------------------------
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Environment activation failed. CONDA_PREFIX not set."
    return 1 2>/dev/null || exit 1
fi

# --- shared library paths --------------------------------------------------
echo "Setting shared library paths..."
if [[ "$OSTYPE" == linux-gnu* ]] || [[ -z "${OSTYPE:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
    export RUSTFLAGS="-C link-arg=-Wl,-rpath,$CONDA_PREFIX/lib"

    echo "Ensuring RDKit .so symlinks exist..."
    for full in "$CONDA_PREFIX"/lib/libRDKit*.so.1.2024.*; do
        base="${full%%.so.*}.so"
        if [ -e "$full" ] && [ ! -e "$base" ]; then
            ln -s "$full" "$base"
        fi
    done
elif [[ "$OSTYPE" == darwin* ]]; then
    unset DYLD_LIBRARY_PATH
    export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi

# --- isolate from Homebrew and any system Python ---------------------------
unset PYTHONPATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# ---------------------------------------------------------------------------
# The four things no channel can supply.
#
# They run ONLY when the recipe has changed. The stamp is a hash of env.yml,
# pip-constraints.txt and this file's extras block, so editing any of the three
# triggers exactly one rebuild of the extras and a job task never writes into a
# shared environment while another task is reading it.
# ---------------------------------------------------------------------------
setup_extras_stamp() {
    cat "$YML_FILE" "$CONSTRAINTS" 2>/dev/null | \
        { shasum -a 256 2>/dev/null || sha256sum; } | awk '{print $1}'
}
STAMP_FILE="$CONDA_PREFIX/.env_test_extras"
WANT_STAMP="$(setup_extras_stamp)"
HAVE_STAMP="$(cat "$STAMP_FILE" 2>/dev/null)"

if [ "${SETUP_FORCE_EXTRAS:-0}" = "1" ] || [ "$WANT_STAMP" != "$HAVE_STAMP" ]; then
    echo "Installing the packages no channel carries (stamp changed)..."

    # RDKit needs a newer libstdc++ than some cluster images ship.
    if [[ "$OSTYPE" == linux-gnu* ]] || [[ -z "${OSTYPE:-}" ]]; then
        "$SETUP_TOOL" install -y "${ENV_SELECT[@]}" -c conda-forge 'libstdcxx-ng>=12'
    fi

    # torchsort has no linux wheel and its extension must be compiled against
    # the torch that is INSTALLED. --no-build-isolation is the whole point:
    # with isolation pip builds it against a fresh torch it downloads into a
    # throwaway environment, which is how the live environment ended up with
    # `torchsort/isotonic_cpu...so: undefined symbol: _ZNK3c105Error4whatEv`
    # and a conformal roster that could not import (RERUN_PLAN.md 2.8d).
    python -m pip install --no-cache-dir --no-binary :all: --no-build-isolation \
        "torchsort==0.1.10"
    # torchcp is the conformal backend. The three conformal models sit in
    # EXCLUDED_MODELS, but the probe checks the whole roster, and a guard that
    # blocks a job for the guard's own reason is worse than no guard.
    python -m pip install --no-cache-dir "torchcp==1.1.0"

    # The noise injector and the validation pipeline's own package, from their
    # checkouts. The validation pipeline imports `noiseInject` and `kirby` with
    # no sys.path help, so without these the whole KIRBy half cannot start --
    # and an editable install is what keeps the injector in step with the
    # checkout instead of freezing a copy at whatever version it was.
    for pair in \
        "NoiseInject:$HOME/repos/NoiseInject:/data/stat-cadd/scat9264/NoiseInject:/data/stat-ecr/scat9264/NoiseInject" \
        "KIRBy:$HOME/repos/KIRBy:/data/stat-cadd/scat9264/KIRBy:/data/stat-ecr/scat9264/KIRBy"
    do
        name="${pair%%:*}"; rest="${pair#*:}"
        found=""
        IFS=':' read -ra cands <<< "$rest"
        for c in "${cands[@]}"; do
            if [ -d "$c" ]; then found="$c"; break; fi
        done
        if [ -n "$found" ]; then
            echo "  editable install: $name -> $found"
            python -m pip install --no-deps -e "$found"
        else
            echo "  WARNING: no $name checkout found; the validation pipeline"
            echo "           will not import until one is installed."
        fi
    done

    echo "$WANT_STAMP" > "$STAMP_FILE"
else
    echo "Extras already match the recipe; nothing to install."
fi

# NOTE, deliberately absent: the torch-scatter / torch-sparse / torch-cluster /
# torch-spline-conv wheels this script used to install from
# https://data.pyg.org/whl/torch-2.3.1+cu121.html. That line was the only thing
# pinning the environment to torch 2.3.1, it was added to match the machine
# rather than the recipe, and nothing in either roster uses the operators those
# packages provide -- the six representations are PDV, MHG-GNN, Avalon, ECFP4,
# ChemBERTa and Sort & Slice, and KIRBy already guards the one SchNet path that
# would want torch_cluster.

echo "Environment '$ENV_LABEL' ready ($(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null))"
