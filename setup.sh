#!/usr/bin/env bash

ENV_NAME="env_test"
YML_FILE="env.yml"

if command -v micromamba &>/dev/null; then
    echo "Using micromamba..."
    eval "$(micromamba shell hook --shell bash)"

    if ! micromamba env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "Creating micromamba env from $YML_FILE..."
        micromamba create -f "$YML_FILE" -n "$ENV_NAME"
    fi

    micromamba activate "$ENV_NAME"

elif command -v mamba &>/dev/null; then
    echo "Using mamba..."
    source "$(mamba info --base)/etc/profile.d/conda.sh"

    if ! mamba env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "Creating mamba env from $YML_FILE..."
        mamba env create -f "$YML_FILE"
    fi

    conda activate "$ENV_NAME"

elif command -v conda &>/dev/null; then
    echo "Using conda..."
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "Creating conda env from $YML_FILE..."
        conda env create -f "$YML_FILE"
    fi

    conda activate "$ENV_NAME"

else
    echo "Neither micromamba, mamba, nor conda was found. Aborting."
    exit 1
fi

# --- Verify environment was activated ---
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Environment activation failed. CONDA_PREFIX not set."
    exit 1
fi

echo "Setting shared library paths..."
if [[ "$OSTYPE" == linux-gnu* ]] || [[ -z "${OSTYPE:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    echo "Ensuring RDKit .so symlinks exist..."
    for full in "$CONDA_PREFIX"/lib/libRDKit*.so.1.2024.*; do
        base="${full%%.so.*}.so"
        if [ ! -e "$base" ]; then
            ln -s "$full" "$base"
        fi
    done

elif [[ "$OSTYPE" == darwin* ]]; then
    unset DYLD_LIBRARY_PATH
    export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib"
fi



# --- Isolate from Homebrew ---
unset PYTHONPATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# --- PyTorch Geometric extensions ---
echo "Installing PyTorch Geometric extensions..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# --- Other pip packages ---
pip install tf-keras

echo "Environment '$ENV_NAME' setup complete!"
