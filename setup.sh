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

# Where env_test actually lives, whether it was asked for by name or by path.
setup_prefix_of_name() {
    "$SETUP_TOOL" env list 2>/dev/null \
        | awk -v n="$ENV_NAME" '$1==n{print $NF}' | head -1
}
SETUP_TARGET="${ENV_TEST_PREFIX:-$(setup_prefix_of_name)}"

setup_activate() {
    if [ "$SETUP_TOOL" = "micromamba" ]; then
        micromamba activate "${ENV_SELECT[@]}" 2>/dev/null || micromamba activate "$ENV_LABEL"
    else
        conda activate "$ENV_LABEL"
    fi
}

# --- build -----------------------------------------------------------------
# The old environment is NEVER deleted before its replacement exists. On
# 2026-08-27 it was: SETUP_REBUILD=1 removed env_test, the conda solve was then
# killed for memory on a login node, and the account was left with no
# environment at all. So: move the old one aside, build at the real path, and
# put the old one back if the build fails.
#
# Aside rather than build-then-move, because a conda environment is not
# relocatable -- every console script in bin/ carries the build path in its
# shebang, so an environment built at one prefix and moved to another has a pip
# that cannot run.
setup_create() {  # setup_create <select args...>
    # strict priority is set here because an environment file cannot carry it:
    # conda's valid keys are name, dependencies, prefix, channels, variables,
    # and a `channel_priority:` line in the yml is silently ignored.
    #
    # SETUP_CREATE_WITH names a binary to BUILD with, while everything after the
    # build still goes through conda. It exists because conda 4.12's solver
    # needs gigabytes to parse the conda-forge index and gets killed by the
    # login-node memory cap, where micromamba solves the same file in a few
    # hundred megabytes. The environment it produces is an ordinary conda
    # environment; `conda activate` reads it exactly the same.
    if [ -n "${SETUP_CREATE_WITH:-}" ]; then
        CONDA_CHANNEL_PRIORITY=strict "$SETUP_CREATE_WITH" create -y "$@" -f "$YML_FILE"
    elif [ "$SETUP_TOOL" = "micromamba" ]; then
        CONDA_CHANNEL_PRIORITY=strict micromamba create -y "$@" -f "$YML_FILE"
    else
        CONDA_CHANNEL_PRIORITY=strict "$SETUP_TOOL" env create "$@" -f "$YML_FILE"
    fi
}

setup_build_failed() {  # setup_build_failed <exit code>
    echo ""
    echo "BUILD FAILED (exit $1). Nothing has been installed."
    echo "  A solve that stops at 'Killed' with no other message is memory, not"
    echo "  the recipe. Build inside an allocation, never on a login node:"
    echo "    srun --account=stat-cadd --cpus-per-task=8 --mem=48G --time=03:00:00 \\"
    echo "         bash scripts/rebuild_env.sh"
}

if [ "${SETUP_REBUILD:-0}" = "1" ] || ! setup_env_exists; then
    if [ -n "$SETUP_TARGET" ] && [ -e "$SETUP_TARGET" ]; then
        SETUP_ASIDE="${SETUP_TARGET}.old"
        rm -rf "$SETUP_ASIDE"
        echo "Moving the old environment aside: $SETUP_TARGET -> $SETUP_ASIDE"
        mv "$SETUP_TARGET" "$SETUP_ASIDE" || {
            echo "Could not move it. Refusing to build over a live environment."
            return 1 2>/dev/null || exit 1
        }
        echo "Creating $SETUP_TARGET from $YML_FILE (one OpenMP runtime; see the header there)"
        setup_create -p "$SETUP_TARGET"
        SETUP_RC=$?
        if [ "$SETUP_RC" -ne 0 ] || [ ! -x "$SETUP_TARGET/bin/python" ]; then
            setup_build_failed "$SETUP_RC"
            echo "  Putting the old environment back."
            rm -rf "$SETUP_TARGET"
            mv "$SETUP_ASIDE" "$SETUP_TARGET"
            return 1 2>/dev/null || exit 1
        fi
        echo "Built. The previous environment is still on disk at $SETUP_ASIDE"
        echo "  -- delete it once the gate passes:  rm -rf $SETUP_ASIDE"
    else
        echo "Creating $ENV_LABEL from $YML_FILE (one OpenMP runtime; see the header there)"
        setup_create "${ENV_SELECT[@]}"
        SETUP_RC=$?
        if [ "$SETUP_RC" -ne 0 ]; then
            setup_build_failed "$SETUP_RC"
            return 1 2>/dev/null || exit 1
        fi
    fi
fi

setup_activate

# --- verify activation ------------------------------------------------------
# Not "is something active" -- "is the RIGHT thing active". The weaker check let
# a failed build fall through on 2026-08-27 with the system Anaconda still
# active, and the four extras below then installed themselves into
# /apps/.../Anaconda3 and ~/.local/lib/python3.9. Refuse instead.
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "Environment activation failed. CONDA_PREFIX not set."
    return 1 2>/dev/null || exit 1
fi
if [ -n "$SETUP_TARGET" ] && [ "$CONDA_PREFIX" != "$SETUP_TARGET" ]; then
    echo "Activation landed in the WRONG environment. Refusing to install anything."
    echo "  wanted: $SETUP_TARGET"
    echo "  got:    $CONDA_PREFIX"
    return 1 2>/dev/null || exit 1
fi
if [ -z "$SETUP_TARGET" ] && [ "$(basename "$CONDA_PREFIX")" != "$ENV_NAME" ]; then
    echo "Activation landed in '$CONDA_PREFIX', which is not $ENV_NAME."
    echo "  Refusing to install anything into it."
    return 1 2>/dev/null || exit 1
fi
SETUP_TARGET="$CONDA_PREFIX"

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
# ~/.local/lib/python3.X/site-packages is read by every interpreter of that
# version, this environment included, and this account has a torch and a
# scikit-learn sitting in one -- a second set of OpenMP runtimes arriving by the
# back door, outside anything env.yml or pip-constraints.txt can control.
export PYTHONNOUSERSITE=1
# and pip may not write there either: with no writable site-packages it silently
# falls back to --user, which is how packages meant for env_test ended up in
# ~/.local/lib/python3.9 on 2026-08-27.
export PIP_USER=0

# --- the libstdc++ torch needs ----------------------------------------------
# The cluster image ships a libstdc++ older than torch's extensions were built
# against, and the failure is
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
# on `import torch` -- which stops the KIRBy pipeline at its imports.
#
# This used to live in the extras block below, so it ran only when the recipe
# changed. A restored environment has no stamp for that recipe and is not
# described by it, so the one package that makes torch importable was tied to
# a condition that has nothing to do with it. It is its own check now: it looks
# at the environment's own libstdc++ and installs only if that is too old, so
# it costs one grep per task after the first time.
if [[ "$OSTYPE" == linux-gnu* ]] || [[ -z "${OSTYPE:-}" ]]; then
    if ! grep -aq GLIBCXX_3.4.30 "$CONDA_PREFIX/lib/libstdc++.so.6" 2>/dev/null; then
        echo "Installing libstdcxx-ng: this environment's libstdc++ is older than torch needs..."
        "$SETUP_TOOL" install -y "${ENV_SELECT[@]}" -c conda-forge 'libstdcxx-ng>=12'
    fi
fi

# ---------------------------------------------------------------------------
# The four things no channel can supply.
#
# They run ONLY when the recipe has changed. The stamp is a hash of env.yml,
# pip-constraints.txt and this file's extras block, so editing any of the three
# triggers exactly one rebuild of the extras and a job task never writes into a
# shared environment while another task is reading it.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Make the interpreter match the recipe.
#
# This is the step that should have been here all along. A restored environment,
# or one anything has been installed into by hand, drifts from env.yml -- and
# every round of that drift was being closed by pasting pip commands into a
# terminal, which is exactly what a setup script is for.
#
# check_environment.py --print-recipe-gaps is the single implementation of
# "what does env.yml pin, and what is actually here". It prints one line per
# unsatisfied pin: `pip <dist> <version>` or `conda <dist> <version>`.
#
# THE SPLIT MATTERS. Anything carrying compiled code is marked `conda` and is
# NEVER pip-installed here: a PyPI wheel of torch, scikit-learn, lightgbm,
# xgboost or grakel brings its own OpenMP runtime, which is the defect this
# environment exists to keep out. Those are reported with the command to run.
# ---------------------------------------------------------------------------
setup_reconcile() {
    local gaps
    gaps="$(python "$REPO_ROOT/scripts/check_environment.py" --print-recipe-gaps 2>/dev/null)"
    [ -z "$gaps" ] && return 0

    local pip_specs=() conda_specs=()
    while read -r kind dist want; do
        [ -z "${dist:-}" ] && continue
        if [ "$kind" = "conda" ]; then conda_specs+=("$dist=$want")
        else pip_specs+=("$dist==$want"); fi
    done <<< "$gaps"

    echo "The interpreter does not match env.yml. ${#pip_specs[@]} to install with pip,"
    echo "${#conda_specs[@]} that must come from conda."

    # Never install from inside an array task. 390 tasks writing into one
    # site-packages at once is the failure this file was rewritten to stop; the
    # environment has to be right BEFORE a launch, not repaired during one.
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "  REFUSING to install: this is array task ${SLURM_ARRAY_TASK_ID}."
        echo "  Run '. ./setup.sh' once on a login node before submitting."
        return 1
    fi

    if [ "${#pip_specs[@]}" -gt 0 ]; then
        echo "  pip: ${pip_specs[*]}"
        python -m pip install --no-cache-dir "${pip_specs[@]}" || {
            echo "  One of those could not be installed. Nothing else was changed."
            return 1
        }
    fi
    if [ "${#conda_specs[@]}" -gt 0 ]; then
        echo ""
        echo "  These carry compiled code, so they are NOT pip-installed -- a wheel of"
        echo "  any of them brings its own OpenMP runtime. Install from conda-forge,"
        echo "  with defaults excluded (that channel's mkl carries a second runtime):"
        echo ""
        echo "    micromamba install -y -p \"$CONDA_PREFIX\" --override-channels \\"
        echo "        -c conda-forge ${conda_specs[*]}"
        echo ""
        echo "  Then re-run:  python scripts/check_environment.py --deep --validation"
        return 1
    fi
    return 0
}

# The extras below are a different question: torchsort, torchcp and the two
# editable installs, none of which any channel carries. Read the installed torch
# from package metadata, never `import torch`: this file is sourced by every
# task, and importing torch in an environment with two threading runtimes is
# the hang this whole section is about.
SETUP_SKIP_EXTRAS=0
# Compare on the release only. conda-forge's torch 2.5.1 reports itself as
# 2.5.1.post108 and a pip wheel as 2.5.1+cu121; both ARE the pinned 2.5.1, and
# a comparison that says otherwise skips the extras on a correctly built
# environment -- which it did on ARC on 2026-08-28.
# Strip a trailing `.*` too: the pin is written `torch==2.5.1.*` so that PEP 440
# accepts conda-forge's post-release, and without this the comparison reads
# "2.5.1.post108 is not 2.5.1.*" and skips the extras on a matching environment.
setup_release_of() { local v="${1%%+*}"; v="${v%%.post*}"; echo "${v%.\*}"; }
_installed_torch="$(python -c "import importlib.metadata as m; print(m.version('torch'))" 2>/dev/null)"
_pinned_torch="$(grep -E '^torch==' "$CONSTRAINTS" 2>/dev/null | head -1 | cut -d'=' -f3)"
if [ -n "$_installed_torch" ] && [ -n "$_pinned_torch" ] \
   && [ "$(setup_release_of "$_installed_torch")" != "$(setup_release_of "$_pinned_torch")" ]; then
    echo "NOTE: torch here is $_installed_torch, and the recipe pins $_pinned_torch."
    echo "  This environment is not the one env.yml describes -- most likely restored"
    echo "  from research_archive/. Skipping the extras so they cannot change it."
    echo "  Everything above this line still applied: activation, library paths,"
    echo "  the RDKit symlinks and libstdc++."
    echo "  To install them anyway:  SETUP_FORCE_EXTRAS=1 . ./setup.sh"
    SETUP_SKIP_EXTRAS=1
fi

setup_reconcile
SETUP_RECONCILE_RC=$?

setup_extras_stamp() {
    cat "$YML_FILE" "$CONSTRAINTS" 2>/dev/null | \
        { shasum -a 256 2>/dev/null || sha256sum; } | awk '{print $1}'
}
STAMP_FILE="$CONDA_PREFIX/.env_test_extras"
WANT_STAMP="$(setup_extras_stamp)"
HAVE_STAMP="$(cat "$STAMP_FILE" 2>/dev/null)"

if [ "${SETUP_FORCE_EXTRAS:-0}" = "1" ] \
   || { [ "$SETUP_SKIP_EXTRAS" = "0" ] && [ "$WANT_STAMP" != "$HAVE_STAMP" ]; }; then
    echo "Installing the packages no channel carries (stamp changed)..."

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
elif [ "$SETUP_SKIP_EXTRAS" = "1" ]; then
    echo "Extras skipped: see the note above."
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
if [ "${SETUP_RECONCILE_RC:-0}" -ne 0 ]; then
    echo "  ...but it does not match env.yml yet -- see the lines above. Verify with:"
    echo "     python scripts/check_environment.py --deep --validation"
fi
