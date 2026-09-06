#!/bin/bash
# The paths and queue settings every command in RERUN_PLAN.md 13.19 uses.
#
#     . /data/stat-cadd/scat9264/qsar_qm_models/scripts/runenv.sh
#
# SOURCE IT, do not run it -- it sets variables in your shell, and a subshell would
# throw them away. Source it again after every login: these were shell variables typed
# by hand, so they vanished with the session and did not follow the operator from
# arc-login01 to arc-login02.
#
# QSAR is derived from where this file is, so a different checkout needs no edit here.
# Everything else is named because it is a decision, not a location.

# --- this checkout, found from this file rather than hardcoded ---------------
_RUNENV_SRC="${BASH_SOURCE[0]:-$0}"
export QSAR="$(cd "$(dirname "$_RUNENV_SRC")/.." && pwd)"

# --- the KIRBy checkout the laboratory runner lives in -----------------------
# stat-ECR, not stat-cadd. KIRBy moved on 2026-05-07 when stat-cadd hit its quota,
# and 125 of KIRBy's own 127 job scripts use the stat-ecr path (RERUN_PLAN.md 0.4).
export KIRBY="${KIRBY:-/data/stat-ecr/scat9264/KIRBy}"

# --- the two selection files the deep run and censoring read AT RUN TIME -----
# Repository root, never results/: results/* is gitignored, so a file there cannot be
# committed or pulled and the tasks would exit 2 having fitted nothing.
export SEL="$QSAR/deep_run_pairs.json"
export CEN="$QSAR/censoring_pairs.json"

# --- account and partition ---------------------------------------------------
# The account is PINNED. where_to_submit.sh --emit returns the highest-fairshare
# association, and breaks an exact tie towards stat-ecr; this study bills to stat-cadd.
export ACCT="${ACCT:-stat-cadd}"
# --emit does not measure the partition either -- it hard-codes medium unless
# EMIT_PARTITION is set. The uncertainty runs are the only step that uses $PART; the
# grids say --partition=long literally, because they carry walls past medium's ceiling.
export PART="${PART:-medium}"

# --- the environment ----------------------------------------------------------
# EVERY python command in 13.19 needs env_test, not the system Anaconda. Sourcing this
# without it got as far as `import scipy` and then died on
#   /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
# -- env_test's scipy is built against a newer libstdc++ than the login node's, and
# setup.sh is what puts the conda one on LD_LIBRARY_PATH ("Setting shared library
# paths..."). The generated job scripts source setup.sh themselves and refuse to start
# without it; nothing the operator types by hand did, so the same command worked in one
# session and failed in the next.
if [ -z "${CONDA_PREFIX:-}" ] || [ "$(basename "${CONDA_PREFIX:-none}")" != "env_test" ]; then
    if [ -f "$QSAR/setup.sh" ]; then
        . "$QSAR/setup.sh"
    else
        echo "WARN  no setup.sh at $QSAR -- python commands will run under whatever is"
        echo "      on PATH, which on a login node is the system Anaconda."
    fi
fi

echo "QSAR  = $QSAR"
echo "KIRBY = $KIRBY"
echo "SEL   = $SEL"
echo "CEN   = $CEN"
echo "ACCT  = $ACCT      PART = $PART   (PART is used by the uncertainty runs only)"
echo "python= $(command -v python)"
echo "env   = ${CONDA_PREFIX:-NONE}"

_runenv_missing=0
for _p in "$QSAR" "$KIRBY"; do
    [ -d "$_p" ] || { echo "MISSING directory: $_p"; _runenv_missing=1; }
done
for _p in "$SEL" "$CEN"; do
    [ -f "$_p" ] || { echo "MISSING file: $_p  -- pull first; every deep-run and"
                      echo "        censoring task exits 2 without it."; _runenv_missing=1; }
done
[ "$_runenv_missing" -eq 0 ] && echo "ok    all four paths exist"
unset _RUNENV_SRC _p _runenv_missing
