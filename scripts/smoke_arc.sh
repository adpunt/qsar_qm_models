#!/usr/bin/env bash
# ============================================================================
# The staged cluster smoke test. One stage per srun allocation.
# ============================================================================
#
#   bash scripts/smoke_arc.sh <stage>
#
# Every stage is independent and says PASS or FAIL on its last line. Run them in
# order and stop at the first failure -- each one is a precondition for the next.
# Nothing here submits a job; the point is to find out, in an allocation you are
# watching, what would otherwise be found by 294 array tasks at once.
#
# Run each stage in its OWN allocation, so a stage that hangs takes only itself:
#
#   srun --account=stat-cadd --partition=interactive --cpus-per-task=8 \
#        --mem=32G --time=01:00:00 --pty bash scripts/smoke_arc.sh 1
#
# `interactive` is the partition for --pty. `short` was advised twice and had
# ZERO idle nodes both times (RERUN_PLAN.md 2.8i).
#
#   1  the environment          ~5 min    32G
#   2  the binary and the noise ~20 min   32G
#   3  two tasks at once        ~15 min   32G
#   4  the uncertainty gates    ~20 min   64G
#   5  THE FINAL SMOKE TEST     ~2-4 h    64G    one pair, every condition, 5,000
#   6  the laboratory side      ~30 min   32G
#
# Stage 5 is the one the whole thing is for, and it is the only one that costs
# real time. Stages 1-4 are what stop stage 5 failing for a reason that has
# nothing to do with the pipeline.
# ============================================================================
set -uo pipefail

STAGE="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

OUT="${SMOKE_OUT:-$REPO/results/smoke_arc}"
mkdir -p "$OUT"

FAILED=0
step() { echo; echo "--- $* ---"; }
verdict() {
    if [ "$FAILED" -eq 0 ]; then echo; echo "PASS — stage $STAGE"; exit 0
    else echo; echo "FAIL — stage $STAGE: $FAILED check(s) failed"; exit 1; fi
}
run() {  # run <label> <command...>
    local label="$1"; shift
    echo; echo ">>> $label"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then echo "    FAILED (exit $rc): $label"; FAILED=$((FAILED+1));
    else echo "    ok: $label"; fi
    return 0
}

# --------------------------------------------------------------------------
# Every stage needs the environment, and it must be the one the jobs use.
# --------------------------------------------------------------------------
activate() {
    . "$REPO/setup.sh" || { echo "setup.sh failed"; exit 2; }
    if [ -z "${CONDA_PREFIX:-}" ]; then
        echo "ERROR: no environment active (CONDA_PREFIX unset). micromamba has"
        echo "       never worked here; setup.sh falls through to conda."
        exit 2
    fi
    if [ "$(basename "$CONDA_PREFIX")" != "env_test" ]; then
        echo "ERROR: active environment is $(basename "$CONDA_PREFIX"), not env_test"
        exit 2
    fi
    local py; py="$(command -v python)"
    case "$py" in
        "$CONDA_PREFIX"/*) : ;;
        /apps/system/*)
            echo "ERROR: python is the system Anaconda ($py). It has no gpytorch,"
            echo "       no quantile_forest and no ngboost, so a job would run,"
            echo "       find nothing to do and write no rows."; exit 2 ;;
        *) echo "ERROR: python is $py, outside $CONDA_PREFIX"; exit 2 ;;
    esac
    echo "=== interpreter: $py"
    echo "=== repository:  $(git -C "$REPO" log --oneline -1 2>/dev/null)"
    echo "=== branch:      $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    # The cluster's only route in is `git pull --ff-only origin <branch>`, so a
    # gate that passed on an unpushed commit proved nothing about what runs.
    local behind
    behind="$(git -C "$REPO" rev-list --count HEAD..@{u} 2>/dev/null || echo '?')"
    echo "=== behind its upstream by: $behind commit(s)"
}

case "$STAGE" in

# ==========================================================================
1)  echo "STAGE 1 — the environment. Nothing else is worth running until this passes."
    activate
    run "the full roster can be constructed, and the two that fail on contact fit" \
        python scripts/check_environment.py --deep --validation
    run "every model label all three generators can emit is known to the probe" \
        python scripts/check_environment.py --audit-roster
    step "what the shared spec hashes to (this must match the laboratory side)"
    python -c "
import sys; sys.path.insert(0, 'models')
import model_defaults as m
print('  model_defaults spec_version', m.SPEC_VERSION if hasattr(m,'SPEC_VERSION') else '?', 'hash', m.spec_hash())"
    verdict
    ;;

# ==========================================================================
2)  echo "STAGE 2 — the binary, and the noise it injects."
    activate
    step "rebuild — the binary on this cluster was six months old on 2026-08-27"
    ( cd rust && cargo build --release ) || { echo "cargo build failed"; exit 2; }
    ls -l --time-style=full-iso rust/target/release/rust_processor 2>/dev/null \
        || ls -l rust/target/release/rust_processor
    step "the held-out-noise fix is IN the binary you just built"
    grep -n "apply_noise" rust/src/main.rs | head -5

    run "33+ gates over real mmap files" bash -c "cd rust && cargo test --release"

    step "real QM9 labels and real Murcko groups for the preflight"
    python scripts/make_selftest_inputs.py 5000 "$OUT/qm9_5k"
    run "the preflight self-test, on 5,000 REAL QM9 labels" \
        ./rust/target/release/rust_processor --self-test "$OUT/qm9_5k.csv" \
            --scaffold-file "$OUT/qm9_5k.groups.json"

    python scripts/make_selftest_inputs.py 133885 "$OUT/qm9_full"
    run "the preflight self-test, on the whole label column" \
        ./rust/target/release/rust_processor --self-test "$OUT/qm9_full.csv" \
            --scaffold-file "$OUT/qm9_full.groups.json"

    run "the two injectors agree — 342 checks on all 133,885 labels" \
        python scripts/crosscheck_injectors.py
    # The SHAPE column, on real labels. It takes --labels, so the gate sweep
    # skips it unless it is given some, and the only ones lying around were
    # synthetic -- where the labels are independent of the scaffolds and the
    # clustering that widens a grouped condition's shape does not exist.
    run "the two injectors agree on the level-free shape, at 5,000" \
        python scripts/crosscheck_noise_pattern.py \
            --labels "$OUT/qm9_5k.csv" --groups "$OUT/qm9_5k.groups.json"
    run "...and on the whole column" \
        python scripts/crosscheck_noise_pattern.py \
            --labels "$OUT/qm9_full.csv" --groups "$OUT/qm9_full.groups.json"
    run "the settled condition set, on three sides" \
        python scripts/test_noise_conditions.py
    verdict
    ;;

# ==========================================================================
3)  echo "STAGE 3 — two tasks at once. This check MEANS NOTHING on a laptop."
    activate
    # The substantive half starts two real training tasks side by side. It skips
    # on macOS, because keopscore writes two fixed-name files under /tmp during
    # import and two simultaneous imports race there for reasons that have
    # nothing to do with this pipeline. The cluster is the only place it counts.
    run "two concurrent tasks each read their own configuration" \
        python scripts/test_config_isolation.py --end-to-end
    run "a hard failure in the Rust half stops the run instead of writing to a pipe nobody reads" \
        python scripts/test_failure_propagation.py

    step "the KeOps import race (RERUN_PLAN.md 2.25), on this cluster's /tmp"
    # Ten simultaneous imports. If the stagger is doing its job in the generated
    # scripts this is the only place the collision is ever seen again.
    rm -f "$OUT"/keops_*.log
    for i in $(seq 1 10); do
        ( python -c "import gpytorch" > "$OUT/keops_$i.log" 2>&1 ) &
    done
    wait
    if grep -l "compiler_version.txt\|brew_prefix.txt" "$OUT"/keops_*.log >/dev/null 2>&1; then
        echo "    REPRODUCED: $(grep -lc . "$OUT"/keops_*.log 2>/dev/null | wc -l) of 10"
        echo "    concurrent imports died on the hard-coded /tmp path. The random"
        echo "    wait at the top of every generated script is the mitigation; this"
        echo "    is not a stage-3 failure, it is the evidence it is needed."
        grep -h "FileNotFoundError" "$OUT"/keops_*.log | head -3
    else
        echo "    ok: 10 simultaneous imports, no collision this time (it is a race;"
        echo "        absence here is not proof, which is why the stagger stays)"
    fi
    verdict
    ;;

# ==========================================================================
4)  echo "STAGE 4 — the uncertainty split. The gates that need a real model fitted."
    activate
    run "the shared definition, its arithmetic and its support table" \
        python scripts/test_uncertainty_decomposition.py
    run "the writer: variances in, one conversion out, a fake per-molecule term refused" \
        python scripts/test_uncertainty_writer.py
    run "the variational noise head and the switch that reaches it" \
        python scripts/test_heteroscedastic_vbll.py
    run "the three checks' designs and wiring" \
        python scripts/test_decomposition_controls.py
    # THE ONE THAT FITS REAL MODELS. It is reported as BLOCKED, not as a pass,
    # on an interpreter whose quantile forest will not build.
    run "the three checks THEMSELVES, on real QM9, out of fold on scaffold groups" \
        python scripts/test_decomposition_controls.py --measured
    run "the validation split through the writer, the scorer and every trainer" \
        python scripts/test_validation_split_scoring.py
    run "one scale, one name, one condition per row" bash -c "
        python scripts/test_uncertainty_stats.py &&
        python scripts/test_model_names.py &&
        python scripts/test_condition_names.py &&
        python scripts/test_uncertainty_pairs.py"
    verdict
    ;;

# ==========================================================================
5)  echo "STAGE 5 — THE FINAL SMOKE TEST."
    echo "One model-and-representation pair against EVERY settled condition,"
    echo "5,000 QM9 molecules, one replicate, cross-fitted five ways."
    activate
    MODEL="${SMOKE_MODEL:-qrf}"
    REP="${SMOKE_REP:-ecfp4}"
    N="${SMOKE_N:-5000}"
    echo "=== pair: $MODEL / $REP at n=$N"

    # The processed cache is rebuilt ONCE, here, rather than by 294 tasks at
    # once. torch_geometric's QM9 takes no lock, and the ChemBERTa encoder change
    # of 2026-08-27 moved the record layout, so anything cached before that
    # decodes every later field at the wrong offset.
    step "the processed QM9 cache"
    ls -l data/QM9/processed/data_v3.pt 2>/dev/null \
        || echo "  absent — this run rebuilds it, which is the intention"

    step "the tuned-hyperparameter files (the screen wants them ABSENT)"
    ls -l results/master_tuned_hyperparameters.json \
          results/hyperparameter_decisions.json 2>/dev/null \
        || echo "  absent — every row will say params_source=default"

    DOSE="0.0 0.2 0.3 0.5 0.75 1.0 1.5"
    CENS="0.0 0.10 0.20 0.25 0.30 0.40 0.50"
    cd scripts

    one() {  # one <condition> <levels> <flags...>
        local cond="$1"; shift
        local levels="$1"; shift
        # The same random wait the generated scripts carry (RERUN_PLAN.md 2.25).
        sleep $(( RANDOM % 60 ))
        echo; echo ">>> condition: $cond"
        TMPDIR="$OUT/tmp_$cond"; mkdir -p "$TMPDIR"; export TMPDIR
        python -u process_and_train.py -d QM9 -t homo_lumo_gap \
            -m "$MODEL" -u True --oof-folds 5 \
            -r "$REP" \
            --noise-level $levels \
            --dose-units spread \
            "$@" \
            -n "$N" --repetitions 1 --start-iteration 0 \
            -s scaffold --normalize True \
            -f "$OUT/smoke_${cond}.csv" 2>&1 | tee "$OUT/smoke_${cond}.log" | tail -40
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then echo "    FAILED (exit $rc): $cond"; FAILED=$((FAILED+1));
        else echo "    ok: $cond"; fi
    }

    one gaussian        "$DOSE" --noise-shape gaussian  --noise-targeting uniform
    one grouped_wider   "$DOSE" --noise-shape gaussian  --noise-targeting grouped_wide
    one grouped_shifted "$DOSE" --noise-shape gaussian  --noise-targeting grouped_shift
    one censoring       "$CENS"                         --noise-targeting censoring --censor-side upper
    one student_t_nu5   "$DOSE" --noise-shape student_t --noise-targeting uniform --nu 5.0
    one outlier_p10     "$DOSE" --noise-shape gaussian  --noise-targeting outlier --outlier-p 0.1
    one laplace         "$DOSE" --noise-shape laplace   --noise-targeting uniform

    cd "$REPO"
    step "what the run actually wrote"
    python scripts/check_smoke_output.py "$OUT" || FAILED=$((FAILED+1))
    verdict
    ;;

# ==========================================================================
6)  echo "STAGE 6 — the laboratory side, and whether it is the same study."
    KIRBY_DIR="${KIRBY_DIR:-/data/stat-cadd/scat9264/KIRBy}"
    activate
    # The runner loads three files from THIS checkout -- every hyperparameter,
    # the uncertainty split, and the settled condition set -- and finds them by
    # walking up from its own directory unless told. Tell it.
    export QSAR_QM_MODELS_ROOT="$REPO"
    echo "=== KIRBy:      $KIRBY_DIR ($(git -C "$KIRBY_DIR" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
    echo "=== shared spec: $QSAR_QM_MODELS_ROOT"
    [ -f "$KIRBY_DIR/tests/alternative_data_noise_robustness.py" ] || {
        echo "ERROR: no runner at $KIRBY_DIR. The other checkout is"
        echo "       /data/stat-ecr/scat9264/KIRBy, which 125 of KIRBy's own 127"
        echo "       job scripts use (RERUN_PLAN.md 2.8b). Set KIRBY_DIR."; exit 2; }

    run "the two pipelines name the same models the same way" \
        python scripts/test_tuning_rosters.py
    run "the parity audit" \
        python scripts/audit_pipeline_parity.py --strict

    step "one real laboratory run — one dataset, one model, one rep, one condition"
    ( cd "$KIRBY_DIR/tests" && \
      python -u alternative_data_noise_robustness.py \
        --datasets logd --models QRF --reps ECFP4 \
        --conditions gaussian --unc-conditions all --oof-folds 3 \
        --results-root "$OUT/kirby_logd" 2>&1 | tee "$OUT/kirby_logd.log" | tail -40 )
    [ ${PIPESTATUS[0]:-0} -eq 0 ] || FAILED=$((FAILED+1))

    step "the three lines that say which checkout it read"
    grep -E "uncertainty split:|noise conditions:|model_defaults|spec.*hash" \
        "$OUT/kirby_logd.log" | head -10
    verdict
    ;;

*)  sed -n '2,40p' "${BASH_SOURCE[0]}"
    exit 2 ;;
esac
