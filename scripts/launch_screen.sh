#!/bin/bash
# =============================================================================
# Launch the QM9 screen -- one replicate, full grid. Everything after the
# environment is set up.
#
# Written 2026-08-28 because doing this by pasting blocks put a Ctrl-C somewhere
# nobody could identify. Every step here is idempotent: run it again after an
# interruption and it picks up where it left off rather than redoing damage.
#
#   bash scripts/launch_screen.sh              # do everything EXCEPT submit
#   bash scripts/launch_screen.sh --submit     # ...and submit the 294 tasks
#
# It refuses to submit unless every check above it passed.
# =============================================================================
set -uo pipefail

SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOG="$REPO/launch_screen_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

ACCT=stat-cadd
PART=medium

say()  { printf '\n=== %s\n' "$*"; }
die()  { printf '\nSTOPPED: %s\n\nNothing was submitted. The log is %s\n' "$*" "$LOG"; exit 1; }
ok()   { printf '  ok    %s\n' "$*"; }

say "log: $LOG"

# --- 1. the environment -------------------------------------------------------
say "1. environment"
[ -n "${CONDA_PREFIX:-}" ] || die "no environment is active. Run '. setup.sh' first."
[ "$(basename "$CONDA_PREFIX")" = "env_test" ] \
    || die "the active environment is $(basename "$CONDA_PREFIX"), not env_test."
case "$(command -v python)" in
    "$CONDA_PREFIX"/*) ok "python is inside $CONDA_PREFIX" ;;
    *) die "python is $(command -v python), outside the active environment." ;;
esac
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && die "this is an array task. Run it interactively."
ok "environment env_test"

# --- 2. the binary ------------------------------------------------------------
say "2. rust binary"
[ -x rust/target/release/rust_processor ] \
    || die "rust/target/release/rust_processor is missing. Run: cd rust && cargo build --release"
flags=$(./rust/target/release/rust_processor --help 2>&1 \
        | grep -cE -- "--noise-level|--dose-units|--noise-shape|--noise-targeting")
[ "$flags" -ge 4 ] \
    || die "the binary does not accept the current noise flags (found $flags of 4). It is the old build. Run: cd rust && cargo build --release"
ok "binary accepts all four noise flags"

# --- 3. clear what a killed run leaves behind ---------------------------------
say "3. clearing leftovers (safe to repeat)"
rm -f  train_*.mmap test_*.mmap val_*.mmap config_*.json config.json
rm -f  noise_manifest_*.json noise_provenance_*.csv scaffold_groups_*.json
rm -f  scripts/train_*.mmap scripts/test_*.mmap scripts/val_*.mmap
rm -f  scripts/config_*.json scripts/config.json scripts/noise_provenance_*.csv
rm -f  results/SMOKE_rf_ecfp4.csv
ok "intermediates and any interrupted smoke output cleared"

# The raw data is the only irreplaceable thing here. Never touched; only checked.
[ -d data/QM9/raw ] || die "data/QM9/raw is missing -- that is the source data. Stop and restore it."
ok "data/QM9/raw present"

# --- 4. QM9 processed, built ONCE, here --------------------------------------
say "4. QM9 processed cache"
if [ -f data/QM9/processed/data_v3.pt ]; then
    if python -c "
import torch, sys
try:
    torch.load('data/QM9/processed/data_v3.pt', map_location='cpu', weights_only=False)
except Exception as e:
    print('  unreadable:', e); sys.exit(1)
" ; then
        ok "data_v3.pt present and loads"
    else
        echo "  it exists but cannot be loaded -- a killed run left it half written. Rebuilding."
        rm -rf data/QM9/processed
    fi
fi
if [ ! -f data/QM9/processed/data_v3.pt ]; then
    echo "  building it once, single threaded (the array must never do this)..."
    ( cd scripts && python -c "
import process_and_train as p
d = p.load_qm9('homo_lumo_gap')
print('  QM9 loaded:', len(d), 'molecules')
" ) || die "QM9 processing failed. The log has the traceback."
    [ -f data/QM9/processed/data_v3.pt ] || die "processing finished but data_v3.pt is still absent."
    ok "data_v3.pt built"
fi

# --- 5. the tuned files must be absent or complete ---------------------------
say "5. tuned hyperparameters"
m=results/master_tuned_hyperparameters.json
h=results/hyperparameter_decisions.json
if [ -f "$m" ] && [ -f "$h" ]; then
    ok "both present -- tasks that start from now will use them"
elif [ ! -f "$m" ] && [ ! -f "$h" ]; then
    ok "neither present -- the screen runs on the shared defaults"
else
    die "exactly one of the two tuned files exists. That is a half written state; move it aside."
fi

# --- 6. prove the pipeline end to end ----------------------------------------
say "6. smoke run (rf / ecfp4, 500 molecules, 2 levels)"
( cd scripts && python -u process_and_train.py -d QM9 -t homo_lumo_gap \
      -m rf -r ecfp4 --noise-level 0.0 0.5 --dose-units spread \
      --noise-shape gaussian --noise-targeting uniform \
      -n 500 --repetitions 1 --start-iteration 0 \
      -s scaffold --normalize True -f ../results/SMOKE_rf_ecfp4.csv ) \
    || die "the smoke run failed. The log has the traceback."

python - <<'PY' || die "the smoke run produced rows that do not make sense. See above."
import sys, pandas as pd
d = pd.read_csv('results/SMOKE_rf_ecfp4.csv')
print(d[['sigma','r2','rmse','params_source','noise_type','delivered_dose']].to_string(index=False))
bad = []
if len(d) != 2: bad.append(f'expected 2 rows, got {len(d)}')
r = d.set_index('sigma')['r2']
if not (0.0 in r.index and 0.5 in r.index): bad.append('missing a noise level')
elif r[0.5] >= r[0.0]: bad.append(f'R2 did not fall with noise: {r[0.0]:.4f} -> {r[0.5]:.4f}')
row = d[d.sigma == 0.5]
if row['delivered_dose'].isna().all(): bad.append('delivered_dose is blank at level 0.5')
if bad:
    print('\n  PROBLEM: ' + '; '.join(bad)); sys.exit(1)
print('\n  two rows, R2 falls with noise, the injector recorded what it delivered')
PY
rm -f results/SMOKE_rf_ecfp4.csv
ok "pipeline runs end to end on this cluster"

# --- 7. old results out of the way -------------------------------------------
say "7. the previous grid"
n=$(ls results/anova_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -gt 0 ]; then
    # ARCHIVE TO A TEMPORARY NAME AND VERIFY BEFORE DELETING ANYTHING.
    #
    # On 2026-08-28 a Ctrl-C during this tar left a 459 MB archive that `tar tzf`
    # refuses -- "unexpected end of file" -- under the final name. It only cost
    # nothing because the delete that followed had not run. A half written archive
    # must never be able to look like a finished one.
    arch=~/results_anova_superseded_$(date +%Y%m%d_%H%M%S).tar.gz
    tmp="$arch.partial"
    echo "  archiving $n files (this takes a while; interrupting it deletes nothing)..."
    if ! tar czf "$tmp" results/anova_*.csv; then
        rm -f "$tmp"; die "the archive was interrupted or failed. NOTHING was deleted."
    fi
    if ! tar tzf "$tmp" > /dev/null 2>&1; then
        rm -f "$tmp"; die "the archive did not verify. NOTHING was deleted."
    fi
    mv "$tmp" "$arch"
    ok "archive verified: $arch"
    rm -f results/anova_*.csv
    ok "$n old anova files removed"
else
    ok "none present"
fi

# --- 8. generate ---------------------------------------------------------------
say "8. generating the screen"
( cd slurm_scripts_qm9_rerun && python generate_scripts.py --stage 0 ) \
    || die "the generator refused. Read its message -- it does not cap a wall time silently."
n=$(ls slurm_scripts_qm9_rerun/qm9_s0_*.sh 2>/dev/null | wc -l | tr -d ' ')
[ "$n" -eq 17 ] || die "expected 17 scripts, found $n."
ok "17 scripts"

# --- 9. submit ------------------------------------------------------------------
if [ "$SUBMIT" != "1" ]; then
    say "READY. Nothing submitted."
    echo "  Every check passed. To submit the 294 tasks:"
    echo "    bash scripts/launch_screen.sh --submit"
    echo "  log: $LOG"
    exit 0
fi

say "9. submitting to $PART under $ACCT"
cd slurm_scripts_qm9_rerun
subs=0
submit() { sbatch --account=$ACCT --partition=$PART --array="$1" "$2" && subs=$((subs+1)); }
for s in rf xgboost lgb svm ngboost dnn mlp;                   do submit "0-17%5" qm9_s0_$s.sh; done
for s in dnn_bnn_full mlp_bnn_full dnn_bnn_full_variational \
         mlp_bnn_full_variational heteroscedastic_gp \
         dnn_bnn_full_variational_hetero mlp_bnn_full_variational_hetero; do
    submit "0-17%4" qm9_s0_$s.sh; done
submit "0-17%5" qm9_s0_qrf.sh
submit "0-17%4" qm9_s0_gauche_rbf.sh
submit "0-5%4"  qm9_s0_gauche.sh

say "submitted $subs of 17 arrays"
[ "$subs" -eq 17 ] || echo "  WARNING: $((17-subs)) did not submit. Re-run with --submit; sbatch is not idempotent, so check squeue first."
squeue -u "$USER" -o "%.12i %.24j %.8T %.10M %R" | head -25
say "done. log: $LOG"
