#!/usr/bin/env bash
# Run a real generated uncertainty job script, off the cluster, and check what
# it does.
#
# scripts/test_uncertainty_job_scripts.py checks that the command line a script
# EMITS is one the runner accepts. This checks the other half: that the script
# gets far enough to emit it. Every guard between the top of the file and the
# python call is executed, in both directions -- the happy path must reach the
# runner, and each failure the guard exists for must stop before it.
#
# Two things are stubbed and nothing else:
#
#   setup.sh   a fake one that activates a fake environment named env_test, so
#              the activation guards see what they see on the cluster
#   python     a wrapper that runs the REAL python for the script's own
#              preflight (so the noiseInject check genuinely imports it), and
#              records the argument list instead of running the 47-hour job
#
# The KIRBy checkout is the real one -- found from --kirby-dir, $KIRBY_DIR, or a
# sibling of this repository -- so the check that it carries the redesigned
# command line is a real check.
#
#   bash scripts/smoke_uncertainty_job_scripts.sh
#   bash scripts/smoke_uncertainty_job_scripts.sh --kirby-dir ~/repos/KIRBy

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR="$REPO/slurm_scripts_uncertainty_rerun/generate_scripts.py"

KIRBY="${KIRBY_DIR:-$(dirname "$REPO")/KIRBy}"
if [ "${1:-}" = "--kirby-dir" ]; then KIRBY="$2"; shift 2; fi
if [ ! -f "$KIRBY/tests/alternative_data_noise_robustness.py" ]; then
    echo "No KIRBy checkout at $KIRBY. Pass --kirby-dir <path> or set KIRBY_DIR." >&2
    exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0

ok()   { PASS=$((PASS+1)); echo "    ok    $1"; }
bad()  { FAIL=$((FAIL+1)); echo "    FAIL  $1"; }

# ---------------------------------------------------------------------------
# the stubs
# ---------------------------------------------------------------------------
mkdir -p "$TMP/fake_qsar/scripts" "$TMP/env_test/bin" "$TMP/run"

# The generated script runs the model-backend check before anything else, and
# the stub repository had no copy of it -- so every task and every guard below
# died on `can't open file .../scripts/check_environment.py` instead of doing
# what it was written to test, and the two guards that print their own message
# reported "exit 2 but no message about it". The real check needs the model's
# backend installed and this harness deliberately has no backends, so it is
# stubbed to pass; what it guards is covered by the preflight, on the machine
# that will run the jobs.
cat > "$TMP/fake_qsar/scripts/check_environment.py" <<'EOF'
import sys
sys.exit(0)
EOF

cat > "$TMP/fake_qsar/setup.sh" <<EOF
# Stands in for the cluster's setup.sh: activate an environment called env_test
# and put its bin first on the path, which is what the guards look for.
export CONDA_PREFIX="$TMP/env_test"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
EOF

REAL_PYTHON="$(command -v python3 || command -v python)"
cat > "$TMP/env_test/bin/python" <<EOF
#!/usr/bin/env bash
# The script's own preflight pipes a here-document in on stdin; run that for
# real, so the noiseInject check actually imports noiseInject. The job itself
# is recorded, not run.
#
# SMOKE_FAST skips the real interpreter. It is set only for the sweep over
# every array index, which is checking the index arithmetic and needs no imports
# -- importing noiseInject takes about 3 seconds, so a real one there would put
# three minutes on a test that otherwise takes twenty seconds. The four
# representative tasks above it, and every guard below, use the real one.
for a in "\$@"; do
    if [ "\$a" = "alternative_data_noise_robustness.py" ]; then
        printf '%s\n' "\$@" > "\$SMOKE_ARGV"
        exit 0
    fi
done
[ -n "\${SMOKE_FAST:-}" ] && exit 0
exec "$REAL_PYTHON" "\$@"
EOF
chmod +x "$TMP/env_test/bin/python"

# A KIRBy checkout that predates the redesign: a runner with no --conditions.
mkdir -p "$TMP/old_kirby/tests"
echo "parser.add_argument('--strategies', nargs='+')" \
    > "$TMP/old_kirby/tests/alternative_data_noise_robustness.py"

# ---------------------------------------------------------------------------
# generate the real scripts
# ---------------------------------------------------------------------------
echo "Smoke test: a real generated uncertainty job script, executed"
echo "  KIRBy: $KIRBY"
python "$GENERATOR" --out-dir "$TMP/run" \
    --kirby-dir "$KIRBY" --qsar-dir "$TMP/fake_qsar" > "$TMP/gen.log" 2>&1 \
    || { echo "the generator failed:"; cat "$TMP/gen.log"; exit 1; }
SCRIPT="$TMP/run/unc_qrf.sh"
[ -f "$SCRIPT" ] || { echo "no unc_qrf.sh was written"; exit 1; }

CONDS=($(sed -n 's/^CONDS=(\(.*\))$/\1/p' "$SCRIPT"))
N_COND=${#CONDS[@]}
N_TASKS=$(( 3 * 4 * N_COND ))
echo "  $N_COND conditions, $N_TASKS tasks per script"

# Run one array task. $1 = SLURM_ARRAY_TASK_ID, rest = environment overrides.
run_task() {
    local idx="$1"; shift
    export SMOKE_ARGV="$TMP/argv.txt"
    rm -f "$SMOKE_ARGV"
    env -i HOME="$HOME" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        SMOKE_ARGV="$SMOKE_ARGV" SMOKE_FAST="${SMOKE_FAST:-}" \
        SLURM_ARRAY_TASK_ID="$idx" SLURM_JOB_PARTITION="medium" \
        SLURM_JOB_ID="99999" "$@" \
        bash "$SCRIPT" > "$TMP/out.txt" 2>&1
    echo $?
}

field() { grep -m1 "^$1" "$TMP/out.txt" | sed "s/^$1//"; }

# ---------------------------------------------------------------------------
# 1. the happy path, at several indices
# ---------------------------------------------------------------------------
echo
echo "1. the happy path reaches the runner"
for idx in 0 1 $((N_COND)) $((N_TASKS - 1)); do
    status="$(run_task "$idx")"
    if [ "$status" != "0" ]; then
        bad "task $idx exited $status"
        sed -n '1,25p' "$TMP/out.txt" | sed 's/^/          /'
        continue
    fi
    if [ ! -s "$TMP/argv.txt" ]; then
        bad "task $idx never reached the runner"
        continue
    fi
    argv="$(tr '\n' ' ' < "$TMP/argv.txt")"
    desc="$(field '=== task ')"
    for want in --datasets --models --reps --conditions --unc-conditions --oof-folds --results-root; do
        case "$argv" in *"$want"*) ;; *) bad "task $idx: $want missing from the command"; continue 2 ;; esac
    done
    case "$argv" in
        *--strategies*|*--unc-strategies*|*--threshold-quantile*|*--sigma*)
            bad "task $idx: a retired flag reached the runner: $argv"; continue ;;
    esac
    ok "task $idx ->$desc"
done

# every index produces a distinct output directory
SMOKE_FAST=1
: > "$TMP/dirs.txt"
for ((i = 0; i < N_TASKS; i++)); do
    status="$(run_task "$i")"
    [ "$status" = "0" ] || { bad "task $i exited $status"; break; }
    field '=== out: ' >> "$TMP/dirs.txt"
done
n_seen="$(wc -l < "$TMP/dirs.txt" | tr -d ' ')"
n_uniq="$(sort -u "$TMP/dirs.txt" | wc -l | tr -d ' ')"
if [ "$n_seen" = "$N_TASKS" ] && [ "$n_uniq" = "$N_TASKS" ]; then
    ok "$N_TASKS tasks wrote $n_uniq distinct results directories"
else
    bad "$N_TASKS tasks wrote $n_seen directories, $n_uniq of them distinct"
fi

# the condition really does change with the index, and only settled names appear
seen_conds="$(sed 's|.*__||' "$TMP/dirs.txt" | sort -u | tr '\n' ' ')"
want_conds="$(printf '%s\n' "${CONDS[@]}" | sort -u | tr '\n' ' ')"
if [ "$seen_conds" = "$want_conds" ]; then
    ok "the conditions actually run are: $seen_conds"
else
    bad "ran [$seen_conds], the script lists [$want_conds]"
fi

# ---------------------------------------------------------------------------
# 2. every guard stops the job it exists to stop
# ---------------------------------------------------------------------------
SMOKE_FAST=
echo
echo "2. each guard fires, and says why"

guard() {  # name, expected message fragment, then the run
    local name="$1" want="$2" status="$3"
    if [ "$status" = "0" ]; then
        bad "$name: the job ran anyway"
    elif grep -q "$want" "$TMP/out.txt"; then
        ok "$name: exit $status, \"$(grep -m1 "$want" "$TMP/out.txt" | cut -c1-72)\""
    else
        bad "$name: exit $status but no message about it"
        sed -n '1,6p' "$TMP/out.txt" | sed 's/^/          /'
    fi
}

status="$(run_task "$N_TASKS")"
guard "an index past the end" "out of range" "$status"

status="$(env -i HOME="$HOME" PATH="/usr/bin:/bin" SMOKE_ARGV="$TMP/argv.txt" \
    SLURM_ARRAY_TASK_ID=0 bash "$SCRIPT" > "$TMP/out.txt" 2>&1; echo $?)"
guard "no partition" "no partition" "$status"

# a KIRBy checkout that predates the redesign
python "$GENERATOR" --out-dir "$TMP/run_old" --kirby-dir "$TMP/old_kirby" \
    --qsar-dir "$TMP/fake_qsar" > /dev/null 2>&1
SAVED="$SCRIPT"; SCRIPT="$TMP/run_old/unc_qrf.sh"
status="$(run_task 0)"
guard "a KIRBy checkout with no --conditions" "has no --conditions flag" "$status"

python "$GENERATOR" --out-dir "$TMP/run_none" --kirby-dir "$TMP/nowhere" \
    --qsar-dir "$TMP/fake_qsar" > /dev/null 2>&1
SCRIPT="$TMP/run_none/unc_qrf.sh"
status="$(run_task 0)"
guard "no KIRBy checkout at all" "no KIRBy checkout" "$status"

# an environment that is not env_test
mkdir -p "$TMP/fake_qsar_wrong" "$TMP/base/bin"
cp "$TMP/env_test/bin/python" "$TMP/base/bin/python"
cat > "$TMP/fake_qsar_wrong/setup.sh" <<EOF
export CONDA_PREFIX="$TMP/base"
export PATH="\$CONDA_PREFIX/bin:\$PATH"
EOF
python "$GENERATOR" --out-dir "$TMP/run_wrongenv" --kirby-dir "$KIRBY" \
    --qsar-dir "$TMP/fake_qsar_wrong" > /dev/null 2>&1
SCRIPT="$TMP/run_wrongenv/unc_qrf.sh"
status="$(run_task 0)"
guard "the wrong environment" "not env_test" "$status"

# setup.sh that activates nothing
mkdir -p "$TMP/fake_qsar_dead"
echo "# activates nothing" > "$TMP/fake_qsar_dead/setup.sh"
python "$GENERATOR" --out-dir "$TMP/run_deadenv" --kirby-dir "$KIRBY" \
    --qsar-dir "$TMP/fake_qsar_dead" > /dev/null 2>&1
SCRIPT="$TMP/run_deadenv/unc_qrf.sh"
status="$(run_task 0)"
guard "no environment activated" "CONDA_PREFIX unset" "$status"
SCRIPT="$SAVED"

echo
if [ "$FAIL" -eq 0 ]; then
    echo "OK: $PASS checks passed — the generated scripts run, and every guard fires"
    exit 0
fi
echo "FAIL: $FAIL of $((PASS + FAIL)) checks failed"
exit 1
