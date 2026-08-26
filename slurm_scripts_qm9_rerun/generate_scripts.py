#!/usr/bin/env python3
"""Generate the SLURM job arrays for the QM9 re-run.

WHY EVERYTHING ABOVE ZERO NOISE HAS TO BE REDONE
------------------------------------------------
The noise map is keyed by TRAINING index, and write_data restarted its counter
for every split -- so validation and test molecules were handed the noise drawn
for the training molecule at the same position. Held-out labels were corrupted
contrary to the Methods, AND the corruption was attached to the wrong molecules.
Every QM9 R2 above sigma = 0 was scored against a moving target.

Fixed in rust/src/main.rs (apply_noise flag, true only for the training call).
The fix is in the repo; the binary must be REBUILT on the cluster.

Results at sigma = 0 are unaffected -- process_and_train.py sets
`'noise': s > 0`, so at zero the noise path is switched off for all splits.
They are re-run here anyway because splicing two runs is more error-prone than
re-running one grid, and sigma=0 is 1/11th of the cost.

THE GRID
--------
11 ANOVA models x 5 representations x 6 strategies x 11 noise levels x 10
replicates.  Model and representation are the two factors of the variance
decomposition, so neither can be cut without gutting the paper's first
question.  Plus two arms outside the ANOVA: QRF (redundant with RF for
accuracy, but it is the strongest error-ranker in the uncertainty results) and
the RBF Gaussian process (absent from the primary representation entirely).

One array TASK per (strategy, representation); one SCRIPT per model.
"""
import argparse
from pathlib import Path

# Representation sets.
#   ANOVA_REPS  the five the variance decomposition uses.
#   ALL_REPS    adds SNS, which is excluded from the ANOVA as redundant with
#               ECFP4 but IS still reported -- generate_paper_figures_v2.py:2313
#               prints SNS specifically, and :4065 slices on it, because
#               table1_supp_simple_effects_all_reps.csv deliberately runs with no
#               exclusions (:2251).
#   FP_REPS     binary fingerprints only. The Tanimoto kernel is defined on
#               binary vectors, so the Tanimoto GP can only run here.
ANOVA_REPS = ['ecfp4', 'continuous_pdv', 'smiles', 'mhggnn', 'mol2vec']
ALL_REPS = ANOVA_REPS + ['sns']
FP_REPS = ['ecfp4', 'sns']

# label -> (extra CLI flags, tier, hours, note, reps)
# The label is the filename suffix and MUST match what the figure script parses:
# generate_paper_figures_v2.py:592-615 maps '<base>_bnn_full_variational' to the
# VBLL models, so that spelling is load-bearing.
MODELS = {
    'rf':                       ('-m rf',        1, 23, 'random forest', ALL_REPS),
    'xgboost':                  ('-m xgboost',   1, 23, '', ALL_REPS),
    'lgb':                      ('-m lgb',       1, 23, 'LightGBM', ALL_REPS),
    'svm':                      ('-m svm',       1, 35, 'RBF kernel on every representation', ALL_REPS),
    'ngboost':                  ('-m ngboost -u True', 1, 47, 'slowest tree model; emits per-molecule uncertainty', ALL_REPS),
    'dnn':                      ('-m dnn',       1, 35, '', ALL_REPS),
    'mlp':                      ('-m mlp',       1, 35, '', ALL_REPS),
    'dnn_bnn_full':             ('-m dnn --bayesian-transformation full -u True', 2, 47, 'BNN-alpha', ALL_REPS),
    'mlp_bnn_full':             ('-m mlp --bayesian-transformation full -u True', 2, 47, 'BNN-beta', ALL_REPS),
    'dnn_bnn_full_variational': ('-m dnn --bayesian-transformation full_variational -u True', 2, 47, 'VBLL-alpha (figure script reads this as dnn_vbll)', ALL_REPS),
    'mlp_bnn_full_variational': ('-m mlp --bayesian-transformation full_variational -u True', 2, 47, 'VBLL-beta (figure script reads this as mlp_vbll)', ALL_REPS),
    # Outside the ANOVA roster, but they feed figures and supplementary tables:
    'qrf':        ('-m qrf -u True', 3, 23, 'not in the ANOVA (rho 0.996 with rf) but the best error-ranker', ALL_REPS),
    'gauche_rbf': ('-m gauche --kernel rbf -u True', 3, 47, 'RBF GP on EVERY rep, so the GP can finally enter the cross-rep ANOVA', ALL_REPS),
    'gauche':     ('-m gauche --kernel tanimoto -u True', 3, 47,
                   'Tanimoto GP. Only defined on BINARY fingerprints, so ecfp4/sns only. '
                   'This is the RBF-vs-Tanimoto head-to-head; the figure script gives it its '
                   'own colour and marker and labels it GP', FP_REPS),
}

# Excluded from EVERY figure by GLOBAL_MODELS_EXCLUDE, so re-running them
# produces files nothing reads. They survive only in
# table1_supp_simple_effects_all_reps.csv, which runs with no exclusions.
# Off by default; --include-excluded turns them on.
EXCLUDED_MODELS = {
    'conformal_rf':             ('-m conformal --cp-base-model rf -u True',  4, 35, 'conformal wrapper (rho > 0.99 with rf)', ANOVA_REPS),
    'conformal_qrf':            ('-m conformal --cp-base-model qrf -u True', 4, 35, 'conformal wrapper', ANOVA_REPS),
    'conformal_dnn':            ('-m conformal --cp-base-model dnn -u True', 4, 47, 'conformal wrapper', ANOVA_REPS),
    'dnn_bnn_last':             ('-m dnn --bayesian-transformation last -u True', 4, 47, 'last-layer BNN (no significant gain over base)', ANOVA_REPS),
    'mlp_bnn_last':             ('-m mlp --bayesian-transformation last -u True', 4, 47, 'last-layer BNN', ANOVA_REPS),
    'dnn_bnn_variational':      ('-m dnn --bayesian-transformation variational -u True', 4, 47, 'pre-VBLL variational (was identical to last-layer, a bug)', ANOVA_REPS),
    'mlp_bnn_variational':      ('-m mlp --bayesian-transformation variational -u True', 4, 47, 'pre-VBLL variational', ANOVA_REPS),
    'flexible_dnn':             ('-m flexible_dnn', 4, 35, 'architecture variant, not discussed in the paper', ANOVA_REPS),
}

# ANOVA representations. 'pdv' (binary) and 'sns' are excluded by the figure
# script (ANOVA_REPS_EXCLUDE), so running them would be wasted compute.
REPS = ['ecfp4', 'continuous_pdv', 'smiles', 'mhggnn', 'mol2vec']
STRATEGIES = ['legacy', 'value_proportional', 'quantile', 'threshold', 'heteroscedastic', 'outlier']
# Filename stem for each strategy, as the figure script's glob expects.
STRATEGY_TAG = {'legacy': 'legacy', 'value_proportional': 'valprop', 'quantile': 'quantile',
                'threshold': 'threshold', 'heteroscedastic': 'hetero', 'outlier': 'outlier'}

QSAR_DIR = '/data/stat-cadd/scat9264/qsar_qm_models'
SIGMAS = '0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0'

TEMPLATE = '''#!/bin/bash
# ============================================================================
# QM9 re-run — model: {model}
# {note}
# ============================================================================
# Array task -> (strategy, representation).
#   {n_st} strategies x {n_rep} representations = {n_tasks} tasks
#   each task = {n_sig} noise levels x {boot} replicates = {runs} training runs
#
# REQUIRES the rebuilt rust binary (the held-out-noise fix). Run
# `cargo build --release` in {qsar_dir}/rust FIRST -- see RUNBOOK.
#
# --account and --partition are LIVE STATE; pass them at submit time:
#   sbatch --account=<acct> --partition=medium --array=0-{last}%{throttle} {script_name}
#
# Resubmit only failed indices:
#   sbatch --account=<acct> --partition=medium --array=3,17 {script_name}
# ============================================================================
#SBATCH --job-name=qm9_{jobslug}
#SBATCH --output=qm9_{jobslug}_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

# micromamba has never worked on this cluster -- setup.sh has always fallen
# through to its conda branch. The two `export MAMBA_EXE=...` lines that used to
# sit here were dead: the hook failed, and because this script runs under
# `set -uo pipefail` with no `-e`, the failure did not stop anything. The task
# simply carried on unactivated, in whatever python was on PATH.
cd {qsar_dir}
. setup.sh

# Activation is not optional, and until now nothing checked it. An unactivated
# task falls through to the system Anaconda at /apps/system/..., which has no
# gpytorch, no quantile_forest and no ngboost -- so the job runs, finds nothing
# to do, and produces no rows. That is what happened to 12822693 and 12822694
# (RERUN_PLAN.md §2.8d). Confirmed on the cluster 2026-08-26.
if [ -z "${{CONDA_PREFIX:-}}" ]; then
    echo "ERROR: setup.sh did not activate an environment (CONDA_PREFIX unset)."
    exit 2
fi
PY_PATH="$(command -v python)"
case "$PY_PATH" in
    "$CONDA_PREFIX"/*) : ;;
    *)
        echo "ERROR: python is $PY_PATH, which is not inside the activated"
        echo "       environment ($CONDA_PREFIX). setup.sh did not activate 'env_test'."
        case "$PY_PATH" in
            /apps/system/*)
                echo "       That is the system Anaconda. It has no gpytorch, no"
                echo "       quantile_forest and no ngboost, so this job would run,"
                echo "       find nothing to do, and write no rows." ;;
        esac
        exit 2 ;;
esac
echo "=== interpreter: $PY_PATH  (CONDA_PREFIX=$CONDA_PREFIX)"

# The binary carries the held-out-noise fix. Refuse to run without it rather
# than silently regenerating the same invalid results.
if [ ! -x rust/target/release/rust_processor ]; then
    echo "ERROR: rust/target/release/rust_processor missing. Run:"
    echo "  cd {qsar_dir}/rust && cargo build --release"
    exit 2
fi

cd scripts

# The interpreter has to be able to BUILD what this job is about to ask for.
# Jobs 12822693 and 12822694 (2026-08-19) ran to completion and produced nothing
# because gpytorch was missing: the experiment list came out empty, five folds
# looped over nothing, and the job crashed reading its own empty output. This
# costs seconds and answers that before the queue does (RERUN_PLAN.md §2.8d).
python check_environment.py --models {model} || {{
    echo "ERROR: this interpreter cannot build model '{model}'. See the output above."
    exit 2
}}

STRATS=({strategies})
REPS=({reps})

n_rep=${{#REPS[@]}}
n_tasks=$(( ${{#STRATS[@]}} * n_rep ))
i="${{SLURM_ARRAY_TASK_ID:-0}}"
if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
if [ -z "${{SLURM_JOB_PARTITION:-}}" ]; then
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK)."; exit 2
fi

rep="${{REPS[$(( i % n_rep ))]}}"
strat="${{STRATS[$(( i / n_rep ))]}}"
# Filename tag the figure script globs for (it differs from the CLI spelling).
case "$strat" in
  value_proportional) tag=valprop ;;
  heteroscedastic)    tag=hetero ;;
  *)                  tag="$strat" ;;
esac

OUT="../results/anova_${{tag}}_${{rep}}_{model}.csv"

echo "=== task $i: model={model} strategy=$strat rep=$rep"
echo "=== out: $OUT"
echo "=== started: $(date)"

python -u process_and_train.py -d QM9 -t homo_lumo_gap \\
    {flags} \\
    -r "$rep" \\
    --sigma {sigmas} \\
    --noise-strategy "$strat" \\
    -n 10000 \\
    -b {boot} \\
    -s scaffold \\
    --normalize True \\
    -f "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrapping', type=int, default=10,
                    help='Replicates per configuration (default 10). The analysis needs at '
                         'least 5 (MIN_CELL_ITERS in generate_paper_figures_v2.py); 5 halves '
                         'the cost and costs precision on the residual term, which is itself '
                         'a reported result.')
    ap.add_argument('--throttle', type=int, default=5)
    ap.add_argument('--models', nargs='+', default=None, help='Subset of model labels.')
    ap.add_argument('--include-excluded', action='store_true',
                    help='Also generate the model variants that GLOBAL_MODELS_EXCLUDE drops from '
                         'every figure (conformal wrappers, last-layer and pre-VBLL variational '
                         'BNNs, flexible DNN). They feed only the no-exclusions supplementary '
                         'table. Off by default because the files nothing reads.')
    ap.add_argument('--out-dir', default=str(Path(__file__).parent))
    args = ap.parse_args()

    out = Path(args.out_dir)
    n_sig = len(SIGMAS.split())
    pool = dict(MODELS)
    if args.include_excluded:
        pool.update(EXCLUDED_MODELS)
    chosen = {k: v for k, v in pool.items() if not args.models or k in args.models}

    written = []
    grand = 0
    for model, (flags, tier, hours, note, reps) in chosen.items():
        n_tasks = len(STRATEGIES) * len(reps)
        grand += n_tasks
        script_name = f'qm9_{model}.sh'
        (out / script_name).write_text(TEMPLATE.format(
            model=model, note=note or model, jobslug=model,
            cpus=8, mem='128G', hours=hours, flags=flags,
            qsar_dir=QSAR_DIR, sigmas=SIGMAS, boot=args.bootstrapping,
            strategies=' '.join(STRATEGIES), reps=' '.join(reps),
            n_st=len(STRATEGIES), n_rep=len(reps), n_tasks=n_tasks,
            n_sig=n_sig, runs=n_sig * args.bootstrapping,
            last=n_tasks - 1, throttle=args.throttle, script_name=script_name))
        (out / script_name).chmod(0o755)
        written.append((tier, script_name, model, hours, n_tasks, len(reps)))

    print(f"Wrote {len(written)} array scripts, {grand} tasks total")
    print(f"Each task: {n_sig} noise levels x {args.bootstrapping} replicates "
          f"= {n_sig * args.bootstrapping} training runs")
    print(f"Grid total: {grand * n_sig * args.bootstrapping:,} training runs")
    for tier, name in [(1, 'ANOVA roster — tree and deterministic'),
                       (2, 'ANOVA roster — Bayesian networks'),
                       (3, 'outside the ANOVA but feeding figures/supplementary'),
                       (4, 'excluded from every figure — only with --include-excluded')]:
        rows = [w for w in written if w[0] == tier]
        if rows:
            print(f"\n  Tier {tier} ({name}):")
            for _, nm, m, h, nt, nr in rows:
                print(f"    {nm:36s} {m:26s} {nr} reps  {nt:3d} tasks  --time={h}:59:00")


if __name__ == '__main__':
    main()
