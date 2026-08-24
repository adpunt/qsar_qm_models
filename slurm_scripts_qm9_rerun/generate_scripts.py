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

# label -> (extra CLI flags, tier, hours, note)
# The label is the filename suffix and MUST match what the figure script parses:
# generate_paper_figures_v2.py:592-615 maps '<base>_bnn_full_variational' to the
# VBLL models, so that spelling is load-bearing.
MODELS = {
    'rf':                       ('-m rf',                                             1, 23, 'random forest'),
    'xgboost':                  ('-m xgboost',                                        1, 23, ''),
    'lgb':                      ('-m lgb',                                            1, 23, 'LightGBM'),
    'svm':                      ('-m svm',                                            1, 35, 'RBF kernel on every representation'),
    'ngboost':                  ('-m ngboost -u True',                                1, 47, 'slowest tree model; emits per-molecule uncertainty'),
    'dnn':                      ('-m dnn',                                            1, 35, ''),
    'mlp':                      ('-m mlp',                                            1, 35, ''),
    'dnn_bnn_full':             ('-m dnn --bayesian-transformation full -u True',     2, 47, 'BNN-alpha'),
    'mlp_bnn_full':             ('-m mlp --bayesian-transformation full -u True',     2, 47, 'BNN-beta'),
    'dnn_bnn_full_variational': ('-m dnn --bayesian-transformation full_variational -u True', 2, 47, 'VBLL-alpha (figure script reads this as dnn_vbll)'),
    'mlp_bnn_full_variational': ('-m mlp --bayesian-transformation full_variational -u True', 2, 47, 'VBLL-beta (figure script reads this as mlp_vbll)'),
    # Outside the ANOVA roster, but needed:
    'qrf':                      ('-m qrf -u True',                                    3, 23, 'not in the ANOVA (rho 0.996 with rf) but the best error-ranker'),
    'gauche_rbf':               ('-m gauche --kernel rbf -u True',                    3, 47, 'RBF GP on EVERY rep, so the GP can finally enter the cross-rep ANOVA'),
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

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
# Guarded: under `set -u` an unset CONDA_PREFIX aborts here, before python.
export LD_LIBRARY_PATH="${{CONDA_PREFIX:-}}/lib:${{LD_LIBRARY_PATH:-}}"

cd {qsar_dir}
. setup.sh

# The binary carries the held-out-noise fix. Refuse to run without it rather
# than silently regenerating the same invalid results.
if [ ! -x rust/target/release/rust_processor ]; then
    echo "ERROR: rust/target/release/rust_processor missing. Run:"
    echo "  cd {qsar_dir}/rust && cargo build --release"
    exit 2
fi

cd scripts

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
    ap.add_argument('--out-dir', default=str(Path(__file__).parent))
    args = ap.parse_args()

    out = Path(args.out_dir)
    n_tasks = len(STRATEGIES) * len(REPS)
    n_sig = len(SIGMAS.split())
    chosen = {k: v for k, v in MODELS.items() if not args.models or k in args.models}

    written = []
    for model, (flags, tier, hours, note) in chosen.items():
        script_name = f'qm9_{model}.sh'
        (out / script_name).write_text(TEMPLATE.format(
            model=model, note=note or model, jobslug=model,
            cpus=8, mem='128G', hours=hours, flags=flags,
            qsar_dir=QSAR_DIR, sigmas=SIGMAS, boot=args.bootstrapping,
            strategies=' '.join(STRATEGIES), reps=' '.join(REPS),
            n_st=len(STRATEGIES), n_rep=len(REPS), n_tasks=n_tasks,
            n_sig=n_sig, runs=n_sig * args.bootstrapping,
            last=n_tasks - 1, throttle=args.throttle, script_name=script_name))
        (out / script_name).chmod(0o755)
        written.append((tier, script_name, model, hours))

    total = len(written) * n_tasks
    print(f"Wrote {len(written)} array scripts, {n_tasks} tasks each = {total} tasks")
    print(f"Each task: {n_sig} noise levels x {args.bootstrapping} replicates "
          f"= {n_sig * args.bootstrapping} training runs")
    print(f"Grid total: {total * n_sig * args.bootstrapping:,} training runs")
    for tier, name in [(1, 'ANOVA roster — tree and deterministic'),
                       (2, 'ANOVA roster — Bayesian networks'),
                       (3, 'outside the ANOVA (uncertainty + GP)')]:
        rows = [w for w in written if w[0] == tier]
        if rows:
            print(f"\n  Tier {tier} ({name}):")
            for _, nm, m, h in rows:
                print(f"    {nm:34s} {m:26s} --time={h}:59:00")


if __name__ == '__main__':
    main()
