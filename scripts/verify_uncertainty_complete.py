#!/usr/bin/env python3
"""
Verify all uncertainty experiments completed.
Run on server: python verify_uncertainty_complete.py /data/stat-cadd/scat9264/qsar_qm_models/results
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Expected setup (from generate_uncertainty_slurm_scripts.py)
STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
REPS = ['pdv', 'sns']  # Only 2 reps for uncertainty
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Uncertainty models
NATIVE_MODELS = ['qrf', 'ngboost', 'gauche']
BNN_MODELS = ['dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational',
              'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational']
CONFORMAL_BASES = ['rf', 'qrf', 'dnn']

# Required columns for uncertainty analysis
REQUIRED_COLS = ['sigma', 'y_pred_mean', 'y_true', 'injected_noise']
OPTIONAL_COLS = ['epistemic_uncertainty', 'aleatoric_uncertainty', 'y_pred_std_calibrated']


def check_uncertainty_file(filepath):
    """Check if uncertainty file has required columns and coverage."""
    try:
        df = pd.read_csv(filepath)

        # Check sigmas
        found_sigmas = set(df['sigma'].unique())
        missing_sigmas = set(SIGMAS) - found_sigmas

        # Check required columns
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

        # Check optional columns
        has_epistemic = 'epistemic_uncertainty' in df.columns
        has_aleatoric = 'aleatoric_uncertainty' in df.columns
        has_injected_noise = 'injected_noise' in df.columns

        return {
            'exists': True,
            'rows': len(df),
            'sigmas_found': len(found_sigmas),
            'sigmas_missing': sorted(missing_sigmas),
            'missing_required_cols': missing_cols,
            'has_epistemic': has_epistemic,
            'has_aleatoric': has_aleatoric,
            'has_injected_noise': has_injected_noise,
            'complete': len(missing_sigmas) == 0 and len(missing_cols) == 0
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results")

    print("=" * 70)
    print("UNCERTAINTY EXPERIMENT VERIFICATION")
    print("=" * 70)

    all_expected = []

    # 1. Native uncertainty models
    for strategy in STRATEGIES:
        for model in NATIVE_MODELS:
            for rep in REPS:
                filename = f"uncertainty_{strategy}_{rep}_{model}.csv"
                filepath = results_dir / filename
                all_expected.append(('native', strategy, model, rep, filepath))

    # 2. BNN models
    for strategy in STRATEGIES:
        for model in BNN_MODELS:
            for rep in REPS:
                filename = f"uncertainty_{strategy}_{rep}_{model}.csv"
                filepath = results_dir / filename
                all_expected.append(('bnn', strategy, model, rep, filepath))

    # 3. Conformal models
    for strategy in STRATEGIES:
        for base in CONFORMAL_BASES:
            for rep in REPS:
                filename = f"uncertainty_{strategy}_{rep}_conformal_{base}.csv"
                filepath = results_dir / filename
                all_expected.append(('conformal', strategy, f'conformal_{base}', rep, filepath))

    # Check all files
    complete = []
    incomplete = []
    missing = []

    for category, strategy, model, rep, filepath in all_expected:
        status = check_uncertainty_file(filepath)

        if not status.get('exists'):
            missing.append((category, strategy, model, rep, filepath.name))
        elif not status.get('complete'):
            incomplete.append((category, strategy, model, rep, filepath.name, status))
        else:
            complete.append((category, strategy, model, rep, filepath.name, status))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(all_expected)
    print(f"\nTotal expected: {total}")
    print(f"Complete: {len(complete)} ({100*len(complete)/total:.1f}%)")
    print(f"Incomplete: {len(incomplete)}")
    print(f"Missing: {len(missing)}")

    # Check for epistemic/aleatoric coverage
    has_decomp = sum(1 for _, _, _, _, _, s in complete if s.get('has_epistemic') and s.get('has_aleatoric'))
    print(f"\nWith epistemic/aleatoric decomposition: {has_decomp}/{len(complete)}")

    has_noise = sum(1 for _, _, _, _, _, s in complete if s.get('has_injected_noise'))
    print(f"With injected_noise column: {has_noise}/{len(complete)}")

    if missing:
        print("\n" + "=" * 70)
        print("MISSING FILES")
        print("=" * 70)
        by_category = defaultdict(list)
        for cat, strat, model, rep, fname in missing:
            by_category[cat].append((strat, model, rep))

        for cat, items in sorted(by_category.items()):
            print(f"\n{cat.upper()}: {len(items)} missing")
            for strat, model, rep in items[:10]:
                print(f"  {strat}/{model}/{rep}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")

    if incomplete:
        print("\n" + "=" * 70)
        print("INCOMPLETE FILES")
        print("=" * 70)
        for cat, strat, model, rep, fname, status in incomplete[:20]:
            print(f"  {fname}")
            if status.get('sigmas_missing'):
                print(f"    Sigmas missing: {status['sigmas_missing']}")
            if status.get('missing_required_cols'):
                print(f"    Columns missing: {status['missing_required_cols']}")

    # Return exit code
    if missing or incomplete:
        print("\n❌ VERIFICATION FAILED - gaps remain")
        return 1
    else:
        print("\n✓ ALL UNCERTAINTY EXPERIMENTS COMPLETE")
        return 0


if __name__ == "__main__":
    sys.exit(main())
