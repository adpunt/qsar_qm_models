#!/usr/bin/env python3
"""
Verify validation experiments (KIRBy) completed.
Run on server: python verify_validation_complete.py /path/to/kirby/results_validation

Expected datasets: LogD, Caco2_Efflux, hERG-Ki
"""

import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Expected setup based on SESSION_LOG.md
DATASETS = ['openadmet_logd', 'openadmet_caco2', 'herg_fluid']  # Directory names
DATASET_LABELS = {
    'openadmet_logd': 'LogD',
    'openadmet_caco2': 'Caco2_Efflux',
    'herg_fluid': 'hERG-Ki',
}

STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
REPS = ['ecfp4', 'pdv', 'sns', 'mhggnn']  # MHG-GNN-pretrained
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Models
REGRESSION_MODELS = ['rf', 'qrf', 'xgboost', 'dnn', 'dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational']
GP_MODELS = ['gp']  # PDV only

# Required columns
REQUIRED_SUMMARY_COLS = ['dataset', 'model', 'rep', 'strategy', 'baseline_r2', 'NDS_r2']
REQUIRED_UNCERTAINTY_COLS = ['sigma', 'y_true', 'y_pred', 'uncertainty']
OPTIONAL_UNCERTAINTY_COLS = ['injected_noise', 'aleatoric_uncertainty', 'epistemic_uncertainty']


def check_summary_file(filepath):
    """Check if summary file has required columns and coverage."""
    try:
        df = pd.read_csv(filepath)

        # Check required columns
        missing_cols = [c for c in REQUIRED_SUMMARY_COLS if c not in df.columns]

        # Check coverage
        strategies_found = set(df['strategy'].unique()) if 'strategy' in df.columns else set()
        models_found = set(df['model'].unique()) if 'model' in df.columns else set()
        reps_found = set(df['rep'].unique()) if 'rep' in df.columns else set()

        return {
            'exists': True,
            'rows': len(df),
            'missing_cols': missing_cols,
            'strategies': len(strategies_found),
            'models': len(models_found),
            'reps': len(reps_found),
            'complete': len(missing_cols) == 0 and len(strategies_found) >= 5 and len(models_found) >= 5
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }


def check_uncertainty_files(dataset_dir):
    """Check uncertainty files in a dataset directory."""
    results = {'found': 0, 'with_decomposition': 0, 'with_injected_noise': 0, 'files': []}

    for f in dataset_dir.glob("*_uncertainty_values.csv"):
        try:
            df = pd.read_csv(f)

            has_decomp = ('aleatoric_uncertainty' in df.columns and
                         'epistemic_uncertainty' in df.columns)
            has_noise = 'injected_noise' in df.columns

            results['found'] += 1
            if has_decomp:
                results['with_decomposition'] += 1
            if has_noise:
                results['with_injected_noise'] += 1

            results['files'].append({
                'name': f.name,
                'rows': len(df),
                'has_decomp': has_decomp,
                'has_noise': has_noise
            })
        except Exception as e:
            results['files'].append({'name': f.name, 'error': str(e)})

    return results


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results/validation")

    print("=" * 70)
    print("VALIDATION EXPERIMENT VERIFICATION")
    print(f"Directory: {results_dir}")
    print("=" * 70)

    if not results_dir.exists():
        print(f"\nERROR: Directory does not exist: {results_dir}")
        return 1

    # Check for combined summary
    combined = results_dir / 'combined_summary.csv'
    if combined.exists():
        print("\n[COMBINED SUMMARY]")
        status = check_summary_file(combined)
        if status['exists']:
            print(f"  Rows: {status['rows']}")
            print(f"  Strategies: {status['strategies']}")
            print(f"  Models: {status['models']}")
            print(f"  Representations: {status['reps']}")
            if status['missing_cols']:
                print(f"  Missing columns: {status['missing_cols']}")
        else:
            print(f"  Error: {status.get('error', 'unknown')}")
    else:
        print("\n[COMBINED SUMMARY] Not found")

    # Check each dataset
    dataset_status = {}

    for ds_name in DATASETS:
        ds_dir = results_dir / ds_name
        label = DATASET_LABELS.get(ds_name, ds_name)

        print(f"\n[{label.upper()}]")
        print("-" * 50)

        if not ds_dir.exists():
            print(f"  Directory not found: {ds_dir}")
            dataset_status[ds_name] = {'exists': False}
            continue

        # Check summary.csv
        summary = ds_dir / 'summary.csv'
        if summary.exists():
            status = check_summary_file(summary)
            dataset_status[ds_name] = status
            print(f"  summary.csv: {status['rows']} rows")
            print(f"    Strategies: {status['strategies']}, Models: {status['models']}, Reps: {status['reps']}")
            if status['missing_cols']:
                print(f"    Missing columns: {status['missing_cols']}")
        else:
            print("  summary.csv: NOT FOUND")
            dataset_status[ds_name] = {'exists': False, 'error': 'No summary.csv'}

        # Check uncertainty files
        unc_status = check_uncertainty_files(ds_dir)
        print(f"  Uncertainty files: {unc_status['found']} found")
        if unc_status['found'] > 0:
            print(f"    With aleatoric/epistemic: {unc_status['with_decomposition']}")
            print(f"    With injected_noise: {unc_status['with_injected_noise']}")

    # Overall summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    complete = sum(1 for ds, s in dataset_status.items() if s.get('complete'))
    found = sum(1 for ds, s in dataset_status.items() if s.get('exists'))

    print(f"\nDatasets expected: {len(DATASETS)}")
    print(f"Datasets found: {found}")
    print(f"Datasets complete: {complete}")

    if found < len(DATASETS):
        missing = [DATASET_LABELS.get(ds, ds) for ds, s in dataset_status.items() if not s.get('exists')]
        print(f"\nMissing datasets: {', '.join(missing)}")

    # Check what the analysis script needs
    print("\n" + "=" * 70)
    print("FOR generate_paper_figures.py")
    print("=" * 70)

    if combined.exists():
        print("\n✓ combined_summary.csv exists - Figure 3C (cross-dataset) can be generated")
    else:
        if all(s.get('exists') for s in dataset_status.values()):
            print("\n✓ Individual summaries exist - will be combined automatically")
        else:
            print("\n✗ Missing data - Figure 3C will show partial results")

    # Return exit code
    if complete >= len(DATASETS):
        print("\n✓ ALL VALIDATION EXPERIMENTS COMPLETE")
        return 0
    elif found > 0:
        print(f"\n⚠ PARTIAL DATA - {found}/{len(DATASETS)} datasets found")
        return 0
    else:
        print("\n✗ NO VALIDATION DATA FOUND")
        return 1


if __name__ == "__main__":
    sys.exit(main())
