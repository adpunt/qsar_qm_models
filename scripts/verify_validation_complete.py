#!/usr/bin/env python3
"""
Verify the three laboratory datasets' noise-robustness runs are complete.

Run on server: python verify_validation_complete.py /path/to/kirby/results/validation

WHAT THIS EXPECTS, AND WHY IT USED TO PASS NOTHING
--------------------------------------------------
Four things in this file described a run that no longer exists, and each one on
its own was enough to make every dataset report incomplete whatever was on disk:

  * three directory names -- openadmet_logd, openadmet_caco2, herg_fluid -- that
    the laboratory runner does not create. It writes logd, caco2 and herg
    (KIRBy tests/alternative_data_noise_robustness.py, --results-root / name), so
    this script reported NO VALIDATION DATA FOUND against a finished run
  * a required column 'strategy'; the runner writes the noise condition under
    'noise_type'
  * a required column 'NDS_r2'; NDS was replaced by auc_norm in July 2026 and
    the runner writes auc_norm
  * a completeness rule of "at least five noise conditions", left over from the
    retired six-condition design. The settled main grid has four, so the rule
    could never be met again

The condition names are read from noise_conditions.json, the one settled list,
rather than restated -- restating them is how the six retired names survived
here for two days after the injectors stopped producing any of them.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# The three directories the laboratory runner creates under --results-root.
DATASETS = ['logd', 'caco2', 'herg']
DATASET_LABELS = {
    'logd': 'LogD',
    'caco2': 'Caco2_Efflux',
    'herg': 'hERG-Ki',
}

# The settled noise conditions, read not restated. The main grid is what every
# dataset is expected to carry; the depth-only conditions run on a narrower
# selection, so they are reported when present and never demanded.
_SETTLED_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
_SETTLED = json.loads(_SETTLED_FILE.read_text())
MAIN_GRID_CONDITIONS = [c['name'] for c in _SETTLED['stage_1_full_grid']]
DEPTH_CONDITIONS = [c['name'] for c in _SETTLED['stage_2_depth_only']]

# Columns the runner writes into summary.csv (KIRBy
# alternative_data_noise_robustness.py, the summary_rows dict).
REQUIRED_SUMMARY_COLS = ['dataset', 'model', 'rep', 'noise_type',
                         'baseline_r2', 'auc_norm']
REQUIRED_UNCERTAINTY_COLS = ['sigma', 'y_true', 'y_pred', 'uncertainty']
OPTIONAL_UNCERTAINTY_COLS = ['injected_noise', 'aleatoric_uncertainty', 'epistemic_uncertainty']


def check_summary_file(filepath):
    """Columns, and which settled noise conditions the summary actually carries.

    Completeness is a SET test against the settled conditions, not a count. The
    count rule it replaces ("at least five") was the retired six-condition
    design's, and the settled main grid has four, so nothing could satisfy it.
    """
    try:
        df = pd.read_csv(filepath)

        missing_cols = [c for c in REQUIRED_SUMMARY_COLS if c not in df.columns]

        conditions_found = (set(df['noise_type'].unique())
                            if 'noise_type' in df.columns else set())
        missing_conditions = [c for c in MAIN_GRID_CONDITIONS
                              if c not in conditions_found]
        extra_conditions = sorted(conditions_found
                                  - set(MAIN_GRID_CONDITIONS)
                                  - set(DEPTH_CONDITIONS))
        models_found = set(df['model'].unique()) if 'model' in df.columns else set()
        reps_found = set(df['rep'].unique()) if 'rep' in df.columns else set()

        return {
            'exists': True,
            'rows': len(df),
            'missing_cols': missing_cols,
            'conditions_found': sorted(conditions_found),
            'missing_conditions': missing_conditions,
            # A condition in the file that the settled list does not name at
            # all: either a retired one still being run, or a new one nobody
            # recorded. Both are worth saying out loud.
            'unrecognised_conditions': extra_conditions,
            'models': len(models_found),
            'reps': len(reps_found),
            'complete': not missing_cols and not missing_conditions,
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
            print(f"  Noise conditions: {status['conditions_found']}")
            print(f"  Models: {status['models']}")
            print(f"  Representations: {status['reps']}")
            if status['missing_cols']:
                print(f"  Missing columns: {status['missing_cols']}")
            if status['missing_conditions']:
                print(f"  Settled conditions absent: {status['missing_conditions']}")
            if status['unrecognised_conditions']:
                print(f"  Conditions no settled list names: "
                      f"{status['unrecognised_conditions']}")
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
            print(f"    Conditions: {status['conditions_found']}")
            print(f"    Models: {status['models']}, Reps: {status['reps']}")
            if status['missing_cols']:
                print(f"    Missing columns: {status['missing_cols']}")
            if status['missing_conditions']:
                print(f"    Settled conditions absent: {status['missing_conditions']}")
            if status['unrecognised_conditions']:
                print(f"    Conditions no settled list names: "
                      f"{status['unrecognised_conditions']}")
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

    # Check what the analysis script needs. v2 is the live figure script; v1
    # (generate_paper_figures.py) still computes the retired NDS measure and its
    # output directory is stale.
    print("\n" + "=" * 70)
    print("FOR generate_paper_figures_v2.py")
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
