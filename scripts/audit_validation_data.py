#!/usr/bin/env python
"""Audit validation data in KIRBy alternative_full/ directory.

Checks:
1. Which directories exist and what format they contain
2. Whether data uses scaffold CV folds
3. Baseline R² per model/rep/dataset
4. Which model×rep×strategy×dataset combos are missing
5. Which experiments need re-running

Usage:
    python audit_validation_data.py --validation-dir /path/to/alternative_full
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


# Expected models and representations for full validation coverage
EXPECTED_MODELS = ['RF', 'QRF', 'XGBoost', 'DNN', 'GP', 'NGBoost', 'SVM', 'LightGBM']
EXPECTED_REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']
EXPECTED_STRATEGIES = ['legacy', 'hetero', 'outlier', 'quantile', 'threshold', 'valprop']
# Classification datasets — skip for regression NDS analysis
CLASSIFICATION_DIRS = {'herg_fluid'}
# Duplicate directories — map to canonical name
DIR_ALIASES = {
    'openadmet_logd': 'OpenADMET-LogD',
    'logd': 'OpenADMET-LogD',
    'openadmet_caco2': 'OpenADMET-Caco2_Efflux',
    'caco2': 'OpenADMET-Caco2_Efflux',
    'herg': 'ChEMBL-hERG-Ki',
}


def audit_directory(subdir):
    """Audit a single dataset directory."""
    info = {
        'dir_name': subdir.name,
        'has_all_results': False,
        'has_summary': False,
        'is_classification': False,
        'has_fold_column': False,
        'has_scaffold_info': False,
        'columns': [],
        'n_rows': 0,
        'models': [],
        'reps': [],
        'strategies': [],
        'sigma_levels': [],
        'dataset_name': None,
        'baseline_r2': {},
    }

    results_file = subdir / 'all_results.csv'
    summary_file = subdir / 'summary.csv'

    if results_file.exists():
        info['has_all_results'] = True
        df = pd.read_csv(results_file)
    elif summary_file.exists():
        info['has_summary'] = True
        df = pd.read_csv(summary_file)
    else:
        return info

    info['columns'] = list(df.columns)
    info['n_rows'] = len(df)
    info['is_classification'] = 'r2' not in df.columns and ('accuracy' in df.columns or 'auc' in df.columns)

    if 'dataset' in df.columns:
        info['dataset_name'] = df['dataset'].unique().tolist()
    if 'fold' in df.columns:
        info['has_fold_column'] = True
        info['n_folds'] = df['fold'].nunique()
    if 'model' in df.columns:
        info['models'] = sorted(df['model'].unique().tolist())
    if 'rep' in df.columns:
        info['reps'] = sorted(df['rep'].unique().tolist())
    if 'strategy' in df.columns:
        info['strategies'] = sorted(df['strategy'].unique().tolist())
    if 'sigma' in df.columns:
        info['sigma_levels'] = sorted(df['sigma'].unique().tolist())

    # Compute baseline R² per model/rep
    if 'r2' in df.columns and 'sigma' in df.columns and 'model' in df.columns and 'rep' in df.columns:
        baseline = df[df['sigma'] == 0.0]
        if len(baseline) > 0:
            for (model, rep), group in baseline.groupby(['model', 'rep']):
                mean_r2 = group['r2'].mean()
                info['baseline_r2'][(model, rep)] = mean_r2

    return info


def check_scaffold_splits(subdir):
    """Check if scaffold splits were used by examining config or data patterns."""
    # Check for scaffold_assignments or config files
    config_files = list(subdir.glob('*.json')) + list(subdir.glob('*.yaml')) + list(subdir.glob('*.yml'))
    scaffold_files = list(subdir.glob('*scaffold*'))
    return {
        'config_files': [f.name for f in config_files],
        'scaffold_files': [f.name for f in scaffold_files],
    }


def main():
    parser = argparse.ArgumentParser(description='Audit validation data')
    parser.add_argument('--validation-dir', required=True, help='Path to alternative_full/')
    args = parser.parse_args()

    val_dir = Path(args.validation_dir)
    if not val_dir.exists():
        print(f"ERROR: {val_dir} does not exist")
        return

    print("=" * 80)
    print("VALIDATION DATA AUDIT")
    print(f"Directory: {val_dir}")
    print("=" * 80)

    # Phase 1: Directory inventory
    print("\n## 1. Directory Inventory\n")
    all_info = {}
    for subdir in sorted(val_dir.iterdir()):
        if not subdir.is_dir():
            continue
        info = audit_directory(subdir)
        all_info[subdir.name] = info

        canonical = DIR_ALIASES.get(subdir.name, subdir.name)
        skip_tag = ""
        if subdir.name in CLASSIFICATION_DIRS:
            skip_tag = " [CLASSIFICATION — skip]"
        elif info['is_classification']:
            skip_tag = " [CLASSIFICATION — no r2]"

        print(f"### {subdir.name}/ → {canonical}{skip_tag}")
        if info['has_all_results']:
            print(f"  File: all_results.csv ({info['n_rows']} rows)")
        elif info['has_summary']:
            print(f"  File: summary.csv ({info['n_rows']} rows)")
        else:
            print(f"  No data files found!")
            continue

        print(f"  Columns: {', '.join(info['columns'])}")
        if info['dataset_name']:
            print(f"  Dataset label in CSV: {info['dataset_name']}")
        print(f"  Models: {info['models']}")
        print(f"  Reps: {info['reps']}")
        print(f"  Strategies: {info['strategies']}")
        if info['sigma_levels']:
            print(f"  Sigma levels: {len(info['sigma_levels'])} ({min(info['sigma_levels']):.1f}–{max(info['sigma_levels']):.1f})")
        print(f"  Has fold column: {info['has_fold_column']}", end="")
        if info['has_fold_column']:
            print(f" ({info['n_folds']} folds)")
        else:
            print(" ⚠ NO FOLD COLUMN — may not be scaffold CV")
        print()

    # Phase 2: Baseline R² summary
    print("\n## 2. Baseline R² by Model × Rep × Dataset\n")
    print(f"{'Dataset':<25} {'Model':<12} {'Rep':<20} {'Baseline R²':>12}")
    print("-" * 72)

    for dir_name, info in sorted(all_info.items()):
        if info['is_classification'] or dir_name in CLASSIFICATION_DIRS:
            continue
        canonical = DIR_ALIASES.get(dir_name, dir_name)
        for (model, rep), r2 in sorted(info['baseline_r2'].items()):
            flag = "  ⚠ LOW" if r2 < 0.4 else ("  ~ marginal" if r2 < 0.6 else "")
            print(f"{canonical:<25} {model:<12} {rep:<20} {r2:>12.4f}{flag}")

    # Phase 3: Duplicate directory analysis
    print("\n\n## 3. Duplicate Directory Analysis\n")
    canonical_dirs = defaultdict(list)
    for dir_name, info in all_info.items():
        if dir_name in CLASSIFICATION_DIRS or info['is_classification']:
            continue
        canonical = DIR_ALIASES.get(dir_name, dir_name)
        canonical_dirs[canonical].append(dir_name)

    for canonical, dirs in sorted(canonical_dirs.items()):
        if len(dirs) > 1:
            print(f"⚠ {canonical} has duplicate directories: {dirs}")
            for d in dirs:
                info = all_info[d]
                fold_info = f"fold column: {'YES' if info['has_fold_column'] else 'NO'}"
                print(f"    {d}/: {info['n_rows']} rows, {len(info['models'])} models, {fold_info}")
            print()
        else:
            print(f"  {canonical}: {dirs[0]}/")

    # Phase 4: Coverage matrix (what's missing)
    print("\n\n## 4. Coverage Matrix (Missing Experiments)\n")
    for canonical in sorted(set(DIR_ALIASES.values())):
        dirs = canonical_dirs.get(canonical, [])
        if not dirs:
            print(f"\n### {canonical}: NO DATA")
            continue

        # Merge data from all duplicate dirs
        all_models = set()
        all_reps = set()
        all_strategies = set()
        seen_combos = set()

        for d in dirs:
            info = all_info[d]
            for m in info['models']:
                all_models.add(m)
            for r in info['reps']:
                all_reps.add(r)
            for s in info['strategies']:
                all_strategies.add(s)
            # Track model×rep×strategy combos
            results_file = Path(args.validation_dir) / d / 'all_results.csv'
            if results_file.exists():
                df = pd.read_csv(results_file)
                if 'model' in df.columns and 'rep' in df.columns and 'strategy' in df.columns:
                    for _, row in df[['model', 'rep', 'strategy']].drop_duplicates().iterrows():
                        seen_combos.add((row['model'], row['rep'], row['strategy']))

        print(f"\n### {canonical}")
        print(f"  Models present: {sorted(all_models)}")
        print(f"  Reps present: {sorted(all_reps)}")
        print(f"  Strategies present: {sorted(all_strategies)}")

        # Check what's missing vs expected
        regression_strategies = [s for s in all_strategies if s in EXPECTED_STRATEGIES or s in
                                 {'legacy', 'hetero', 'outlier', 'quantile', 'threshold', 'valprop'}]
        classification_strategies = [s for s in all_strategies if s not in regression_strategies]

        if classification_strategies:
            print(f"  ⚠ Classification strategies found: {sorted(classification_strategies)}")
            print(f"    Regression strategies: {sorted(regression_strategies)}")

        missing = []
        for model in EXPECTED_MODELS:
            for rep in EXPECTED_REPS:
                for strategy in EXPECTED_STRATEGIES:
                    if (model, rep, strategy) not in seen_combos:
                        missing.append((model, rep, strategy))

        if missing:
            # Group by model for readable output
            missing_by_model = defaultdict(list)
            for m, r, s in missing:
                missing_by_model[m].append((r, s))

            print(f"\n  Missing experiments ({len(missing)} combos):")
            for model in EXPECTED_MODELS:
                if model in missing_by_model:
                    items = missing_by_model[model]
                    if len(items) == len(EXPECTED_REPS) * len(EXPECTED_STRATEGIES):
                        print(f"    {model}: ALL missing (not run yet)")
                    else:
                        reps_missing = set(r for r, s in items)
                        strats_missing = set(s for r, s in items)
                        print(f"    {model}: {len(items)} missing — reps: {sorted(reps_missing)}, strategies: {sorted(strats_missing)}")
        else:
            print(f"\n  ✓ Full coverage!")

    # Phase 5: Scaffold split check
    print("\n\n## 5. Scaffold Split Verification\n")
    print("Directories WITH fold column (likely scaffold CV):")
    for dir_name, info in sorted(all_info.items()):
        if info['has_fold_column']:
            print(f"  ✓ {dir_name}/ ({info.get('n_folds', '?')} folds)")

    print("\nDirectories WITHOUT fold column (may need re-running):")
    for dir_name, info in sorted(all_info.items()):
        if not info['has_fold_column'] and (info['has_all_results'] or info['has_summary']):
            if dir_name in CLASSIFICATION_DIRS or info['is_classification']:
                continue
            print(f"  ⚠ {dir_name}/ — no fold column, verify scaffold splits were used")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
