#!/usr/bin/env python3
"""
Analyze anova_* file coverage to find gaps.
Run on server: python analyze_anova_coverage.py /data/stat-cadd/scat9264/qsar_qm_models/results
"""

import sys
from pathlib import Path
from collections import defaultdict

# Expected combinations
STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
REPS = ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn']
MODELS = ['rf', 'xgboost', 'qrf', 'ngboost', 'dnn', 'mlp', 'gauche']
BNN_MODELS = ['dnn_bnn_full', 'mlp_bnn_full']  # Only for legacy

def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    # Parse existing files
    existing = defaultdict(set)  # strategy -> set of (rep, model)
    uncertainty_files = set()

    for f in results_dir.glob("anova_*.csv"):
        name = f.stem  # anova_strategy_rep_model
        if '_uncertainty_values' in name:
            uncertainty_files.add(name.replace('_uncertainty_values', ''))
            continue

        parts = name.split('_')
        if len(parts) >= 4:
            strategy = parts[1]
            # Handle bnn_full models
            if 'bnn_full' in name:
                model = '_'.join(parts[3:])
                rep = parts[2]
            else:
                rep = parts[2]
                model = '_'.join(parts[3:])
            existing[strategy].add((rep, model))

    print("=" * 70)
    print("ANOVA COVERAGE ANALYSIS")
    print("=" * 70)

    # Summary per strategy
    print("\nCOVERAGE BY STRATEGY:")
    print("-" * 70)

    all_missing = []

    for strategy in STRATEGIES:
        expected = set()
        for rep in REPS:
            for model in MODELS:
                expected.add((rep, model))
            # BNN only for legacy
            if strategy == 'legacy':
                for model in BNN_MODELS:
                    expected.add((rep, model))

        have = existing.get(strategy, set())
        missing = expected - have

        pct = len(have) / len(expected) * 100 if expected else 0
        print(f"{strategy:15} {len(have):3}/{len(expected):3} ({pct:5.1f}%)", end="")

        if missing:
            print(f"  Missing: {len(missing)}")
            for rep, model in sorted(missing):
                all_missing.append((strategy, rep, model))
        else:
            print("  COMPLETE")

    # Detailed missing
    if all_missing:
        print("\n" + "=" * 70)
        print("MISSING COMBINATIONS")
        print("=" * 70)

        by_strategy = defaultdict(list)
        for s, r, m in all_missing:
            by_strategy[s].append((r, m))

        for strategy, combos in sorted(by_strategy.items()):
            print(f"\n{strategy}:")
            for rep, model in sorted(combos):
                print(f"  {rep}/{model}")

    # Uncertainty files
    print("\n" + "=" * 70)
    print("UNCERTAINTY VALUE FILES")
    print("=" * 70)
    print(f"Total: {len(uncertainty_files)}")
    for f in sorted(uncertainty_files):
        print(f"  {f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total anova files: {sum(len(v) for v in existing.values())}")
    print(f"Total missing: {len(all_missing)}")
    print(f"Uncertainty files: {len(uncertainty_files)}")

if __name__ == "__main__":
    main()
