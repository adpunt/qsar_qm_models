#!/usr/bin/env python3
"""Merge validation rerun results with existing PDV data.

Run from: /data/stat-cadd/scat9264/KIRBy/tests
After all SLURM jobs have completed.

Reads:
  - Existing PDV data from results/validation/{caco2,logd}/all_results.csv
  - New data from results/validation_rerun/{rep}_{dataset}/{dataset}/all_results*.csv

Writes:
  - results/validation/{dataset}/all_results.csv (backed up first)

Usage:
    python merge_results.py --dry-run   # check first
    python merge_results.py             # do it
"""
import argparse
import glob
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

VALIDATION_DIR = Path('results/validation')
RERUN_DIR = Path('results/validation_rerun')
DATASETS = ['logd', 'caco2', 'herg']


def merge_dataset(dataset, dry_run=False):
    print(f"\n=== {dataset} ===")
    dataset_dir = VALIDATION_DIR / dataset

    # Load existing
    existing = pd.DataFrame()
    for fname in ['all_results.csv', 'all_results_partial.csv']:
        p = dataset_dir / fname
        if p.exists():
            existing = pd.read_csv(p)
            print(f"  Existing ({fname}): {len(existing)} rows, reps={sorted(existing['rep'].unique())}, models={sorted(existing['model'].unique())}")
            break

    # Load all rerun data for this dataset
    rerun_dfs = []
    for d in sorted(RERUN_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith(f'_{dataset}'):
            continue
        for csv in sorted((d / dataset).glob('all_results*.csv')):
            df = pd.read_csv(csv)
            if len(df) > 0:
                rerun_dfs.append(df)
                print(f"  Rerun {d.name}: {len(df)} rows, models={sorted(df['model'].unique())}, reps={sorted(df['rep'].unique())}")

    if not rerun_dfs:
        print(f"  No rerun data found")
        return

    rerun = pd.concat(rerun_dfs, ignore_index=True)

    # Deduplicate rerun data (multiple models wrote to same rep_dataset dir)
    dedup_cols = [c for c in ['model', 'rep', 'strategy', 'sigma', 'fold'] if c in rerun.columns]
    rerun = rerun.drop_duplicates(subset=dedup_cols, keep='last')
    print(f"  Rerun total: {len(rerun)} rows")

    # Combine
    if not existing.empty:
        # For herg: rerun includes fresh PDV, so remove existing PDV rows that overlap
        # For logd/caco2: rerun is non-PDV only, no overlap expected
        rerun_keys = rerun[['model', 'rep']].drop_duplicates()
        mask = pd.Series(False, index=existing.index)
        for _, row in rerun_keys.iterrows():
            mask |= (existing['model'] == row['model']) & (existing['rep'] == row['rep'])
        kept = existing[~mask]
        combined = pd.concat([kept, rerun], ignore_index=True)
        print(f"  Kept {len(kept)} existing + {len(rerun)} new = {len(combined)} total")
    else:
        combined = rerun

    sort_cols = [c for c in ['dataset', 'model', 'rep', 'strategy', 'sigma', 'fold'] if c in combined.columns]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    print(f"  Final: {len(combined)} rows")
    print(f"  Models: {sorted(combined['model'].unique())}")
    print(f"  Reps: {sorted(combined['rep'].unique())}")

    if dry_run:
        print(f"  [DRY RUN] Would write to {dataset_dir}/all_results.csv")
    else:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # Backup
        for fname in ['all_results.csv', 'all_results_partial.csv']:
            src = dataset_dir / fname
            if src.exists():
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                dst = dataset_dir / f'{fname}.backup_{ts}'
                shutil.copy2(src, dst)
                print(f"  Backed up {fname}")
        combined.to_csv(dataset_dir / 'all_results.csv', index=False)
        print(f"  Wrote {len(combined)} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not RERUN_DIR.exists():
        print(f"ERROR: {RERUN_DIR} not found")
        sys.exit(1)

    for dataset in DATASETS:
        merge_dataset(dataset, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == '__main__':
    main()
