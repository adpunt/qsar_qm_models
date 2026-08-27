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
# Directory labels, not command-line names. The runner writes hERG output to a
# directory called 'herg' while its --datasets flag spells it 'herg_ki'
# (RERUN_PLAN.md, 'hERG was never cut').
DATASETS = ['logd', 'caco2', 'herg']


def condition_column_of(df):
    """Which column names the noise condition.

    It is `noise_type`. It used to be `strategy`, and the rename is why this
    exists as one function rather than a literal in three places.
    """
    col = next((c for c in ('noise_type', 'strategy') if c in df.columns), None)
    assert col is not None, (
        f"the results have no condition column -- looked for 'noise_type' and "
        f"'strategy', found {sorted(df.columns)}. Deduplicating without one "
        f"collapses every noise condition onto a single row.")
    return col


def deduplicate(rerun):
    """Drop repeated (model, rep, condition, level, fold) rows, keeping the last.

    Separate from merge_dataset so a test can call it. The bug it now prevents
    was invisible from the outside: the column list named `strategy` after the
    runner had renamed it to `noise_type`, and the list comprehension that
    filters to present columns dropped it silently rather than raising. The key
    became (model, rep, sigma, fold), so the four noise conditions on one cell
    deduplicated against each other. Four rows in, one row out.
    """
    condition_col = condition_column_of(rerun)
    dedup_cols = [c for c in ['model', 'rep', condition_col, 'sigma', 'fold']
                  if c in rerun.columns]
    return rerun.drop_duplicates(subset=dedup_cols, keep='last'), dedup_cols


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

    # Deduplicate rerun data (multiple models wrote to same rep_dataset dir).
    #
    # The condition column is `noise_type`. It used to be `strategy`, and this
    # list still said `strategy` after the rename -- which did not raise, because
    # the comprehension silently drops a column that is not there. The key became
    # (model, rep, sigma, fold), so FOUR noise conditions on one cell deduplicated
    # against each other and three of them were discarded, keep='last'. Measured:
    # four rows in, one row out. Both names are accepted so old files still merge.
    condition_col = condition_column_of(rerun)
    before = len(rerun)
    rerun, dedup_cols = deduplicate(rerun)
    print(f"  Rerun total: {len(rerun)} rows "
          f"(deduplicated on {dedup_cols}, dropped {before - len(rerun)})")
    print(f"  Conditions present: {sorted(rerun[condition_col].unique())}")

    # Combine
    if not existing.empty:
        # For herg: rerun includes fresh PDV, so remove existing PDV rows that overlap
        # For logd/caco2: rerun is non-PDV only, no overlap expected
        # Match on the condition too where both sides carry one. Dropping every
        # existing row for a (model, rep) would discard conditions the re-run did
        # not produce -- and the re-run now runs four of the seven by default.
        key_cols = ['model', 'rep']
        if condition_col in existing.columns and condition_col in rerun.columns:
            key_cols.append(condition_col)
        rerun_keys = rerun[key_cols].drop_duplicates()
        mask = pd.Series(False, index=existing.index)
        for _, row in rerun_keys.iterrows():
            m = pd.Series(True, index=existing.index)
            for c in key_cols:
                m &= (existing[c] == row[c])
            mask |= m
        kept = existing[~mask]
        combined = pd.concat([kept, rerun], ignore_index=True)
        print(f"  Kept {len(kept)} existing + {len(rerun)} new = {len(combined)} total")
    else:
        combined = rerun

    sort_cols = [c for c in ['dataset', 'model', 'rep', 'noise_type', 'strategy', 'sigma', 'fold']
                 if c in combined.columns]
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
