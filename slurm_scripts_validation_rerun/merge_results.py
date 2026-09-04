#!/usr/bin/env python3
"""Merge the laboratory re-run's results into the archive the figures are built from.

Run it from ANYWHERE. It used to say "run from /data/stat-cadd/scat9264/KIRBy/tests",
which was wrong twice over: KIRBy moved to /data/stat-ecr on 2026-05-07, and there is no
working directory that reaches both halves of what this reads. The generated jobs
`cd "$KIRBY_DIR"/tests` and pass `--results-root "../results/validation_rerun/..."`, so
the new grid lands in <KIRBy>/results/validation_rerun while every earlier run and the
archive sit under <KIRBy>/tests/results. Both are searched, and what was found is
printed before anything is written.

Reads:
  - the re-run, from <KIRBy>/results/validation_rerun/<model>_<rep>_<dataset>/<dataset>/
    and <KIRBy>/tests/results/validation_rerun/... -- whichever exist
  - the archive, from <KIRBy>/tests/results/validation/<dataset>/all_results.csv

Writes:
  - that same archive file, backed up first with a timestamp

Archive rows whose noise condition is not one of the settled seven are DROPPED, and the
count is printed. They come from the scheme this whole re-run exists to replace; nothing
downstream reads a condition name and asks whether it is still a condition.
`--allow-retired-conditions` keeps them.

Usage:
    python merge_results.py --dry-run   # says what it found and what it would write
    python merge_results.py             # do it
"""
import argparse
import glob
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# RESOLVED AT RUN TIME, NOT AT IMPORT. These used to be bare relative paths, so the
# working directory decided everything and there was no working directory that was
# right: the generated jobs `cd "$KIRBY_DIR"/tests` and pass `--results-root
# "../results/validation_rerun/..."`, which lands the new grid in
# <KIRBy>/results/validation_rerun, while every earlier run and the archive this merges
# into sit under <KIRBy>/tests/results. Run from tests/ the new grid was invisible; run
# from the checkout root the archive was. Both are searched now, and what was found is
# printed before anything is written.
KIRBY_DIR = Path('/data/stat-ecr/scat9264/KIRBy')
RERUN_SUBPATHS = ('results/validation_rerun', 'tests/results/validation_rerun')
VALIDATION_SUBPATHS = ('tests/results/validation', 'results/validation')
VALIDATION_DIR = Path('results/validation')
RERUN_DIRS = [Path('results/validation_rerun')]

# The conditions this study runs. Anything else in the archive was produced under the
# noise scheme that was replaced (NOISE_DESIGN.md), and merging it forward would put
# pre-redesign rows into the file the figures are built from.
_SETTLED_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
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

    # Load all rerun data for this dataset, from every place the grid may have written
    rerun_dfs = []
    for root in RERUN_DIRS:
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.endswith(f'_{dataset}'):
                continue
            for csv in sorted((d / dataset).glob('all_results*.csv')):
                df = pd.read_csv(csv)
                if len(df) > 0:
                    rerun_dfs.append(df)
                    print(f"  Rerun {root}/{d.name}: {len(df)} rows, "
                          f"models={sorted(df['model'].unique())}, "
                          f"reps={sorted(df['rep'].unique())}")

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

    # DROP PRE-REDESIGN ROWS FROM THE ARCHIVE BEFORE THEY ARE CARRIED FORWARD.
    #
    # results/validation is the archive from before the noise scheme was replaced
    # (NOISE_DESIGN.md). Its conditions are not the settled seven, and its levels do not
    # mean what the same numbers mean now. Carrying them forward puts rows from the
    # scheme this whole re-run exists to replace into the file the figures are built
    # from -- silently, because nothing downstream reads the condition name and asks
    # whether it is still a condition.
    if not existing.empty and not ALLOW_RETIRED:
        col = condition_column_of(existing)
        keep = existing[col].isin(SETTLED_CONDITIONS)
        if (~keep).any():
            dropped = sorted(existing.loc[~keep, col].unique())
            print(f"  Dropped {int((~keep).sum())} archive row(s) under retired "
                  f"conditions: {', '.join(map(str, dropped))}")
            print(f"    (--allow-retired-conditions keeps them; the settled set is "
                  f"{', '.join(SETTLED_CONDITIONS)})")
            existing = existing[keep]

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
    global VALIDATION_DIR, RERUN_DIRS, SETTLED_CONDITIONS, ALLOW_RETIRED
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--kirby-dir', default=str(KIRBY_DIR),
                        help=f'KIRBy checkout the jobs wrote into (default '
                             f'{KIRBY_DIR}). Both {" and ".join(RERUN_SUBPATHS)} are '
                             f'searched under it, because the generated jobs and this '
                             f'script disagreed about which one for a while.')
    parser.add_argument('--rerun-dir', default=None, action='append',
                        help='Look here for re-run output instead of searching under '
                             '--kirby-dir. Repeatable.')
    parser.add_argument('--validation-dir', default=None,
                        help='The archive that is merged into and written back. '
                             'Default: whichever of '
                             f'{", ".join(VALIDATION_SUBPATHS)} exists.')
    parser.add_argument('--allow-retired-conditions', action='store_true',
                        help='Carry archive rows forward even when their condition is '
                             'not one of the settled seven. Off by default: those rows '
                             'come from the noise scheme this re-run replaced.')
    args = parser.parse_args()
    ALLOW_RETIRED = args.allow_retired_conditions

    settled = json.loads(_SETTLED_FILE.read_text())
    SETTLED_CONDITIONS = [c['name'] for g in ('stage_1_full_grid', 'stage_2_depth_only')
                          for c in settled.get(g, [])]

    kirby = Path(args.kirby_dir)
    if args.rerun_dir:
        RERUN_DIRS = [Path(p) for p in args.rerun_dir]
    else:
        RERUN_DIRS = [kirby / s for s in RERUN_SUBPATHS]
    missing = [d for d in RERUN_DIRS if not d.is_dir()]
    RERUN_DIRS = [d for d in RERUN_DIRS if d.is_dir()]
    for d in missing:
        print(f"  not present, skipped: {d}")
    if not RERUN_DIRS:
        print("ERROR: found no re-run output. Looked in:")
        for s in RERUN_SUBPATHS:
            print(f"  {kirby / s}")
        print("Pass --rerun-dir if the jobs wrote somewhere else. The generated jobs "
              "cd into <KIRBy>/tests and pass a RELATIVE --results-root, so the answer "
              "depends on how many '../' that path carries.")
        sys.exit(1)
    print(f"reading re-run output from: {', '.join(str(d) for d in RERUN_DIRS)}")

    if args.validation_dir:
        VALIDATION_DIR = Path(args.validation_dir)
    else:
        VALIDATION_DIR = next((kirby / s for s in VALIDATION_SUBPATHS
                               if (kirby / s).is_dir()), kirby / VALIDATION_SUBPATHS[0])
    print(f"merging into and writing back to:  {VALIDATION_DIR}")
    print(f"settled conditions: {', '.join(SETTLED_CONDITIONS)}"
          + ("   (retired rows KEPT by request)" if ALLOW_RETIRED else ""))

    for dataset in DATASETS:
        merge_dataset(dataset, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == '__main__':
    main()
