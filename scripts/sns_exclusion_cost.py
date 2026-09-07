#!/usr/bin/env python
"""How big the Sort & Slice exclusion is against the replicate-to-replicate wobble.

    python scripts/sns_exclusion_cost.py --results results

THE QUESTION THIS ANSWERS. `62f1fe2` drops methane, ammonia and water from the training,
validation and test lists of EVERY representation, because Sort & Slice cannot give them
a vector. A sample of 10,000 out of 132,480 draws at least one of the three about one time
in five, so about one replicate in five now trains on 9,999 or 9,998 molecules where it
used to train on 10,000. A QM9 row written before that commit is on the larger set and a
row written after it is on the smaller one.

Measured on three pairs of runs (`RERUN_PLAN.md` 13.27 D4g), that moved the clean R2 by
between 0.001262 and 0.023893. The author's decision is whether to re-run what ran before
the fix or to state the difference in Methods, and it turns on ONE number that nobody has
computed: how that shift compares with how much the clean R2 already moves between
replicates of the same run.

WHAT IT PRINTS. One row per combination of a model, a representation and a noise type,
giving the clean R2 range across that combination's own replicates. Then how many
combinations have a replicate range SMALLER than the observed shift, which is where the
shift would not be lost in the wobble that is already there.

IT PRINTS NUMBERS AND NOTHING ELSE. The choice is the author's; nothing here recommends
one, and nothing here is written into a document.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_load as FL  # noqa: E402

# The two ends of the shift measured on real runs, RERUN_PLAN.md 13.27 D4g. They are
# quoted, not recomputed: the three pairs they come from are named there.
SHIFT_SMALLEST = 0.001262
SHIFT_LARGEST = 0.023893


def clean_rows(df):
    """The no-noise rows. `sigma` is the noise level and 0 is the clean label."""
    return df[np.isclose(df['sigma'].astype(float), 0.0)]


def spread_per_combination(df):
    """One row per model, representation and noise type: how far apart its replicates are.

    `n_replicates` matters as much as the range. Two replicates give a range that says
    very little, so the summary below counts only combinations with at least three.
    """
    key = ['model', 'rep', 'condition']
    grouped = df.groupby(key, dropna=False)['r2']
    out = pd.DataFrame({
        'n_replicates': grouped.size(),
        'r2_min': grouped.min(),
        'r2_max': grouped.max(),
        'r2_range': grouped.max() - grouped.min(),
        'r2_sd': grouped.std(ddof=1),
    }).reset_index()
    return out.sort_values('r2_range', ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results',
                    help='the directory holding anova_*.csv (default: results)')
    ap.add_argument('--min-replicates', type=int, default=3,
                    help='anything with fewer replicates than this is listed but left '
                         'out of the summary (default 3)')
    ap.add_argument('--show', type=int, default=15,
                    help='how many rows to print in full (default 15)')
    cli = ap.parse_args()

    df = FL.load_qm9(cli.results)
    if df is None or not len(df):
        print(f'  no QM9 rows under {cli.results}. Nothing to compare.')
        return 1
    if 'condition' not in df.columns:
        print('  the rows carry no noise type, so nothing can be grouped.')
        return 1

    clean = clean_rows(df)
    if not len(clean):
        print('  no rows at noise level 0, so there is no clean R2 to compare.')
        return 1

    table = spread_per_combination(clean)
    usable = table[table['n_replicates'] >= cli.min_replicates]

    print(f'\n  {len(clean)} clean row(s) over {len(table)} combination(s) of a model, '
          f'a representation\n  and a noise type. Every number below is R2 on QM9.\n')
    print(f'  {"model":<26s} {"rep":<11s} {"noise type":<16s} {"reps":>4s} '
          f'{"lowest R2":>10s} {"highest R2":>11s} {"range":>9s} {"SD":>9s}')
    for _, r in table.head(cli.show).iterrows():
        sd = f"{r['r2_sd']:.6f}" if pd.notna(r['r2_sd']) else '        -'
        print(f"  {str(r['model'])[:26]:<26s} {str(r['rep'])[:11]:<11s} "
              f"{str(r['condition'])[:16]:<16s} {int(r['n_replicates']):>4d} "
              f"{r['r2_min']:>10.6f} {r['r2_max']:>11.6f} {r['r2_range']:>9.6f} "
              f"{sd:>9s}")
    if len(table) > cli.show:
        print(f'  ... and {len(table) - cli.show} more')

    if not len(usable):
        print(f'\n  Nothing has {cli.min_replicates} replicates or more, so there is '
              f'no replicate range to compare against yet.')
        return 1

    q = usable['r2_range'].quantile([0.25, 0.5, 0.75])
    print(f'\n  Across the {len(usable)} combination(s) with at least '
          f'{cli.min_replicates} replicates, the clean\n  R2 range between replicates '
          f'of the SAME combination is:')
    print(f'      lowest        {usable["r2_range"].min():.6f} R2')
    print(f'      lower quarter {q.loc[0.25]:.6f} R2')
    print(f'      middle        {q.loc[0.5]:.6f} R2')
    print(f'      upper quarter {q.loc[0.75]:.6f} R2')
    print(f'      highest       {usable["r2_range"].max():.6f} R2')

    # THE COMPARISON THE DECISION TURNS ON. A shift smaller than a combination's own
    # replicate range is inside the wobble that combination already carries. A shift that
    # is larger is not, and mixing a pre-fix row with a post-fix row there moves the
    # number by more than repeating the run does.
    print(f'\n  The exclusion moved the clean R2 by {SHIFT_SMALLEST:.6f} to '
          f'{SHIFT_LARGEST:.6f} R2 on the three\n  pairs measured (RERUN_PLAN.md 13.27 '
          f'D4g). Against the replicate range:')
    for shift, name in ((SHIFT_SMALLEST, 'the smallest'), (SHIFT_LARGEST, 'the largest')):
        bigger = int((usable['r2_range'] < shift).sum())
        print(f'      {name} shift, {shift:.6f} R2, is LARGER than the replicate '
              f'range in\n        {bigger} of {len(usable)} combination(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
