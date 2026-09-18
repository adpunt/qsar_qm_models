#!/usr/bin/env python3
"""The group-share statistic has to actually compute, and it never has.

WHAT WAS WRONG
--------------
`q7_group_correlated_error` measures how much of a model's out-of-fold error a
whole scaffold family holds in common:

    group_share = Var(group mean error) / Var(error)

It tested for a column called `canonical_smiles`. The writer does emit that
column -- it is field 19 of the per-molecule schema -- but `load_uncertainty`
builds a fresh frame and puts it on under the name `mol_id`:

    out['mol_id'] = df['canonical_smiles'] if 'canonical_smiles' in df.columns

So the test was False for every row that has ever been loaded, and the statistic
returned NaN on all 20,699 rows of the 2026-09-17 harvest with the reason "no
canonical_smiles on the row" written beside it. That reads as a gap in the runs.
It is a rename inside this file, and nothing is missing from the data.

WHY IT MATTERS BEYOND Q7
------------------------
This is the one statistic in the study that measures the group structure of a
model's error WITHOUT knowing the injected noise, so it is the only one that can
be pointed at a real dataset whose true labels nobody has. The artificial
conditions calibrate it -- grouped-shifted injects a group share of 0.781 and
every other condition sits at 0.11 to 0.14 -- and the clean level of an assay
dataset then reads against that scale.

WHAT IT CHECKS
--------------
1. The statistic computes from a frame carrying `mol_id`, which is what the
   loader produces, and from one carrying `canonical_smiles`, which is what the
   writer produces.
2. It separates group-correlated error from independent error: a model whose
   error is a per-scaffold offset scores near 1, and one whose error is drawn
   per molecule scores near the finite-group-size floor.
3. `n_groups` and `mean_group_size` travel with every row, because a share over
   groups of one is exactly 1 and means nothing.
4. A frame with no identifier says so, and says which column it wanted.

    python scripts/test_q7_group_share_computes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as unc  # noqa: E402

#: Nine scaffolds, ten molecules each. Real SMILES, so rdkit's Murcko
#: decomposition has something to do; each base is a distinct ring system and the
#: substituent varies within a family, so the scaffold groups are the families.
#: Every one of these stays a valid molecule when the substituent is appended --
#: checked with rdkit, because three of the twelve first tried here (quinoxaline,
#: oxazole and thiazole) do not, and rdkit then drops those rows and the group
#: count silently comes out lower than the fixture intended.
SCAFFOLDS = [
    'c1ccccc1', 'c1ccncc1', 'c1ccc2ccccc2c1', 'C1CCCCC1',
    'C1CCNCC1', 'c1ccsc1', 'c1cc[nH]c1', 'c1ccc2[nH]ccc2c1',
    'C1CCOC1',
]
PER_GROUP = 10


def molecules():
    """(smiles, scaffold index) for 90 molecules in 9 families."""
    out = []
    for index, core in enumerate(SCAFFOLDS):
        for n in range(PER_GROUP):
            out.append((core + 'C' * (n + 1), index))
    return out


def frame(group_correlated, identifier='mol_id', seed=0):
    """One out-of-fold cell whose error is per-group or per-molecule."""
    rng = np.random.default_rng(seed)
    mols = molecules()
    smiles = [m for m, _ in mols]
    groups = np.array([g for _, g in mols])
    if group_correlated:
        offsets = rng.normal(0, 1.0, len(SCAFFOLDS))
        error = offsets[groups] + rng.normal(0, 0.05, len(mols))
    else:
        error = rng.normal(0, 1.0, len(mols))
    y = rng.normal(0, 1, len(mols))
    return pd.DataFrame({
        'dataset': 'qm9', 'model': 'rf', 'rep': 'ecfp4',
        'condition': 'gaussian', 'sigma': 1.0, 'fold': 0, 'split': 'train_oof',
        'y_true_clean': y, 'y_pred': y + error,
        identifier: smiles,
    })


def run(df):
    got = unc.q7_group_correlated_error(df, min_n=5)
    assert len(got) == 1, f'expected one cell, got {len(got)}'
    return got.iloc[0]


def check_both_column_names():
    for identifier in ('mol_id', 'canonical_smiles'):
        row = run(frame(True, identifier=identifier, seed=1))
        assert np.isfinite(row['group_share']), (
            f'{identifier}: the statistic did not compute -- {row["reason"]!r}')
        print(f'  {identifier}: computes, share {row["group_share"]:.3f}, '
              f'{int(row["n_groups"])} groups of {row["mean_group_size"]:.1f}')


def check_it_separates():
    correlated = run(frame(True, seed=2))
    independent = run(frame(False, seed=2))
    print(f'  per-scaffold offset: share {correlated["group_share"]:.3f}')
    print(f'  per-molecule draw:   share {independent["group_share"]:.3f}')
    assert correlated['group_share'] > 0.8, (
        'error that IS a per-scaffold offset should score near 1, got '
        f'{correlated["group_share"]:.3f}')
    assert independent['group_share'] < 0.4, (
        'error drawn per molecule should sit near the finite-group-size floor, '
        f'got {independent["group_share"]:.3f}')
    assert correlated['group_share'] > independent['group_share'] + 0.4, (
        'the two cases have to be far apart for the statistic to be readable')


def check_group_size_travels():
    row = run(frame(True, seed=3))
    assert int(row['n_groups']) == len(SCAFFOLDS), (
        f'expected {len(SCAFFOLDS)} scaffold groups, got {int(row["n_groups"])}')
    assert abs(row['mean_group_size'] - PER_GROUP) < 1e-6, (
        f'expected groups of {PER_GROUP}, got {row["mean_group_size"]}')
    print(f'  n_groups and mean_group_size travel: {int(row["n_groups"])} '
          f'groups of {row["mean_group_size"]:.1f} -- a share over groups of '
          f'one is exactly 1 and means nothing, so both are needed to read it')


def check_missing_identifier_says_so():
    df = frame(True, seed=4).drop(columns=['mol_id'])
    row = run(df)
    assert not np.isfinite(row['group_share']), (
        'a frame with no identifier produced a number')
    assert 'identifier' in str(row['reason']), (
        f'the reason should name what was missing, got {row["reason"]!r}')
    print(f'  no identifier: {row["reason"]!r}')


def main():
    print(__doc__.split('\n')[0])
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        print('  rdkit is not importable here, so the statistic cannot be '
              'exercised. NOT a pass.')
        return 0
    check_both_column_names()
    check_it_separates()
    check_group_size_travels()
    check_missing_identifier_says_so()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
