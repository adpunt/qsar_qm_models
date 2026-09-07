#!/usr/bin/env python
"""The Gaussian process's capped training set and its noise record are the same
molecules, in the same order.

    python scripts/test_gp_cap_matches_noise.py

WHAT THIS CATCHES, AND WHAT IT COST.

An exact Gaussian process is cubic in its training count, so `cap_gp_training_set`
subsamples to GP_DEFAULTS['max_train_n'] -- 5,000. On QM9's 8,000 training
molecules that always fires.

It used to return only the COUNT and throw the selection away. The caller then
handed `score_training_molecules_out_of_fold` 5,000 rows while the noise record
still described all 8,000, and the guard refused:

    RuntimeError: out-of-fold scoring for gauche_rbf: the model fits 5000 rows but
    the recorded noise covers 8000.

The guard was right -- pairing them by position attributes one molecule's noise to
another, which is the original QM9 defect. But nothing could reconstruct the
selection afterwards: it is a seeded draw made inside the function. So EVERY
Gaussian process in the study -- gauche, gauche_rbf, heteroscedastic_gp -- failed
its out-of-fold pass, and because the runner exits non-zero on an incomplete
results file, those tasks lost their ACCURACY rows too.

Three things are checked, because fixing only the first would leave the bug:
  1. the selection is returned at all;
  2. the molecules it names are the ones actually fitted, in the same order;
  3. it is deterministic in the seed -- the ordinary and heteroscedastic processes
     must fit the SAME molecules on one run, or the comparison between them differs
     by training set as well as by noise model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'models'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402

from model_defaults import GP_DEFAULTS, cap_gp_training_set  # noqa: E402

FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f'\n          {detail}' if detail and not ok else ''))


def main():
    cap = GP_DEFAULTS['max_train_n']
    check(f"the cap is a real number ({cap})", bool(cap) and cap > 0)

    n = 8000                                   # QM9's training count
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 7))
    y = rng.normal(size=n)

    out = cap_gp_training_set(X, y, seed=12345)
    check('cap_gp_training_set returns the selection, not just a count',
          len(out) == 4, f'returned {len(out)} value(s)')
    if len(out) != 4:
        print('\n  cannot go on without the selection.')
        return 1
    Xc, yc, n_fit, kept = out

    check(f'{n} molecules are capped to {cap}',
          len(Xc) == cap and len(yc) == cap and n_fit == cap,
          f'{len(Xc)} rows, n_fit={n_fit}')
    check('the selection has one index per fitted row',
          len(kept) == len(yc), f'{len(kept)} indices for {len(yc)} rows')
    check('THE SELECTION NAMES THE MOLECULES ACTUALLY FITTED',
          np.array_equal(X[kept], Xc) and np.array_equal(y[kept], yc),
          'the noise record would be lined up against different molecules')
    check('the selection is in the original molecule order',
          np.all(np.diff(kept) > 0), 'out of order, or a repeat')
    check('every index is inside the training set',
          kept.min() >= 0 and kept.max() < n)

    # The noise record is indexed with exactly this, so it must work as a selector.
    record = np.arange(n) * 10
    check('it works as a numpy row selection on the noise arrays',
          len(record[kept]) == cap and np.array_equal(record[kept], kept * 10))

    # Determinism: the ordinary and heteroscedastic processes cap with the same seed
    # on one run and must land on the same molecules.
    _, _, _, again = cap_gp_training_set(X, y, seed=12345)
    check('the same seed keeps the same molecules', np.array_equal(kept, again))
    _, _, _, other = cap_gp_training_set(X, y, seed=999)
    check('a different seed keeps different molecules', not np.array_equal(kept, other))

    # Below the cap it is a no-op, and the selection is still usable.
    Xs, ys, n_s, kept_s = cap_gp_training_set(X[:100], y[:100], seed=1)
    check('below the cap nothing is dropped and the selection is every index',
          n_s == 100 and len(kept_s) == 100 and np.array_equal(kept_s, np.arange(100)))

    # Both callers must pass it. Reading the source is the only way to check this
    # without a GPU and a real fit.
    src = (ROOT / 'models' / 'models.py').read_text()
    check('both GP callers unpack the selection',
          src.count('gp_n_train, gp_kept = cap_gp_training_set') == 2,
          f"found {src.count('gp_n_train, gp_kept = cap_gp_training_set')}")
    check('both GP callers hand it to the out-of-fold scorer as train_slice',
          src.count('train_slice=gp_kept') == 2,
          f"found {src.count('train_slice=gp_kept')}")

    print(f"\n  {len(FAILS)} failure(s)" if FAILS else '\n  all checks passed.')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
