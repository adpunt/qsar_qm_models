#!/usr/bin/env python3
"""Should sparse substructure COUNTS be standardised before a kernel model?

THE QUESTION, AND WHY IT IS OPEN
--------------------------------
`models/model_defaults.py` decides feature scaling by looking at the matrix: a
0/1 matrix is left alone, anything else is standardised per feature. That rule
was measured, on Morgan bits with a radial-kernel support vector machine:

    raw 0/1 bits    R2 = +0.815, +0.809, +0.832
    standardised    R2 = +0.320, +0.124, +0.182

Standardising a sparse binary fingerprint gives its rare bits enormous
magnitudes and lets them dominate every distance.

The Sort & Slice substructure fingerprint is COMPUTED as counts -- the featuriser
is called with sub_counts=True (process_and_train.py:681) -- and then thrown away
at storage, because write_to_mmap packs it with np.packbits (:539-541), which is
a presence bit per substructure. So the matrix the models see today is binary,
the rule leaves it unscaled, and nobody has had to decide anything.

The moment that storage defect is fixed the matrix stops being binary, the rule
starts standardising it, and the setting changes underneath every kernel model
with nothing measured. `should_standardise` says so in its own docstring:
"NOT YET MEASURED: substructure COUNTS ... sparse counts may have the same
problem as sparse bits."

This measures it.

THE DECISION RULE, FIXED BEFORE THE RUN
---------------------------------------
Standardise sparse counts if and only if, for the radial-kernel support vector
machine:

  (a) standardised counts beat unscaled counts on BOTH datasets, and
  (b) on at least 4 of the 5 seeds in each, and
  (c) the mean paired gain exceeds the seed-to-seed standard deviation of that
      same paired difference on both.

Otherwise leave counts unscaled -- extend the binary exemption to sparse counts.

Ties go to leaving them unscaled, because that is what the representation gets
today and a change has to earn itself.

A SECOND NUMBER THIS PRODUCES FOR FREE
--------------------------------------
Counts against bits, both unscaled, under the setting the rule selects. That is
what the storage fix actually buys, and nobody has measured that either.

WHAT IS HELD FIXED
------------------
Scaffold split, training-only fitting of the Sort & Slice pooling operator, the
support vector machine on the shared spec's own settings, labels standardised on
the training split. The random forest is carried as a control: trees are
invariant to a monotone per-feature transform, so it should show nothing, and if
it does the harness is wrong rather than the answer interesting.

    python scripts/parity_test_count_scaling.py
    python scripts/parity_test_count_scaling.py --seeds 3 --qm9-n 2000   # quicker
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from models.model_defaults import SKLEARN_DEFAULTS, SPEC_VERSION, spec_hash   # noqa: E402
import parity_test_common as common                                          # noqa: E402

OUT_DIR = REPO / 'results' / 'parity_tests'
OUT_CSV = OUT_DIR / 'count_scaling.csv'

# What the models are fed. 'bits' is today's storage; 'counts' is what the
# featuriser actually computes.
FEATURE_SETS = ['bits_raw', 'bits_std', 'counts_raw', 'counts_std']


def build_models():
    """The two estimators, from the shared spec -- not retyped here.

    The support vector machine is the one that decided the binary case and is
    the one at risk: a radial kernel is a function of distances, and a rare
    feature standardised to a huge value moves every distance. The forest is a
    control.
    """
    svm = dict(SKLEARN_DEFAULTS['svm'])
    rf = dict(SKLEARN_DEFAULTS['rf'])
    return {
        'svm_rbf': lambda seed: SVR(**svm),
        'rf': lambda seed: RandomForestRegressor(random_state=seed, n_jobs=-1, **rf),
    }


def count_profile(X):
    """What 'counts' actually means on this dataset, so the answer can be read."""
    nz = X[X > 0]
    return {
        'max_count': float(X.max()),
        'mean_nonzero_count': float(nz.mean()) if nz.size else 0.0,
        'frac_nonzero_above_one': float((nz > 1).mean()) if nz.size else 0.0,
        'density': float((X > 0).mean()),
    }


def run_dataset(name, smiles, y, scaffolds, seeds, models, rows, subsample=None):
    for seed in seeds:
        t0 = time.time()
        idx = np.arange(len(smiles))
        if subsample is not None and subsample < len(smiles):
            idx = np.random.RandomState(seed).choice(len(smiles), subsample, replace=False)
        sub_smiles = [smiles[i] for i in idx]
        sub_y = y[idx]
        sub_scaf = scaffolds[idx]

        train, _, test = common.scaffold_split(sub_scaf, seed)
        counts = common.sort_and_slice(sub_smiles, train)
        bits = (counts > 0).astype(np.float32)

        profile = count_profile(counts[train])

        # Labels standardised on the TRAINING split, clean -- the same convention
        # both pipelines now use.
        mu, sd = sub_y[train].mean(), sub_y[train].std()
        sd = sd if sd > 0 else 1.0
        y_train = (sub_y[train] - mu) / sd
        y_test = (sub_y[test] - mu) / sd

        matrices = {}
        for base, X in (('bits', bits), ('counts', counts)):
            matrices[f'{base}_raw'] = (X[train], X[test])
            std_train, std_test = common.standardise(X[train], X[test])
            matrices[f'{base}_std'] = (std_train, std_test)

        for feature_set in FEATURE_SETS:
            X_train, X_test = matrices[feature_set]
            for model_name, factory in models.items():
                t1 = time.time()
                model = factory(seed)
                model.fit(X_train, y_train)
                r2 = float(r2_score(y_test, model.predict(X_test)))
                rows.append(dict(
                    dataset=name, seed=seed, feature_set=feature_set,
                    model=model_name, r2=r2,
                    n_train=len(train), n_test=len(test),
                    n_features=X_train.shape[1],
                    seconds=round(time.time() - t1, 1),
                    spec_version=SPEC_VERSION, spec_hash=spec_hash(),
                    **profile))
                print(f'  {name} seed {seed} {feature_set:11s} {model_name:8s} '
                      f'R2 = {r2:+.4f}  ({time.time() - t1:.0f}s)', flush=True)
        print(f'  {name} seed {seed} done in {time.time() - t0:.0f}s '
              f'(train {len(train)}, test {len(test)}, '
              f'max count {profile["max_count"]:.0f}, '
              f'{profile["frac_nonzero_above_one"]:.1%} of present substructures '
              f'appear more than once)', flush=True)


def paired(rows, dataset, model, a, b):
    """Per-seed a-minus-b, and the summary the decision rule reads."""
    by_seed = {}
    for r in rows:
        if r['dataset'] == dataset and r['model'] == model:
            by_seed.setdefault(r['seed'], {})[r['feature_set']] = r['r2']
    seeds = sorted(s for s in by_seed if a in by_seed[s] and b in by_seed[s])
    diffs = np.array([by_seed[s][a] - by_seed[s][b] for s in seeds])
    return seeds, diffs


def report(rows):
    datasets = sorted({r['dataset'] for r in rows})
    print('\n' + '=' * 78)
    print('THE DECISION: standardise sparse substructure counts, or leave them raw?')
    print('=' * 78)

    verdict_parts = []
    for dataset in datasets:
        seeds, diffs = paired(rows, dataset, 'svm_rbf', 'counts_std', 'counts_raw')
        wins = int((diffs > 0).sum())
        mean, sd = diffs.mean(), diffs.std(ddof=1) if len(diffs) > 1 else 0.0
        print(f'\n{dataset}: standardised counts minus raw counts, radial-kernel SVM')
        print(f'  per seed: {", ".join(f"{d:+.4f}" for d in diffs)}')
        print(f'  mean {mean:+.4f}, seed-to-seed spread {sd:.4f}, '
              f'standardising wins on {wins}/{len(diffs)} seeds')
        verdict_parts.append(dict(dataset=dataset, mean=mean, sd=sd,
                                  wins=wins, n=len(diffs)))

    passes = all(v['mean'] > 0 and v['wins'] >= max(1, v['n'] - 1) and v['mean'] > v['sd']
                 for v in verdict_parts) and len(verdict_parts) == len(datasets)
    decision = 'STANDARDISE' if passes else 'LEAVE RAW'
    print(f'\n  --> decision rule says: {decision} sparse counts')
    if not passes:
        print('      (the rule needed a gain on both datasets, on at least all-but-one seed '
              'each,\n       and larger than the seed-to-seed spread. Ties go to leaving '
              'them raw.)')

    chosen = 'counts_std' if passes else 'counts_raw'
    print(f'\n{"=" * 78}')
    print(f'AND WHAT THE STORAGE FIX BUYS: {chosen} minus bits_raw (today\'s storage)')
    print('=' * 78)
    for dataset in datasets:
        for model in ('svm_rbf', 'rf'):
            seeds, diffs = paired(rows, dataset, model, chosen, 'bits_raw')
            if not len(diffs):
                continue
            print(f'  {dataset:6s} {model:8s} mean {diffs.mean():+.4f}  '
                  f'(per seed: {", ".join(f"{d:+.4f}" for d in diffs)})')

    print(f'\n{"=" * 78}')
    print('CONTROL: the forest should be indifferent -- it splits on thresholds, and')
    print('standardising is monotone per feature, so any movement is sampling noise.')
    print('=' * 78)
    for dataset in datasets:
        seeds, diffs = paired(rows, dataset, 'rf', 'counts_std', 'counts_raw')
        print(f'  {dataset:6s} mean {diffs.mean():+.4f}, largest {np.abs(diffs).max():.4f}')

    return decision


def self_test():
    """Fails if the rule this measurement set is taken back out.

    Four matrices, one per branch. A run of the full measurement takes twelve
    minutes and needs the network; this takes no time and is what a preflight can
    afford, so it is the guard that actually gets run.
    """
    from models.model_defaults import (is_binary_matrix, is_sparse_count_matrix,
                                       should_standardise)
    rng = np.random.RandomState(0)
    n, d = 600, 1024

    binary = (rng.random_sample((n, d)) < 0.03).astype(np.float32)

    counts = binary.copy()
    hit = counts > 0
    counts[hit] = rng.randint(1, 8, size=int(hit.sum()))

    continuous = rng.normal(size=(n, 208)).astype(np.float32)

    # Non-negative integers, but dense -- a count-like descriptor block, which is
    # NOT the sparse-fingerprint case and must still be standardised.
    dense_counts = rng.randint(0, 12, size=(n, 30)).astype(np.float32)

    cases = [
        ('binary fingerprint', binary, False, dict(binary=True, sparse=False)),
        ('sparse substructure counts', counts, False, dict(binary=False, sparse=True)),
        ('continuous descriptors', continuous, True, dict(binary=False, sparse=False)),
        ('dense integer block', dense_counts, True, dict(binary=False, sparse=False)),
    ]
    failures = []
    for name, X, want_std, want_kind in cases:
        got_std = should_standardise(X)
        got_kind = dict(binary=is_binary_matrix(X), sparse=is_sparse_count_matrix(X))
        mark = 'ok ' if (got_std == want_std and got_kind == want_kind) else 'FAIL'
        if mark == 'FAIL':
            failures.append(f'{name}: standardise={got_std} (want {want_std}), '
                            f'{got_kind} (want {want_kind})')
        print(f'  {mark} {name:28s} density {float((X != 0).mean()):.3f}  '
              f'standardise={got_std}  {got_kind}')
    if failures:
        print('\nFAIL — the scaling rule has moved:')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('\nPASS — sparse counts are exempt, dense integers are not, '
          'binary is still exempt.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true',
                    help='Check the rule this measurement set, in a second, without data. '
                         'This is the guard that fails if the fix is removed.')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--qm9-n', type=int, default=4000,
                    help='QM9 molecules per seed, drawn fresh (default 4000, the size '
                         'chat G used for the setting-selection run).')
    ap.add_argument('--qm9-pool', type=int, default=20000,
                    help='How many QM9 molecules to load before subsampling.')
    ap.add_argument('--skip-herg', action='store_true',
                    help='QM9 only. The decision rule needs both, so this cannot settle it.')
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    models = build_models()
    rows = []

    print(f'spec {SPEC_VERSION} ({spec_hash()[:12]}), {args.seeds} seeds')
    print('QM9...', flush=True)
    smiles, y, scaffolds = common.load_qm9(args.qm9_pool)
    run_dataset('qm9', smiles, y, scaffolds, seeds, models, rows, subsample=args.qm9_n)

    if not args.skip_herg:
        print('hERG Ki...', flush=True)
        h_smiles, h_y, h_scaffolds = common.load_herg()
        print(f'  {len(h_smiles)} compounds, label spread {h_y.std():.3f} log units')
        run_dataset('herg', h_smiles, h_y, h_scaffolds, seeds, models, rows)

    with open(OUT_CSV, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nwrote {OUT_CSV} ({len(rows)} rows)')

    report(rows)
    print(f'\nQuote the numbers from {OUT_CSV}, not from this log.')
    # This measures; it does not gate. The exit status says the run completed,
    # not which way the answer went -- the answer is in the CSV.
    return 0


if __name__ == '__main__':
    sys.exit(main())
