#!/usr/bin/env python3
"""Which max_features setting should the forests use, on BOTH pipelines?

THE QUESTION
------------
QM9 pins max_features='sqrt' for the random forest and the quantile forest
(models/models.py:1367). The experimental pipeline passes nothing, so it takes
scikit-learn's default of 1.0 -- every feature at every split. That is not a
tuning difference, it is a different forest under one name, and it has to be
resolved in one direction before either study is re-run.

Three candidates:
    'sqrt'  what QM9 does today. ~14 of 208 descriptors, ~45 of 2048 bits.
    0.3     30% of features.
    1.0     what the experimental side takes by omission.

THE DECISION RULE, fixed before the run
---------------------------------------
Take the setting that (a) wins or ties on the descriptor vector UNDER NOISE --
that is the paper's primary representation and the regime the paper is about --
and (b) does not lose on Morgan. If two settings tie inside the seed spread,
take the faster one.

Read the PAIRED differences, not the raw R2. Each seed reshuffles the scaffold
split, so the unpaired spread is dominated by which scaffolds land in test.

WHY THE QUANTILE FOREST IS EMULATED
-----------------------------------
quantile_forest 1.4.1 cannot be fitted against scikit-learn 1.3.2 -- every fit
raises "Invalid parameter 'monotonic_cst'". That is a live launch blocker, not a
quirk of this test (the uncertainty preflight already checks for it). So the
quantile forest here is computed from a plain RandomForestRegressor by pooling
the in-bag training labels that share a leaf with the query point, which is
exactly what quantile_forest computes. Rows are labelled qrf_emulated=True.
Run --verify-emulation in an environment where the library fits to confirm.

USAGE
-----
    python scripts/parity_test_forest.py                 # the real run
    python scripts/parity_test_forest.py --quick         # 2,000 molecules, 2 seeds
    python scripts/parity_test_forest.py --verify-emulation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_test_common import (  # noqa: E402
    HAS_QRF, QRF_UNAVAILABLE_REASON, build_representations, env_fingerprint,
    forest_quantiles, inject_gaussian, load_qm9, scaffold_split, standardise,
)

OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'parity_tests'
SETTINGS = {'sqrt': 'sqrt', '0.3': 0.3, '1.0': 1.0}
BASELINE = 'sqrt'          # what QM9 does today; everything is paired against it
RF_TREES, QRF_TREES = 100, 300


def run(n_molecules, seeds, sigmas, reps, incremental_path=None):
    smiles, y, scaffolds = load_qm9(n_molecules)
    print(f'molecules {len(smiles)}   label SD {y.std():.4f} eV   '
          f'scaffolds {len(np.unique(scaffolds))}', flush=True)
    X_all = build_representations(smiles, which=reps)
    for name, X in X_all.items():
        print(f'  {name:12s} {X.shape}', flush=True)

    rows = []
    for seed in seeds:
        train, _, test = scaffold_split(scaffolds, seed, train_frac=0.8, val_frac=0.0)
        print(f'\nseed {seed}: train {len(train)}  test {len(test)}', flush=True)
        for rep in reps:
            X = X_all[rep]
            X_tr, X_te = X[train], X[test]
            if rep == 'descriptors':
                X_tr, X_te = standardise(X_tr, X_te)
            for sigma in sigmas:
                y_tr, _ = inject_gaussian(y[train], sigma, seed)
                y_te = y[test]                      # held-out labels stay CLEAN
                for label, max_features in SETTINGS.items():
                    # --- ordinary forest ---
                    t0 = time.time()
                    rf = RandomForestRegressor(
                        n_estimators=RF_TREES, max_depth=None, min_samples_leaf=1,
                        min_samples_split=2, max_features=max_features,
                        bootstrap=True, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
                    pred = rf.predict(X_te)
                    rows.append(dict(
                        family='RF', seed=seed, rep=rep, sigma=sigma, setting=label,
                        r2=r2_score(y_te, pred),
                        rmse=float(np.sqrt(np.mean((y_te - pred) ** 2))),
                        fit_secs=time.time() - t0, interval_width=np.nan,
                        coverage_1sd=np.nan, qrf_emulated=False))
                    print(f'  {rep:11s} sigma={sigma:.1f} RF  max_features={label:5s} '
                          f'R2={rows[-1]["r2"]:.4f}  {rows[-1]["fit_secs"]:6.1f}s',
                          flush=True)

                    # --- quantile forest ---
                    t0 = time.time()
                    qf = RandomForestRegressor(
                        n_estimators=QRF_TREES, max_depth=None, min_samples_leaf=1,
                        min_samples_split=2, max_features=max_features,
                        bootstrap=True, random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
                    q16, q50, q84 = forest_quantiles(
                        qf, X_tr, y_tr, X_te, quantiles=(0.16, 0.5, 0.84)).T
                    inside = (y_te >= q16) & (y_te <= q84)
                    rows.append(dict(
                        family='QRF', seed=seed, rep=rep, sigma=sigma, setting=label,
                        r2=r2_score(y_te, q50),
                        rmse=float(np.sqrt(np.mean((y_te - q50) ** 2))),
                        fit_secs=time.time() - t0,
                        interval_width=float(np.mean(q84 - q16)),
                        coverage_1sd=float(inside.mean()), qrf_emulated=True))
                    print(f'  {rep:11s} sigma={sigma:.1f} QRF max_features={label:5s} '
                          f'R2={rows[-1]["r2"]:.4f}  width={rows[-1]["interval_width"]:.3f} '
                          f'cov={rows[-1]["coverage_1sd"]:.3f}  {rows[-1]["fit_secs"]:6.1f}s',
                          flush=True)
                    # Written after every fit. A full run is hours on a loaded
                    # machine; a partial file that can be read beats a complete
                    # one that never lands.
                    if incremental_path:
                        pd.DataFrame(rows).to_csv(incremental_path, index=False)
    return pd.DataFrame(rows)


def report(df):
    print('\n' + '=' * 78)
    print(f"PAIRED DIFFERENCES vs max_features='{BASELINE}' (QM9 today)")
    print('=' * 78)
    lines = []
    for (family, rep, sigma), g in df.groupby(['family', 'rep', 'sigma']):
        piv = g.pivot_table(index='seed', columns='setting', values='r2')
        if BASELINE not in piv:
            continue
        base = piv[BASELINE]
        for setting in piv.columns:
            if setting == BASELINE:
                continue
            # dropna: on a partial run a seed may have the baseline but not yet
            # the comparison, and a NaN difference would silently count as a loss.
            d = (piv[setting] - base).dropna()
            if not len(d):
                continue
            lines.append(dict(family=family, rep=rep, sigma=sigma, setting=setting,
                              mean_delta=d.mean(), wins=int((d > 0).sum()),
                              n=len(d), lo=d.min(), hi=d.max()))
            print(f'{family:3s} {rep:11s} sigma={sigma:.1f}  {setting:5s} vs {BASELINE:5s}  '
                  f'delta R2 = {d.mean():+.4f}   wins {int((d > 0).sum())}/{len(d)}   '
                  f'range {d.min():+.4f} .. {d.max():+.4f}')

    print('\n' + '=' * 78)
    print('MEAN FIT SECONDS')
    print('=' * 78)
    print(df.groupby(['family', 'rep', 'setting']).fit_secs.mean().round(1).to_string())

    qrf = df[df.family == 'QRF']
    if len(qrf):
        print('\n' + '=' * 78)
        print('QUANTILE FOREST — interval width and coverage at 1 sd (nominal 0.68)')
        print('=' * 78)
        print(qrf.groupby(['rep', 'sigma', 'setting'])[['interval_width', 'coverage_1sd']]
              .mean().round(4).to_string())

    print('\n' + '=' * 78)
    print('DECISION RULE: win or tie on descriptors under noise, and do not lose '
          'on Morgan;\n               ties inside the seed spread go to the faster '
          'setting.')
    print('=' * 78)
    verdict = pd.DataFrame(lines)
    for setting in sorted(set(verdict.setting)):
        s = verdict[verdict.setting == setting]
        desc_noisy = s[(s.rep == 'descriptors') & (s.sigma > 0)]
        morgan = s[s.rep == 'morgan']
        ok_desc = bool((desc_noisy.mean_delta >= -0.002).all()) if len(desc_noisy) else None
        ok_morgan = bool((morgan.mean_delta >= -0.002).all()) if len(morgan) else None
        print(f"  {setting:5s}  descriptors-under-noise ok={ok_desc}   morgan ok={ok_morgan}   "
              f"mean delta over all cells {s.mean_delta.mean():+.4f}")
    print(f"  {BASELINE:5s}  is the baseline (delta 0 by construction)")
    return verdict


def _naive_forest_quantiles(forest, X_train, y_train, X_query, quantiles):
    """Obviously-correct reference: for every query row and every tree, collect
    the in-bag training labels that land in the same leaf, pool them, take
    quantiles. Slow and stupid on purpose — it is what forest_quantiles' index
    bookkeeping has to reproduce."""
    from sklearn.ensemble._forest import (_generate_sample_indices,
                                          _get_n_samples_bootstrap)
    y_train = np.asarray(y_train, dtype=float).ravel()
    train_leaves = forest.apply(X_train)
    query_leaves = forest.apply(X_query)
    n_boot = _get_n_samples_bootstrap(len(X_train), forest.max_samples)
    out = np.empty((len(X_query), len(quantiles)))
    for r in range(len(X_query)):
        pool = []
        for i, est in enumerate(forest.estimators_):
            idx = _generate_sample_indices(est.random_state, len(X_train), n_boot)
            for j in idx:
                if train_leaves[j, i] == query_leaves[r, i]:
                    pool.append(y_train[j])
        out[r] = np.quantile(pool, quantiles)
    return out


def verify_emulation():
    """Two checks. The naive reference always runs; the library comparison runs
    wherever quantile_forest can actually be fitted."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(600, 12))
    y = X[:, 0] * 2 - X[:, 3] + rng.normal(scale=0.4, size=600)
    Xtr, ytr, Xte = X[:500], y[:500], X[500:530]
    quantiles = (0.16, 0.5, 0.84)
    kw = dict(n_estimators=50, max_features='sqrt', random_state=7, n_jobs=-1)
    rf = RandomForestRegressor(**kw).fit(Xtr, ytr)

    mine = forest_quantiles(rf, Xtr, ytr, Xte, quantiles=quantiles)
    naive = _naive_forest_quantiles(rf, Xtr, ytr, Xte, quantiles)
    err_naive = np.abs(mine - naive).max()
    print(f'vs naive reference   max absolute difference {err_naive:.3e}')
    ok = err_naive < 1e-9
    print('  the fast index bookkeeping is exact'
          if ok else '  ✗ BOOKKEEPING BUG — do not trust the emulation')

    if HAS_QRF:
        from quantile_forest import RandomForestQuantileRegressor
        real = RandomForestQuantileRegressor(**kw).fit(Xtr, ytr).predict(
            Xte, quantiles=list(quantiles))
        err_lib = np.abs(real - mine).max()
        print(f'vs quantile_forest   max absolute difference {err_lib:.6f}  '
              f'(label SD {y.std():.3f})')
        ok = ok and err_lib < 0.05 * y.std()
    else:
        print(f'vs quantile_forest   SKIPPED: {QRF_UNAVAILABLE_REASON}')
        print('  run this on the server, in the job environment, to close it out')
    print('\nEMULATION VERIFIED' if ok else '\nEMULATION NOT VERIFIED')
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', '--n-molecules', type=int, default=10000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.5])
    ap.add_argument('--reps', nargs='+', default=['descriptors', 'morgan'])
    ap.add_argument('--quick', action='store_true',
                    help='2,000 molecules and 2 seeds — a smoke test, NOT evidence')
    ap.add_argument('--verify-emulation', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.verify_emulation:
        return verify_emulation()

    if args.quick:
        args.n_molecules, args.seeds = 2000, [0, 1]

    print(json.dumps(env_fingerprint(), indent=2))
    if not HAS_QRF:
        print(f'\nNOTE quantile forest is EMULATED here: {QRF_UNAVAILABLE_REASON}\n')

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out or ('forest_max_features'
                        + ('_quick' if args.quick else '')
                        + f'_n{args.n_molecules}')
    out_csv = OUT_DIR / f'{stem}.csv'
    df = run(args.n_molecules, args.seeds, args.sigmas, args.reps,
             incremental_path=out_csv)
    df.to_csv(out_csv, index=False)
    verdict = report(df)
    verdict.to_csv(OUT_DIR / f'{stem}_paired.csv', index=False)
    (OUT_DIR / f'{stem}_env.json').write_text(json.dumps(env_fingerprint(), indent=2))
    print(f'\nwrote {OUT_DIR / f"{stem}.csv"}')
    print(f'wrote {OUT_DIR / f"{stem}_paired.csv"}')
    print(f'total {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
