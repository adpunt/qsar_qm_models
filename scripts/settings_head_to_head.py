#!/usr/bin/env python3
"""Every candidate setting against every model in the paired comparisons.

    export OMP_NUM_THREADS=1
    python scripts/settings_head_to_head.py --dataset qm9 --reps pdv chemberta

WHY THIS EXISTS
---------------
`results/master_tuned_hyperparameters.json` gives four models a searched
setting and the other fifteen the shared default, so seven of the eight paired
comparisons in `d10_probabilistic.csv` compare two different networks as well
as two different amounts of probabilistic machinery. Choosing one setting per
family closes that, and this measures which setting to choose.

It is not a search. The candidates are exactly the settings already on disk in
the two tuned files, plus the shared default, and each one is fitted on every
model in the family rather than only on the model it was picked for.

WHAT IS HELD FIXED
------------------
One scaffold split, one seed, no replicates. Training and validation labels are
noised, each with its own independent draw, at a dose set against the CLEAN
TRAINING spread. The test split stays clean and is what R-squared is measured
on, because robustness means predicting the truth after being taught something
false. The candidate reaches the model through `load_best_hyperparameters`,
which is the branch `--use-best-params` fires on the cluster.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import lightgbm as _lgb_first  # noqa: F401  (before torch; see the tuner)

import argparse
import csv
import json
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

# The settled dose grid, as a fraction of the clean training label spread.
# Clean, the level the paper reports at, and one above it.
LEVELS = [0.0, 0.5, 1.0]

# ---------------------------------------------------------------------------
# THE CANDIDATES
#
# Every distinct setting in results/master_tuned_hyperparameters.json and
# results/master_tuned_hyperparameters_lab.json, named by the model and dataset
# it was picked for, plus the shared default from models/model_defaults.py.
# `None` means the default: it is delivered by leaving the tuned branch alone,
# so the default row is the pipeline's own default and not a retyped copy.
#
# NN-alpha reads {activation, hidden_size1, hidden_size2}; the two remaining
# keys, lr and dropout_rate, fall back to the spec in both families.
# NN-beta reads {hidden_size, num_hidden_layers, dropout_rate, lr}.
# ---------------------------------------------------------------------------
ALPHA = [
    ('default',        None),
    ('a_qm9_bnn',      {'activation': 'tanh', 'hidden_size1': 64,  'hidden_size2': 32}),
    ('a_qm9_vbll',     {'activation': 'tanh', 'hidden_size1': 64,  'hidden_size2': 64}),
    ('a_caco2_vbll',   {'activation': 'relu', 'hidden_size1': 64,  'hidden_size2': 32}),
    ('a_logd_vbll',    {'activation': 'tanh', 'hidden_size1': 256, 'hidden_size2': 32}),
]

BETA = [
    ('default',        None),
    ('b_qm9_bnn',      {'hidden_size': 64,  'num_hidden_layers': 1,
                        'dropout_rate': 0.379, 'lr': 0.004285944143830873}),
    ('b_qm9_vbll',     {'hidden_size': 64,  'num_hidden_layers': 1,
                        'dropout_rate': 0.357, 'lr': 0.001185606743480818}),
    ('b_caco2_bnn',    {'hidden_size': 64,  'num_hidden_layers': 3,
                        'dropout_rate': 0.401, 'lr': 0.005865018723574268}),
    ('b_herg_bnn',     {'hidden_size': 64,  'num_hidden_layers': 2,
                        'dropout_rate': 0.169, 'lr': 0.006017629069106814}),
    ('b_logd_vbll',    {'hidden_size': 128, 'num_hidden_layers': 2,
                        'dropout_rate': 0.079, 'lr': 0.00011017131165932221}),
]

# The forests carry no tuned entry at all. What separates them is the spec:
# the plain forest is built with 100 trees and the quantile forest with 300
# (models/model_defaults.py), everything else matching. So the candidates here
# are the two tree counts, each fitted on both forests.
FOREST_TREES = [100, 300]

ALPHA_MODELS = ['dnn', 'dnn_bnn_full', 'dnn_bnn_full_variational',
                'dnn_bnn_full_mve', 'dnn_bnn_full_variational_hetero']
BETA_MODELS = ['mlp', 'mlp_bnn_full', 'mlp_bnn_full_variational',
               'mlp_bnn_full_mve', 'mlp_bnn_full_variational_hetero']
FOREST_MODELS = ['rf', 'qrf']


def candidates_for(model, rosters):
    """(setting name, parameter dict or None) for one model."""
    if model in ALPHA_MODELS:
        return list(ALPHA)
    if model in BETA_MODELS:
        return list(BETA)
    if model in FOREST_MODELS:
        import model_defaults as D
        out = []
        for n in FOREST_TREES:
            params = dict(D.sklearn_params('rf'))
            params['n_estimators'] = n
            out.append((f'trees{n}', params))
        return out
    raise KeyError(f'no candidate list for {model!r}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='qm9',
                    choices=['qm9', 'logd', 'caco2', 'herg'])
    ap.add_argument('--reps', nargs='+', default=['pdv', 'chemberta'])
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--sample-size', type=int, default=10000,
                    help='QM9 only; the grid runs 10000.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--condition', default='gaussian')
    ap.add_argument('--levels', nargs='+', type=float, default=None)
    ap.add_argument('--openadmet-csv', default=None)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--time', action='store_true',
                    help='Fit the default of each model once, clean, and stop. '
                         'Costs the full run before committing to it.')
    cli = ap.parse_args()

    from noiseInject import NoiseInjectorRegression
    import tune_hyperparameters as T
    import tuning_rosters as R

    T.unblock_quantile_forest()
    tag = cli.tag or cli.dataset
    models = cli.models or (ALPHA_MODELS + BETA_MODELS + FOREST_MODELS)
    levels = [0.0] if cli.time else (cli.levels or LEVELS)

    pat, M = T._import_pipeline()
    K = cache_key = None
    if cli.dataset == 'qm9':
        smiles, y, tr, va, te = T.build_split(pat, cli.sample_size, cli.seed)
    else:
        smiles, y, tr, va, te, K = T.build_validation_split(
            cli.dataset, 0, cli.seed, cli.openadmet_csv)
        cache_key = f'{cli.dataset}_seed{cli.seed}'
    y = np.asarray(y)

    spread = float(np.std(y[tr].astype(np.float64)))
    print(f'{cli.dataset}: {len(tr)} train, {len(va)} validation, {len(te)} test '
          f'molecules; clean training label spread {spread:.4f}', flush=True)

    scratch = os.path.join(OUT_DIR, f'scratch_h2h_{tag}.csv')
    out = os.path.join(OUT_DIR, f'head_to_head_{tag}.csv')
    fresh = not os.path.exists(out)
    fh = open(out, 'a', newline='')
    w = csv.DictWriter(fh, fieldnames=['dataset', 'rep', 'level', 'model',
                                       'setting', 'r2', 'seconds', 'status',
                                       'detail'])
    if fresh:
        w.writeheader()

    for rep in cli.reps:
        t0 = time.perf_counter()
        data0, scaled = T.prepared_data(pat, rep, smiles, y, tr, va, te,
                                        cli.sample_size, cli.seed,
                                        K=K, cache_key=cache_key)
        print(f'featurised {rep} in {time.perf_counter() - t0:.1f}s '
              f'(standardised: {scaled})', flush=True)

        for level in levels:
            if level == 0.0:
                y_tr, y_va = data0['y_train'], data0['y_val']
            else:
                # Independent draws for the two splits: validation carries its
                # own noise, settled 2026-08-27. Same condition, same dose.
                inj_t = NoiseInjectorRegression.from_condition(
                    cli.condition, random_state=1000 + int(level * 100))
                inj_v = NoiseInjectorRegression.from_condition(
                    cli.condition, random_state=9000 + int(level * 100))
                dose = level * spread
                noisy_tr, _s, _e = inj_t.inject_verbose(
                    data0['y_train'].astype(np.float64), dose)
                noisy_va, _s, _e = inj_v.inject_verbose(
                    data0['y_val'].astype(np.float64), dose)
                y_tr = np.asarray(noisy_tr, dtype=np.float32)
                y_va = np.asarray(noisy_va, dtype=np.float32)
            data = dict(data0, y_train=y_tr, y_val=y_va)

            for model in models:
                if rep not in R.MODELS[model][4]:
                    continue
                cands = candidates_for(model, R)
                if cli.time:
                    cands = [c for c in cands if c[1] is None] or cands[:1]
                for name, params in cands:
                    t1 = time.perf_counter()
                    try:
                        r2, _ = T.fit_once(pat, M, model, rep, data, params,
                                           cli.sample_size, scratch, R,
                                           score_on='test')
                        status, detail = 'ok', json.dumps(params or {})
                    except Exception as exc:
                        r2, status = '', 'error'
                        detail = f'{type(exc).__name__}: {exc}'
                    secs = time.perf_counter() - t1
                    shown = r2 if r2 == '' else f'{r2:+.4f}'
                    print(f'  {rep:10s} level {level:<5} {model:34s} '
                          f'{name:14s} R2={shown}  {secs:7.1f}s '
                          f'{status if status != "ok" else ""}', flush=True)
                    w.writerow(dict(dataset=cli.dataset, rep=rep, level=level,
                                    model=model, setting=name, r2=r2,
                                    seconds=round(secs, 1), status=status,
                                    detail=detail))
                    fh.flush()
    fh.close()
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
