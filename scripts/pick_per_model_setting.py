#!/usr/bin/env python3
"""Pick ONE setting per model, chosen at noise, that no representation pays for.

    export OMP_NUM_THREADS=1
    python scripts/pick_per_model_setting.py --models mlp_bnn_full_variational

WHAT IT DOES
------------
The two-way decomposition breaks if each model-and-representation cell carries
its own hyperparameters, so each model needs ONE setting used everywhere. The
search cannot supply it: every pairing drew its own twelve candidates, so the
six winners come from six different pools and cannot be compared.

This scores a shared pool on every representation. The pool is the model's own
six winners; each is fitted on all six representations and the default is fitted
alongside as the baseline.

CHOSEN AT NOISE, NOT ON CLEAN LABELS
------------------------------------
Selecting on clean labels is what produced the problem in the first place: clean
scoring prefers larger, less restricted models, and those are the ones that
memorise noisy labels. Training labels are noised to the level the author judges
at; the test split stays clean.

THE RULE IS MAXIMIN, WHICH IS THE POINT
---------------------------------------
A setting is scored by its WORST representation, measured against that
representation's own default, and the winner is the setting whose worst case is
best. Picking by the average would let a setting that is excellent on four
representations and ruinous on PDV come out ahead. The author's concern is
exactly that PDV, the strongest representation under noise, must not pay for a
shared choice -- so the rule protects every representation rather than the mean.

SEEDS
-----
Three, and the median is taken. The two plain Bayesian families swing by 0.3 to
0.7 between adjacent noise levels on one seed, which is larger than every effect
being chosen between; one seed cannot pick a setting for them.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import lightgbm as _lgb_first  # noqa: F401

import argparse
import csv
import json
import statistics
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

FOUR = ['mlp_bnn_full_variational', 'dnn_bnn_full_variational',
        'mlp_bnn_full', 'dnn_bnn_full']


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=FOUR)
    ap.add_argument('--reps', nargs='+',
                    default=['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns'])
    ap.add_argument('--level', type=float, default=0.5)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--sample-size', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dataset', default='qm9',
                    choices=['qm9', 'herg', 'caco2', 'logd'])
    ap.add_argument('--tag', default='pick')
    cli = ap.parse_args()

    from noiseInject import NoiseInjectorRegression
    import report_tuning_results as RT
    import tune_hyperparameters as T
    import tuning_rosters as R

    _default, cands, _s, _r = RT.load(cli.dataset)
    if not cands:
        print(f'no search results for {cli.dataset}; nothing to choose between',
              file=sys.stderr)
        return 1

    pat, M = T._import_pipeline()
    K = cache_key = None
    if cli.dataset == 'qm9':
        smiles, y, tr, va, te = T.build_split(pat, cli.sample_size, cli.seed)
    else:
        smiles, y, tr, va, te, K = T.build_validation_split(
            cli.dataset, cli.sample_size, cli.seed, None)
        cache_key = f'{cli.dataset}_seed{cli.seed}'
    spread = float(np.std(y[tr].astype(np.float64)))
    scratch = os.path.join(OUT_DIR, f'scratch_{cli.tag}.csv')

    out = os.path.join(OUT_DIR, f'per_model_{cli.tag}.csv')
    print(f'{cli.dataset}: train {len(tr)}  validation {len(va)}  '
          f'test {len(te)}  label spread {spread:.4f}', flush=True)
    fresh = not os.path.exists(out)
    fh = open(out, 'a', newline='')
    w = csv.DictWriter(fh, fieldnames=['dataset', 'model', 'from_rep',
                                       'applied_to', 'noise_seed', 'r2',
                                       'seconds', 'status', 'detail'])
    if fresh:
        w.writeheader()

    for model in cli.models:
        # The pool: this model's own winners, one per representation, deduped.
        pool, seen = [], set()
        for rep in cli.reps:
            c = cands.get((model, rep))
            if not c:
                continue
            _sc, params = max(c, key=lambda t: t[0])
            if not params:
                continue
            sig = json.dumps(params, sort_keys=True)
            if sig not in seen:
                seen.add(sig)
                pool.append((rep, params))
        pool.append(('DEFAULT', None))
        print(f'\n=== {model}: {len(pool)} settings x {len(cli.reps)} '
              f'representations x {cli.seeds} seeds ===', flush=True)

        for rep in cli.reps:
            if rep not in R.MODELS[model][4]:
                continue
            data0, _ = T.prepared_data(pat, rep, smiles, y, tr, va, te,
                                       cli.sample_size, cli.seed,
                                       K=K, cache_key=cache_key)
            for ns in range(cli.seeds):
                inj_t = NoiseInjectorRegression.from_condition(
                    'gaussian', random_state=2000 + ns)
                inj_v = NoiseInjectorRegression.from_condition(
                    'gaussian', random_state=8000 + ns)
                dose = cli.level * spread
                ntr, _a, _b = inj_t.inject_verbose(
                    data0['y_train'].astype(np.float64), dose)
                nva, _a, _b = inj_v.inject_verbose(
                    data0['y_val'].astype(np.float64), dose)
                data = dict(data0, y_train=np.asarray(ntr, dtype=np.float32),
                            y_val=np.asarray(nva, dtype=np.float32))
                for from_rep, params in pool:
                    t0 = time.perf_counter()
                    try:
                        r2, _ = T.fit_once(pat, M, model, rep, data, params,
                                           cli.sample_size, scratch, R,
                                           score_on='test')
                        status, detail = 'ok', json.dumps(params or {})
                    except Exception as exc:
                        r2, status = '', 'error'
                        detail = f'{type(exc).__name__}: {exc}'
                    secs = time.perf_counter() - t0
                    print(f'  {model[:22]:22s} on {rep:10s} using '
                          f'{from_rep:10s} seed {ns} '
                          f'R2={r2 if r2 == "" else f"{r2:.4f}"} {secs:6.1f}s',
                          flush=True)
                    w.writerow(dict(dataset=cli.dataset, model=model,
                                    from_rep=from_rep,
                                    applied_to=rep, noise_seed=ns, r2=r2,
                                    seconds=round(secs, 1), status=status,
                                    detail=detail))
                    fh.flush()
    fh.close()
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
