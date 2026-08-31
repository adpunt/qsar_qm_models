#!/usr/bin/env python3
"""Does a setting tuned on clean labels still help once the labels are noisy?

    export OMP_NUM_THREADS=1
    python scripts/tuned_under_noise.py --reps pdv chemberta

WHY THIS EXISTS
---------------
The whole search scored candidates on clean labels. The study is about noise. A
setting that is best at zero noise is not automatically best at any other level,
and if the advantage shrinks or reverses as noise rises, then adopting tuned
settings biases the very comparison the paper makes.

Nothing in the search can answer that, because the search never saw a noisy
label. This fits each pairing twice at every noise level -- once at its default,
once at the setting the search picked -- and reports both.

WHAT IS HELD FIXED
------------------
One scaffold split of QM9, one seed, no replicates. Training and validation
labels are noised, each with its own independent draw, at a dose set against the
CLEAN TRAINING spread, which is the settled scale. The test split stays clean,
because robustness means predicting the truth after being taught something
false. Scores are R-squared on that clean test split.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
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
LEVELS = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]

# Cheap first, so the answer starts arriving before the slow families finish.
ORDER = ['svm', 'mlp', 'xgboost', 'lgb', 'rf', 'dnn', 'qrf',
         'mlp_bnn_full', 'dnn_bnn_full',
         'mlp_bnn_full_variational', 'dnn_bnn_full_variational', 'ngboost']


def winners(dataset='qm9'):
    """The setting the completed search picked, per (model, representation)."""
    import report_tuning_results as RT
    import tuning_rosters as R
    default, cands, _sizes, _rng = RT.load(dataset)
    out = {}
    for k, c in cands.items():
        score, params = max(c, key=lambda t: t[0])
        if params:
            out[k] = params
    return out, default


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', nargs='+', default=['pdv', 'chemberta'])
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--sample-size', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--condition', default='gaussian')
    ap.add_argument('--tag', default='noise')
    cli = ap.parse_args()

    from noiseInject import NoiseInjectorRegression
    import tune_hyperparameters as T
    import tuning_rosters as R

    win, _default = winners()
    models = cli.models or [m for m in ORDER if m in R.MODELS]

    pat, M = T._import_pipeline()
    smiles, y, tr, va, te = T.build_split(pat, cli.sample_size, cli.seed)
    scratch = os.path.join(OUT_DIR, f'scratch_{cli.tag}.csv')

    # The dose is a fraction of the CLEAN TRAINING spread, which is the settled
    # scale. It must be computed once, from the clean labels, and reused at every
    # level -- computing it per level from noisy labels is the confound that was
    # removed from the pipeline in August.
    spread = float(np.std(y[tr].astype(np.float64)))
    print(f'clean training label spread: {spread:.4f} '
          f'({len(tr)} train, {len(va)} validation, {len(te)} test)', flush=True)

    out = os.path.join(OUT_DIR, f'noise_vs_tuned_{cli.tag}.csv')
    fresh = not os.path.exists(out)
    fh = open(out, 'a', newline='')
    w = csv.DictWriter(fh, fieldnames=['model', 'rep', 'level', 'setting', 'r2',
                                       'seconds', 'status', 'detail'])
    if fresh:
        w.writeheader()

    for rep in cli.reps:
        data0, _ = T.prepared_data(pat, rep, smiles, y, tr, va, te,
                                   cli.sample_size, cli.seed)
        for level in LEVELS:
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
                # InjectionResult unpacks as (labels, scale, epsilon); it does
                # not index.
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
                for label, params in (('default', None),
                                      ('tuned', win.get((model, rep)))):
                    if label == 'tuned' and params is None:
                        continue
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
                    print(f'  {rep:10s} level {level:<5} {model:26s} {label:8s} '
                          f'R2={r2 if r2 == "" else f"{r2:+.4f}"}  '
                          f'{secs:6.1f}s {status if status != "ok" else ""}',
                          flush=True)
                    w.writerow(dict(model=model, rep=rep, level=level,
                                    setting=label, r2=r2, seconds=round(secs, 1),
                                    status=status, detail=detail))
                    fh.flush()
    fh.close()
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
