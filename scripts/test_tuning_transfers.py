#!/usr/bin/env python3
"""Does a setting tuned on one pairing work on another? The decisive test.

    export OMP_NUM_THREADS=1
    python scripts/test_tuning_transfers.py --models svm rf --dataset qm9

WHY A SEPARATE EXPERIMENT IS NEEDED
------------------------------------
Comparing the WINNING settings across representations, or across datasets, looks
like it answers "should we tune per representation" and does not. Each pairing
drew its own twelve random settings, so the winners come from different candidate
pools. They would differ even if the true optimum were identical everywhere.
Counting how many parameter values differ between winners measures the draws, not
the models.

The question can only be answered by APPLYING one pairing's winner to another and
seeing what it costs:

    for every pairing B, fit B with its OWN winner, and fit B with the winner
    from every other pairing A.

If A's setting scores about as well on B as B's own does, one setting serves both
and tuning per pairing is wasted effort. If it costs a lot, per-pairing tuning is
doing real work and the paper should say so.

The diagonal of that matrix is each pairing's own winner, which is already known,
so it doubles as a check: a transfer run should reproduce it.

COST
----
n pairings squared fits. Six representations is 36 per model, which is minutes
for the support vector machine and about an hour for the forest. Deliberately
restricted to the cheap models -- the answer is a property of tuning, not of any
one model, and there is no sense spending a day of Gaussian process time on it.
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
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=['svm'])
    ap.add_argument('--dataset', default='qm9')
    ap.add_argument('--sample-size', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    cli = ap.parse_args()

    import report_tuning_results as RT
    import tune_hyperparameters as T
    import tuning_rosters as R

    default, cands, sizes, _ = RT.load(cli.dataset)
    if not cands:
        print(f'no results for {cli.dataset} yet')
        return 1
    winners = {}
    for k, v in cands.items():
        if k[0] not in cli.models:
            continue
        score, params = max(v, key=lambda t: t[0])
        if params:
            winners[k] = (score, params)
    if len(winners) < 2:
        print(f'need at least two tuned pairings; have {len(winners)}')
        return 1

    pat, M = T._import_pipeline()
    smiles, y, tr, va, te = T.build_split(pat, cli.sample_size, cli.seed)
    scratch = os.path.join(OUT_DIR, 'scratch_transfer.csv')

    out = os.path.join(OUT_DIR, f'transfer_{cli.dataset}.csv')
    fh = open(out, 'w', newline='')
    w = csv.DictWriter(fh, fieldnames=['model', 'applied_to', 'setting_from',
                                       'r2', 'own_best', 'default', 'seconds',
                                       'status', 'detail'])
    w.writeheader()

    cache = {}
    for model in cli.models:
        reps = [k[1] for k in winners if k[0] == model]
        print(f'\n=== {model}: {len(reps)} representations, '
              f'{len(reps) ** 2} fits ===', flush=True)
        for target in reps:
            if target not in cache:
                cache[target] = T.prepared_data(pat, target, smiles, y, tr, va,
                                                te, cli.sample_size, cli.seed)[0]
            data = cache[target]
            for source in reps:
                params = winners[(model, source)][1]
                t0 = time.perf_counter()
                try:
                    r2, _ = T.fit_once(pat, M, model, target, data, params,
                                       cli.sample_size, scratch, R)
                    status, detail = 'ok', json.dumps(params)
                except Exception as exc:
                    r2, status, detail = '', 'error', f'{type(exc).__name__}: {exc}'
                secs = time.perf_counter() - t0
                own = winners[(model, target)][0]
                d = default.get((model, target))
                mark = '  (its own)' if source == target else ''
                print(f'  {model} on {target:10s} using {source:10s} '
                      f'R2={r2 if r2 == "" else f"{r2:+.4f}"}  '
                      f'(its own best {own:+.4f}){mark}', flush=True)
                w.writerow(dict(model=model, applied_to=target,
                                setting_from=source, r2=r2, own_best=own,
                                default=d, seconds=round(secs, 1),
                                status=status, detail=detail))
                fh.flush()
    fh.close()
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
