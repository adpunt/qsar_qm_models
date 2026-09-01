#!/usr/bin/env python3
"""Score every candidate setting for a model on every representation.

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

TWO RUNS, AND THIS SCRIPT IS BOTH OF THEM
-----------------------------------------
--level 0.0 on all six representations is the MAIN experiment. Its table --
every candidate's change from the default on each representation, and the mean
of those changes -- is what the setting is ranked on.

--level 0.5 on PDV and ChemBERTa is a separate run that supplies ONE COLUMN of
that table. Training labels are noised to half the clean training label spread;
the test split stays clean. It is a filter, not the ranking.

The noise run exists because clean scoring prefers larger, less restricted
models, and those are the ones that memorise noisy labels: LightGBM's clean
winner on ECFP4 ends at 0.226 under noise where its own default holds 0.634. Two
representations are enough to catch that, by the author's decision.

RANKED BY THE MEAN CHANGE FROM THE DEFAULT
------------------------------------------
Not by the raw score, which would rank the representations rather than the
settings, and not by the worst representation, which was the rule until
2026-09-01. If ECFP4's default scores 0.6 and this setting scores 0.7 that is
+0.1; if PDV's default scores 0.7 and this setting scores 0.5 that is -0.2; the
setting is ranked on mean(+0.1, -0.2) = -0.05.

The ranking itself is applied in scripts/write_chosen_settings.py. This script
only measures.

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


def _defuse_keops_compiler_probe():
    """Stop KeOps racing every other process over one fixed file.

    keopscore's get_use_Apple_clang writes `c++ --version` to the HARD-CODED
    path /tmp/compiler_version.txt, reads it, then deletes it. Every process on
    this machine uses the same path, so two of them starting close together
    means one deletes the file while the other is reading, and the loser dies
    with FileNotFoundError before fitting anything. It killed LogD/SNS for an
    hour on 2026-08-31 and Caco-2 variational beta repeatedly on 2026-09-01,
    and it looks exactly like a memory kill from the outside.

    The probe only asks whether the compiler is Apple clang. On macOS with the
    default toolchain it is -- `c++ --version` reports Apple clang 16 here -- so
    answering directly is both correct and race-free. On anything other than
    macOS the real probe is left alone.
    """
    import platform
    if platform.system() != 'Darwin':
        return
    try:
        from keopscore.config.base_config import Config
        Config.get_use_Apple_clang = lambda self: True
    except Exception:
        pass  # KeOps not installed, or its internals moved; nothing to defuse


_defuse_keops_compiler_probe()
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
    # Avalon dropped from the study 2026-09-01 (author's decision).
    ap.add_argument('--reps', nargs='+',
                    default=['pdv', 'chemberta', 'ecfp4', 'mhggnn', 'sns'])
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
    FIELDS = ['dataset', 'level', 'model', 'from_rep', 'applied_to',
              'noise_seed', 'r2', 'seconds', 'status', 'detail']
    # A RESTART MUST NOT APPEND A WIDER ROW THAN THE HEADER. When 'dataset'
    # and 'level' were added, restarted jobs wrote ten fields under a
    # nine-field header, and every later row was read one or two columns out of
    # step -- which is how rows appeared claiming the model was called 'qm9'
    # or '0.5'. If the layout on disk is not the layout being written, the old
    # file is set aside rather than appended to.
    if os.path.exists(out):
        with open(out) as _fh:
            head = _fh.readline().strip().split(',')
        if head != FIELDS:
            os.rename(out, out + '.oldlayout')
            print(f'set aside {out}: header {head} is not {FIELDS}', flush=True)
    fresh = not os.path.exists(out)
    fh = open(out, 'a', newline='')
    # THE LEVEL IS A COLUMN. Without it a clean run and a noise run are
    # indistinguishable once written, and the only thing separating them was the
    # tag someone happened to choose.
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        w.writeheader()
        # FLUSH THE HEADER NOW. It was sitting in the buffer until the first row
        # was written, so a run killed during its first fit -- which is what
        # memory pressure does -- left a ZERO-BYTE file. The next run then read
        # its header as [''], decided the layout had changed, set the empty file
        # aside and started again, so a memory kill looked like a layout
        # problem and burned a restart each time.
        fh.flush()

    # WHAT IS ALREADY ON DISK. A restart used to refit everything from the top
    # of the pool, so stopping a run to change its representations threw away
    # every fit it had done. Combinations already recorded are skipped instead.
    # ACROSS EVERY FILE, not just this run's own. A fit is a fit: it is
    # identified by dataset, noise level, model, representation and setting, and
    # which process happened to write it is irrelevant. Reading only this run's
    # own file meant that renaming a file, splitting a job by model, or
    # recovering rows from an older layout all caused the same fits to be
    # computed again -- 79 of them on 2026-09-01.
    import glob as _glob
    already = set()
    for _f in _glob.glob(os.path.join(OUT_DIR, 'per_model_*.csv')):
        try:
            with open(_f) as _fh:
                for _r in csv.DictReader(_fh):
                    if _r.get('status') != 'ok':
                        continue
                    if (_r.get('dataset') or 'qm9') != cli.dataset:
                        continue
                    try:
                        if abs(float(_r.get('level') or 0.0) - cli.level) > 1e-9:
                            continue
                    except ValueError:
                        continue
                    already.add((_r.get('model'), _r.get('applied_to'),
                                 _r.get('from_rep')))
        except OSError:
            continue
    if already:
        print(f'resuming: {len(already)} fits already recorded', flush=True)

    for model in cli.models:
        # The pool: this model's own winners, one per representation, deduped.
        # THE POOL IS REPRESENTATION-AGNOSTIC. It was built from cli.reps,
        # so asking for two representations silently shrank the candidate list
        # from seven settings to three -- and the noise filter then covered
        # fewer settings than the clean table it was meant to filter.
        # A candidate setting belongs to the MODEL. Which representations it
        # gets scored on is a separate question.
        pool, seen = [], set()
        searched = [r for (m, r) in cands if m == model]
        for rep in sorted(set(searched), key=lambda r: (r not in cli.reps, r)):
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
        # THE DEFAULT IS FITTED FIRST, not last. Every number in the table is
        # a change FROM the default, so a representation's whole column stays
        # blank until its default lands. With the default last, ECFP4 showed as
        # empty on QM9 Bayesian alpha when three of its six fits were already
        # done, and the table read as though no data had been collected.
        pool.insert(0, ('DEFAULT', None))
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
                    if (model, rep, from_rep) in already:
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
                    print(f'  {model[:22]:22s} on {rep:10s} using '
                          f'{from_rep:10s} seed {ns} '
                          f'R2={r2 if r2 == "" else f"{r2:.4f}"} {secs:6.1f}s',
                          flush=True)
                    w.writerow(dict(dataset=cli.dataset, level=cli.level,
                                    model=model, from_rep=from_rep,
                                    applied_to=rep, noise_seed=ns, r2=r2,
                                    seconds=round(secs, 1), status=status,
                                    detail=detail))
                    fh.flush()
    fh.close()
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
