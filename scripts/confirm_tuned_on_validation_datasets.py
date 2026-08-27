#!/usr/bin/env python3
"""Does a QM9-tuned setting still beat the default on LogD, Caco-2 and hERG?

    export OMP_NUM_THREADS=1
    python scripts/confirm_tuned_on_validation_datasets.py --time
    python scripts/confirm_tuned_on_validation_datasets.py --run
    python scripts/confirm_tuned_on_validation_datasets.py --prune

WHY THIS EXISTS
---------------
A hyperparameter tuned on QM9 is tuned on 10,000 small organic molecules with a
computed label and essentially no measurement error. LogD, Caco-2 and hERG are
drug-like molecules with real assay error, an order of magnitude fewer of them,
and a different label spread. There is no reason a setting that helps on the
first must help on the others, and a setting that HURTS them is not an
improvement -- those three datasets are what the paper's validation section
rests on.

So this runs the same head-to-head the QM9 confirmation runs, on their ground:
each adopted setting and the shared default, refitted on every fold of the same
5-fold scaffold cross-validation the experimental pipeline uses, on CLEAN
labels, and both scored on that fold's held-out molecules.

WHAT IT READS AND WHAT IT CAN CHANGE
------------------------------------
It reads results/master_tuned_hyperparameters.json -- what the cluster will
actually use -- so what is checked is what would run. `--prune` rewrites that
file, dropping any pairing that loses on more folds than it wins, and says which
ones it dropped and why. Nothing is dropped silently.

NO AVERAGING. Every fold's R-squared for both settings is written to the output,
and the summary is a COUNT of folds won, never a mean or a median over folds.

WHAT IS NOT COVERED, AND WHY
----------------------------
Only the families the experimental pipeline builds from a parameter dict --
`sklearn_params(...)` -- can be checked this way: the forest, the quantile
forest, XGBoost, LightGBM, NGBoost and the SVM. Its neural models go through
`train_neural_regression` and its Gaussian process through
`GaussianProcessGauche`, neither of which takes a tuned dict, and neither of
those families can be addressed in the tuned file today anyway
(models/tuning_rosters.py). If that changes, they need the same treatment here
before anything of theirs is adopted.

The quantile forest additionally cannot be fitted on this laptop at all --
quantile_forest 1.4.1 against scikit-learn 1.3.2 raises
"Invalid parameter 'monotonic_cst'" -- so it is reported as blocked, not skipped.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Before torch and tensorflow arrive through the KIRBy import. See
# RERUN_PLAN.md 5.7b: imported after them and fitted with its default thread
# count, LightGBM segfaults with no traceback.
import lightgbm as _lightgbm_first  # noqa: F401

import argparse
import csv
import importlib.util
import json
import time
import traceback
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tuning_rosters as rosters  # noqa: E402

MASTER = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

# The datasets, in the experimental pipeline's own names.
DATASETS = ('logd', 'caco2', 'herg_ki')

BLOCKED = {
    'qrf': ("quantile_forest 1.4.1 passes monotonic_cst to scikit-learn 1.3.2's "
            "DecisionTreeRegressor, which has no such parameter."),
}


def load_kirby():
    """The experimental pipeline, loaded as a module.

    Its loaders, its representations, its scaffold grouping and its fold count
    are used as they are. Restating any of them here would be restating the
    thing this check is supposed to be checking against.
    """
    spec = importlib.util.spec_from_file_location(
        'kirby_validation_pipeline', rosters.KIRBY_PIPELINE_PATH)
    if spec is None:
        raise RuntimeError(f'no experimental pipeline at '
                           f'{rosters.KIRBY_PIPELINE_PATH}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset_molecules(K, name, openadmet_csv=None):
    """(smiles, labels) for one dataset, through the pipeline's own loaders."""
    if name == 'herg_ki':
        return K.load_chembl_herg()
    df = K.download_openadmet(csv_path=openadmet_csv)
    if name == 'logd':
        col = next((c for c in df.columns if 'LogD' in c), None)
        if col is None:
            raise RuntimeError('no LogD column in the OpenADMET table')
        return K.load_openadmet_endpoint(df, col, log_transform=False)
    if name == 'caco2':
        col = next((c for c in df.columns
                    if 'Caco' in c and 'Efflux' in c), None)
        if col is None:
            raise RuntimeError('no Caco-2 efflux column in the OpenADMET table')
        return K.load_openadmet_endpoint(df, col, log_transform=True)
    raise KeyError(name)


def build_model(K, key, params):
    """One estimator, built the way the experimental pipeline builds it.

    `params` None means the shared default, taken from models/model_defaults.py
    through the pipeline's own accessor -- so the baseline in this comparison is
    the value both studies actually run, not a library default.
    """
    if key == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=42, n_jobs=-1,
                                     **(params or K.sklearn_params('rf')))
    if key == 'qrf':
        from quantile_forest import RandomForestQuantileRegressor
        return RandomForestQuantileRegressor(
            random_state=42, n_jobs=-1, **(params or K.sklearn_params('qrf')))
    if key == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42,
                            **(params or K.sklearn_params('xgboost')))
    if key == 'lgb':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1,
                             **(params or K.sklearn_params('lightgbm')))
    if key == 'ngboost':
        from ngboost import NGBRegressor
        if params is None:
            return NGBRegressor(random_state=42, verbose=False,
                                **K._ngboost_kwargs())
        # A tuned dict carries neither Dist nor Score; they come from the spec
        # by name, exactly as models.py resolves them.
        kw = K._ngboost_kwargs()
        kw.update(params)
        return NGBRegressor(random_state=42, verbose=False, **kw)
    if key == 'svm':
        from sklearn.svm import SVR
        return SVR(**(params or K.sklearn_params('svm')))
    raise KeyError(f'no experimental builder for tuned key {key!r}. The neural '
                   f'and Gaussian-process families do not take a parameter '
                   f'dict on that side -- see the module docstring.')


def score_fold(K, model, X_train, y_train, X_test, y_test, rep_name):
    """Fit and score one fold, with the pipeline's own scaling conventions.

    Features are standardised only when the shared spec says so, fitted on the
    training fold. Targets are standardised on the training fold, because the
    SVM's C, the Gaussian process's noise and NGBoost's learning rate are all
    stated relative to unit variance; predictions come back to raw units before
    R-squared, so the number means the same thing on every dataset.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    if K.should_standardise(X_train, rep_name):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    y_scaler = StandardScaler().fit(np.asarray(y_train, float).reshape(-1, 1))
    y_fit = y_scaler.transform(np.asarray(y_train, float).reshape(-1, 1)).ravel()
    model.fit(X_train, y_fit)
    pred = y_scaler.inverse_transform(
        np.asarray(model.predict(X_test), float).reshape(-1, 1)).ravel()
    return float(r2_score(y_test, pred))


def adopted_pairings():
    """What the master file would hand the cluster, as (tuned key, QM9 rep)."""
    if not os.path.exists(MASTER):
        return None
    with open(MASTER) as fh:
        master = json.load(fh)
    return [(key, rep) for key, by_rep in sorted(master.items())
            for rep in sorted(by_rep)], master


def run(cli):
    from sklearn.model_selection import GroupKFold

    K = load_kirby()
    os.makedirs(OUT_DIR, exist_ok=True)

    found = adopted_pairings()
    if found is None:
        print(f'{MASTER} does not exist. Run the sweep, then --confirm, then '
              f'--write-master, and this checks what that produced.')
        return 1
    pairings, master = found
    if not pairings:
        print(f'{MASTER} is empty -- nothing was adopted, so there is nothing '
              f'to check here.')
        return 0

    datasets = cli.datasets or list(DATASETS)
    n_folds = K.N_FOLDS
    print(f'{len(pairings)} adopted pairing(s) x {len(datasets)} dataset(s) x '
          f'{n_folds} folds x 2 settings')
    if cli.time:
        print('TIMING: one fold of each pairing on each dataset, at the '
              'DEFAULT only, then the arithmetic.')

    out = os.path.join(OUT_DIR, 'validation_dataset_confirmation.csv')
    fields = ['dataset', 'model_key', 'qm9_rep', 'experimental_model',
              'experimental_rep', 'fold', 'default_r2', 'tuned_r2', 'delta',
              'seconds', 'status', 'detail', 'n_molecules', 'written']
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    fh_out = open(out, 'w', newline='')
    writer = csv.DictWriter(fh_out, fieldnames=fields)
    writer.writeheader()
    rows = []

    for ds in datasets:
        print(f'\n=== {ds} ===', flush=True)
        smiles, labels = dataset_molecules(K, ds, cli.openadmet_csv)
        print(f'  {len(smiles)} molecules', flush=True)
        groups, n_scaffolds = K.assign_scaffold_groups(smiles)
        print(f'  {n_scaffolds} scaffolds', flush=True)

        wanted_reps = sorted({rosters.REP_NAME_MAP[r] for _, r in pairings})
        reps = K.generate_representations(smiles, rep_filter=wanted_reps)

        splits = list(GroupKFold(n_splits=n_folds).split(smiles, labels, groups))
        if cli.time:
            splits = splits[:1]

        for key, qm9_rep in pairings:
            exp_rep = rosters.REP_NAME_MAP[qm9_rep]
            exp_model = next(
                (rosters.MODEL_NAME_MAP[m] for m in rosters.TUNED_KEY
                 if rosters.TUNED_KEY[m] == key), key)
            base = dict(dataset=ds, model_key=key, qm9_rep=qm9_rep,
                        experimental_model=exp_model, experimental_rep=exp_rep,
                        n_molecules=len(smiles), written=stamp)

            if key in BLOCKED:
                row = dict(base, fold='', default_r2='', tuned_r2='', delta='',
                           seconds='', status='blocked', detail=BLOCKED[key])
                writer.writerow(row); fh_out.flush(); rows.append(row)
                print(f'  {key} x {qm9_rep}: BLOCKED — {BLOCKED[key]}', flush=True)
                continue
            if exp_rep not in reps:
                row = dict(base, fold='', default_r2='', tuned_r2='', delta='',
                           seconds='', status='error',
                           detail=f'{exp_rep} was not built for {ds}')
                writer.writerow(row); fh_out.flush(); rows.append(row)
                print(f'  {key} x {qm9_rep}: {exp_rep} unavailable', flush=True)
                continue

            X = reps[exp_rep]
            tuned = master[key][qm9_rep]
            for fold, (tr, te) in enumerate(splits):
                t0 = time.perf_counter()
                try:
                    d = score_fold(K, build_model(K, key, None),
                                   X[tr], labels[tr], X[te], labels[te], exp_rep)
                    if cli.time:
                        t, delta = '', ''
                    else:
                        t = score_fold(K, build_model(K, key, tuned),
                                       X[tr], labels[tr], X[te], labels[te],
                                       exp_rep)
                        delta = t - d
                    status, detail = 'ok', ''
                except Exception as exc:
                    d = t = delta = ''
                    status, detail = 'error', f'{type(exc).__name__}: {exc}'
                    if cli.traceback:
                        traceback.print_exc()
                secs = time.perf_counter() - t0
                row = dict(base, fold=fold, default_r2=d, tuned_r2=t,
                           delta=delta, seconds=secs, status=status,
                           detail=detail)
                writer.writerow(row); fh_out.flush(); rows.append(row)
                print(f'  {key} x {qm9_rep} fold {fold}: '
                      f'default {d if d == "" else f"{d:+.4f}"}  '
                      f'tuned {t if t == "" else f"{t:+.4f}"}  '
                      f'{secs:.1f}s  {status}{"  " + detail[:90] if detail else ""}',
                      flush=True)

    fh_out.close()
    print(f'\nwrote {out}  ({len(rows)} rows)')
    if cli.time:
        report_cost(rows, len(DATASETS if not cli.datasets else cli.datasets),
                    n_folds)
    else:
        report_counts(rows)
    return 0


def report_cost(rows, n_datasets, n_folds):
    """One fold at the default, then the multiplication. Two settings per fold,
    so the sweep of this check is two fits where the timing did one."""
    ok = [r for r in rows if r['status'] == 'ok']
    one = sum(float(r['seconds']) for r in ok)
    print('\n' + '=' * 68)
    print(f'ONE FOLD AT THE DEFAULT, {len(ok)} pairing-dataset combinations')
    print('=' * 68)
    for r in sorted(ok, key=lambda r: -float(r['seconds']))[:10]:
        print(f'  {float(r["seconds"]):7.1f}s  {r["dataset"]}  '
              f'{r["model_key"]} x {r["qm9_rep"]}')
    print(f'\n  one fold, default only, everything: {one / 60:.1f} min')
    print(f'  the full check ({n_folds} folds x 2 settings): '
          f'{one * n_folds * 2 / 3600:.2f} h')
    print('  representation building is NOT in that number and is paid once '
          'per dataset.')


def report_counts(rows):
    """Folds won, not a mean over folds. A mean would hide a setting that wins
    big on one fold and loses on the other four."""
    print('\n' + '=' * 68)
    print('FOLDS WON BY THE TUNED SETTING')
    print('=' * 68)
    tally = {}
    for r in rows:
        if r['status'] != 'ok' or r['delta'] in ('', None):
            continue
        k = (r['dataset'], r['model_key'], r['qm9_rep'])
        won, total = tally.get(k, (0, 0))
        tally[k] = (won + (1 if float(r['delta']) > 0 else 0), total + 1)
    for (ds, key, rep), (won, total) in sorted(tally.items()):
        verdict = 'tuned' if won * 2 > total else 'DEFAULT'
        print(f'  {ds:8s} {key:8s} {rep:15s} {won}/{total} folds  -> {verdict}')
    losers = [k for k, (won, total) in tally.items() if won * 2 <= total]
    if losers:
        print(f'\n  {len(losers)} pairing-dataset combination(s) where the '
              f'default holds. --prune drops those pairings from the master '
              f'file.')


def prune(cli):
    """Drop from the master file every pairing that loses on the real datasets.

    Losing means winning no more than half the folds it ran, on any dataset. A
    setting that costs accuracy on LogD, Caco-2 or hERG is not adopted, whatever
    it did on QM9 -- those three are the datasets with real measurement error in
    them, and they are what the validation section rests on.
    """
    path = os.path.join(OUT_DIR, 'validation_dataset_confirmation.csv')
    if not os.path.exists(path):
        print(f'{path} does not exist -- run --run first.')
        return 1
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    tally = {}
    for r in rows:
        if r['status'] != 'ok' or r['delta'] in ('', None):
            continue
        k = (r['model_key'], r['qm9_rep'], r['dataset'])
        won, total = tally.get(k, (0, 0))
        tally[k] = (won + (1 if float(r['delta']) > 0 else 0), total + 1)

    drop = {}
    for (key, rep, ds), (won, total) in tally.items():
        if won * 2 <= total:
            drop.setdefault((key, rep), []).append(f'{ds} {won}/{total} folds')

    with open(MASTER) as fh:
        master = json.load(fh)
    removed = []
    for (key, rep), why in sorted(drop.items()):
        if key in master and rep in master[key]:
            del master[key][rep]
            removed.append((key, rep, '; '.join(sorted(why))))
    master = {k: v for k, v in master.items() if v}

    with open(MASTER, 'w') as fh:
        json.dump(master, fh, indent=2)
    decisions_path = os.path.join(_ROOT, 'results',
                                  'hyperparameter_decisions.json')
    with open(decisions_path, 'w') as fh:
        json.dump({k: 'USE_TUNED' for k in master}, fh, indent=2)

    print(f'dropped {len(removed)} pairing(s) from {MASTER}:')
    for key, rep, why in removed:
        print(f'  {key} x {rep} — the default holds on {why}')
    kept = sum(len(v) for v in master.values())
    print(f'{kept} pairing(s) kept, across {len(master)} model key(s)')
    print(f'rewrote {decisions_path}')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--time', action='store_true',
                      help='One fold of each pairing on each dataset, at the '
                           'default only, then the arithmetic.')
    mode.add_argument('--run', action='store_true',
                      help='The full head-to-head, every fold, both settings.')
    mode.add_argument('--prune', action='store_true',
                      help='Drop from the master file every pairing the default '
                           'beats on these datasets.')
    ap.add_argument('--datasets', nargs='*', choices=DATASETS,
                    help=f'Subset of {DATASETS}.')
    ap.add_argument('--openadmet-csv', default=None,
                    help='Local OpenADMET table, if it is not to be downloaded.')
    ap.add_argument('--traceback', action='store_true')
    cli = ap.parse_args()

    if cli.prune:
        return prune(cli)
    return run(cli)


if __name__ == '__main__':
    sys.exit(main())
