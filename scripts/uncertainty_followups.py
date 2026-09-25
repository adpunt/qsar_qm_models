#!/usr/bin/env python
"""Two tables the uncertainty subsection (guide §R6) quotes, written from the
harvest rather than typed into the text (the author, 2026-09-25).

1. decomposition_ratio.csv -- for each model, dataset, representation, noise
   condition and molecule set: how much each component of the uncertainty grew
   between no added noise and the highest level, and the epistemic growth
   divided by the aleatoric growth. Near 0 means the added noise went into the
   aleatoric component; 1 or above means the epistemic component grew as much or
   more. Censoring is left out: its level is a fraction clipped, not an amount.

2. censoring_spread.csv -- for each assay dataset and QM9 and each censoring
   level: the SD of the labels after the top fraction is clipped, divided by the
   clean SD, beside each model's total uncertainty at that level divided by its
   value at no censoring. The clipping rule is the injector's
   (`NoiseInject._censored_set`): the top round(fraction * n) labels by rank are
   set to the k-th largest. The spread is computed on the whole dataset, not per
   training fold, and QM9 comes from the training molecules in this repository's
   noise_provenance_*.csv files, a sample of the run's own subset.

    python scripts/uncertainty_followups.py [--harvest results/decisions_arc_20260916]
"""
import argparse
import contextlib
import glob
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
KIRBY_TESTS = ROOT.parent / 'KIRBy' / 'tests'
DECOMPOSING = ('dnn_bnn_full_mve', 'mlp_bnn_full_mve', 'het_gp_rbf', 'qrf')
DATASET_KEYS = {'logD': 'openadmet-logd', 'Caco-2': 'openadmet-caco2_efflux',
                'hERG': 'chembl-herg-ki', 'QM9': 'qm9'}


def decomposition_ratio(q5):
    d = q5[q5['condition'].notna() & (q5['condition'] != 'censoring')
           & q5['component'].isin(['aleatoric', 'epistemic'])
           & q5['model'].isin(DECOMPOSING)].copy()
    d['rep'] = d['rep'].astype(str).str.lower()
    m = (d.groupby(['split', 'model', 'dataset', 'rep', 'condition', 'component',
                    'sigma'])['mean_uncertainty'].median().unstack('sigma'))
    lo, hi = min(m.columns), max(m.columns)
    growth = (m[hi] - m[lo]).unstack('component')
    growth.columns = [f'{c}_growth' for c in growth.columns]
    growth['epistemic_over_aleatoric'] = (growth['epistemic_growth']
                                          / growth['aleatoric_growth'])
    growth['levels'] = f'{lo} to {hi}'
    return growth.reset_index()


def _clean_labels():
    sys.path.insert(0, str(KIRBY_TESTS))
    import alternative_data_noise_robustness as A  # noqa: E402
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        df = A.download_openadmet()
        logd = next(c for c in df.columns if 'LogD' in c)
        caco = next(c for c in df.columns if 'Caco' in c and 'Efflux' in c)
        out['logD'] = A.load_openadmet_endpoint(df, logd)[1]
        out['Caco-2'] = A.load_openadmet_endpoint(df, caco, log_transform=True)[1]
        out['hERG'] = A.load_chembl_herg()[1]
    files = [f for f in glob.glob(str(ROOT / 'noise_provenance_*.csv'))
             if sum(1 for _ in open(f)) > 1000]
    q = pd.concat([pd.read_csv(f) for f in files])
    q = q[q['split'] == 'train'].drop_duplicates('canonical_smiles')
    out['QM9'] = q['y_clean_raw'].to_numpy()
    return out


def clipped_spread_ratio(y, frac):
    y = np.asarray(y, dtype=float)
    k = int(round(frac * len(y)))
    if k == 0:
        return 1.0
    limit = y[np.argsort(-y, kind='stable')[k - 1]]
    return float(np.minimum(y, limit).std(ddof=1) / y.std(ddof=1))


def censoring_spread(q5):
    labels = _clean_labels()
    c = q5[(q5['condition'] == 'censoring') & (q5['split'] == 'train_oof')
           & (q5['component'] == 'total')].copy()
    c['rep'] = c['rep'].astype(str).str.lower()
    m = (c.groupby(['dataset', 'model', 'rep', 'sigma'])['mean_uncertainty']
         .median().unstack('sigma'))
    ratio = m.div(m[0.0], axis=0).stack().rename('uncertainty_ratio').reset_index()
    rows = []
    for name, key in DATASET_KEYS.items():
        for frac in sorted(ratio['sigma'].unique()):
            rows.append({'dataset': key, 'sigma': frac, 'n_labels': len(labels[name]),
                         'label_spread_ratio': clipped_spread_ratio(labels[name], frac)})
    spread = pd.DataFrame(rows)
    return ratio.merge(spread, on=['dataset', 'sigma'], how='left')


def total_uncertainty_rise(q5):
    """Per fold: mean total uncertainty at the top level divided by its value
    with no noise. One row per dataset, model, representation, condition and
    molecule set, with the lowest and highest fold. `rises_in_every_fold` is
    False when any fold's ratio is 1 or below. Censoring is left out."""
    t = q5[(q5['component'] == 'total') & q5['condition'].notna()
           & (q5['condition'] != 'censoring')].copy()
    t['rep'] = (t['rep'].astype(str).str.lower()
                .replace({'mhg-gnn-pretrained': 'mhggnn'}))
    keys = ['dataset', 'model', 'rep', 'condition', 'split']
    rows = []
    for k, g in t.groupby(keys + ['fold']):
        g = g.sort_values('sigma')
        u = g['mean_uncertainty'].to_numpy()
        rows.append(dict(zip(keys + ['fold'], k), clean=u[0], top=u[-1],
                         top_level=g['sigma'].iloc[-1],
                         ratio=u[-1] / u[0] if u[0] > 0 else np.nan))
    f = pd.DataFrame(rows)
    out = f.groupby(keys).agg(folds=('fold', 'size'), top_level=('top_level', 'max'),
                              clean_median=('clean', 'median'),
                              top_median=('top', 'median'),
                              ratio_lowest_fold=('ratio', 'min'),
                              ratio_highest_fold=('ratio', 'max')).reset_index()
    out['rises_in_every_fold'] = out['ratio_lowest_fold'] > 1
    return out


def per_sample_by_level(q4):
    """The Spearman correlation between predicted uncertainty and the size of
    the injected noise (`rho_plain_NOT_THE_ANSWER` in d7_q4.csv), out of fold,
    one row per dataset, model, representation, condition and level. Folds are
    not pooled: the median, lowest and highest fold are reported, and how many
    folds fall above and below a chance range of 1.96 / sqrt(n - 1). That range
    assumes the molecules are independent, which grouped noise breaks; it is
    an approximation, not the harvest's permutation band (that band is for the
    error, not the uncertainty)."""
    d = q4[q4['sigma'] > 0].copy()
    d['rep'] = d['rep'].astype(str).str.lower()
    r = d['rho_plain_NOT_THE_ANSWER']
    d['chance'] = 1.96 / np.sqrt(d['n'] - 1)
    d['above'] = r > d['chance']
    d['below'] = r < -d['chance']
    return (d.groupby(['dataset', 'model', 'rep', 'condition', 'sigma'])
            .agg(folds=('fold', 'size'), molecules_per_fold=('n', 'min'),
                 chance=('chance', 'max'),
                 rho_median=('rho_plain_NOT_THE_ANSWER', 'median'),
                 rho_lowest_fold=('rho_plain_NOT_THE_ANSWER', 'min'),
                 rho_highest_fold=('rho_plain_NOT_THE_ANSWER', 'max'),
                 folds_above_chance=('above', 'sum'),
                 folds_below_chance=('below', 'sum')).reset_index())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--harvest', default=str(ROOT / 'results' / 'decisions_arc_20260916'))
    args = ap.parse_args(argv)
    harvest = Path(args.harvest)
    q5 = pd.read_csv(harvest / 'unc_q5.csv', low_memory=False)
    decomposition_ratio(q5).to_csv(harvest / 'decomposition_ratio.csv', index=False)
    print(f'wrote {harvest / "decomposition_ratio.csv"}')
    censoring_spread(q5).to_csv(harvest / 'censoring_spread.csv', index=False)
    print(f'wrote {harvest / "censoring_spread.csv"}')
    total_uncertainty_rise(q5).to_csv(harvest / 'total_uncertainty_rise.csv', index=False)
    print(f'wrote {harvest / "total_uncertainty_rise.csv"}')
    q4 = pd.read_csv(harvest / 'd7_q4.csv', low_memory=False)
    per_sample_by_level(q4).to_csv(harvest / 'per_sample_by_level.csv', index=False)
    print(f'wrote {harvest / "per_sample_by_level.csv"}')


if __name__ == '__main__':
    main()
