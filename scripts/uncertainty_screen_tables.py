#!/usr/bin/env python3
"""The uncertainty screen: which models and which representations earn a place
in the uncertainty runs.

WHAT THIS DECIDES
-----------------
Two lists at the top of `slurm_scripts_uncertainty_rerun/generate_scripts.py`:
MODELS (seven) and REPS (four of the study's six). Those two lists are what
makes that run 420 jobs. A model whose predicted uncertainty says nothing about
which labels were corrupted, and nothing about where the error is, produces rows
nobody can use.

It decides nothing about the noise types. Those are settled
(`noise_conditions.json`, 2026-08-27) and this screen runs ONE of them, plain
Gaussian, so every job is spent on the open question.

WHICH STATISTIC DECIDES IT, AND WHY IT IS NOT THE PAPER'S
---------------------------------------------------------
`paper.tex` `tab:top_unc_noise` reports the plain Spearman correlation between
predicted uncertainty and the size of the injected noise. That column is
reported here, because it is the one a reader will look for -- but it CANNOT
rank the models. Under cross-fitting the model that scores a molecule never saw
that molecule's noise draw, and under uniform Gaussian noise every molecule
receives the same amount, so nothing in the label carries the individual draw
and the honest answer is near zero for every model alike
(`uncertainty_stats.q4_plain_correlation` says so in its own docstring). A large
value there is evidence of leakage, not of skill.

What separates the models is three things, all out of fold, all computed inside
one noise level:

  rho_error / auc_error   does the cross-fitted ERROR find the corrupted labels
                          (the same for every model up to its accuracy -- the
                          baseline the uncertainty has to beat)
  rho_delta / auc_delta   does dividing that error by the predicted uncertainty
                          find them BETTER. This is the answer to the question
                          the uncertainty runs exist to ask, and zero means the
                          uncertainty added nothing
  rho_unc_vs_clean_error  does the uncertainty rank the size of the error
                          against the CLEAN label -- the everyday use, and the
                          one a practitioner acts on

with coverage at 1 and 2 sigma beside them as the calibration column the paper
already reports, and the clean R2 beside that so a model is never praised for
ranking error it has plenty of.

USAGE
-----
    python scripts/uncertainty_screen_tables.py \
        --run-dir /Volumes/seagate/chatN_screen \
        --out-dir results/uncertainty_screen

One table per representation, one row per model, nothing pooled across
representations and nothing averaged across noise levels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as us  # noqa: E402

# The screen's own grid. Level 0 is the negative control: with no noise injected
# every noise-based statistic is undefined there by construction, and coverage
# and the error ranking are still defined.
REPORT_LEVEL = 1.5
CONTROL_LEVEL = 0.0

# Rows are the seven models the uncertainty generator would queue, in the
# generator's own spelling, keyed by what the QM9 pipeline writes.
MODEL_LABELS = {
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    'gauche': 'GP',
    'gauche_rbf': 'GP',
    'dnn_bnn_full': 'BNN-Full',
    'dnn_bnn_full_variational': 'VBLL-Full',
    'mlp_bnn_full': 'MLP-BNN-Full',
    'mlp_bnn_full_variational': 'MLP-VBLL-Full',
}
REP_LABELS = {
    'ecfp4': 'ECFP4',
    'continuous_pdv': 'PDV',
    'sns': 'SNS',
    'mhggnn': 'MHG-GNN-pretrained',
    'avalon': 'Avalon',
    'chemberta': 'ChemBERTa',
}


def coverage(y_true, y_pred, uncertainty, k=1):
    """Empirical coverage of the k-sigma interval.

    The same definition as `calculate_coverage` in `generate_paper_figures_v2.py`,
    computed against the label the model was TRAINED on -- which on a corrupted
    training row is the corrupted one. A model cannot be asked to cover a label
    it was never shown.
    """
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(uncertainty)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[ok] - y_pred[ok]) <= k * uncertainty[ok]))


def coverage_table(df, split='train_oof'):
    """Coverage per (model, rep, level), on the noisy label the model saw."""
    rows = []
    keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
    sub = df[df['split'] == split] if split else df
    for key, cell in sub.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        y_noisy = (cell['y_true_clean'].to_numpy(dtype=float)
                   + cell['injected_noise'].to_numpy(dtype=float))
        y_pred = cell['y_pred'].to_numpy(dtype=float)
        unc = cell['uncertainty'].to_numpy(dtype=float)
        rec['n'] = len(cell)
        rec['coverage_1sigma'] = coverage(y_noisy, y_pred, unc, k=1)
        rec['coverage_2sigma'] = coverage(y_noisy, y_pred, unc, k=2)
        rec['mean_uncertainty'] = float(np.nanmean(unc))
        rows.append(rec)
    return pd.DataFrame(rows)


def r2_from_results(run_dir):
    """Held-out R2 per (model, rep, level), read from the pipeline's own result
    rows rather than recomputed, so the accuracy column and the paper's accuracy
    numbers come from one place."""
    frames = []
    for path in sorted(Path(run_dir).glob('*.csv')):
        if path.name.endswith('_uncertainty_values.csv'):
            continue
        if path.name.endswith('_noise_manifest.csv'):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not read {path.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    res = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in res.columns}
    model_c = cols.get('model')
    rep_c = cols.get('representation') or cols.get('rep')
    sig_c = cols.get('sigma') or cols.get('noise_level')
    r2_c = cols.get('r2') or cols.get('r_squared') or cols.get('test_r2')
    if not all([model_c, rep_c, sig_c, r2_c]):
        print(f"  ! result files carry {sorted(res.columns)}; no R2 column found")
        return pd.DataFrame()
    out = res[[model_c, rep_c, sig_c, r2_c]].copy()
    out.columns = ['model', 'rep', 'sigma', 'r2']
    # One row per cell. The screen runs a single replicate, so this collapses
    # nothing in practice; if a cell were ever run twice the median across
    # REPLICATES is the aggregation the project allows, and n_runs says how many
    # went into it rather than hiding the repeat.
    out = (out.groupby(['model', 'rep', 'sigma'], as_index=False)
              .agg(r2=('r2', 'median'), n_runs=('r2', 'size')))
    return out


def build(run_dir, out_dir, report_level=REPORT_LEVEL):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(run_dir.glob('*_uncertainty_values.csv'))
    if not paths:
        raise SystemExit(f"no *_uncertainty_values.csv under {run_dir}")
    print(f"reading {len(paths)} uncertainty files from {run_dir}")
    df = us.load_uncertainty(paths)
    print(f"  {len(df):,} rows | models {sorted(df['model'].unique())}")
    print(f"  reps {sorted(df['rep'].unique())} | levels {sorted(df['sigma'].unique())}")
    print(f"  conditions {sorted(df['condition'].unique())}")

    # Every statistic below is out of fold, on training molecules, inside one
    # noise level. assert_single_cell fires inside uncertainty_stats.
    plain = us.q4_plain_correlation(df, split='train_oof')
    ratio = us.q4_error_ratio(df, split='train_oof')
    rank = us.q6_error_ranking(df, split='train_oof')
    cov = coverage_table(df, split='train_oof')
    r2 = r2_from_results(run_dir)

    keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
    tab = (ratio[keys + ['n', 'rho_error', 'rho_ratio', 'rho_delta',
                         'auc_error', 'auc_ratio', 'auc_delta']]
           .merge(plain[keys + ['rho_raw']], on=keys, how='outer')
           .merge(rank[keys + ['rho_unc_vs_clean_error', 'mean_abs_error_clean']],
                  on=keys, how='outer')
           .merge(cov[keys + ['coverage_1sigma', 'coverage_2sigma',
                              'mean_uncertainty']], on=keys, how='outer'))
    if not r2.empty:
        tab = tab.merge(r2, on=['model', 'rep', 'sigma'], how='left')

    tab['model_label'] = tab['model'].map(lambda m: MODEL_LABELS.get(m, m))
    tab['rep_label'] = tab['rep'].map(lambda r: REP_LABELS.get(r, r))
    full = out_dir / 'uncertainty_screen_full.csv'
    tab.sort_values(['rep', 'model', 'sigma']).to_csv(full, index=False)
    print(f"\nwrote {full}")

    # ---- one table per representation, level by level, nothing pooled -------
    shown = ['model_label', 'n', 'r2', 'rho_raw', 'rho_error', 'rho_ratio',
             'rho_delta', 'auc_error', 'auc_ratio', 'auc_delta',
             'rho_unc_vs_clean_error', 'coverage_1sigma', 'coverage_2sigma',
             'mean_uncertainty']
    for rep, part in tab.groupby('rep'):
        label = REP_LABELS.get(rep, rep)
        at = part[np.isclose(part['sigma'], report_level)]
        control = part[np.isclose(part['sigma'], CONTROL_LEVEL)]
        path = out_dir / f'uncertainty_screen_{rep}.csv'
        part.sort_values(['sigma', 'model']).to_csv(path, index=False)
        print(f"\n=== {label} — QM9, Gaussian noise at level {report_level} "
              f"(fraction of the clean training label spread), "
              f"training molecules scored out of fold ===")
        cols = [c for c in shown if c in at.columns]
        print(at.sort_values('rho_delta', ascending=False)[cols]
              .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        if not control.empty:
            ccols = [c for c in ['model_label', 'r2', 'rho_unc_vs_clean_error',
                                 'coverage_1sigma', 'coverage_2sigma']
                     if c in control.columns]
            print(f"  -- the same cells at level 0 (no noise injected: the noise "
                  f"columns are undefined there, these are not)")
            print(control.sort_values('model_label')[ccols]
                  .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nper-representation tables in {out_dir}")
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', default='/Volumes/seagate/chatN_screen')
    ap.add_argument('--out-dir', default='results/uncertainty_screen')
    ap.add_argument('--report-level', type=float, default=REPORT_LEVEL)
    args = ap.parse_args()
    build(args.run_dir, args.out_dir, args.report_level)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
