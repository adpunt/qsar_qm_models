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
    # The pipeline writes the DNN Bayesian variants without the 'dnn_' prefix.
    'bnn_full': 'BNN-Full',
    'bnn_full_variational': 'VBLL-Full',
    'dnn_bnn_full': 'BNN-Full',
    'dnn_bnn_full_variational': 'VBLL-Full',
    'mlp_bnn_full': 'MLP-BNN-Full',
    'mlp_bnn_full_variational': 'MLP-VBLL-Full',
}
REP_LABELS = {
    'ecfp4': 'ECFP4',
    'pdv': 'PDV',
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


def _spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.unique(a[ok]).size < 2 or np.unique(b[ok]).size < 2:
        return np.nan
    from scipy.stats import rankdata
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_cis(df, split='train_oof', n_boot=300, seed=20260827):
    """A spread for each cell's two deciding statistics.

    The screen runs ONE replicate, so there is no replicate spread to quote and
    the cells cannot be separated by eye. Resampling the molecules inside a cell
    gives the interval that says whether one model is really ahead of another,
    which is the difference between a decision and a ranking of noise. It is a
    statement about these molecules, not a replicate spread, and is labelled so.
    """
    rng = np.random.default_rng(seed)
    keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
    sub = df[df['split'] == split] if split else df
    rows = []
    for key, cell in sub.groupby(keys, dropna=False):
        rec = dict(zip(keys, key))
        eps = cell['injected_noise'].to_numpy(dtype=float)
        y = cell['y_true_clean'].to_numpy(dtype=float)
        p = cell['y_pred'].to_numpy(dtype=float)
        u = cell['uncertainty'].to_numpy(dtype=float)
        size = np.abs(eps)
        err = np.abs(y + eps - p)
        clean_err = np.abs(y - p)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.isfinite(u) & (u > 0), err / u, np.nan)
        n = len(cell)
        deltas, ranks = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            deltas.append(_spearman(ratio[idx], size[idx])
                          - _spearman(err[idx], size[idx]))
            ranks.append(_spearman(u[idx], clean_err[idx]))
        for name, vals in (('rho_delta', deltas), ('rho_unc_vs_clean_error', ranks)):
            arr = np.asarray(vals, dtype=float)
            if np.isfinite(arr).sum() < 10:
                rec[f'{name}_lo'] = rec[f'{name}_hi'] = np.nan
                continue
            rec[f'{name}_lo'] = float(np.nanpercentile(arr, 2.5))
            rec[f'{name}_hi'] = float(np.nanpercentile(arr, 97.5))
        rec['n_boot'] = n_boot
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


def build(run_dir, out_dir, report_level=REPORT_LEVEL, n_boot=300):
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(run_dir.glob('*_uncertainty_values.csv'))
    if not paths:
        raise SystemExit(f"no *_uncertainty_values.csv under {run_dir}")
    print(f"reading {len(paths)} uncertainty files from {run_dir}")
    df = us.load_uncertainty(paths)

    # A molecule in an out-of-fold part that was not run is written as NaN, so
    # the file says which molecules carry a score (process_and_train.py
    # --oof-folds-scored). Drop those rows here rather than carrying them into
    # the counts: every statistic already masks non-finite values, but `n` would
    # otherwise report molecules nobody scored.
    unscored = (df['split'].eq('train_oof')
                & (~np.isfinite(df['y_pred']) | ~np.isfinite(df['uncertainty'])))
    if unscored.any():
        print(f"  dropping {int(unscored.sum()):,} training rows with no "
              f"out-of-fold score (a fold that was not run, or one that failed)")
        df = df[~unscored].copy()
    print(f"  {len(df):,} rows | models {sorted(df['model'].unique())}")
    print(f"  reps {sorted(df['rep'].unique())} | levels {sorted(df['sigma'].unique())}")
    print(f"  conditions {sorted(df['condition'].unique())}")

    # Every statistic below is out of fold, on training molecules, inside one
    # noise level. assert_single_cell fires inside uncertainty_stats.
    plain = us.q4_plain_correlation(df, split='train_oof')
    ratio = us.q4_error_ratio(df, split='train_oof')
    rank = us.q6_error_ranking(df, split='train_oof')
    cov = coverage_table(df, split='train_oof')
    boot = bootstrap_cis(df, split='train_oof', n_boot=n_boot)
    r2 = r2_from_results(run_dir)

    keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
    tab = (ratio[keys + ['n', 'rho_error', 'rho_ratio', 'rho_delta',
                         'auc_error', 'auc_ratio', 'auc_delta']]
           .merge(plain[keys + ['rho_raw']], on=keys, how='outer')
           .merge(rank[keys + ['rho_unc_vs_clean_error', 'mean_abs_error_clean']],
                  on=keys, how='outer')
           .merge(cov[keys + ['coverage_1sigma', 'coverage_2sigma',
                              'mean_uncertainty']], on=keys, how='outer')
           .merge(boot[keys + ['rho_delta_lo', 'rho_delta_hi',
                               'rho_unc_vs_clean_error_lo',
                               'rho_unc_vs_clean_error_hi', 'n_boot']],
                  on=keys, how='outer'))
    if not r2.empty:
        tab = tab.merge(r2, on=['model', 'rep', 'sigma'], how='left')

    tab['model_label'] = tab['model'].map(lambda m: MODEL_LABELS.get(m, m))
    tab['rep_label'] = tab['rep'].map(lambda r: REP_LABELS.get(r, r))
    full = out_dir / 'uncertainty_screen_full.csv'
    tab.sort_values(['rep', 'model', 'sigma']).to_csv(full, index=False)
    print(f"\nwrote {full}")

    # ---- one table per representation, level by level, nothing pooled -------
    shown = ['model_label', 'n', 'r2', 'rho_raw', 'rho_error', 'rho_ratio',
             'rho_delta', 'rho_delta_lo', 'rho_delta_hi',
             'auc_error', 'auc_ratio', 'auc_delta',
             'rho_unc_vs_clean_error', 'rho_unc_vs_clean_error_lo',
             'rho_unc_vs_clean_error_hi', 'coverage_1sigma', 'coverage_2sigma',
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
        if at.empty:
            print(f"  (no rows at level {report_level} yet)")
        else:
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
    decide(tab, out_dir, report_level)
    return tab


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------
# Written into RERUN_PLAN.md chat N before the first number came back, and
# executed here so the two lists follow from the table rather than from a
# reading of it. Change the rule in both places or in neither.

INCUMBENT_REPS = ['ecfp4', 'pdv', 'sns', 'mhggnn']
N_REPS_REQUIRED = 3          # of six, for a model to stay
RANK_FLOOR = 0.2             # rho_unc_vs_clean_error a model must reach
COVERAGE_FLOOR, COVERAGE_CEILING = 0.02, 0.99
DATASETS_IN_THE_RUN = 3
CONDITIONS_IN_THE_RUN = 5


def _clears_zero(lo, hi):
    return np.isfinite(lo) and np.isfinite(hi) and lo > 0


def decide(tab, out_dir, report_level=REPORT_LEVEL):
    at = tab[np.isclose(tab['sigma'], report_level)].copy()
    if at.empty:
        print(f"\nno rows at level {report_level}: the decision needs the noised "
              f"half of the run, and it has not landed yet.")
        return None

    at['adds_over_error'] = [
        _clears_zero(lo, hi) for lo, hi in zip(at['rho_delta_lo'], at['rho_delta_hi'])]
    at['ranks_the_error'] = [
        (r >= RANK_FLOOR) and _clears_zero(lo, hi)
        for r, lo, hi in zip(at['rho_unc_vs_clean_error'],
                             at['rho_unc_vs_clean_error_lo'],
                             at['rho_unc_vs_clean_error_hi'])]
    at['coverage_usable'] = at['coverage_1sigma'].between(COVERAGE_FLOOR,
                                                          COVERAGE_CEILING)
    at['cell_earns_its_place'] = ((at['adds_over_error'] | at['ranks_the_error'])
                                  & at['coverage_usable'])

    per_model = (at.groupby(['model', 'model_label'], as_index=False)
                   .agg(reps_scored=('rep', 'nunique'),
                        reps_earning=('cell_earns_its_place', 'sum'),
                        best_rank=('rho_unc_vs_clean_error', 'max'),
                        best_delta=('rho_delta', 'max'),
                        worst_coverage=('coverage_1sigma', 'min'),
                        best_coverage=('coverage_1sigma', 'max')))
    per_model['keep'] = per_model['reps_earning'] >= N_REPS_REQUIRED

    per_rep = (at.groupby(['rep', 'rep_label'], as_index=False)
                 .agg(models_scored=('model', 'nunique'),
                      best_rank=('rho_unc_vs_clean_error', 'max'),
                      best_delta=('rho_delta', 'max'),
                      best_r2=('r2', 'max')))
    incumbent = per_rep[per_rep['rep'].isin(INCUMBENT_REPS)]
    floor = float(incumbent['best_rank'].min()) if not incumbent.empty else np.nan
    per_rep['keep'] = (per_rep['best_rank'] >= floor) & (per_rep['best_r2'] > 0)

    print("\n" + "=" * 78)
    print(f"THE DECISION — QM9, Gaussian at level {report_level}, out of fold")
    print("=" * 78)
    print(f"\nMODELS — kept if {N_REPS_REQUIRED} of 6 representations earn their "
          f"place: the uncertainty adds over the error alone, or ranks the clean "
          f"error at {RANK_FLOOR}+, with coverage neither 0 nor 1")
    print(per_model[['model_label', 'reps_scored', 'reps_earning', 'best_delta',
                     'best_rank', 'worst_coverage', 'best_coverage', 'keep']]
          .sort_values('reps_earning', ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nREPRESENTATIONS — kept if the best model on them reaches the "
          f"weakest of the four already in the run ({floor:.3f})")
    print(per_rep[['rep_label', 'models_scored', 'best_rank', 'best_delta',
                   'best_r2', 'keep']]
          .sort_values('best_rank', ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:7.3f}"))

    models_kept = per_model.loc[per_model['keep'], 'model_label'].tolist()
    reps_kept = per_rep.loc[per_rep['keep'], 'rep_label'].tolist()
    before = (len(per_model) * DATASETS_IN_THE_RUN * len(INCUMBENT_REPS)
              * CONDITIONS_IN_THE_RUN)
    after = (len(models_kept) * DATASETS_IN_THE_RUN * len(reps_kept)
             * CONDITIONS_IN_THE_RUN)
    print(f"\nMODELS  -> {models_kept}")
    print(f"REPS    -> {reps_kept}")
    print(f"JOBS    -> {before} before, {after} after "
          f"({len(models_kept)} models x {DATASETS_IN_THE_RUN} datasets x "
          f"{len(reps_kept)} reps x {CONDITIONS_IN_THE_RUN} conditions)")

    per_model.to_csv(out_dir / 'decision_models.csv', index=False)
    per_rep.to_csv(out_dir / 'decision_reps.csv', index=False)
    at.to_csv(out_dir / 'decision_cells.csv', index=False)
    return per_model, per_rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', default='/Volumes/seagate/chatN_screen')
    ap.add_argument('--out-dir', default='results/uncertainty_screen')
    ap.add_argument('--report-level', type=float, default=REPORT_LEVEL)
    ap.add_argument('--n-boot', type=int, default=300,
                    help='molecule resamples behind each interval (0 skips them)')
    args = ap.parse_args()
    build(args.run_dir, args.out_dir, args.report_level, args.n_boot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
