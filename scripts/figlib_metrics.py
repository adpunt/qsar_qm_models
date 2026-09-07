#!/usr/bin/env python
"""Every statistic the paper reports, computed at the granularity it claims.

THE ONE CHANGE THAT MATTERS
---------------------------
`robustness` integrates the retention curve PER REPLICATE and then aggregates.
The old `calculate_robustness` averaged the replicates first and integrated the
averaged curve (generate_paper_figures_v2.py:2603), so no robustness number in
the submitted paper has a spread, and one diverged replicate is baked into the
mean accuracy at that level.

Worse, the script computed BOTH: `run_robustness_anova` (:2871) and
`run_simple_effects_analysis` (:2982) build auc_norm per replicate while
`calculate_robustness` averages first -- two different statistics under one
name, which is failure mode 12 with a number attached rather than a caption.

ONE ANOVA, NOT THREE
--------------------
`two_way_eta2` is the Type-I sequential version. The old script also carried two
balanced hand-rolled sum-of-squares implementations whose residual can come out
NEGATIVE on an unbalanced design -- and every design here is unbalanced, because
`gauche` runs on fingerprints only and the Bayesian variants do not train on
every representation. Those two are not carried over.

The response is read AT THE REPORTING LEVEL, from `models/model_defaults.py`.
The old one took `sigma_value=0.3` as a default argument and the call site never
passed anything, so every published eta-squared is at 0.3 on a scale that no
longer exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402

CELL = ['dataset', 'model', 'rep', 'condition']


# ---------------------------------------------------------------------------
# AUC_norm
# ---------------------------------------------------------------------------

def retention_auc_norm(sigma, r2, baseline):
    """Normalised area under the retention curve R2(sigma)/R2(0), trapezoidal.

    Higher is more robust. This is the study's SOLE robustness metric -- the
    noise degradation slope and the Weibull fit are both retired. Say AUC_norm;
    describing it instead of naming it reads as reopening a settled question.
    """
    sigma = np.asarray(sigma, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if baseline is None or not np.isfinite(baseline) or baseline == 0:
        return np.nan
    span = float(sigma.max() - sigma.min())
    if span <= 0:
        return np.nan
    return float(trapezoid(r2 / baseline, sigma) / span)


def robustness(df, baseline_threshold=None, min_levels=3):
    """One AUC_norm per REPLICATE, never one per cell.

    Returns (per_replicate, excluded). `per_replicate` carries one row per
    (dataset, model, rep, condition, replicate) with its own auc_norm and its
    own clean baseline -- so the spread across replicates is available to every
    figure that needs an error bar, and the variance decomposition has a
    residual that is real within-cell variance.
    """
    threshold = (C.BASELINE_THRESHOLD if baseline_threshold is None
                 else baseline_threshold)
    keys = [c for c in CELL if c in df.columns] + ['replicate']
    rows, excluded = [], []

    for key, group in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        curve = (group.dropna(subset=['sigma', 'r2'])
                 .groupby('sigma', as_index=False)['r2'].mean()
                 .sort_values('sigma'))
        if len(curve) < min_levels:
            excluded.append(dict(record, reason='fewer than '
                                 f'{min_levels} levels',
                                 n_levels=int(len(curve))))
            continue
        clean = curve[curve['sigma'] == curve['sigma'].min()]
        if clean.empty or curve['sigma'].min() != 0:
            excluded.append(dict(record, reason='no clean level',
                                 n_levels=int(len(curve))))
            continue
        baseline = float(clean['r2'].iloc[0])
        if not np.isfinite(baseline) or baseline < threshold:
            excluded.append(dict(record, reason='clean R2 below the gate',
                                 baseline_r2=baseline,
                                 n_levels=int(len(curve))))
            continue
        auc = retention_auc_norm(curve['sigma'].to_numpy(),
                                 curve['r2'].to_numpy(), baseline)
        units = group.get('level_units')
        rows.append(dict(
            record,
            auc_norm=auc,
            baseline_r2=baseline,
            r2_at_max_level=float(curve['r2'].iloc[-1]),
            n_levels=int(len(curve)),
            level_max=float(curve['sigma'].max()),
            level_units=(units.dropna().iloc[0]
                         if units is not None and units.notna().any() else ''),
        ))

    per_replicate = pd.DataFrame(rows)
    if len(per_replicate):
        high = per_replicate['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH
        if high.any():
            print(f'  {int(high.sum())} of {len(per_replicate)} replicate '
                  f'AUC_norm values exceed {C.AUC_NORM_IMPLAUSIBLE_HIGH} -- a '
                  f'model scoring better with noise added. Counted, not '
                  f'patched (RERUN_PLAN.md 7.3, 14.6 row 10).')
    return per_replicate, pd.DataFrame(excluded)


def summarise_robustness(per_replicate):
    """Median across replicates, with the spread that the median hides.

    The median is the summary; the spread is the error bar. Both come from the
    per-replicate values, which is the whole reason they are computed that way.
    """
    if per_replicate is None or not len(per_replicate):
        return pd.DataFrame()
    keys = [c for c in CELL if c in per_replicate.columns]
    out = per_replicate.groupby(keys, dropna=False).agg(
        auc_norm=('auc_norm', 'median'),
        auc_norm_lo=('auc_norm', lambda s: s.quantile(0.25)),
        auc_norm_hi=('auc_norm', lambda s: s.quantile(0.75)),
        auc_norm_spread=('auc_norm', lambda s: float(s.max() - s.min())),
        baseline_r2=('baseline_r2', 'median'),
        n_replicates=('auc_norm', 'size'),
    ).reset_index()
    return out


def accuracy_at_reporting_level(df, dataset=None):
    """Accuracy at the ONE level a table quotes, per replicate.

    `reporting_level` RAISES for a dataset whose level is unset rather than
    returning a default, because every previous default silently became the
    answer.
    """
    frames = []
    datasets = ([dataset] if dataset is not None
                else sorted(df['dataset'].dropna().unique()))
    for name in datasets:
        level = C.reporting_level(name)
        sub = df[df['dataset'] == name]
        at = sub[np.isclose(sub['sigma'].astype(float), float(level))]
        if len(at) == 0:
            print(f'  WARNING: {name} has no rows at its reporting level '
                  f'{level:g}; accuracy cannot be quoted for it yet.')
            continue
        frames.append(at.assign(reporting_level=level))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# The variance decomposition
# ---------------------------------------------------------------------------

def two_way_eta2(df, response, f1='model', f2='rep'):
    """Type-I sequential (model -> representation -> their pairing) eta-squared.

    Computed from nested least-squares fits, so the residual is pure within-cell
    variance, is always at least zero, and the four shares sum to 100. Every
    design in this study is unbalanced -- the Tanimoto process runs on
    fingerprints only -- and the balanced hand-rolled formula the old script
    also carried returns a negative residual on exactly those designs.
    """
    d = df.dropna(subset=[response, f1, f2])
    y = d[response].to_numpy(dtype=float)
    n = len(y)
    if n < 4:
        return None
    total_ss = float(((y - y.mean()) ** 2).sum())
    if total_ss == 0:
        return None

    def rss(*factors):
        X = np.ones((n, 1))
        for f in factors:
            dm = pd.get_dummies(f, drop_first=True).to_numpy(dtype=float)
            if dm.shape[1]:
                X = np.hstack([X, dm])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(((y - X @ beta) ** 2).sum())

    a = d[f1].astype(str)
    b = d[f2].astype(str)
    cell = a.str.cat(b, sep='|')
    rss_a = rss(a)
    rss_ab = rss(a, b)
    rss_full = rss(cell)
    return {
        'eta2_model': (total_ss - rss_a) / total_ss * 100,
        'eta2_rep': (rss_a - rss_ab) / total_ss * 100,
        'eta2_interaction': (rss_ab - rss_full) / total_ss * 100,
        'eta2_residual': rss_full / total_ss * 100,
        'n_models': int(d[f1].nunique()),
        'n_reps': int(d[f2].nunique()),
        'n': int(n),
    }


def two_way_eta2_by_condition(df, response, min_cell=None, where='the ANOVA'):
    """One decomposition per noise condition, with the replicate spread.

    The spread comes from repeating the decomposition on each replicate
    separately, which is the column table T3 asks for and no version of this
    figure has ever carried.
    """
    min_cell = C.MIN_CELL_ITERS if min_cell is None else min_cell
    rows = []
    for condition, group in df.groupby('condition', dropna=False):
        G.assert_replicates(group, ['model', 'rep'], min_n=min_cell,
                            where=f'{where}, condition {condition}')
        overall = two_way_eta2(group, response)
        if overall is None:
            continue
        per_replicate = []
        for _, one in group.groupby('replicate', dropna=False):
            got = two_way_eta2(one, response)
            if got is not None:
                per_replicate.append(got)
        record = dict(condition=condition, response=response, **overall)
        for share in ('eta2_model', 'eta2_rep', 'eta2_interaction',
                      'eta2_residual'):
            values = [p[share] for p in per_replicate]
            record[f'{share}_spread'] = (float(np.max(values) - np.min(values))
                                         if len(values) > 1 else np.nan)
        record['n_replicates'] = len(per_replicate)
        rows.append(record)
    return pd.DataFrame(rows)


def simple_effects(df, response, group_col, factor_col):
    """One-way eta-squared for `factor_col` within each level of `group_col`.

    When the interaction term is large the main effects are misleading, and this
    answers the question they cannot: how much does the model matter AT each
    representation.
    """
    results = []
    for level, group in df.groupby(group_col, dropna=False):
        groups = [g[response].to_numpy() for _, g in group.groupby(factor_col)
                  if len(g) >= 2]
        if len(groups) < 2:
            continue
        values = group[response]
        grand = values.mean()
        total_ss = float(((values - grand) ** 2).sum())
        if total_ss == 0:
            continue
        means = group.groupby(factor_col)[response].mean()
        counts = group.groupby(factor_col).size()
        ss_factor = float((counts * (means - grand) ** 2).sum())
        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_value = np.nan, np.nan
        results.append({group_col: level, 'eta2': ss_factor / total_ss * 100,
                        'f_stat': f_stat, 'p_value': p_value,
                        'n': int(len(group)), 'n_levels': len(groups)})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Rank agreement and paired tests
# ---------------------------------------------------------------------------

def kendalls_w(auc_frame, rep, conditions=None, where="Kendall's W"):
    """Do model rankings agree across noise conditions, WITHIN one representation.

    The old computation had no representation filter, so a model's value at a
    condition was its mean across every representation it happened to run on --
    and a model present on six was ranked against one present on two, on
    differently-constituted means. "Rankings are stable across noise types" then
    became a claim about an average that describes no representation.
    """
    frame = auc_frame[auc_frame['rep'] == rep]
    G.declare(frame, where, fixed={'rep': rep},
              varies=('model', 'condition', 'dataset'))
    usable = G.ranking_conditions(
        conditions if conditions is not None
        else sorted(frame['condition'].dropna().unique()))
    frame = frame[frame['condition'].isin(usable)]

    table = frame.pivot_table(index='condition', columns='model',
                              values='auc_norm', aggfunc='median')
    table = table.dropna(axis=1, how='any')
    n_raters, n_items = table.shape
    if n_raters < 2 or n_items < 3:
        return {'kendall_w': np.nan, 'p_value': np.nan, 'n_models': n_items,
                'n_conditions': n_raters, 'rep': rep, 'models': list(table.columns)}
    ranks = table.rank(axis=1, ascending=False).to_numpy()
    rank_sums = ranks.sum(axis=0)
    ss_between = float(((rank_sums - rank_sums.mean()) ** 2).sum())
    max_ss = (n_raters ** 2) * (n_items ** 3 - n_items) / 12
    w = ss_between / max_ss if max_ss > 0 else np.nan
    chi2 = n_raters * (n_items - 1) * w
    p = float(1 - stats.chi2.cdf(chi2, n_items - 1))
    return {'kendall_w': float(w), 'p_value': p, 'n_models': n_items,
            'n_conditions': n_raters, 'rep': rep, 'models': list(table.columns)}


def wilcoxon_paired(frame, value, group_col, a, b, pair_on, where='a paired test'):
    """Signed-rank test comparing two levels of `group_col`, paired on `pair_on`.

    The old implementation ACCEPTED a representation and a condition and used
    neither, pairing across every representation and every condition at once to
    manufacture enough pairs. Filter first; pair on the replicate.

    Note the floor: a two-sided signed-rank test on five pairs cannot go below
    p = 0.0625, so on the assay datasets -- five folds, no replicates -- nothing
    can be significant however large the effect. That is arithmetic, and it
    belongs in the caption rather than being discovered later.
    """
    left = frame[frame[group_col] == a].set_index(pair_on)[value]
    right = frame[frame[group_col] == b].set_index(pair_on)[value]
    shared = left.index.intersection(right.index)
    x = left.loc[shared].astype(float)
    y = right.loc[shared].astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(len(x))
    record = {'a': a, 'b': b, 'n_pairs': n, 'paired_on': pair_on,
              'median_a': float(x.median()) if n else np.nan,
              'median_b': float(y.median()) if n else np.nan,
              'median_change': float((y - x).median()) if n else np.nan,
              'statistic': np.nan, 'p_value': np.nan, 'significant': False,
              'p_floor': np.nan}
    if n < 5:
        record['note'] = (f'{n} pairs: a two-sided signed-rank test needs at '
                          f'least 5 and cannot reach 0.05 below 6')
        return record
    record['p_floor'] = float(2 ** (1 - n))
    if np.allclose(x, y):
        record['note'] = 'identical values; the test is undefined'
        return record
    stat, p = stats.wilcoxon(x, y)
    record.update(statistic=float(stat), p_value=float(p),
                  significant=bool(p < 0.05))
    return record


def _one_row_per(series, frame, profile_over, compare, level, where):
    """A profile must have ONE value per key, or it is not a profile.

    This is failure mode 1 with a crash attached. A robustness summary carries
    one row per (dataset, model, rep, condition), so `set_index('model')` on a
    frame spanning seven conditions gives a model seven index entries; `.loc`
    then returns seven rows for one key and the two sides of the correlation
    come out different lengths. Before it crashed it would have been correlating
    a pooled mixture of conditions, which is the thing the averaging rules
    exist to stop.
    """
    if not series.index.has_duplicates:
        return series
    extra = sorted(set(frame.columns) - {profile_over, compare, 'auc_norm',
                                         'auc_norm_lo', 'auc_norm_hi',
                                         'auc_norm_spread', 'baseline_r2',
                                         'n_replicates'})
    counts = series.index.value_counts()
    worst = counts.index[0]
    raise GuardishError(
        f'{where}: {profile_over!r} is not unique within {compare}={level!r} '
        f'-- {worst!r} appears {int(counts.iloc[0])} times.\n'
        f'The frame still varies over {extra}, so this would correlate a '
        f'mixture of them and call it a profile.\n'
        f'Hold those fixed first: a rank agreement between two '
        f'{compare}s is computed WITHIN one noise condition, never across '
        f'several (RERUN_PLAN.md 14.2).')


class GuardishError(AssertionError):
    """A statistic asked for at a granularity the data does not have."""


def profile_spearman(frame, value, profile_over, compare, a, b,
                     where='a profile correlation'):
    """Rank correlation between two levels of `compare`, over a shared profile.

    `frame` must already be narrowed so that `profile_over` is unique within
    each level of `compare` -- one row per model, not one per model per
    condition. It raises rather than pooling.
    """
    left = frame[frame[compare] == a].set_index(profile_over)[value]
    right = frame[frame[compare] == b].set_index(profile_over)[value]
    left = _one_row_per(left, frame, profile_over, compare, a, where)
    right = _one_row_per(right, frame, profile_over, compare, b, where)
    shared = sorted(set(left.index) & set(right.index))
    if len(shared) < 3:
        return {'a': a, 'b': b, 'rho': np.nan, 'p_value': np.nan,
                'n': len(shared)}
    rho, p = stats.spearmanr(left.loc[shared].astype(float),
                             right.loc[shared].astype(float))
    return {'a': a, 'b': b, 'rho': float(rho), 'p_value': float(p),
            'n': len(shared)}


def icc_1_1(frame, value, subject, rater, a, b):
    """ICC(1,1) between two raters over shared subjects.

    Near 1.0 means the two rank configurations almost identically, which is the
    non-independence the ANOVA has to screen for before it runs.
    """
    left = frame[frame[rater] == a].set_index(subject)[value]
    right = frame[frame[rater] == b].set_index(subject)[value]
    left = _one_row_per(left, frame, subject, rater, a, 'ICC(1,1)')
    right = _one_row_per(right, frame, subject, rater, b, 'ICC(1,1)')
    shared = sorted(set(left.index) & set(right.index))
    if len(shared) < 3:
        return {'a': a, 'b': b, 'icc': np.nan, 'n': len(shared)}
    data = np.column_stack([left.loc[shared].astype(float),
                            right.loc[shared].astype(float)])
    n, k = data.shape
    row_means = data.mean(axis=1)
    grand = data.mean()
    bms = k * float(((row_means - grand) ** 2).sum()) / (n - 1)
    wms = float(((data - row_means.reshape(-1, 1)) ** 2).sum()) / (n * (k - 1))
    denominator = bms + (k - 1) * wms
    icc = (bms - wms) / denominator if denominator else np.nan
    return {'a': a, 'b': b, 'icc': float(icc), 'n': n}


def coverage_at_k(y_true, y_pred, uncertainty, k=1):
    """Fraction of molecules whose value lies inside k predicted deviations.

    Targets 0.68 at k=1 and 0.95 at k=2. Read off the RAW uncertainty: the
    calibration multiplier is refitted at each noise level, so calibrated
    coverage is nominal at every level by construction and says nothing about
    how the interval responds to noise.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(uncertainty)
    if not ok.any():
        return np.nan
    inside = np.abs(y_true[ok] - y_pred[ok]) <= k * uncertainty[ok]
    return float(inside.mean())
