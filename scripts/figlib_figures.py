#!/usr/bin/env python
"""The paper's figures. Six slots, four shapes, and every one declares what it
holds fixed before it draws.

RERUN_PLAN.md 14.5. Each builder takes the tidy frames, calls `G.declare` to
state what it holds fixed and what it varies -- which RAISES if the data still
carries a factor the figure has not accounted for -- and gets back the title
text, generated from the data rather than typed, so a caption cannot drift away
from the numbers behind it.

  F1  what each noise type does to the labels          (Methods)
  F2  model, representation, or their pairing          SHAPE D
  F3  WHICH model and WHICH representation             SHAPE C
  F4  what label noise costs you                       SHAPE A + SHAPE C
  F8  the assay datasets                               SHAPE C
  R1  rank against noise level (contingent, 14.6 r15)  SHAPE A
  R2  retention against clean baseline (contingent, r16)

F6 and F7 need the uncertainty runs and are built when those land.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402
import figlib_metrics as M  # noqa: E402
import figlib_shapes as S  # noqa: E402


#: The bottom axis of every level chart, worded once. Spelled short because
#: two of these side by side ran into each other at full length.
LEVEL_AXIS = 'Noise level (fraction of label spread)'


def _fig(width_fraction=1.0, height=3.2, nrows=1, ncols=1, **kwargs):
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols,
                        figsize=(C.TEXTWIDTH_IN * width_fraction, height),
                        **kwargs)


# ---------------------------------------------------------------------------
# F2 -- Q1: model, representation, or their pairing
# ---------------------------------------------------------------------------

FACTOR_COLUMNS = [('eta2_model', 'Model'), ('eta2_rep', 'Representation'),
                  ('eta2_interaction', 'Interaction'),
                  ('eta2_residual', 'Residual')]


def f2_variance_decomposition(anova, output_dir, dataset='qm9'):
    """Two panels stacked: how much of the variance each factor explains.

    Top panel is predictive accuracy, bottom is robustness. Bottom axis of both
    is the noise conditions; side axis is share of variance, 0 to 100 per cent.
    Four bars at each condition -- the model, the representation, their pairing,
    and the leftover. A thin whisker on each bar is how much that share moved
    across the replicates.

    NO version of this figure has ever carried that whisker, because the metric
    behind it was computed on an averaged curve and had no spread to show.
    """
    if anova is None or not len(anova):
        return None
    frame = anova[anova.get('dataset', dataset) == dataset] \
        if 'dataset' in anova.columns else anova
    outcomes = list(dict.fromkeys(frame['outcome']))
    if not outcomes:
        return None

    long = []
    for _, row in frame.iterrows():
        for column, label in FACTOR_COLUMNS:
            long.append({'condition': row['condition'], 'factor': label,
                         'outcome': row['outcome'],
                         'share': row.get(column, np.nan),
                         'spread': row.get(f'{column}_spread', np.nan)})
    long = pd.DataFrame(long)

    fig, axes = _fig(height=5.4, nrows=len(outcomes), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (ax, outcome) in enumerate(zip(axes, outcomes)):
        panel = long[long['outcome'] == outcome]
        S.grouped_bars(ax, panel, 'condition', 'factor', 'share',
                       spread='spread', labeller=str,
                       colours=C.ANOVA_FACTOR_COLORS,
                       legend=(index == 0))
        ax.set_ylabel('Share of variance (%)')
        ax.set_ylim(0, 100)
        S.panel_letter(ax, 'ab'[index] if index < 2 else str(index))
        ax.set_title(outcome, fontsize=9, loc='right', color='#444444')
    axes[-1].set_xlabel('Noise condition')

    n_reps = int(frame['n_replicates'].max()) if 'n_replicates' in frame else 0
    fig.suptitle(f'{C.dataset_label(dataset)} — whiskers span the '
                 f'{n_reps} replicates', fontsize=9, y=1.0)
    fig.tight_layout()
    return S.save(fig, Path(output_dir) / 'F2_variance_decomposition.png')


# ---------------------------------------------------------------------------
# F3 -- Q1 continued: WHICH model and WHICH representation
# ---------------------------------------------------------------------------

def f3_model_by_representation(summary, output_dir, conditions,
                               dataset='qm9', value='auc_norm'):
    """One grid per noise condition: models down the side, representations
    across the bottom, robustness printed on each square.

    Which conditions appear is not a taste. D2 keeps the conditions whose grids
    DIFFER from each other and sends the repeats to an additional file; if two
    conditions give the same grid, showing both is showing one thing twice.
    """
    frame = summary[summary['dataset'] == dataset]
    frame = frame[frame['condition'].isin(conditions)]
    if not len(frame):
        return None
    title = G.declare(frame, 'F3', fixed={'dataset': dataset},
                      varies=('model', 'rep', 'condition'))

    models = C.sort_models(frame['model'].unique())
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    shown = [c for c in C.sort_conditions(frame['condition'].unique())]

    height = min(C.MAX_HEIGHT_IN, 1.1 + 0.24 * len(models))
    fig, axes = _fig(height=height, ncols=len(shown), sharey=True)
    axes = np.atleast_1d(axes)
    lo = float(frame[value].quantile(0.02))
    hi = float(frame[value].quantile(0.98))
    image = None
    for index, (ax, condition) in enumerate(zip(axes, shown)):
        panel = frame[frame['condition'] == condition]
        image, _ = S.grid(ax, panel, 'model', 'rep', value,
                          column_labeller=C.rep_label, vmin=lo, vmax=hi,
                          row_order=models, column_order=reps)
        ax.set_title(C.condition_label(condition), fontsize=9)
        S.panel_letter(ax, 'abcdefg'[index])
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
        bar.set_label(G.metric_label(value), fontsize=9)
    fig.suptitle(title, fontsize=9, y=1.02)
    return S.save(fig, Path(output_dir) / 'F3_model_by_representation.png')


# ---------------------------------------------------------------------------
# F4 -- Q2 and Q3: what label noise costs you
# ---------------------------------------------------------------------------

def f4_overview(accuracy, summary, output_dir, rep, dataset='qm9',
                reference_condition='gaussian', focus_model=None,
                top_models=8):
    """Three panels.

    a) Accuracy against noise level, one line per model, at one noise condition.
    b) Accuracy against noise level, one line per noise CONDITION, for one
       model -- the picture of "does the kind of noise matter" that nothing else
       in the paper shows.
    c) A grid: models down the side, conditions across the bottom, robustness on
       each square, WITH THE CLEAN ACCURACY AS A SEPARATE FIRST COLUMN behind a
       white gutter. That column is what stops a retention fraction ever being
       printed without the thing it is a fraction of.

    Censoring is on none of them: its bottom axis is a fraction of labels
    clipped, not a fraction of the label spread, so it cannot share an axis or a
    colour scale. It also runs on five named pairs and cannot rank models at
    all, which figlib_guard refuses outright.
    """
    acc = accuracy[(accuracy['dataset'] == dataset) & (accuracy['rep'] == rep)]
    rankable = G.ranking_conditions(sorted(acc['condition'].dropna().unique()))
    acc = acc[acc['condition'].isin(rankable)]
    if not len(acc):
        return None

    summ = summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)]
    summ = summ[summ['condition'].isin(rankable)]

    order = (summ[summ['condition'] == reference_condition]
             .sort_values('auc_norm', ascending=False)['model'].tolist())
    order = order or C.sort_models(acc['model'].unique())
    focus_model = focus_model or (order[0] if order else None)
    keep = order[:top_models]

    curves = (acc[acc['model'].isin(keep)
                  & (acc['condition'] == reference_condition)]
              .groupby(['model', 'sigma'], as_index=False)
              .agg(r2=('r2', 'median'),
                   spread=('r2', lambda v: float(v.std(ddof=0)))))
    title = G.declare(
        acc[(acc['condition'] == reference_condition)], 'F4a',
        fixed={'dataset': dataset, 'rep': rep,
               'condition': reference_condition},
        varies=('model', 'sigma'), aggregates=('replicate',))

    by_condition = (acc[acc['model'] == focus_model]
                    .groupby(['condition', 'sigma'], as_index=False)
                    .agg(r2=('r2', 'median'),
                         spread=('r2', lambda v: float(v.std(ddof=0)))))

    level = C.reporting_level(dataset)
    height = min(C.MAX_HEIGHT_IN, 3.0 + 0.22 * len(order))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, height))
    # The top row needs room for a legend UNDER it: eight model names inside
    # the axes covered the curves entirely.
    spec = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.20 * len(order) + 0.6],
                            hspace=0.95, wspace=0.30)

    ax_a = fig.add_subplot(spec[0, 0])
    S.line_chart(ax_a, curves, 'sigma', 'r2', 'model', spread='spread',
                 labeller=C.model_label, reference_x=level,
                 reference_label='reported at', legend=False)
    ax_a.set_xlabel(LEVEL_AXIS, fontsize=8)
    ax_a.set_ylabel(G.metric_label('r2'))
    ax_a.set_title(f'{C.condition_label(reference_condition)}, '
                   f'{len(keep)} most robust models', fontsize=8,
                   color='#444444', pad=14)
    ax_a.legend(ncol=3, fontsize=6.5, loc='upper center',
                bbox_to_anchor=(0.5, -0.46), frameon=False,
                handlelength=1.2, columnspacing=1.0)
    S.panel_letter(ax_a, 'a')

    ax_b = fig.add_subplot(spec[0, 1])
    S.line_chart(ax_b, by_condition, 'sigma', 'r2', 'condition',
                 spread='spread', labeller=C.condition_label,
                 colours=C.CONDITION_COLORS, reference_x=level, legend=False)
    ax_b.set_xlabel(LEVEL_AXIS, fontsize=8)
    ax_b.set_ylabel(G.metric_label('r2'))
    ax_b.set_title(f'{C.model_label(focus_model)}, every condition',
                   fontsize=8, color='#444444', pad=14)
    ax_b.legend(ncol=2, fontsize=6.5, loc='upper center',
                bbox_to_anchor=(0.5, -0.46), frameon=False,
                handlelength=1.2, columnspacing=1.0)
    S.panel_letter(ax_b, 'b')

    ax_c = fig.add_subplot(spec[1, :])
    # The clean baseline becomes a column of its own, first, behind a gutter.
    baseline = (summ.groupby('model', as_index=False)['baseline_r2'].median()
                .assign(condition='clean R²')
                .rename(columns={'baseline_r2': 'auc_norm'}))
    both = pd.concat([baseline, summ[['model', 'condition', 'auc_norm']]],
                     ignore_index=True)
    S.grid(ax_c, both, 'model', 'condition', 'auc_norm',
           row_order=order, separate_first_column=True,
           column_order=['clean R²'] + C.sort_conditions(
               [c for c in summ['condition'].unique()]),
           column_labeller=lambda c: ('Clean R²' if c == 'clean R²'
                                      else C.condition_label(c)))
    ax_c.set_title(f'{G.metric_label("auc_norm")} by model and condition. The '
                   f'first column is what each retains a fraction OF.',
                   fontsize=8, color='#444444', pad=8)
    S.panel_letter(ax_c, 'c')

    fig.suptitle(title, fontsize=9, y=1.005)
    return S.save(fig, Path(output_dir) / 'F4_overview.png')


# ---------------------------------------------------------------------------
# F8 -- the assay datasets
# ---------------------------------------------------------------------------

def f8_assay(summary, output_dir, rep, value='auc_norm'):
    """One grid per assay dataset, on one shared colour scale so the three can
    be read against each other, each with its clean-accuracy column at the left.

    NO ERROR BARS, and the Methods has to say so: one fit per cell with the seed
    pinned, and the five scaffold folds are a partition of one dataset rather
    than repeats of an experiment (RERUN_PLAN.md 3.2b).
    """
    frame = summary[(summary['rep'] == rep)
                    & (summary['dataset'] != 'qm9')]
    rankable = G.ranking_conditions(sorted(frame['condition'].dropna().unique()))
    frame = frame[frame['condition'].isin(rankable)]
    if not len(frame):
        return None
    title = G.declare(frame, 'F8', fixed={'rep': rep},
                      varies=('dataset', 'model', 'condition'))

    datasets = [d for d in C.DATASET_ORDER if d in set(frame['dataset'])]
    models = C.sort_models(frame['model'].unique())
    conditions = C.sort_conditions(frame['condition'].unique())
    lo = float(frame[value].quantile(0.02))
    hi = float(frame[value].quantile(0.98))

    height = min(C.MAX_HEIGHT_IN, 1.2 + 0.22 * len(models))
    fig, axes = _fig(height=height, ncols=len(datasets), sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for index, (ax, dataset) in enumerate(zip(axes, datasets)):
        panel = frame[frame['dataset'] == dataset]
        baseline = (panel.groupby('model', as_index=False)['baseline_r2']
                    .median().assign(condition='clean R²')
                    .rename(columns={'baseline_r2': value}))
        both = pd.concat([baseline, panel[['model', 'condition', value]]],
                         ignore_index=True)
        image, _ = S.grid(ax, both, 'model', 'condition', value,
                          row_order=models, separate_first_column=True,
                          column_order=['clean R²'] + conditions,
                          column_labeller=lambda c: ('Clean R²'
                                                     if c == 'clean R²'
                                                     else C.condition_label(c)),
                          vmin=lo, vmax=hi)
        ax.set_title(C.dataset_label(dataset), fontsize=9)
        S.panel_letter(ax, 'abc'[index] if index < 3 else str(index))
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
        bar.set_label(G.metric_label(value), fontsize=9)
    fig.suptitle(f'{title} — one fit per cell, five scaffold folds; the folds '
                 f'are a partition, not repeats, so there is no error bar',
                 fontsize=8.5, y=1.02)
    return S.save(fig, Path(output_dir) / 'F8_assay_datasets.png')


# ---------------------------------------------------------------------------
# Contingent: rank against noise level (14.6 row 15, the author's spec in 5.4a)
# ---------------------------------------------------------------------------

def r15_rank_against_level(accuracy, output_dir, rep, condition,
                           dataset='qm9', baseline_gate=None):
    """Hold one noise type and one representation. Plot every model. The bottom
    axis is the noise level; the side axis is where that model ranks against the
    others. Each model is one line, and the lines cross as the noise rises.

    The author's spec, 2026-08-27. The one figure that does not have to pick a
    single noise level -- every other ranking table does.

    Three rules for a model that stops working, and they are the author's:
    a run that broke is re-run and is not a data point; a model that worked and
    then fell below the accuracy floor has its line CUT SHORT there rather than
    plunging to last place; a model that never worked is left out entirely.
    Ranks are taken within each replicate and then aggregated, because ranking
    an averaged score answers a different question.
    """
    gate = C.BASELINE_THRESHOLD if baseline_gate is None else baseline_gate
    frame = accuracy[(accuracy['dataset'] == dataset)
                     & (accuracy['rep'] == rep)
                     & (accuracy['condition'] == condition)]
    if not len(frame):
        return None
    title = G.declare(frame, 'R15',
                      fixed={'dataset': dataset, 'rep': rep,
                             'condition': condition},
                      varies=('model', 'sigma'), aggregates=('replicate',))

    clean = (frame[frame['sigma'] == frame['sigma'].min()]
             .groupby('model')['r2'].median())
    alive = clean[clean >= gate].index                     # never worked -> out
    frame = frame[frame['model'].isin(alive)]
    if not len(frame):
        return None

    ranked = []
    for (sigma, replicate), group in frame.groupby(['sigma', 'replicate']):
        order = group.set_index('model')['r2'].rank(ascending=False)
        for model, rank in order.items():
            ranked.append({'model': model, 'sigma': sigma,
                           'replicate': replicate, 'rank': rank})
    ranked = pd.DataFrame(ranked)
    median = (ranked.groupby(['model', 'sigma'], as_index=False)['rank']
              .median())
    accuracy_at = (frame.groupby(['model', 'sigma'], as_index=False)['r2']
                   .median())
    median = median.merge(accuracy_at, on=['model', 'sigma'])

    # Cut a line short where the model drops below the floor, rather than
    # letting it plunge to last place and read as a ranking result.
    cut = []
    for model, group in median.sort_values('sigma').groupby('model'):
        below = group[group['r2'] < gate]
        limit = below['sigma'].min() if len(below) else np.inf
        cut.append(group[group['sigma'] <= limit])
    median = pd.concat(cut, ignore_index=True)

    height = min(C.MAX_HEIGHT_IN, 3.4)
    fig, ax = _fig(height=height)
    start = (median[median['sigma'] == median['sigma'].min()]
             .sort_values('rank')['model'].tolist())
    # Ordered by rank on CLEAN labels, so each line starts where that model
    # starts. Ordering the legend by average rank was rejected on 2026-08-14.
    median['model'] = pd.Categorical(median['model'], categories=start,
                                     ordered=True)
    S.line_chart(ax, median.sort_values(['model', 'sigma']), 'sigma', 'rank',
                 'model', labeller=C.model_label, legend_ncol=3)
    ax.invert_yaxis()
    ax.set_xlabel(LEVEL_AXIS)
    ax.set_ylabel('Rank (1 = most accurate)')
    ax.set_title(f'{title} — a line stops where that model falls below '
                 f'{G.metric_label("r2")} {gate:g}', fontsize=8.5,
                 color='#444444')
    fig.tight_layout()
    return S.save(fig, Path(output_dir)
                  / f'R15_rank_against_level_{rep}_{condition}.png')


# ---------------------------------------------------------------------------
# Contingent: retention against clean baseline (14.6 row 16, spec in 10b.5)
# ---------------------------------------------------------------------------

def r16_decoupling(summary, output_dir, rep, dataset='qm9'):
    """Is robustness really decoupled from accuracy?

    a) Retention against clean accuracy, one point per model.
    b) Accuracy delivered at the reporting level against retention. The
       bottom-right quadrant is the flattered one -- retains well, delivers
       little.

    ⚠️ Part of this is ARITHMETIC, not a finding: the metric divides the clean
    baseline out by construction, so a model with a weak baseline scores well by
    having less to lose. That is why the baseline is an axis here rather than a
    footnote, and why guard 4 refuses to print the ratio without it.
    """
    frame = summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)]
    if not len(frame) or 'r2_at_reporting_level' not in frame.columns:
        frame = frame.assign(r2_at_reporting_level=np.nan)
    if not len(frame):
        return None
    title = G.declare(frame, 'R16', fixed={'dataset': dataset, 'rep': rep},
                      varies=('model', 'condition'))

    fig, axes = _fig(height=3.0, ncols=2)
    for ax, (x, y, xlabel, ylabel, letter) in zip(axes, [
        ('baseline_r2', 'auc_norm', 'Clean ' + G.metric_label('r2'),
         G.metric_label('auc_norm'), 'a'),
        ('auc_norm', 'r2_at_reporting_level', G.metric_label('auc_norm'),
         G.metric_label('r2') + ' at the reported level', 'b'),
    ]):
        for model, group in frame.groupby('model'):
            ax.scatter(group[x], group[y], s=22, color=C.model_color(model),
                       marker=C.model_marker(model), label=C.model_label(model))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        S.panel_letter(ax, letter)
    if frame['auc_norm'].notna().sum() > 2:
        rho = frame[['baseline_r2', 'auc_norm']].corr(method='spearman').iloc[0, 1]
        axes[0].set_title(f'Spearman ρ = {rho:.2f}', fontsize=8.5,
                          color='#444444')
    fig.suptitle(f'{title} — the metric divides the baseline out, so part of '
                 f'any decoupling is arithmetic', fontsize=8.5, y=1.02)
    fig.tight_layout()
    return S.save(fig, Path(output_dir) / f'R16_decoupling_{rep}.png')
