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
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402
import figlib_metrics as M  # noqa: E402
import figlib_shapes as S  # noqa: E402


#: The bottom axis of every level chart, worded once. Spelled short because
#: two of these side by side ran into each other at full length.
LEVEL_AXIS = 'Noise level (fraction of label spread)'

#: What each figure MEANS goes here, not in its title. A title carrying the
#: caveats makes a figure look crowded before a number is read, and the journal
#: expects the explanation in the caption anyway. write_captions() puts them in
#: one file beside the images, ready to paste.
CAPTIONS = {}


def caption(name, text):
    CAPTIONS[name] = ' '.join(text.split())


def write_captions(output_dir):
    if not CAPTIONS:
        return None
    path = Path(output_dir) / 'captions.md'
    lines = ['# Figure captions', '',
             'Generated with the figures. Everything here was kept OUT of the '
             'titles on purpose.', '']
    for name in sorted(CAPTIONS):
        lines += [f'**{name}.** {CAPTIONS[name]}', '']
    path.write_text('\n'.join(lines))
    print(f'    wrote captions.md ({len(CAPTIONS)} caption(s))')
    return path


#: Beyond this many columns across a row of grids, the printed numbers stop
#: being legible at 170 mm. Three assay datasets with a clean column and seven
#: conditions is 24, which is where "text overlay nightmare" comes from.
MAX_COLUMNS_ACROSS = 14


def panel_layout(n_panels, n_rows, n_columns_each, name):
    """Stacked, side by side, or one file each -- decided, not guessed.

    The old script stacks panels vertically every time, and says why in three
    separate comments: side by side at full width squeezes the grid and the
    legend. So stacked is the default. But nineteen models stacked three deep is
    over 450 mm and the journal allows 225, and a grid squeezed to 7 mm a column
    is unreadable however tall it is. So:

      stacked        if the stack fits the page
      side by side   if it does not, and the columns still get enough width
      one file each  otherwise -- an unreadable single figure is worth less than
                     three readable ones, and the journal caps neither

    Returns 'stacked', 'across' or 'split', having said which and why.
    """
    stacked = C.grid_height(n_rows, n_panels)
    if stacked <= C.MAX_HEIGHT_IN:
        return 'stacked'
    across = n_panels * n_columns_each
    if across <= MAX_COLUMNS_ACROSS:
        print(f'    {name}: {n_panels} stacked panels of {n_rows} rows is '
              f'{stacked * 25.4:.0f} mm, past the '
              f'{C.MAX_HEIGHT_IN * 25.4:.0f} mm the journal allows. Side by '
              f'side instead -- {across} columns still get '
              f'{170 / across:.0f} mm each.')
        return 'across'
    print(f'    {name}: {n_panels} stacked panels is {stacked * 25.4:.0f} mm '
          f'(limit {C.MAX_HEIGHT_IN * 25.4:.0f}) and side by side would give '
          f'{across} columns {170 / across:.0f} mm each. Writing one figure per '
          f'panel instead.')
    return 'split'


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
        S.title(ax, 'abcd'[index] if index < 4 else str(index), outcome)
    axes[-1].set_xlabel('Noise condition')

    n_reps = int(frame['n_replicates'].max()) if 'n_replicates' in frame else 0
    caption('F2', f"""
        How much of the variation in each outcome is explained by the choice of
        model, the choice of representation, the pairing of the two, and what is
        left over, on {C.dataset_label(dataset)}. Bars are the share of variance
        from a two-way analysis with sequential sums of squares; the four shares
        sum to 100 per cent. Whiskers span the {n_reps} replicates -- the
        decomposition is repeated on each one separately, which no previous
        version of this figure could do because the robustness metric was
        computed on a curve that had already been averaged over them.""")
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

    lo, hi = C.AUC_RANGE_QM9
    caption('F3', f"""
        Robustness ({G.metric_label(value)}) of every model on every
        representation, one panel per noise condition, on
        {C.dataset_label(dataset)}. Rows are models ordered by family, columns
        are representations. Which conditions appear is decided from the data:
        conditions whose grids repeat another's are held back to an additional
        file, because showing both is showing one thing twice. Colour is on one
        fixed range across panels. Grey cells were never run.""")

    layout = panel_layout(len(shown), len(models), len(reps), 'F3')
    if layout == 'stacked':
        fig, axes = _fig(height=C.grid_height(len(models), len(shown)),
                         nrows=len(shown), sharex=True)
    else:
        fig, axes = _fig(height=C.grid_height(len(models)), ncols=len(shown),
                         sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for index, (ax, condition) in enumerate(zip(axes, shown)):
        panel = frame[frame['condition'] == condition]
        image, _ = S.grid(ax, panel, 'model', 'rep', value,
                          column_labeller=C.rep_label, vmin=lo, vmax=hi,
                          row_order=models, column_order=reps)
        S.title(ax, 'abcdefg'[index], C.condition_label(condition))
    if image is not None:
        bar = fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
        bar.set_label(G.metric_label(value), fontsize=8)
        bar.ax.tick_params(labelsize=7)
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
    # Three FULL-WIDTH panels, stacked. Two line charts side by side ran their
    # axis labels into each other and left no room for a key.
    import matplotlib.pyplot as plt
    height = min(C.MAX_HEIGHT_IN, 3.4 + C.grid_height(len(order)))
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, height))
    spec = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.20 * len(order) + 1.0],
                            hspace=0.42)

    ax_a = fig.add_subplot(spec[0])
    S.line_chart(ax_a, curves, 'sigma', 'r2', 'model', spread='spread',
                 labeller=C.model_label, reference_x=level,
                 reference_label='reported at', legend=False)
    ax_a.set_ylabel(G.metric_label('r2'))
    S.title(ax_a, 'a', f'{C.condition_label(reference_condition)}, '
            f'{len(keep)} most robust models')

    ax_b = fig.add_subplot(spec[1], sharex=ax_a)
    S.line_chart(ax_b, by_condition, 'sigma', 'r2', 'condition',
                 spread='spread', labeller=C.condition_label,
                 colours=C.CONDITION_COLORS, reference_x=level, legend=False)
    ax_b.set_xlabel(LEVEL_AXIS)
    ax_b.set_ylabel(G.metric_label('r2'))
    S.title(ax_b, 'b', f'{C.model_label(focus_model)}, every condition')

    ax_c = fig.add_subplot(spec[2])
    baseline = (summ.groupby('model', as_index=False)['baseline_r2'].median()
                .assign(condition='clean')
                .rename(columns={'baseline_r2': 'auc_norm'}))
    both = pd.concat([baseline, summ[['model', 'condition', 'auc_norm']]],
                     ignore_index=True)
    S.grid(ax_c, both, 'model', 'condition', 'auc_norm',
           vmin=C.AUC_RANGE[0], vmax=C.AUC_RANGE[1],
           row_order=order, separate_first_column=True,
           column_order=['clean'] + C.sort_conditions(
               [c for c in summ['condition'].unique()]),
           column_labeller=lambda c: ('Clean R²' if c == 'clean'
                                      else C.condition_label(c)))
    S.title(ax_c, 'c', f'{G.metric_label("auc_norm")} by model and condition')

    # A KEY PER PANEL, inside it. Panel a's key is models and panel b's is noise
    # conditions -- different things that cannot share a legend, and merging
    # them produced one strip of eighteen entries that then sat on top of panel
    # c's column labels. Each panel's key is small, in two columns, in the
    # corner its own lines leave empty.
    # BESIDE the panel, not inside it. Lines that fall left to right leave no
    # corner free -- every placement inside the axes sat on data.
    for ax in (ax_a, ax_b):
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.02), ncol=1,
                  fontsize=6.5, frameon=False, handlelength=1.2,
                  handletextpad=0.4, labelspacing=0.35)

    caption('F4', f"""
        What label noise costs you, on {title}. (a) Accuracy against the amount
        of noise for the most robust models under one condition; the shaded band
        is the spread across replicates and the dashed line marks the level
        every table reports at. (b) The same for one model under every
        condition, which is the picture of whether the KIND of noise matters
        rather than only the amount. (c) Robustness by model and condition,
        models ordered by robustness under the reference condition. The first
        column is clean accuracy and is deliberately uncoloured: it is the
        quantity the rest are a fraction of. Censoring appears nowhere here --
        its level axis is a fraction of labels clipped rather than a fraction of
        the label spread, so it shares neither an axis nor a colour scale, and
        it runs on a named subset of pairs that cannot rank models.""")
    return S.save(fig, Path(output_dir) / 'F4_overview.png')


# ---------------------------------------------------------------------------
# F8 -- the assay datasets
# ---------------------------------------------------------------------------

def f8_assay(summary, output_dir, rep, value='auc_norm'):
    """One grid per assay dataset: models down the side, conditions across.

    THE PANELS SPLIT WHEN THEY WOULD NOT BE LEGIBLE. Three datasets, each with a
    clean column and seven conditions, is twenty-four columns across 170 mm --
    seven millimetres per printed number. Past MAX_COLUMNS_ACROSS this writes one
    figure per dataset instead, and says so, because an unreadable single figure
    is worth less than three readable ones and the journal caps neither.

    The clean-accuracy column is drawn UNCOLOURED. It is a different quantity
    from the retention beside it -- on the assay sets clean accuracy runs about
    0.45 to 0.61 while retention runs 0.67 to 0.94 -- so sharing one colour
    scale paints the whole reference column in the map's dark end and it reads
    as every model being terrible.
    """
    frame = summary[(summary['rep'] == rep) & (summary['dataset'] != 'qm9')]
    rankable = G.ranking_conditions(sorted(frame['condition'].dropna().unique()))
    frame = frame[frame['condition'].isin(rankable)]
    if not len(frame):
        return None
    title = G.declare(frame, 'F8', fixed={'rep': rep},
                      varies=('dataset', 'model', 'condition'))

    datasets = [d for d in C.DATASET_ORDER if d in set(frame['dataset'])]
    models = C.sort_models(frame['model'].unique())
    conditions = C.sort_conditions(frame['condition'].unique())
    lo, hi = C.AUC_RANGE_ASSAY

    caption('F8', f"""
        Robustness ({G.metric_label(value)}) of every model on the three assay
        datasets, on the {C.rep_label(rep)} representation. Rows are models,
        ordered by family. The first column of each panel is clean
        {G.metric_label('r2')} and is deliberately uncoloured: it is the
        quantity the others are a fraction of, not a measurement on the same
        scale. Colour is on one fixed range across all panels so they can be
        compared directly. Grey cells were never run. There are no error bars:
        one fit per cell with the seed pinned, and the five scaffold folds are a
        partition of one dataset rather than repeats of an experiment.
        Censoring is absent because it runs on a named subset of pairs and
        cannot rank models.""")

    # STACKED. Panels side by side at 170 mm gave each dataset 55 mm for eight
    # columns, which is the "text overlay nightmare". Stacked, each gets the
    # full width -- but three grids of nineteen models is 18 inches and the
    # journal allows 225 mm, so past that it becomes one figure per dataset.
    # An unreadable single figure is worth less than three readable ones and the
    # journal caps neither.
    layout = panel_layout(len(datasets), len(models), len(conditions) + 1, 'F8')
    written = []
    panels = [[d] for d in datasets] if layout == 'split' else [datasets]
    for group in panels:
        if layout == 'across':
            fig, axes = _fig(height=C.grid_height(len(models)),
                             ncols=len(group), sharey=True)
        else:
            fig, axes = _fig(height=C.grid_height(len(models), len(group)),
                             nrows=len(group), sharex=True)
        axes = np.atleast_1d(axes)
        image = None
        for index, (ax, dataset) in enumerate(zip(axes, group)):
            panel = frame[frame['dataset'] == dataset]
            baseline = (panel.groupby('model', as_index=False)['baseline_r2']
                        .median().assign(condition='clean')
                        .rename(columns={'baseline_r2': value}))
            both = pd.concat([baseline, panel[['model', 'condition', value]]],
                             ignore_index=True)
            image, _ = S.grid(ax, both, 'model', 'condition', value,
                              row_order=models, separate_first_column=True,
                              column_order=['clean'] + conditions,
                              column_labeller=lambda c: (
                                  'Clean R²' if c == 'clean'
                                  else C.condition_label(c)),
                              vmin=lo, vmax=hi)
            letter = 'abc'[datasets.index(dataset)]
            S.title(ax, letter, C.dataset_label(dataset))
        if image is not None:
            bar = fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
            bar.set_label(G.metric_label(value), fontsize=8)
            bar.ax.tick_params(labelsize=7)
        suffix = f'_{group[0]}' if layout == 'split' else ''
        written.append(S.save(
            fig, Path(output_dir) / f'F8_assay_datasets{suffix}.png'))
    return written[0] if written else None


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

    # Room for the chart AND a key of up to nineteen models beneath it.
    fig, ax = _fig(height=min(C.MAX_HEIGHT_IN, 4.6))
    start = (median[median['sigma'] == median['sigma'].min()]
             .sort_values('rank')['model'].tolist())
    # Ordered by rank on CLEAN labels, so each line starts where that model
    # starts. Ordering the legend by average rank was rejected on 2026-08-14.
    median['model'] = pd.Categorical(median['model'], categories=start,
                                     ordered=True)
    S.line_chart(ax, median.sort_values(['model', 'sigma']), 'sigma', 'rank',
                 'model', labeller=C.model_label, legend=False)
    ax.invert_yaxis()
    # A rank is a whole number. Matplotlib's default ticks put 2.5 and 7.5 on
    # the axis, which are not ranks anything can hold.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(LEVEL_AXIS)
    ax.set_ylabel('Rank (1 = most accurate)')
    S.title(ax, 'a', f'Rank against noise level — {title}')
    S.shared_legend(fig, ax, ncol=4)
    caption('R15', f"""
        Where each model ranks against the others as the noise rises, on
        {title}. Rank 1 is the most accurate. Ranks are taken within each
        replicate and then aggregated, because ranking an averaged score answers
        a different question. A line stops where that model falls below
        {G.metric_label('r2')} {gate:g} rather than plunging to last place, and
        a model that never cleared it is absent. The legend is ordered by rank
        on clean labels, so each line starts where that model starts. This is
        the only figure here that does not have to pick a single noise
        level.""")
    return S.save(fig, Path(output_dir)
                  / f'R15_rank_against_level_{rep}_{condition}.png')


# ---------------------------------------------------------------------------
# Contingent: retention against clean baseline (14.6 row 16, spec in 10b.5)
# ---------------------------------------------------------------------------

def r16_decoupling(summary, output_dir, rep, dataset='qm9'):
    """Is robustness decoupled from accuracy?

    Built the way `create_figure3` in the old script is: ONE full-width panel,
    a marker per model, points at alpha 0.7 and size 50 so overlaps are
    readable, the correlation in a white box so it survives being drawn over
    data, a y-range padded to the data so the flatness fills the panel, and the
    key BELOW the axes in four columns rather than covering the points.

    ⚠️ Part of any decoupling here is arithmetic: the robustness metric divides
    the clean baseline out by construction, so a model with a weak baseline
    scores well by having less to lose. That is why the baseline is an axis and
    not a footnote.
    """
    frame = summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)]
    if not len(frame):
        return None
    title = G.declare(frame, 'R16', fixed={'dataset': dataset, 'rep': rep},
                      varies=('model', 'condition'))

    fig, ax = _fig(height=C.TEXTWIDTH_IN * 0.85)
    for model in C.sort_models(frame['model'].unique()):
        one = frame[frame['model'] == model]
        ax.scatter(one['baseline_r2'], one['auc_norm'], s=50, alpha=0.7,
                   color=C.model_color(model), marker=C.model_marker(model),
                   label=C.model_label(model))

    # Padded to the data, so the spread fills the panel instead of being
    # squeezed under a 1.0 reference line that is not there.
    values = frame['auc_norm'].dropna()
    if len(values):
        pad = max((values.max() - values.min()) * 0.25, 0.02)
        ax.set_ylim(values.min() - pad, values.max() + pad)
    ax.margins(x=0.08)
    ax.set_xlabel(f'Clean {G.metric_label("r2")} (no noise added)')
    ax.set_ylabel(G.metric_label('auc_norm'))
    S.title(ax, 'a', f'Clean accuracy against robustness — {title}')
    ax.spines[['top', 'right']].set_visible(False)

    pair = frame[['baseline_r2', 'auc_norm']].dropna()
    if len(pair) >= 3:
        rho, pval = stats.spearmanr(pair['baseline_r2'], pair['auc_norm'])
        significance = 'n.s.' if pval >= 0.05 else f'p = {pval:.1e}'
        S.stat_box(ax, f'Spearman ρ = {rho:.2f} ({significance}), '
                       f'{len(pair)} cells')

    S.shared_legend(fig, ax, ncol=4)
    caption('R16', f"""
        Clean accuracy against robustness on {title}, one point per model and
        noise condition. Marker shape is the model; variants of one family share
        a colour on purpose, so the shape is what tells them apart and it
        survives greyscale printing. Read this beside the clean-accuracy axis
        rather than alone: the robustness metric divides the clean baseline out
        by construction, so a model with a weak baseline scores well by having
        less to lose, and part of any apparent decoupling is arithmetic rather
        than a finding.""")
    return S.save(fig, Path(output_dir) / f'R16_decoupling_{rep}.png')
