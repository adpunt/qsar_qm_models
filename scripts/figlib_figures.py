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
  F6  the aleatoric/epistemic decomposition            SHAPE A, one panel per model
  F7  does the uncertainty find the bad labels         SHAPE A (7A, 7B) or C (7C)
  F8  the assay datasets                               SHAPE C

The contingent ones. Each is drawn only when its decision fires, and the
decision is 14.6's row number:

  R6   representation profiles, F3 option 3B (row 6)   SHAPE A sideways
  R9   rank transfer promoted from T7 (row 9)          SHAPE B
  R10  where AUC_norm exceeds 1 (row 10)               scatter against baseline
  R15  rank against noise level (row 15)               SHAPE A
  R15b the same, holding a MODEL fixed (5.4a)          SHAPE A
  R16  retention against clean baseline (row 16)       SHAPE A

F6 and F7 need the per-molecule uncertainty rows and draw nothing without them.
F7 is ONE of three options and D7 picks which; drawing all three would hand the
choice back to the reader, which is what rows 1 to 3 exist to prevent.
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
# F1 -- METHODS: what each noise condition does to a label
# ---------------------------------------------------------------------------

def _methods_groups(n, n_groups=40, seed=7):
    """Scaffold-like groups, so the grouped conditions have something to act on."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=n)


def f1_noise_conditions(output_dir, level=0.5, censored_fraction=0.25,
                        n_samples=2000, replicates=10):
    """What each settled condition does to a label distribution, plus the panel
    that shows the doses are matched.

    DRAWN WITH THE REAL INJECTOR. The old version of this figure carried its own
    reimplementation of six conditions -- all six retired in August -- and fell
    through to zeros for every condition the study actually runs, so a Methods
    figure claiming to show the noise scheme would have shown panels of no noise
    at all. It also plotted thirteen names into a hard-coded six-panel grid and
    raised before any other figure was reached. Conditions come from
    noise_conditions.json and the panel count follows from it.

    The last panel is the one nothing in the paper shows today and the one the
    whole comparison rests on: **the amount of noise actually delivered by each
    condition**. If the conditions do not deliver the same amount, every
    difference between them is a difference in dose rather than in kind -- which
    is what the six retired strategies turned out to be.
    """
    try:
        from noiseInject import NoiseInjectorRegression, CONDITIONS
    except Exception as exc:
        print(f'  F1 not drawn: the Methods figure is drawn with the real '
              f'injector and noiseInject will not import here ({exc}). Drawing '
              f'it from a local reimplementation is what made this figure '
              f'describe a scheme the study had already replaced.')
        return None

    conditions = [c for c in C.SETTLED_CONDITIONS if c in CONDITIONS]
    if not conditions:
        print(f'  F1 not drawn: the installed injector knows none of '
              f'{C.SETTLED_CONDITIONS}')
        return None
    missing = [c for c in C.SETTLED_CONDITIONS if c not in CONDITIONS]
    if missing:
        print(f'  F1: settled conditions the installed injector does not know, '
              f'omitted: {missing}')

    rng = np.random.default_rng(42)
    clean = np.concatenate([
        rng.normal(-0.5, 0.30, n_samples // 3),
        rng.normal(0.2, 0.40, n_samples // 3),
        rng.normal(0.8, 0.25, n_samples // 3 + n_samples % 3)])
    groups = _methods_groups(len(clean))
    spread = float(np.std(clean))

    def inject(condition, seed):
        censoring = CONDITIONS[condition].get('strategy') == 'censoring'
        dose = censored_fraction if censoring else level * spread
        injector = NoiseInjectorRegression.from_condition(
            condition, random_state=seed, selection_state=seed + 1337)
        return injector.inject(clean, dose, groups=groups, reference=clean)

    drawn = {c: inject(c, 42) for c in conditions}

    # ONE bin grid for every panel. Recomputing bins per panel shifts the edges,
    # and the SAME clean labels then draw a visibly different outline in each
    # panel from identical data.
    lo = min([clean.min()] + [v.min() for v in drawn.values()])
    hi = max([clean.max()] + [v.max() for v in drawn.values()])
    bins = np.linspace(lo, hi, 51)

    # The delivered amount, over replicates, per condition.
    delivered = []
    for condition in conditions:
        for seed in range(replicates):
            noised = inject(condition, 1000 + seed)
            delivered.append({
                'condition': condition, 'seed': seed,
                'delivered': float(np.sqrt(np.mean((noised - clean) ** 2)))})
    delivered = pd.DataFrame(delivered)

    ncols = 2
    nrows = int(np.ceil(len(conditions) / ncols)) + 1      # +1 for the dose panel
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN,
                              min(C.MAX_HEIGHT_IN, 1.35 * nrows)))
    spec = fig.add_gridspec(nrows, ncols, hspace=0.95, wspace=0.18)

    for index, condition in enumerate(conditions):
        ax = fig.add_subplot(spec[index // ncols, index % ncols])
        noised = drawn[condition]
        colour = C.CONDITION_COLORS.get(condition, '#666666')
        ax.hist(clean, bins=bins, density=True, alpha=0.15, color=C.CLEAN_COLOR)
        ax.hist(noised, bins=bins, density=True, alpha=0.15, color=colour)
        ax.hist(clean, bins=bins, density=True, histtype='step', linewidth=1.6,
                color=C.CLEAN_COLOR, alpha=0.85)
        ax.hist(noised, bins=bins, density=True, histtype='step', linewidth=1.6,
                color=colour, alpha=0.85)
        S.title(ax, 'abcdefghij'[index], C.condition_label(condition))
        ax.set_yticks([])
        # No numbers on the side axis -- it is a density and the number means
        # nothing to a reader -- but the axis must still SAY what it is. Without
        # this the panels had an unlabelled vertical direction and nothing on
        # the figure said which of the two outlines was the clean labels.
        if index % ncols == 0:
            ax.set_ylabel('Share of\nlabels', fontsize=7)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        # Headroom so the annotation clears the bars.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
        amount = float(np.sqrt(np.mean((noised - clean) ** 2)))
        ax.text(0.97, 0.94, f'delivered {amount:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7, color='#333333')
        # Only the bottom histogram in each column, or the label lands on the
        # title of the panel underneath it.
        if index >= len(conditions) - ncols:
            ax.set_xlabel('Label value', fontsize=8)

    # The dose panel, across the full width.
    ax = fig.add_subplot(spec[nrows - 1, :])
    order = C.sort_conditions(conditions)
    for position, condition in enumerate(order):
        values = delivered[delivered['condition'] == condition]['delivered']
        colour = C.CONDITION_COLORS.get(condition, '#666666')
        ax.plot([position, position], [values.min(), values.max()],
                color=colour, linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter([position], [values.median()], s=50, color=colour, zorder=3)
    target = level * spread
    ax.axhline(target, color='#444444', linestyle='--', linewidth=0.9)
    ax.annotate(f'asked for {target:.2f}', xy=(1.0, target), xycoords=('axes fraction', 'data'),
                xytext=(3, 0), textcoords='offset points', fontsize=7,
                va='center', color='#444444', annotation_clip=False)
    # Censoring is NOT dose-matched and cannot be: it has no variance parameter
    # and is not zero-mean, so it runs on its own axis. Saying that on the panel
    # stops its lower point being read as a condition that failed to hit target.
    censoring = [i for i, c in enumerate(order) if c.startswith('censoring')]
    for position in censoring:
        ax.annotate('own axis,\nnot dose-matched', xy=(position, 0),
                    xycoords=('data', 'axes fraction'), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=6, style='italic', color='#555555')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([C.condition_label(c) for c in order], rotation=25,
                       ha='right', fontsize=8)
    ax.set_ylabel('Delivered noise', fontsize=8)
    S.title(ax, 'abcdefghij'[len(conditions)],
            f'Amount actually delivered, {replicates} draws each')
    ax.spines[['top', 'right']].set_visible(False)

    # WHICH OUTLINE IS WHICH. Every histogram panel draws the clean labels in
    # one colour and the noised ones in that condition's colour, and nothing on
    # the figure said so -- a reader had two outlines and no way to tell which
    # was the before and which the after.
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], color=C.CLEAN_COLOR, linewidth=1.6,
               label='Clean labels'),
        Line2D([0], [0], color='#666666', linewidth=1.6,
               label='After the noise (each panel in its own colour)')],
        loc='lower center', ncol=2, frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, -0.012))

    caption('F1', f"""
        What each settled noise condition does to a label distribution.
        Panels (a) onward: the clean labels and the noised labels over each
        other, same bins in every panel, at a level of {level:g} of the clean
        label spread — censoring at {censored_fraction:.0%} of labels clipped,
        because its level is a fraction clipped rather than a fraction of the
        spread. Drawn with the injector the pipeline runs, not a
        reimplementation. The final panel is the evidence that comparing
        conditions is fair at all: the amount of noise actually delivered by
        each, over {replicates} draws, against the amount the dose solver was
        asked for. Without it, a difference between two conditions could be a
        difference in dose rather than in kind — which is what the six
        strategies this scheme replaced turned out to be.""")
    return S.save(fig, Path(output_dir) / 'F1_noise_conditions.png')


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

    # ONE category list for both panels. A condition can be missing from one
    # outcome and not the other -- censoring has no accuracy at the reporting
    # level, because its level axis is a fraction of labels clipped -- and
    # letting each panel derive its own categories put a seven-group panel over
    # a six-group axis, so every bar in it was labelled as its neighbour.
    conditions = C.sort_conditions(long['condition'].unique())
    fig, axes = _fig(height=5.8, nrows=len(outcomes), sharex=True)
    axes = np.atleast_1d(axes)
    for index, (ax, outcome) in enumerate(zip(axes, outcomes)):
        panel = long[long['outcome'] == outcome]
        S.grouped_bars(ax, panel, 'condition', 'factor', 'share',
                       spread='spread', labeller=str,
                       colours=C.ANOVA_FACTOR_COLORS, legend=False,
                       categories=conditions, clip=(0, 100))
        ax.set_ylabel('Share of variance (%)')
        ax.set_ylim(0, 100)
        S.title(ax, 'abcd'[index] if index < 4 else str(index), outcome)
        # AN EMPTY SLOT SAYS WHY IT IS EMPTY. Both panels share one category
        # list so they line up, which means each one has gaps where the other
        # has bars -- censoring has no accuracy at a reported level, and the
        # three deep-run conditions have no robustness because their cells were
        # dropped. Four blank slots with nothing written in them read as four
        # zeroes.
        have = set(panel.dropna(subset=['share'])['condition'])
        for position, condition in enumerate(conditions):
            if condition in have:
                continue
            ax.annotate('no value\n(see the excluded table)',
                        xy=(position, 2), xycoords=('data', 'data'),
                        ha='center', va='bottom', fontsize=5.5,
                        style='italic', color='#777777', rotation=90)
    axes[-1].set_xlabel('Noise condition')
    S.shared_legend(fig, axes[0], ncol=4)

    n_reps = int(frame['n_replicates'].max()) if 'n_replicates' in frame else 0
    empty = sorted(set(conditions) - set(
        long[long['outcome'] == outcomes[-1]]['condition']))
    missing_note = (
        f' {", ".join(C.condition_label(c) for c in empty)} has no bar in the '
        f'lower panel: its level axis is a fraction of labels clipped rather '
        f'than a fraction of the label spread, so there is no accuracy at a '
        f'reported level to decompose.' if empty else '')
    caption('F2', f"""
        How much of the variation in each outcome is explained by the choice of
        model, the choice of representation, the pairing of the two, and what is
        left over, on {C.dataset_label(dataset)}. Bars are the share of variance
        from a two-way analysis with sequential sums of squares; the four shares
        sum to 100 per cent. Whiskers span the {n_reps} replicates -- the
        decomposition is repeated on each one separately, which no previous
        version of this figure could do, because the robustness metric was
        computed on a curve that had already been averaged over them.{missing_note}""")
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

def _excluded_keys(excluded, keys, **where):
    """The grid cells that RAN and were then dropped, keyed as the grid is.

    `robustness()` returns these with a reason: no clean level on the ladder, a
    clean accuracy under the gate, fewer than three levels to integrate. Every
    one of them ran. Printing "not run" on them states that the experiment was
    never done, which on the deep-run conditions is false -- those tasks
    finished and their accuracy is in the line panels of the same figure.
    """
    if excluded is None or not len(excluded):
        return set()
    frame = excluded
    for column, value in where.items():
        if column in frame.columns:
            frame = frame[frame[column] == value]
    if not len(frame) or any(k not in frame.columns for k in keys):
        return set()
    return set(map(tuple, frame[list(keys)].drop_duplicates().to_numpy()))


def f4_overview(accuracy, summary, output_dir, rep, dataset='qm9',
                reference_condition='gaussian', focus_model=None,
                top_models=8, excluded=None):
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
    dropped = _excluded_keys(excluded, ('model', 'condition'),
                             dataset=dataset, rep=rep)
    S.grid(ax_c, both, 'model', 'condition', 'auc_norm',
           vmin=C.AUC_RANGE[0], vmax=C.AUC_RANGE[1], excluded=dropped,
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
        quantity the rest are a fraction of. A grey cell marked "not run" was
        never fitted; one marked "excluded" ran and was then dropped, and the
        reason -- no clean level on its ladder, a clean accuracy under the gate,
        or fewer than three levels to integrate -- is in the excluded table. Censoring appears nowhere here --
        its level axis is a fraction of labels clipped rather than a fraction of
        the label spread, so it shares neither an axis nor a colour scale, and
        it runs on a named subset of pairs that cannot rank models.""")
    return S.save(fig, Path(output_dir) / 'F4_overview.png')


# ---------------------------------------------------------------------------
# F8 -- the assay datasets
# ---------------------------------------------------------------------------

def f8_assay(summary, output_dir, rep, value='auc_norm', excluded=None):
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
        compared directly. A grey cell marked "not run" is a combination that
        was never fitted; one marked "excluded" ran and was then dropped, with
        the reason in the excluded table. There are no error bars:
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
                              vmin=lo, vmax=hi,
                              excluded=_excluded_keys(
                                  excluded, ('model', 'condition'),
                                  dataset=dataset, rep=rep))
            # ONE FIGURE PER DATASET GETS NO PANEL LETTER. A lone panel called
            # "a)" promises a "b)" that is in a different file, and the
            # representation has to be on the panel because every grid in this
            # study is one representation and pooling them is the defect the
            # whole guard exists for.
            name = f'{C.dataset_label(dataset)}, {C.rep_label(rep)}'
            if len(group) > 1:
                S.title(ax, 'abc'[datasets.index(dataset)], name)
            else:
                ax.set_title(name, fontweight='bold', fontsize=9, loc='left')
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


# ---------------------------------------------------------------------------
# F6 -- Q5 and the aleatoric/epistemic decomposition
# ---------------------------------------------------------------------------

def f6_decomposition(q5, output_dir, rep, condition, slopes=None, support=None,
                     dataset='qm9', max_panels=6):
    """One small chart per model. Two lines in each: aleatoric and epistemic.

    Option 6A, settled 2026-09-04 as a headline. The bottom axis is how much
    noise was added to the training labels, seven positions, none on the left.
    The side axis is how much uncertainty the model reported, in the label's own
    units. Both lines are in every chart and keep the same two colours, so the
    colours are learned once.

    What you are looking for: the aleatoric line climbs and the epistemic line
    stays flat. Both climbing is a model that has failed to separate them, and
    that is drawn rather than hidden -- row 5 of the contingent list says the
    failure looks like two lines climbing together, so the figure is the same.

    A component that is one number per fit is NOT drawn: a flat line there is
    arithmetic about the fit, not a property of the molecules. `figlib_uncertainty.q5`
    has already dropped those rows, and the support flag is printed on the panel
    instead so the absence is legible (row 13 -- a guard, not a choice).
    """
    frame = q5[(q5['dataset'] == dataset) & (q5['rep'] == rep)
               & (q5['condition'] == condition)]
    frame = frame[frame['component'].isin(['aleatoric', 'epistemic'])]
    if 'split' in frame.columns:
        oof = frame[frame['split'] == 'train_oof']
        frame = oof if len(oof) else frame
    if not len(frame):
        return None

    models = C.sort_models(frame['model'].unique())[:max_panels]
    frame = frame[frame['model'].isin(models)]
    title = G.declare(frame, 'F6',
                      fixed={'dataset': dataset, 'rep': rep,
                             'condition': condition},
                      varies=('model', 'sigma'), aggregates=('fold',))

    flags = {}
    if support is not None and len(support):
        for _, row in support.iterrows():
            flags[str(row.get('model'))] = (str(row.get('aleatoric_support', '')),
                                            str(row.get('epistemic_support', '')))
    verdicts = {}
    if slopes is not None and len(slopes):
        for _, row in slopes.iterrows():
            if str(row.get('condition')) == condition and str(row.get('rep')) == rep:
                verdicts[str(row.get('model'))] = str(row.get('verdict', ''))

    caption('F6', f"""
        The two halves of the predicted uncertainty against the amount of noise
        added to the training labels, one panel per model, on
        {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}. One mark is the mean predicted
        uncertainty over the molecules of one out-of-fold pass, in the label's
        own units; the band is the range across folds. A model that is
        attributing added label noise to the data has an aleatoric line that
        climbs and an epistemic line that holds. A component that is one number
        per fit is not drawn as a line, and its support flag is printed on the
        panel instead. The bottom axis is the noise level, a fraction of the
        clean training label spread. Every panel starts at zero, so a component
        that holds still looks like one. Verdicts are from the slopes, not read
        off the picture.""")

    ncols = min(len(models), 4)
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = _fig(height=2.5 * nrows + 1.1, nrows=nrows, ncols=ncols,
                     sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for index, (ax, model) in enumerate(zip(axes, models)):
        panel = frame[frame['model'] == model]
        curve = (panel.groupby(['component', 'sigma'], dropna=False)
                 .agg(value=('mean_uncertainty', 'median'),
                      lo=('mean_uncertainty', 'min'),
                      hi=('mean_uncertainty', 'max'))
                 .reset_index())
        for component in ('aleatoric', 'epistemic'):
            one = curve[curve['component'] == component].sort_values('sigma')
            if not len(one):
                continue
            colour = C.COMPONENT_COLORS[component]
            ax.plot(one['sigma'], one['value'], marker='o', markersize=3.5,
                    linewidth=1.4, color=colour,
                    label=C.component_label(component))
            ax.fill_between(one['sigma'], one['lo'], one['hi'], color=colour,
                            alpha=0.15, linewidth=0)
        drawn = set(curve['component'])
        missing = [c for c in ('aleatoric', 'epistemic') if c not in drawn]
        if missing and model in flags:
            said = dict(zip(('aleatoric', 'epistemic'), flags[model]))
            note = 'not drawn: ' + '; '.join(
                f'{c} is {said.get(c, "unrecorded").replace("_", " ")}'
                for c in missing)
        else:
            note = verdicts.get(model, '')
        # From zero. A model whose model half really is flat wobbles by a
        # thousandth, and a panel scaled to its own range turns that into a
        # mountain -- which is the opposite of what the panel is claiming.
        ax.set_ylim(bottom=0)
        S.title(ax, 'abcdefgh'[index], C.model_label(model))
        if note:
            _panel_note(ax, note)
        ax.spines[['top', 'right']].set_visible(False)
        if index % ncols == 0:
            ax.set_ylabel('Mean predicted uncertainty\n(label units)')
    for ax in axes[len(models):]:
        ax.set_visible(False)
    # The short form of the bottom-axis label. The full one -- "fraction of
    # label spread" -- is four times the panel width here, and the caption
    # carries the unit instead.
    # sharex hides the tick NUMBERS on every row but the last. With six panels
    # in a four-wide grid the top row's last two panels have nothing beneath
    # them, so they were carrying the words "Noise level" over a bare line with
    # no numbers on it. Any panel that is the bottom of its own column gets its
    # numbers back.
    for ax in axes[len(models) - ncols:len(models)]:
        ax.set_xlabel('Noise level')
        ax.tick_params(labelbottom=True)
    # The representation is on the figure, not only in the caption. Every grid
    # and every panel in this study is ONE representation and pooling them is
    # the defect the guard exists for; a figure that does not say which one it
    # is cannot be checked.
    fig.suptitle(f'{C.dataset_label(dataset)}, {C.rep_label(rep)}, '
                 f'{C.condition_label(condition)}',
                 fontsize=9, fontweight='bold', x=0.01, ha='left')
    S.shared_legend(fig, axes[0], ncol=2)
    return S.save(fig, Path(output_dir) / 'F6_decomposition.png')


def _panel_note(ax, text, width=24):
    """A verdict printed inside a panel, wrapped to the panel's width.

    Not `stat_box`, and not the title: at four panels across, a box wide enough
    to hold "BOTH components rise -- the split failed" is wider than the panel
    and lands on its neighbour, and the title slot belongs to the model's name.

    Top left where the lines rise from the bottom, bottom left where they do
    not -- a model whose uncertainty is flat and high fills the top of its
    panel, and the note landed on the line.
    """
    import textwrap
    low, high = ax.get_ylim()
    span = (high - low) or 1.0
    left = [line.get_ydata()[:3] for line in ax.get_lines()
            if len(line.get_ydata())]
    crowded = any(float(np.nanmax(y)) > low + 0.6 * span for y in left if len(y))
    y, va = (0.03, 'bottom') if crowded else (0.97, 'top')
    ax.text(0.03, y, '\n'.join(textwrap.wrap(text, width)),
            transform=ax.transAxes, fontsize=6, va=va, ha='left',
            color='#444444', linespacing=1.25)


# ---------------------------------------------------------------------------
# F7 -- Q4 and Q6: does the uncertainty find the bad labels
# ---------------------------------------------------------------------------

def f7_uncertainty(option, output_dir, rep, condition, retention=None,
                   enrichment=None, q4=None, dataset='qm9', sigma=None):
    """Whichever of 7A, 7B and 7C the results chose. D7 makes that choice.

    7A the error-retention curve, 7B the enrichment curve, 7C the grid of
    `auc_delta` across conditions. All three answer the same question and only
    one goes in the main text, so only one is drawn -- and which one is decided
    from the numbers rather than from taste (RERUN_PLAN.md 14.6 rows 1 to 3).
    """
    if option == '7A':
        return _f7a_retention(retention, output_dir, rep, condition,
                              dataset=dataset, sigma=sigma)
    if option == '7B':
        return _f7b_enrichment(enrichment, output_dir, rep, condition,
                               dataset=dataset, sigma=sigma)
    if option == '7C':
        return _f7c_grid(q4, output_dir, rep, dataset=dataset, sigma=sigma)
    return None


def _one_level(frame, sigma):
    """The noise level a curve figure is drawn at, and it is one level.

    A curve pooled over levels is a different curve at every level averaged
    together. The reporting level is the default because it is the level every
    accuracy number in the paper is quoted at.
    """
    if not len(frame):
        return frame, None
    levels = sorted(pd.unique(frame['sigma'].dropna()))
    if sigma is None:
        # reporting_level RAISES for a dataset whose level is unset, which is
        # the point -- a silent default became the answer three times.
        wanted = C.reporting_level(str(frame['dataset'].iloc[0]))
        sigma = wanted if any(np.isclose(float(wanted), levels)) else max(levels)
    return frame[np.isclose(frame['sigma'].astype(float), float(sigma))], sigma


def _f7a_retention(retention, output_dir, rep, condition, dataset='qm9',
                   sigma=None):
    """SHAPE A. Bottom axis: the fraction of molecules thrown away, most
    uncertain first. Side axis: the error left on the ones you keep.

    One line per model, plus two grey reference lines -- throwing molecules away
    at random, and throwing them away in order of true error, which is the best
    any ordering could do. The gap between a model's line and the flat one is
    what its uncertainty is worth.
    """
    if retention is None or not len(retention):
        return None
    frame = retention[(retention['dataset'] == dataset)
                      & (retention['rep'] == rep)
                      & (retention['condition'] == condition)]
    frame, sigma = _one_level(frame, sigma)
    if not len(frame):
        return None
    title = G.declare(frame, 'F7A',
                      fixed={'dataset': dataset, 'rep': rep,
                             'condition': condition, 'sigma': sigma},
                      varies=('model',), aggregates=('fold', 'molecule'))

    models = C.sort_models(frame[frame['series'] == 'uncertainty']['model'].unique())
    caption('F7A', f"""
        Error against the clean label after discarding the most uncertain
        molecules, on {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}, at noise level {sigma}. Bottom axis: the
        fraction of out-of-fold molecules discarded, most uncertain first. Side
        axis: the root-mean-square error on the molecules that remain, in the
        label's own units. One coloured line per model. The flat grey line is
        discarding molecules at random and the lower grey line is discarding
        them in order of true error, which is the best any ordering could do.
        The gap between a model's line and the flat one is what its uncertainty
        is worth.""")

    fig, ax = _fig(height=3.4)
    for model in models:
        one = (frame[(frame['model'] == model) & (frame['series'] == 'uncertainty')]
               .sort_values('fraction'))
        ax.plot(one['fraction'], one['value'], marker=C.model_marker(model),
                markersize=3.5, linewidth=1.4, alpha=0.9,
                color=C.model_color(model), label=C.model_label(model))
    for series in ('random', 'oracle'):
        one = (frame[frame['series'] == series]
               .groupby('fraction', dropna=False)['value'].median().reset_index())
        if not len(one):
            continue
        ax.plot(one['fraction'], one['value'], linewidth=1.2,
                linestyle=C.CURVE_STYLES[series], color=C.CURVE_COLORS[series],
                label=C.curve_label(series), zorder=1)
    ax.set_xlabel('Fraction of molecules discarded, most uncertain first')
    ax.set_ylabel('RMSE on the molecules kept\n(label units)')
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=3)
    return S.save(fig, Path(output_dir) / 'F7_error_retention.png')


def _f7b_enrichment(enrichment, output_dir, rep, condition, dataset='qm9',
                    sigma=None):
    """SHAPE A. Bottom axis: the fraction of molecules looked at, most
    suspicious first. Side axis: the fraction of the corrupted labels found.

    The diagonal is what random picking gives. The second grey line is what
    ordering by out-of-fold error alone gives -- and that is the reference that
    matters, because the error already tracks the injected noise and the
    question is whether the uncertainty ADDS to it.
    """
    if enrichment is None or not len(enrichment):
        return None
    frame = enrichment[(enrichment['dataset'] == dataset)
                       & (enrichment['rep'] == rep)
                       & (enrichment['condition'] == condition)]
    frame, sigma = _one_level(frame, sigma)
    if not len(frame):
        return None
    title = G.declare(frame, 'F7B',
                      fixed={'dataset': dataset, 'rep': rep,
                             'condition': condition, 'sigma': sigma},
                      varies=('model',), aggregates=('fold', 'molecule'))

    models = C.sort_models(frame[frame['series'] == 'ratio']['model'].unique())
    top = frame['top_frac'].dropna()
    top_frac = float(top.iloc[0]) if len(top) else 0.10
    caption('F7B', f"""
        How many of the corrupted labels you have found, on
        {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}, at noise level {sigma}. Bottom axis: the
        fraction of out-of-fold molecules inspected, most suspicious first. Side
        axis: the fraction of the corrupted labels among them. Corrupted means
        the {top_frac:.0%} of molecules that received the largest injected
        noise, which is the same definition the Q4 statistic uses. One coloured
        line per model, ordered by out-of-fold error divided by predicted
        uncertainty. The straight grey diagonal is picking at random; the dashed
        grey line is ordering by out-of-fold error alone, which is what the
        uncertainty has to beat to have added anything.""")

    fig, ax = _fig(height=3.4)
    for model in models:
        one = (frame[(frame['model'] == model) & (frame['series'] == 'ratio')]
               .sort_values('fraction'))
        ax.plot(one['fraction'], one['value'], marker=C.model_marker(model),
                markersize=3.5, linewidth=1.4, alpha=0.9,
                color=C.model_color(model), label=C.model_label(model))
    # ON TOP, NOT UNDERNEATH. The error-alone curve is the line the whole
    # decision rests on -- whether the uncertainty adds anything to it -- and at
    # zorder 1 it sat behind a cluster of six model curves that lie on top of
    # each other, so it was in the key and invisible on the panel. Drawn last,
    # thicker, above the models, and in white-edged black so it reads against
    # any of them.
    for series in ('error', 'random'):
        one = (frame[frame['series'] == series]
               .groupby('fraction', dropna=False)['value'].median().reset_index())
        if not len(one):
            continue
        ax.plot(one['fraction'], one['value'],
                linewidth=2.6, color='white', alpha=0.85, zorder=4,
                solid_capstyle='round')
        ax.plot(one['fraction'], one['value'], linewidth=1.4,
                linestyle=C.CURVE_STYLES[series], color=C.CURVE_COLORS[series],
                label=C.curve_label(series), zorder=5)
    ax.set_title(f'{C.dataset_label(dataset)}, {C.rep_label(rep)}, '
                 f'{C.condition_label(condition)}, noise level {sigma}',
                 fontweight='bold', fontsize=9, loc='left')
    ax.set_xlabel('Fraction of molecules inspected, most suspicious first')
    ax.set_ylabel('Fraction of the corrupted\nlabels found')
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=3)
    return S.save(fig, Path(output_dir) / 'F7_enrichment.png')


def _f7c_grid(q4, output_dir, rep, dataset='qm9', sigma=None):
    """SHAPE C. Rows are the models, columns are the noise conditions, and the
    number on each square is how much dividing the error by the uncertainty
    improved the ranking of corrupted labels. Zero means it added nothing.

    This is the option that shows a null across every condition rather than
    asserting one in a sentence.

    ONE noise level, like the two curve options. A square holding the median
    across levels would be averaging over a factor: the same square would then
    mean something different for a condition whose ladder is short. The guard
    refuses it, which is how this was caught.
    """
    if q4 is None or not len(q4) or 'auc_delta' not in q4.columns:
        return None
    frame = q4[(q4['dataset'] == dataset) & (q4['rep'] == rep)]
    frame = frame[frame['auc_delta'].notna()]
    frame, sigma = _one_level(frame, sigma)
    if not len(frame):
        return None
    title = G.declare(frame, 'F7C',
                      fixed={'dataset': dataset, 'rep': rep, 'sigma': sigma},
                      varies=('model', 'condition'),
                      aggregates=('fold', 'molecule'))
    cell = (frame.groupby(['model', 'condition'], dropna=False)
            .agg(auc_delta=('auc_delta', 'median'),
                 outside=('outside_null', 'mean') if 'outside_null'
                 in frame.columns else ('auc_delta', 'size'))
            .reset_index())
    models = C.sort_models(cell['model'].unique())
    conditions = C.sort_conditions(cell['condition'].unique())
    caption('F7C', f"""
        How much dividing the out-of-fold error by the predicted uncertainty
        improved the ranking of corrupted labels, on
        {C.dataset_label(dataset)} at {C.rep_label(rep)}. Rows are models,
        columns are noise conditions, and one square is the median across folds
        at noise level {sigma}. Zero means the uncertainty added nothing to the
        error alone. Grey squares were never run.""")
    fig, ax = _fig(height=C.grid_height(len(models)))
    image, _ = S.grid(ax, cell, 'model', 'condition', 'auc_delta',
                      column_labeller=C.condition_label, vmin=-0.2, vmax=0.2,
                      cmap='RdBu_r', row_order=models, column_order=conditions,
                      fmt='{:+.2f}')
    if image is not None:
        bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        bar.set_label('Improvement in ranking corrupted labels', fontsize=8)
        bar.ax.tick_params(labelsize=7)
    return S.save(fig, Path(output_dir) / 'F7_uncertainty_grid.png')


# ---------------------------------------------------------------------------
# R6 -- the representation profiles (14.6 row 6, F3 option 3B)
# ---------------------------------------------------------------------------

def r6_representation_profile(summary, output_dir, condition, dataset='qm9',
                              value='auc_norm'):
    """SHAPE A turned sideways. Bottom axis: the six representations. Side axis:
    robustness. One mark is a dot; dots belonging to one model are joined by a
    line.

    A model whose line is flat is robust whatever the representation; a line
    that dives at one representation is the pairing, and you can see WHICH
    representation causes it. This is the only figure that shows the case the
    averaging rules exist to catch, and D5 fires it (row 6).
    """
    frame = summary[(summary['dataset'] == dataset)
                    & (summary['condition'] == condition)]
    if not len(frame):
        return None
    title = G.declare(frame, 'R6',
                      fixed={'dataset': dataset, 'condition': condition},
                      varies=('model', 'rep'))
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    position = {r: i for i, r in enumerate(reps)}
    models = C.sort_models(frame['model'].unique())
    caption('R6', f"""
        Robustness ({G.metric_label(value)}) of every model at every
        representation, on {C.dataset_label(dataset)} under
        {C.condition_label(condition)}. Bottom axis: the six representations.
        Side axis: robustness. One mark is one model at one representation, the
        median over replicates; dots belonging to one model are joined. A flat
        line is a model that is robust whatever the representation. A line that
        dives at one representation is the model-representation pairing, and the
        dive names the representation responsible.""")
    fig, ax = _fig(height=4.2)
    for model in models:
        one = frame[frame['model'] == model]
        one = one[one['rep'].isin(reps)].copy()
        if not len(one):
            continue
        one['x'] = one['rep'].map(position)
        one = one.sort_values('x')
        ax.plot(one['x'], one[value], marker=C.model_marker(model),
                markersize=4, linewidth=1.3, alpha=0.85,
                color=C.model_color(model), label=C.model_label(model))
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([C.rep_label(r) for r in reps], rotation=20, ha='right')
    ax.set_ylabel(G.metric_label(value))
    ax.set_xlabel('Representation')
    ax.set_title(f'{C.dataset_label(dataset)}, {C.condition_label(condition)}',
                 fontweight='bold', fontsize=9, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=4)
    return S.save(fig,
                  Path(output_dir) / f'R6_representation_profile_{condition}.png')


# ---------------------------------------------------------------------------
# R9 -- rank transfer as a figure (14.6 row 9)
# ---------------------------------------------------------------------------

def r9_rank_transfer(transfer, output_dir, rep, condition='gaussian'):
    """SHAPE B. Models down the side, rank along the bottom, one dot for the
    QM9 rank and one for the rank on each assay dataset.

    Promoted from T7 by D9, which fires when the two sides disagree on the
    ranking. A model whose dots sit together transfers; a model whose dots are
    far apart does not, and the spread along its row is the disagreement.
    """
    if transfer is None or not len(transfer):
        return None
    frame = transfer[(transfer['rep'] == rep)
                     & (transfer['condition'] == condition)]
    if not len(frame):
        return None
    G.declare(frame, 'R9', fixed={'rep': rep, 'condition': condition},
              varies=('model', 'dataset'))

    order = (frame.groupby('model', dropna=False)['qm9_rank'].median()
             .sort_values().index.tolist())
    rows = []
    for model in order:
        one = frame[frame['model'] == model]
        rows.append({'model': model, 'dataset': 'qm9',
                     'rank': float(one['qm9_rank'].median())})
        for _, r in one.iterrows():
            rows.append({'model': model, 'dataset': str(r['dataset']),
                         'rank': float(r['assay_rank'])})
    plotted = pd.DataFrame(rows)
    caption('R9', f"""
        Where each model ranks by robustness on QM9 and where it ranks on each
        assay dataset, at {C.rep_label(rep)} under
        {C.condition_label(condition)}. Models run down the side, ordered by
        their QM9 rank. Rank 1 is the most robust. One mark is that model's rank
        on one dataset; colour says which dataset. A model whose marks sit
        together keeps its place across datasets. A model whose marks are spread
        out does not, and the width of its row is the disagreement.""")
    fig, ax = _fig(height=C.grid_height(len(order)))
    colours = {'qm9': C.CLEAN_COLOR, 'logd': '#E69F00', 'caco2': '#009E73',
               'herg': '#CC79A7'}
    S.dot_rows(ax, plotted, 'model', 'rank', series='dataset',
               labeller=C.dataset_label, colours=colours, legend_ncol=4)
    # dot_rows puts its key inside the axes, which on nineteen rows lands on
    # the bottom model's row and hides it. One key below the figure instead.
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_xlabel('Rank by robustness (1 = most robust)')
    S.shared_legend(fig, ax, ncol=4)
    return S.save(fig, Path(output_dir) / f'R9_rank_transfer_{rep}.png')


# ---------------------------------------------------------------------------
# R10 -- where AUC_norm exceeds 1 (14.6 row 10)
# ---------------------------------------------------------------------------

def r10_auc_above_one(per_replicate, output_dir, dataset='qm9'):
    """Where a model scores better with noise added than without, against the
    clean baseline it started from.

    Not a patched metric -- a count and a picture of where it happens. A cell
    scoring above one almost always started from a low clean baseline, and that
    is exactly what the bottom axis shows.
    """
    if per_replicate is None or not len(per_replicate):
        return None
    frame = per_replicate[per_replicate['dataset'] == dataset]
    if 'baseline_r2' not in frame.columns or not len(frame):
        return None
    G.declare(frame, 'R10', fixed={'dataset': dataset},
              varies=('model', 'rep', 'condition'), aggregates=('replicate',))
    high = frame[frame['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH]
    caption('R10', f"""
        Every replicate's robustness against the clean baseline it was measured
        from, on {C.dataset_label(dataset)}. One mark is one replicate of one
        model at one representation under one noise condition. The horizontal
        line is {C.AUC_NORM_IMPLAUSIBLE_HIGH}, above which a model retained more
        than it started with. {len(high)} of {len(frame)} replicate values sit
        above it. They cluster at low clean baselines, which is what the metric
        does when the denominator is small; it is reported rather than
        patched.""")
    fig, ax = _fig(height=3.2)
    ax.scatter(frame['baseline_r2'], frame['auc_norm'], s=8, alpha=0.25,
               color='#999999', linewidth=0, label='every replicate')
    if len(high):
        ax.scatter(high['baseline_r2'], high['auc_norm'], s=14, alpha=0.85,
                   color='#D55E00', linewidth=0,
                   label=f'above {C.AUC_NORM_IMPLAUSIBLE_HIGH}')
        # NAME THEM. The whole point of this figure is WHERE it happens, and a
        # bare orange dot names nothing -- six of them stacked at one clean
        # baseline are six replicates of a single cell that the reader cannot
        # identify. Each cell is labelled once, at its highest replicate.
        for key, group in high.groupby(['model', 'rep', 'condition'],
                                       dropna=False):
            worst = group.loc[group['auc_norm'].idxmax()]
            model, representation, cond = key
            ax.annotate(f'{C.model_label(model)} / {C.rep_label(representation)}'
                        f' / {C.condition_label(cond)}'
                        f'  ({len(group)} rep.)',
                        xy=(float(worst['baseline_r2']),
                            float(worst['auc_norm'])),
                        xytext=(6, 2), textcoords='offset points',
                        fontsize=5.5, color='#8A3B00', va='bottom')
    ax.axhline(C.AUC_NORM_IMPLAUSIBLE_HIGH, color='#444444', linestyle='--',
               linewidth=0.9)
    ax.axvline(C.BASELINE_THRESHOLD, color='#444444', linestyle=':',
               linewidth=0.9)
    ax.annotate('clean-accuracy floor', xy=(C.BASELINE_THRESHOLD, 0.02),
                xycoords=('data', 'axes fraction'), fontsize=6.5,
                color='#444444', rotation=90, ha='right', va='bottom')
    ax.set_xlabel('Clean R² the replicate started from')
    ax.set_ylabel(G.metric_label('auc_norm'))
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=2)
    return S.save(fig, Path(output_dir) / 'R10_auc_above_one.png')


# ---------------------------------------------------------------------------
# R15b -- the mirror of the rank chart (5.4a)
# ---------------------------------------------------------------------------

def r15b_rank_against_level_by_rep(accuracy, output_dir, model, condition,
                                   dataset='qm9', baseline_gate=None):
    """The mirror the author asked for: hold one noise type and one MODEL, and
    plot every representation.

    Same chart as R15 with the two factors swapped. The three rules for a
    representation that stops working are the same three, and the R² is printed
    beside the ranks for the same reason -- a rank hides whether two are a
    thousandth apart or a fifth apart.
    """
    gate = C.BASELINE_THRESHOLD if baseline_gate is None else baseline_gate
    frame = accuracy[(accuracy['dataset'] == dataset)
                     & (accuracy['model'] == model)
                     & (accuracy['condition'] == condition)]
    if not len(frame):
        return None
    title = G.declare(frame, 'R15b',
                      fixed={'dataset': dataset, 'model': model,
                             'condition': condition},
                      varies=('rep', 'sigma'), aggregates=('replicate',))
    clean = (frame[frame['sigma'] == frame['sigma'].min()]
             .groupby('rep')['r2'].median())
    alive = clean[clean >= gate].index
    frame = frame[frame['rep'].isin(alive)]
    if not len(frame):
        return None

    ranked = []
    for (sigma, replicate), group in frame.groupby(['sigma', 'replicate']):
        order = group.set_index('rep')['r2'].rank(ascending=False)
        for name, rank in order.items():
            ranked.append({'rep': name, 'sigma': sigma, 'rank': rank})
    ranks = (pd.DataFrame(ranked).groupby(['rep', 'sigma'], dropna=False)['rank']
             .median().reset_index())
    scores = (frame.groupby(['rep', 'sigma'], dropna=False)['r2'].median()
              .reset_index())
    ranks = ranks.merge(scores, on=['rep', 'sigma'], how='left')
    # A representation that worked and then fell below the accuracy floor has
    # its line CUT SHORT there rather than plunging to last place.
    ranks = ranks[ranks['r2'] >= gate]

    caption('R15b', f"""
        Where each representation ranks against the others as the noise rises,
        for {C.model_label(model)} on {C.dataset_label(dataset)} under
        {C.condition_label(condition)}. Bottom axis: the noise level. Side axis:
        rank, 1 at the top. One line per representation, and lines crossing are
        representations trading places. Ranks are taken within each replicate
        and then the median is shown. A line stops where that representation's
        R² falls below {gate}, rather than plunging to last place. R² is printed
        beside each mark, because a rank hides whether two are a thousandth
        apart or a fifth apart.""")

    fig, ax = _fig(height=3.4)
    order = (ranks[ranks['sigma'] == ranks['sigma'].min()]
             .sort_values('rank')['rep'].tolist())
    # TWO REPRESENTATIONS TIED AT ONE LEVEL SHARE A MARK, and at a fixed offset
    # their two R2 labels printed over each other -- a tie at level 0 came out
    # as one unreadable smear of two numbers. Each label after the first at a
    # position is lifted clear.
    taken = {}
    for name in order:
        one = ranks[ranks['rep'] == name].sort_values('sigma')
        if not len(one):
            continue
        ax.plot(one['sigma'], one['rank'], marker='o', markersize=4,
                linewidth=1.4, label=C.rep_label(name))
        for _, row in one.iterrows():
            spot = (round(float(row['sigma']), 6), round(float(row['rank']), 3))
            nth = taken.get(spot, 0)
            taken[spot] = nth + 1
            ax.annotate(f"{row['r2']:.2f}",
                        xy=(row['sigma'], row['rank']), fontsize=5.5,
                        xytext=(0, 5 + 7 * nth), textcoords='offset points',
                        ha='center', color='#555555')
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_xlabel(LEVEL_AXIS)
    ax.set_ylabel('Rank (1 = most accurate)')
    ax.set_title(f'{C.model_label(model)} on {C.dataset_label(dataset)}, '
                 f'{C.condition_label(condition)}',
                 fontweight='bold', fontsize=9, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=3)
    return S.save(
        fig, Path(output_dir) / f'R15b_rank_by_rep_{model}_{condition}.png')
