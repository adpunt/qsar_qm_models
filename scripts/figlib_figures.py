#!/usr/bin/env python
"""The paper's figures. Six slots, four shapes, and every one declares what it
holds fixed before it draws.

RERUN_PLAN.md 14.5. Each builder takes the tidy frames, calls `G.declare` to
state what it holds fixed and what it varies -- which RAISES if the data still
carries a factor the figure has not accounted for -- and gets back the title
text, generated from the data rather than typed, so a caption cannot drift away
from the numbers behind it.

  F1  what each noise condition does to the labels          (Methods)
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


def panel_layout(n_panels, n_rows, n_columns_each, name,
                 longest_label=0):
    """Stacked, side by side, or one file each -- decided, not guessed.

    The old script stacks panels vertically every time, and says why in three
    separate comments: side by side at full width squeezes the grid and the
    legend. So stacked is the default. But nineteen models stacked three deep is
    over 450 mm and the journal allows 225, and a grid squeezed to 7 mm a column
    is unreadable however tall it is. So:

      stacked        if the stack fits the page
      side by side   if it does not, and the columns still get enough width
                     FOR THEIR LABELS
      one file each  otherwise -- an unreadable single figure is worth less than
                     three readable ones, and the journal caps neither

    `longest_label` is the character count of the longest bottom-axis label.
    Column width alone is not the constraint: three four-column grids at 14 mm a
    column passed this test and then drew "Grouped, shifted" rotated 35 degrees
    out of its own panel and across the panel beside it. A rotated label needs
    horizontal room of roughly its length times the character width times
    cos(35 degrees), and that is what decides whether panels can sit side by
    side (the author, 2026-09-16).

    Returns 'stacked', 'across' or 'split', having said which and why.
    """
    stacked = C.grid_height(n_rows, n_panels)
    if stacked <= C.MAX_HEIGHT_IN:
        return 'stacked'
    across = n_panels * n_columns_each
    per_column = 170 / across if across else 170
    # An 8-point character is about 1.6 mm wide; rotated 35 degrees a label of
    # `longest_label` characters sweeps this far sideways.
    needed = max(12.0, longest_label * 1.6 * 0.82)
    if across <= MAX_COLUMNS_ACROSS and per_column >= needed:
        print(f'    {name}: {n_panels} stacked panels of {n_rows} rows is '
              f'{stacked * 25.4:.0f} mm, past the '
              f'{C.MAX_HEIGHT_IN * 25.4:.0f} mm the journal allows. Side by '
              f'side instead -- {across} columns get {per_column:.0f} mm each, '
              f'and the longest label needs {needed:.0f} mm.')
        return 'across'
    why = ('too many columns' if across > MAX_COLUMNS_ACROSS else
           f'each column would get {per_column:.0f} mm and the longest bottom '
           f'label needs {needed:.0f} mm, so the labels would run into the '
           f'panel beside them')
    print(f'    {name}: {n_panels} stacked panels is {stacked * 25.4:.0f} mm '
          f'(limit {C.MAX_HEIGHT_IN * 25.4:.0f}) and side by side is out -- '
          f'{why}. Writing one figure per panel instead.')
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
    # +1 row for the scaffold-group panel. A HISTOGRAM CANNOT SHOW WHAT THE
    # GROUPED CONDITIONS DO. Grouped-shifted is a per-group offset plus a
    # per-molecule error, both Gaussian, with the two variances summing to the
    # same total -- so its marginal distribution IS the Gaussian condition's,
    # exactly, by construction. The difference lives entirely in how much of the
    # noise a scaffold group shares, and this panel is the only place in the
    # figure that shows it (the author, 2026-09-13).
    nrows = int(np.ceil(len(conditions) / ncols)) + 1
    import matplotlib.pyplot as plt
    # 1.35 inches a row with hspace 0.95 put each panel's title into the panel
    # above it and left the bottom panel's rotated tick labels over the legend.
    # Taller rows and more room between them (the author, 2026-09-14).
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN,
                              min(C.MAX_HEIGHT_IN, 1.62 * nrows + 0.6)))
    spec = fig.add_gridspec(nrows, ncols, hspace=1.15, wspace=0.26,
                            bottom=0.09, top=0.96)

    for index, condition in enumerate(conditions):
        ax = fig.add_subplot(spec[index // ncols, index % ncols])
        noised = drawn[condition]
        colour = C.CONDITION_COLORS.get(condition, '#666666')
        ax.hist(clean, bins=bins, density=True, alpha=0.18,
                color=C.CLEAN_BACKDROP_COLOR)
        ax.hist(noised, bins=bins, density=True, alpha=0.15, color=colour)
        ax.hist(clean, bins=bins, density=True, histtype='step', linewidth=1.6,
                color=C.CLEAN_BACKDROP_COLOR, alpha=0.9)
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
        # The delivered amount is NOT written on the panel. It is a check that
        # the doses match, not a property of the distribution being drawn, and
        # seven copies of it were seven pieces of text over seven histograms
        # (the author, 2026-09-16). It is in F1_delivered_dose.csv and in one
        # sentence of Methods.
        # Only the bottom histogram in each column, or the label lands on the
        # title of the panel underneath it.
        if index >= len(conditions) - ncols:
            ax.set_xlabel('Label value', fontsize=8)

    # The scaffold-group panel, across the full width.
    ax = fig.add_subplot(spec[nrows - 1, :])
    # CENSORING IS NOT ON THIS PANEL. It is not dose-matched to the others and
    # has no variance parameter, so the share of its noise carried by a group
    # is not comparable with theirs and invites a comparison that cannot be
    # made (the author, 2026-09-16). Said in the caption instead.
    order = [c for c in C.sort_conditions(conditions)
             if not str(c).startswith('censoring')]
    shared = []
    for position, condition in enumerate(order):
        noised = drawn[condition]
        error = noised - clean
        # How much of the noise a whole scaffold group shares: the spread of the
        # group means, against the spread of the noise overall. Near zero means
        # the noise scatters within a group; near one means the group moves as a
        # block.
        means = pd.Series(error).groupby(groups).mean()
        overall = float(np.std(error))
        share = float(np.std(means) / overall) if overall > 0 else np.nan
        shared.append({'condition': condition, 'group_share': share})
        colour = C.CONDITION_COLORS.get(condition, '#666666')
        ax.bar([position], [share], width=0.62, color=colour, alpha=0.9)
        ax.annotate(f'{share:.2f}', xy=(position, share), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=7,
                    color='#333333')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([C.condition_label(c) for c in order], rotation=30,
                       ha='right', fontsize=7)
    # The side axis is a share of one, so it says so in three words and the
    # title carries the meaning. It used to repeat the title down the side.
    ax.set_ylabel('Share of the\nnoise, 0 to 1', fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    S.title(ax, 'abcdefghij'[len(conditions)],
            'How much of the noise a whole scaffold group shares')
    ax.spines[['top', 'right']].set_visible(False)
    pd.DataFrame(shared).to_csv(
        Path(output_dir) / 'F1_group_share.csv', index=False)

    # THE DOSE PANEL IS GONE -- the author's call, 2026-09-13. It showed that
    # every condition delivers the amount it was asked for, which is a
    # precondition of the comparison rather than a result, and it is one
    # sentence of text. The numbers are still computed and printed here so that
    # sentence can be written from a measurement rather than from memory, and
    # they are written to a CSV beside the figure.
    summary = (delivered.groupby('condition')['delivered']
               .agg(['median', 'min', 'max']).reset_index())
    summary['asked_for'] = level * spread
    summary['off_by'] = (summary['median'] - summary['asked_for']).abs()
    dose_path = Path(output_dir) / 'F1_delivered_dose.csv'
    summary.to_csv(dose_path, index=False)
    matched = summary[~summary['condition'].str.startswith('censoring')]
    if len(matched):
        worst = matched.loc[matched['off_by'].idxmax()]
        print(f'  F1: the dose check is a CSV, not a panel. Asked for '
              f'{level * spread:.4f} in label units; the dose-matched '
              f'conditions delivered between {matched["median"].min():.4f} and '
              f'{matched["median"].max():.4f}, worst off by '
              f'{worst["off_by"]:.4f} ({C.condition_label(worst["condition"])}). '
              f'Censoring is not dose-matched and cannot be. '
              f'Written to {dose_path.name}.')

    caption('F1', f"""
        What each settled noise condition does to a label distribution.
        Panels (a) onward: the clean labels and the noised labels over each
        other, same bins in every panel, at a level of {level:g} of the clean
        label spread — censoring at {censored_fraction:.0%} of labels clipped,
        because its level is a fraction clipped rather than a fraction of the
        spread. Drawn with the injector the pipeline runs, not a
        reimplementation. The amount each condition actually delivered is
        reported in the text and in F1_delivered_dose.csv rather than written on
        each panel, and censoring is absent from the final panel because it has
        no variance parameter to dose-match and its group share is therefore not
        comparable with the others'. The final panel is the one that separates
        the grouped
        conditions from the plain one, and it is there because the panels above
        it cannot: grouped-shifted gives every scaffold group a constant offset
        and every molecule its own error on top, both drawn from the same shape,
        with the two variances summing to the same total — so the distribution
        of its noise is the same as the plain condition's, exactly. What differs
        is how much of the noise a whole scaffold group shares, which is what
        the final panel measures: the spread of the group mean errors divided by
        the spread of all the errors. Near zero means the noise scatters inside
        a group; near one means the group moves as a block. That the conditions
        deliver the same amount of noise,
        and so differ in kind rather than in dose, is reported in the text and
        measured in F1_delivered_dose.csv rather than drawn as a panel.""")
    return S.save(fig, Path(output_dir) / 'F1_noise_conditions.png')


# ---------------------------------------------------------------------------
# F2 -- Q1: model, representation, or their pairing
# ---------------------------------------------------------------------------

#: FOUR BARS, AND THE WHISKERS STAY TOO. The author dropped the residual bar on
#: 2026-09-16 and reinstated it on 2026-09-17 after reading the whiskers on the
#: rendered figure: they are narrow everywhere except grouped-shifted, so they
#: do not crowd it. RERUN_PLAN.md 14.11ab. Neither the residual nor the spread
#: may leave without her word -- test_figure_slots checks for both.
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
        # WHICH LEVEL. The accuracy panel is R2 at ONE noise level and the
        # title never said which, so a reader could not tell whether the shares
        # were a property of the study or of a setting.
        level = C.reporting_level(dataset)
        heading = str(outcome)
        for phrase in ('the reported level', 'the reporting level'):
            heading = heading.replace(phrase, f'noise level {level:g}')
        S.title(ax, 'abcd'[index] if index < 4 else str(index), heading)
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
    # BESIDE THE PANELS, not beneath them: four short factor names cost a
    # whole band of height under a two-panel figure (the author, 2026-09-18).
    S.shared_legend(fig, axes[0], side=True)

    n_reps = int(frame['n_replicates'].max()) if 'n_replicates' in frame else 0
    n_reps_less = max(n_reps - 1, 0)
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
        sum to 100 per cent. The whiskers are NOT confidence intervals: they are
        a leave-one-out jackknife over the {n_reps} replicates -- drop one
        replicate, decompose the remaining {n_reps_less}, repeat -- so they say
        how much the answer depends on any one replicate, not how precisely the
        share is known. The leftover share is the variation between replicates
        of one identical configuration: same model, same representation, same
        noise condition, same noise level, different seed.{missing_note}""")
    return S.save(fig, Path(output_dir) / 'F2_variance_decomposition.png')


def f2b_clean_decomposition(anova_clean, output_dir):
    """One panel: the same four terms as F2, on accuracy with no noise added,
    with the four datasets along the bottom instead of the noise conditions.

    F2 answers how the model and the representation divide the variance once
    label noise is in the training set. This answers the prior question -- how
    they divide it before any noise is added -- and it is the only decomposition
    in the study that can put the computed property and the three measured
    endpoints on one axis, because the clean fit is the one thing all four
    datasets have in common.

    There is no noise-condition axis here on purpose: the clean fit is made once
    per replicate and shared by every condition, so a condition axis would be
    the same bar drawn seven times.
    """
    if anova_clean is None or not len(anova_clean):
        return None
    long = []
    for _, row in anova_clean.iterrows():
        for column, label in FACTOR_COLUMNS:
            long.append({'dataset': row['dataset'], 'factor': label,
                         'share': row.get(column, np.nan),
                         'spread': row.get(f'{column}_spread', np.nan)})
    long = pd.DataFrame(long)
    order = [d for d in C.DATASET_ORDER if d in set(long['dataset'])]
    order += [d for d in dict.fromkeys(long['dataset']) if d not in order]

    fig, ax = _fig(height=3.4)
    S.grouped_bars(ax, long, 'dataset', 'factor', 'share', spread='spread',
                   labeller=str, category_labeller=C.dataset_label,
                   colours=C.ANOVA_FACTOR_COLORS,
                   legend=False, categories=order, clip=(0, 100))
    ax.set_ylabel('Share of variance (%)')
    ax.set_xlabel('Dataset')
    ax.set_ylim(0, 100)
    S.shared_legend(fig, ax, side=True)

    counts = {str(r['dataset']): (int(r['n_models']), int(r['n_reps']),
                                  int(r['n_replicates']))
              for _, r in anova_clean.iterrows()}
    spelled = '; '.join(
        f'{C.dataset_label(d)} {m} models by {p} representations by {k} '
        f'replicates, {m * p * k} values'
        for d, (m, p, k) in counts.items())
    caption('F2b', f"""
        How much of the variation in predictive accuracy on CLEAN labels is
        explained by the choice of model, the choice of representation, the
        pairing of the two, and what is left over, on each dataset. Accuracy is
        R$^2$ on held-out molecules with no noise added to the training labels.
        The bottom axis is the dataset and the side axis is the share of
        variance, from a two-way analysis with sequential sums of squares; the
        four shares within a dataset sum to 100 per cent. One bar is one term on
        one dataset. There is no noise-condition axis: the clean fit is made
        once per replicate and every noise condition's ladder starts from that
        same fit, so a condition axis would repeat each bar seven times. The
        whiskers are NOT confidence intervals: they are a leave-one-replicate-out
        jackknife -- drop one replicate, decompose the rest, repeat -- so they
        say how much the answer depends on any one replicate, not how precisely
        the share is known. The leftover share is the variation between
        replicates of one identical configuration: same model, same
        representation, same dataset, different seed. {spelled}.""")
    return S.save(fig, Path(output_dir) / 'F2b_clean_decomposition.png')


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
    frame = C.cross_model(summary[summary['dataset'] == dataset], 'F3')
    frame = frame[frame['condition'].isin(conditions)]
    if not len(frame):
        return None
    title = G.declare(frame, 'F3', fixed={'dataset': dataset},
                      varies=('model', 'rep', 'condition'))

    models = C.sort_models(frame['model'].unique())
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    shown = [c for c in C.sort_conditions(frame['condition'].unique())]

    lo, hi = C.auc_range(dataset)
    caption('F3', f"""
        Robustness ({G.metric_label(value)}) of every model on every
        representation, one panel per noise condition, on
        {C.dataset_label(dataset)}. Rows are models ordered by family, columns
        are representations. Which conditions appear is decided from the data:
        conditions whose grids repeat another's are held back to an additional
        file, because showing both is showing one thing twice. Colour is on one
        fixed range across the panels, printed on the colour bar; the range is
        narrower than the metric's full span because every value on this dataset
        falls inside it, and a scale wider than the data paints every cell the
        same shade. A grey cell marked "not run" was never fitted; one marked
        "excluded" ran and was dropped.""")

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


def f3b_every_full_roster_condition(summary, output_dir, dataset='qm9',
                                    value='auc_norm'):
    """F3 again, with NO condition left out of it.

    F3 shows the conditions whose grids differ, because two grids that agree to
    inside the replicate wobble are one picture drawn twice and the main text
    has room for one of them. That is a decision about the MAIN TEXT and not a
    decision to stop looking: a condition held back as a repeat still has to be
    written about, and it cannot be written about from a figure that does not
    exist (the author, 2026-09-23).

    So this draws every condition the whole roster was run on, repeats included,
    at the same fixed colour range as F3, for the additional files. The
    conditions that ran on a named subset of pairs are not here -- their rows
    would be mostly grey against these -- they are R19, and between the two
    every condition in the study has a grid.
    """
    frame = C.cross_model(summary[summary['dataset'] == dataset], 'F3b')
    if not len(frame):
        return None
    models = C.sort_models(frame['model'].unique())
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    full = len(models) * len(reps)

    # WHICH CONDITIONS RAN ON THE WHOLE ROSTER, asked of the data rather than
    # named here. A list written into the code goes stale the moment a run
    # fills in a condition, and then the figure quietly omits it.
    counts = frame.groupby('condition')['auc_norm'].size()
    shown = [c for c in C.sort_conditions(counts.index)
             if int(counts[c]) >= full]
    partial = [c for c in C.sort_conditions(counts.index) if c not in shown]
    if not shown:
        return None
    frame = frame[frame['condition'].isin(shown)]
    G.declare(frame, 'F3b', fixed={'dataset': dataset},
              varies=('model', 'rep', 'condition'))

    lo, hi = C.auc_range(dataset)
    longest = max((len(C.rep_label(r)) for r in reps), default=0)
    layout = panel_layout(len(shown), len(models), len(reps), 'F3b',
                          longest_label=longest)
    # THREE GRIDS OF THIRTEEN ROWS DO NOT FIT ON ONE PAGE, and eighteen columns
    # across do not either. One file per condition then, as F8 already does for
    # the three measured datasets: three readable figures beat one that the
    # journal rejects on height. The colour range is fixed, so they still
    # compare (the author's rule for F8, 2026-09-18).
    groups = [[c] for c in shown] if layout == 'split' else [shown]
    written = []
    for group in groups:
        if layout == 'stacked' or len(group) == 1:
            fig, axes = _fig(height=C.grid_height(len(models), len(group)),
                             nrows=len(group), sharex=True)
        else:
            fig, axes = _fig(height=C.grid_height(len(models)),
                             ncols=len(group), sharey=True)
        axes = np.atleast_1d(axes)
        image = None
        for index, (ax, condition) in enumerate(zip(axes, group)):
            panel = frame[frame['condition'] == condition]
            image, _ = S.grid(ax, panel, 'model', 'rep', value,
                              column_labeller=C.rep_label, vmin=lo, vmax=hi,
                              row_order=models, column_order=reps)
            letter = 'abcdefg'[shown.index(condition)]
            S.title(ax, letter, C.condition_label(condition))
        if image is not None:
            bar = fig.colorbar(image, ax=list(axes), fraction=0.02, pad=0.02)
            bar.set_label(G.metric_label(value), fontsize=8)
            bar.ax.tick_params(labelsize=7)
        suffix = f'_{group[0]}' if len(groups) > 1 else ''
        written.append(S.save(
            fig,
            Path(output_dir) / f'F3b_every_condition_{dataset}{suffix}.png'))

    elsewhere = (' The remaining condition(s) -- '
                 + ', '.join(C.condition_label(c) for c in partial)
                 + ' -- ran on a named subset of the pairs rather than on the '
                   'whole roster and have their own figure, so between the two '
                   'every noise condition in the study has a grid.'
                 if partial else '')
    caption('F3b', f"""
        Robustness ({G.metric_label(value)}) of every model on every
        representation, for every noise condition the whole roster was run on,
        on {C.dataset_label(dataset)}: {", ".join(C.condition_label(c) for c in shown)}.
        Rows are the {len(models)} models ordered by family, columns are the
        {len(reps)} representations, one panel per condition. One cell is one
        model on one representation, the median over the replicates, with its
        value printed on it. Colour is on the same fixed range as the main-text
        grid, printed on the colour bar. The main text carries only the
        conditions whose grids differ from one another; this carries all of
        them, including the one held back there as a repeat, so that each can be
        read on its own. The panels are {'separate image files, one per '
        'condition, because three grids of this many rows do not fit one page; '
        'they share one colour range and one row order so they still compare'
        if len(written) > 1 else 'one image'}.{elsewhere}""")
    return written


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
                top_models=8, excluded=None, second_rep='pdv'):
    """THREE FIGURES, not three panels -- the author's call, 2026-09-13.

    One figure was 170 mm wide and 250 mm tall carrying two line charts and a
    nineteen-row grid. The bands across replicates are a real measurement and
    at that size none of them could be read, and the two line charts sat side
    by side looking like two views of one thing when they hold different things
    fixed.

    F4a  accuracy against noise level, one line per MODEL, at one noise condition.
    F4b  accuracy against noise level, one line per NOISE CONDITION, for one model.
    F4c  the grid: models down the side, noise conditions across, robustness on each
         square, with clean accuracy as an uncoloured first column.

    Censoring is on none of them: its bottom axis is a fraction of labels
    clipped, not a fraction of the label spread, so it cannot share an axis or a
    colour scale, and it runs on five named pairs and cannot rank models.
    """
    acc = C.cross_model(
        accuracy[(accuracy['dataset'] == dataset) & (accuracy['rep'] == rep)],
        'F4')
    rankable = G.ranking_conditions(sorted(acc['condition'].dropna().unique()))
    acc = acc[acc['condition'].isin(rankable)]
    if not len(acc):
        return None

    summ = summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)]
    summ = summ[summ['condition'].isin(rankable)]

    # BASE MODELS BEFORE THE TOP-N IS TAKEN, AND THAT ORDER MATTERS. The panels
    # drop the variant models (14.11k), but this ranking did not, so a variant
    # could take one of the N slots and then be filtered out of the drawing --
    # leaving N-1 lines under a caption built from len(keep), which says N. On
    # the 2026-09-17 harvest GP (het.) took slot 8 at ECFP4 and the figure drew
    # seven models while the caption promised eight. Rank what will be drawn.
    summ = C.cross_model(summ, 'F4a ranking')
    order = (summ[summ['condition'] == reference_condition]
             .sort_values('auc_norm', ascending=False)['model'].tolist())
    order = order or C.sort_models(acc['model'].unique())
    # RF, BY INSTRUCTION (the author, 2026-09-13). The data's own pick is
    # NGBoost, which is top by AUC_norm and near LAST by clean accuracy -- an
    # interesting case that earns its own figure, not the model a reader should
    # meet first as the example of what noise does. --focus-model overrides.
    if focus_model is None:
        focus_model = (C.DEFAULT_FOCUS_MODEL
                       if C.DEFAULT_FOCUS_MODEL in set(acc['model'])
                       else (order[0] if order else None))
    keep = order[:top_models]
    level = C.reporting_level(dataset)
    written = []

    # ---- F4a: the models, under one noise condition ----------------------------
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
    # TWO PANELS, ONE PER REPRESENTATION. The same models under the same noise,
    # on the representation the tables hold and on one more, so the reader can
    # see that a model's curve is a fact about the pairing and not about the
    # model (the author, 2026-09-16). NGBoost is the case that makes it: its
    # curve is the flattest on both and it starts far lower on ECFP4.
    # `acc` is filtered to ONE representation at line 586, so asking it whether
    # it holds `second_rep` is always False and the companion panel could never
    # be drawn -- while the caption below promised it. Ask the unfiltered frame,
    # which is what the per-panel loop reads from.
    available = set(accuracy[accuracy['dataset'] == dataset]['rep'].dropna())
    companions = [r for r in (second_rep,)
                  if r and r != rep and r in available]
    panes = [rep] + companions
    fig, axes = _fig(height=4.0, ncols=len(panes), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for index, (ax, held) in enumerate(zip(axes, panes)):
        here = (accuracy[(accuracy['dataset'] == dataset)
                         & (accuracy['rep'] == held)
                         & (accuracy['condition'] == reference_condition)
                         & (accuracy['model'].isin(keep))]
                .groupby(['model', 'sigma'], as_index=False)['r2'].median())
        here = C.cross_model(here, f'F4a {held}')
        if not len(here):
            continue
        # NO BANDS. Eight overlapping shaded ranges made the lines unreadable
        # and the spread is in T4 (the author, 2026-09-13).
        S.line_chart(ax, here, 'sigma', 'r2', 'model',
                     labeller=C.model_label, reference_x=level,
                     reference_label='reported at' if index == 0 else None,
                     legend=False)
        ax.set_xlabel(LEVEL_AXIS)
        ax.spines[['top', 'right']].set_visible(False)
        S.title(ax, 'ab'[index], C.rep_label(held))
        if index == 0:
            ax.set_ylabel(G.metric_label('r2'))
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).
    S.shared_legend(fig, axes[0], ncol=4)
    # The two-panel sentences are only true when the companion panel drew. A
    # caption that claims a panel the figure does not have is the failure this
    # guard exists to stop.
    two_panels = (f"One panel per representation, the same models and the same "
                  f"noise on each, sharing a side axis. " if len(panes) > 1 else "")
    why_two = ("Two panels rather than one because how far a model falls is a "
               "property of the pairing and not of the model: the same curve "
               "can start high on one representation and low on another while "
               "keeping the same shape. " if len(panes) > 1 else "")
    caption('F4a', f"""
        What label noise costs you, on {title}. One line per model, the
        {len(keep)} most robust under {C.condition_label(reference_condition)}.
        {two_panels}Bottom axis: the amount of noise put into the
        training labels, as a fraction of the clean training label spread. Side
        axis: R2 on held-out molecules, median over the ten replicates. {why_two}The dashed vertical line
        marks the level every table in the paper reports at. There are no bands:
        eight overlapping ranges hid the lines, and the spread across replicates
        is in T4.""")
    written.append(S.save(fig, Path(output_dir) / 'F4a_models_under_noise.png'))

    # ---- F4b: one model, across every noise condition ---------------------------
    if focus_model:
        by_condition = (acc[acc['model'] == focus_model]
                        .groupby(['condition', 'sigma'], as_index=False)['r2']
                        .median())
        G.declare(acc[acc['model'] == focus_model], 'F4b',
                  fixed={'dataset': dataset, 'rep': rep, 'model': focus_model},
                  varies=('condition', 'sigma'), aggregates=('replicate',))
        fig, ax = _fig(height=4.0)
        # No reference line: this panel is about the SHAPE of the curves
        # against each other, and a vertical rule through them was clutter
        # (the author, 2026-09-13).
        # NO BANDS HERE EITHER. Six overlapping shaded ranges on one panel hid
        # the lines they belonged to, and the lines are the point (the author,
        # 2026-09-16). The spread across replicates is in T4.
        S.line_chart(ax, by_condition, 'sigma', 'r2', 'condition',
                     labeller=C.condition_label,
                     colours=C.CONDITION_COLORS, legend=False)
        ax.set_ylabel(G.metric_label('r2'))
        ax.set_xlabel(LEVEL_AXIS)
        # NO FIGURE TITLE. A journal caption carries what this is; a title
        # above it says the same thing twice (the author, 2026-09-17).
        ax.spines[['top', 'right']].set_visible(False)
        # BESIDE THE CHART. Six condition names under a single wide panel
        # pushed the lines into the top half (the author, 2026-09-18).
        S.shared_legend(fig, ax, side=True)
        caption('F4b', f"""
            Whether the KIND of noise matters or only the amount, for
            {C.model_label(focus_model)} on {C.dataset_label(dataset)} at
            {C.rep_label(rep)}. One line per noise condition, same axes as the
            previous figure, each the median over the ten replicates. There are
            no shaded ranges: six of them overlapping hid the lines, and the
            spread across replicates is in T4. Lines that lie on top of each other mean this model
            cannot tell the noise conditions apart at a matched dose; lines that
            separate mean the shape of the noise costs something beyond its
            size. Censoring is absent: its level is a fraction of labels clipped
            rather than a fraction of the label spread, so it does not share
            this bottom axis.""")
        written.append(S.save(
            fig, Path(output_dir) /
            f'F4b_{focus_model}_across_noise_conditions.png'))

    # ---- F4c: the grid -----------------------------------------------------
    # A MOSTLY GREY COLUMN IS NOT PUBLISHABLE and will never fill in: the deep
    # run covers a named subset of pairs by design. Those conditions leave the
    # grid and get their own figure (the author, 2026-09-13).
    wide, thin = G.full_roster_conditions(summ, where='F4c')
    summ = summ[summ['condition'].isin(wide)] if wide else summ
    baseline = (summ.groupby('model', as_index=False)['baseline_r2'].median()
                .assign(condition='clean')
                .rename(columns={'baseline_r2': 'auc_norm'}))
    both = pd.concat([baseline, summ[['model', 'condition', 'auc_norm']]],
                     ignore_index=True)
    dropped = _excluded_keys(excluded, ('model', 'condition'),
                             dataset=dataset, rep=rep)
    lo, hi = C.auc_range(dataset)
    fig, ax = _fig(height=C.grid_height(len(order)))
    image, _ = S.grid(ax, both, 'model', 'condition', 'auc_norm',
                      vmin=lo, vmax=hi,
                      excluded=dropped,
                      row_order=order, separate_first_column=True,
                      column_order=['clean'] + C.sort_conditions(
                          [c for c in summ['condition'].unique()]),
                      column_labeller=lambda c: ('Clean R²' if c == 'clean'
                                                 else C.condition_label(c)))
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(G.metric_label('auc_norm'), fontsize=8)
    bar.ax.tick_params(labelsize=7)
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).
    caption('F4c', f"""
        Robustness ({G.metric_label('auc_norm')}) by model and noise condition on
        {C.dataset_label(dataset)} at {C.rep_label(rep)}, models ordered by
        robustness under {C.condition_label(reference_condition)}. The first
        column is clean accuracy and is deliberately uncoloured: it is the
        quantity the rest are a fraction of. A grey square marked "not run" was
        never fitted; one marked "excluded" ran and was dropped, with the reason
        -- no clean level on its ladder, a clean accuracy under the gate, or
        fewer than three levels to integrate -- in the excluded table. Only
        the noise conditions the whole roster ran are here; the ones given to a
        named subset of models would leave a mostly empty column and are in
        R19, and censoring is on neither figure's axis. Every condition,
        including those, is in T4.""")
    written.append(S.save(fig, Path(output_dir) / 'F4c_robustness_grid.png'))
    return written[0] if written else None


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
    frame = C.cross_model(
        summary[(summary['rep'] == rep) & (summary['dataset'] != 'qm9')], 'F8')
    rankable = G.ranking_conditions(sorted(frame['condition'].dropna().unique()))
    frame = frame[frame['condition'].isin(rankable)]
    if not len(frame):
        return None
    title = G.declare(frame, 'F8', fixed={'rep': rep},
                      varies=('dataset', 'model', 'condition'))

    wide, thin = G.full_roster_conditions(frame, where='F8')
    frame = frame[frame['condition'].isin(wide)] if wide else frame
    conditions_thin = thin
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
        compared directly, printed on the colour bar. The assay datasets and QM9
        use different ranges because their robustness spans differ by a factor
        of three; panels within a figure and figures on the same datasets are
        directly comparable, a QM9 cell and an assay cell are not. A grey cell marked "not run" is a combination that
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
    longest = max([len('Clean R²')]
                  + [len(C.condition_label(c)) for c in conditions])
    layout = panel_layout(len(datasets), len(models), len(conditions) + 1, 'F8',
                          longest_label=longest)
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
            # A PANEL title, which is the kind that stays (the author,
            # 2026-09-17). When the datasets are split into separate files each
            # keeps its letter, so the three read as panels a, b and c of one
            # figure rather than as three unrelated grids. The representation is
            # in the caption, not repeated on every panel.
            S.title(ax, 'abc'[datasets.index(dataset)],
                    C.dataset_label(dataset))
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
    """Hold one noise condition and one representation. Plot every model. The bottom
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
    frame = C.cross_model(accuracy[(accuracy['dataset'] == dataset)
                                   & (accuracy['rep'] == rep)
                                   & (accuracy['condition'] == condition)],
                          'R15')
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

def r16_decoupling(summary, output_dir, rep, dataset='qm9',
                   condition='gaussian', conditions=None):
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
    # ONE POINT PER MODEL PER PANEL, AND ONE PANEL PER NOISE CONDITION. It used
    # to draw every condition into ONE panel, and clean accuracy does not depend
    # on the condition -- so each model came out as a vertical stack of
    # identical markers at one x, which reads as duplicates (the author,
    # 2026-09-14). Splitting them into panels keeps every point and lets the
    # reader see whether the decoupling holds under each condition, which
    # collapsing or averaging them both hide (the author, 2026-09-16).
    shown = [c for c in (conditions or [condition])
             if c in set(summary['condition'])]
    shown = C.sort_conditions(shown) or [condition]
    frame = C.cross_model(
        summary[(summary['dataset'] == dataset) & (summary['rep'] == rep)
                & (summary['condition'].isin(shown))],
        'R16')
    if not len(frame):
        return None
    title = G.declare(frame, 'R16', fixed={'dataset': dataset, 'rep': rep},
                      varies=('model', 'condition'))

    fig, axes = _fig(height=min(C.MAX_HEIGHT_IN, 3.6),
                     ncols=len(shown), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    models = C.sort_models(frame['model'].unique())
    notes = []
    for index, (ax, this) in enumerate(zip(axes, shown)):
        panel = frame[frame['condition'] == this]
        for model in models:
            one = panel[panel['model'] == model]
            if not len(one):
                continue
            ax.scatter(one['baseline_r2'], one['auc_norm'], s=50, alpha=0.75,
                       color=C.model_color(model),
                       marker=C.model_marker(model),
                       linewidth=0.4, edgecolor='white',
                       label=C.model_label(model) if index == 0 else None)
        rho_here, p_here = (stats.spearmanr(panel['baseline_r2'],
                                            panel['auc_norm'])
                            if len(panel) >= 4 else (np.nan, np.nan))
        notes.append(f'({"abcdef"[index]}) {C.condition_label(this)}: '
                     f'rho {rho_here:.2f}'
                     + ('' if not np.isfinite(p_here)
                        else f', p {p_here:.2g}')
                     + f', {len(panel)} models')
        ax.margins(x=0.10)
        ax.set_xlabel(f'Clean {G.metric_label("r2")}')
        ax.spines[['top', 'right']].set_visible(False)
        S.title(ax, 'abcdef'[index], C.condition_label(this))
        if index == 0:
            ax.set_ylabel(G.metric_label('auc_norm'))

    # Padded to the data, so the spread fills the panels instead of being
    # squeezed under a 1.0 reference line that is not there. Shared, so the
    # panels can be read against each other.
    values = frame['auc_norm'].dropna()
    if len(values):
        pad = max((values.max() - values.min()) * 0.15, 0.02)
        axes[0].set_ylim(values.min() - pad, values.max() + pad)
    ax = axes[0]
    ax.set_xlabel(f'Clean {G.metric_label("r2")} (no noise added)')
    ax.set_ylabel(G.metric_label('auc_norm'))
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).
    # NOTHING WRITTEN OVER THE POINTS. The correlations are in the caption, one
    # per panel, where they can be read without covering the data. The key gets
    # a wider band than the default: at thirteen models in four family columns
    # the entries were touching (the author, 2026-09-17).
    S.shared_legend(fig, axes[0], ncol=4, margin=0.30)
    caption('R16', f"""
        Clean accuracy against robustness on {title}, one panel per noise
        condition and one point per model in each. Bottom axis: R2 with no noise
        added. Side axis: the share of that accuracy the model keeps as noise
        rises, on one scale across the panels. Marker shape is the model.
        Correlation between the two axes, panel by panel: {'; '.join(notes)}.
        Read this beside the clean-accuracy axis rather than alone: the
        robustness metric divides the clean baseline out by construction, so a
        model with a weak baseline scores well by having less to lose, and part
        of any apparent decoupling is arithmetic rather than a finding.""")
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
    notes = []
    verdicts = {}
    if slopes is not None and len(slopes):
        for _, row in slopes.iterrows():
            if str(row.get('condition')) == condition and str(row.get('rep')) == rep:
                verdicts[str(row.get('model'))] = str(row.get('verdict', ''))

    # A SHAPE THAT DIVIDES. Six panels came out as four across the top and two
    # underneath, which reads as two figures. The columns are chosen so the rows
    # are equal where the count allows it: six becomes three by two.
    # A PANEL WITH ONE LINE IS NOT A COMPARISON. Where one of the two halves is
    # a single number per fit there is nothing to plot against anything, and the
    # panel was a line with a note beside it saying the other half is missing.
    # Those models come out of the figure and go into a sentence (the author,
    # 2026-09-13); the sentence is built below and the caption carries it.
    drawn_components = (frame.groupby('model')['component'].nunique()
                        if 'component' in frame.columns else None)
    single = ([m for m in models if drawn_components.get(m, 0) < 2]
              if drawn_components is not None else [])
    models = [m for m in models if m not in single]
    if not models:
        print('  F6: every model has only one component that varies per '
              'molecule, so there is nothing to draw. The support flags are '
              'in T6.')
        return None

    ncols = next((c for c in (3, 4, 2) if len(models) % c == 0 and len(models) // c <= 3),
                 min(len(models), 3))
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
        # THE NOTE GOES TO THE CAPTION. Text laid over the data is text a reader
        # has to work around; the caption is where a journal expects it and
        # where it can be read at its own pace. Collected here, printed there.
        if note:
            notes.append(f'({"abcdefgh"[index]}) {C.model_label(model)}: {note}')
        ax.spines[['top', 'right']].set_visible(False)
        if index % ncols == 0:
            ax.set_ylabel(f'Mean predicted uncertainty\n({C.dataset_unit(dataset)})')
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
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).
    S.shared_legend(fig, axes[0], ncol=2)

    # WRITTEN AFTER THE PANELS, because it carries what each panel found. Those
    # sentences used to be printed over the data.
    caption('F6', f"""
        The two halves of the predicted uncertainty against the amount of noise
        added to the training labels, one panel per model, on
        {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}. One mark is the mean predicted
        uncertainty over the molecules of one out-of-fold pass, in
        {C.dataset_unit(dataset)}; the band is the range across folds. The
        bottom axis is the noise level, a fraction of the clean training label
        spread. A model that is attributing added label noise to the data has a
        data line that climbs and a model line that holds. Every panel starts at
        zero, so a component that holds still looks like one. A component that
        is one number per fit rather than one per molecule is not drawn, and
        which those are is said here: {' '.join(notes) if notes else
        'every component shown varies per molecule.'}
        {('Not shown at all, because only one of their two halves varies per '
          'molecule and a panel of one line is not a comparison: '
          + ', '.join(C.model_label(m) for m in single) + '. Their support '
          'flags are in T6.') if single else ''}
        These readings come from the fitted slopes, not from the picture.""")
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
        # SEMI-TRANSPARENT, because six of these lie on top of each other and
        # an opaque line hides every line drawn before it -- a reader counts one
        # curve where six agree (the author, 2026-09-13).
        ax.plot(one['fraction'], one['value'], marker=C.model_marker(model),
                markersize=3.5, linewidth=1.4, alpha=0.55,
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
                              value='auc_norm', outliers=None):
    """A GRID, not a line chart -- the author's call, 2026-09-13.

    Models down the side, representations across the bottom, robustness on each
    square, and a heavy outline around every cell that sits outside the range of
    that model's other representations by more than the replicate spread.

    It was nineteen lines over six categories, and the two things it had to show
    -- which cells are outliers, and that most models are flat -- were both lost
    in the hairball. A line between two representations also draws a slope
    between two categories that have no order, so the eye reads a trend that
    cannot exist. The grid keeps the outlier marks, which are the finding (D5),
    and drops the false slope.
    """
    frame = C.cross_model(summary[(summary['dataset'] == dataset)
                                  & (summary['condition'] == condition)], 'R6')
    if not len(frame):
        return None
    G.declare(frame, 'R6',
              fixed={'dataset': dataset, 'condition': condition},
              varies=('model', 'rep'))
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    models = C.sort_models(frame['model'].unique())

    marked = set()
    if outliers is not None and len(outliers):
        flagged = outliers[(outliers['condition'] == condition)
                           & outliers.get('is_outlier', False)]
        marked = set(map(tuple, flagged[['model', 'rep']].to_numpy()))

    fig, ax = _fig(height=C.grid_height(len(models)))
    lo, hi = C.auc_range(dataset)
    image, table = S.grid(ax, frame, 'model', 'rep', value,
                          row_order=models, column_order=reps,
                          column_labeller=C.rep_label, vmin=lo, vmax=hi)
    # The outline is the point of the figure: one cell of a model's row that its
    # other representations do not reach.
    import matplotlib.patches as mpatches
    drawn = 0
    for i, model in enumerate(table.index):
        for j, representation in enumerate(table.columns):
            if (model, representation) not in marked:
                continue
            ax.add_patch(mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#111111',
                linewidth=2.0, zorder=4))
            drawn += 1
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(G.metric_label(value), fontsize=8)
    bar.ax.tick_params(labelsize=7)
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).

    caption('R6', f"""
        Robustness ({G.metric_label(value)}) of every model at every
        representation, on {C.dataset_label(dataset)} under
        {C.condition_label(condition)}. Rows are models, columns are the six
        representations, and the number on each square is the median over
        replicates. {drawn} square(s) are outlined: those are the cells that sit
        outside the range of that model's other representations by more than the
        spread across replicates, which is what a model-representation pairing
        looks like when it is real rather than noise. A model whose row is one
        flat colour is a model that does not care which representation it is
        given.""")
    return S.save(fig,
                  Path(output_dir) / f'R6_representation_profile_{condition}.png')


# ---------------------------------------------------------------------------
# F9 -- can the uncertainty find the corrupted labels (the author, 2026-09-14)
# ---------------------------------------------------------------------------

def f9_uncertainty_finds_noise(q4, output_dir, rep, condition, dataset='qm9',
                               sigma=None, statistic='rho_ratio'):
    """One bar per model: how well the predicted uncertainty tracks the noise.

    SHAPE D. Bottom axis: the models. Side axis: the Spearman correlation
    between what the model said it was unsure about and how much noise was
    actually put into each label. One bar per model, and behind each bar the
    grey band the permutation null covers -- the range the same statistic takes
    when the noise is shuffled. A bar that clears its band is a model whose
    uncertainty found the corrupted labels.

    This replaces F7, which drew the same result as a lift curve with an axis
    nobody asked for. D7 fires on the same numbers; this is the picture of it.
    The paper's title is about uncertainty and this is its headline question.
    """
    if q4 is None or not len(q4):
        return None
    frame = q4[(q4['dataset'].map(C.canonical_dataset) == dataset)
               & (q4['rep'] == rep) & (q4['condition'] == condition)]
    if statistic not in frame.columns:
        return None
    frame = frame.dropna(subset=[statistic])
    if not len(frame):
        return None
    frame, sigma = _one_level(frame, sigma)
    if not len(frame):
        return None
    G.declare(frame, 'F9',
              fixed={'dataset': dataset, 'rep': rep, 'condition': condition,
                     'sigma': sigma},
              varies=('model',), aggregates=('fold',))

    per = (frame.groupby('model', dropna=False)
           .agg(value=(statistic, 'median'),
                lo=('null_lo', 'median'), hi=('null_hi', 'median'),
                outside=('outside_null', 'mean'), folds=(statistic, 'size'))
           .reset_index())
    per = per.reindex(index=[i for m in C.sort_models(per['model'])
                             for i in per.index[per['model'] == m]])
    if not len(per):
        return None

    fig, ax = _fig(height=3.6)
    x = np.arange(len(per))
    # The null band FIRST and behind, so a bar is read against it.
    for i, row in enumerate(per.itertuples()):
        if np.isfinite(row.lo) and np.isfinite(row.hi):
            ax.add_patch(plt_rect(i - 0.44, row.lo, 0.88, row.hi - row.lo))
    clears = per['outside'] >= 0.5
    ax.bar(x, per['value'], width=0.66, zorder=3,
           color=[C.model_color(m) for m in per['model']],
           alpha=0.9,
           edgecolor=['#111111' if c else '#BBBBBB' for c in clears],
           linewidth=[1.1 if c else 0.5 for c in clears])
    ax.axhline(0, color='#444444', linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([C.model_label(m) for m in per['model']], rotation=35,
                       ha='right', fontsize=8)
    ax.set_ylabel('Correlation between estimated\nuncertainty and label noise')
    ax.set_title(f'{C.dataset_label(dataset)}, {C.rep_label(rep)}, '
                 f'{C.condition_label(condition)}, noise level {sigma}',
                 fontweight='bold', fontsize=9, loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Patch(facecolor='#DDDDDD', edgecolor='none',
              label='what shuffling the noise gives'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#888888',
               markeredgecolor='#111111', markeredgewidth=1.1, markersize=8,
               label='clears its band'),
    ], loc='lower center', ncol=2, frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, -0.02))

    n_clear = int(clears.sum())
    caption('F9', f"""
        Whether a model's own uncertainty points at the labels that were
        corrupted, on {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}, at noise level {sigma}. One bar per
        model: the Spearman correlation between the uncertainty the model
        predicted for each held-out molecule and the amount of noise actually
        added to that molecule's training label, divided by the model's own
        out-of-fold error so that the comparison is against what the error
        already tells you. The grey band behind each bar is what the same
        statistic gives when the noise is shuffled and the error recomputed from
        the shuffled noise, over the permutations; a bar that clears its band is
        a model that found something the error alone does not.
        {n_clear} of {len(per)} models clear it. Bars are the median over the
        out-of-fold folds.""")
    return S.save(fig, Path(output_dir) /
                  f'F9_uncertainty_finds_noise_{rep}_{condition}.png')


def plt_rect(x, y, width, height):
    """A grey null band, drawn behind the bars."""
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), width, height, facecolor='#DDDDDD',
                     edgecolor='none', zorder=1)


# ---------------------------------------------------------------------------
# R17 -- each model against its own variant (recovered from paper.tex,
# fig_nn_family_comparison.png, 2026-09-13)
# ---------------------------------------------------------------------------

#: The three families, each as a list of models that belong on ONE panel. The
#: author's shape, 2026-09-13: the alpha architecture with its Bayesian version
#: and its variance-head version, the same for beta, and the two forests. The
#: Gaussian process has no deterministic counterpart, so it has no panel -- a
#: panel of one line is not a comparison.
VARIANT_FAMILIES = [
    ('NN-α family', ['dnn', 'dnn_bnn_full', 'dnn_bnn_full_mve']),
    ('NN-β family', ['mlp', 'mlp_bnn_full', 'mlp_bnn_full_mve']),
    ('Forests', ['rf', 'qrf']),
]

#: One colour per ROLE, shared across the two neural panels so the same kind of
#: model is the same colour in both, and its own pair for the forests. Solid
#: lines throughout -- a dashed line and a solid line of the same colour were
#: not separable in the key (the author, 2026-09-13).
FAMILY_ROLE_COLORS = {
    'dnn': '#0072B2', 'mlp': '#0072B2',
    'dnn_bnn_full': '#D55E00', 'mlp_bnn_full': '#D55E00',
    'dnn_bnn_full_mve': '#009E73', 'mlp_bnn_full_mve': '#009E73',
    'rf': '#CC79A7', 'qrf': '#7B3294',
}


def r17_variant_families(accuracy, output_dir, rep, dataset='qm9',
                         condition='gaussian', families=None):
    """Three panels: does making a model probabilistic change how noise hurts it?

    Panel one is the alpha architecture -- the plain network, its Bayesian
    version, and the Bayesian version with a variance head. Panel two is the
    same for beta. Panel three is RF against QRF. One line per model, solid, one
    colour per role, so the same kind of model is the same colour in both neural
    panels.

    The Gaussian process is not here: it has no deterministic counterpart in the
    roster, so there is nothing to compare it against.
    """
    frame = accuracy[(accuracy['dataset'] == dataset)
                     & (accuracy['rep'] == rep)
                     & (accuracy['condition'] == condition)]
    if not len(frame):
        return None
    have = set(frame['model'].unique())
    use = [(name, [m for m in members if m in have])
           for name, members in (families or VARIANT_FAMILIES)]
    use = [(name, members) for name, members in use if len(members) >= 2]
    if not use:
        return None
    G.declare(frame, 'R17',
              fixed={'dataset': dataset, 'rep': rep, 'condition': condition},
              varies=('model', 'sigma'), aggregates=('replicate',))

    fig, axes = _fig(height=3.2, nrows=1, ncols=len(use), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for index, ((name, members), ax) in enumerate(zip(use, axes)):
        for model in members:
            one = (frame[frame['model'] == model]
                   .groupby('sigma', as_index=False)['r2'].median()
                   .sort_values('sigma'))
            if not len(one):
                continue
            ax.plot(one['sigma'], one['r2'], marker='o', markersize=3.5,
                    linewidth=1.6,
                    color=FAMILY_ROLE_COLORS.get(model, C.model_color(model)),
                    label=C.model_label(model))
        ax.legend(loc='lower left', fontsize=7, frameon=False,
                  handlelength=1.6, labelspacing=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        S.title(ax, 'abc'[index], name)
        if index == 0:
            ax.set_ylabel(G.metric_label('r2'))
    # ONE bottom-axis label under the middle panel. Three copies of the same
    # words across three panels ran into each other and said nothing three
    # times (the author, 2026-09-14).
    axes[len(use) // 2].set_xlabel(LEVEL_AXIS)
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).

    caption('R17', f"""
        Does making a model probabilistic change how label noise hurts it, on
        {C.dataset_label(dataset)} at {C.rep_label(rep)} under
        {C.condition_label(condition)}. Bottom axis: the amount of noise put
        into the training labels, as a fraction of the clean training label
        spread. Side axis: R2 on held-out molecules, median over the ten
        replicates, shared across the panels. (a) the first neural architecture
        as a plain network, as a Bayesian network, and as a Bayesian network
        with a variance head; (b) the same three for the second architecture;
        (c) the random forest against the quantile forest. Lines that stay
        together mean the probabilistic version neither gains nor loses
        robustness. The Gaussian process has no deterministic counterpart in the
        roster and so has no panel here.""")
    return S.save(fig, Path(output_dir) /
                  f'R17_variant_families_{rep}_{condition}.png')


# ---------------------------------------------------------------------------

def r18_representation_against_representation(summary, output_dir, a, b,
                                              dataset='qm9',
                                              condition='gaussian',
                                              value='auc_norm'):
    """Robustness on one representation against robustness on another.

    One point per model, both axes the same quantity, a diagonal for equality
    and a Spearman correlation. This is NOT R16 -- R16 is clean accuracy against
    robustness, one representation at a time. This asks whether a model's
    robustness carries from one representation to the next, which is the
    compact form of the evidence behind the open choice of which representation
    the main-text tables hold fixed (RERUN_PLAN.md 14.9 item 5).

    NOTHING IS AVERAGED OVER REPRESENTATIONS here: each axis is one named
    representation and the pairing is on the model.
    """
    frame = C.cross_model(summary[(summary['dataset'] == dataset)
                                  & (summary['condition'] == condition)], 'R18')
    left = frame[frame['rep'] == a].set_index('model')[value]
    right = frame[frame['rep'] == b].set_index('model')[value]
    shared = left.index.intersection(right.index)
    if len(shared) < 4:
        return None
    x, y = left.loc[shared].astype(float), right.loc[shared].astype(float)
    G.declare(frame[frame['rep'].isin([a, b])], 'R18',
              fixed={'dataset': dataset, 'condition': condition},
              varies=('model', 'rep'))
    rho, p_value = stats.spearmanr(x, y)

    fig, ax = _fig(height=4.2)
    for model in shared:
        ax.scatter(float(left[model]), float(right[model]), s=60, alpha=0.85,
                   marker=C.model_marker(model), color=C.model_color(model),
                   linewidth=0.4, edgecolor='white', zorder=3,
                   label=C.model_label(model))
    lo = float(min(x.min(), y.min())) - 0.02
    hi = float(max(x.max(), y.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], color='#777777', linestyle=':', linewidth=1.0,
            zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f'{G.metric_label(value)} on {C.rep_label(a)}')
    ax.set_ylabel(f'{G.metric_label(value)} on {C.rep_label(b)}')
    # NOTHING WRITTEN ON THE PANEL. The correlation and the reading of the
    # diagonal are in the caption (the author, 2026-09-13); see RERUN_PLAN.md
    # 14.11k for whether the number belongs in the figure at all.
    # NO FIGURE TITLE. A journal caption carries what this is; a title
    # above it says the same thing twice (the author, 2026-09-17).
    ax.spines[['top', 'right']].set_visible(False)
    S.shared_legend(fig, ax, ncol=4)

    caption('R18', f"""
        Robustness ({G.metric_label(value)}) on {C.rep_label(a)} against the
        same quantity on {C.rep_label(b)}, one point per model, on
        {C.dataset_label(dataset)} under {C.condition_label(condition)}. The
        dotted diagonal is equal robustness on both: a point above it is a
        model that holds up better on {C.rep_label(b)}, a point below it one
        that holds up better on {C.rep_label(a)}. Spearman ρ = {rho:.2f} over
        {len(shared)} models: how far a model's robustness carries from one
        representation to the other, which is what choosing one representation
        for the main-text tables costs. Nothing here is averaged over
        representations -- each axis is one named representation and the points
        are paired on the model.""")
    return S.save(fig, Path(output_dir) /
                  f'R18_{a}_against_{b}_{condition}.png')


# ---------------------------------------------------------------------------
# R19 -- the deep-run conditions, over the pairs that actually ran
# ---------------------------------------------------------------------------

def r19_deep_conditions(summary, output_dir, conditions, dataset='qm9',
                        value='auc_norm'):
    """The conditions the main grid cannot hold, on the pairs they were run on.

    Student-t, Outlier and Laplace run on a NAMED SUBSET of model-and-
    representation pairs -- six models across three representations, not the
    full cross. In a nineteen-row heatmap their column is mostly grey and no
    queued task will ever fill it. Here the rows are only the models that were
    run, so every cell carries a number, and Gaussian sits beside them as the
    reference the deep conditions are read against.

    The author's call, 2026-09-13: fill the grey cells or take the columns out.
    They cannot be filled, so they come out and land here.
    """
    if summary is None or not len(summary) or not conditions:
        return None
    # NOT cross_model. That filter drops the variant models because they
    # cannot stand as independent architectures in a CROSS-MODEL comparison,
    # and it was silently taking three of the seven models the deep run was
    # chosen to include -- GP (het.), BNN-Full-MVE and VBLL-Full-Hetero -- so
    # the figure drew four rows while its caption reported those four as
    # everything that ran. This grid is the deep run's own roster, named in
    # deep_run_pairs.json, not a cross-model ranking (the author, 2026-09-23).
    frame = summary[summary['dataset'] == dataset]
    show = ['gaussian'] + [c for c in C.sort_conditions(conditions)
                           if c != 'gaussian']
    frame = frame[frame['condition'].isin(show)]
    if not len(frame):
        return None
    # Only the pairs the deep conditions were actually run on -- otherwise the
    # Gaussian reference column is nineteen rows against their six.
    deep = frame[frame['condition'].isin([c for c in show if c != 'gaussian'])]
    pairs = set(map(tuple, deep[['model', 'rep']].drop_duplicates().to_numpy()))
    if not pairs:
        return None
    frame = frame[[(m, r) in pairs for m, r
                   in zip(frame['model'], frame['rep'])]]
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'])]
    models = C.sort_models(frame['model'].unique())
    G.declare(frame, 'R19', fixed={'dataset': dataset},
              varies=('model', 'rep', 'condition'))

    # ONE COLUMN LIST FOR EVERY PANEL. It used to be derived per panel, from
    # the conditions THAT panel had, while `sharex=True` gave all three one
    # x-axis -- so a panel with five conditions drew five columns onto an axis
    # scaled for the four of the last panel, and the fifth column's numbers
    # landed outside the grid and on top of the colour bar (the author,
    # 2026-09-18). Same failure as the seven-group panel over a six-group axis
    # in F2. A condition a panel does not have is now a grey cell, which is what
    # it is.
    columns = [c for c in show if c in set(frame['condition'])]
    lo, hi = C.auc_range(dataset)
    fig, axes = _fig(height=C.grid_height(len(models), len(reps)),
                     nrows=len(reps), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    image = None
    for index, (ax, rep) in enumerate(zip(axes, reps)):
        panel = frame[frame['rep'] == rep]
        image, _ = S.grid(ax, panel, 'model', 'condition', value,
                          row_order=models, column_order=columns,
                          vmin=lo, vmax=hi)
        S.title(ax, 'abcdef'[index], C.rep_label(rep))
    if image is not None:
        # More room between the panels and the key. At pad 0.02 the colour bar
        # sat on the rightmost column's labels (the author, 2026-09-17).
        bar = fig.colorbar(image, ax=list(axes), fraction=0.024, pad=0.055)
        bar.set_label(G.metric_label(value), fontsize=8, labelpad=8)
        bar.ax.tick_params(labelsize=7)

    named = ', '.join(C.condition_label(c) for c in show if c != 'gaussian')
    roster = ', '.join(C.model_label(m) for m in models)
    caption('R19', f"""
        The noise conditions that run on a named subset of models and
        representations rather than on the whole roster: {named}, on
        {C.dataset_label(dataset)}. One panel per representation. The rows are
        the models these conditions were run on -- {roster} -- chosen before
        the run to cover one model from each family, and Gaussian is the left
        column, as the reference the rest are read against. Every value is the
        median over replicates. These conditions are absent from the main
        robustness grid because their column there would be mostly empty by
        design, not because anything is still running. Censoring is here rather
        than on the curves because its level is a fraction of labels clipped
        rather than a fraction of the label spread, so it shares no bottom axis
        with the others; a grey square marked "not run" is a pair it was never
        given.""")
    return S.save(fig, Path(output_dir) / f'R19_deep_conditions_{dataset}.png')


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
    frame = C.cross_model(transfer[(transfer['rep'] == rep)
                                   & (transfer['condition'] == condition)],
                          'R9')
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

# ---------------------------------------------------------------------------
# Two findings that are sentences, not figures (the author, 2026-09-13)
# ---------------------------------------------------------------------------

def sentence_auc_above_one(per_replicate, dataset='qm9'):
    """Which cells retain more than they started with, in one sentence.

    R10 drew this as eight marked points among eight thousand grey ones. The
    picture named nothing; the sentence names every cell.
    """
    if per_replicate is None or not len(per_replicate):
        return ''
    frame = per_replicate[per_replicate['dataset'] == dataset]
    if not len(frame) or 'baseline_r2' not in frame.columns:
        return ''
    high = frame[frame['auc_norm'] > C.AUC_NORM_IMPLAUSIBLE_HIGH]
    if not len(high):
        return (f'No replicate on {C.dataset_label(dataset)} scored above '
                f'{C.AUC_NORM_IMPLAUSIBLE_HIGH} -- no model retained more '
                f'accuracy under noise than it had without it.')
    named = []
    for key, group in high.groupby(['model', 'rep', 'condition'], dropna=False):
        model, representation, condition = key
        named.append(
            f'{C.model_label(model)} on {C.rep_label(representation)} under '
            f'{C.condition_label(condition)} '
            f'({len(group)} of its replicates, clean R2 '
            f'{group["baseline_r2"].median():.2f}, highest '
            f'{group["auc_norm"].max():.2f})')
    return (f'{len(high)} of {len(frame)} replicate values on '
            f'{C.dataset_label(dataset)} scored above '
            f'{C.AUC_NORM_IMPLAUSIBLE_HIGH}, meaning the model retained more '
            f'accuracy with noise added than it had without it. Every one '
            f'began from a low clean baseline, which is what the ratio does '
            f'when its denominator is small: '
            + '; '.join(named) + '. Reported rather than patched.')


def sentence_rank_by_rep(accuracy, model, condition, dataset='qm9',
                         baseline_gate=None):
    """Whether the representations change places as the noise rises.

    R15b drew six nearly flat lines to say "they do not". This says it, and
    names the one that moves.
    """
    gate = C.BASELINE_THRESHOLD if baseline_gate is None else baseline_gate
    frame = accuracy[(accuracy['dataset'] == dataset)
                     & (accuracy['model'] == model)
                     & (accuracy['condition'] == condition)]
    if not len(frame):
        return ''
    ranks = (frame.groupby(['rep', 'sigma'], dropna=False)['r2'].median()
             .reset_index())
    ranks = ranks[ranks['r2'] >= gate]
    if not len(ranks):
        return ''
    wide = ranks.pivot_table(index='rep', columns='sigma', values='r2')
    order = wide.rank(ascending=False, axis=0)
    lo, hi = float(ranks['sigma'].min()), float(ranks['sigma'].max())
    if lo not in order.columns or hi not in order.columns:
        return ''
    moved = (order[hi] - order[lo])
    movers = moved[moved.abs() >= 1].sort_values()
    steady = [C.rep_label(r) for r in moved[moved.abs() < 1].index]
    if not len(movers):
        return (f'For {C.model_label(model)} on {C.dataset_label(dataset)} '
                f'under {C.condition_label(condition)}, the six '
                f'representations hold their order from clean labels to noise '
                f'level {hi:g}: none changes place.')
    named = '; '.join(
        f'{C.rep_label(r)} moves {abs(int(round(d)))} place(s) '
        f'{"up" if d < 0 else "down"}' for r, d in movers.items())
    return (f'For {C.model_label(model)} on {C.dataset_label(dataset)} under '
            f'{C.condition_label(condition)}, the representations hold their '
            f'order from clean labels to noise level {hi:g} with '
            f'{len(movers)} exception(s) -- {named}. '
            + (f'The other {len(steady)} ({", ".join(steady)}) do not change '
               f'place.' if steady else ''))


def r15b_rank_against_level_by_rep(accuracy, output_dir, model, condition,
                                   dataset='qm9', baseline_gate=None):
    """The mirror the author asked for: hold one noise condition and one MODEL, and
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


# ---------------------------------------------------------------------------
# F4d -- F4b's curves, on every representation instead of one
# ---------------------------------------------------------------------------

def f4d_conditions_by_representation(accuracy, output_dir, dataset='qm9',
                                     models=None):
    """One panel per representation, one line per noise condition.

    F4b answers "does the KIND of noise matter" for one model on one
    representation, and that single panel was the only drawing of the curves
    that existed. Whether a condition costs more early or late is a property
    of the pairing, exactly as F4a's two panels show for the models -- so the
    answer from one representation cannot stand for the other five (the
    author, 2026-09-23).

    A check figure, not a paper figure: it is here so the shapes can be looked
    at, and what it shows decides what the paper carries. One file per model.
    Censoring is absent for the same reason it is absent from F4b -- its level
    is a fraction of labels clipped, not a fraction of the label spread, so it
    does not share this bottom axis.
    """
    if accuracy is None or not len(accuracy):
        return []
    frame = accuracy[accuracy['dataset'] == dataset]
    rankable = G.ranking_conditions(sorted(frame['condition'].dropna().unique()))
    frame = frame[frame['condition'].isin(rankable)]
    if not len(frame):
        return []

    if models is None:
        models = [C.DEFAULT_FOCUS_MODEL] if C.DEFAULT_FOCUS_MODEL in set(
            frame['model']) else sorted(frame['model'].unique())[:1]
    reps = [r for r in C.REP_LABELS if r in set(frame['rep'].dropna())]
    if not reps:
        return []

    written = []
    for model in models:
        here = frame[frame['model'] == model]
        if not len(here):
            continue
        # A representation this model never ran on would draw an empty panel
        # under a letter, which reads as a finding rather than a gap.
        panes = [r for r in reps if len(here[here['rep'] == r])]
        if not panes:
            continue
        columns = min(3, len(panes))
        rows = int(np.ceil(len(panes) / columns))
        G.declare(here, 'F4d', fixed={'dataset': dataset, 'model': model},
                  varies=('rep', 'condition', 'sigma'),
                  aggregates=('replicate',))
        fig, axes = _fig(height=2.6 * rows, nrows=rows, ncols=columns,
                         sharey=True, sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for index, (ax, held) in enumerate(zip(axes, panes)):
            curves = (here[here['rep'] == held]
                      .groupby(['condition', 'sigma'], as_index=False)['r2']
                      .median())
            S.line_chart(ax, curves, 'sigma', 'r2', 'condition',
                         labeller=C.condition_label,
                         colours=C.CONDITION_COLORS, legend=False)
            ax.spines[['top', 'right']].set_visible(False)
            S.title(ax, 'abcdef'[index], C.rep_label(held))
            if index % columns == 0:
                ax.set_ylabel(G.metric_label('r2'))
            if index >= len(panes) - columns:
                ax.set_xlabel(LEVEL_AXIS)
        for spare in axes[len(panes):]:
            spare.set_visible(False)
        S.shared_legend(fig, axes[0], side=True)
        caption('F4d', f"""
            Whether the KIND of noise matters or only the amount, for
            {C.model_label(model)} on {C.dataset_label(dataset)}, one panel per
            representation. One line per noise condition, each the median over
            replicates, all panels sharing a side axis so the panels can be
            read against each other. Bottom axis: the amount of noise put into
            the training labels, as a fraction of the clean training label
            spread. Lines that lie on top of each other mean this pairing
            cannot tell the noise conditions apart at a matched dose; lines
            that separate mean the shape of the noise costs something beyond
            its size, and WHERE they separate is what the area under the curve
            cannot tell you. Censoring is absent: its level is a fraction of
            labels clipped rather than a fraction of the label spread, so it
            does not share this bottom axis. The numbers behind every line are
            in r2_by_level.csv.""")
        written.append(S.save(
            fig, Path(output_dir) /
            f'F4d_conditions_by_representation_{model}.png'))
    return written
