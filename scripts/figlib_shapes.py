#!/usr/bin/env python
"""Four shapes, and every figure in the paper is one of them.

RERUN_PLAN.md 14.4. The old script had no saving helper, no error-band helper
and no heatmap wrapper: the same seaborn call was written out four times with
three different colour ranges, and every figure repeated the same three lines to
save itself. Learning four shapes once is the whole visual vocabulary.

  SHAPE A -- the line chart.
      Bottom axis is the amount of noise, left to right from none to the most.
      Side axis is a score. One mark is a dot; dots belonging to the same thing
      are joined left to right by a line. Colour tells you which thing each line
      is.

  SHAPE B -- the dot-per-row chart.
      The models run down the side, one row each. A score runs along the bottom.
      One mark is a dot sitting on a model's row. When a row carries several
      dots, colour tells you what each dot is. A vertical line marks a reference
      value.

  SHAPE C -- the grid of squares.
      Rows are one factor, columns are another. Every square is one combination.
      The number is printed on the square and the colour repeats it, dark for
      low and bright for high. Squares that were never run are left blank and
      labelled.

  SHAPE D -- the bar chart.
      Categories along the bottom, a quantity up the side, one bar per category.
      Where bars are grouped, colour tells you which member of the group.

Every one of these takes an axes and returns nothing. The figure builders own
the layout; these own the drawing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402


def save(fig, path, dpi=300):
    """One place figures are written, so they cannot drift in size or format.

    Journal of Cheminformatics: 170 mm full width, 225 mm maximum height. The
    height is checked rather than trusted -- a grid that grows a row per model
    passes 225 mm without anyone noticing, and the journal rejects it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = fig.get_size_inches()
    if height > C.MAX_HEIGHT_IN + 0.01:
        print(f'  WARNING: {path.name} is {height * 25.4:.0f} mm tall; the '
              f'journal allows {C.MAX_HEIGHT_IN * 25.4:.0f} mm. Split it or '
              f'drop rows.')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'    wrote {path.name} ({width * 25.4:.0f} x {height * 25.4:.0f} mm)')
    return path


#: Reserve this much of the figure height for a legend placed below it.
#: `plt.tight_layout(rect=[0, LEGEND_MARGIN, 1, 1])` in the old script, with the
#: comment that 6% overlapped the x-axis ticks and the axis title.
LEGEND_MARGIN = 0.09


def shared_legend(fig, sources, ncol=4, extra=None, margin=None):
    """ONE legend, below the whole figure, reading left to right.

    Per-panel legends inside the axes cover the data -- with nineteen models
    they cover most of it -- and repeating the same models in three panels eats
    the plot area three times. The old script says so in as many words: "single
    shared legend below all three panels (per-panel legends were redundant and
    ate plot area)".

    matplotlib fills a multi-column legend column-major, so the handles are
    permuted first or the list reads down instead of across.
    """
    handles, labels = [], []
    for ax in (sources if isinstance(sources, (list, tuple)) else [sources]):
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if extra:
        handles += [h for h, _ in extra]
        labels += [l for _, l in extra]
    if not handles:
        return None
    ncol = min(ncol, len(labels))
    rows = -(-len(handles) // ncol) if ncol > 1 else len(handles)

    def colmajor(items):
        if ncol <= 1:
            return list(items)
        out = []
        for c in range(ncol):
            for r in range(rows):
                i = r * ncol + c
                if i < len(items):
                    out.append(items[i])
        return out

    legend = fig.legend(colmajor(handles), colmajor(labels), loc='lower center',
                        bbox_to_anchor=(0.5, 0.005), ncol=ncol, frameon=False,
                        fontsize=8, columnspacing=1.2, handletextpad=0.4)
    # Reserve exactly the band the legend occupies, and no more. Reserving a
    # fixed fraction per row left a hand's width of white between the axes and
    # a five-row key.
    used = margin if margin is not None else 0.032 * max(rows, 1) + 0.03
    fig.tight_layout(rect=[0, min(used, 0.34), 1, 1])
    return legend


def stat_box(ax, text, x=0.03, y=0.95):
    """A statistic printed ON a panel, in a box so it stays readable over data."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85,
                      edgecolor='#CCCCCC'))


def _order_legend(ax, ncol, **kwargs):
    """matplotlib fills legends column-major, which scrambles a logically
    ordered list. This restores reading order."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    n = len(handles)
    rows = -(-n // ncol) if ncol > 1 else n

    def colmajor(items):
        if ncol <= 1:
            return list(items)
        out = []
        for c in range(ncol):
            for r in range(rows):
                i = r * ncol + c
                if i < n:
                    out.append(items[i])
        return out

    return ax.legend(colmajor(handles), colmajor(labels), ncol=ncol, **kwargs)


# ---------------------------------------------------------------------------
# SHAPE A -- the line chart
# ---------------------------------------------------------------------------

def line_chart(ax, frame, x, y, series, spread=None, labeller=None,
               colours=None, markers=None, reference_x=None,
               reference_label=None, legend_ncol=3, legend=True):
    """One line per level of `series`, over `x`.

    `spread` names a column holding the half-height of a shaded band around each
    line -- the run-to-run spread. Drawn as a band rather than error bars
    because seven levels of bars on nineteen models is unreadable, and the point
    is the shape of the curve, not the value at one level.
    """
    labeller = labeller or (lambda v: str(v))
    # observed=True: `series` is a Categorical wherever a caller fixed the
    # line order, and without it pandas walks every unused category and
    # warns. Only categories actually in the frame are drawn.
    for name, group in frame.groupby(series, dropna=False, sort=False,
                                     observed=True):
        group = group.sort_values(x)
        colour = (colours or {}).get(name, C.model_color(name))
        # A marker per model, not a colour alone. Variants of one family SHARE a
        # colour on purpose, so with nineteen models the marker is what tells
        # two lines apart -- and it survives being printed in greyscale.
        marker = (markers or {}).get(name, C.model_marker(name))
        ax.plot(group[x], group[y], marker=marker, markersize=4,
                linewidth=1.4, alpha=0.85, color=colour, label=labeller(name))
        if spread and spread in group.columns:
            band = group[spread].to_numpy(dtype=float)
            ax.fill_between(group[x], group[y] - band, group[y] + band,
                            color=colour, alpha=0.15, linewidth=0)
    if reference_x is not None:
        ax.axvline(reference_x, color='#444444', linestyle='--', linewidth=0.9,
                   zorder=0)
        if reference_label:
            # INSIDE the axes. Above them it collided with the panel title,
            # which is the one place a reader looks first.
            ax.annotate(reference_label, xy=(reference_x, 0.03),
                        xycoords=('data', 'axes fraction'), ha='right',
                        va='bottom', fontsize=6.5, color='#444444',
                        rotation=90, xytext=(-3, 0), textcoords='offset points')
    if legend:
        _order_legend(ax, legend_ncol, loc='best', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)


# ---------------------------------------------------------------------------
# SHAPE B -- the dot-per-row chart
# ---------------------------------------------------------------------------

def dot_rows(ax, frame, row, value, series=None, labeller=None,
             row_labeller=None, colours=None, reference=None,
             reference_label=None, spread_low=None, spread_high=None,
             legend_ncol=3):
    """One row per level of `row`; one dot per level of `series` on that row."""
    labeller = labeller or (lambda v: str(v))
    row_labeller = row_labeller or C.model_label
    rows = list(dict.fromkeys(frame[row]))
    position = {name: i for i, name in enumerate(rows)}

    if series is None:
        ax.scatter(frame[value], [position[r] for r in frame[row]],
                   s=50, alpha=0.7, color=C.CLEAN_COLOR, zorder=3)
        if spread_low and spread_high:
            for _, r in frame.iterrows():
                ax.plot([r[spread_low], r[spread_high]],
                        [position[r[row]]] * 2, color=C.CLEAN_COLOR,
                        linewidth=1.2, alpha=0.5, zorder=2)
    else:
        for name, group in frame.groupby(series, dropna=False, sort=False,
                                         observed=True):
            ax.scatter(group[value], [position[r] for r in group[row]], s=50,
                       alpha=0.7, label=labeller(name), zorder=3,
                       color=(colours or {}).get(name, C.model_color(name)))
        _order_legend(ax, legend_ncol, loc='best', fontsize=8)

    if reference is not None:
        ax.axvline(reference, color='#444444', linestyle='--', linewidth=0.9,
                   zorder=1)
        if reference_label:
            ax.annotate(reference_label, xy=(reference, 1.005),
                        xycoords=('data', 'axes fraction'), ha='center',
                        va='bottom', fontsize=8, color='#444444')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row_labeller(r) for r in rows])
    ax.invert_yaxis()
    ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


# ---------------------------------------------------------------------------
# SHAPE C -- the grid of squares
# ---------------------------------------------------------------------------

def grid(ax, frame, rows, columns, value, row_labeller=None,
         column_labeller=None, fmt='{:.2f}', vmin=None, vmax=None,
         cmap='viridis', separate_first_column=False, row_order=None,
         column_order=None, never_run='not run', annotate=True):
    """A square per combination, the number printed on it.

    `vmin` and `vmax` are REQUIRED in practice: pass the fixed anchor for the
    quantity (figlib_config.AUC_RANGE_*), never the panel's own range. Two
    panels scaled to their own data make the same colour mean different numbers,
    and a reader compares panels by colour long before reading a value.

    A combination that was never run is grey and labelled, so an absence cannot
    be read as a low value -- the colour map's own dark end is a low value.

    `separate_first_column` draws a gutter after the first column, which is how
    the clean baseline sits beside the robustness grid without being mistaken
    for another noise condition.
    """
    row_labeller = row_labeller or C.model_label
    column_labeller = column_labeller or C.condition_label

    table = frame.pivot_table(index=rows, columns=columns, values=value,
                              aggfunc='median')
    if row_order:
        table = table.reindex([r for r in row_order if r in table.index])
    if column_order:
        table = table.reindex(
            columns=[c for c in column_order if c in table.columns])

    data = table.to_numpy(dtype=float)
    if vmin is None or vmax is None:
        raise ValueError(
            'grid() needs an explicit vmin and vmax. A panel scaled to its own '
            'data makes the same colour mean a different number in the panel '
            'beside it; use the fixed anchor for the quantity.')
    # A REFERENCE COLUMN is a different quantity and must not share the colour
    # scale. Clean accuracy runs 0.45-0.61 while retention runs 0.67-0.94, so
    # putting both on one scale paints the whole reference column in the map's
    # dark end -- and a reader sees a black column and concludes every model is
    # terrible. It is drawn uncoloured, with black text, so it reads as what it
    # is: the thing the other columns are a fraction OF.
    coloured = data.copy()
    if separate_first_column and data.shape[1] > 1:
        coloured[:, 0] = np.nan

    # Clipping is fine; clipping SILENTLY is not. A cell outside the shared
    # range is drawn at the end of the map, and how many and how far is said out
    # loud so nobody reads a saturated cell as an ordinary one.
    finite = coloured[np.isfinite(coloured)]
    if finite.size:
        outside = int(((finite < vmin) | (finite > vmax)).sum())
        if outside:
            print(f'      {outside} cell(s) fall outside the shared colour '
                  f'range [{vmin:g}, {vmax:g}] and are drawn at its end '
                  f'(range in this panel: {finite.min():.2f} to '
                  f'{finite.max():.2f}). Their printed values are exact.')

    ax.set_facecolor(C.MISSING_CELL_COLOUR)
    image = ax.imshow(np.ma.masked_invalid(coloured), aspect='auto', cmap=cmap,
                      vmin=vmin, vmax=vmax)
    image.cmap.set_bad(C.MISSING_CELL_COLOUR)
    if separate_first_column and data.shape[1] > 1:
        # Paint the reference column white behind its numbers.
        import matplotlib.patches as mpatches
        ax.add_patch(mpatches.Rectangle(
            (-0.5, -0.5), 1.0, data.shape[0], facecolor='white',
            edgecolor='none', zorder=1.5))

    # Cell borders. Without them a run of similar values reads as one block and
    # the eye cannot tell which number belongs to which row.
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0)

    span = (vmax - vmin) or 1.0
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    ax.text(j, i, never_run, ha='center', va='center',
                            fontsize=6, color='#333333', style='italic',
                            zorder=3)
                    continue
                reference = separate_first_column and j == 0
                light = reference or (v - vmin) / span > 0.55
                ax.text(j, i, fmt.format(v), ha='center', va='center',
                        fontsize=7, zorder=3,
                        color='#111111' if light else '#FFFFFF')

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([column_labeller(c) for c in table.columns],
                       rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([row_labeller(r) for r in table.index], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if separate_first_column and data.shape[1] > 1:
        ax.axvline(0.5, color='white', linewidth=4)
        ax.axvline(0.5, color='#333333', linewidth=0.9)
    return image, table


def title(ax, letter, text):
    """A panel title: the letter and WHAT THE PANEL IS, and nothing else.

    Why the panel matters, what to look for, what the caveats are -- all of that
    belongs in the caption, where a reader can take it at their own pace and
    where the journal expects it. A title carrying an explanation makes the
    figure look crowded before a single number is read.
    """
    ax.set_title(f'{letter}) {text}', fontweight='bold', fontsize=9, loc='left')


# ---------------------------------------------------------------------------
# SHAPE D -- the bar chart
# ---------------------------------------------------------------------------

def grouped_bars(ax, frame, category, series, value, spread=None,
                 labeller=None, category_labeller=None, colours=None,
                 legend=True, legend_ncol=4, stacked=False):
    """One group of bars per category, one bar per series member.

    `spread` names a column holding a half-height whisker -- how much that share
    moved across the replicates. No version of the variance figure has ever
    carried one, because the metric behind it was computed on an averaged curve
    and had no spread to show.
    """
    labeller = labeller or (lambda v: str(v))
    category_labeller = category_labeller or C.condition_label
    categories = list(dict.fromkeys(frame[category]))
    members = list(dict.fromkeys(frame[series]))
    index = np.arange(len(categories))
    width = 0.8 / max(len(members), 1)

    bottom = np.zeros(len(categories))
    for k, member in enumerate(members):
        sub = frame[frame[series] == member].set_index(category)
        heights = np.array([sub[value].get(c, np.nan) for c in categories],
                           dtype=float)
        colour = (colours or {}).get(member, C.ANOVA_FACTOR_COLORS.get(
            member, C.model_color(member)))
        if stacked:
            ax.bar(index, np.nan_to_num(heights), 0.7, bottom=bottom,
                   color=colour, label=labeller(member), linewidth=0)
            bottom += np.nan_to_num(heights)
        else:
            offset = (k - (len(members) - 1) / 2) * width
            ax.bar(index + offset, heights, width * 0.92, color=colour,
                   label=labeller(member), linewidth=0)
            if spread and spread in frame.columns:
                whisk = np.array([sub[spread].get(c, np.nan) for c in categories],
                                 dtype=float)
                ax.errorbar(index + offset, heights, yerr=whisk, fmt='none',
                            ecolor='#333333', elinewidth=0.8, capsize=1.5)

    ax.set_xticks(index)
    ax.set_xticklabels([category_labeller(c) for c in categories],
                       rotation=25, ha='right')
    if legend:
        _order_legend(ax, legend_ncol, loc='upper left', fontsize=8,
                      bbox_to_anchor=(0, 1.18))
    ax.spines[['top', 'right']].set_visible(False)


def panel_letter(ax, letter):
    """a), b), c) in the corner, one place so they cannot drift in style."""
    ax.annotate(f'{letter})', xy=(0, 1.02), xycoords='axes fraction',
                ha='left', va='bottom', fontweight='bold', fontsize=11)
