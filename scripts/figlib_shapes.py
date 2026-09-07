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
    for name, group in frame.groupby(series, dropna=False, sort=False):
        group = group.sort_values(x)
        colour = (colours or {}).get(name, C.model_color(name))
        ax.plot(group[x], group[y], marker=(markers or {}).get(name, 'o'),
                markersize=3.5, color=colour, label=labeller(name))
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
                   s=26, color=C.CLEAN_COLOR, zorder=3)
        if spread_low and spread_high:
            for _, r in frame.iterrows():
                ax.plot([r[spread_low], r[spread_high]],
                        [position[r[row]]] * 2, color=C.CLEAN_COLOR,
                        linewidth=1.2, alpha=0.5, zorder=2)
    else:
        for name, group in frame.groupby(series, dropna=False, sort=False):
            ax.scatter(group[value], [position[r] for r in group[row]], s=26,
                       label=labeller(name), zorder=3,
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
         cmap='viridis', separate_first_column=False, cbar_label=None,
         row_order=None, column_order=None, never_run='not run'):
    """A square per combination, the number printed on it.

    A combination that was never run is left blank and labelled, so an absence
    can never be read as a low value. `separate_first_column` draws a gap after
    the first column, which is how the clean baseline sits beside the robustness
    grid without being mistaken for another noise condition.
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
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    image = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=cmap,
                      vmin=vmin, vmax=vmax)
    image.cmap.set_bad('#F0F0F0')

    span = (vmax - vmin) or 1.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, never_run, ha='center', va='center', fontsize=6,
                        color='#888888', style='italic')
                continue
            # Dark squares need light text and vice versa, or the number is
            # unreadable on exactly the cells a reader looks hardest at.
            light = (v - vmin) / span > 0.55
            ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7,
                    color='#111111' if light else '#FFFFFF')

    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([column_labeller(c) for c in table.columns],
                       rotation=35, ha='right')
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([row_labeller(r) for r in table.index])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if separate_first_column and data.shape[1] > 1:
        # A white gutter, so the clean baseline cannot be read as a condition.
        ax.axvline(0.5, color='white', linewidth=4)
        ax.axvline(0.5, color='#666666', linewidth=0.8)
    return image, table


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
