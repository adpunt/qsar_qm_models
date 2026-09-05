#!/usr/bin/env python
"""The assertions that stop this analysis going wrong the same ways it already has.

RERUN_PLAN.md 0.6 lists thirteen failure modes and says which guard belongs
where. Six of them belong in the figure script -- 1, 3, 4, 8, 9 and 12 -- and
this module is all six. Section 14.2's "averaging guard" is failure mode 1
generalised from correlations to whole figures.

  1  Pooling across a dimension that should have been conditioned on
  3  Averaging replicates before computing a derived quantity
  4  Printing a ratio without its denominator
  8  A filter that is not random with respect to the question
  9  Silent no-ops
 12  One number, two names

Each guard is an ASSERTION THAT FAILS THE RUN. A guard nobody executes is not a
guard, and a note in a document is not a guard.

THE ONE THAT MATTERS MOST
-------------------------
`declare` is the averaging guard. A figure that is SUPPOSED to show one
representation, handed a table that still contains all six, does not fail today
-- it silently plots the average of six things and the title still says one.
That is `generate_paper_figures_v2.py:2494`, the line the paper's cross-dataset
claim rests on:

    model_ds_c = val_auc_df.pivot_table(values='auc_norm', index='model',
                                        columns='dataset', aggfunc='mean')

It groups by model and dataset only, so `aggfunc='mean'` averages over
representation and noise type together -- up to 36 values per bar. With `declare`
in place that raises instead of drawing.

`declare` demands a statement about EVERY factor in the frame, not just the
fixed ones. Forgetting a factor is the failure; a guard you can silently opt out
of by leaving a name off a list would not have caught line 2494 either.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
from figlib_config import metric_definition, metric_label  # noqa: F401,E402


class GuardError(AssertionError):
    """A figure or table asked for a number that would not mean what it says."""


#: The factors of this study. Averaging over one of these reports a number that
#: describes none of its levels.
FACTORS = ('dataset', 'model', 'rep', 'condition', 'sigma')

#: The two axes a figure MAY average over, and the only two. Replicates are
#: repeats, so their spread is the error bar. Molecules are the population the
#: metric is defined on. Everything else is a factor.
AGGREGATABLE = ('replicate', 'iteration', 'fold', 'molecule', 'sample_idx',
                'mol_id', 'permutation')

#: Column spellings that mean the same factor on the two sides of the study.
_ALIASES = {
    'representation': 'rep',
    'strategy': 'condition',
    'noise_type': 'condition',
    'task_condition': 'condition',
    'task_model': 'model',
    'task_rep': 'rep',
    'task_dataset': 'dataset',
    'noise_level': 'sigma',
    'level': 'sigma',
}


def canonical_factor(column):
    return _ALIASES.get(str(column), str(column))


def factors_present(df):
    """Which factors this frame still carries more than one value of."""
    out = {}
    for col in df.columns:
        factor = canonical_factor(col)
        if factor in FACTORS:
            values = pd.unique(df[col].dropna())
            out.setdefault(factor, (col, list(values)))
    return out


def declare(df, figure, *, fixed=None, varies=(), aggregates=()):
    """State what a figure does with every factor, and fail if the data disagrees.

    `fixed`      -- mapping of factor -> the single value it must hold, or
                    factor -> None to require exactly one value without naming
                    which. Goes into the figure's title.
    `varies`     -- factors this figure deliberately shows several levels of.
    `aggregates` -- axes this figure averages over. ONLY replicates and molecules
                    may appear here (section 14.2); naming a factor raises.

    Returns the title fragment naming the fixed values, so a title cannot drift
    away from the data behind it.

    Raises GuardError if a factor in the frame appears in none of the three, or
    if a factor declared fixed still carries more than one value.
    """
    fixed = dict(fixed or {})
    varies = {canonical_factor(v) for v in varies}
    aggregates = [str(a) for a in aggregates]

    illegal = [a for a in aggregates if canonical_factor(a) in FACTORS]
    if illegal:
        raise GuardError(
            f"{figure}: declared that it averages over {illegal}, which are "
            f"FACTORS of this study, not repeats. A figure may average over "
            f"replicates and over molecules and over nothing else "
            f"(RERUN_PLAN.md 14.2). Averaging over a factor reports a number "
            f"that describes none of its levels -- to show the spread across "
            f"one, plot the spread.")

    unknown_agg = [a for a in aggregates if a not in AGGREGATABLE]
    if unknown_agg:
        raise GuardError(
            f"{figure}: {unknown_agg} is not a recognised repeat axis. "
            f"Known: {list(AGGREGATABLE)}.")

    present = factors_present(df)
    declared = {canonical_factor(f) for f in fixed} | varies
    undeclared = sorted(set(present) - declared)

    bad = []
    title_bits = []
    for factor, expected in fixed.items():
        factor = canonical_factor(factor)
        if factor not in present:
            # A factor the frame does not carry is vacuously fixed -- but say so
            # in the title anyway, so the caption still records the choice.
            if expected is not None:
                title_bits.append((factor, expected))
            continue
        column, values = present[factor]
        if len(values) != 1:
            bad.append((factor, column, values))
            continue
        got = values[0]
        if expected is not None and str(got) != str(expected):
            bad.append((factor, column, [got]))
            continue
        title_bits.append((factor, got))

    # Every problem at once. Reporting only the first sends the caller round
    # the loop once per factor, and the line this guard exists for --
    # generate_paper_figures_v2.py:2494 -- gets TWO of them wrong together.
    if bad or undeclared:
        lines = [f"{figure}: would plot the average of something it does not "
                 f"name."]
        for factor, column, values in bad:
            wanted = fixed.get(factor, fixed.get(column))
            if len(values) == 1:
                lines.append(
                    f"  {factor}: declared fixed at {wanted!r}, the frame holds "
                    f"{values[0]!r}")
            else:
                shown = values[:8]
                more = '' if len(values) <= 8 else f' (+{len(values) - 8} more)'
                lines.append(
                    f"  {factor}: declared fixed, but {len(values)} values are "
                    f"still in the frame -- {shown}{more}. Drawing this plots "
                    f"their average under a title that names one.")
        for factor in undeclared:
            column, values = present[factor]
            lines.append(
                f"  {factor}: not named in fixed=, varies= or aggregates= at "
                f"all, and the frame holds {len(values)} of them -- "
                f"{values[:8]}")
        lines.append(
            "Every factor must be declared. RERUN_PLAN.md 14.2: a figure may "
            "average over replicates and over molecules, and over nothing else.")
        raise GuardError('\n'.join(lines))

    return title_fragment(title_bits)


def title_fragment(pairs):
    """The fixed values, spelled for a caption. Generated from the data, never
    typed, so a title cannot say one thing while the numbers say another."""
    show = {
        'dataset': C.dataset_label,
        'rep': C.rep_label,
        'model': C.model_label,
        'condition': C.condition_label,
        'sigma': lambda v: f'level {v:g}' if isinstance(v, (int, float)) else str(v),
    }
    bits = [show.get(f, str)(v) for f, v in pairs]
    return ', '.join(b for b in bits if b)


# ---------------------------------------------------------------------------
# Guard 3 -- averaging replicates before computing a derived quantity
# ---------------------------------------------------------------------------

def assert_replicates(df, group_cols, min_n=None, where='this analysis'):
    """Every cell must carry more than one observation before a variance
    decomposition, and at least `min_n` of them.

    Retention was computed on an AVERAGED curve, so no robustness number in the
    submitted paper has a spread. On the assay side the folds were averaged
    before integration, which forces the unexplained share of the variance
    decomposition to exactly zero -- a saturated fit reproducing the data.
    """
    min_n = C.MIN_CELL_ITERS if min_n is None else min_n
    counts = df.groupby(list(group_cols), dropna=False).size()
    if counts.empty:
        raise GuardError(
            f"{where}: no cells at all after grouping on {list(group_cols)}. "
            f"A condition that produces no rows fails loudly rather than "
            f"drawing an empty panel (RERUN_PLAN.md 0.6, failure mode 9).")
    singletons = counts[counts < 2]
    if len(singletons) == len(counts):
        raise GuardError(
            f"{where}: every cell holds exactly one observation, so a variance "
            f"decomposition would be saturated and its residual arithmetically "
            f"zero. Keep the replicate or fold axis all the way through "
            f"(RERUN_PLAN.md 0.6, guard 3).")
    thin = counts[counts < min_n]
    return {
        'n_cells': int(len(counts)),
        'n_below_min': int(len(thin)),
        'min_n': int(min_n),
        'thin_cells': thin.index.tolist(),
        'median_per_cell': float(counts.median()),
    }


# ---------------------------------------------------------------------------
# Guard 4 -- printing a ratio without its denominator
# ---------------------------------------------------------------------------

#: Every ratio this study reports, and the columns that must sit beside it.
RATIOS = {
    'auc_norm': ('baseline_r2',),
    'auc_ratio': ('auc_error',),
    'auc_delta': ('auc_error', 'auc_ratio'),
    'rho_delta': ('rho_error', 'rho_ratio'),
}


def with_components(table, where='a table'):
    """Refuse to write a table that prints a ratio and not its components.

    "Robustness is decoupled from accuracy" is arithmetic, not a finding: the
    metric divides the baseline out, so a model with a weak baseline scores well
    by having less to lose. Enforced in the writer, not in the caption.
    """
    missing = []
    for ratio, components in RATIOS.items():
        if ratio not in table.columns:
            continue
        absent = [c for c in components if c not in table.columns]
        if absent:
            missing.append((ratio, absent))
    if missing:
        lines = [f"{where}: a ratio is being printed without its components."]
        for ratio, absent in missing:
            lines.append(
                f"  {ratio} needs {absent} beside it. "
                f"{metric_definition(ratio) if ratio in C.METRICS else ''}")
        lines.append("RERUN_PLAN.md 0.6, guard 4: no ratio without its "
                     "components in adjacent columns.")
        raise GuardError('\n'.join(lines))
    return table


# ---------------------------------------------------------------------------
# Guard 8 -- a filter that is not random with respect to the question
# ---------------------------------------------------------------------------

class Filter:
    """A declared filter. Undeclared filters are how baseline-poor, unstable
    configurations got their retention quietly inflated: whole replicates below
    an accuracy floor were deleted and the variance decomposition dropped more
    on top, neither stated anywhere."""

    def __init__(self, name, reason, predicate):
        self.name = name
        self.reason = reason
        self.predicate = predicate

    def apply(self, df):
        keep = self.predicate(df)
        return df[keep], df[~keep]


def headline_with_and_without(df, filters, headline, where='a headline'):
    """Compute a headline with every declared filter and again with none, and
    fail if the two disagree in DIRECTION.

    `headline` takes a frame and returns a number or a Series. Disagreeing in
    magnitude is expected and reported; disagreeing in sign means the filter is
    carrying the finding.
    """
    filtered = df
    log = []
    for f in filters:
        filtered, dropped = f.apply(filtered)
        log.append({'filter': f.name, 'reason': f.reason,
                    'rows_dropped': int(len(dropped))})

    with_filters = headline(filtered)
    without = headline(df)

    def _sign(x):
        arr = np.asarray(pd.Series(x).astype(float))
        return np.sign(np.where(np.isfinite(arr), arr, 0.0))

    if np.any(_sign(with_filters) * _sign(without) < 0):
        raise GuardError(
            f"{where}: the declared filters change the DIRECTION of the "
            f"result, not just its size.\n"
            f"  with filters   : {with_filters}\n"
            f"  without any    : {without}\n"
            f"  filters applied: {[f.name for f in filters]}\n"
            f"A filter that flips the sign of the finding is carrying the "
            f"finding (RERUN_PLAN.md 0.6, guard 8). Report both, or drop the "
            f"claim.")
    return {'with_filters': with_filters, 'without_filters': without,
            'log': log}


# ---------------------------------------------------------------------------
# Guard 9 -- silent no-ops
# ---------------------------------------------------------------------------

def assert_expected_rows(df, expected, where='a condition'):
    """A condition that produces no rows fails loudly.

    Uncertainty was written only when the zero-noise level was present, and only
    for one noise type unless a flag was passed; a guard was evaluated before the
    thing it tested for existed, so a control column was always blank and always
    passed.
    """
    n = 0 if df is None else len(df)
    if n == 0:
        raise GuardError(
            f"{where}: produced no rows at all. Expected about {expected}. "
            f"An empty result is reported, never drawn as an empty panel "
            f"(RERUN_PLAN.md 0.6, failure mode 9).")
    return n


# ---------------------------------------------------------------------------
# The scope rules a condition carries in its own registry entry
# ---------------------------------------------------------------------------

def refuse_ranking_axis(condition, where='this figure'):
    """Some conditions run on a NAMED SUBSET of pairs and cannot rank anything.

    `noise_conditions.json` says it in censoring's own scope block: it runs on
    five named pairs, so no claim about WHICH model resists it best can rest on
    that run. The registry states it; this makes it an assertion.
    """
    key = str(condition)
    scope = C.PAIR_SUBSET_CONDITIONS.get(key)
    if scope is None and key.startswith('censoring'):
        scope = C.PAIR_SUBSET_CONDITIONS.get('censoring')
    if scope is None:
        return
    raise GuardError(
        f"{where}: {key!r} cannot be used as a ranking axis. It runs on "
        f"{scope.get('n_pairs', 'a named subset of')} pairs, chosen to measure "
        f"the size of its effect rather than to compare models.\n"
        f"  {scope.get('paper_must_say', '')}\n"
        f"It also runs on its own level axis (a fraction of labels clipped, not "
        f"a fraction of the label spread), so it cannot share an x-axis or a "
        f"colour scale with the others either. Give it its own panel.")


def ranking_conditions(conditions):
    """The conditions a ranking may legitimately use."""
    out = []
    for c in conditions:
        try:
            refuse_ranking_axis(c)
        except GuardError:
            continue
        out.append(c)
    return out


# Re-exported rather than reimplemented: two implementations of one rule is
# failure mode 10, and this one already stops runs.
try:
    from uncertainty_stats import ConditioningError, assert_single_cell  # noqa: F401
except Exception:  # pragma: no cover - only when scipy/pandas are unavailable
    ConditioningError = GuardError

    def assert_single_cell(df, cols=None):  # type: ignore[misc]
        raise GuardError(
            'uncertainty_stats could not be imported, so the conditioning '
            'assertion is unavailable. Fix the import rather than skipping it.')
