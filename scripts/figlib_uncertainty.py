#!/usr/bin/env python
"""The uncertainty statistics, conditioned on the support flags before use.

WHY THIS WRAPPER EXISTS RATHER THAN CALLING uncertainty_stats DIRECTLY
---------------------------------------------------------------------
`uncertainty_stats.py` carries `aleatoric_support` and `epistemic_support`
through every statistic and applies NO filter to them. Its own docstring says
why:

    NOTHING IN THIS MODULE CORRELATES A COMPONENT WITH ANYTHING YET. They are
    carried so the split is readable at all; a statistic built on them must
    first condition on the support column, because a rank correlation against a
    constant column is undefined rather than zero.

So the conditioning has to happen somewhere, and this is where. A component that
is one number per fit may not be drawn as a line: a flat line there is
arithmetic, not a result (RERUN_PLAN.md 14.1, 14.6 row 13).

WHAT EACH STATISTIC IS FOR, IN ONE LINE EACH
--------------------------------------------
  q4_plain_correlation   NOT the answer. Near zero is expected and correct; a
                         large value would mean leakage. Reported so nobody
                         assumes it was hidden.
  q4_error_ratio         THE answer to Q4. Does dividing the out-of-fold error
                         by the predicted uncertainty rank corrupted labels
                         better than the error alone? `auc_delta` is the gain.
  permutation_null       The band every Q4 number is read against.
  q5_mean_uncertainty    Does noisier training make a model less sure? A
                         POPULATION statement, and it carries a column saying so.
  q6_error_ranking       Does the uncertainty rank the error against the CLEAN
                         label, within one level?
"""
from __future__ import annotations

import sys
from pathlib import Path

import gc

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402
import uncertainty_stats as unc  # noqa: E402
import uncertainty_decomposition as decomp  # noqa: E402

PER_MOLECULE = decomp.PER_MOLECULE
CONSTANT = decomp.CONSTANT
NONE = decomp.NONE

COMPONENTS = {
    'aleatoric': ('aleatoric_uncertainty', 'aleatoric_support'),
    'epistemic': ('epistemic_uncertainty', 'epistemic_support'),
}

PAIR = ['dataset', 'model', 'rep']

#: The model half counts as HOLDING if its slope spans zero, or if it is under
#: this fraction of the data half's slope. Stated once, here, so the verdict is
#: a declared rule rather than a judgement made while looking at a figure.
EPISTEMIC_FLAT_FRACTION = 0.25


# ---------------------------------------------------------------------------
# What each model is allowed to claim
# ---------------------------------------------------------------------------

def support_table(df):
    """One row per (dataset, model, representation): what each component is.

    The flags are read from the DATA, then checked against the registry that
    the pipelines write them from. A disagreement is reported rather than
    silently preferred one way, because either side being wrong is a finding.
    """
    rows = []
    keys = [c for c in PAIR if c in df.columns]
    for key, group in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        for name, (_, flag_col) in COMPONENTS.items():
            values = sorted(set(group[flag_col].dropna().astype(str))) \
                if flag_col in group.columns else []
            record[f'{name}_support'] = values[0] if len(values) == 1 else (
                '|'.join(values) if values else 'unrecorded')
            record[f'{name}_drawable'] = (len(values) == 1
                                          and values[0] == PER_MOLECULE)
        try:
            expected = decomp.support(record['model'])
            record['registry_aleatoric'], record['registry_epistemic'] = expected
            record['registry_agrees'] = (
                record['aleatoric_support'] == expected[0]
                and record['epistemic_support'] == expected[1])
        except decomp.DecompositionError as exc:
            record['registry_aleatoric'] = record['registry_epistemic'] = ''
            record['registry_agrees'] = False
            record['registry_note'] = str(exc)[:160]
        rows.append(record)
    return pd.DataFrame(rows)


def drawable(df, component):
    """The rows whose named component varies per molecule, and may be drawn.

    Anything else keeps its value in the table with its flag beside it. This is
    a guard, not a choice (RERUN_PLAN.md 14.6 row 13).
    """
    if component not in COMPONENTS:
        raise KeyError(f'{component!r}: expected one of {sorted(COMPONENTS)}')
    _, flag = COMPONENTS[component]
    if flag not in df.columns:
        return df.iloc[0:0]
    return df[df[flag].astype(str) == PER_MOLECULE]


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------

def q4(df, permutations=200, where='Q4'):
    """The answer to Q4, with its null band attached to every row.

    A Q4 number without its permutation band is not readable: `auc_delta` of
    0.03 is a finding or noise depending entirely on how wide the band is, and
    the two were reported in separate tables.
    """
    oof = df[df['split'] == 'train_oof'] if 'split' in df.columns else df
    if len(oof) == 0:
        print(f'  {where}: no out-of-fold rows, so Q4 cannot be answered. '
              f'A model scored on molecules it was fitted on would be '
              f'measuring memorisation.')
        return pd.DataFrame()
    answer = unc.q4_error_ratio(oof)
    plain = unc.q4_plain_correlation(oof)
    if permutations:
        null = unc.permutation_null(oof, statistic='error_noise_spearman',
                                    n_permutations=permutations)
        keys = [c for c in unc.PERM_GROUP_COLS if c in answer.columns
                and c in null.columns]
        answer = answer.merge(
            null[keys + ['null_lo', 'null_hi', 'p_value', 'observed',
                         'observed_inside_null', 'null_kind']],
            on=keys, how='left')
        answer['outside_null'] = ~answer['observed_inside_null'].fillna(True)
    if len(plain):
        keys = [c for c in unc.CELL_COLS if c in answer.columns
                and c in plain.columns]
        answer = answer.merge(plain[keys + ['rho_raw']], on=keys, how='left')
        answer = answer.rename(columns={'rho_raw': 'rho_plain_NOT_THE_ANSWER'})
    return answer


def q5(df):
    """Mean predicted uncertainty against noise level -- total, and per component.

    The total is every model. The two components are computed only where the
    support flag says that component varies per molecule; a slope through a
    column that is one number per fit is arithmetic about the fit, not a
    property of the molecules.
    """
    frames = []
    total = unc.q5_mean_uncertainty(df)
    if len(total):
        frames.append(total.assign(component='total', component_support='n/a'))
    for name, (column, flag) in COMPONENTS.items():
        if column not in df.columns:
            continue
        eligible = drawable(df, name)
        if len(eligible) == 0:
            continue
        swapped = eligible.assign(uncertainty=eligible[column])
        got = unc.q5_mean_uncertainty(swapped)
        if len(got):
            frames.append(got.assign(component=name,
                                     component_support=PER_MOLECULE))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def q6(df):
    """Does the uncertainty rank the error against the CLEAN label, within level."""
    return unc.q6_error_ranking(df)


def component_slopes(q5_frame):
    """Per pair and condition: how fast each component rises with the noise.

    The claim F6 makes is that the data-driven half climbs and the model half
    holds. That is two slopes and their spread, so it is stated as two slopes
    and their spread rather than read off a picture.
    """
    if q5_frame is None or not len(q5_frame):
        return pd.DataFrame()
    keys = [c for c in ['dataset', 'model', 'rep', 'condition']
            if c in q5_frame.columns]
    wide = (q5_frame[q5_frame['component'].isin(['aleatoric', 'epistemic'])]
            .groupby(keys + ['component'], dropna=False)
            .agg(slope=('slope_mean_unc_vs_sigma', 'median'),
                 slope_lo=('slope_mean_unc_vs_sigma', 'min'),
                 slope_hi=('slope_mean_unc_vs_sigma', 'max'),
                 n_folds=('slope_mean_unc_vs_sigma', 'size'))
            .reset_index())
    if not len(wide):
        return pd.DataFrame()
    pivot = wide.pivot_table(index=keys, columns='component',
                             values=['slope', 'slope_lo', 'slope_hi'])
    pivot.columns = [f'{a}_{b}' for a, b in pivot.columns]
    out = pivot.reset_index()
    have_a = 'slope_aleatoric' in out.columns
    have_e = 'slope_epistemic' in out.columns
    if not (have_a and have_e):
        # One component is `constant` or `none`, so there is nothing to compare
        # it against. Say which, rather than reporting a null that is a property
        # of the model rather than of the data.
        present = 'aleatoric' if have_a else ('epistemic' if have_e else 'neither')
        out['separates'] = False
        out['verdict'] = (f'only the {present} component varies per molecule, '
                          f'so the two cannot be compared')
        return out

    a = out['slope_aleatoric']
    e_lo, e_hi = out['slope_lo_epistemic'], out['slope_hi_epistemic']

    # A model whose component is `constant` or `none` has NaN here, and the
    # column exists only because OTHER models filled it. Testing the column's
    # presence would call that "neither component moves clearly", which reads as
    # a measured null when it is a property of the model. Test per row.
    only_aleatoric = out['slope_epistemic'].isna() & a.notna()
    only_epistemic = a.isna() & out['slope_epistemic'].notna()
    neither = a.isna() & out['slope_epistemic'].isna()

    # The claim is "the data-driven half climbs and the model half HOLDS", so
    # the test is whether the epistemic slope is flat -- not merely whether it
    # is the smaller of the two. Asking only whether one exceeds the other
    # passes a model whose components rise together at nearly the same rate,
    # which is exactly the failure the ordinary forest shows at x5.7 against
    # x5.3: both climb, because one bootstrap causes both.
    holds = ((e_lo <= 0) & (e_hi >= 0)) | (e_hi.abs() < EPISTEMIC_FLAT_FRACTION * a)
    rises = out['slope_lo_aleatoric'] > 0
    both_climb = rises & ~holds & (e_lo > 0)
    out['epistemic_holds'] = holds.fillna(False)
    out['separates'] = (rises & holds).fillna(False) & ~(
        only_aleatoric | only_epistemic | neither)
    out['verdict'] = np.select(
        [neither, only_aleatoric, only_epistemic, out['separates'], both_climb],
        ['neither component varies per molecule, so nothing can be drawn',
         'only the aleatoric component varies per molecule; the epistemic one '
         'is one number per fit, so the two cannot be compared',
         'only the epistemic component varies per molecule; the aleatoric one '
         'is one number per fit, so the two cannot be compared',
         'aleatoric rises, epistemic holds',
         'BOTH components rise -- the split failed'],
        default='neither component moves clearly')
    return out


def discover(sources, pattern='*_uncertainty_values.csv'):
    """Every per-molecule file under the given roots, de-duplicated."""
    from pathlib import Path as _Path
    found, seen = [], set()
    for source in (sources or []):
        if not source:
            continue
        path = _Path(source)
        candidates = ([path] if path.is_file()
                      else sorted(path.rglob(pattern)) if path.is_dir() else [])
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                found.append(c)
    return found


def load(sources, dataset_name=None, strict=True, where='per-molecule rows',
         paths=None):
    """Read per-molecule rows into ONE frame.

    ⚠️ Only for a handful of files. The whole set does not fit in memory: every
    row is one molecule at one noise level in one fold, so the real grid is
    hundreds of millions of them and a login node's per-user cap is reached
    while allocating a few megabytes. Use `statistics()` for anything larger --
    it never holds more than one file at a time.
    """
    targets = paths if paths is not None else [str(s) for s in (sources or [])
                                               if s]
    if not targets:
        return None
    try:
        df = unc.load_uncertainty(
            [str(t) for t in targets], strict=strict,
            uncertainty_column=('uncalibrated'
                                if C.UNCERTAINTY_PRIMARY == 'raw'
                                else 'calibrated'),
            dataset_name=dataset_name)
    except unc.UncertaintySchemaError as exc:
        print(f'  {where}: NOT loaded -- {exc}')
        return None
    return df


def statistics(sources, permutations=200, dataset_name=None, strict=True,
               max_files=None, where='per-molecule rows', progress_every=10):
    """Every uncertainty statistic, computed ONE FILE AT A TIME.

    WHY IT STREAMS
    --------------
    Loading the whole set first is what a login node kills. Every row is one
    molecule at one noise level in one fold; the real grid is hundreds of
    millions of them, and pandas dies while allocating a few megabytes because
    the cap was reached long before.

    IT IS SAFE TO SPLIT THIS WAY, AND THAT IS NOT OBVIOUS
    -----------------------------------------------------
    A statistic here is computed inside one cell -- dataset, model, rep,
    condition, sigma, fold, split (uncertainty_stats.CELL_COLS) -- and one file
    holds every level and every fold for one (condition, representation, model).
    So no cell spans two files, and the zero-level subtraction that
    q4_plain_correlation and confound_controlled_effect need finds its sigma = 0
    rows in the same file it is already holding. Splitting by file therefore
    changes no number. Splitting by anything coarser would.

    Returns the small statistic tables, never the rows.
    """
    files = discover(sources)
    if not files:
        print(f'  {where}: no *_uncertainty_values.csv found')
        return {}
    if max_files is not None and len(files) > max_files:
        print(f'  {where}: {len(files)} files, reading the first {max_files} '
              f'(--max-uncertainty-files). The rest are NOT included and the '
              f'uncertainty answers are partial.')
        files = files[:max_files]

    print(f'  {where}: {len(files)} file(s), one at a time so the whole set '
          f'never has to fit in memory')
    parts = {'support': [], 'q4': [], 'q5': [], 'q6': []}
    skipped = []
    for index, path in enumerate(files, 1):
        try:
            df = load(None, dataset_name=dataset_name, strict=strict,
                      where=str(path.name), paths=[path])
        except Exception as exc:  # noqa: BLE001
            skipped.append((path.name, f'{type(exc).__name__}: {exc}'))
            continue
        if df is None or not len(df):
            skipped.append((path.name, 'no usable rows'))
            continue
        try:
            parts['support'].append(support_table(df))
            got = q4(df, permutations=permutations, where=path.name)
            if len(got):
                parts['q4'].append(got)
            got = q5(df)
            if len(got):
                parts['q5'].append(got)
            got = q6(df)
            if len(got):
                parts['q6'].append(got)
        except Exception as exc:  # noqa: BLE001
            skipped.append((path.name, f'{type(exc).__name__}: {exc}'))
        finally:
            del df
            gc.collect()
        if progress_every and index % progress_every == 0:
            print(f'    {index}/{len(files)}')

    if skipped:
        print(f'  {where}: {len(skipped)} file(s) contributed nothing:')
        for name, why in skipped[:5]:
            print(f'      {name}: {why}')
        if len(skipped) > 5:
            print(f'      ... and {len(skipped) - 5} more')

    unmapped = unc.unmapped_model_names()
    if unmapped:
        print(f'  WARNING: {where}: model name(s) unknown to model_names.json '
              f'and joining to nothing: {unmapped}')

    out = {}
    for key, frames in parts.items():
        out[key] = (pd.concat(frames, ignore_index=True) if frames
                    else pd.DataFrame())
    out['slopes'] = component_slopes(out['q5'])
    out['n_files'] = len(files)
    return out
