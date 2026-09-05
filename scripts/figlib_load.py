#!/usr/bin/env python
"""Reading the three producers into one tidy frame, and saying what is missing.

THREE PRODUCERS, THREE LAYOUTS
------------------------------
  QM9 accuracy      results/anova_<condition>_<rep>_<model>.csv
                    the 22 RESULT_COLUMNS; the repeat axis is `iteration`
  Assay accuracy    <KIRBy>/results/validation_rerun/<model>_<rep>_<ds>/<ds>/
                        all_results.csv
                    the repeat axis is `fold`; the condition column is
                    `noise_type` or `strategy`, depending on vintage
  Assay uncertainty <KIRBy>/results/uncertainty_rerun/_merged/*.csv
                    already collated, with task_* provenance columns

WHAT THIS MODULE NORMALISES, AND WHAT IT REFUSES TO
---------------------------------------------------
Column spellings are normalised: `representation` -> `rep`, `strategy` and
`noise_type` -> `condition`, `iteration` and `fold` -> `replicate`.

`replicate` carries a companion column `replicate_kind`, and the two words are
NOT interchangeable (RERUN_PLAN.md 3.2b). QM9 has ten independent replicates and
their spread is an error bar. The assay datasets have five scaffold folds, which
are a PARTITION of one dataset, not repeats, and carry no error bar. Collapsing
the distinction is how the assay variance decomposition came to have a residual
of arithmetically zero.

FOUR THINGS THAT LOOK LIKE HOUSEKEEPING AND ARE NOT
---------------------------------------------------
1. `anova_*.csv` matches three siblings the same run writes -- the noise
   manifest, the per-epoch metrics, and the per-molecule uncertainty. Two of
   them carry the results columns, so only the NAME rule rejects them, and they
   were being concatenated into the results frame (RERUN_PLAN.md 2.13).
2. A row whose condition cannot be established is labelled `unknown_<file>`,
   never left blank. `drop_duplicates` treats blanks as equal, so a blank
   collapsed every condition for one (model, rep, level, replicate) onto
   whichever file was read last.
3. Settled condition names are matched before retired ones, longest first, so
   `outlier_p10` can never be read as the retired `outlier` -- which was the
   value-proportional strategy, a different mechanism entirely.
4. The QM9 injector puts the clipped percentage inside the name (`censoring_25`)
   where the assay runner writes plain `censoring` and carries 0.25 in the level
   column. Left alone, each censoring level arrives as a condition of its own
   holding one level, and every robustness function drops it for having no
   curve -- so QM9's censoring was silently absent from every figure while the
   assay one was present.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

#: Retired 2026-08-26. Kept ONLY so files written before then still parse.
#: `outlier` here is the value-proportional strategy, a different mechanism from
#: the settled `outlier_p10`; settled names are matched first so the two can
#: never be pooled.
RETIRED_CONDITIONS = ['legacy', 'outlier', 'quantile', 'threshold', 'hetero',
                      'valprop', 'heteroscedastic', 'value_proportional']
CONDITION_NORMALISE = {'heteroscedastic': 'hetero',
                       'value_proportional': 'valprop'}

#: The no-noise reference: `gaussian` under the settled scheme, `legacy` under
#: the retired one.
BASELINE_CONDITIONS = ('gaussian', 'legacy')

#: `condition_name` in rust/src/main.rs composes two names it does not
#: enumerate: a contaminated fraction and a censoring percentage, each optionally
#: suffixed with a non-Gaussian shape.
_COMPOSED_CONDITION = re.compile(
    r'^(outlier_p\d{2}(?:_[a-z_0-9]+?)?|censoring(?:_lower)?_\d+)_')
_CENSORING_LEVEL_SUFFIX = re.compile(r'^(censoring(?:_lower)?)_\d+$')

#: Siblings the same run writes off the same base path. Two of them carry the
#: results columns, so only the name rule rejects them.
_SIBLING_SUFFIXES = ('_uncertainty_values.csv', '_noise_manifest.csv',
                     '_per_epoch.csv', '_calibration.csv')

#: A file must carry all of these to be a results file, whatever it is called.
_RESULT_MARKERS = {'sigma', 'model', 'rep', 'r2'}

_RENAME = {
    'representation': 'rep',
    'strategy': 'condition',
    'noise_type': 'condition',
    'task_model': 'model',
    'task_rep': 'rep',
    'task_dataset': 'dataset',
    'task_condition': 'condition',
}


def strip_censoring_level(name):
    """`censoring_25` -> `censoring`. The level is already in its own column."""
    if not isinstance(name, str):
        return name
    m = _CENSORING_LEVEL_SUFFIX.match(name)
    return m.group(1) if m else name


def condition_from_stem(rest):
    """The condition a filename stem begins with, or None.

    Settled names first, longest first, so a settled name is never read as the
    retired name it happens to start with.
    """
    for name in sorted(C.SETTLED_CONDITIONS, key=len, reverse=True):
        if rest.startswith(name + '_'):
            return name
    m = _COMPOSED_CONDITION.match(rest)
    if m:
        return m.group(1)
    for name in sorted(RETIRED_CONDITIONS, key=len, reverse=True):
        if rest.startswith(name + '_'):
            return CONDITION_NORMALISE.get(name, name)
    return None


def attach_condition(df, stem, prefixes=('anova_', 'uncertainty_')):
    """Put the row's condition in `condition`, from a column or from the name.

    Never leaves it blank. A file whose condition cannot be established is
    labelled `unknown_<stem>`, which groups with nothing else.
    """
    for column in ('noise_type', 'condition', 'strategy'):
        if column in df.columns and df[column].notna().any():
            df['condition'] = (df[column].astype(str)
                               .map(strip_censoring_level))
            return df, None
    rest = stem
    for prefix in prefixes:
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    name = condition_from_stem(rest)
    if name is None:
        df['condition'] = f'unknown_{stem}'
        return df, stem
    df['condition'] = strip_censoring_level(name)
    return df, None


def apply_model_map(df, mapping, where):
    """Map `model` through `mapping`, and SAY what did not map.

    An unmapped name is lower-cased and kept, as it always was, so a legacy file
    still loads -- but it is now named in the output instead of disappearing
    into a spelling nothing else in the study uses, which is how four models
    were absent from every cross-pipeline table.
    """
    if 'model' not in df.columns or len(df) == 0:
        return df, []
    raw = df['model'].astype(str)
    unmapped = sorted(set(raw[~raw.isin(mapping)]))
    if unmapped:
        print(f'  WARNING: {len(unmapped)} model name(s) in {where} are not in '
              f'model_names.json and will not join to the other pipeline: '
              f'{", ".join(unmapped)}')
    df = df.copy()
    df['model'] = raw.map(mapping).fillna(raw.str.lower())
    return df, unmapped


def baseline_rows(frame, where):
    """The rows in the reference condition, refusing to pool when it is absent.

    Every one of these filters used to read
    `frame[frame.strategy == 'legacy'] if 'strategy' in frame else frame`, so a
    frame with no condition column silently became EVERY condition pooled under
    a table titled with one of them.
    """
    if 'condition' not in frame.columns:
        raise RuntimeError(
            f'{where}: the frame carries no noise condition, so the reference '
            f'condition cannot be selected. Falling through would pool every '
            f'condition under the reference condition\'s name.')
    out = frame[frame['condition'].isin(BASELINE_CONDITIONS)]
    if len(out) == 0:
        present = sorted(frame['condition'].dropna().unique())
        print(f'  WARNING: {where}: no rows in the reference condition '
              f'{BASELINE_CONDITIONS}. Present: {present}')
    return out


def _tidy(df, dataset, replicate_kind):
    """One spelling per column, and the repeat axis named for what it is."""
    df = df.rename(columns={k: v for k, v in _RENAME.items()
                            if k in df.columns and v not in df.columns})
    if 'dataset' not in df.columns:
        df['dataset'] = dataset
    df['dataset'] = df['dataset'].astype(str).str.lower()
    if 'rep' in df.columns:
        df['rep'] = df['rep'].astype(str).str.lower()
    for source in ('iteration', 'fold'):
        if source in df.columns and 'replicate' not in df.columns:
            df['replicate'] = df[source]
            break
    if 'replicate' not in df.columns:
        df['replicate'] = 0
    # RERUN_PLAN.md 3.2b: a replicate is QM9's; the other three have folds. Both
    # may carry an error bar and the two must be labelled differently.
    df['replicate_kind'] = replicate_kind
    for numeric in ('sigma', 'r2', 'rmse', 'mae', 'delivered_dose'):
        if numeric in df.columns:
            df[numeric] = pd.to_numeric(df[numeric], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# A cache, because the old script re-parsed thousands of CSVs on every run
# ---------------------------------------------------------------------------

def _fingerprint(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(str(p).encode())
        h.update(str(int(st.st_mtime)).encode())
        h.update(str(st.st_size).encode())
    return h.hexdigest()[:16]


def _cached(cache_dir, key, paths, build):
    if cache_dir is None:
        return build()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f'{key}_{_fingerprint(paths)}.parquet'
    if target.exists():
        try:
            print(f'  cache hit: {target.name}')
            return pd.read_parquet(target)
        except Exception as exc:  # pragma: no cover
            print(f'  cache unreadable ({exc}); rebuilding')
    frame = build()
    if frame is not None and len(frame):
        try:
            frame.to_parquet(target, index=False)
        except Exception as exc:  # pragma: no cover
            print(f'  could not write cache ({exc}); continuing')
    return frame


# ---------------------------------------------------------------------------
# QM9
# ---------------------------------------------------------------------------

def load_qm9(results_dir, cache_dir=None):
    """Every `anova_*.csv` in one directory, as one tidy frame."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        print(f'  no QM9 directory at {results_dir}')
        return None
    paths = [p for p in sorted(results_dir.glob('anova_*.csv'))
             if not any(p.name.endswith(s) for s in _SIBLING_SUFFIXES)]
    if not paths:
        print(f'  no anova_*.csv in {results_dir}')
        return None

    def build():
        frames, unnamed, skipped = [], [], 0
        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f'  WARNING: {path.name} unreadable ({exc}); skipped')
                continue
            if not _RESULT_MARKERS <= set(df.columns):
                skipped += 1
                continue
            df, unnamed_stem = attach_condition(df, path.stem)
            if unnamed_stem:
                unnamed.append(unnamed_stem)
            df['source_file'] = path.name
            frames.append(df)
        if not frames:
            return None
        if skipped:
            print(f'  {skipped} file(s) matched anova_*.csv without the results '
                  f'columns and were skipped')
        if unnamed:
            print(f'  WARNING: {len(unnamed)} file(s) carry no condition in a '
                  f'column or in their name; labelled unknown_<file> so they '
                  f'cannot pool with a named condition: {unnamed[:5]}')
        df = pd.concat(frames, ignore_index=True)
        df, _ = apply_model_map(df, C.QM9_MODEL_MAP, 'QM9')
        df = _tidy(df, 'qm9', 'replicate')
        # The replicate is IN the key. Without it, appended runs of the same
        # cell overwrite one another.
        key = ['model', 'rep', 'condition', 'sigma', 'replicate']
        missing = [k for k in key if k not in df.columns]
        if missing:
            raise RuntimeError(
                f'QM9 rows are missing {missing}, so duplicate runs of one cell '
                f'cannot be told apart and would silently overwrite each other.')
        before = len(df)
        df = df.drop_duplicates(subset=key, keep='last')
        if before != len(df):
            print(f'  {before - len(df)} duplicate QM9 row(s) dropped '
                  f'(same model, rep, condition, level and replicate)')
        return df.reset_index(drop=True)

    return _cached(cache_dir, 'qm9', paths, build)


# ---------------------------------------------------------------------------
# Assay accuracy
# ---------------------------------------------------------------------------

_ASSAY_DATASET_ALIASES = {
    'herg_ki': 'herg', 'chembl-herg-ki': 'herg', 'herg-ki': 'herg',
    'openadmet-logd': 'logd', 'openadmet-caco2_efflux': 'caco2',
    'caco2_efflux': 'caco2',
}


def _assay_dataset(name):
    key = str(name).strip().lower()
    return _ASSAY_DATASET_ALIASES.get(key, key)


def load_assay_accuracy(dirs, cache_dir=None):
    """Every `all_results.csv` under one or more validation-rerun trees."""
    dirs = [Path(d) for d in (dirs or []) if d]
    paths = []
    for root in dirs:
        if not root.is_dir():
            print(f'  no assay directory at {root}')
            continue
        paths.extend(sorted(root.rglob('all_results.csv')))
        paths.extend(sorted(root.rglob('summary.csv')))
    # Prefer all_results.csv wherever both exist in one directory.
    by_parent = {}
    for p in paths:
        current = by_parent.get(p.parent)
        if current is None or p.name == 'all_results.csv':
            by_parent[p.parent] = p
    paths = sorted(by_parent.values())
    if not paths:
        return None

    def build():
        frames = []
        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f'  WARNING: {path} unreadable ({exc}); skipped')
                continue
            if not _RESULT_MARKERS <= set(df.columns):
                continue
            if 'dataset' not in df.columns:
                df['dataset'] = path.parent.name
            df, _ = attach_condition(df, path.stem)
            df['source_file'] = str(path)
            frames.append(df)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        df['dataset'] = df['dataset'].map(_assay_dataset)
        df, _ = apply_model_map(df, C.VALIDATION_MODEL_MAP, 'the assay datasets')
        df = _tidy(df, 'assay', 'fold')
        # `fold` is IN the key. The old loader deduplicated on dataset, model,
        # rep, condition and level with no fold, kept the first row, and
        # silently discarded four fifths of the data.
        key = ['dataset', 'model', 'rep', 'condition', 'sigma', 'replicate']
        before = len(df)
        df = df.drop_duplicates(subset=key, keep='last')
        if before != len(df):
            print(f'  {before - len(df)} duplicate assay row(s) dropped')
        return df.reset_index(drop=True)

    return _cached(cache_dir, 'assay', paths, build)


# ---------------------------------------------------------------------------
# Assay uncertainty -- already collated by the run's own merge step
# ---------------------------------------------------------------------------

def load_merged_uncertainty(dirs):
    """The `_merged/` tables written by
    slurm_scripts_uncertainty_rerun/merge_results.py.

    `coverage.csv` is read rather than rebuilt: it already carries a `status`
    per (dataset, model, rep, condition) -- MISSING, NO_OOF, OOF_ALL_NAN,
    TRUNCATED_OOF, PARTIAL_FOLDS, PARTIAL_LEVELS or OK -- computed by the code
    that knows what each task was asked to produce.
    """
    out = {'summary': None, 'coverage': None, 'all_results': None}
    for root in [Path(d) for d in (dirs or []) if d]:
        merged = root / '_merged' if (root / '_merged').is_dir() else root
        for key, name in (('summary', 'summary.csv'),
                          ('coverage', 'coverage.csv'),
                          ('all_results', 'all_results.csv')):
            path = merged / name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                print(f'  WARNING: {path} unreadable ({exc})')
                continue
            df = df.rename(columns={k: v for k, v in _RENAME.items()
                                    if k in df.columns and v not in df.columns})
            if 'dataset' in df.columns:
                df['dataset'] = df['dataset'].map(_assay_dataset)
            out[key] = df if out[key] is None else pd.concat(
                [out[key], df], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# The per-molecule rows
# ---------------------------------------------------------------------------

def uncertainty_column(df):
    """Which column is "the model's uncertainty".

    `models/model_defaults.py` settles this on 36 measured fits: raw, because
    the calibration multiplier is refitted at every noise level, so calibrated
    coverage is nominal at each level BY CONSTRUCTION and says nothing about how
    uncertainty responds to noise. Both rank questions are unaffected either way
    -- the maximum difference across all 36 fits was exactly 0.0.
    """
    raw = ['y_pred_std_uncalibrated', 'y_pred_std', 'uncertainty', 'std']
    calibrated = ['y_pred_std_calibrated']
    order = raw + calibrated if C.UNCERTAINTY_PRIMARY == 'raw' \
        else calibrated + raw
    for name in order:
        if name in df.columns and df[name].notna().any():
            return name
    raise RuntimeError(
        f'no uncertainty column in the frame. Looked for {order}.')


def load_per_molecule(sources, dataset_name=None, strict=True):
    """The per-molecule rows, through `uncertainty_stats.load_uncertainty`.

    That module owns the schema detection for both producers and the settled
    -scale reconciliation; a second reader here would be failure mode 10.
    """
    import uncertainty_stats as unc

    paths = [str(s) for s in (sources or []) if s]
    if not paths:
        return None
    try:
        return unc.load_uncertainty(
            paths, strict=strict,
            uncertainty_column='uncalibrated'
            if C.UNCERTAINTY_PRIMARY == 'raw' else 'calibrated',
            dataset_name=dataset_name)
    except unc.UncertaintySchemaError as exc:
        print(f'  per-molecule rows NOT loaded: {exc}')
        return None


# ---------------------------------------------------------------------------
# The two declared filters (guard 8)
# ---------------------------------------------------------------------------

def catastrophic_filter():
    """A replicate in which any level failed to train is dropped WHOLE.

    Dropping only the failed level would leave a curve with a hole in it and
    integrate across the gap as though nothing happened. The repeat axis has two
    names and the old implementation returned the frame untouched the moment
    `iteration` was missing -- so it ran on QM9 and was a silent no-op on the
    three assay datasets.
    """
    def predicate(df):
        if 'r2' not in df.columns or len(df) == 0:
            return pd.Series(True, index=df.index)
        keys = [c for c in ('dataset', 'model', 'rep', 'condition', 'replicate')
                if c in df.columns]
        failed = df.loc[df['r2'] < C.CATASTROPHIC_R2_THRESHOLD, keys]
        if failed.empty:
            return pd.Series(True, index=df.index)
        bad = set(map(tuple, failed.drop_duplicates().to_numpy()))
        keyed = list(map(tuple, df[keys].to_numpy()))
        return pd.Series([k not in bad for k in keyed], index=df.index)

    return G.Filter(
        'catastrophic replicate',
        f'any level with R2 below {C.CATASTROPHIC_R2_THRESHOLD}; the whole '
        f'replicate goes, because integrating across a hole is worse',
        predicate)


def baseline_filter(auc_frame_column='baseline_r2'):
    """A configuration whose CLEAN accuracy is too low is not asked how much of
    it it retains -- a near-zero denominator makes the ratio unstable."""
    def predicate(df):
        if auc_frame_column not in df.columns:
            return pd.Series(True, index=df.index)
        return df[auc_frame_column] >= C.BASELINE_THRESHOLD

    return G.Filter(
        'weak clean baseline',
        f'clean R2 below {C.BASELINE_THRESHOLD}; the retention ratio is '
        f'unstable when there is almost nothing to retain',
        predicate)


def collapsed_gp_filter():
    """A Gaussian-process fit that returned its prior everywhere.

    Nothing filtered these before. A collapsed fit still writes a number and the
    number reads as a weak REPRESENTATION rather than as a fit that answered
    nothing -- which is how the two learned embeddings were written off.
    """
    def predicate(df):
        if 'gp_collapsed' not in df.columns:
            return pd.Series(True, index=df.index)
        flag = pd.to_numeric(df['gp_collapsed'], errors='coerce').fillna(0)
        return flag <= 0

    return G.Filter(
        'collapsed Gaussian process',
        'gp_collapsed is set: the fit returned its prior for every molecule, '
        'so the number describes the fit and not the representation',
        predicate)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(df, where='QM9'):
    """What landed, against what the ladder says should be there.

    The screen lands in pieces, so a partial grid is the normal case. This makes
    the gaps a table instead of a silent absence.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    keys = [c for c in ('dataset', 'model', 'rep', 'condition') if c in df.columns]
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        condition = record.get('condition', '')
        expected = C.expected_levels(str(condition))
        got = sorted(pd.unique(group['sigma'].dropna()))
        n_rep = group.groupby('sigma')['replicate'].nunique()
        record.update({
            'n_rows': int(len(group)),
            'levels_present': len(got),
            'levels_expected': len(expected),
            'levels_missing': sorted(set(np.round(expected, 6))
                                     - set(np.round(got, 6))),
            'replicates_min': int(n_rep.min()) if len(n_rep) else 0,
            'replicates_max': int(n_rep.max()) if len(n_rep) else 0,
            'has_clean_level': bool(len(got) and min(got) == 0),
            # The assay frames carry no gp_collapsed column at all, so this
            # has to survive its absence rather than assume a Series.
            'gp_collapsed': (int(pd.to_numeric(group['gp_collapsed'],
                                               errors='coerce').fillna(0).sum())
                             if 'gp_collapsed' in group.columns else 0),
        })
        record['status'] = _coverage_status(record)
        rows.append(record)
    out = pd.DataFrame(rows)
    n_ok = int((out['status'] == 'OK').sum()) if len(out) else 0
    print(f'  {where}: {len(out)} cells, {n_ok} complete')
    return out


def _coverage_status(record):
    if not record['has_clean_level']:
        return 'NO_CLEAN_BASELINE'
    if record['levels_missing']:
        return 'PARTIAL_LEVELS'
    if record['replicates_min'] < C.MIN_CELL_ITERS:
        return 'THIN_REPLICATES'
    return 'OK'


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qm9-dir', default=str(C.ROOT / 'results'))
    args = ap.parse_args()
    frame = load_qm9(args.qm9_dir)
    if frame is None:
        print('nothing loaded')
    else:
        print(frame[['dataset', 'model', 'rep', 'condition', 'sigma',
                     'replicate', 'r2']].head(10).to_string(index=False))
        print(coverage(frame).to_string(index=False))
