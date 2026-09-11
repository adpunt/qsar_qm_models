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
        # Not just lower-cased: the assay runner writes `MHG-GNN-pretrained`,
        # whose lower-cased form joins to nothing QM9 ever wrote.
        df['rep'] = df['rep'].map(C.canonical_rep)
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
            frame = pd.read_parquet(target)
            for extra in sorted(cache_dir.glob(f'{target.stem}.*.parquet')):
                name = extra.name[len(target.stem) + 1:-len('.parquet')]
                frame.attrs[name] = pd.read_parquet(extra)
            return frame
        except Exception as exc:  # pragma: no cover
            print(f'  cache unreadable ({exc}); rebuilding')
    frame = build()
    if frame is not None and len(frame):
        try:
            # `.attrs` goes into the parquet metadata as JSON, and the QM9
            # loader puts a DataFrame there -- the duplicate disagreements. So
            # every write failed with "Object of type DataFrame is not JSON
            # serializable" and every re-run re-read the whole grid. The frames
            # in attrs are written beside it and restored on a hit, because
            # they are a finding rather than a detail.
            plain = frame.copy()
            side = {k: v for k, v in frame.attrs.items()
                    if isinstance(v, pd.DataFrame)}
            plain.attrs = {k: v for k, v in frame.attrs.items()
                           if k not in side}
            plain.to_parquet(target, index=False)
            for name, extra in side.items():
                extra.to_parquet(target.with_suffix(f'.{name}.parquet'),
                                 index=False)
        except Exception as exc:  # pragma: no cover
            print(f'  could not write cache ({exc}); continuing')
    return frame


# ---------------------------------------------------------------------------
# QM9
# ---------------------------------------------------------------------------

#: How a cell that was run more than once is resolved. "last" is file append
#: order and is what the loader did first -- it is not a choice, it is whichever
#: task finished second.
DUPLICATE_RULES = ('median', 'last', 'first')


def _resolve_duplicates(df, key, rule='median'):
    """Reduce cells that were run more than once, by a STATED rule.

    Some tasks ran twice and appended to the same file. Nothing can be re-run,
    so the copies have to be resolved rather than fixed, and the rule matters:
    on the real data 326 duplicated cells disagree, 105 of them because the two
    copies had different training data.

    THE RULE, in order:

    1. Where `standardisation_sd` differs between copies, the two runs did not
       share a training set -- the standardisation is computed from the clean
       training labels -- so they are not repeat measurements of one thing. The
       copy matching the MAJORITY standardisation for that (model, rep,
       condition) is kept, because the majority is the split the rest of the
       cell was run on. The minority copy is dropped and counted.
    2. Otherwise the copies are the same nominal run, differing only by
       nondeterminism in the fit, and the MEDIAN across them is taken. That is
       a summary of a repeated measurement; "the last one written" is a summary
       of nothing.

    `rule='last'` or `'first'` restores positional selection for comparison.
    """
    if rule not in DUPLICATE_RULES:
        raise ValueError(f'duplicate rule {rule!r}; expected {DUPLICATE_RULES}')
    duplicated = df.duplicated(subset=key, keep=False)
    if not duplicated.any():
        return df
    if rule in ('last', 'first'):
        return df.drop_duplicates(subset=key, keep=rule)

    cell = [c for c in ('dataset', 'model', 'rep', 'condition') if c in df.columns]

    # Two columns say a pair of copies are NOT repeat measurements of one thing,
    # and a median across them would average two different experiments:
    #   standardisation_sd -- computed from the clean TRAINING labels, so a
    #     difference means a different training set;
    #   params_source -- a copy fitted at the shared default against one fitted
    #     at a tuned setting, which happens because --use-best-params re-reads
    #     the tuned files inside every run and a task that ran before they
    #     landed used the defaults.
    # In both cases the copy matching the MAJORITY for that cell is kept: the
    # majority is what the rest of the cell was run under.
    for column, why in (('standardisation_sd',
                         'their standardisation disagrees with the rest of '
                         'their cell, so they were fitted on a different '
                         'training set'),
                        ('params_source',
                         'they were fitted under different hyperparameters '
                         'from the rest of their cell')):
        if column not in df.columns or not cell:
            continue
        majority = (df.groupby(cell, dropna=False)[column]
                    .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None)
                    .rename('_majority').reset_index())
        df = df.merge(majority, on=cell, how='left')
        if pd.api.types.is_numeric_dtype(df[column]):
            mismatched = (df[column].notna() & df['_majority'].notna()
                          & ~np.isclose(df[column].astype(float),
                                        df['_majority'].astype(float),
                                        rtol=0, atol=1e-9))
        else:
            mismatched = (df[column].notna() & df['_majority'].notna()
                          & (df[column].astype(str) != df['_majority'].astype(str)))
        # Only drop where a matching copy of that cell survives, or the cell
        # would vanish entirely.
        good_keys = set(map(tuple, df.loc[~mismatched, key].to_numpy()))
        drop = mismatched & pd.Series(
            [tuple(k) in good_keys for k in df[key].to_numpy()], index=df.index)
        if int(drop.sum()):
            print(f'  {int(drop.sum())} row(s) dropped: {why}, so they are not '
                  f'a repeat measurement of the same thing and a median across '
                  f'them would average two different experiments')
        df = df[~drop].drop(columns=['_majority'])

    numeric = [c for c in df.columns
               if c not in key and pd.api.types.is_numeric_dtype(df[c])]
    still = df.duplicated(subset=key, keep=False)
    if still.any():
        print(f'  {int(still.sum())} row(s) in {int(df.loc[still].groupby(key, dropna=False).ngroups)} '
              f'cell(s) were run more than once with the same settings; taking '
              f'the median across copies rather than whichever finished last')
        aggregated = (df.groupby(key, dropna=False, as_index=False)
                      .agg({**{c: 'median' for c in numeric},
                            **{c: 'first' for c in df.columns
                               if c not in key and c not in numeric}}))
        return aggregated
    return df


def load_qm9(results_dir, cache_dir=None, duplicate_rule='median'):
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
        duplicated = df.duplicated(subset=key, keep='last')
        _disagreements = None
        if duplicated.any():
            # WHETHER THE TWO COPIES AGREE. Keeping the last is only safe if the
            # copies are the same computation. They need not be: the deep run
            # recomputes every replicate of gaussian, grouped_wider and
            # grouped_shifted that the screen and the main grid already ran
            # (generate_scripts.py STAGE_DEFAULTS -- stage 2 is replicates 0-9 on
            # STAGE2_CONDITIONS, which contains all three), and `--use-best-params`
            # re-reads the tuned hyperparameter files inside every training run. A
            # task that ran before those files landed fitted at the shared defaults;
            # one that ran after fitted at the tuned setting. `params_source` says
            # which, per row. "Last written" is file append order, not the better
            # number, so a disagreement has to be visible.
            _both = df.duplicated(subset=key, keep=False)
            _grp = df.loc[_both].groupby(key, dropna=False)
            if 'params_source' in df.columns:
                _mixed = int((_grp['params_source'].nunique(dropna=False) > 1).sum())
                if _mixed:
                    print(f'  ⚠ {_mixed} duplicated cell-and-replicate(s) have copies '
                          f'fitted under DIFFERENT hyperparameters (params_source '
                          f'differs between the copies). The copy matching the '
                          f'majority for its cell is kept -- a median across '
                          f'them would average a tuned fit with a default one.')
            if 'r2' in df.columns:
                _spread = _grp['r2'].agg(lambda v: v.max() - v.min())
                _far = _spread[_spread > 0.001]
                if len(_far):
                    # Carried out on the frame, because six lines of log cannot
                    # be chased and 275 disagreeing cells is a thing to look at.
                    _sub_all = df.loc[_both]
                    _all_keys = pd.MultiIndex.from_frame(_sub_all[key])
                    _disagreements = (_sub_all[_all_keys.isin(_far.index)]
                                      .sort_values(key + ['r2']))
                    print(f'  ⚠ {len(_far)} duplicated cell-and-replicate(s) differ by '
                          f'more than 0.001 R2 between copies (largest '
                          f'{_spread.max():.4f}). Same molecules and same seed should '
                          f'give the same number, so something else moved.')
                    # WHAT moved. The seed is random_seed XOR (iteration * const)
                    # and the generator never passes --random-seed, so both
                    # copies ran at 42 and the same iteration; file_no is a
                    # scratch-file NAME and never a seed. That leaves two
                    # candidates, and every row already carries the column that
                    # separates them.
                    _differs = []
                    for _col in ('spec_hash', 'spec_version', 'loss_function',
                                 'sample_size', 'gp_collapsed', 'gp_fit_method',
                                 'params_source', 'standardisation_sd'):
                        if _col in df.columns:
                            _n = int((_grp[_col].nunique(dropna=False) > 1).sum())
                            if _n:
                                _differs.append((_col, _n))
                    if _differs:
                        for _col, _n in _differs:
                            if _col == 'gp_collapsed':
                                print(f'      {_n} of them differ in '
                                      f'gp_collapsed -- the process returned its '
                                      f'PRIOR in one copy and fitted in the '
                                      f'other. That is not two measurements of '
                                      f'one thing; the collapsed copy answered '
                                      f'nothing and is filtered anyway.')
                            elif _col == 'standardisation_sd':
                                print(f'      {_n} of them differ in '
                                      f'standardisation_sd -- the labels were '
                                      f'scaled by different amounts, so the two '
                                      f'copies are not on one scale and their '
                                      f'errors are not comparable.')
                            else:
                                print(f'      {_n} of them also differ in '
                                      f'{_col!r} -- the copies were fitted under '
                                      f'different code or settings, so keeping '
                                      f'the last written is right, and it should '
                                      f'be the NEWER one')
                    else:
                        # Which models. Build the key as an index to test
                        # membership WITHOUT set_index, which would consume the
                        # `model` column the message needs.
                        _sub = df.loc[_both]
                        _keys = pd.MultiIndex.from_frame(_sub[key])
                        _models = (_sub.loc[_keys.isin(_far.index), 'model']
                                   .unique() if 'model' in _sub.columns else [])
                        print(f'      and NOTHING else on the row differs -- same '
                              f'spec_hash, same hyperparameters, same seed, and '
                              f'both noise seeds derive from the iteration. '
                              f'Affects: {sorted(set(_models))[:8]}')
                        # THE DECIDING TEST, and it is free. At level zero no
                        # noise is applied at all, so a clean row that disagrees
                        # cannot be the noise draw -- it has to be the training
                        # data or the split. A clean row that AGREES while the
                        # noised ones do not points the other way.
                        _clean_key = [k for k in key if k != 'sigma']
                        _z = _sub[_sub['sigma'] == 0]
                        if len(_z):
                            _zs = (_z.groupby(_clean_key, dropna=False)['r2']
                                   .agg(lambda v: v.max() - v.min()))
                            _zbad = int((_zs > 0.001).sum())
                            if _zbad:
                                print(f'      {_zbad} of them disagree at level '
                                      f'ZERO, where no noise is applied at all '
                                      f'(largest {_zs.max():.4f}). That cannot '
                                      f'be the noise draw: the training data or '
                                      f'the scaffold split differed between the '
                                      f'two runs.')
                            else:
                                print(f'      but they AGREE at level zero, so '
                                      f'the split and the data match and it is '
                                      f'the noise draw or the fit that moved.')
                        _worst = _spread.idxmax()
                        print(f'      largest single disagreement: '
                              f'{dict(zip(key, _worst if isinstance(_worst, tuple) else (_worst,)))}')
                        # HOW MUCH IT MATTERS, not just that it happens. The
                        # replicate spread is supposed to measure variation
                        # between independent noise draws. If re-running the
                        # SAME nominal replicate moves the number by as much as
                        # a different replicate does, then the error bar is
                        # partly training jitter and does not mean what the
                        # caption says.
                        _rep_key = [k for k in key if k != 'replicate']
                        _wobble = (df.groupby(_rep_key, dropna=False)['r2']
                                   .agg(lambda v: v.max() - v.min()))
                        _rerun = float(_far.median())
                        _between = float(_wobble.median())
                        _share = (_rerun / _between) if _between else float('inf')
                        print(f'      Re-running the same replicate moves R2 by '
                              f'{_rerun:.4f} at the median; DIFFERENT replicates '
                              f'of the same cell differ by {_between:.4f}. '
                              f'Re-run jitter is {_share:.0%} of the spread the '
                              f'error bars are drawn from.')
                        if _share > 0.5:
                            print(f'      That is not a floor under the error '
                                  f'bar, it is most of it. A replicate spread '
                                  f'quoted from this data is largely measuring '
                                  f'the fit being re-run, not the noise draw '
                                  f'changing.')
            # WHICH cells, not just how many. A cell appearing twice means the
            # same task ran twice -- a resubmit that was not needed, or two
            # array indices writing the same output path. Keeping the last is
            # right either way, but the count alone hides which.
            where = (df.loc[duplicated, ['model', 'rep', 'condition']]
                     .value_counts())
            print(f'  {int(duplicated.sum())} duplicate QM9 row(s) dropped '
                  f'(same model, rep, condition, level and replicate), across '
                  f'{len(where)} cell(s), resolved by the stated rule:')
            for (model, rep, condition), count in list(where.items())[:6]:
                print(f'      {model} / {rep} / {condition}: {count} row(s)')
            if len(where) > 6:
                print(f'      ... and {len(where) - 6} more cell(s)')
            df = _resolve_duplicates(df, key, duplicate_rule)
        out = df.reset_index(drop=True)
        if _disagreements is not None and len(_disagreements):
            # Carried on the frame so the entry point can write it
            # somewhere it can be opened.
            out.attrs['duplicate_disagreements'] = _disagreements
        return out

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


def load_assay_accuracy(dirs, cache_dir=None,
                        duplicate_rule='median'):
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
        duplicated = df.duplicated(subset=key, keep='last')
        if duplicated.any():
            where = (df.loc[duplicated, ['dataset', 'model', 'rep', 'condition']]
                     .value_counts())
            print(f'  {int(duplicated.sum())} duplicate assay row(s) dropped, '
                  f'across {len(where)} cell(s), resolved by the stated rule:')
            for keys, count in list(where.items())[:6]:
                print('      ' + ' / '.join(str(k) for k in keys)
                      + f': {count} row(s)')
            if len(where) > 6:
                print(f'      ... and {len(where) - 6} more cell(s)')
            df = df[~duplicated]
        return df.reset_index(drop=True)

    return _cached(cache_dir, 'assay', paths, build)


# ---------------------------------------------------------------------------
# Assay uncertainty -- already collated by the run's own merge step
# ---------------------------------------------------------------------------

def load_merged_uncertainty(dirs):
    """The uncertainty runs' own tables, merged or not.

    THE MERGE STEP IS NOT A PREREQUISITE. Every task writes its own
    all_results.csv and summary.csv into its own directory, and this reads those
    directly when `_merged/` is not there. Requiring a collation pass before any
    number could be looked at would mean waiting on a housekeeping step to see
    results that are already on disk.

    `_merged/coverage.csv` is the one thing the merge adds that cannot be
    rebuilt here as well: it carries a `status` per cell -- MISSING, NO_OOF,
    OOF_ALL_NAN, TRUNCATED_OOF, PARTIAL_FOLDS, PARTIAL_LEVELS or OK -- from the
    code that knows what each task was ASKED to produce. When it is absent the
    analysis still runs and D0 reports coverage from the rows themselves.
    """
    out = {'summary': None, 'coverage': None, 'all_results': None}
    for root in [Path(d) for d in (dirs or []) if d]:
        if not Path(root).is_dir():
            continue
        merged = root / '_merged'
        if merged.is_dir():
            sources = {name: [merged / name] for name in
                       ('summary.csv', 'coverage.csv', 'all_results.csv')}
        else:
            # Un-merged: one directory per task, each with its own tables.
            sources = {name: sorted(Path(root).rglob(name)) for name in
                       ('summary.csv', 'coverage.csv', 'all_results.csv')}
        for key, name in (('summary', 'summary.csv'),
                          ('coverage', 'coverage.csv'),
                          ('all_results', 'all_results.csv')):
            for path in sources.get(name, []):
                if not path.exists():
                    continue
                try:
                    df = pd.read_csv(path)
                except Exception as exc:
                    print(f'  WARNING: {path} unreadable ({exc})')
                    continue
                df = df.rename(columns={
                    k: v for k, v in _RENAME.items()
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
