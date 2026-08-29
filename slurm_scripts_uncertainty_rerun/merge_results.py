#!/usr/bin/env python3
"""Stitch the per-task uncertainty-run outputs together — WITHOUT loading them all.

Every array task writes its own results-root so concurrent tasks cannot race on
a shared file. This walks those directories and concatenates them.

  python merge_results.py --root /data/stat-cadd/scat9264/KIRBy/tests/results/uncertainty_rerun

Writes into <root>/_merged/:
  coverage.csv      what ran and what is missing — READ THIS FIRST
  all_results.csv   R2/RMSE/MAE per dataset x model x rep x condition x level x fold (small)
  summary.csv       per-config robustness rows (small)
  uncertainty.csv   every per-molecule row (LARGE — streamed, never held in memory)

Why streaming: with the out-of-fold training split, one LogD task writes roughly
6 levels x 5 folds x (1,000 test + 3,200 train) rows = ~126,000 rows, and there
are 336 tasks — tens of millions of rows and many GB. Building one DataFrame
would run out of memory on any normal node, at the very end of a multi-day run.
So the big file is appended chunk by chunk and the coverage report is
accumulated from per-file counts, never from the merged file.

The expected cells are NOT restated here. The conditions come from the generated
job scripts (falling back to noise_conditions.json), and the number of noise
levels each (dataset, condition) should have is read out of the runner itself —
the grid is the runner's to set and it has already changed twice, so "11 levels
everywhere" was wrong the moment the ladder did.

Pass --parquet to write a partitioned Parquet dataset instead (much smaller and
far faster to query); needs pyarrow.
"""
import argparse
import ast
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

# THE LAST RESORT, USED ONLY WHEN THERE ARE NO GENERATED SCRIPTS TO READ.
#
# These lists were the coverage grid until 2026-08-29, and by then they named
# three models the author's decision of 2026-08-28 dropped and two representations the run
# does not use, while missing the one it added. The report the runbook tells you
# to read FIRST would then have invented 462 missing cells for jobs nobody
# submitted and omitted all 84 ChemBERTa cells the run really produces --
# neither listing them as present nor as missing. The whole point of coverage.csv
# is to say what did not run, so a grid that cannot name the run is worse than
# no grid.
#
# The generated scripts carry the answer, exactly as they already do for the
# condition list, and they carry it PER MODEL because the representation set is
# no longer shared (VBLL runs ChemBERTa alone).
FALLBACK_MODELS = ['QRF', 'NGBoost', 'GP', 'VBLL-Full']
FALLBACK_DATASETS = ['logd', 'caco2', 'herg_ki']
FALLBACK_REPS = ['ecfp4', 'pdv', 'chemberta']

# The dataset name the runner uses internally, which is the key its per-dataset
# level grid is stored under.
RUNNER_DATASET_NAME = {'logd': 'OpenADMET-LogD',
                       'caco2': 'OpenADMET-Caco2_Efflux',
                       'herg_ki': 'ChEMBL-hERG-Ki'}

HERE = Path(__file__).resolve().parent
NOISE_CONDITIONS_FILE = HERE.parent / 'noise_conditions.json'

CHUNK = 200_000


def expected_conditions(explicit=None):
    """Which conditions this merge should expect to find.

    In order: what was asked for, then what the generated job scripts actually
    run, then the settled main grid. The six that used to be hard-coded here —
    legacy, outlier, quantile, hetero, threshold, valprop — were deleted in
    noiseInject 1.0.0, so every cell of the coverage report was MISSING by
    construction and nothing that did run was ever checked.
    """
    if explicit:
        return list(dict.fromkeys(explicit)), 'named on the command line'
    for sh in sorted(HERE.glob('unc_*.sh')):
        m = re.search(r'^CONDS=\((.*?)\)$', sh.read_text(), re.M)
        if m and m.group(1).split():
            return m.group(1).split(), f'read from {sh.name}'
    settled = json.loads(NOISE_CONDITIONS_FILE.read_text())
    return ([c['name'] for c in settled['stage_1_full_grid']],
            f'the main grid ({NOISE_CONDITIONS_FILE.name})')


def expected_level_counts(kirby_dir):
    """How many noise levels each (dataset, condition) should carry.

    Parsed out of the runner's source rather than copied, because the grids are
    the runner's to set, and it sweeps censoring on a different axis entirely.
    Returns None if the source cannot be read, and the coverage report then
    reports the level count without judging it.
    """
    runner = Path(kirby_dir) / 'tests' / 'alternative_data_noise_robustness.py'
    if not runner.exists():
        return None
    try:
        tree = ast.parse(runner.read_text())
    except SyntaxError:
        return None
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                    'NOISE_LEVELS_BY_DATASET', 'CENSORING_LEVELS', 'NOISE_LEVELS'):
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    if 'CENSORING_LEVELS' not in found or 'NOISE_LEVELS_BY_DATASET' not in found:
        return None
    by_dataset = found['NOISE_LEVELS_BY_DATASET']
    fallback = found.get('NOISE_LEVELS', [])
    n_censor = len(found['CENSORING_LEVELS'])

    def n_levels(ds, condition):
        if condition == 'censoring':
            return n_censor
        return len(by_dataset.get(RUNNER_DATASET_NAME.get(ds, ds), fallback)) or None

    return n_levels


def _task_of(path: Path):
    """Task dir is <model_slug>__<dataset>__<rep_slug>__<condition>.

    Split on the DOUBLE underscore, so a condition name with single underscores
    in it (grouped_wider, student_t_nu5) comes back whole.
    """
    for part in path.parts:
        if part.count('__') == 3:
            m, ds, rep, cond = part.split('__')
            return {'task_model': m, 'task_dataset': ds,
                    'task_rep': rep, 'task_condition': cond}
    return {'task_model': '', 'task_dataset': '', 'task_rep': '', 'task_condition': ''}


def _small_concat(root: Path, pattern: str, out_path: Path):
    """For the small files (thousands of rows) an in-memory concat is fine."""
    frames = []
    for f in sorted(root.glob(pattern)):
        if '_merged' in f.parts:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  WARN unreadable {f}: {e}")
            continue
        if df.empty:
            continue
        for k, v in _task_of(f).items():
            df[k] = v
        frames.append(df)
    if not frames:
        print(f"  NOTHING FOUND for {pattern}")
        return
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    print(f"  {out_path.name}: {len(out):,} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--expected-oof-folds', type=int, default=5,
                    help='The --oof-folds the jobs were submitted with. Cells whose '
                         'cross-fitting was truncated below this are flagged TRUNCATED_OOF.')
    ap.add_argument('--conditions', nargs='+', default=None,
                    help='The noise conditions the jobs were submitted with. Default: read '
                         'from the generated unc_*.sh, falling back to the main grid in '
                         'noise_conditions.json.')
    ap.add_argument('--kirby-dir', default='/data/stat-cadd/scat9264/KIRBy',
                    help='KIRBy checkout, read to find how many noise levels each '
                         '(dataset, condition) should have. Without it the level count is '
                         'reported but not judged.')
    ap.add_argument('--parquet', action='store_true',
                    help='Write partitioned Parquet instead of one big CSV (needs pyarrow).')
    args = ap.parse_args()
    root = Path(args.root)
    expected_oof = args.expected_oof_folds
    conditions, conditions_source = expected_conditions(args.conditions)
    n_levels = expected_level_counts(args.kirby_dir)
    print(f"Expected conditions ({conditions_source}): {', '.join(conditions)}")
    if n_levels is None:
        print(f"  NOTE could not read the level grids from {args.kirby_dir} — the coverage "
              f"report will show each cell's level count but not flag a short one.")
    out = root / '_merged'
    out.mkdir(parents=True, exist_ok=True)

    print("Small tables:")
    _small_concat(root, '*/*/all_results.csv', out / 'all_results.csv')
    # A task killed by the wall clock never writes all_results.csv -- only the
    # per-fold checkpoint. Collect those separately so nothing is invisible.
    _small_concat(root, '*/*/all_results_partial.csv', out / 'all_results_partial.csv')
    _small_concat(root, '*/*/summary.csv', out / 'summary.csv')

    unc_files = [f for f in sorted(root.glob('*/*/*_uncertainty_values.csv'))
                 if '_merged' not in f.parts]
    print(f"\nUncertainty: {len(unc_files)} task files, streaming")

    # A single header is written and every later chunk appended, so the column
    # set MUST be identical for every chunk. Task files can legitimately differ
    # (a task whose out-of-fold block was skipped has fewer columns), and
    # appending a narrower chunk under a wider header silently shifts every
    # column after the missing one. Build the union up front and reindex.
    COLUMNS = []
    for f in unc_files:
        try:
            cols = list(pd.read_csv(f, nrows=0).columns)
        except Exception:
            continue
        for c in cols:
            if c not in COLUMNS:
                COLUMNS.append(c)
    COLUMNS += [c for c in ('task_model', 'task_dataset', 'task_rep', 'task_condition')
                if c not in COLUMNS]
    # The four columns added 2026-08-28: the two halves of the uncertainty and,
    # for each, whether it varies per molecule or is one number per fit. The
    # union above already carries them through without shifting anything, and a
    # task file written before they existed is padded with blanks -- which reads
    # exactly like a model that cannot produce them. Say which it is, here,
    # rather than leaving the reader to guess from an empty column
    # (RERUN_PLAN.md 5.5).
    _SPLIT_COLUMNS = ('aleatoric_uncertainty', 'epistemic_uncertainty',
                      'aleatoric_support', 'epistemic_support')
    _with_split = 0
    for f in unc_files:
        try:
            cols = set(pd.read_csv(f, nrows=0).columns)
        except Exception:
            continue
        if set(_SPLIT_COLUMNS) <= cols:
            _with_split += 1
    print(f"  column union across task files: {len(COLUMNS)} columns")
    print(f"  uncertainty split: {_with_split}/{len(unc_files)} task files carry "
          f"all four component columns"
          + ("" if _with_split == len(unc_files)
             else "  <-- the rest predate them and are padded with blanks, "
                  "which is NOT the same as a model that has no split"))
    _ragged = set()

    big = out / 'uncertainty.csv'
    if big.exists():
        big.unlink()

    # coverage counters, accumulated per file so nothing large is ever resident
    cov = defaultdict(lambda: {'test_rows': 0, 'oof_rows': 0, 'oof_finite': 0,
                               'folds': set(), 'sigmas': set(), 'files': 0,
                               'oof_folds_min': None})
    header_written = False
    total_rows = 0
    pq_writer = None

    for i, f in enumerate(unc_files, 1):
        meta = _task_of(f)
        try:
            reader = pd.read_csv(f, chunksize=CHUNK)
        except Exception as e:
            print(f"  WARN unreadable {f}: {e}")
            continue
        for chunk in reader:
            if chunk.empty:
                continue
            for k, v in meta.items():
                chunk[k] = v
            missing = [c for c in COLUMNS if c not in chunk.columns]
            if missing:
                _ragged.add((f.name, tuple(missing)))
            chunk = chunk.reindex(columns=COLUMNS)

            key = (meta['task_dataset'], chunk['model'].iloc[0] if 'model' in chunk else meta['task_model'],
                   meta['task_rep'], meta['task_condition'])
            c = cov[key]
            c['files'] = 1
            if 'split' in chunk:
                is_test = chunk['split'] == 'test'
                c['test_rows'] += int(is_test.sum())
                oof = chunk[~is_test]
                c['oof_rows'] += len(oof)
                if 'uncertainty' in oof:
                    c['oof_finite'] += int(oof['uncertainty'].notna().sum())
                # A cross-fitting pass where some inner folds failed is written
                # as a normal block; the only marker is this column.
                if 'oof_folds_ok' in oof and len(oof):
                    v = pd.to_numeric(oof['oof_folds_ok'], errors='coerce')
                    v = v[v >= 0]
                    if len(v):
                        m = int(v.min())
                        c['oof_folds_min'] = m if c['oof_folds_min'] is None \
                            else min(c['oof_folds_min'], m)
            if 'fold' in chunk:
                c['folds'].update(chunk['fold'].dropna().unique().tolist())
            if 'sigma' in chunk:
                c['sigmas'].update(chunk['sigma'].dropna().unique().tolist())

            if args.parquet:
                import pyarrow as pa, pyarrow.parquet as pq
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if pq_writer is None:
                    pq_writer = pq.ParquetWriter(out / 'uncertainty.parquet', table.schema)
                pq_writer.write_table(table)
            else:
                chunk.to_csv(big, mode='a', index=False, header=not header_written,
                             quoting=csv.QUOTE_MINIMAL)
                header_written = True
            total_rows += len(chunk)
        if i % 25 == 0 or i == len(unc_files):
            print(f"  {i}/{len(unc_files)} files, {total_rows:,} rows")

    if pq_writer is not None:
        pq_writer.close()
    if _ragged:
        print(f"  NOTE {len(_ragged)} task file(s) were missing columns and were padded:")
        for name, miss in sorted(_ragged)[:10]:
            print(f"    {name}: missing {list(miss)}")
    print(f"  wrote {total_rows:,} rows to "
          f"{'uncertainty.parquet' if args.parquet else 'uncertainty.csv'}")

    # ---- coverage ---------------------------------------------------------
    rows = []
    for ds in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            slug = model.lower().replace('-', '_')
            for rep in EXPECTED_REPS:
                for cond in conditions:
                    c = cov.get((ds, model, rep, cond)) or cov.get((ds, slug, rep, cond))
                    want = n_levels(ds, cond) if n_levels else None
                    if c is None:
                        status, t, o, of, nf, ns, okf = 'MISSING', 0, 0, 0, 0, 0, None
                    else:
                        t, o, of = c['test_rows'], c['oof_rows'], c['oof_finite']
                        nf, ns = len(c['folds']), len(c['sigmas'])
                        okf = c['oof_folds_min']
                        status = ('NO_OOF' if o == 0 else
                                  'OOF_ALL_NAN' if of == 0 else
                                  'TRUNCATED_OOF' if (okf is not None and okf < expected_oof) else
                                  'PARTIAL_FOLDS' if nf < 5 else
                                  'PARTIAL_LEVELS' if (want and ns < want) else 'OK')
                    rows.append({'dataset': ds, 'model': model, 'rep': rep, 'condition': cond,
                                 'test_rows': t, 'oof_rows': o, 'oof_finite': of,
                                 'folds': nf, 'levels': ns, 'levels_expected': want,
                                 'oof_folds_min': okf, 'status': status})
    covdf = pd.DataFrame(rows)
    covdf.to_csv(out / 'coverage.csv', index=False)
    print(f"\ncoverage.csv: {len(covdf)} expected cells")
    print(covdf['status'].value_counts().to_string())
    bad = covdf[covdf['status'] != 'OK']
    if len(bad):
        print(f"\n{len(bad)} cells not OK — first 20:")
        print(bad.head(20).to_string(index=False))
    print(f"\nMerged into {out}")


if __name__ == '__main__':
    main()
