#!/usr/bin/env python3
"""Stitch the per-task uncertainty-run outputs together — WITHOUT loading them all.

Every array task writes its own results-root so concurrent tasks cannot race on
a shared file. This walks those directories and concatenates them.

  python merge_results.py --root /data/stat-cadd/scat9264/KIRBy/tests/results/uncertainty_rerun

Writes into <root>/_merged/:
  coverage.csv      what ran and what is missing — READ THIS FIRST
  all_results.csv   R2/RMSE/MAE per dataset x model x rep x strategy x sigma x fold (small)
  summary.csv       per-config robustness rows (small)
  uncertainty.csv   every per-molecule row (LARGE — streamed, never held in memory)

Why streaming: with all six strategies and the out-of-fold training split, one
LogD task writes roughly 11 sigmas x 5 folds x (1,000 test + 3,200 train) rows
= ~230,000 rows, and there are 504 tasks — on the order of 100 million rows and
tens of GB. Building one DataFrame would run out of memory on any normal node,
at the very end of a multi-day run. So the big file is appended chunk by chunk
and the coverage report is accumulated from per-file counts, never from the
merged file.

Pass --parquet to write a partitioned Parquet dataset instead (much smaller and
far faster to query); needs pyarrow.
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

EXPECTED_MODELS = ['QRF', 'NGBoost', 'GP',
                   'BNN-Full', 'VBLL-Full', 'MLP-BNN-Full', 'MLP-VBLL-Full']
EXPECTED_DATASETS = ['logd', 'caco2', 'herg_ki']
EXPECTED_REPS = ['ecfp4', 'pdv', 'sns', 'mhggnnpretrained']
EXPECTED_STRATEGIES = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']

CHUNK = 200_000


def _task_of(path: Path):
    """Task dir is <model_slug>__<dataset>__<rep_slug>__<strategy>."""
    for part in path.parts:
        if part.count('__') == 3:
            m, ds, rep, st = part.split('__')
            return {'task_model': m, 'task_dataset': ds,
                    'task_rep': rep, 'task_strategy': st}
    return {'task_model': '', 'task_dataset': '', 'task_rep': '', 'task_strategy': ''}


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
    ap.add_argument('--parquet', action='store_true',
                    help='Write partitioned Parquet instead of one big CSV (needs pyarrow).')
    args = ap.parse_args()
    root = Path(args.root)
    out = root / '_merged'
    out.mkdir(parents=True, exist_ok=True)

    print("Small tables:")
    _small_concat(root, '*/*/all_results.csv', out / 'all_results.csv')
    _small_concat(root, '*/*/summary.csv', out / 'summary.csv')

    unc_files = [f for f in sorted(root.glob('*/*/*_uncertainty_values.csv'))
                 if '_merged' not in f.parts]
    print(f"\nUncertainty: {len(unc_files)} task files, streaming")

    big = out / 'uncertainty.csv'
    if big.exists():
        big.unlink()

    # coverage counters, accumulated per file so nothing large is ever resident
    cov = defaultdict(lambda: {'test_rows': 0, 'oof_rows': 0, 'oof_finite': 0,
                               'folds': set(), 'sigmas': set(), 'files': 0})
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

            key = (meta['task_dataset'], chunk['model'].iloc[0] if 'model' in chunk else meta['task_model'],
                   meta['task_rep'], meta['task_strategy'])
            c = cov[key]
            c['files'] = 1
            if 'split' in chunk:
                is_test = chunk['split'] == 'test'
                c['test_rows'] += int(is_test.sum())
                oof = chunk[~is_test]
                c['oof_rows'] += len(oof)
                if 'uncertainty' in oof:
                    c['oof_finite'] += int(oof['uncertainty'].notna().sum())
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
    print(f"  wrote {total_rows:,} rows to "
          f"{'uncertainty.parquet' if args.parquet else 'uncertainty.csv'}")

    # ---- coverage ---------------------------------------------------------
    rows = []
    for ds in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            slug = model.lower().replace('-', '_')
            for rep in EXPECTED_REPS:
                for st in EXPECTED_STRATEGIES:
                    c = cov.get((ds, model, rep, st)) or cov.get((ds, slug, rep, st))
                    if c is None:
                        status, t, o, of, nf, ns = 'MISSING', 0, 0, 0, 0, 0
                    else:
                        t, o, of = c['test_rows'], c['oof_rows'], c['oof_finite']
                        nf, ns = len(c['folds']), len(c['sigmas'])
                        status = ('NO_OOF' if o == 0 else
                                  'OOF_ALL_NAN' if of == 0 else
                                  'PARTIAL_FOLDS' if nf < 5 else
                                  'PARTIAL_SIGMAS' if ns < 11 else 'OK')
                    rows.append({'dataset': ds, 'model': model, 'rep': rep, 'strategy': st,
                                 'test_rows': t, 'oof_rows': o, 'oof_finite': of,
                                 'folds': nf, 'sigmas': ns, 'status': status})
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
