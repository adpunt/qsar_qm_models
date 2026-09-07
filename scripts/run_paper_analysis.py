#!/usr/bin/env python
"""Every number and every figure in the paper, from one script.

    python scripts/run_paper_analysis.py --qm9-dir results \
        --validation-dir <KIRBy>/results/validation_rerun \
        --uncertainty-dir <KIRBy>/results/uncertainty_rerun \
        --output-dir results/paper_figures --only decisions

STAGE 1 IS THE DECISION REPORT, AND IT RUNS FIRST ON PURPOSE
------------------------------------------------------------
RERUN_PLAN.md 14.6 lists fourteen figures that exist only if the results say so,
and 14.9 eight open choices. F3's panel count, F7's option and which
representation is held constant are all inputs to DRAWING, so they are settled
from the numbers before a single figure is built. `--only decisions` does that
and nothing else.

THE NAME
--------
`generate_paper_figures.py` is TAKEN: it is the dead v1 script, 4,688 lines and
519 mentions of the retired slope metric. `generate_paper_figures_v2.py` is the
one that currently produces the paper. RERUN_PLAN.md 5.4 says the end state is
one file under the plain name with no versioning -- so when the author retires
those two, this becomes `generate_paper_figures.py`. Until then it has its own
name and destroys nothing.

WHAT THIS REPLACES
------------------
It supersedes `generate_paper_figures_v2.py`, which is left on disk untouched
until this has run on real cluster data. Five things it does are not carried
over:

  * it averaged the ten replicates before integrating the retention curve, so no
    robustness number had a spread -- and it separately computed the SAME metric
    per replicate elsewhere, under the same name;
  * it averaged over representation and noise type together to build the
    cross-dataset figure, up to 36 values per bar;
  * it ranked models on representation-averaged means for Kendall's W;
  * it pooled every noise level and every replicate into the uncertainty table's
    headline columns;
  * it took the variance decomposition's noise level from a default argument
    that no call site ever passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _preflight():
    """Turn the one environment failure this hits into a sentence.

    scipy's compiled extensions are built against the conda environment's
    libstdc++, which is newer than the one in /lib64 on ARC. Without the
    environment's own lib directory ahead of the system one, `from scipy import
    stats` dies forty lines deep in scipy.optimize with

        ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found

    which says nothing about what to do. `setup.sh` sets the path and every
    generated job script sets it again; a shell that ran `conda activate` alone
    has not. Checked here, before any of this module's imports, so the answer
    arrives instead of the traceback.
    """
    try:
        from scipy import stats  # noqa: F401
    except ImportError as exc:
        text = str(exc)
        if 'GLIBCXX' not in text and 'libstdc++' not in text:
            raise
        prefix = os.environ.get('CONDA_PREFIX', '<your env>')
        sys.stderr.write(
            '\nThis is the environment, not the analysis.\n\n'
            f'  {text}\n\n'
            "scipy's compiled parts need the conda environment's libstdc++, "
            'which is newer\nthan the one in /lib64. Put the environment '
            'ahead of the system:\n\n'
            f'    export LD_LIBRARY_PATH="{prefix}/lib:$LD_LIBRARY_PATH"\n\n'
            'Better, source the one line that sets every path this run needs:'
            '\n\n    . /data/stat-cadd/scat9264/qsar_qm_models/scripts/'
            'runenv.sh\n\n'
            'It sources setup.sh (which is what sets the library path) and '
            'exports\nQSAR, KIRBY, SEL, CEN, ACCT and PART with it.\n\n'
            'A shell that ran `conda activate` on its own has NOT set this.\n')
        raise SystemExit(3)


_preflight()

import pandas as pd  # noqa: E402

import figlib_config as C  # noqa: E402
import figlib_decisions as D  # noqa: E402
import figlib_guard as G  # noqa: E402
import figlib_load as L  # noqa: E402
import figlib_metrics as M  # noqa: E402
import figlib_figures as FIG  # noqa: E402
import figlib_uncertainty as U  # noqa: E402

STAGE_2_MESSAGE = (
    'The figures and tables are Stage 2 and are not built yet. Stage 1 is the '
    'guard, the loaders, the metrics and the decision report, because F3\'s '
    'panel count, F7\'s option and the held-constant representation are all '
    'inputs to drawing and are settled from the numbers first '
    '(RERUN_PLAN.md 14.9). Run with --only decisions.')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--qm9-dir', default=str(C.ROOT / 'results'),
                   help='where the anova_*.csv live')
    p.add_argument('--validation-dir', action='append', default=None,
                   help='a validation_rerun tree; repeatable')
    p.add_argument('--uncertainty-dir', action='append', default=None,
                   help='an uncertainty_rerun tree; repeatable')
    p.add_argument('--output-dir', default=str(C.ROOT / 'results'
                                               / 'paper_figures'))
    p.add_argument('--primary-rep', default=None,
                   help='the representation held constant where a figure must '
                        'hold one. Open (RERUN_PLAN.md 14.9 item 5); D1 lays '
                        'out the evidence and this flag applies the answer')
    p.add_argument('--cache-dir', default=None,
                   help='parquet cache for the parsed results')
    p.add_argument('--only', default='decisions',
                   choices=['decisions', 'figures', 'tables', 'all'])
    p.add_argument('--permutations', type=int, default=200,
                   help='permutations for the Q4 null band; 0 skips it')
    p.add_argument('--no-filters', action='store_true',
                   help='skip the declared filters, to see what they carry')
    p.add_argument('--skip-uncertainty', action='store_true',
                   help='the accuracy half only. The uncertainty statistics '
                        'read every per-molecule file, which is the expensive '
                        'part; this gets the other eight answers in seconds')
    p.add_argument('--max-uncertainty-files', type=int, default=None,
                   help='read only the first N per-molecule files. Makes the '
                        'uncertainty answers PARTIAL, and the run says so')
    return p.parse_args(argv)


def load_everything(args):
    print('[1/3] reading')
    qm9 = L.load_qm9(args.qm9_dir, cache_dir=args.cache_dir)
    assay = L.load_assay_accuracy(args.validation_dir,
                                  cache_dir=args.cache_dir)
    merged = L.load_merged_uncertainty(args.uncertainty_dir)

    # NOT loaded here. Every row is one molecule at one noise level in one
    # fold, so the whole set is hundreds of millions of rows and reading it into
    # one frame is what a login node kills. The statistics are computed one file
    # at a time, in run_decisions.
    per_molecule = [args.qm9_dir] + list(args.uncertainty_dir or []) \
        + list(args.validation_dir or [])

    # Anything the loader flagged as needing a human decision travels on the
    # frame; write it where it can be opened rather than leaving it in a log.
    for label, frame in (('qm9', qm9), ('assay', assay)):
        detail = (frame.attrs.get('duplicate_disagreements')
                  if frame is not None else None)
        if detail is not None and len(detail):
            path = Path(args.output_dir) / f'd0_duplicate_disagreements_{label}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            detail.to_csv(path, index=False)
            print(f'  {len(detail)} row(s) from disagreeing duplicate cells '
                  f'written to {path.name} -- both copies of each, side by '
                  f'side, so the cause can be chased')

    for name, frame in (('QM9', qm9), ('assay', assay)):
        print(f'  {name}: '
              + ('nothing' if frame is None else
                 f'{len(frame)} rows, {frame["model"].nunique()} models, '
                 f'{frame["rep"].nunique()} representations, '
                 f'{frame["condition"].nunique()} conditions'))
    found = U.discover(per_molecule)
    print(f'  per-molecule: {len(found)} file(s), read one at a time')
    return qm9, assay, merged, per_molecule


def apply_declared_filters(frame, where, skip=False):
    """Guard 8: the filters are declared, applied, and logged -- never silent."""
    if frame is None or not len(frame) or skip:
        return frame, pd.DataFrame()
    log = []
    keys = [c for c in ('dataset', 'model', 'rep', 'condition', 'replicate')
            if c in frame.columns]
    for filt in (L.catastrophic_filter(), L.collapsed_gp_filter()):
        frame, dropped = filt.apply(frame)
        cells = (dropped[keys].drop_duplicates() if len(dropped)
                 else pd.DataFrame(columns=keys))
        log.append({'source': where, 'filter': filt.name, 'reason': filt.reason,
                    'rows_dropped': int(len(dropped)),
                    'cells_dropped': int(len(cells))})
        if len(dropped):
            # WHICH, not just how many. "dropped 7 rows" on a seven-level ladder
            # is one whole replicate of one cell, and knowing which one is the
            # difference between "a network diverged once" and "a model is
            # failing everywhere".
            print(f'  {where}: {filt.name} dropped {len(dropped)} row(s) '
                  f'= {len(cells)} whole replicate(s) of:')
            for row in cells.head(6).itertuples(index=False):
                print('      ' + ' / '.join(str(v) for v in row))
            if len(cells) > 6:
                print(f'      ... and {len(cells) - 6} more')
    return frame, pd.DataFrame(log)


def run_decisions(args, qm9, assay, merged, per_molecule):
    print('[2/3] deciding')
    tables, verdicts = {}, []

    def collect(result):
        got, verdict = result
        tables.update(got)
        verdicts.append(verdict)
        mark = 'FIRED' if verdict['fired'] else '    -'
        print(f'  {mark}  {verdict["id"]}: {verdict["says"][:110]}')

    collect(D.d0_coverage(qm9, assay, merged.get('coverage')))

    qm9, qm9_log = apply_declared_filters(qm9, 'QM9', args.no_filters)
    assay, assay_log = apply_declared_filters(assay, 'assay', args.no_filters)
    filter_log = pd.concat([f for f in (qm9_log, assay_log) if len(f)],
                           ignore_index=True) if (len(qm9_log) or len(assay_log)) \
        else pd.DataFrame()
    if len(filter_log):
        tables['d0_filters_applied'] = filter_log

    qm9_per, qm9_excluded = (M.robustness(qm9) if qm9 is not None
                             else (pd.DataFrame(), pd.DataFrame()))
    assay_per, assay_excluded = (M.robustness(assay) if assay is not None
                                 else (pd.DataFrame(), pd.DataFrame()))
    qm9_summary = M.summarise_robustness(qm9_per)
    assay_summary = M.summarise_robustness(assay_per)
    if len(qm9_summary):
        tables['auc_norm_qm9'] = G.with_components(qm9_summary, 'auc_norm_qm9')
    if len(assay_summary):
        tables['auc_norm_assay'] = G.with_components(assay_summary,
                                                     'auc_norm_assay')
    for name, frame in (('excluded_qm9', qm9_excluded),
                        ('excluded_assay', assay_excluded)):
        if len(frame):
            tables[name] = frame

    primary = args.primary_rep
    if primary is None and len(qm9_summary):
        counts = qm9_summary.groupby('rep')['model'].nunique()
        primary = counts.idxmax()
        print(f'  no --primary-rep given; using {primary!r} (the widest '
              f'coverage) for the tests that must hold one. D1 is the evidence.')

    collect(D.d1_representation(qm9_summary, qm9))
    collect(D.d2_grid_similarity(qm9_summary, qm9_per))
    if primary is not None and len(qm9_summary):
        collect(D.d3_condition_separation(qm9_per, qm9_summary, primary))

    anova_rows = []
    if len(qm9_per):
        anova_rows.append(M.two_way_eta2_by_condition(qm9_per, 'auc_norm')
                          .assign(outcome='robustness (AUC_norm)'))
        at_level = M.accuracy_at_reporting_level(qm9, 'qm9')
        if len(at_level):
            anova_rows.append(
                M.two_way_eta2_by_condition(at_level, 'r2')
                .assign(outcome='performance (R2 at the reporting level)'))
    anova = pd.concat(anova_rows, ignore_index=True) if anova_rows \
        else pd.DataFrame()
    if len(anova):
        tables['anova_eta2'] = anova
    collect(D.d4_interaction(anova))
    collect(D.d5_representation_outlier(qm9_summary))
    collect(D.d6_auc_above_one([qm9_summary, assay_summary],
                              [qm9_per, assay_per]))

    support = slopes = q4 = q6 = None
    if args.skip_uncertainty:
        print('  uncertainty skipped (--skip-uncertainty); D7 and D8 will say '
              'so rather than report a null nothing measured')
    else:
        stats = U.statistics(per_molecule, permutations=args.permutations,
                             max_files=args.max_uncertainty_files)
        if stats:
            support = stats.get('support')
            q4 = stats.get('q4')
            q6 = stats.get('q6')
            slopes = stats.get('slopes')
    collect(D.d7_uncertainty_option(q4, q6, support))
    collect(D.d8_decomposition(slopes, support))
    collect(D.d9_rank_transfer(qm9_summary, assay_summary))
    collect(D.d10_probabilistic(qm9_per))
    # Carried for the figures, not written out: the leading underscore keeps
    # them out of the CSV sweep in write_report.
    tables['_qm9_accuracy'] = qm9
    tables['_primary_rep'] = primary
    return tables, verdicts


def draw_figures(args, tables, verdicts):
    """The figures that need only the accuracy results.

    F6 and F7 need the uncertainty runs; they are drawn when those land. Each
    builder declares what it holds fixed and RAISES if the data still carries a
    factor it has not accounted for, so a failure here is a real one and is not
    caught.
    """
    out = Path(args.output_dir) / 'figures'
    print(f'[3/3] drawing into {out}')
    qm9 = tables.get('auc_norm_qm9')
    assay = tables.get('auc_norm_assay')
    accuracy = tables.get('_qm9_accuracy')
    anova = tables.get('anova_eta2')
    rep = args.primary_rep or tables.get('_primary_rep')

    said = {v['id']: v for v in verdicts}
    # D2 chose which conditions differ from each other; F3 shows those and the
    # rest become an additional file. The choice is the data's, not a taste.
    conditions = said.get('D2', {}).get('main_text') or []

    drawn = []
    if anova is not None and len(anova):
        drawn.append(FIG.f2_variance_decomposition(anova, out))
    if qm9 is not None and len(qm9) and conditions:
        drawn.append(FIG.f3_model_by_representation(qm9, out, conditions))
    if accuracy is not None and len(accuracy) and qm9 is not None and rep:
        drawn.append(FIG.f4_overview(accuracy, qm9, out, rep))
        drawn.append(FIG.r15_rank_against_level(accuracy, out, rep,
                                                'gaussian'))
        drawn.append(FIG.r16_decoupling(qm9, out, rep))
    if assay is not None and len(assay) and rep:
        drawn.append(FIG.f8_assay(assay, out, rep))

    drawn = [d for d in drawn if d]
    print(f'  {len(drawn)} figure(s). F6 and F7 wait on the uncertainty runs.')
    return drawn


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    # Unbuffered, because this runs for minutes and a SLURM log that shows
    # nothing until the end is indistinguishable from a hung job.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover
        pass
    C.apply_style()

    if args.only == 'tables':
        print(STAGE_2_MESSAGE)
        return 2

    qm9, assay, merged, per_molecule = load_everything(args)
    if qm9 is None and assay is None:
        print('\nNothing to analyse. Point --qm9-dir at the anova_*.csv and '
              '--validation-dir at a validation_rerun tree.')
        return 1

    tables, verdicts = run_decisions(args, qm9, assay, merged, per_molecule)

    if args.only in ('figures', 'all'):
        draw_figures(args, tables, verdicts)

    print('[3/3] writing')
    context = dict(C.provenance())
    context['qm9_dir'] = args.qm9_dir
    context['validation_dirs'] = ', '.join(args.validation_dir or []) or 'none'
    context['uncertainty_dirs'] = ', '.join(args.uncertainty_dir or []) or 'none'
    context['baseline_gate'] = C.BASELINE_THRESHOLD
    context['filters'] = 'skipped (--no-filters)' if args.no_filters \
        else 'catastrophic replicate, collapsed Gaussian process'
    path = D.write_report(verdicts, tables, args.output_dir, context)

    fired = [v['id'] for v in verdicts if v['fired']]
    print(f'  {len(tables)} table(s) and {path.name} in {args.output_dir}')
    print(f'  fired: {", ".join(fired) if fired else "nothing"}')
    print(f'  {time.time() - started:.1f}s')

    if args.only == 'all':
        print()
        print(STAGE_2_MESSAGE)
    return 0


if __name__ == '__main__':
    sys.exit(main())
