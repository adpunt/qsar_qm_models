#!/usr/bin/env python3
"""Fill in the clean row for every noise condition that did not run it.

WHY THERE IS ANYTHING TO COPY
-----------------------------
At noise level 0 the pipeline does not add noise at all -- process_and_train.py
switches the noise step off for every split -- and the replicate seed depends
only on the replicate number. So the clean run is bit-identical whichever
condition it is labelled with. Measured on 400 QM9 molecules, random forest on
ECFP4, all four stage-1 conditions:

    R2   = 0.7579128047581825      RMSE = 0.5176004014184159

to the last digit, in all four.

Running it once per condition costs 11% of the QM9 grid to recompute a number
that is already on disk. So the job scripts run the clean level under the
reference condition only, and this fills in the rest.

WHY IT CANNOT SIMPLY BE LEFT OUT
--------------------------------
auc_norm -- the retention measure the paper reports -- divides each condition's
accuracy curve by that same condition's accuracy at zero noise. A condition with
no clean row has nothing to divide by, so it would produce nothing at all rather
than produce something wrong.

WHAT THIS REFUSES TO DO
-----------------------
It will not overwrite a clean row that a job actually computed. If one is there
it CHECKS it against the reference instead, and says so -- which is the free
version of the four-way agreement test, on real production runs rather than a
400-molecule sample. It will not invent a row for a configuration whose
reference file is missing. It will not run twice over the same file.

    python slurm_scripts_qm9_rerun/copy_zero_rows.py --results ../results
    python slurm_scripts_qm9_rerun/copy_zero_rows.py --results ../results --dry-run
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_scripts as gen                                   # noqa: E402

# The accuracy columns that must match for a copy to be honest. The rest of the
# row -- the condition name, the file number, the delivered dose -- is per-run
# bookkeeping and is expected to differ.
ACCURACY = ('mae', 'mse', 'rmse', 'r2', 'pearson_corr')

LOG_NAME = 'zero_row_copies.csv'


def is_clean(row):
    try:
        return float(row['sigma']) == 0.0
    except (KeyError, TypeError, ValueError):
        return False


def parse_name(path, conditions, reps):
    """anova_<condition>_<rep>_<model>.csv -> (condition, rep, model), or None."""
    stem = path.stem
    if not stem.startswith('anova_') or '_uncertainty_values' in stem:
        return None
    rest = stem[len('anova_'):]
    for condition in sorted(conditions, key=len, reverse=True):
        if rest.startswith(condition + '_'):
            tail = rest[len(condition) + 1:]
            for rep in sorted(reps, key=len, reverse=True):
                if tail.startswith(rep + '_'):
                    return condition, rep, tail[len(rep) + 1:]
            return None
    return None


def read(path):
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(HERE.parent / 'results'),
                    help='Directory holding the anova_*.csv files.')
    ap.add_argument('--conditions', nargs='+', default=None,
                    help='Default: the stage-1 set from noise_conditions.json.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Say what would be written and write nothing.')
    args = ap.parse_args()

    results = Path(args.results)
    if not results.is_dir():
        raise SystemExit(f'no such directory: {results}')
    conditions = args.conditions or gen.STAGE1_CONDITIONS
    reference = gen.REFERENCE_CONDITION
    if reference not in conditions:
        raise SystemExit(f'the reference condition {reference!r} is not in {conditions}; there is '
                         f'nothing to copy from')

    # Index every results file by the configuration it belongs to.
    files = defaultdict(dict)
    for path in sorted(results.glob('anova_*.csv')):
        parsed = parse_name(path, conditions, gen.ALL_REPS)
        if parsed:
            condition, rep, model = parsed
            files[(rep, model)][condition] = path

    copied = written = checked = disagreed = skipped = 0
    log_rows = []

    for (rep, model), by_condition in sorted(files.items()):
        source = by_condition.get(reference)
        if source is None:
            print(f"  SKIP  {rep}/{model}: no {reference} file to copy from")
            skipped += 1
            continue
        header, source_rows = read(source)
        clean = [r for r in source_rows if is_clean(r)]
        if not clean:
            print(f"  SKIP  {rep}/{model}: {source.name} has no clean row")
            skipped += 1
            continue

        for condition in conditions:
            if condition == reference:
                continue
            target = by_condition.get(condition)
            if target is None:
                print(f"  SKIP  {rep}/{model}/{condition}: no results file")
                skipped += 1
                continue
            target_header, target_rows = read(target)
            if target_header != header:
                print(f"  SKIP  {target.name}: its columns differ from {source.name}")
                skipped += 1
                continue

            existing = {r['iteration']: r for r in target_rows if is_clean(r)}
            missing = [r for r in clean if r['iteration'] not in existing]

            # A clean row the job actually computed is CHECKED, never replaced.
            for row in clean:
                have = existing.get(row['iteration'])
                if have is None:
                    continue
                checked += 1
                differs = [c for c in ACCURACY if have.get(c) != row.get(c)]
                if differs:
                    disagreed += 1
                    print(f"  DISAGREES  {target.name} replicate {row['iteration']}: "
                          f"{', '.join(differs)} differ from {source.name}. The clean run is "
                          f"supposed to be identical across conditions -- something adds noise "
                          f"at level 0, or the seeds have diverged. Not copying anything here.")

            if not missing:
                continue
            if disagreed:
                print(f"  STOPPING: a computed clean row disagreed with the reference, so the "
                      f"premise this copy rests on is false. Fix that first.")
                return 1

            for row in missing:
                new = dict(row)
                new['noise_type'] = condition
                # The dose delivered at level 0 is zero whatever the condition,
                # and the censoring axis measures a clipped fraction rather than
                # a dose -- so the units column follows the target, not the source.
                if 'level_units' in new and condition == 'censoring':
                    new['level_units'] = 'fraction_censored'
                log_rows.append(dict(target=target.name, source=source.name,
                                     iteration=row['iteration'], rep=rep, model=model,
                                     condition=condition, r2=row.get('r2')))
                if not args.dry_run:
                    with open(target, 'a', newline='') as fh:
                        csv.DictWriter(fh, fieldnames=header).writerow(new)
                    written += 1
                copied += 1

    print()
    print(f"  {copied} clean row(s) {'would be ' if args.dry_run else ''}copied "
          f"({written} written)")
    print(f"  {checked} computed clean row(s) checked against the reference, "
          f"{disagreed} disagreed")
    if skipped:
        print(f"  {skipped} configuration(s) skipped -- listed above, none silently")

    if log_rows and not args.dry_run:
        log = results / LOG_NAME
        new_file = not log.exists()
        with open(log, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
            if new_file:
                writer.writeheader()
            writer.writerows(log_rows)
        print(f"  what was copied, and from where: {log}")

    return 1 if disagreed else 0


if __name__ == '__main__':
    sys.exit(main())
