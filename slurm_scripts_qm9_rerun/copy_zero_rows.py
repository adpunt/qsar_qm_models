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


def previously_copied(results):
    """(target file, replicate) pairs this script wrote on an earlier run.

    WHY THIS IS NEEDED. A copied clean row is written into the target with the same
    columns and the same values as a computed one and NOTHING marks it as a copy -- the
    only record is zero_row_copies.csv beside the results. So on a second run every copy
    was read back as a row "the job actually computed" and checked against the reference.

    That is fine while the reference stands still. It does not. The DEEP RUN recomputes
    gaussian's clean level for the pairs it selects -- `--stage 2` gives gaussian seven
    levels, starting at 0.0, over replicates 0-9 -- into the same anova_gaussian_*.csv
    the screen wrote. When one of those lands, the reference moves, and every copy made
    before it is then reported as a seed divergence and stops the run. That happened on
    2026-09-06 on chemberta/rf, nine replicates of ten, hours after the deep run's rf
    array finished.

    A copy is not evidence about anything. It gets refreshed. A row that is NOT in this
    log and disagrees is the real thing the guard is for, and still stops the run.
    """
    log = Path(results) / LOG_NAME
    if not log.exists():
        return set()
    try:
        with open(log, newline='') as fh:
            return {(r['target'], r['iteration']) for r in csv.DictReader(fh)
                    if r.get('target') and r.get('iteration') is not None}
    except (OSError, KeyError, ValueError):
        return set()


def as_copy(row, condition, header):
    """The reference row, relabelled for the target condition."""
    new = dict(row)
    new['noise_type'] = condition
    # The dose delivered at level 0 is zero whatever the condition, and the censoring
    # axis measures a clipped fraction rather than a dose -- so the units column follows
    # the target, not the source.
    if 'level_units' in new and condition == 'censoring':
        new['level_units'] = 'fraction_censored'
    return {k: new.get(k, '') for k in header}


def rewrite(path, header, rows):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in header} for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=str(HERE.parent / 'results'),
                    help='Directory holding the anova_*.csv files.')
    ap.add_argument('--conditions', nargs='+', default=None,
                    help='Default: EVERY settled condition from noise_conditions.json, '
                         'the stage-2 depth-only ones included. It used to be the '
                         'stage-1 set alone, so student_t_nu5, outlier_p10 and laplace '
                         'never got a clean row -- and auc_norm is retention against '
                         'the clean point, so those three were dropped from every '
                         'ranking in silence. A condition that has not run yet simply '
                         'has nothing to copy into and is reported as such.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Say what would be written and write nothing.')
    args = ap.parse_args()

    results = Path(args.results)
    if not results.is_dir():
        raise SystemExit(f'no such directory: {results}')
    conditions = args.conditions or gen.STAGE2_CONDITIONS
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

    copied = written = checked = disagreed = skipped = refreshed = 0
    # WHY THIS IS COLLECTED RATHER THAN ONLY PRINTED. A refusal is the one line in
    # this output that needs an action, and it is printed in the middle of several
    # hundred SKIP lines for configurations that have not run yet. On 2026-09-07
    # the run reported "13 disagreed" in its summary and every one of the thirteen
    # names had scrolled away. They are now repeated at the end, beside the counts.
    refusals = []
    log_rows = []
    copies = previously_copied(results)

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

        # ONE FILE, MORE THAN ONE CLEAN BLOCK. The deep run recomputes gaussian's
        # clean level for the pairs it selects and APPENDS into the same
        # anova_gaussian_*.csv the screen wrote, so a replicate can appear twice
        # with slightly different numbers -- training on a different machine gives
        # a slightly different network. Comparing a target's computed clean row
        # against the newest block alone then calls the screen's own agreement a
        # divergence, which is what stopped the whole copy on 2026-09-07
        # (RERUN_PLAN.md 13.28). A target row is judged against EVERY clean row the
        # reference holds for that replicate: matching any of them means it came
        # from that run and agrees. Copies are still made from the newest.
        by_iteration = defaultdict(list)
        for r in clean:
            by_iteration[r['iteration']].append(r)
        clean = [rows[-1] for rows in by_iteration.values()]

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

            # A clean row the job actually COMPUTED is checked, never replaced. One this
            # script copied on an earlier run is REFRESHED -- it is not evidence about
            # anything, and the reference legitimately moves under it when the deep run
            # recomputes gaussian's clean level. See previously_copied().
            stale = []
            refused = False
            for row in clean:
                have = existing.get(row['iteration'])
                if have is None:
                    continue
                candidates = by_iteration[row['iteration']]
                if any(all(have.get(c) == candidate.get(c) for c in ACCURACY)
                       for candidate in candidates):
                    checked += 1
                    continue
                if (target.name, row['iteration']) in copies:
                    stale.append(row['iteration'])
                    continue
                differs = [c for c in ACCURACY if have.get(c) != row.get(c)]
                checked += 1
                disagreed += 1
                refused = True
                print(f"  DISAGREES  {target.name} replicate {row['iteration']}: "
                      f"{', '.join(differs)} differ from every clean row {source.name} "
                      f"holds for that replicate. The clean run is supposed to be "
                      f"identical across conditions -- something adds noise at level 0, "
                      f"or the seeds have diverged.")

            if stale:
                print(f"  REFRESHED  {target.name}: {len(stale)} clean row(s) this script "
                      f"copied earlier no longer match {source.name}, which has been re-run "
                      f"since -- the deep run recomputes gaussian's clean level. "
                      f"Replicate(s) {', '.join(str(i) for i in stale)}. NOT a seed "
                      f"divergence; a copy of a reference that moved.")
                if not args.dry_run:
                    fresh = {i: as_copy(r, condition, target_header)
                             for i in stale for r in clean if r['iteration'] == i}
                    rows_out = [fresh.get(r['iteration'], r)
                                if is_clean(r) and r['iteration'] in fresh else r
                                for r in target_rows]
                    rewrite(target, target_header, rows_out)
                refreshed += len(stale)

            # A DISAGREEMENT REFUSES ITS OWN FILE, NOT THE WHOLE COPY. `disagreed`
            # used to be one counter for the run, so the first bad file stopped
            # every file after it -- on 2026-09-07 one neural model on ChemBERTa
            # left every other condition without a clean row, and auc_norm is
            # retention against that row, so those conditions produced nothing at
            # all (RERUN_PLAN.md 13.28). The exit code is still 1, so nothing
            # passes silently.
            if refused:
                refusals.append((target.name, len(missing)))
                print(f"  REFUSING  {target.name}: nothing copied into this file until "
                      f"the disagreement above is explained. Every other file is "
                      f"unaffected and is still being filled.")
                continue
            if not missing:
                continue

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
    if refreshed:
        print(f"  {refreshed} earlier copy/copies refreshed from a reference that has "
              f"been re-run since -- listed above, none silently")
    if skipped:
        print(f"  {skipped} configuration(s) skipped -- listed above, none silently")

    if refusals:
        print()
        print(f"  {len(refusals)} file(s) REFUSED -- a clean row they COMPUTED matches "
              f"no clean row the reference holds. These are the only ones that need a "
              f"decision:")
        for name, still_missing in refusals:
            cost = (f"{still_missing} replicate(s) left with no clean row"
                    if still_missing else
                    "nothing was waiting on a copy here -- every replicate already "
                    "has its own computed clean row, so the refusal costs no data; "
                    "the disagreement itself is the finding")
            print(f"      {name}   {cost}")

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
