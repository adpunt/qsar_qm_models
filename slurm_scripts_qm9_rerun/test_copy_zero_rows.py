#!/usr/bin/env python
"""A copy of a moving reference is not a seed divergence, and must not stop the run.

WHY THIS EXISTS. On 2026-09-06 `copy_zero_rows.py` stopped with nine DISAGREES on
chemberta/rf and refused to do anything further. Nothing was wrong with the data.

What happened: a copied clean row is written into the target with the same columns and
values as a computed one, and nothing marks it as a copy -- the only record is
zero_row_copies.csv beside the results. So on the second run every copy was read back as
a row "the job actually computed" and checked against the reference. That is safe only
while the reference stands still, and it does not: the DEEP RUN recomputes gaussian's
clean level for the pairs it selects (`--stage 2` gives gaussian seven levels starting at
0.0, over replicates 0-9) into the same anova_gaussian_*.csv the screen wrote. The rf
deep-run array had finished hours earlier, the reference moved, and every copy made
before it looked like a divergence.

So this drives the real script over a fixture in three states:

  1. first run, nothing copied yet          -> copies, exit 0
  2. reference re-run, copies now stale     -> REFRESHES them, exit 0, values updated
  3. a row that was never a copy disagrees  -> DISAGREES, REFUSES that file, exit 1,
                                               and every other file is still filled
  4. the reference holds two clean blocks   -> a target matching the OLDER one agrees

State 3 is the guard's actual purpose and must survive the fix for state 2.

STATES 3 AND 4 CHANGED 2026-09-07. A disagreement used to stop the whole run, and the
comparison used to be against the newest clean row alone. The deep run recomputes
gaussian's clean level into the file the screen wrote, so a reference file holds two
clean blocks and a target written by the screen matches the older one. That was read as
a divergence, and because one counter stopped everything, thirteen combinations of
model, representation and noise type were left with no clean row at all -- and auc_norm
is retention against that row (RERUN_PLAN.md 13.28).
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'copy_zero_rows.py'
COLUMNS = ['dataset', 'model', 'rep', 'noise_type', 'sigma', 'iteration',
           'mae', 'mse', 'rmse', 'r2', 'pearson_corr']


def row(condition, it, r2, sigma='0.0'):
    return dict(dataset='qm9', model='rf', rep='chemberta', noise_type=condition,
                sigma=sigma, iteration=str(it), mae='0.1', mse='0.02', rmse='0.14',
                r2=str(r2), pearson_corr='0.9')


def write(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def clean_rows(path):
    with open(path, newline='') as fh:
        return {r['iteration']: r for r in csv.DictReader(fh)
                if float(r['sigma']) == 0.0}


def run(results, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), '--results', str(results), *extra],
        capture_output=True, text=True)


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp)
        ref = results / 'anova_gaussian_chemberta_rf.csv'
        tgt = results / 'anova_grouped_wider_chemberta_rf.csv'

        # 1. First run: the reference has clean rows, the target has none.
        write(ref, [row('gaussian', i, 0.800 + i / 1000) for i in range(3)]
                   + [row('gaussian', i, 0.7, sigma='0.5') for i in range(3)])
        write(tgt, [row('grouped_wider', i, 0.7, sigma='0.5') for i in range(3)])
        p = run(results)
        if p.returncode != 0:
            failures.append(f'first run exited {p.returncode}\n{p.stdout[-500:]}')
        got = clean_rows(tgt)
        if len(got) != 3:
            failures.append(f'first run copied {len(got)} clean row(s), expected 3')
        if not (results / 'zero_row_copies.csv').exists():
            failures.append('no zero_row_copies.csv, so a later run cannot tell a copy '
                            'from a computed row -- which is the whole defect')

        # 2. The deep run recomputes gaussian's clean level. The reference moves.
        write(ref, [row('gaussian', i, 0.900 + i / 1000) for i in range(3)]
                   + [row('gaussian', i, 0.7, sigma='0.5') for i in range(3)])
        p = run(results)
        if p.returncode != 0:
            failures.append(
                f'a moved reference stopped the run (exit {p.returncode}); a copy is not '
                f'evidence and must be refreshed, not reported as a divergence\n'
                f'{p.stdout[-700:]}')
        if 'REFRESHED' not in p.stdout:
            failures.append('the run did not say it refreshed anything, so a silently '
                            'changed number is now on disk')
        got = clean_rows(tgt)
        stale = [i for i, r in got.items()
                 if abs(float(r['r2']) - (0.900 + int(i) / 1000)) > 1e-12]
        if stale:
            failures.append(f'replicate(s) {stale} still hold the old value after the '
                            f'refresh, so the target disagrees with the reference')

        # 3. A clean row that was NEVER copied, and disagrees. The guard must fire
        # for THAT file, and the file after it must still be filled.
        other = results / 'anova_grouped_shifted_chemberta_rf.csv'
        write(other, [row('grouped_shifted', 0, 0.111)]
                     + [row('grouped_shifted', i, 0.7, sigma='0.5') for i in range(3)])
        innocent = results / 'anova_student_t_nu5_chemberta_rf.csv'
        write(innocent, [row('student_t_nu5', i, 0.7, sigma='0.5') for i in range(3)])
        p = run(results)
        if p.returncode == 0:
            failures.append('a computed clean row disagreed with the reference and the '
                            'run exited 0 -- that is the check this script exists for')
        if 'DISAGREES' not in p.stdout:
            failures.append('a genuine disagreement was not reported')
        if 'REFUSING' not in p.stdout:
            failures.append('a genuine disagreement did not refuse its own file')
        if len(clean_rows(other)) != 1:
            failures.append('the refused file was written into anyway')
        if len(clean_rows(innocent)) != 3:
            failures.append(
                f'one file disagreed and a DIFFERENT file was left with '
                f'{len(clean_rows(innocent))} clean row(s) instead of 3 -- one bad '
                f'combination must not stop the rest, which is how thirteen '
                f'combinations were left with nothing to divide by')

        # 4. TWO clean blocks in the reference, which is what the deep run leaves
        # behind. A target holding the OLDER block's values came from that run and
        # agrees; only a value in neither block is a divergence.
        results2 = results / 'two_blocks'
        results2.mkdir()
        ref2 = results2 / 'anova_gaussian_chemberta_rf.csv'
        tgt2 = results2 / 'anova_grouped_wider_chemberta_rf.csv'
        write(ref2, [row('gaussian', i, 0.800 + i / 1000) for i in range(3)]
                    + [row('gaussian', i, 0.900 + i / 1000) for i in range(3)]
                    + [row('gaussian', i, 0.7, sigma='0.5') for i in range(3)])
        write(tgt2, [row('grouped_wider', 0, 0.800)]
                    + [row('grouped_wider', i, 0.7, sigma='0.5') for i in range(3)])
        p = run(results2)
        if 'DISAGREES' in p.stdout:
            failures.append(
                'the target holds the value of the reference\'s FIRST clean block, so '
                'it came from that run and agrees -- it was reported as a divergence')
        if p.returncode != 0:
            failures.append(f'a reference with two clean blocks exited {p.returncode}')
        if len(clean_rows(tgt2)) != 3:
            failures.append(
                f'the two remaining replicates were not copied; the target has '
                f'{len(clean_rows(tgt2))} clean row(s) instead of 3')

    if failures:
        print(f'FAIL — {len(failures)} problem(s):\n')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('PASS — copies are refreshed when the reference is re-run, a genuine '
          'disagreement refuses its own file and no other, and a target matching an '
          'earlier clean block of the reference is not a divergence.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
