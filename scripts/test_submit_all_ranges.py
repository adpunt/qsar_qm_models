#!/usr/bin/env python
"""Every submit_all.sh submits every script at that script's OWN array range.

WHY THIS EXISTS. The array range is on the sbatch line, not in the job script, and it
differs between scripts in the same directory and between generations of the same
script:

  QM9 --stage 1     18 tasks, but `gauche` runs on ECFP4 alone and holds 3
  QM9 --stage 2     36 tasks, `gauche` 6
  QM9 censoring      6 tasks, `gauche` 1
  uncertainty       27 tasks on three conditions, 36 on the four that follow

A runbook or a hand-written loop that submits everything at one range over-queues the
short ones, and the out-of-range guard exits 2 -- so those indices land as FAILED, with
a mail alert each, and any coverage check reads them as missing cells. That has gone
wrong three times: the QM9 runbook, the uncertainty runbook at 0-62 against scripts
holding 27, and the deep run in RERUN_PLAN.md 13.19.

All three generators now write submit_all.sh. This checks that what it submits matches
what each script will accept, across every form of every generator, so the operator
never types a range again.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QM9 = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
LAB = ROOT / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
UNC = ROOT / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py'
DEEP = ROOT / 'deep_run_pairs.json'
CENS = ROOT / 'censoring_pairs.json'

FORMS = [
    ('QM9 --stage 0', QM9, ['--stage', '0', '--max-hours', '720']),
    ('QM9 --stage 1', QM9, ['--stage', '1', '--max-hours', '720']),
    ('QM9 deep run', QM9, ['--stage', '2', '--max-hours', '720',
                           '--runtime-selection', str(DEEP)]),
    ('QM9 censoring', QM9, ['--stage', '2', '--conditions', 'censoring',
                            '--max-hours', '720', '--runtime-selection', str(CENS)]),
    ('laboratory breadth', LAB, []),
    ('laboratory depth', LAB, ['--include-depth-conditions',
                               '--runtime-selection', str(DEEP)]),
    ('laboratory censoring', LAB, ['--conditions', 'censoring',
                                   '--runtime-selection', str(CENS)]),
    ('uncertainty, the three', UNC, []),
    ('uncertainty, the four that follow', UNC,
     ['--conditions', 'censoring', 'student_t_nu5', 'outlier_p10', 'laplace']),
]

def _selected_pairs(spec):
    """The (model, representation) pairs a selection file names — both shapes.

    The deep run's file is a model list crossed with a representation list; the
    censoring file names its pairs outright, because a cross product cannot express
    five. Same two shapes the generator reads.
    """
    named = spec.get('generator_pairs') or spec.get('pairs')
    if named:
        return [(p[0], p[1]) for p in named]
    return [(m, r) for m in spec.get('generator_labels', [])
            for r in spec.get('representations', [])]


def tasks_in(text):
    """The number of array tasks a generated script will accept.

    Read off the script's OWN `n_tasks=$(( ... ))` formula, not off a comment and not
    off an assumption about which arrays a pipeline happens to use: the laboratory
    multiplies representations by datasets, QM9 multiplies conditions by
    representations, and the uncertainty runs multiply all three. Whichever arrays the
    formula names, those are the ones counted.
    """
    # QM9 and the uncertainty runs call it n_tasks; the laboratory calls it n_task.
    formula = re.search(r'^n_tasks?=\$\(\(([^)]*)\)\)', text, re.M)
    if not formula:
        return None

    def count(name):
        m = re.search(r'^%s=\((.*?)\)$' % re.escape(name), text, re.M | re.S)
        return len(m.group(1).split()) if m else None

    n = 1
    for token in re.findall(r'\$\{#(\w+)\[@\]\}|\b(n_\w+)\b', formula.group(1)):
        array_ref, var_ref = token
        if array_ref:
            c = count(array_ref)
        else:
            # n_rep=${#REPS[@]} and friends: follow the indirection one step.
            alias = re.search(r'^%s=\$\{#(\w+)\[@\]\}' % re.escape(var_ref), text, re.M)
            c = count(alias.group(1)) if alias else None
        if c is None:
            return None
        n *= c
    return n if n > 1 or 'n_tasks' in formula.group(1) else n


def main():
    failures, checked = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for label, gen, extra in FORMS:
            out = Path(tmp) / re.sub(r'\W+', '_', label)
            out.mkdir(parents=True)
            proc = subprocess.run(
                [sys.executable, str(gen), *extra, '--out-dir', str(out)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f'{label}: the generator exited {proc.returncode}\n'
                                f'      {proc.stderr.strip()[-400:]}')
                continue
            sub = out / 'submit_all.sh'
            if not sub.exists():
                failures.append(f'{label}: wrote no submit_all.sh, so the ranges are '
                                f'left to whoever types the sbatch line')
                continue
            lines = re.findall(r'--array=0-(\d+)%\S*\s+([A-Za-z0-9_.-]+\.sh)',
                               sub.read_text())
            if not lines:
                failures.append(f'{label}: submit_all.sh submits nothing')
                continue
            job_scripts = {p.name for p in out.glob('*.sh')} - {
                'submit_all.sh', 'preflight.sh', 'smoke_test.sh',
                'resubmit_selected.sh'}
            submitted = {name for _, name in lines}
            for missing in sorted(job_scripts - submitted):
                failures.append(f'{label}: {missing} is generated but in no sbatch line')
            for last, name in lines:
                script = out / name
                if not script.exists():
                    failures.append(f'{label}: submit_all.sh submits {name}, which '
                                    f'was not written')
                    continue
                want = tasks_in(script.read_text())
                checked += 1
                if want is None:
                    failures.append(f'{label}: cannot read the task count out of '
                                    f'{name}; this check has gone blind')
                elif want != int(last) + 1:
                    failures.append(
                        f'{label}: submit_all.sh submits {name} at 0-{last} '
                        f'({int(last) + 1} tasks) but the script holds {want}; '
                        f'{abs(want - int(last) - 1)} task(s) differ')

            # THE WIDENING SUBMITTER, WHICH NAMES INDIVIDUAL INDICES.
            #
            # submit_all.sh sends a whole array and this file sends the tasks of one
            # model that a widened selection has just added. It is the only place in
            # the study where an index list is written out, so it is the only place a
            # typo could queue an out-of-range task -- the failure that happened three
            # times before submit_all.sh existed. Every index is checked against the
            # task count the SCRIPT ITSELF computes, and against that script's own
            # CONDS and REPS arrays, which is where the mapping actually lives.
            widen = out / 'resubmit_selected.sh'
            if widen.exists():
                for arr, name in re.findall(
                        r'--array=([0-9,]+)%\S*\s+([A-Za-z0-9_.-]+\.sh)',
                        widen.read_text()):
                    script = out / name
                    if not script.exists():
                        failures.append(f'{label}: resubmit_selected.sh submits '
                                        f'{name}, which was not written')
                        continue
                    body = script.read_text()
                    want = tasks_in(body)
                    idx = [int(i) for i in arr.split(',')]
                    checked += 1
                    over = [i for i in idx if want is not None and i >= want]
                    if over:
                        failures.append(
                            f'{label}: resubmit_selected.sh sends {name} task(s) '
                            f'{over}, and the script holds {want} -- an out-of-range '
                            f'task exits 2 and computes nothing')
                    if len(set(idx)) != len(idx):
                        failures.append(f'{label}: resubmit_selected.sh repeats an '
                                        f'index for {name}')
                    # The script maps index -> (rep, condition) itself. Rebuild that
                    # map and check every index really is a selected representation.
                    reps_m = re.search(r'^REPS=\((.*?)\)$', body, re.M)
                    conds_m = re.search(r'^CONDS=\((.*?)\)$', body, re.M)
                    sel_file = next((e for e in extra if e.endswith('.json')), None)
                    if reps_m and conds_m and sel_file:
                        reps_l = reps_m.group(1).split()
                        conds_l = conds_m.group(1).split()
                        spec = json.loads(Path(sel_file).read_text())
                        model = name[len('qm9_s2_'):-len('.sh')] \
                            if name.startswith('qm9_s2_') else None
                        want_reps = {r for m, r in _selected_pairs(spec)
                                     if m == model and r in reps_l}
                        got = {(reps_l[i % len(reps_l)],
                                conds_l[i // len(reps_l)]) for i in idx}
                        expect = {(r, c) for r in want_reps for c in conds_l}
                        if want_reps and got != expect:
                            failures.append(
                                f'{label}: resubmit_selected.sh decodes {name} to '
                                f'{sorted(got - expect)} that the selection does not '
                                f'name, and misses {sorted(expect - got)}')

    if failures:
        print(f'FAIL — {len(failures)} problem(s):\n')
        for f in failures:
            print(f'  - {f}')
        return 1
    print(f'PASS — {checked} sbatch lines across {len(FORMS)} generator forms, every '
          f'one at its own script\'s task count.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
