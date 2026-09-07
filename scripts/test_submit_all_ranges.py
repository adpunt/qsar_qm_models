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
    out = []
    for pairs_key, models_key in (('generator_pairs', 'generator_labels'),
                                  ('validation_pairs', 'validation_labels')):
        named = spec.get(pairs_key)
        if named:
            out += [(p[0], p[1]) for p in named]
        else:
            out += [(m, r) for m in spec.get(models_key, [])
                    for r in spec.get('representations', [])]
    for p in spec.get('pairs', []):
        out.append((p[0], p[1]))
    return out


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
            # THE QM9 GENERATOR NAMES ITS SUBMITTER FOR THE STAGE, because stages 0,
            # 1 and 2 share one directory and a single `submit_all.sh` was overwritten
            # by whichever ran last. `submit_all.sh` is now the dispatcher and holds no
            # sbatch line at all, so the real one is `submit_all_s<stage>.sh`.
            staged = sorted(out.glob('submit_all_s*.sh'))
            sub = staged[0] if staged else out / 'submit_all.sh'
            if not sub.exists():
                failures.append(f'{label}: wrote no submitter, so the ranges are '
                                f'left to whoever types the sbatch line')
                continue
            lines = re.findall(r'--array=0-(\d+)%\S*\s+([A-Za-z0-9_.-]+\.sh)',
                               sub.read_text())
            if not lines:
                failures.append(f'{label}: submit_all.sh submits nothing')
                continue
            job_scripts = {p.name for p in out.glob('*.sh')
                           if not p.name.startswith('submit_all')} - {
                'preflight.sh', 'smoke_test.sh', 'resubmit_selected.sh'}
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
            # submit_all.sh sends a whole array; this file sends the tasks of one model
            # that a widened selection has just added. It is the only place in the
            # study where an index list is written out, so it is the only place a typo
            # could queue an out-of-range task -- the failure that happened three times
            # before submit_all.sh existed. Every index is checked against the task
            # count the SCRIPT ITSELF computes, and decoded back through that script's
            # own arrays: REPS by CONDS on QM9, REPS by DATASETS on the laboratory.
            widen = out / 'resubmit_selected.sh'
            sel_file = next((e for e in extra if e.endswith('.json')), None)
            if widen.exists() and sel_file:
                spec = json.loads(Path(sel_file).read_text())
                blocks = re.findall(
                    r'^# ADDED (.+?)$.*?--array=([0-9,]+)%\S*\s+([A-Za-z0-9_.-]+\.sh)',
                    widen.read_text(), re.M | re.S)
                if not blocks:
                    failures.append(f'{label}: resubmit_selected.sh names no model, so '
                                    f'a widened selection has nothing to submit with')
                for model, arr, name in blocks:
                    script = out / name
                    if not script.exists():
                        failures.append(f'{label}: resubmit_selected.sh submits {name}, '
                                        f'which was not written')
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
                    reps_m = re.search(r'^REPS=\((.*?)\)$', body, re.M)
                    outer_m = (re.search(r'^CONDS=\((.*?)\)$', body, re.M)
                               or re.search(r'^DATASETS=\((.*?)\)$', body, re.M))
                    if not (reps_m and outer_m):
                        failures.append(f'{label}: cannot read REPS and its partner out '
                                        f'of {name}; this check has gone blind')
                        continue
                    reps_l = reps_m.group(1).split()
                    outer_l = outer_m.group(1).split()
                    # The selection spells representations in lower case and the
                    # laboratory runner spells them ECFP4 / ChemBERTa. The run-time
                    # gate case-folds, so this does too.
                    fold = {r.lower(): r for r in reps_l}
                    want_reps = {fold[r.lower()] for m, r in _selected_pairs(spec)
                                 if m == model and r.lower() in fold}
                    # Only decode what is in range: an out-of-range index is already
                    # reported above, and indexing outer_l with it would raise here
                    # instead of failing the check.
                    got = {(reps_l[i % len(reps_l)], outer_l[i // len(reps_l)])
                           for i in idx if i // len(reps_l) < len(outer_l)}
                    expect = {(r, o) for r in want_reps for o in outer_l}
                    if not want_reps:
                        failures.append(
                            f'{label}: resubmit_selected.sh has a block for {model!r}, '
                            f'which {Path(sel_file).name} does not name')
                    elif got != expect:
                        failures.append(
                            f'{label}: resubmit_selected.sh decodes {name} to '
                            f'{sorted(got - expect)} that the selection does not name, '
                            f'and misses {sorted(expect - got)}')

        # TWO STAGES IN ONE DIRECTORY, which is what actually happens: the QM9 repair
        # regenerates stage 1 and then stage 2 into `slurm_scripts_qm9_rerun` so both
        # pick up a fix in process_and_train.py. Until 2026-09-07 the second run
        # overwrote the first's submitter, leaving 19 stage-2 arrays listed, none for
        # stage 1, and all 19 `qm9_s1_*.sh` on disk with nothing to submit them. Every
        # other check here generates into a fresh directory, so none of them saw it.
        shared = Path(tmp) / 'both_stages'
        shared.mkdir(parents=True)
        for extra in (['--stage', '1', '--max-hours', '720'],
                      ['--stage', '2', '--max-hours', '720',
                       '--runtime-selection', str(DEEP)]):
            proc = subprocess.run(
                [sys.executable, str(QM9), *extra, '--out-dir', str(shared)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f'two stages in one directory: the generator exited '
                                f'{proc.returncode}\n      '
                                f'{proc.stderr.strip()[-300:]}')
        for stage, n_tasks in (('1', 18), ('2', 36)):
            sub = shared / f'submit_all_s{stage}.sh'
            if not sub.exists():
                failures.append(f'two stages in one directory: no submit_all_s{stage}'
                                f'.sh, so stage {stage} cannot be submitted at all')
                continue
            names = re.findall(r'--array=0-\d+%\S*\s+([A-Za-z0-9_.-]+\.sh)',
                               sub.read_text())
            wrong = [n for n in names if not n.startswith(f'qm9_s{stage}_')]
            checked += len(names)
            if wrong:
                failures.append(f'two stages in one directory: submit_all_s{stage}.sh '
                                f'submits {wrong}, which belong to another stage')
            on_disk = sorted(p.name for p in shared.glob(f'qm9_s{stage}_*.sh'))
            for missing in sorted(set(on_disk) - set(names)):
                failures.append(f'two stages in one directory: {missing} is on disk '
                                f'and in no sbatch line')
        # And the dispatcher must refuse rather than pick one.
        disp = shared / 'submit_all.sh'
        if disp.exists():
            ran = subprocess.run(['bash', 'submit_all.sh'], cwd=str(shared),
                                 capture_output=True, text=True)
            if ran.returncode == 0:
                failures.append('two stages in one directory: submit_all.sh with no '
                                'stage exited 0, so it submitted one of them')
            for stage in ('1', '2'):
                if f'submit_all_s{stage}.sh' not in ran.stdout:
                    failures.append(f'two stages in one directory: submit_all.sh does '
                                    f'not name submit_all_s{stage}.sh, so the operator '
                                    f'is not told what to run')

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
