#!/usr/bin/env python3
"""Prove that every job script this generator emits is one the pipeline accepts.

WHY THIS EXISTS
---------------
The previous generator emitted `--sigma` and `--noise-strategy` for weeks after
process_and_train.py started refusing both by name, and two representations that
parse_mmap raises on. Nobody noticed, because a generator whose output has never
been executed looks exactly like one that works: the Python runs, the files
appear, and the failure only happens on the cluster at submit time.

So this does not read the generator. It GENERATES, RUNS each script far enough
to see the command it builds, and feeds that command through the real argument
parser in scripts/process_and_train.py. If a flag is retired, misspelt or given
a value outside its choices, the parser says so here rather than in the queue.

WHAT IT COVERS
--------------
  * every model script, every array index, in stages 0, 1 and 2
  * the array index -> (condition, representation) mapping is a bijection
  * the output filename the figure script has to parse
  * every emitted command line, against the real parser
  * the guards: no partition, index out of range, unactivated environment,
    wrong environment, missing rust binary -- each must FAIL, and the run
    with everything satisfied must pass
  * the generator's own refusals, which never produce a script at all: a deep
    run with no models, a deep run missing a shortlisted model, and a
    pair-subset condition generated over the top of the main-grid scripts

WHAT IT DOES NOT COVER
----------------------
Whether a run produces sensible numbers. That needs a real training run; see
--end-to-end, which does exactly one, at a small molecule count.

    python slurm_scripts_qm9_rerun/test_generate_scripts.py
    python slurm_scripts_qm9_rerun/test_generate_scripts.py --end-to-end
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GENERATOR = HERE / 'generate_scripts.py'

sys.path.insert(0, str(HERE))
import generate_scripts as gen                                   # noqa: E402

# The flags process_and_train.py refuses BY NAME (:420-431), and the
# representations parse_mmap raises on (:1173). A generated script containing any
# of these cannot run, whatever else is true of it.
RETIRED_FLAGS = ['--sigma', '--distribution', '--noise-strategy', '--strategy-params',
                 '--bootstrapping']
DROPPED_REPS = ['mol2vec', 'randomized_smiles']


def build_stub(root: Path) -> Path:
    """A fake checkout that satisfies every guard in the job template.

    The point is to let the script run its own logic -- index arithmetic, the
    condition case statement, the filename -- without a cluster and without
    training anything. Every `python` call is captured instead of executed.
    """
    (root / 'env_test' / 'bin').mkdir(parents=True)
    (root / 'scripts').mkdir()
    (root / 'rust' / 'target' / 'release').mkdir(parents=True)

    # The processed QM9 cache. The template refuses to start without it, because
    # torch_geometric builds data_v3.pt on first access and takes no lock, so an
    # array submitted cold has 294 tasks writing one file. The stub only needs the
    # path to exist -- nothing here reads it -- and `guard_refuses_unprocessed_qm9`
    # below deletes it again to prove the refusal still fires.
    (root / 'data' / 'QM9' / 'processed').mkdir(parents=True)
    (root / 'data' / 'QM9' / 'processed' / 'data_v3.pt').write_text('stub')

    binary = root / 'rust' / 'target' / 'release' / 'rust_processor'
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)

    setup = root / 'setup.sh'
    setup.write_text(
        f'export CONDA_PREFIX="{root}/env_test"\n'
        f'export PATH="$CONDA_PREFIX/bin:$PATH"\n')

    shim = root / 'env_test' / 'bin' / 'python'
    shim.write_text(
        '#!/bin/sh\n'
        '# Record the call and succeed. One line per invocation.\n'
        'printf "%s\\n" "$*" >> "$ARGV_LOG"\n'
        'exit 0\n')
    shim.chmod(0o755)
    return root


def run_task(script: Path, stub: Path, index, extra_env=None):
    """Run one array task of a generated script against the stub.

    Returns (exit status, the command lines it would have run).
    """
    log = stub / f'argv_{os.getpid()}.log'
    if log.exists():
        log.unlink()
    env = dict(os.environ)
    env.update({'ARGV_LOG': str(log),
                'SLURM_JOB_PARTITION': 'medium',
                'SLURM_JOB_ID': '1',
                'PATH': f"{stub / 'env_test' / 'bin'}:{env['PATH']}"})
    if index is not None:
        env['SLURM_ARRAY_TASK_ID'] = str(index)
    env.pop('CONDA_PREFIX', None)
    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    proc = subprocess.run(['bash', str(script)], env=env,
                          capture_output=True, text=True, cwd=str(stub))
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


def generate(out_dir: Path, stub: Path, *args):
    cmd = [sys.executable, str(GENERATOR), '--out-dir', str(out_dir),
           '--qsar-dir', str(stub), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'generator failed: {cmd}\n{proc.stdout}\n{proc.stderr}')
    return sorted(out_dir.glob('*.sh'))


def parse_header(script: Path):
    text = script.read_text()
    conds = re.search(r'^CONDS=\((.*?)\)$', text, re.M).group(1).split()
    reps = re.search(r'^REPS=\((.*?)\)$', text, re.M).group(1).split()
    return conds, reps


def check_stage(stage_args, label, failures, checked):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = build_stub(tmp / 'stub')
        out = tmp / 'scripts'
        out.mkdir()
        scripts = generate(out, stub, *stage_args)
        if not scripts:
            failures.append(f'{label}: the generator wrote no scripts')
            return []

        emitted = []
        for script in scripts:
            text = script.read_text()
            for flag in RETIRED_FLAGS:
                # Word-boundary so --bootstrapping does not match inside a word,
                # and so --sigma does not match --sigmas.
                if re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', text):
                    failures.append(f'{label}: {script.name} contains retired flag {flag}')
            for rep in DROPPED_REPS:
                if re.search(r'(?<![\w])' + re.escape(rep) + r'(?![\w])', text):
                    failures.append(f'{label}: {script.name} names dropped representation {rep}')

            conds, reps = parse_header(script)
            seen = {}
            for index in range(len(conds) * len(reps)):
                proc, calls = run_task(script, stub, index)
                checked[0] += 1
                if proc.returncode != 0:
                    failures.append(f'{label}: {script.name} task {index} exited '
                                    f'{proc.returncode}\n{proc.stdout}\n{proc.stderr}')
                    continue
                train = [c for c in calls if 'process_and_train.py' in c]
                if len(train) != 1:
                    failures.append(f'{label}: {script.name} task {index} made '
                                    f'{len(train)} training calls, expected 1')
                    continue
                argv = shlex.split(train[0])
                rep = arg_value(argv, '-r')
                cond = conds[index // len(reps)]

                # Two tasks must never run the same noise configuration on the
                # same representation: that is a whole cell of the grid silently
                # traded for a duplicate. Keyed on what was actually emitted, not
                # on the index, or the check would be circular.
                cell = (noise_signature(argv), rep)
                if cell in seen:
                    failures.append(f'{label}: {script.name} tasks {seen[cell]} and {index} '
                                    f'both run {cell} -- the index mapping is not a bijection')
                seen[cell] = index

                # And the case statement must map the tag to the flags the
                # generator declares for it, so the filename means what it says.
                want_flags, want_levels, _, _ = gen.CONDITIONS[cond]
                got = ' '.join(noise_flags(argv))
                if got != want_flags:
                    failures.append(f'{label}: {script.name} task {index} is filed under '
                                    f'"{cond}" but emitted "{got}", not "{want_flags}"')
                if ' '.join(levels_of(argv)) != want_levels:
                    failures.append(f'{label}: {script.name} task {index} ran levels '
                                    f'{levels_of(argv)} under "{cond}", expected {want_levels}')

                out_path = arg_value(argv, '-f')
                model = script.stem.split('_', 2)[2]
                expect_stem = f'anova_{cond}_{rep}_{model}.csv'
                if Path(out_path).name != expect_stem:
                    failures.append(f'{label}: {script.name} task {index} writes '
                                    f'{Path(out_path).name}, expected {expect_stem}')
                emitted.append((script.name, index, argv))

            # Every task must be reachable and nothing beyond the last.
            proc, _ = run_task(script, stub, len(conds) * len(reps))
            if proc.returncode == 0:
                failures.append(f'{label}: {script.name} accepted an out-of-range task index')
        return emitted


NOISE_ARGS = ('--noise-shape', '--noise-targeting', '--nu', '--noise-lambda',
              '--group-fraction', '--group-variance-share', '--outlier-p', '--censor-side')


def arg_value(argv, flag):
    return argv[argv.index(flag) + 1]


def values_after(argv, flag):
    """Every value following a flag, up to the next flag. --noise-level takes a list."""
    i = argv.index(flag) + 1
    out = []
    while i < len(argv) and not argv[i].startswith('-'):
        out.append(argv[i])
        i += 1
    return out


def levels_of(argv):
    return values_after(argv, '--noise-level')


def noise_flags(argv):
    """The noise flags the case statement supplied, in the order it supplied them."""
    out = []
    for i, a in enumerate(argv):
        if a in NOISE_ARGS:
            out += [a, argv[i + 1]]
    return out


def noise_signature(argv):
    return tuple(noise_flags(argv)) + ('levels',) + tuple(levels_of(argv))


def check_guards(failures, checked):
    """Each guard must refuse. A guard that never fires is decoration."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = build_stub(tmp / 'stub')
        out = tmp / 'scripts'
        out.mkdir()
        script = generate(out, stub, '--stage', '0', '--models', 'rf')[0]

        cases = [
            ('no partition', {'SLURM_JOB_PARTITION': None}, None),
            ('index out of range', {}, 999),
        ]
        for name, env, index in cases:
            proc, _ = run_task(script, stub, index if index is not None else 0, env)
            checked[0] += 1
            if proc.returncode == 0:
                failures.append(f'guard "{name}" did not fire')

        # An environment that activates nothing, and one that activates the
        # wrong thing -- the two failures that produced jobs 12822693/12822694.
        setup = stub / 'setup.sh'
        original = setup.read_text()
        for name, body in [
            ('unactivated environment', 'true\n'),
            ('wrong environment',
             f'export CONDA_PREFIX="{stub}/env_other"\n'
             f'export PATH="{stub}/env_test/bin:$PATH"\n'),
        ]:
            setup.write_text(body)
            proc, _ = run_task(script, stub, 0)
            checked[0] += 1
            if proc.returncode == 0:
                failures.append(f'guard "{name}" did not fire')
        setup.write_text(original)

        # And the rust binary the held-out-noise fix lives in.
        binary = stub / 'rust' / 'target' / 'release' / 'rust_processor'
        binary.chmod(0o644)
        proc, _ = run_task(script, stub, 0)
        checked[0] += 1
        if proc.returncode == 0:
            failures.append('guard "rust binary not executable" did not fire')
        binary.chmod(0o755)

        # The processed QM9 cache. torch_geometric builds data_v3.pt on first
        # access and takes NO lock, so an array submitted with the directory
        # cleared has 294 tasks writing one file: 294x the work at best, and at
        # worst a task loading a .pt another is still writing. The guard must
        # refuse BEFORE anything is trained -- a guard that fires after the
        # command line has been issued has already lost the race it exists to
        # prevent, so the recorded call log is what this asserts on.
        cache = stub / 'data' / 'QM9' / 'processed' / 'data_v3.pt'
        cache.unlink()
        proc, calls = run_task(script, stub, 0)
        checked[0] += 1
        if proc.returncode == 0:
            failures.append('guard "QM9 not processed" did not fire')
        elif any('process_and_train.py' in c for c in calls):
            failures.append('guard "QM9 not processed" fired only AFTER issuing '
                            'the training command, which is too late')
        cache.write_text('stub')


def check_clean_level_control(failures, checked):
    """The condition that needs its own clean level must actually emit it.

    copy_zero_rows.py refuses any `_uncertainty_values` file, so a condition
    without its own level-zero run gets a clean ACCURACY row and no clean
    uncertainty row -- and the uncertainty statistic subtracts the level-zero
    correlation to remove the label-magnitude confound. Without it the effect is
    NaN: an empty table, not a wrong one. The reference condition's clean run
    cannot stand in, because what differs at level zero is the SHAPE column and
    the reference condition has no shape (RERUN_PLAN.md 3.1f, 2.26b).

    Censoring is the one that needs it: it is keyed to the label, so it is the
    only condition on which the question is answerable at all. Author's decision,
    2026-08-28.
    """
    import generate_scripts as gen

    # Anchored on the CONDITION, not on the set. Written as a loop over
    # NEEDS_OWN_CLEAN_LEVEL this passed vacuously when the set was emptied --
    # which is the shape of guard the fix harness exists to catch.
    for name in sorted({'censoring'} | set(gen.NEEDS_OWN_CLEAN_LEVEL)):
        checked[0] += 1
        levels = gen.CONDITIONS[name][1].split()
        if '0.0' not in levels:
            failures.append(
                f'{name} needs its own clean level as the uncertainty control and '
                f'levels_for gave it {levels}. It is the only condition on which '
                f'"is the model less certain where the labels are unreliable" has '
                f'an answer, and without a level-zero row of its own that answer '
                f'is NaN.')

    # And the ones that do NOT need it must still be stripped, or the saving the
    # copy step exists for is gone.
    checked[0] += 1
    borrowers = [c for c in gen.CONDITIONS
                 if c != gen.REFERENCE_CONDITION and c not in gen.NEEDS_OWN_CLEAN_LEVEL]
    kept = [c for c in borrowers if '0.0' in gen.CONDITIONS[c][1].split()]
    if kept:
        failures.append(
            f'{kept} run the clean level and do not need to; copy_zero_rows.py fills '
            f'their accuracy row in for free')

    # The copy step must still refuse to invent per-molecule numbers -- that
    # refusal is WHY the level has to be run rather than copied.
    checked[0] += 1
    sys.path.insert(0, str(GENERATOR.parent))
    import copy_zero_rows
    if copy_zero_rows.parse_name(Path('anova_censoring_pdv_qrf_uncertainty_values.csv'),
                                 ['censoring'], ['pdv']) is not None:
        failures.append('copy_zero_rows.py no longer refuses an uncertainty file; if it '
                        'has learned to carry them, NEEDS_OWN_CLEAN_LEVEL can shrink')


def check_generator_refusals(failures, checked):
    """The generator's OWN refusals, which check_guards does not reach.

    check_guards runs generated shell scripts; these three never get that far.
    Each one exists because the alternative is compute spent on a grid nobody
    chose: a blind deep run, a deep run missing the model the screen singled
    out, and censoring generated at full breadth over the top of the main-grid
    scripts (RERUN_PLAN.md §13.12 A3/A4). A refusal nothing exercises is a
    refusal that gets deleted the next time it is inconvenient.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = build_stub(tmp / 'stub')
        out = tmp / 'scripts'
        out.mkdir()

        cases = [
            ('deep run with no models or reps',
             ('--stage', '2'), '--models'),
            ('deep run missing a shortlisted model',
             ('--stage', '2', '--models', 'rf', '--reps', 'ecfp4'), 'ngboost'),
        ]
        for name, args, expect in cases:
            cmd = [sys.executable, str(GENERATOR), '--out-dir', str(out),
                   '--qsar-dir', str(stub), *args]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            checked[0] += 1
            if proc.returncode == 0:
                failures.append(f'generator guard "{name}" did not fire')
            elif expect not in proc.stderr + proc.stdout:
                failures.append(f'generator guard "{name}" fired without naming {expect!r}')

        # The escape hatch has to work, or the guard above is a wall.
        cmd = [sys.executable, str(GENERATOR), '--out-dir', str(out),
               '--qsar-dir', str(stub), '--stage', '2', '--models', 'rf',
               '--reps', 'ecfp4', '--drop-shortlisted']
        proc = subprocess.run(cmd, capture_output=True, text=True)
        checked[0] += 1
        if proc.returncode != 0:
            failures.append('--drop-shortlisted did not let the deep run through:\n'
                            f'{proc.stdout}\n{proc.stderr}')

        # A pair-subset condition written into the generator's own directory
        # overwrites the main-grid scripts for those models, exit 0, untracked.
        pair_subset = sorted(gen.PAIR_SUBSET_CONDITIONS)
        if not pair_subset:
            failures.append('no pair-subset condition to test the overwrite guard with')
        else:
            own_dir = Path(GENERATOR).parent
            cmd = [sys.executable, str(GENERATOR), '--out-dir', str(own_dir),
                   '--qsar-dir', str(stub), '--stage', '0',
                   '--conditions', pair_subset[0], '--models', 'rf']
            proc = subprocess.run(cmd, capture_output=True, text=True)
            checked[0] += 1
            if proc.returncode == 0:
                failures.append('generator guard "pair-subset into its own directory" did not '
                                'fire -- it may have just overwritten the main-grid scripts')
            elif pair_subset[0] not in proc.stderr + proc.stdout:
                failures.append('the pair-subset refusal did not name the condition')


def check_against_real_parser(emitted, failures):
    """Feed every emitted command line through scripts/process_and_train.py's own parser.

    This is the check the old generator would have failed: the parser refuses the
    retired flags by name, and rejects a value outside a --choices list, so a
    command that survives here is one the cluster will accept.
    """
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        import process_and_train
    except Exception as exc:                                    # pragma: no cover
        failures.append(f'could not import process_and_train to check the parser: {exc!r}')
        return
    seen = set()
    saved = sys.argv
    for name, index, argv in emitted:
        key = tuple(argv)
        if key in seen:
            continue
        seen.add(key)
        # argparse ignores argv[0] and parses the rest, so the interpreter and
        # its -u have to come off first or the script name is read as a stray
        # positional and every command line "fails" for the harness's reason.
        trimmed = list(argv)
        while trimmed and not trimmed[0].endswith('process_and_train.py'):
            trimmed.pop(0)
        sys.argv = trimmed
        try:
            process_and_train.parse_arguments()
        except SystemExit as exc:
            failures.append(f'{name} task {index}: the real parser refused its own command '
                            f'line (exit {exc.code}): {" ".join(argv)}')
        except Exception as exc:
            failures.append(f'{name} task {index}: parser raised {exc!r}')
    sys.argv = saved
    print(f'  checked {len(seen)} distinct command lines against the real parser')


def end_to_end(failures):
    """One real training run, built by the generator, at a small molecule count."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        stub = build_stub(tmp / 'stub')
        out = tmp / 'scripts'
        out.mkdir()
        script = generate(out, stub, '--stage', '0', '--models', 'rf',
                          '--reps', 'pdv', '--conditions', 'groupedshift',
                          '--sample-size', '400')[0]
        proc, calls = run_task(script, stub, 0)
        train = [c for c in calls if 'process_and_train.py' in c][0]
        argv = shlex.split(train)
        # Two levels rather than the full grid: clean, and the reporting level.
        i = argv.index('--noise-level')
        j = next(k for k in range(i + 1, len(argv)) if argv[k].startswith('-'))
        argv[i + 1:j] = ['0.0', '0.5']
        results = tmp / 'smoke.csv'
        argv[argv.index('-f') + 1] = str(results)
        argv[0] = sys.executable
        print('  running: ' + ' '.join(argv[1:]))
        run = subprocess.run(argv, cwd=str(REPO / 'scripts'),
                             capture_output=True, text=True)
        tail = '\n'.join((run.stdout + run.stderr).splitlines()[-25:])
        if run.returncode != 0:
            failures.append(f'end-to-end run exited {run.returncode}:\n{tail}')
            return
        if not results.exists():
            failures.append(f'end-to-end run wrote no results file:\n{tail}')
            return
        rows = results.read_text().splitlines()
        header = rows[0].split(',')
        body = rows[1:]
        if len(body) != 2:
            failures.append(f'end-to-end run wrote {len(body)} rows, expected 2 '
                            f'(two levels x one replicate)')
        print(f'  wrote {len(body)} rows, columns: {", ".join(header)}')
        for row in body:
            print('    ' + row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--end-to-end', action='store_true',
                    help='Also run one real training task at 400 molecules. Slow, needs the '
                         'rust binary built and the QM9 data present.')
    ap.add_argument('--skip-parser', action='store_true',
                    help='Skip the check against the real parser (it imports the whole '
                         'pipeline, which takes about a minute).')
    args = ap.parse_args()

    failures, checked = [], [0]
    emitted = []
    for label, stage_args in [
        ('stage 0', ('--stage', '0', '--max-hours', '720')),
        # The main grid needs `long`. With the cross-fit costed in, ngboost is
        # 202 h per task and the generator refuses rather than capping, so it is
        # generated with that partition's ceiling -- the way the RUNBOOK submits it.
        ('stage 1', ('--stage', '1', '--max-hours', '720')),
        # ngboost is on the deep run's shortlist, so the generator refuses a deep
        # run without it. Ask for the run the author would actually submit rather
        # than passing --drop-shortlisted here, which would test the escape hatch
        # instead of the path. The refusal itself is checked in
        # check_generator_refusals.
        # The deep run is 10 replicates -- 70 training runs per task -- so with the
        # cross-fit costed in it needs `long` for the same reason the main grid does.
        ('stage 2', ('--stage', '2', '--models', 'rf', 'lgb', 'ngboost',
                     '--reps', 'pdv', 'ecfp4', '--max-hours', '720')),
        ('stage 0, excluded models', ('--stage', '0', '--include-excluded',
                                     '--max-hours', '720')),
    ]:
        print(f'{label}...')
        emitted += check_stage(stage_args, label, failures, checked)

    print('guards...')
    check_guards(failures, checked)

    print('the clean-level control...')
    check_clean_level_control(failures, checked)

    print('the generator\'s own refusals...')
    check_generator_refusals(failures, checked)

    if not args.skip_parser:
        print('the real parser...')
        check_against_real_parser(emitted, failures)

    if args.end_to_end:
        print('end to end...')
        end_to_end(failures)

    print()
    if failures:
        print(f'FAIL — {len(failures)} problem(s):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print(f'PASS — {checked[0]} task runs, {len(emitted)} emitted command lines, all accepted')
    return 0


if __name__ == '__main__':
    sys.exit(main())
