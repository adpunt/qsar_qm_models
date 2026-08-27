#!/usr/bin/env python
"""Every command line the uncertainty job generator emits must actually parse.

The generator listed the six noise strategies — legacy, outlier, quantile,
hetero, threshold, valprop — for a day after they were deleted in noiseInject
1.0.0, and emitted `--strategies`, `--unc-strategies` and `--threshold-quantile`,
none of which the runner has any more. All 504 tasks would have died at argument
parsing. Nothing checked, because the check would have had to reach across two
repositories.

This reaches across them. It generates real job scripts, pulls the command line
out of each one, substitutes the shell variables with values the array dispatch
can actually produce, and runs the result through the runner's OWN argument
parser — not a string match against its source, which passes whether or not the
line it matched ever runs.

Run it directly:

    python scripts/test_uncertainty_job_scripts.py
    python scripts/test_uncertainty_job_scripts.py --kirby-dir ~/repos/KIRBy

The KIRBy checkout is found from --kirby-dir, then $KIRBY_DIR, then a sibling
of this repository. Importing it is slow (about 40 seconds — it pulls in torch,
gpytorch and rdkit) and that is the price of checking the real parser.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GENERATOR = REPO / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py'
SETTLED_FILE = REPO / 'noise_conditions.json'

# Values the array dispatch can actually produce, for the shell variables that
# appear inside the command line. Substituted before parsing so the VALUES are
# checked too, not only the flag names: --conditions and --reps both carry
# choices=, and a deleted condition name fails there.
SHELL_VALUES = {
    'ds': 'herg_ki',
    'rep': 'MHG-GNN-pretrained',   # the hyphenated one, which quoting can break
    'cond': None,                  # filled per condition below
    'OUT': 'results/uncertainty_rerun/qrf__herg_ki__mhggnnpretrained__gaussian',
}


def find_kirby(explicit=None):
    candidates = [explicit, os.environ.get('KIRBY_DIR'), REPO.parent / 'KIRBy']
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if (p / 'tests' / 'alternative_data_noise_robustness.py').exists():
            return p
    raise SystemExit(
        "Could not find a KIRBy checkout with tests/alternative_data_noise_robustness.py.\n"
        "Tried: " + ", ".join(str(c) for c in candidates if c) + "\n"
        "Pass --kirby-dir <path>, or set KIRBY_DIR. This test cannot be run without it: "
        "checking the flags against anything other than the runner's real parser is "
        "the string match that let the deleted strategies through in the first place.")


def load_runner(kirby_dir):
    path = Path(kirby_dir) / 'tests' / 'alternative_data_noise_robustness.py'
    spec = importlib.util.spec_from_file_location('kirby_unc_runner', path)
    module = importlib.util.module_from_spec(spec)
    saved_argv, sys.argv = sys.argv, ['alternative_data_noise_robustness.py']
    saved_path = list(sys.path)
    sys.path.insert(0, str(Path(kirby_dir) / 'tests'))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.argv, sys.path = saved_argv, saved_path
    return module


def load_generator():
    spec = importlib.util.spec_from_file_location('unc_generate_scripts', GENERATOR)
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ['generate_scripts.py']
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def generate(tmp, *extra):
    result = subprocess.run(
        [sys.executable, str(GENERATOR), '--out-dir', str(tmp), *extra],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    scripts = sorted(f for f in os.listdir(tmp) if f.endswith('.sh'))
    assert scripts, f'the generator wrote no scripts: {result.stdout[-800:]}'
    return scripts, result.stdout


INVOCATION = 'python -u alternative_data_noise_robustness.py'


def command_line_of(text):
    """The runner invocation, with its backslash continuations joined.

    Matched on the whole `python -u <file>` invocation, not on the file name:
    the script also names the file in its own guard, and half a shell string is
    not a command line.
    """
    joined = text.replace('\\\n', ' ')
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or INVOCATION not in stripped:
            continue
        return stripped.split(INVOCATION, 1)[1]
    raise AssertionError('no runner invocation found in the generated script')


def conditions_of(text):
    m = re.search(r'^CONDS=\((.*?)\)$', text, re.M)
    assert m, 'the generated script has no CONDS=(...) line'
    return m.group(1).split()


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def every_emitted_command_parses(runner, _gen):
    """Generate the default scripts and parse what each one would run."""
    parser = runner.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        checked = 0
        for name in scripts:
            text = Path(tmp, name).read_text()
            after = command_line_of(text)
            for cond in conditions_of(text):
                values = dict(SHELL_VALUES, cond=cond)
                # shlex strips the quoting, so "$ds" arrives as the bare token
                # $ds and the substitution is a straight lookup.
                argv = [values.get(a[1:], a) if a.startswith('$') else a
                        for a in shlex.split(after)]
                left = [a for a in argv if a.startswith('$')]
                assert not left, f'{name}: unsubstituted shell variable {left}'
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err), \
                            contextlib.redirect_stdout(io.StringIO()):
                        parsed = runner.validate_args(parser, parser.parse_args(argv))
                except SystemExit:
                    raise AssertionError(
                        f'{name} (condition {cond}) emits a command the runner refuses:\n'
                        f'      {" ".join(argv)}\n'
                        f'      {err.getvalue().strip().splitlines()[-1]}')
                assert parsed.conditions == [cond], parsed.conditions
                assert parsed.oof_folds > 1, (
                    'out-of-fold scoring is what makes question A answerable; '
                    f'{name} asks for {parsed.oof_folds} folds')
                assert parsed.unc_conditions == 'all', (
                    'question B needs the per-molecule uncertainty written for every '
                    f'condition; {name} asks for {parsed.unc_conditions!r}')
                checked += 1
        print(f'    {len(scripts)} scripts x their conditions = {checked} command lines, '
              f'all accepted')


def no_deleted_name_survives(_runner, _gen):
    """The six deleted strategies, and the flags that carried them."""
    gone = ('legacy', 'quantile', 'valprop', 'hetero', 'threshold',
            '--strategies', '--unc-strategies', '--drop-strategies',
            '--threshold-quantile')
    with tempfile.TemporaryDirectory() as tmp:
        scripts, stdout = generate(tmp)
        bad = []
        for name in scripts:
            text = Path(tmp, name).read_text()
            body = command_line_of(text) + ' ' + ' '.join(conditions_of(text))
            for g in gone:
                if g in body:
                    bad.append(f'{name}: {g}')
        # 'outlier' is not in the list: outlier_p10 is a live condition of the
        # deep run, and the deleted strategy of the same name is caught by
        # --conditions carrying choices=.
        assert not bad, 'deleted names still emitted:\n    ' + '\n    '.join(bad)
        assert 'strateg' not in stdout, f'the generator still says "strategy":\n{stdout}'
    print(f'    none of {len(gone)} deleted names appear in what the scripts run')


def conditions_come_from_the_settled_file(_runner, gen):
    settled = json.loads(SETTLED_FILE.read_text())
    main_grid = [c['name'] for c in settled['stage_1_full_grid']]
    deep = [c['name'] for c in settled['stage_2_depth_only']]
    assert gen.MAIN_GRID_CONDITIONS == main_grid, (
        f'{gen.MAIN_GRID_CONDITIONS} != {main_grid}')
    assert gen.DEEP_RUN_CONDITIONS == deep, f'{gen.DEEP_RUN_CONDITIONS} != {deep}'
    # The recorded default (RERUN_PLAN.md 13.1 item 2): the main grid's four
    # plus the one depth-only condition that can answer question B at all.
    added = gen.ADDED_FOR_QUESTION_B
    assert all(c in deep for c in added), f'{added} is not depth-only'
    assert all(c not in gen.FLAT_BY_DESIGN for c in added), (
        f'{added} gives every molecule the same amount, so adding it buys nothing '
        f'for question B — which is the only reason it is added')
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        got = conditions_of(Path(tmp, scripts[0]).read_text())
        assert got == main_grid + added, (
            f'the default run is {got}, not {main_grid + added}')
        scripts, _ = generate(tmp, '--main-grid-only')
        got = conditions_of(Path(tmp, scripts[0]).read_text())
        assert got == main_grid, f'--main-grid-only gave {got}'
        scripts, _ = generate(tmp, '--include-deep-conditions')
        got = conditions_of(Path(tmp, scripts[0]).read_text())
        assert got == main_grid + deep, got
    print(f'    default = the main grid {main_grid} plus {added}; '
          f'--main-grid-only drops it, --include-deep-conditions runs all of {deep}')


def the_run_can_still_answer_both_questions(_runner, gen):  # noqa: D401
    """A condition set with no even condition, or no patterned one, is announced."""
    with tempfile.TemporaryDirectory() as tmp:
        _, stdout = generate(tmp, '--conditions', 'grouped_wider', 'censoring')
        assert 'question A has no clean reference' in stdout, stdout
        _, stdout = generate(tmp, '--conditions', 'gaussian', 'grouped_shifted')
        assert 'question B is undefined throughout' in stdout, stdout
        # The one that is added is added BECAUSE it is not flat.
        _, stdout = generate(tmp, '--conditions', *gen.ADDED_FOR_QUESTION_B)
        assert 'question B is undefined throughout' not in stdout, stdout
        _, stdout = generate(tmp)
        assert 'WARNING' not in stdout, f'the default run warns, and should not:\n{stdout}'
    assert gen.QUESTION_A_CONDITION in gen.MAIN_GRID_CONDITIONS
    print('    dropping every even condition, or every patterned one, is called out')


def the_rosters_match_the_runner(runner, gen):
    """Models, representations and datasets the generator asks for must exist."""
    assert list(gen.MODELS) == list(runner.UNCERTAINTY_MODELS), (
        f'the generator writes scripts for {list(gen.MODELS)}, but the runner '
        f'cross-fits {list(runner.UNCERTAINTY_MODELS)} — a model outside that list is '
        f'refused, and a model missing from it never gets an out-of-fold pass')
    unknown = [r for r in gen.REPS if r not in runner.ALL_REPS]
    assert not unknown, f'representations the runner does not have: {unknown}'
    parser = runner.build_parser()
    ds_choices = next(a.choices for a in parser._actions if a.dest == 'datasets')
    unknown = [d for d in gen.DATASETS if d not in ds_choices]
    assert not unknown, f'datasets the runner does not have: {unknown}'
    print(f'    {len(gen.MODELS)} models, {len(gen.REPS)} of the runner\'s '
          f'{len(runner.ALL_REPS)} reps, {len(gen.DATASETS)} datasets — all known')


def the_scripts_are_valid_bash(_runner, _gen):
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts:
            r = subprocess.run(['bash', '-n', str(Path(tmp, name))],
                               capture_output=True, text=True)
            assert r.returncode == 0, f'{name}: {r.stderr.strip()}'
    print(f'    {len(scripts)} scripts, bash -n clean')


def the_array_dispatch_covers_every_task_once(_runner, _gen):
    """Walk the index arithmetic the script does, for every index."""
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        text = Path(tmp, scripts[0]).read_text()
        datasets = re.search(r'^DATASETS=\((.*?)\)$', text, re.M).group(1).split()
        reps = re.findall(r'"([^"]+)"',
                          re.search(r'^REPS=\((.*?)\)$', text, re.M).group(1))
        conds = conditions_of(text)
        n_c, n_r = len(conds), len(reps)
        n = len(datasets) * n_r * n_c
        seen = [(datasets[i // (n_c * n_r)], reps[(i // n_c) % n_r], conds[i % n_c])
                for i in range(n)]
        assert len(set(seen)) == n, (
            f'{n} array indices cover only {len(set(seen))} distinct tasks')
        # The header must promise the same count the arithmetic produces.
        assert f'--array=0-{n - 1}%' in text, (
            f'the header tells you to submit a range that is not 0-{n - 1}')
    print(f'    {n} indices -> {n} distinct (dataset, rep, condition) tasks, header agrees')


CHECKS = [
    ('every command line the generator emits parses', every_emitted_command_parses),
    ('no deleted strategy name or flag survives', no_deleted_name_survives),
    ('the conditions come from noise_conditions.json', conditions_come_from_the_settled_file),
    ('both uncertainty questions stay answerable', the_run_can_still_answer_both_questions),
    ('the model, rep and dataset rosters match the runner', the_rosters_match_the_runner),
    ('the generated scripts are valid bash', the_scripts_are_valid_bash),
    ('the array dispatch covers every task once', the_array_dispatch_covers_every_task_once),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--kirby-dir', default=None)
    args = ap.parse_args()

    kirby = find_kirby(args.kirby_dir)
    print(f'Uncertainty job generator (RERUN_PLAN.md §2.8j)')
    print(f'  KIRBy: {kirby}')
    runner = load_runner(kirby)
    gen = load_generator()

    ok = True
    for name, fn in CHECKS:
        print(f'  {name}')
        try:
            fn(runner, gen)
        except Exception as exc:  # noqa: BLE001
            print(f'    FAIL: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            ok = False
        else:
            print('    ok')
    if not ok:
        print('\nFAIL: the uncertainty jobs would not run as generated')
        return 1
    print('\nOK: every generated uncertainty job parses against the runner itself')
    return 0


if __name__ == '__main__':
    sys.exit(main())
