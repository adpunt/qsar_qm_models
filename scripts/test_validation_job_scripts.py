#!/usr/bin/env python
"""Every command line the VALIDATION job generator emits must actually parse.

The uncertainty family has had this check since 2026-08-27
(`test_uncertainty_job_scripts.py`) and the QM9 family has its own
(`test_generated_job_flags.py`). The validation family had neither, and two
defects lived in it because of that:

  1. It emitted `--datasets herg`. The runner's `--datasets` carries
     `choices=['logd', 'caco2', 'herg_ki', 'all']`, so 28 of the 87 scripts
     would have died at argument parsing before loading a molecule. Two
     scripts had been hand-corrected to `herg_ki` at some point, which is the
     only reason anything ran; regenerating the directory would have reverted
     even those.
  2. It passed no `--conditions` at all, so every job inherited the runner's
     own `NOISE_CONDITIONS` literal -- which contains `outlier_p05`, retired
     on 2026-08-27 in favour of `outlier_p10`. All 87 scripts would have run a
     setting `noise_conditions.json` lists under `not_run`, and skipped the
     settled one, silently: nothing in a result file makes you read a
     condition name.

Both are the same shape. A job script is a string until something runs it, and
a string that is nearly right fails at the far end of a queue. So this does not
grep the generator: it generates real scripts into a temporary directory, pulls
the command line out of each one, and puts it through the runner's OWN parser.

Run it directly:

    python scripts/test_validation_job_scripts.py
    python scripts/test_validation_job_scripts.py --kirby-dir ~/repos/KIRBy

The KIRBy checkout is found from --kirby-dir, then $KIRBY_DIR, then a sibling
of this repository. Importing it is slow -- it pulls in torch, gpytorch and
rdkit -- and that is the price of checking the real parser rather than a copy
of it.
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
GENERATOR = REPO / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
SETTLED = json.loads((REPO / 'noise_conditions.json').read_text())

INVOCATION = 'python alternative_data_noise_robustness.py'


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
        "Pass --kirby-dir <path>, or set KIRBY_DIR. Checking these flags against "
        "anything other than the runner's real parser is the string match that let "
        "`--datasets herg` through for weeks.")


def load_runner(kirby_dir):
    path = Path(kirby_dir) / 'tests' / 'alternative_data_noise_robustness.py'
    spec = importlib.util.spec_from_file_location('kirby_val_runner', path)
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


def generate(tmp, *extra):
    result = subprocess.run(
        [sys.executable, str(GENERATOR), '--out-dir', str(tmp), *extra],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    scripts = sorted(f for f in os.listdir(tmp) if f.startswith('val_'))
    assert scripts, f'the generator wrote no job scripts: {result.stdout[-800:]}'
    return scripts, result.stdout


def command_line_of(text):
    """The runner invocation, with its backslash continuations joined."""
    joined = text.replace('\\\n', ' ')
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or INVOCATION not in stripped:
            continue
        return stripped.split(INVOCATION, 1)[1]
    raise AssertionError('no runner invocation found in the generated script')


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def every_emitted_command_parses(runner):
    """The check that would have caught `--datasets herg`."""
    parser = runner.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts:
            argv = shlex.split(command_line_of(Path(tmp, name).read_text()))
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    parser.parse_args(argv)
            except SystemExit:
                raise AssertionError(
                    f"{name} emits a command line the runner rejects:\n"
                    f"    {' '.join(argv)}\n"
                    f"    {err.getvalue().strip().splitlines()[-1]}")
    return f"{len(scripts)} scripts, every command line accepted by the runner's own parser"


def the_smoke_test_parses_too(runner):
    """It is not a val_ script, so the sweep above skips it -- and it was the
    last file here still carrying the dead micromamba hook."""
    parser = runner.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp)
        text = Path(tmp, 'smoke_test.sh').read_text()
        joined = text.replace('\\\n', ' ')
        calls = [l.strip().split(INVOCATION, 1)[1] for l in joined.splitlines()
                 if INVOCATION in l.strip() and not l.strip().startswith('#')]
        assert len(calls) == 2, f'expected two runner calls in the smoke test, found {len(calls)}'
        for c in calls:
            argv = shlex.split(re.sub(r'\$TESTDIR', '/tmp/x', c))
            with contextlib.redirect_stderr(io.StringIO()) as err:
                try:
                    parser.parse_args(argv)
                except SystemExit:
                    raise AssertionError(
                        f"smoke_test.sh emits a command the runner rejects:\n    {c}\n"
                        f"    {err.getvalue().strip().splitlines()[-1]}")
    return "both of the smoke test's runner calls parse"


def no_retired_condition_is_ever_run():
    """The check that would have caught outlier_p05."""
    retired = {c['name'] for c in SETTLED['not_run']}
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts + ['smoke_test.sh']:
            text = Path(tmp, name).read_text()
            for bad in retired:
                assert bad not in text, (
                    f"{name} names {bad}, which noise_conditions.json lists under "
                    f"not_run. Conditions must come from that file.")
    # and with the depth conditions on, which is the other way to run this
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp, '--include-depth-conditions')
        for name in scripts:
            text = Path(tmp, name).read_text()
            for bad in retired:
                assert bad not in text, f"--include-depth-conditions puts {bad} in {name}"
    return f"neither default nor --include-depth-conditions emits any of {sorted(retired)}"


def the_conditions_are_stated_not_inherited():
    """Passing no --conditions is how the retired setting got in.

    The runner's default is a literal in ITS source
    (alternative_data_noise_robustness.py NOISE_CONDITIONS), so a job that does
    not state its conditions runs whatever that literal happens to say. This
    asserts every script states them, and that what it states is the settled
    set rather than a copy that can drift.
    """
    expected = [c['name'] for c in SETTLED['stage_1_full_grid']]
    with tempfile.TemporaryDirectory() as tmp:
        scripts, out = generate(tmp)
        for name in scripts:
            argv = shlex.split(command_line_of(Path(tmp, name).read_text()))
            assert '--conditions' in argv, (
                f"{name} states no --conditions, so it would inherit the runner's own "
                f"NOISE_CONDITIONS literal. That literal is how outlier_p05 got in.")
            i = argv.index('--conditions')
            got = []
            for a in argv[i + 1:]:
                if a.startswith('--'):
                    break
                got.append(a)
            assert got == expected, f"{name} runs {got}, the settled full grid is {expected}"
    return f"all {len(scripts)} state --conditions {' '.join(expected)}"


def the_dataset_name_and_the_path_name_are_both_right(runner):
    """hERG is `herg_ki` on the command line and `herg` in every path.

    Collapsing them breaks one end or the other: argparse rejects `herg`, and
    the runner writes to results_root/'herg' while merge_results.py matches
    directories by the '_{dataset}' suffix.
    """
    src = (Path(find_kirby()) / 'tests' / 'alternative_data_noise_robustness.py').read_text()
    assert "Path(args.results_root) / 'herg'" in src, (
        "the runner no longer writes hERG output to a directory called 'herg'; the "
        "generator's two-column name table needs revisiting")
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        herg = [s for s in scripts if s.endswith('_herg.sh')]
        assert herg, 'no hERG scripts were generated'
        for name in herg:
            argv = shlex.split(command_line_of(Path(tmp, name).read_text()))
            assert argv[argv.index('--datasets') + 1] == 'herg_ki', \
                f'{name} passes the wrong dataset name'
            root = argv[argv.index('--results-root') + 1]
            assert root.endswith('_herg'), \
                f'{name} writes to {root}, which merge_results.py will not match'
    return f"{len(herg)} hERG scripts: herg_ki on the command line, _herg in the path"


def every_script_carries_all_three_guards():
    """Activation, model-buildability, injector version.

    Three scripts here sat for weeks with none of them because they had been
    hand-edited and so were skipped at every regeneration.
    """
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts + ['smoke_test.sh']:
            text = Path(tmp, name).read_text()
            assert 'basename "$CONDA_PREFIX"' in text, f'{name} has no env_test assertion'
            assert '--validation-models' in text, f'{name} does not check it can build its model'
            assert 'from noiseInject import CONDITIONS' in text, \
                f'{name} does not check the injector is the redesigned one'
            assert 'MAMBA_EXE=' not in text.replace('`export MAMBA_EXE=...`', ''), \
                f'{name} still carries the dead micromamba hook'
    return f"all {len(scripts) + 1} scripts carry the activation, model and injector guards"


def the_model_names_are_ones_the_probe_knows():
    """A guard that blocks a job for the guard's own reason is worse than none."""
    sys.path.insert(0, str(HERE))
    from check_environment import VALIDATION_MODELS
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts:
            text = Path(tmp, name).read_text()
            m = re.search(r'--validation-models (\S+)', text)
            assert m, f'{name} has no --validation-models line'
            assert m.group(1) in VALIDATION_MODELS, (
                f'{name} guards on {m.group(1)!r}, which check_environment.py does not '
                f'know, so the task would exit 2 for the guard\'s own reason')
    return f"all {len(scripts)} guard on a model name the probe knows"


def the_merge_keeps_every_condition(runner):
    """The merge must key on the column the runner actually writes.

    The runner writes `noise_type`; merge_results.py's key list said `strategy`,
    left over from the rename. The comprehension that filters to present columns
    drops a missing name silently rather than raising, so the key quietly became
    (model, rep, sigma, fold) and the four conditions on one cell deduplicated
    against each other -- three quarters of the validation results discarded at
    merge time, with nothing said.

    This builds a frame with the columns the runner really produces, taken from
    the runner module rather than assumed, and asserts all four survive.
    """
    import pandas as pd
    spec = importlib.util.spec_from_file_location(
        'val_merge', REPO / 'slurm_scripts_validation_rerun' / 'merge_results.py')
    merge = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ['merge_results.py']
    try:
        spec.loader.exec_module(merge)
    finally:
        sys.argv = saved

    # the condition column name, from the runner's own source
    src = (Path(find_kirby()) / 'tests' / 'alternative_data_noise_robustness.py').read_text()
    assert "per_sigma['noise_type'] = condition" in src, (
        "the runner no longer labels its per-level metrics with 'noise_type'; "
        "merge_results.py keys on that name")

    conditions = [c['name'] for c in SETTLED['stage_1_full_grid']]
    frame = pd.DataFrame([
        {'dataset': 'caco2', 'model': 'SVM', 'rep': 'PDV', 'sigma': 0.5,
         'fold': 0, 'noise_type': c, 'r2': 0.6 + i / 100}
        for i, c in enumerate(conditions)])
    out, key = merge.deduplicate(frame)
    assert len(out) == len(conditions), (
        f"the merge collapsed {len(conditions)} conditions to {len(out)} "
        f"(key {key}); every condition but {list(out.noise_type)} would be lost")

    # a genuine duplicate must still be removed
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    out2, _ = merge.deduplicate(dup)
    assert len(out2) == len(conditions), 'a real duplicate row was not removed'

    # and an old-format file must still merge
    old = frame.rename(columns={'noise_type': 'strategy'})
    out3, key3 = merge.deduplicate(old)
    assert len(out3) == len(conditions), f'old-format results collapsed (key {key3})'
    return (f"{len(conditions)} conditions survive the merge (key {key}); duplicates "
            f"still removed; old 'strategy' files still merge")


def the_scripts_are_valid_bash():
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        for name in scripts + ['smoke_test.sh', 'submit_all.sh']:
            r = subprocess.run(['bash', '-n', str(Path(tmp, name))],
                               capture_output=True, text=True)
            assert r.returncode == 0, f'{name} is not valid bash: {r.stderr.strip()}'
    return f'{len(scripts) + 2} files pass bash -n'


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as e:
        print(f"  FAIL  {name}\n        {e}")
        return False
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return False
    print(f"  OK    {name}\n        {detail}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kirby-dir', default=None)
    args = ap.parse_args()

    print("validation job scripts\n")
    kirby = find_kirby(args.kirby_dir)
    print(f"KIRBy: {kirby}")
    runner = load_runner(kirby)

    results = [
        check("every emitted command line parses", lambda: every_emitted_command_parses(runner)),
        check("the smoke test parses too", lambda: the_smoke_test_parses_too(runner)),
        check("no retired condition is ever run", no_retired_condition_is_ever_run),
        check("the conditions are stated, not inherited", the_conditions_are_stated_not_inherited),
        check("hERG's two names are both right",
              lambda: the_dataset_name_and_the_path_name_are_both_right(runner)),
        check("every script carries all three guards", every_script_carries_all_three_guards),
        check("the model names are ones the probe knows", the_model_names_are_ones_the_probe_knows),
        check("the merge keeps every condition", lambda: the_merge_keeps_every_condition(runner)),
        check("the scripts are valid bash", the_scripts_are_valid_bash),
    ]

    print()
    if not all(results):
        print("FAIL: the validation job scripts would not do what they say")
        return 1
    print("OK: every validation job script parses and carries its guards")
    return 0


if __name__ == '__main__':
    sys.exit(main())
