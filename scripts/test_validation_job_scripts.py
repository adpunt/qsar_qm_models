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

# The runner call, matched by the SCRIPT NAME rather than by a fixed prefix.
# It was `'python alternative_data_noise_robustness.py'` until 2026-09-07, and the
# generator started emitting `python -u ...` on 2026-09-01 (commit f4c6cfb). The
# string stopped matching, `command_line_of` raised, and four of these ten checks
# reported "no runner invocation found" for six days while the scripts were fine.
# A test that can be broken by an unbuffering flag was testing the wrong thing.
INVOCATION = 'alternative_data_noise_robustness.py'
INVOKE_RE = re.compile(r'\bpython[0-9.]*\s+(?:-\S+\s+)*'
                       + re.escape(INVOCATION))


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


def generate_raw(tmp, *extra):
    """Run the generator and hand back the result, refusals included."""
    return subprocess.run(
        [sys.executable, str(GENERATOR), '--out-dir', str(tmp), *extra],
        capture_output=True, text=True)


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
        if stripped.startswith('#'):
            continue
        hit = INVOKE_RE.search(stripped)
        if hit:
            return stripped[hit.end():]
    raise AssertionError('no runner invocation found in the generated script')


def datasets_of(text):
    """The command-line dataset names one array script can be given.

    Since 2026-09-01 the generator writes ONE array per model and picks the
    dataset from the task index -- `DATASETS=(logd caco2 herg)` at the top and a
    `case` that turns each into the name argparse accepts. Before that there was
    one script per dataset with the name written into the command line. So the
    command line now holds `"$dataset_cli"`, and a checker that reads it literally
    is reading a shell variable.
    """
    arr = re.search(r'^DATASETS=\(([^)]*)\)', text, re.M)
    names = arr.group(1).split() if arr else []
    cli = dict(re.findall(r'^\s*(\w+)\)\s*dataset_cli="([^"]+)"', text, re.M))
    return [cli.get(n, n) for n in names] or ['logd']


def expanded_command_lines(text):
    """Every command line one array script can actually run, variables resolved.

    A generated script chooses its dataset, its representation and its Gaussian
    process flags from the task index at run time. Handing argparse the raw
    `"$dataset_cli"` proves nothing, so each variable is replaced by a value the
    script itself can produce.
    """
    line = command_line_of(text)
    reps = re.search(r'^REPS=\(([^)]*)\)', text, re.M)
    rep = reps.group(1).split()[0] if reps else 'ECFP4'
    model = re.search(r'--models\s+(\S+)', line)
    model = model.group(1) if model else ''
    gp = ''
    for pats, flags in re.findall(r'^\s*([\w|.-]+)\)\s*GP_FLAGS="([^"]*)"', text, re.M):
        if model in pats.split('|'):
            gp = flags
            break
    out = []
    for ds in datasets_of(text):
        one = line
        one = one.replace('"$dataset_cli"', ds).replace('$dataset_cli', ds)
        one = one.replace('$GP_FLAGS', gp)
        one = one.replace('"$rep"', rep).replace('$rep', rep)
        one = one.replace('"../$OUT_ROOT"', '../results/x').replace('$OUT_ROOT', 'results/x')
        out.append(one)
    return out


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def every_emitted_command_parses(runner):
    """The check that would have caught `--datasets herg`."""
    parser = runner.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp)
        n = 0
        for name in scripts:
            for line in expanded_command_lines(Path(tmp, name).read_text()):
                argv = shlex.split(line)
                n += 1
                try:
                    with contextlib.redirect_stderr(io.StringIO()) as err:
                        parser.parse_args(argv)
                except SystemExit:
                    raise AssertionError(
                        f"{name} emits a command line the runner rejects:\n"
                        f"    {' '.join(argv)}\n"
                        f"    {err.getvalue().strip().splitlines()[-1]}")
    return (f"{len(scripts)} scripts, {n} command lines (one per dataset the array "
            f"can pick), every one accepted by the runner's own parser")


def the_smoke_test_parses_too(runner):
    """It is not a val_ script, so the sweep above skips it -- and it was the
    last file here still carrying the dead micromamba hook."""
    parser = runner.build_parser()
    with tempfile.TemporaryDirectory() as tmp:
        generate(tmp)
        text = Path(tmp, 'smoke_test.sh').read_text()
        joined = text.replace('\\\n', ' ')
        calls = []
        for l in joined.splitlines():
            s = l.strip()
            if s.startswith('#'):
                continue
            hit = INVOKE_RE.search(s)
            if hit:
                calls.append(s[hit.end():])
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
    # and with the deep run on, which is the other way to run this. It takes named
    # pairs since 2026-08-28 -- the validation datasets get the same noise as QM9,
    # on the same shape of run, so the deep conditions are a subset run here too.
    with tempfile.TemporaryDirectory() as tmp:
        scripts, _ = generate(tmp, '--include-depth-conditions',
                              '--models', 'RF', 'NGBoost', '--reps', 'ECFP4', 'PDV')
        for name in scripts:
            text = Path(tmp, name).read_text()
            for bad in retired:
                assert bad not in text, f"the deep run puts {bad} in {name}"
    return f"neither the breadth grid nor the deep run emits any of {sorted(retired)}"


def the_deep_run_is_a_subset_run_here_too():
    """The validation datasets get the same noise as QM9, on the same shape of run.

    The author ruled on 2026-08-28 that the validation datasets see every
    condition QM9 sees. QM9 runs the three depth-only conditions on about a
    dozen model-and-representation pairs, not across its whole grid, and its
    generator refuses `--stage 2` without --models and --reps rather than
    inventing a selection. This asserts the validation generator does the same:
    the depth-only conditions across 8 models x 6 representations is three times
    the breadth grid's cost and was never the design.
    """
    with tempfile.TemporaryDirectory() as tmp:
        r = generate_raw(tmp, '--include-depth-conditions')
        assert r.returncode != 0, (
            "the depth-only conditions were accepted across the whole grid; the deep run "
            "goes on a named subset of pairs, as it does on QM9")
        assert 'named subset' in r.stderr, (
            f"the refusal does not say what the rule is:\n{r.stderr[-400:]}")

        r = generate_raw(tmp, '--include-depth-conditions',
                         '--models', 'RF', 'NGBoost', '--reps', 'ECFP4', 'PDV')
        assert r.returncode == 0, (
            f"the deep run on named pairs was refused:\n{r.stderr[-600:]}")
        deep = {c['name'] for c in SETTLED['stage_2_depth_only']}
        emitted = set()
        for f in Path(tmp).glob('val_*.sh'):
            argv = shlex.split(command_line_of(f.read_text()))
            i = argv.index('--conditions')
            for a in argv[i + 1:]:
                if a.startswith('--'):
                    break
                emitted.add(a)
        missing = deep - emitted
        assert not missing, (
            f"the deep run on named pairs emitted no {', '.join(sorted(missing))}; the "
            f"validation datasets are meant to see every condition QM9 sees")


def the_conditions_are_stated_not_inherited():
    """Passing no --conditions is how the retired setting got in.

    The runner's default is a literal in ITS source
    (alternative_data_noise_robustness.py NOISE_CONDITIONS), so a job that does
    not state its conditions runs whatever that literal happens to say. This
    asserts every script states them, and that what it states is the settled
    set rather than a copy that can drift.
    """
    pair_subset = {c['name'] for g in ('stage_1_full_grid', 'stage_2_depth_only')
                   for c in SETTLED[g]
                   if c.get('scope', {}).get('mode') == 'pair_subset'
                   and 'validation_robustness' in c['scope'].get('applies_to', [])}
    expected = [c['name'] for c in SETTLED['stage_1_full_grid']
                if c['name'] not in pair_subset]
    with tempfile.TemporaryDirectory() as tmp:
        scripts, out = generate(tmp)
        for name in scripts:
            # Expanded, because the raw line ends `--conditions ... $GP_FLAGS` and an
            # unexpanded variable reads as a fourth condition name.
            argv = shlex.split(expanded_command_lines(Path(tmp, name).read_text())[0])
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
        # There are no `val_*_herg.sh` files any more -- one array per model picks
        # its dataset from the task index (commit f4c6cfb, 2026-09-01). The property
        # is unchanged: `herg` in the path, `herg_ki` on the command line.
        checked = 0
        for name in scripts:
            text = Path(tmp, name).read_text()
            arr = re.search(r'^DATASETS=\(([^)]*)\)', text, re.M)
            assert arr, f'{name} names no DATASETS array, so no task can pick one'
            names = arr.group(1).split()
            assert 'herg' in names, (
                f"{name} does not offer hERG at all; its datasets are {names}")
            assert 'herg_ki' not in names, (
                f"{name} puts herg_ki in DATASETS, so OUT_ROOT would end _herg_ki and "
                f"merge_results.py would not match it")
            assert re.search(r'^\s*herg\)\s*dataset_cli="herg_ki"', text, re.M), (
                f"{name} does not turn herg into herg_ki, and argparse rejects herg")
            assert re.search(r'^OUT_ROOT=.*_\$\{dataset\}"?\s*$', text, re.M), (
                f"{name} does not end OUT_ROOT with the path-side dataset name")
            checked += 1
        assert checked, 'no scripts were generated'
    return (f"{checked} array scripts: herg on the path side, herg_ki on the "
            f"command line, chosen from the task index")


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

    conditions = [c['name'] for c in SETTLED['stage_1_full_grid']]  # merge sees all of them
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
        check("the deep run is a subset run here too", the_deep_run_is_a_subset_run_here_too),
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
