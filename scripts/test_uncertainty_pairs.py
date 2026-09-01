#!/usr/bin/env python3
"""Both pipelines run the uncertainty work on the same model-and-representation pairs.

WHY THIS EXISTS
---------------
The author ruled on 2026-08-28 that one set of pairs is used on all four
datasets -- "the same thing that happens on QM9 happens on the others" -- and
settled the membership on 2026-08-29: four models, three representations, the
variational network on ChemBERTa alone.

That cut was applied to the laboratory job generator and to nothing else. QM9 has
no uncertainty run of its own; its uncertainty rows fall out of the main grid, so
every model that could emit one was cross-fitted on every representation. Eleven
models by six representations against the laboratory's ten pairs, and no table
could put the two halves side by side row for row.

`uncertainty_pairs.json` is the one place the pairs are written down, on the model
of `noise_conditions.json` and `model_names.json`. This checks that both
generators agree with it.

WHAT IT CHECKS
--------------
1. The settled file is self-consistent: every restriction names a model and
   representations the file itself lists.
2. Every settled name resolves through `model_names.json` to one canonical name,
   from both pipelines' spellings. Two names that do not meet cannot be compared
   however the runs are configured.
3. The QM9 job generator emits the out-of-fold uncertainty pass for exactly the
   settled pairs, and for no others.
4. The QM9 wall clock is still sized for a task that DOES run the pass. One
   request covers a whole array, and the pass costs `--oof-folds` extra fits, so
   sizing it from the flags -- which no longer carry the pass -- would under-
   request and the job would be killed after its queue wait.
5. The laboratory job generator's own roster agrees with the settled file. Read
   out of its source rather than imported, so this runs on a laptop, and read
   rather than edited, because another session may hold that file.

    python scripts/test_uncertainty_pairs.py
"""
import ast
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
QM9_GENERATOR = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
LAB_GENERATOR = ROOT / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py'

failures = []
checked = [0]


def check(ok, message):
    checked[0] += 1
    if not ok:
        failures.append(message)
    return ok


def load(path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# 1 and 2 — the settled file, and that its names meet
# ---------------------------------------------------------------------------
def check_file_is_consistent(pairs):
    canon_models = {m['canonical'] for m in pairs['models']}
    canon_reps = {r['canonical'] for r in pairs['representations']}
    for model, only in pairs.get('model_representations', {}).items():
        if model.startswith('_'):
            continue
        check(model in canon_models,
              f"model_representations restricts {model!r}, which is not in the "
              f"settled model list {sorted(canon_models)}")
        stray = sorted(set(only) - canon_reps)
        check(not stray,
              f"model_representations gives {model!r} the representation(s) "
              f"{stray}, which the settled list does not contain")
    print(f"  settled file: {len(canon_models)} models, {len(canon_reps)} "
          f"representations, {len(pairs.get('model_representations', {})) - 1} "
          f"restriction(s)")
    return canon_models, canon_reps


def check_decomposition_is_a_subset(pairs):
    """The decomposition list is a subset of the uncertainty list, and its models
    can actually produce both halves.

    A model whose split is reported but which is not run produces nothing; a model
    whose split is reported but which the support table says has no second half
    produces a blank column that reads as a null result. Both are silent, which is
    why this is a check and not a convention.
    """
    canon = {m['canonical'] for m in pairs['models']}
    qm9_name = {m['canonical']: m['qm9'] for m in pairs['models']}
    dec = pairs.get('decomposition', {}).get('models')
    check(isinstance(dec, list) and bool(dec),
          "uncertainty_pairs.json names no decomposition models. The split has to "
          "say which models it is reported from, or every model on the uncertainty "
          "list reads as one it was reported from.")
    if not isinstance(dec, list):
        return
    stray = sorted(set(dec) - canon)
    check(not stray,
          f"the decomposition list names {stray}, which the uncertainty list does "
          f"not run. A split cannot be reported from a model that is not fitted.")

    sys.path.insert(0, str(HERE))
    try:
        from uncertainty_decomposition import support
    except Exception as exc:
        print(f"  NOTE  the support table was NOT consulted "
              f"({type(exc).__name__}: {exc}) -- a skip, not a pass")
        return
    for m in dec:
        if m not in canon:
            continue
        try:
            alea, epis = support(qm9_name[m])
        except Exception as exc:
            check(False, f"{m}: the support table has no entry for "
                         f"{qm9_name[m]!r} ({exc})")
            continue
        # Both halves must EXIST. Either may be one number per fit -- that is
        # enough for the population question this list answers, and the file says
        # so -- but a half that is absent makes the split undefined.
        for half, kind in (('measurement-error', alea), ('model-doubt', epis)):
            check(kind != 'none',
                  f"{m}: the decomposition list reports its {half} half and the "
                  f"support table says it has none, so that column would be blank "
                  f"and read as a null result")
    print(f"  decomposition: {len(dec)} of {len(canon)} uncertainty models, "
          f"both halves present on each")


def check_names_meet(pairs):
    """Every settled name resolves to one canonical name from both spellings."""
    names = load(ROOT / 'model_names.json')
    qm9, val = names['qm9'], names['validation']
    for m in pairs['models']:
        a = qm9.get(m['qm9'])
        b = val.get(m['validation'])
        check(a is not None,
              f"the settled file names the QM9 model {m['qm9']!r} and "
              f"model_names.json does not map it")
        check(b is not None,
              f"the settled file names the laboratory model {m['validation']!r} "
              f"and model_names.json does not map it")
        check(a == b == m['canonical'],
              f"{m['qm9']!r} and {m['validation']!r} resolve to {a!r} and {b!r}, "
              f"and the settled file calls the pair {m['canonical']!r} -- three "
              f"names for one model means the two halves cannot be joined")
    print(f"  names: all {len(pairs['models'])} settled models resolve to one "
          f"canonical name from both spellings")


# ---------------------------------------------------------------------------
# 3 and 4 — the QM9 generator
# ---------------------------------------------------------------------------
def qm9_emitted_pairs(out_dir):
    """(model -> representations that get the pass), read off the real scripts."""
    subprocess.run(
        [sys.executable, str(QM9_GENERATOR), '--stage', '0',
         '--out-dir', str(out_dir), '--max-hours', '720'],
        capture_output=True, text=True, check=True)
    emitted = {}
    hours = {}
    for path in sorted(out_dir.glob('qm9_s0_*.sh')):
        model = path.stem[len('qm9_s0_'):]
        src = path.read_text()
        # Every case line, not the first -- re.search would read one and miss a
        # second if a script ever carried two.
        # The flags string carries `--oof-folds-scored N` for a model that
        # scores fewer folds than it cuts, so the pattern must not end at the
        # fold count. Anchored on the opening quote and the flag name instead.
        cases = re.findall(
            r'^\s*([a-z0-9_ |]+)\)\s*OOF_FLAGS="--oof-folds (\d+)[^"]*"',
            src, re.M)
        emitted[model] = [r.strip() for line, _ in cases for r in line.split('|')]
        h = re.search(r'#SBATCH --time=(\d+):', src)
        hours[model] = int(h.group(1)) if h else 0
        # The out-of-fold flag must never be baked into the literal command: that
        # is what made the pass unconditional on the representation.
        #
        # The first version of this check looked at src.split('case "$rep"')[0]
        # -- the text BEFORE the case block -- while the command line comes
        # AFTER it, so on the four scripts it exists to protect it could not
        # fire at all. Found by the smoke test of 2026-08-29. It now looks at
        # the command line itself.
        cmd = re.search(r'^python -u process_and_train\.py.*?(?=\n\n|\nif |\Z)',
                        src, re.M | re.S)
        check(cmd is not None,
              f'{path.name}: could not find the training command to check')
        if cmd is not None:
            check('--oof-folds' not in cmd.group(0).replace('$OOF_FLAGS', ''),
                  f'{path.name} passes --oof-folds literally in the command, so '
                  f'every representation would pay for the pass')
            # -u True is deliberately NOT restricted: it is free and it writes the
            # test rows either way. A model that emits an uncertainty must still
            # carry it, or seven models silently stop emitting anything.
            if emitted[model]:
                check('-u True' in cmd.group(0),
                      f'{path.name} runs the out-of-fold pass but does not pass '
                      f'-u True, so nothing is written')
    return emitted, hours


def check_the_wall_clock_fits_the_partition(out_dir):
    """No task may ask for more than the longest partition can give it.

    A job that asks for more than the queue allows is rejected at submission,
    which is cheap. The expensive case is the one just under: NGBoost at three
    seeds and five scored folds is eighteen fits per training run and prices the
    main-grid task at 606 hours -- 25 days against a 30-day limit, with nothing
    spare. It would queue for days and then be killed at the wall, losing the
    whole task.

    Three seeds and ONE scored fold is six fits and 202 hours, which is what
    NGBoost cost before the seed ensemble landed. That is what OOF_FOLDS_SCORED
    buys, and this is the check that says so out loud rather than leaving the
    number to be rediscovered.
    """
    # The long partition, in hours. RERUN_PLAN.md 2.8i: medium is 2 days, long
    # is 30.
    LONG_PARTITION_HOURS = 30 * 24
    # How much headroom a job must leave. A task sized at 96% of the limit has
    # no room for a slow node, and it has already queued for days by the time it
    # finds out.
    HEADROOM = 0.75

    subprocess.run(
        [sys.executable, str(QM9_GENERATOR), '--stage', '1', '--max-hours', '720',
         '--out-dir', str(out_dir)],
        capture_output=True, text=True, check=True)
    worst = []
    for path in sorted(out_dir.glob('qm9_s1_*.sh')):
        m = re.search(r'#SBATCH --time=(\d+):', path.read_text())
        if not m:
            continue
        hours = int(m.group(1))
        checked[0] += 1
        if hours > LONG_PARTITION_HOURS * HEADROOM:
            worst.append((path.stem[len('qm9_s1_'):], hours))
    for model, hours in worst:
        check(False,
              f"{model} asks for {hours}h on the main grid, over "
              f"{int(LONG_PARTITION_HOURS * HEADROOM)}h -- {hours / 24:.1f} days "
              f"against a {LONG_PARTITION_HOURS // 24}-day limit. It would queue "
              f"for days and be killed at the wall. Cut the scored folds "
              f"(OOF_FOLDS_SCORED in the generator) rather than the seeds.")
    if not worst:
        peak = max((int(re.search(r'#SBATCH --time=(\d+):', p.read_text()).group(1))
                    for p in out_dir.glob('qm9_s1_*.sh')), default=0)
        print(f"  wall clock: the longest main-grid task asks {peak}h "
              f"({peak / 24:.1f} days), inside the {LONG_PARTITION_HOURS // 24}-day limit")


def check_qm9_matches(pairs, out_dir):
    # The generator itself, for its roster hours and its stage-0 shape. Imported
    # rather than restated: the per-model hours are re-measured from time to
    # time and a copy here would go stale silently.
    sys.path.insert(0, str(QM9_GENERATOR.parent))
    import generate_scripts as gen
    # Stage 0 as this test generates it: one replicate, and the longest
    # condition's level count -- which is what the generator sizes the wall
    # clock from.
    runs_per_task = max(
        len(gen.CONDITIONS[c][1].split()) for c in gen.STAGE_DEFAULTS[0]['conditions'])
    oof_folds = 5

    want = {}          # model -> the representations it should run the pass on
    canon_to_qm9_rep = {r['canonical']: r['qm9'] for r in pairs['representations']}
    all_reps = [r['qm9'] for r in pairs['representations']]
    for m in pairs['models']:
        only = pairs.get('model_representations', {}).get(m['canonical'])
        want[m['qm9']] = ([canon_to_qm9_rep[c] for c in only] if only else list(all_reps))

    emitted, hours = qm9_emitted_pairs(out_dir)

    for model, reps in want.items():
        got = emitted.get(model)
        check(got is not None,
              f"the settled file names the QM9 model {model!r} and the job "
              f"generator emits no script for it")
        if got is not None:
            check(sorted(got) == sorted(reps),
                  f"{model} runs the uncertainty pass on {sorted(got)}; the "
                  f"settled file says {sorted(reps)}")
            # THE WALL CLOCK MUST INCLUDE THE PASS.
            #
            # One request covers a whole array, so it has to fit the worst task
            # in it -- one that runs the out-of-fold pass, which costs
            # (1 + folds) fits per training run instead of one. Sizing it for a
            # task that does not run the pass is how a job queues for days and
            # is then killed at the wall.
            #
            # Checked against the generator's OWN formula rather than against an
            # hour threshold. The first version of this asserted at least 5
            # hours; another session then measured the real per-model timings and
            # the quantile forest came in at 6 hours per 110 runs, so its correct
            # screen request is 3 hours and a correct generator failed this test.
            # An absolute threshold cannot tell "sized without the pass" from
            # "genuinely cheap".
            # A model may score FEWER folds than it cuts, which is the lever
            # that keeps NGBoost inside the partition limit. The fits it pays
            # for are 1 + the folds it SCORES, so this must read the same
            # override the generator does or it demands a wall clock for work
            # the job does not do.
            scored = gen.OOF_FOLDS_SCORED.get(model, 0) or oof_folds
            want_hours = math.ceil(gen.MODELS[model][2] * runs_per_task
                                   * (1 + scored) / 110 * 1.25)
            check(hours.get(model) == want_hours,
                  f"{model} runs the out-of-fold pass and asks for "
                  f"{hours.get(model)}h; the generator's own formula on "
                  f"{runs_per_task} training runs at {1 + scored} fits each "
                  f"gives {want_hours}h. A smaller number means the pass was "
                  f"left out of the sizing.")

    stray = sorted(m for m, r in emitted.items() if r and m not in want)
    check(not stray,
          f"QM9 runs the uncertainty pass for {stray}, which the settled file "
          f"does not list. Every pair costs extra fits per training run and "
          f"produces rows the laboratory side has no counterpart for.")
    n_pairs = sum(len(r) for r in want.values())
    print(f"  QM9: the pass runs on {n_pairs} pairs across "
          f"{len([m for m, r in emitted.items() if r])} scripts, and on no others")


# ---------------------------------------------------------------------------
# 5 — the laboratory generator, read rather than imported or edited
# ---------------------------------------------------------------------------
def lab_roster():
    """MODELS, REPS and MODEL_REPS out of the laboratory generator's source."""
    if not LAB_GENERATOR.is_file():
        return None
    tree = ast.parse(LAB_GENERATOR.read_text())
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name == 'MODELS' and isinstance(node.value, ast.Dict):
            out['MODELS'] = [k.value for k in node.value.keys]
        elif name == 'REPS' and isinstance(node.value, ast.List):
            out['REPS'] = [e.value for e in node.value.elts]
        elif name == 'MODEL_REPS' and isinstance(node.value, ast.Dict):
            out['MODEL_REPS'] = {
                k.value: [e.value for e in v.elts]
                for k, v in zip(node.value.keys, node.value.values)}
    return out


def check_lab_matches(pairs):
    roster = lab_roster()
    if roster is None:
        print('  NOTE  laboratory generator not found; its half is NOT checked')
        return
    want_models = sorted(m['validation'] for m in pairs['models'])
    want_reps = sorted(r['validation'] for r in pairs['representations'])
    check(sorted(roster.get('MODELS', [])) == want_models,
          f"the laboratory generator runs models {sorted(roster.get('MODELS', []))}; "
          f"the settled file says {want_models}")
    check(sorted(roster.get('REPS', [])) == want_reps,
          f"the laboratory generator runs representations "
          f"{sorted(roster.get('REPS', []))}; the settled file says {want_reps}")

    canon_to_val_rep = {r['canonical']: r['validation'] for r in pairs['representations']}
    val_name = {m['canonical']: m['validation'] for m in pairs['models']}
    want_only = {val_name[k]: sorted(canon_to_val_rep[c] for c in v)
                 for k, v in pairs.get('model_representations', {}).items()
                 if not k.startswith('_')}
    got_only = {k: sorted(v) for k, v in roster.get('MODEL_REPS', {}).items()}
    check(got_only == want_only,
          f"the laboratory generator restricts {got_only}; the settled file "
          f"says {want_only}")
    print(f"  laboratory: {len(roster.get('MODELS', []))} models, "
          f"{len(roster.get('REPS', []))} representations, agrees with the file")


def main():
    print('uncertainty_pairs.json — one set of pairs, both pipelines')
    pairs = load(ROOT / 'uncertainty_pairs.json')
    check_file_is_consistent(pairs)
    check_decomposition_is_a_subset(pairs)
    check_names_meet(pairs)
    with tempfile.TemporaryDirectory() as tmp:
        check_qm9_matches(pairs, Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        check_the_wall_clock_fits_the_partition(Path(tmp))
    check_lab_matches(pairs)

    if failures:
        print(f'\nFAIL — {len(failures)} problem(s):')
        for f in failures:
            print(f'  * {f}')
        return 1
    print(f'\nPASS — {checked[0]} checks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
