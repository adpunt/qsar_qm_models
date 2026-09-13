#!/usr/bin/env python3
"""One list of which models the uncertainty work runs, not two.

WHAT WENT WRONG, AND WHEN
-------------------------
`uncertainty_pairs.json` says why it exists in its own first line: both pipelines
answer the same uncertainty questions on the same pairs, so a table can put QM9
beside logD, Caco-2 and hERG row for row.

On 2026-09-07 the author added GP-Hetero to the uncertainty runs. It went into
the MODELS dict of `slurm_scripts_uncertainty_rerun/generate_scripts.py` and into
KIRBy's UNCERTAINTY_MODELS, and into `uncertainty_pairs.json` not at all. For
five days the assay side ran seven models and QM9's out-of-fold pass fired on
six -- so the assay half was about to write GP-Hetero uncertainty rows that QM9
had no counterpart for. Nobody could see it from either file, because neither one
named the other.

Closed 2026-09-12 two ways. GP-Hetero is in the settled file, and the assay
generator reads its membership out of that file instead of holding a copy. This
test is the guard on both halves.

WHAT IT CHECKS
--------------
1. The assay generator's roster IS the settled file's roster -- same models, same
   order, same representations, same per-model narrowing. Imported and run, not
   pattern-matched, so a list rebuilt at import time is checked as it will
   actually behave.
2. Every model the settled file names has a spec in that generator (a tier, a
   core count, a note) and a per-fit rate in the laboratory generator, or its
   wall clock cannot be priced.
3. Deleting a model from the settled file removes exactly its array and nothing
   else, and adding a name with no spec stops the generator instead of writing a
   short roster. Both are run against a temporary copy of the JSON; the real file
   is never written.
4. Every settled model resolves through `model_names.json` to one canonical name
   from all three spellings, so a QM9 row and an assay row can be joined.
5. The QM9 side fires its out-of-fold pass on exactly the settled models. QM9 has
   no uncertainty submission of its own: its uncertainty rows fall out of the main
   grid, and `--oof-folds` is written into the task only for a settled pair.
6. KIRBy's UNCERTAINTY_MODELS contains every settled model, and the two read in
   the same order -- checked only when the KIRBy checkout is present.

    python scripts/test_uncertainty_rosters_agree.py

Read-only against the repository. It generates real scripts into a temporary
directory and throws them away.
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAIRS_FILE = ROOT / 'uncertainty_pairs.json'
UNC_GENERATOR = ROOT / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py'
LAB_GENERATOR = ROOT / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
QM9_GENERATOR = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
KIRBY_RUNNER = Path('/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')

failures = []
checked = [0]


def check(ok, message):
    checked[0] += 1
    if not ok:
        failures.append(message)
    return ok


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1 and 2 — the assay generator's roster is the settled file's roster
# ---------------------------------------------------------------------------
def check_assay_roster(pairs):
    gen = load_module(UNC_GENERATOR, '_unc_generator')

    want_models = [m['validation'] for m in pairs['models']]
    check(list(gen.MODELS) == want_models,
          f"the uncertainty generator runs {list(gen.MODELS)}; "
          f"{PAIRS_FILE.name} says {want_models}. Same names in the same order, "
          f"or a second list has appeared again")

    want_reps = [r['validation'] for r in pairs['representations']]
    check(list(gen.REPS) == want_reps,
          f"the uncertainty generator runs representations {list(gen.REPS)}; "
          f"{PAIRS_FILE.name} says {want_reps}. Order counts as well as "
          f"membership: a task index is (dataset, representation, condition), so "
          f"reordering changes what an already-queued index means")

    canon_rep = {r['canonical']: r['validation'] for r in pairs['representations']}
    canon_model = {m['canonical']: m['validation'] for m in pairs['models']}
    want_only = {canon_model[k]: sorted(canon_rep[c] for c in v)
                 for k, v in pairs.get('model_representations', {}).items()
                 if not k.startswith('_')}
    got_only = {k: sorted(v) for k, v in gen.MODEL_REPS.items()}
    check(got_only == want_only,
          f"the uncertainty generator narrows {got_only}; {PAIRS_FILE.name} "
          f"says {want_only}")

    # 2 — everything the wall clock and the memory request need, per model.
    lab = load_module(LAB_GENERATOR, '_lab_generator')
    for name in want_models:
        check(name in gen.MODEL_SPEC,
              f"{name} is on the settled list and has no MODEL_SPEC entry in "
              f"{UNC_GENERATOR.name}")
        check(name in lab.SECONDS_PER_FIT_PER_1K,
              f"{name} is on the settled list and has no per-fit rate in "
              f"{LAB_GENERATOR.parent.name}/{LAB_GENERATOR.name}, so its wall "
              f"clock cannot be priced")
        mem = gen.memory_for(name)
        check(bool(mem) and str(mem).upper().endswith('G'),
              f"{name} gets no memory request from model_memory.json (got {mem!r})")
    print(f"  assay generator: {len(gen.MODELS)} models, {len(gen.REPS)} "
          f"representations, read from {PAIRS_FILE.name} and agreeing with it")
    return gen


# ---------------------------------------------------------------------------
# 3 — the file really is what decides, proved by changing a copy of it
# ---------------------------------------------------------------------------
def _run_generator_against(pairs_text, out_dir, work):
    """Run the assay generator with a SUBSTITUTE uncertainty_pairs.json.

    The generator resolves its data files against its own parent's parent, so a
    copy of the tree with one JSON replaced is the only way to change what it
    reads without writing to the repository. Everything is copied or symlinked
    into a temporary directory; nothing under the repository is touched.
    """
    pkg = work / 'slurm_scripts_uncertainty_rerun'
    pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(UNC_GENERATOR, pkg / 'generate_scripts.py')
    (work / 'uncertainty_pairs.json').write_text(pairs_text)
    for name in ('model_memory.json', 'model_names.json', 'noise_conditions.json'):
        target = work / name
        if not target.exists():
            os.symlink(ROOT / name, target)
    lab_link = work / 'slurm_scripts_validation_rerun'
    if not lab_link.exists():
        os.symlink(LAB_GENERATOR.parent, lab_link)
    return subprocess.run(
        [sys.executable, str(pkg / 'generate_scripts.py'), '--out-dir', str(out_dir)],
        capture_output=True, text=True)


def check_the_file_decides(pairs):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'shortened'
        out = Path(tmp) / 'out_short'
        dropped = json.loads(PAIRS_FILE.read_text())
        removed = dropped['models'].pop()['validation']
        run = _run_generator_against(json.dumps(dropped), out, work)
        check(run.returncode == 0,
              f"dropping {removed} from a copy of {PAIRS_FILE.name} made the "
              f"generator exit {run.returncode}:\n{run.stderr[-800:]}")
        slug = removed.lower().replace('-', '_')
        wrote = sorted(p.name for p in out.glob('unc_*.sh')) if out.is_dir() else []
        check(f'unc_{slug}.sh' not in wrote,
              f"{removed} was dropped from the copied file and the generator "
              f"still wrote unc_{slug}.sh — the list in the generator is what is "
              f"deciding, not the file")
        check(len(wrote) == len(pairs['models']) - 1,
              f"dropping one model wrote {len(wrote)} arrays; "
              f"{len(pairs['models']) - 1} expected")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / 'widened'
        out = Path(tmp) / 'out_wide'
        added = json.loads(PAIRS_FILE.read_text())
        added['models'].append({'canonical': 'not_a_model', 'qm9': 'not_a_model',
                                'validation': 'Not-A-Model', 'why': 'a test'})
        run = _run_generator_against(json.dumps(added), out, work)
        check(run.returncode != 0,
              "a model with no MODEL_SPEC entry was added to a copy of "
              f"{PAIRS_FILE.name} and the generator exited 0. It must stop: a "
              "roster quietly one model short is the failure this whole test "
              "exists for")
        check('Not-A-Model' in (run.stderr + run.stdout),
              "the generator refused an unspecced model without naming it")
    print("  the settled file decides: dropping a model removes its array, and a "
          "model with no spec stops the run")


# ---------------------------------------------------------------------------
# 4 — the three spellings meet
# ---------------------------------------------------------------------------
def check_names_meet(pairs):
    names = json.loads((ROOT / 'model_names.json').read_text())
    for m in pairs['models']:
        canon = m['canonical']
        for half in ('qm9', 'validation'):
            resolved = names[half].get(m[half], m[half])
            check(resolved == canon,
                  f"{PAIRS_FILE.name} calls this model {m[half]!r} on the {half} "
                  f"side; model_names.json resolves that to {resolved!r} and the "
                  f"file's canonical name is {canon!r}. Two names that do not "
                  f"meet cannot be joined in a table however the runs go")
    print(f"  names: all {len(pairs['models'])} settled models resolve to one "
          f"canonical name from both spellings")


# ---------------------------------------------------------------------------
# 5 — QM9 fires the out-of-fold pass on exactly the settled models
# ---------------------------------------------------------------------------
def check_qm9_fires_on_them(pairs, out_dir):
    run = subprocess.run(
        [sys.executable, str(QM9_GENERATOR), '--stage', '1', '--max-hours', '720',
         '--out-dir', str(out_dir)],
        capture_output=True, text=True)
    check(run.returncode == 0,
          f"the QM9 main-grid generator exited {run.returncode}:\n{run.stderr[-800:]}")
    if run.returncode != 0:
        return
    want = sorted(m['qm9'] for m in pairs['models'])
    canon_rep = {r['canonical']: r['qm9'] for r in pairs['representations']}
    all_reps = [r['qm9'] for r in pairs['representations']]
    want_reps = {}
    for m in pairs['models']:
        only = pairs.get('model_representations', {}).get(m['canonical'])
        want_reps[m['qm9']] = sorted(canon_rep[c] for c in only) if only else sorted(all_reps)
    got = []
    walls = {}
    for path in sorted(out_dir.glob('qm9_s1_*.sh')):
        model = path.stem[len('qm9_s1_'):]
        src = path.read_text()
        cases = re.findall(r'^\s*([a-z0-9_ |]+)\)\s*OOF_FLAGS="--oof-folds \d+',
                           src, re.M)
        if cases:
            got.append(model)
            fires_on = sorted(r.strip() for line in cases for r in line.split('|'))
            check(fires_on == want_reps.get(model, []),
                  f"the QM9 main grid runs {model}'s out-of-fold pass on "
                  f"{fires_on}; the settled file says {want_reps.get(model)}")
            check('-u True' in src,
                  f"{path.name} runs the out-of-fold pass and does not pass "
                  f"-u True, so nothing is written")
        m = re.search(r'#SBATCH --time=(\d+):', src)
        if m:
            walls[model] = int(m.group(1))
    check(sorted(got) == want,
          f"the QM9 main grid runs the out-of-fold pass for {sorted(got)}; the "
          f"settled file names {want}. A model on the settled list that does not "
          f"get the pass writes an uncertainty column scored on the TEST split, "
          f"where the injected noise is zero, and nothing downstream can tell "
          f"that apart from a real one")
    for model in got:
        check(walls.get(model, 0) > 0,
              f"{model} runs the out-of-fold pass and its script names no wall clock")
    print(f"  QM9: the out-of-fold pass fires on {len(got)} models, and on no "
          f"others; longest main-grid wall among them "
          f"{max((walls[m] for m in got), default=0)}h")


# ---------------------------------------------------------------------------
# 6 — KIRBy, when it is on this machine
# ---------------------------------------------------------------------------
def check_kirby(pairs):
    if not KIRBY_RUNNER.is_file():
        print(f'  NOTE  {KIRBY_RUNNER} not on this machine; its half is NOT checked')
        return
    tree = ast.parse(KIRBY_RUNNER.read_text())
    listed = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'UNCERTAINTY_MODELS'
                and isinstance(node.value, ast.List)):
            listed = [e.value for e in node.value.elts]
    if listed is None:
        check(False, f"UNCERTAINTY_MODELS is not a plain list in {KIRBY_RUNNER.name}")
        return
    want = [m['validation'] for m in pairs['models']]
    absent = [m for m in want if m not in listed]
    check(not absent,
          f"{absent} are on the settled list and not in KIRBy's "
          f"UNCERTAINTY_MODELS. The runner writes no uncertainty column for a "
          f"model it does not name, so those arrays would run and produce nothing")
    order = [m for m in listed if m in want]
    check(order == want,
          f"KIRBy reads the settled models in the order {order} and "
          f"{PAIRS_FILE.name} in the order {want}. Keep one order; "
          f"scripts/test_uncertainty_job_scripts.py reads both")
    print(f"  KIRBy: all {len(want)} settled models are in UNCERTAINTY_MODELS, "
          f"in the same order")


def main():
    print('one uncertainty roster, not two — uncertainty_pairs.json against both '
          'generators and KIRBy')
    pairs = json.loads(PAIRS_FILE.read_text())
    print(f"  settled file: {len(pairs['models'])} models, "
          f"{len(pairs['representations'])} representations")
    check_assay_roster(pairs)
    check_the_file_decides(pairs)
    check_names_meet(pairs)
    with tempfile.TemporaryDirectory() as tmp:
        check_qm9_fires_on_them(pairs, Path(tmp))
    check_kirby(pairs)

    if failures:
        print(f'\nFAIL — {len(failures)} problem(s):')
        for f in failures:
            print(f'  * {f}')
        return 1
    print(f'\nPASS — {checked[0]} checks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
