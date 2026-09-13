#!/usr/bin/env python
"""The three depth-only conditions are DECLARED to run on the experimental datasets.

WHAT THIS IS FOR. The author, 2026-08-28, was told that logD, Caco-2 and hERG were
seeing only gaussian, grouped_wider and grouped_shifted while QM9 saw all seven, and
answered: "It should get all the same noise as qm9, update the documentation and wire
it in." That was wired in -- the depth submission on 2026-09-06 asked for six
conditions on this side, the breadth three and the depth three -- but the ruling lived
only in the generator's --include-depth-conditions help text. noise_conditions.json,
which is where every other condition decision lives, gave student_t_nu5, outlier_p10
and laplace no scope at all. Nothing in data said they belonged on this side.

The scope blocks now carry it and the generator reads them. This asserts the pieces
hold together, and in particular the one thing that would quietly undo the ruling:
giving those scopes mode "pair_subset" instead of "pair_grid".

THE DIFFERENCE BETWEEN THE TWO MODES. Censoring is five pairs named outright in
censoring_pairs.json, and both generators cap it at about five. The depth conditions
are the deep run's model list CROSSED with its representation list -- eight models on
three representations in deep_run_pairs.json, twenty-four pairs. Every reader of
noise_conditions.json filters on the literal string "pair_subset", so "pair_grid"
passes through them unchanged; "pair_subset" would put the depth conditions under
censoring's five-pair cap and the depth submission would be refused.

Run: python scripts/test_depth_conditions_scope.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALGEN = ROOT / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
SETTLED = json.loads((ROOT / 'noise_conditions.json').read_text())

DEPTH = {c['name']: c for c in SETTLED['stage_2_depth_only']}

failures = []


def check(ok, message):
    if not ok:
        failures.append(message)
    return ok


def load_generator():
    spec = importlib.util.spec_from_file_location('valgen_for_test', VALGEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def the_file_declares_the_experimental_datasets():
    """Each depth-only condition's scope names validation_robustness."""
    for name, entry in DEPTH.items():
        scope = entry.get('scope')
        if not check(isinstance(scope, dict),
                     f"{name} has no scope in noise_conditions.json, so nothing declares "
                     f"where it runs. The author ruled on 2026-08-28 that the experimental "
                     f"datasets get the same noise as QM9; the file has to say so."):
            continue
        applies = scope.get('applies_to', [])
        check('validation_robustness' in applies,
              f"{name}'s scope does not name validation_robustness; it names {applies}. "
              f"Narrowing this side needs a reason in the file first (author, 2026-08-28).")
        check('qm9_deep_run' in applies,
              f"{name}'s scope does not name qm9_deep_run; it names {applies}. The whole "
              f"point of the 2026-08-28 ruling is that the two halves match.")
        check(scope.get('defined_on_experimental_datasets', '').startswith('Yes'),
              f"{name}'s scope does not record whether the condition is DEFINED on logD, "
              f"Caco-2 and hERG. A condition stays out only if it is degenerate, broken or "
              f"undefined there, so the file has to answer that in writing.")
    return (f"all {len(DEPTH)} depth-only conditions declare validation_robustness: "
            f"{', '.join(DEPTH)}")


def the_depth_conditions_are_not_a_pair_subset():
    """mode is pair_grid, not pair_subset -- censoring's five-pair cap must not reach them.

    Checked against the readers themselves, not against a remembered list of them, so
    a sixth reader added later with the same filter is covered by the same assertion.
    """
    for name, entry in DEPTH.items():
        mode = entry.get('scope', {}).get('mode')
        check(mode != 'pair_subset',
              f"{name}'s scope is mode 'pair_subset'. Censoring is five pairs named in "
              f"censoring_pairs.json and is capped at about five; the depth conditions are "
              f"deep_run_pairs.json's model list crossed with its representation list, "
              f"which is twenty-four pairs today. The cap would refuse the depth run.")
        check(mode == 'pair_grid',
              f"{name}'s scope is mode {mode!r}; the depth conditions run on a cross "
              f"product of models and representations, which this file calls 'pair_grid'.")
        check(entry.get('scope', {}).get('selection_file') == 'deep_run_pairs.json',
              f"{name}'s scope does not name deep_run_pairs.json as the file that decides "
              f"which pairs it runs on.")

    gen = load_generator()
    check(list(gen.PAIR_SUBSET) == ['censoring'],
          f"the validation generator's pair-subset set is {sorted(gen.PAIR_SUBSET)}; "
          f"censoring is meant to be the only one on this side.")
    check(gen.BREADTH_GRID == ['gaussian', 'grouped_wider', 'grouped_shifted'],
          f"the breadth grid changed to {gen.BREADTH_GRID}. Nineteen arrays are already "
          f"running against the old three, and their output paths are keyed to it.")
    # getattr, not an attribute access: with the list removed from the generator this
    # has to print the sentence below rather than raise AttributeError from the harness.
    declared_in_generator = getattr(gen, 'DEPTH_ON_VALIDATION', None)
    check(declared_in_generator == list(DEPTH),
          f"the generator reads {declared_in_generator} as declared for this side; the "
          f"file declares {list(DEPTH)}.")
    return (f"mode is pair_grid on all three, the pair-subset set is still censoring "
            f"alone, and the breadth grid is still {' '.join(gen.BREADTH_GRID)}")


def the_depth_submission_asks_for_six_conditions_and_keeps_its_output_path():
    """--include-depth-conditions emits the breadth three plus the depth three, in order.

    THE ORDER IS LOAD-BEARING and so is the count. The generator tags the results
    directory `_{n}cond_` plus the first four letters of each non-breadth condition, so
    this exact list is what produces `_6cond_stud_outl_lapl`. Nineteen arrays submitted
    on 2026-09-06 are writing into directories with that name. Changing the list, or its
    order, points a resubmission at a different directory and splits the run in two.
    """
    want = ['gaussian', 'grouped_wider', 'grouped_shifted',
            'student_t_nu5', 'outlier_p10', 'laplace']
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, str(VALGEN), '--include-depth-conditions',
             '--runtime-selection', str(ROOT / 'deep_run_pairs.json'),
             '--out-dir', tmp],
            capture_output=True, text=True)
        if not check(r.returncode == 0,
                     f"the depth submission was refused:\n{r.stderr[-800:]}"):
            return None
        line = [l for l in r.stdout.splitlines() if l.startswith('Conditions')][0]
        got = line.split(':', 1)[1].split()
        check(got == want,
              f"the depth submission asks for {got}; it must ask for {want}, in that "
              f"order, because the results directory is named from it.")
        script = Path(tmp, 'val_rf.sh').read_text()
        check('6cond_stud_outl_lapl' in script,
              "the depth scripts no longer write to `..._6cond_stud_outl_lapl_<dataset>`. "
              "Nineteen arrays submitted 2026-09-06 are writing there; a different path "
              "splits the run across two directories with nothing to join them.")
    return f"--include-depth-conditions asks for {' '.join(want)}, tag 6cond_stud_outl_lapl"


def an_undeclared_depth_condition_is_refused():
    """Taking a condition off this side has to be an edit to the settled file.

    The generator is imported and its two derived lists are moved, rather than the
    settled file being edited on disk: the point is the refusal, not a second copy of
    the file. main() reads the module globals, so this is the same code path a real
    narrowing would take.
    """
    gen = load_generator()
    gen.DEPTH_ON_VALIDATION = ['student_t_nu5', 'outlier_p10']
    gen.DEPTH_NOT_DECLARED = ['laplace']
    with tempfile.TemporaryDirectory() as tmp:
        argv = sys.argv
        sys.argv = ['generate_scripts.py', '--conditions', 'laplace',
                    '--models', 'RF', '--reps', 'PDV', '--out-dir', tmp]
        try:
            gen.main()
        except SystemExit as exc:
            check(exc.code != 0, "an undeclared depth condition was accepted")
        else:
            failures.append("an undeclared depth condition was accepted, no refusal at all")
        finally:
            sys.argv = argv

        # And the declared ones still go through on the same code path.
        sys.argv = ['generate_scripts.py', '--conditions', 'student_t_nu5',
                    '--models', 'RF', '--reps', 'PDV', '--out-dir', tmp]
        try:
            gen.main()
        except SystemExit as exc:
            check(exc.code in (0, None),
                  f"a declared depth condition was refused, exit {exc.code}")
        finally:
            sys.argv = argv
    return "laplace refused once undeclared; student_t_nu5 still accepted while declared"


def the_breadth_grid_is_untouched():
    """The nineteen val_*.sh on disk still state the breadth three and nothing else.

    Those nineteen are NOT in git -- `.gitignore` line 87 is `slurm_scripts_*/*.sh` --
    so a fresh checkout, the cluster's included, has none of them. Asserting that
    nineteen exist would then fail for a reason that has nothing to do with the noise
    conditions, so where the directory is empty this generates the breadth default into
    a temporary directory and asserts the same thing about what the generator writes.
    """
    where = ROOT / 'slurm_scripts_validation_rerun'
    scripts = sorted(p for p in where.glob('val_*.sh'))
    tmp = None
    if not scripts:
        tmp = tempfile.TemporaryDirectory()
        r = subprocess.run([sys.executable, str(VALGEN), '--out-dir', tmp.name],
                           capture_output=True, text=True)
        if not check(r.returncode == 0,
                     f"no val_*.sh on disk and the breadth grid would not "
                     f"generate:\n{r.stderr[-800:]}"):
            tmp.cleanup()
            return None
        where = Path(tmp.name)
        scripts = sorted(where.glob('val_*.sh'))
    check(len(scripts) == 19, f"{len(scripts)} val_*.sh in {where}, expected 19")
    for p in scripts:
        text = p.read_text()
        check('--conditions gaussian grouped_wider grouped_shifted ' in text,
              f"{p.name} no longer states the breadth three on its command line")
        for name in DEPTH:
            check(name not in text,
                  f"{p.name} carries {name}. This directory is the BREADTH grid; the depth "
                  f"conditions belong in the depth submission, which writes elsewhere.")
    note = (f"all {len(scripts)} breadth scripts in {where} state the breadth three only")
    if tmp is not None:
        tmp.cleanup()
        note += " (generated: none were on disk)"
    return note


def main():
    print(__doc__.strip().splitlines()[0])
    print()
    for fn in (the_file_declares_the_experimental_datasets,
               the_depth_conditions_are_not_a_pair_subset,
               the_depth_submission_asks_for_six_conditions_and_keeps_its_output_path,
               an_undeclared_depth_condition_is_refused,
               the_breadth_grid_is_untouched):
        before = len(failures)
        # A check that raises is a failure, not a crash of the whole run: with the
        # scope blocks taken out of noise_conditions.json the later checks would
        # otherwise never print, and the point of this file is its message.
        try:
            note = fn()
        except Exception as exc:
            failures.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")
            note = None
        status = 'OK  ' if len(failures) == before else 'FAIL'
        print(f"  {status}  {fn.__name__.replace('_', ' ')}")
        if note:
            print(f"        {note}")
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("OK: the three depth conditions are declared for logD, Caco-2 and hERG, "
          "and the depth submission is unchanged.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
