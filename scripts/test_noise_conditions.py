#!/usr/bin/env python
"""The settled condition set is binding on the Python injector too.

`noise_conditions.json` at the repository root says what the study runs. The Rust
side is held to it by `rust/tests/noise_gates.rs`; this holds the Python side.

WHY A TEST AND NOT A NOTE
-------------------------
`scripts/noise_strategy_params.json` used to be a settings file that nothing read.
It was never passed to the binary, so for the life of the project it silently meant
nothing while everyone believed it was in force, and the defaults that actually ran
were whatever the code happened to say. A file describing the run, and a run that
ignores the file, is worse than no file.

The same failure in its other form is the two injectors drifting apart -- one
scheme, two implementations, nothing checking (RERUN_PLAN.md section 2.3, guard 10
of section 0.6). So the names in the settled set have to resolve on BOTH sides, to
the same parameters, or this fails.

    python scripts/test_noise_conditions.py

Exits non-zero on any failure. It is part of the preflight; no cluster time is
spent until it passes.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS_FILE = os.path.join(REPO, 'noise_conditions.json')

sys.path.insert(0, os.path.expanduser('~/repos/NoiseInject'))


def load_settled():
    with open(CONDITIONS_FILE) as f:
        return json.load(f)


def names(spec, key):
    return [entry['name'] for entry in spec[key]]


def check(condition, message, failures):
    if not condition:
        failures.append(message)
    return condition


def check_generator(spec, failures):
    """The QM9 job generator's condition table must match the settled set.

    Absent (not yet written, or on the cluster only) is not a failure; disagreeing is.
    """
    import importlib.util

    path = os.path.join(REPO, 'slurm_scripts_qm9_rerun', 'generate_scripts.py')
    if not os.path.exists(path):
        print("\n  (the QM9 job generator is not in this checkout — skipping that check)")
        return

    module_spec = importlib.util.spec_from_file_location('qm9_generate_scripts', path)
    module = importlib.util.module_from_spec(module_spec)
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:                                   # pragma: no cover
        failures.append(f"the QM9 job generator does not import ({exc}), so its condition "
                        f"table cannot be checked against the settled set")
        return
    if not hasattr(module, 'CONDITIONS'):
        print("\n  (the job generator has no CONDITIONS table — skipping that check)")
        return

    settled_stage_1 = set(names(spec, 'stage_1_full_grid'))
    settings = spec['settings_that_follow']

    # The generator keys its table on the registry name itself, so there is no
    # translation to keep in step here. The abbreviated job-script labels it used
    # for one day are kept only so an older checkout is reported honestly rather
    # than as an unknown condition.
    RETIRED_LABELS = {
        'groupedwide': 'grouped_wider',
        'groupedshift': 'grouped_shifted',
        'studentt': 'student_t_nu5',
        'outlier': 'outlier_p10',
    }
    known = set(names(spec, 'stage_1_full_grid')) | set(names(spec, 'stage_2_depth_only'))
    generator_stage_1 = set()
    for label, entry in module.CONDITIONS.items():
        flags, _levels, in_stage_1 = entry[0], entry[1], entry[2]
        name = label if label in known else RETIRED_LABELS.get(label)
        if name is None:
            failures.append(f"the job generator has a condition '{label}' that the settled set "
                            f"does not name. Add it to noise_conditions.json with its evidence, "
                            f"or take it out of the generator")
            continue
        if in_stage_1:
            generator_stage_1.add(name)
        # the single-setting decisions, where the generator spells them out
        if name == 'student_t_nu5' and '--nu' in flags:
            got = float(flags.split('--nu')[1].split()[0])
            check(abs(got - settings['nu']) < 1e-9,
                  f"the job generator runs Student-t at nu = {got}; the settled setting is "
                  f"{settings['nu']}", failures)
        if name == 'outlier_p10' and '--outlier-p' in flags:
            got = float(flags.split('--outlier-p')[1].split()[0])
            check(abs(got - settings['outlier_p']) < 1e-9,
                  f"the job generator contaminates {got} of labels; the settled setting is "
                  f"{settings['outlier_p']}", failures)

    check(generator_stage_1 == settled_stage_1,
          f"the job generator's full grid is {sorted(generator_stage_1)}, the settled set is "
          f"{sorted(settled_stage_1)}. One of them is about to run and the other is about to be "
          f"believed", failures)
    if not failures:
        print(f"  job generator     : full grid {sorted(generator_stage_1)} — agrees")


def check_pair_subset_scope():
    """A condition the settled file restricts to a pair subset must actually be restricted.

    Censoring is the only one today (RERUN_PLAN.md 13.13, the author's call 2026-08-27): on
    QM9 it runs on about five model-and-representation pairs rather than all 78, because the
    question there is how big the effect is and not which model resists it best. That is 350
    runs instead of 5,460.

    Three things have to hold together or the saving silently evaporates:
      1. the scope lives in noise_conditions.json, not in the generator
      2. the QM9 generator's default condition set for the screen and the main grid excludes it
      3. asking for it by name without --models and --reps is refused, not defaulted

    The scope is QM9 ONLY. Censoring stays a full condition in the uncertainty runs on the
    experimental datasets, where it is one of only two conditions that can answer the
    which-molecules question -- so this also asserts the uncertainty generator still carries it.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    spec = json.loads((repo / 'noise_conditions.json').read_text())

    scoped = {c['name']: c['scope']
              for g in ('stage_1_full_grid', 'stage_2_depth_only')
              for c in spec[g]
              if c.get('scope', {}).get('mode') == 'pair_subset'}
    assert 'censoring' in scoped, \
        "noise_conditions.json no longer restricts censoring to a pair subset " \
        "(RERUN_PLAN.md 13.13). If that is deliberate the plan has to say so first."
    # Widened 2026-08-27 on the author's ruling: the lab-dataset robustness runs
    # use the same pairing as QM9. The uncertainty runs do not -- censoring is
    # one of only two conditions there that can say WHICH molecules were damaged.
    applies = scoped['censoring'].get('applies_to', [])
    assert 'qm9_grid' in applies and 'validation_robustness' in applies, \
        f"censoring's pair subset must cover the QM9 grid and the validation robustness " \
        f"runs; it covers {applies}"
    assert 'uncertainty_runs' in scoped['censoring'].get('full_breadth_in', []), \
        "censoring must stay a FULL condition in the uncertainty runs"
    n = scoped['censoring']['n_pairs']
    assert 1 <= n <= 20, f"censoring's n_pairs is {n}, which is not a small subset"

    gen = repo / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'

    def run(*args):
        return subprocess.run([sys.executable, str(gen), *args],
                              capture_output=True, text=True, cwd=gen.parent)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # The deep run drops the pair-subset conditions too: it is more pairs than the
        # subset, so leaving censoring in would either refuse or quietly restore its cost.
        for stage, extra in (('0', []), ('1', []),
                             # ngboost is on the author's deep-run shortlist
                             # (RERUN_PLAN.md 13.15) and the generator refuses a
                             # deep run without it, so this fixture carries it.
                             ('2', ['--models', 'lgb', 'rf', 'dnn', 'svm', 'ngboost',
                                    '--reps', 'pdv', 'ecfp4', 'mhggnn'])):
            r = run('--stage', stage, *extra, '--out-dir', tmp)
            assert r.returncode == 0, f"stage {stage} failed to generate:\n{r.stderr[-800:]}"
            line = [l for l in r.stdout.splitlines() if 'conditions:' in l]
            assert line, f"stage {stage} did not report its conditions"
            assert 'censoring' not in line[0], (
                f"stage {stage} still runs censoring at full breadth by default: {line[0].strip()}")

        r = run('--stage', '1', '--conditions', 'censoring', '--out-dir', tmp)
        assert r.returncode != 0, \
            "asking for censoring without --models and --reps was accepted; it must be refused"
        assert '13.13' in r.stderr, \
            f"the refusal does not point at the decision:\n{r.stderr[-400:]}"

        r = run('--stage', '1', '--conditions', 'censoring',
                '--models', 'lgb', 'rf', '--reps', 'pdv', '--out-dir', tmp)
        assert r.returncode == 0, \
            f"censoring on a named pair subset was refused:\n{r.stderr[-800:]}"

    unc = (repo / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py').read_text()
    assert 'scope' not in unc, \
        "the uncertainty generator has picked up the pair-subset scope; censoring must " \
        "stay a full condition there (RERUN_PLAN.md 13.1 item 6)"

    # The validation robustness generator MUST honour it, the same way QM9 does.
    valgen = repo / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(valgen), '--out-dir', tmp],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        line = [l for l in r.stdout.splitlines() if 'Conditions' in l][0]
        assert 'censoring' not in line, \
            f"the validation robustness runs still take censoring at full breadth: {line.strip()}"

        r = subprocess.run([sys.executable, str(valgen), '--conditions', 'censoring',
                            '--out-dir', tmp], capture_output=True, text=True)
        assert r.returncode != 0, \
            "asking for censoring without --models and --reps was accepted there too"
        assert '13.13' in r.stderr, f"the refusal does not point at the decision: {r.stderr[-300:]}"

        r = subprocess.run([sys.executable, str(valgen), '--conditions', 'censoring',
                            '--models', 'LightGBM', 'RF', '--reps', 'PDV', '--out-dir', tmp],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"censoring on named pairs was refused:\n{r.stderr[-800:]}"
    assert 'censoring' in [c['name'] for c in spec['stage_1_full_grid']], \
        "censoring left the main-grid group, which removes it from the uncertainty runs too"


def main():
    spec = load_settled()
    failures = []

    stage_1 = names(spec, 'stage_1_full_grid')
    stage_2 = names(spec, 'stage_2_depth_only')
    dropped = names(spec, 'not_run')
    settings = spec['settings_that_follow']

    print(f"settled {spec['settled']}, evidence: {spec['evidence']}")
    print(f"  stage 1, full grid : {', '.join(stage_1)}")
    print(f"  stage 2, depth only: {', '.join(stage_2)}")
    print(f"  not run            : {', '.join(dropped)}")

    # --- 1. nothing is both run and not run --------------------------------
    for name in dropped:
        check(name not in stage_1 and name not in stage_2,
              f"'{name}' is listed as not run AND as something that runs. The evidence for "
              f"dropping it is RERUN_PLAN.md 13.9; if that has been revisited, move the entry "
              f"rather than listing it twice", failures)

    # --- 2. the four redundant settings stay dropped ------------------------
    # Named explicitly so that putting one back is a deliberate edit to a test, with
    # the measurement in front of whoever does it, not a quiet line in a job script.
    for name in ('student_t_nu10', 'student_t_nu3', 'outlier_p01', 'outlier_p05'):
        check(name in dropped,
              f"'{name}' was dropped on 2026-08-27: twelve replicates on real QM9 put every "
              f"Student-t and Outlier setting within 0.006 R2 of Gaussian at the reporting "
              f"level, against a test that could have detected 0.006 to 0.021. Putting it back "
              f"costs 4,680 training runs, 9.1% of the old design, and needs a reason that "
              f"measurement does not already answer", failures)

    # --- 3. the skewed draw is not built ------------------------------------
    skewed = [e for e in spec['not_run'] if e['name'] == 'skewed_draw']
    check(skewed and skewed[0].get('never_implemented') is True,
          "the skewed draw was tested and rejected; it exists only in "
          "scripts/setting_selection_test.py and is not to be built into either injector",
          failures)

    # --- 4. every name the study runs resolves in the Python injector -------
    try:
        from noiseInject import CONDITIONS
    except ImportError as exc:
        print(f"\nnoiseInject is not importable ({exc}).")
        print("Install it: pip install -e ~/repos/NoiseInject")
        return 2

    for name in stage_1 + stage_2:
        check(name in CONDITIONS,
              f"the study runs '{name}' but the Python injector has no such condition. Known: "
              f"{sorted(CONDITIONS)}. This is the drift the cross-check exists to stop, in the "
              f"form where one side simply cannot build what the other runs", failures)

    # --- 5. the parameters agree with the settled settings ------------------
    # The single-setting decisions are only real if the parameters follow them.
    checks = [
        ('student_t_nu5', 'nu', settings['nu']),
        ('outlier_p10', 'p', settings['outlier_p']),
        ('outlier_p10', 'lam', settings['lambda']),
        ('grouped_wider', 'lam', settings['lambda']),
        ('grouped_wider', 'group_fraction', settings['group_fraction']),
        ('grouped_shifted', 'rho', settings['group_variance_share']),
    ]
    for condition, key, expected in checks:
        if condition not in CONDITIONS:
            continue                      # already reported above
        got = CONDITIONS[condition].get(key)
        check(got is not None and abs(float(got) - float(expected)) < 1e-9,
              f"the Python injector builds '{condition}' with {key} = {got}, but "
              f"noise_conditions.json settles it at {expected}. Two sources of truth for one "
              f"number is how the injectors drifted apart the first time", failures)

    # --- 6. the job generator agrees with the settled set -------------------
    # It restates the conditions as command-line flags rather than reading this file,
    # which is understandable -- it has to turn them into flags -- but it is also
    # exactly how a second source of truth starts. So the restatement is checked.
    check_generator(spec, failures)

    # --- 7. censoring runs on its own axis ----------------------------------
    check('censoring' in stage_1,
          "censoring is not in the full grid. It is not zero-mean and cannot be dose-matched, "
          "so it runs on its own axis -- and it is the largest effect in the whole study "
          "(NOISE_DESIGN.md 5.3b)", failures)

    # --- 8. censoring is restricted to a pair subset on QM9 ------------------
    # RERUN_PLAN.md 13.13, the author's call 2026-08-27. This runs the real generator.
    try:
        check_pair_subset_scope()
    except AssertionError as e:
        failures.append(str(e))

    # --- report -------------------------------------------------------------
    if failures:
        print(f"\nFAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"\n  - {f}")
        return 1
    print(f"\nPASSED — {len(stage_1) + len(stage_2)} conditions resolve on both sides "
          f"with the settled parameters")
    return 0


if __name__ == '__main__':
    sys.exit(main())
