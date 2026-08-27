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

    # Its keys are job-script labels, not registry names, so match on the flags.
    settled_stage_1 = set(names(spec, 'stage_1_full_grid'))
    settings = spec['settings_that_follow']
    label_to_name = {
        'gaussian': 'gaussian',
        'groupedwide': 'grouped_wider',
        'groupedshift': 'grouped_shifted',
        'censoring': 'censoring',
        'studentt': 'student_t_nu5',
        'outlier': 'outlier_p10',
        'laplace': 'laplace',
    }
    generator_stage_1 = set()
    for label, entry in module.CONDITIONS.items():
        flags, _levels, in_stage_1 = entry[0], entry[1], entry[2]
        name = label_to_name.get(label)
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
