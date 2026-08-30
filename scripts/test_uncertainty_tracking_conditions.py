#!/usr/bin/env python
"""Which noise conditions can answer "did the model spot the corrupted molecules"?

THE QUESTION, AND WHY IT NEEDS A GUARD
--------------------------------------
One of the study's three uncertainty questions is whether a model becomes less
sure about the molecules whose labels were actually corrupted. Answering it means
correlating each molecule's predicted uncertainty against the amount of noise its
own label received.

That correlation only exists when the amount VARIES from molecule to molecule.
Six of the eleven conditions give every molecule the same amount by construction:
the three shape-only ones (Gaussian, Laplace, Student-t) perturb everything
equally, and the shifted grouped condition moves whole scaffold families by an
offset drawn from one distribution, so what differs by family is the DIRECTION,
not a magnitude anything could rank.

For those six the correlation is UNDEFINED, not zero. Reporting a zero reads as
"this model cannot find corrupted labels" when the truth is that nothing was
asked. Author's instruction, 2026-08-29: Gaussian is the deliberate negative
baseline, the conditions that CAN answer it are to be named in the plan and used
for the uncertainty work, and the ones that cannot must not be used for it.

WHAT THIS CHECKS
----------------
1. The split is DERIVED from the injector, by asking it for the per-molecule
   amount on real-shaped labels and counting distinct values -- never from a list
   restated here. A restated list is how every other correspondence in this
   project went stale.
2. The derived split matches the two names the pipelines already carry, so a new
   condition cannot join the study on one side only.
3. The analysis module leaves the per-molecule detection answer UNDEFINED, not
   False, for a condition whose amount does not vary.

Run: python scripts/test_uncertainty_tracking_conditions.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.expanduser('~/repos/NoiseInject'))

from noiseInject import CONDITIONS, NoiseInjectorRegression  # noqa: E402

# The negative baseline. Named on its own because it is not a limitation: a
# condition that corrupts every molecule equally is exactly what a control for
# "the model tracks something else about the molecule" looks like.
NEGATIVE_BASELINE = 'gaussian'

# How many molecules and groups the probe uses. Large enough that a condition
# affecting a small fraction still shows at least two distinct amounts.
N_PROBE = 4000
N_GROUPS = 400


def _amount_per_molecule(condition, y, groups):
    """What each molecule's label receives, at a fixed level, without corrupting it."""
    inj = NoiseInjectorRegression.from_condition(condition, random_state=0,
                                                selection_state=1)
    # Censoring's level is a FRACTION of labels clipped, so a level of 1.0 would
    # clip everything and the amount would be constant for a reason that has
    # nothing to do with the condition.
    level = 0.25 if CONDITIONS[condition]['strategy'] == 'censoring' else 1.0
    return inj.noise_scale(y, level, reference=y, groups=groups,
                           reference_groups=groups)


def derive_split():
    """Ask the injector which conditions vary per molecule. Returns two sorted lists."""
    rng = np.random.RandomState(0)
    y = rng.randn(N_PROBE) * 1.27 + 5.0
    groups = rng.randint(0, N_GROUPS, N_PROBE)
    same, varies = [], []
    for name in sorted(CONDITIONS):
        amounts = _amount_per_molecule(name, y, groups)
        distinct = len(np.unique(np.round(np.asarray(amounts, dtype=float), 12)))
        (varies if distinct > 1 else same).append(name)
    return same, varies


def the_split_is_what_the_plan_says():
    """Derived from the injector, then checked against the two settled lists."""
    same, varies = derive_split()

    expected_same = ['gaussian', 'grouped_shifted', 'laplace',
                     'student_t_nu10', 'student_t_nu3', 'student_t_nu5']
    expected_varies = ['censoring', 'grouped_wider',
                       'outlier_p01', 'outlier_p05', 'outlier_p10']

    assert same == expected_same, (
        f"the conditions that give every molecule the same amount have changed.\n"
        f"  derived from the injector: {same}\n"
        f"  RERUN_PLAN.md §3.4.7 says: {expected_same}\n"
        f"If a condition moved between the two lists, the uncertainty runs are "
        f"asking a question of a condition that cannot answer it, or not asking "
        f"it of one that can. Update the plan AND the job generators together.")
    assert varies == expected_varies, (
        f"the conditions whose amount varies per molecule have changed.\n"
        f"  derived from the injector: {varies}\n"
        f"  RERUN_PLAN.md §3.4.7 says: {expected_varies}")

    assert NEGATIVE_BASELINE in same, (
        f"{NEGATIVE_BASELINE} is the negative baseline for the per-molecule "
        f"detection question, so its amount MUST be the same for every molecule. "
        f"It is not, which means the baseline is no longer a baseline.")

    print(f"    same amount for every molecule ({len(same)}): {', '.join(same)}")
    print(f"    varies per molecule ({len(varies)}): {', '.join(varies)}")
    print(f"    negative baseline: {NEGATIVE_BASELINE}")


def a_flat_condition_is_undefined_not_zero():
    """The analysis must not report a detection answer where none was asked."""
    import pandas as pd
    import uncertainty_stats as us

    rng = np.random.RandomState(1)
    n = 300
    # A condition whose amount is the same everywhere: the shape column is a
    # constant, so nothing can correlate with it.
    # `condition` is the column the analysis keys a cell on (CELL_COLS); the
    # pipelines write `noise_type` and the loader renames it. Both are set here
    # so this test does not depend on which side of the rename it sits.
    frame = pd.DataFrame({
        'model': 'qrf', 'rep': 'pdv', 'dataset': 'QM9', 'split': 'train_oof',
        'fold': 0,
        'noise_type': NEGATIVE_BASELINE,
        'condition': NEGATIVE_BASELINE,
        'sigma': np.repeat([0.0, 1.0], n),
        'uncertainty': np.abs(rng.randn(2 * n)) + 0.5,
        'noise_pattern': np.ones(2 * n),          # constant, by construction
        'noise_pattern_pred': np.ones(2 * n),
        'injected_noise': np.concatenate([np.zeros(n), rng.randn(n)]),
        'y_true_clean': rng.randn(2 * n),
        'y_pred': rng.randn(2 * n),
    })
    out = us.confound_controlled_effect(frame)
    assert len(out), "the analysis returned no rows for a flat condition"
    # Undefined can arrive as None, as a float nan, or as pandas' own missing
    # value, depending on the column's dtype. All three mean "not answered"; only
    # a real True or False would be the defect.
    verdicts = list(out['is_detection'])
    assert all(pd.isna(v) for v in verdicts), (
        f"a condition that corrupts every molecule equally has NO per-molecule "
        f"detection answer, so the verdict must be undefined. Got {verdicts}. "
        f"Writing False here reads as 'the model failed to find the corrupted "
        f"molecules' when nothing was asked of it.")
    print(f"    {NEGATIVE_BASELINE}: {len(out)} cell(s), every detection verdict "
          f"left undefined rather than False")


def check(name, fn):
    print(f"  {name}")
    try:
        fn()
        print("    ok")
        return True
    except Exception as exc:
        print(f"    FAIL: {type(exc).__name__}: {exc}")
        return False


def main():
    print("which conditions can answer the per-molecule uncertainty question "
          "(RERUN_PLAN.md §3.4.7)")
    ok = True
    ok &= check("the split is derived from the injector and matches the plan",
                the_split_is_what_the_plan_says)
    ok &= check("a condition that corrupts everything equally is undefined, not zero",
                a_flat_condition_is_undefined_not_zero)
    print("\nOK" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
