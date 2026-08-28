"""Gates on the three checks in `scripts/decomposition_controls.py`.

Each one fails if the fix it guards is removed. They are deliberately of two
kinds:

  FAST      arithmetic and wiring, on small arrays. No QM9, no fitting. These
            run every time and are what a change to the split has to survive.
  MEASURED  the three checks themselves, on real QM9 with every model scored out
            of fold on scaffold groups. Slow -- minutes, not seconds -- so they
            run only with --measured, and a run without it says so rather than
            reporting six passes it did not perform.

Run:  python scripts/test_decomposition_controls.py
      python scripts/test_decomposition_controls.py --measured

THE THREE CHECKS
================
1. EVEN NOISE, AND THE MODEL MUST FIND NOTHING. Every molecule gets the same
   amount, so there is nothing to localise. A model whose data-noise term still
   correlates with the pattern is tracking something else about the molecule,
   and its positive result under an uneven condition is that same artefact.

2. GRADED NOISE, AND THE MODEL MUST RANK. Every earlier correlation came from
   two noise amounts, so it measured whether a model could tell two blocks
   apart -- not whether it could rank molecules. If the correlation collapses
   under a graded scale, that is the answer and it is reported as one.

3. THE VARIATIONAL LAYER, MEASURED THE WAY THE GAUSSIAN PROCESS WAS. It had a
   built-and-tested noise head and no correlation number, no zero-noise control
   and no accuracy comparison against the ordinary variational model. Until it
   has all three it must not be reported beside the Gaussian process as though
   the two carry the same evidence.

Every correlation is reported BESIDE its zero-noise value and never alone.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decomposition_controls as dc  # noqa: E402
from uncertainty_decomposition import (  # noqa: E402
    CONSTANT, NONE, PER_MOLECULE, SUPPORT, support)

FAILURES = []
SKIPPED = []
BLOCKED = []


class Blocked(Exception):
    """This interpreter cannot run the gate. Never reported as a pass."""


def check(name, fn):
    try:
        fn()
        print(f"  ok       {name}")
    except Blocked as e:
        print(f"  BLOCKED  {name}: {e}")
        BLOCKED.append(name)
    except AssertionError as e:
        print(f"  FAIL     {name}: {e}")
        FAILURES.append(name)
    except Exception as e:
        print(f"  ERROR    {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Fast gates -- the designs themselves, and the wiring
# ---------------------------------------------------------------------------

def test_the_even_control_is_scored_against_the_uneven_pattern():
    """The even design's own shape is CONSTANT, and a rank correlation against a
    constant is undefined, not zero. Scoring the control against its own shape
    would produce a blank where the control's answer belongs. It is scored
    against the patterns from the uneven designs instead -- which is the actual
    question: does a model that scores well under two_block or graded still
    score against those same patterns when the noise carries no such structure?
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    refs = dc.reference_patterns('even', x, 2.0, 0)
    assert set(refs) == {'two_block', 'graded'}, (
        f"the even control is scored against {sorted(refs)}; it must be scored "
        f"against the patterns from the uneven cases or it answers nothing")
    for name, pattern in refs.items():
        assert np.ptp(pattern) > 1e-9, f"reference {name} is constant"
    for uneven in ('two_block', 'graded'):
        assert list(dc.reference_patterns(uneven, x, 2.0, 0)) == [uneven], (
            f"{uneven} must be scored against its own shape")


def test_the_even_design_really_is_even():
    """If this drifts, check 1 stops being a control and starts being a second
    copy of check 2."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    scales = dc.noise_scales('even', x, 0.6, 2.0, 0)
    assert np.ptp(scales) <= 1e-12, (
        f"the even design varies by {np.ptp(scales):.6g} across molecules, so "
        f"there IS something to find and it is no longer a negative control")
    assert scales[0] > 0, "the even design delivers no noise at all"


def test_the_graded_design_really_is_graded():
    """Two blocks is what every earlier number came from. A design with three or
    four distinct values would still be blocks, not a ranking."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    graded = dc.noise_scales('graded', x, 0.6, 2.0, 0)
    two = dc.noise_scales('two_block', x, 0.6, 2.0, 0)
    assert len(np.unique(np.round(two, 12))) == 2, (
        "the two-block design no longer has exactly two amounts, so it is not "
        "the design the earlier correlations came from")
    assert len(np.unique(np.round(graded, 12))) > 200, (
        f"the graded design has only {len(np.unique(np.round(graded, 12)))} "
        f"distinct amounts across 300 molecules; a rank correlation against it "
        f"would still be measuring block membership")


def test_the_three_designs_deliver_the_same_amount_of_noise():
    """Dose-matched on the second moment, because that is what adds.

    Without this a design that simply added more noise would look like a design
    a model localises better, and the comparison between the three would be a
    comparison of doses (RERUN_PLAN.md 2.2, the six strategies that were one
    strategy at six doses).
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 4))
    delivered = {c: float(np.sqrt(np.mean(dc.noise_scales(c, x, 0.6, 2.0, 0) ** 2)))
                 for c in dc.CONDITIONS}
    lo, hi = min(delivered.values()), max(delivered.values())
    assert hi / lo - 1.0 < 1e-9, (
        f"the three designs deliver different amounts: {delivered}. A model "
        f"that looks better under one of them would be looking better at a "
        f"different dose.")
    assert abs(lo - 0.6 * 2.0) < 1e-9, (
        f"the delivered amount is {lo:.6g}, not level x spread = 1.2")


def test_zero_level_keeps_the_shape_and_drops_the_amount():
    """The zero-noise control is the SAME condition with nothing delivered. If
    the shape changed too, the correlation at zero would not be the confound the
    correlation at the level has to be corrected for."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    for condition in dc.CONDITIONS:
        at_zero = dc.noise_scales(condition, x, 0.0, 2.0, 0)
        assert np.allclose(at_zero, 0.0), f"{condition} delivers noise at level 0"
        y = np.linspace(0, 1, 300)
        assert np.allclose(dc.apply_noise(y, at_zero, 7), y), (
            f"{condition}: labels moved at level 0")


def test_a_constant_term_is_reported_as_undefined_and_not_as_zero():
    """A rank correlation against a constant column is undefined. Writing 0.0
    would read as `the model found nothing`, which is a claim about the model;
    the truth is the column cannot answer the question at all. The support table
    is what says which (RERUN_PLAN.md 5.5a point 5)."""
    assert support('gauche_rbf')[0] == CONSTANT
    assert support('dnn_bnn_full_variational')[0] == CONSTANT
    assert support('heteroscedastic_gp')[0] == PER_MOLECULE
    assert support('dnn_bnn_full_variational_hetero')[0] == PER_MOLECULE
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'decomposition_controls.py')).read()
    assert "float('nan') if (v is None or constant)" in src, (
        "a constant term is no longer reported as undefined")


def test_the_measurement_calls_the_pipeline_and_not_a_copy():
    """A control that measures a copy of the code proves nothing about the code
    that runs. The variational split here comes from `sample_network_split` in
    models/models.py, and the forest and Gaussian process splits from the shared
    module -- so a change to any of them changes this measurement."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'decomposition_controls.py')).read()
    for name in ('sample_network_split', 'decompose_forest', 'decompose_gp',
                 'decompose_hetero_gp',
                 'apply_bayesian_transformation_full_variational'):
        assert name in src, f"the measurement no longer uses {name}"


def test_every_model_measured_has_a_support_entry():
    for name in dc.MODELS:
        assert name in SUPPORT, name
        alea, epis = support(name)
        assert not (alea == NONE and epis == NONE), (
            f"{name} has no components at all and should not be in this "
            f"measurement")


def test_the_out_of_fold_split_is_scaffold_grouped():
    """Scoring a molecule with a model that fitted its corrupted label measures
    memorisation. And a RANDOM inner split puts close analogues of every
    held-out molecule in the fit set, so the uncertainty comes from an
    interpolation regime while the study's own splits are extrapolation."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'decomposition_controls.py')).read()
    assert 'GroupKFold' in src, "the inner split is no longer grouped"
    assert 'MurckoScaffold' in src, "the groups are no longer scaffolds"
    assert '__acyclic__' in src, (
        "acyclic molecules are no longer singletons, so they collapse into one "
        "group GroupKFold must place entirely in one fold")


# ---------------------------------------------------------------------------
# The measured gates -- real QM9, out of fold on scaffold groups
# ---------------------------------------------------------------------------

_MEASURED = {}


def _measured(models, conditions):
    """One measurement per (models, conditions), reused across the gates."""
    key = (tuple(models), tuple(conditions))
    if key not in _MEASURED:
        _MEASURED[key] = dc.measure(
            models=list(models), conditions=list(conditions),
            level=float(os.environ.get('DECOMP_LEVEL', 0.6)),
            n_molecules=int(os.environ.get('DECOMP_N', 1500)),
            n_folds=int(os.environ.get('DECOMP_FOLDS', 3)),
            seed=0, verbose=True)
    return _MEASURED[key]


def _effect(rows, model, condition, term, level, reference=None):
    """The correlation MINUS its zero-noise value. Never reported alone."""
    at_level = dc.lookup(rows, model, condition, term, level, reference)
    at_zero = dc.lookup(rows, model, condition, term, 0.0, reference)
    if at_level['support'] == 'blocked':
        raise Blocked(at_level.get('blocked', 'model could not be built'))
    return at_level, at_zero


def check_1_even_noise_finds_nothing():
    """Give every molecule the same amount and the data-noise term must find
    nothing. A model that still scores high is tracking the molecule, not the
    corruption -- and its positive result under an uneven condition is the same
    artefact."""
    level = float(os.environ.get('DECOMP_LEVEL', 0.6))
    rows = _measured(['rf', 'heteroscedastic_gp'], ['even'])
    bad = []
    for model in ('rf', 'heteroscedastic_gp'):
        for reference in ('two_block', 'graded'):
            try:
                at_level, at_zero = _effect(rows, model, 'even', 'aleatoric',
                                            level, reference)
            except Blocked:
                continue
            rho = at_level['spearman_vs_pattern']
            rho0 = at_zero['spearman_vs_pattern']
            if np.isnan(rho):
                continue       # a constant term: undefined, not a failure
            effect = rho - (0.0 if np.isnan(rho0) else rho0)
            print(f"      {model:<22} vs {reference:<10} rho {rho:+.4f}, "
                  f"at zero noise {rho0:+.4f}, effect {effect:+.4f}")
            if abs(effect) > 0.25:
                bad.append(f"{model} vs {reference} {effect:+.4f}")
    assert not bad, (
        "under EVEN noise there is nothing to find, and these models' "
        "data-noise term still tracks the pattern: " + ", ".join(bad)
        + ". Their positive result under an uneven condition is that same "
          "artefact and must not be reported as noise localisation.")


def check_2_graded_noise_is_ranked():
    """Every earlier correlation used two noise amounts, so it measured whether
    a model could tell two blocks apart. This asks whether it can RANK. The gate
    is not that the correlation is high -- it is that the answer is reported
    beside the two-block one, so a collapse is visible instead of absent."""
    level = float(os.environ.get('DECOMP_LEVEL', 0.6))
    models = ['rf', 'heteroscedastic_gp']
    rows = _measured(models, ['two_block', 'graded'])
    lines = []
    for model in models:
        for condition in ('two_block', 'graded'):
            try:
                at_level, at_zero = _effect(rows, model, condition,
                                            'aleatoric', level)
            except Blocked:
                continue
            rho = at_level['spearman_vs_pattern']
            rho0 = at_zero['spearman_vs_pattern']
            lines.append((model, condition, rho, rho0))
            print(f"      {model:<22} {condition:<10} rho {rho:+.4f}, "
                  f"at zero noise {rho0:+.4f}, "
                  f"effect {rho - (0.0 if np.isnan(rho0) else rho0):+.4f}")
    assert lines, "nothing was measured"
    for model in models:
        got = {c for m, c, _, _ in lines if m == model}
        if got:
            assert got == {'two_block', 'graded'}, (
                f"{model} was measured under {sorted(got)} only; the graded "
                f"result has to be reported BESIDE the two-block one or a "
                f"collapse is invisible")


def check_3_the_variational_layer_is_measured_like_the_gp():
    """It had a noise head that was built and tested and never measured: no
    correlation number, no zero-noise control, no accuracy comparison. All three
    here, and the ordinary variational model beside it."""
    level = float(os.environ.get('DECOMP_LEVEL', 0.6))
    plain = 'dnn_bnn_full_variational'
    het = 'dnn_bnn_full_variational_hetero'
    rows = _measured([plain, het], ['two_block'])

    at_level, at_zero = _effect(rows, het, 'two_block', 'aleatoric', level)
    rho = at_level['spearman_vs_pattern']
    rho0 = at_zero['spearman_vs_pattern']
    print(f"      {het}: rho {rho:+.4f}, at zero noise {rho0:+.4f}, "
          f"effect {rho - (0.0 if np.isnan(rho0) else rho0):+.4f}")

    plain_level = dc.lookup(rows, plain, 'two_block', 'aleatoric', level)
    print(f"      {plain}: data-noise term is "
          f"{'CONSTANT -- correlation undefined' if plain_level['is_constant'] else 'per molecule'}")
    print(f"      R2 out of fold: {het} {at_level['r2_oof']:+.4f}, "
          f"{plain} {plain_level['r2_oof']:+.4f}")

    assert not at_level['is_constant'], (
        "the heteroscedastic variational layer's data-noise term came out "
        "CONSTANT, so the noise head is attached and not being trained -- the "
        "column would mean nothing")
    assert plain_level['is_constant'], (
        "the ORDINARY variational model's data-noise term varies per molecule, "
        "which it cannot: it holds one learned noise. Either the support table "
        "is wrong or the two models are the same model")
    assert not np.isnan(rho), "no correlation number for the noise head"
    assert not np.isnan(rho0), "no zero-noise control for the noise head"
    assert np.isfinite(at_level['r2_oof']) and np.isfinite(plain_level['r2_oof']), (
        "no accuracy comparison against the ordinary variational model")


MEASURED_GATES = [
    ('check_1_even_noise_finds_nothing', check_1_even_noise_finds_nothing),
    ('check_2_graded_noise_is_ranked', check_2_graded_noise_is_ranked),
    ('check_3_the_variational_layer_is_measured_like_the_gp',
     check_3_the_variational_layer_is_measured_like_the_gp),
]


if __name__ == '__main__':
    measured = '--measured' in sys.argv
    print("The three checks on the uncertainty split -- gates\n")

    fast = [(k, v) for k, v in sorted(globals().items())
            if k.startswith('test_') and callable(v)]
    for name, fn in fast:
        check(name, fn)

    if measured:
        print("\n  MEASURED -- real QM9, out of fold on scaffold groups\n")
        for name, fn in MEASURED_GATES:
            check(name, fn)
    else:
        for name, _ in MEASURED_GATES:
            SKIPPED.append(name)
        print(f"\n  {len(MEASURED_GATES)} measured gate(s) NOT RUN "
              f"(they fit real models and take minutes). Pass --measured.")

    total = len(fast) + (len(MEASURED_GATES) if measured else 0)
    passed = total - len(FAILURES) - len(BLOCKED)
    print(f"\n{passed}/{total} passed")
    if BLOCKED:
        print(f"{len(BLOCKED)} gate(s) BLOCKED on this interpreter -- they are "
              f"not passes: " + ", ".join(BLOCKED))
    if SKIPPED:
        print(f"{len(SKIPPED)} gate(s) skipped: " + ", ".join(SKIPPED))
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    sys.exit(0)
