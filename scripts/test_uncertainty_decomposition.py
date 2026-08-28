"""Gates on the shared aleatoric/epistemic split.

Every check here was confirmed to FAIL when the fix it guards is removed. Run:

    python scripts/test_uncertainty_decomposition.py

Nothing in this file needs the cluster, a GPU or a trained model.
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uncertainty_decomposition import (  # noqa: E402
    CONSTANT, NONE, PER_MOLECULE, SUPPORT, DecompositionError,
    assert_matches_support, decompose_forest, decompose_gp,
    decompose_hetero_gp, decompose_sampling, decompose_seed_ensemble,
    support, to_raw_variance, variance_to_std)

FAILURES = []
BLOCKED = []


class EnvironmentBlocked(Exception):
    """This interpreter cannot run the gate. Never reported as a pass."""


def check(name, fn):
    try:
        fn()
        print(f"  ok      {name}")
    except EnvironmentBlocked as e:
        print(f"  BLOCKED {name}: {e}")
        BLOCKED.append(name)
    except AssertionError as e:
        print(f"  FAIL    {name}: {e}")
        FAILURES.append(name)
    except Exception as e:  # a test that errors is a failing test
        print(f"  ERROR   {name}: {type(e).__name__}: {e}")
        FAILURES.append(name)


def raises(exc, fn, fragment=None):
    try:
        fn()
    except exc as e:
        if fragment is not None:
            assert fragment in str(e), f"wrong message: {e}"
        return
    raise AssertionError(f"expected {exc.__name__}, nothing was raised")


# ---------------------------------------------------------------------------
# The sampling split matches the reference implementations on disk
# ---------------------------------------------------------------------------

def test_sampling_matches_ryu():
    """`ryu_train_cep.py:128-133` computes exactly this, and so does
    `scalia_predict.py:107-118`. Reproduced here on a fixed array so that a
    change to our arithmetic shows up as a failure rather than as a new
    number in a figure."""
    rng = np.random.default_rng(0)
    means = rng.normal(size=(50, 7))
    variances = np.abs(rng.normal(size=(50, 7))) + 0.1

    alea, epis, total = decompose_sampling(means, variances)

    # Ryu: ale_unc = np.mean(P_logvar, axis=0); epi_unc = np.var(P_mean, axis=0)
    assert np.allclose(alea, variances.mean(axis=0))
    assert np.allclose(epis, means.var(axis=0))
    # Ryu: tot_unc = ale_unc + epi_unc -- VARIANCES ADD.
    assert np.allclose(total, alea + epis)


def test_total_is_a_sum_of_variances_not_of_deviations():
    """The 41% overstatement in RERUN_PLAN.md 5.5. If someone re-writes the
    total as a sum of standard deviations this fires."""
    alea, epis, total = decompose_sampling(
        np.array([[0.0, 0.0], [2.0, 2.0]]),
        np.full((2, 2), 1.0))
    wrong = (np.sqrt(alea) + np.sqrt(epis)) ** 2
    assert np.allclose(total, alea + epis)
    assert not np.allclose(total, wrong), (
        "the total equals the square of summed standard deviations, which is "
        "only correct if the two sources are perfectly correlated")


def test_no_variance_head_means_no_aleatoric_term():
    """A plain Bayesian network predicts no observation noise. Kendall & Gal's
    epistemic-only model. It must return None, never zero -- a zero would be
    averaged into a table as a real measurement."""
    alea, epis, total = decompose_sampling(np.random.default_rng(1).normal(
        size=(20, 5)))
    assert alea is None
    assert np.allclose(total, epis)


def test_sampling_refuses_averaged_passes():
    raises(DecompositionError,
           lambda: decompose_sampling(np.zeros(10)),
           'n_passes')


def test_sampling_refuses_a_single_pass():
    raises(DecompositionError,
           lambda: decompose_sampling(np.zeros((1, 10))),
           'exactly zero')


# ---------------------------------------------------------------------------
# The forest split
# ---------------------------------------------------------------------------

def test_forest_split_obeys_the_law_of_total_variance():
    """Both terms per molecule, from trees already fitted, no retraining."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 6))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.3, size=300)

    forest = RandomForestRegressor(n_estimators=40, min_samples_leaf=5,
                                   random_state=0).fit(x, y)
    mean, alea, epis = decompose_forest(forest, x, y, x[:50])

    assert mean.shape == (50,) and alea.shape == (50,) and epis.shape == (50,)
    assert np.all(alea >= 0) and np.all(epis >= 0)
    # Both must actually vary -- a constant here would be a broken walk of the
    # trees that still produces plausible numbers.
    assert alea.std() > 0, "aleatoric term is constant across molecules"
    assert epis.std() > 0, "epistemic term is constant across molecules"
    # The forest's own prediction is the mean of the per-tree leaf means.
    assert np.allclose(mean, forest.predict(x[:50]), atol=1e-8)


def test_forest_refuses_the_degenerate_split_at_the_project_default():
    """min_samples_leaf=1 makes the forest split DEGENERATE, and it must refuse.

    A leaf holding one training molecule has no within-leaf spread, so the
    aleatoric term is identically zero and the epistemic term absorbs the whole
    predictive variance. `models/model_defaults.py:70` and `:84` both set
    min_samples_leaf=1, so this is the setting the re-run would have used.

    Measured on 4,000 real QM9 molecules -- the HOMO-LUMO gap, nine descriptors
    from `data/QM9/raw/gdb9.sdf.csv`: the aleatoric share is 0.000 at leaf 1 and
    0.373 at leaf 5. Writing a column of zeros would read as "this model sees no
    label noise", so the split refuses instead of returning them.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 5))
    y = x[:, 0] * 3.0 + rng.normal(scale=0.5, size=400)

    forest = RandomForestRegressor(
        n_estimators=40, min_samples_leaf=1, random_state=0).fit(x, y)
    raises(DecompositionError,
           lambda: decompose_forest(forest, x, y, x),
           'min_samples_leaf')


def test_forest_aleatoric_rises_with_label_noise():
    """Corrupting labels raises the forest's aleatoric term. Both terms rise.

    THE SHARE IS NOT A RELIABLE SEPARATOR FOR A BAGGED MODEL, and this test does
    not assert that it is. Measured two ways:

      real QM9, 4,000 molecules, leaf 5, ten replicates per level -- the
      aleatoric share rises monotonically, 0.3744 at zero noise to 0.4060 at one
      label spread;
      a synthetic linear set, same leaf size -- the share FALLS monotonically,
      0.5650 to 0.5208.

    The reason is that bootstrap resampling carries label noise into the trees
    twice over: into the spread inside each leaf, which is the aleatoric term,
    and into where the trees choose to split at all, which moves the epistemic
    term. How those two race is a property of the dataset. So for the forests
    the honest reportable statement is that both terms rise, not that one rises
    and the other holds still (RERUN_PLAN.md 5.5d).
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(4)
    x = rng.normal(size=(600, 5))
    signal = x[:, 0] * 3.0

    def split(labels):
        f = RandomForestRegressor(n_estimators=60, min_samples_leaf=5,
                                  random_state=0).fit(x, labels)
        _, a, e = decompose_forest(f, x, labels, x)
        return a.mean(), e.mean()

    alea_c, epis_c = split(signal + rng.normal(scale=0.3, size=600))
    alea_n, epis_n = split(signal + rng.normal(scale=1.5, size=600))

    assert alea_c > 0 and epis_c > 0, "a term was zero on the cleaner labels"
    assert alea_n > alea_c * 3, (
        f"the aleatoric term barely moved when the labels were corrupted: "
        f"{alea_c:.4g} -> {alea_n:.4g}")
    assert epis_n > epis_c, (
        "the epistemic term did not rise; if this ever holds, the note above "
        "about bagging is wrong and the share becomes usable")


def test_forest_split_works_on_the_quantile_forest():
    """The quantile forest is queued on BOTH pipelines, so the walk has to work
    on it and not only on the plain forest. It subclasses the sklearn forest;
    that is checked here rather than assumed.

    `env.yml:81,97` pins scikit-learn 1.6.1 against quantile-forest 1.4.1, which
    are compatible. An interpreter carrying a different scikit-learn cannot fit
    one at all -- that is reported as a blocked environment, never as a pass.
    """
    from quantile_forest import RandomForestQuantileRegressor

    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 4))
    y = x[:, 1] + rng.normal(scale=0.4, size=300)

    try:
        forest = RandomForestQuantileRegressor(
            n_estimators=30, min_samples_leaf=5, random_state=0).fit(x, y)
    except (ValueError, TypeError) as e:
        import sklearn
        raise EnvironmentBlocked(
            f"ENVIRONMENT: quantile-forest cannot fit against scikit-learn "
            f"{sklearn.__version__} ({e}). env.yml pins 1.6.1 with "
            f"quantile-forest 1.4.1. This gate is unverified on this "
            f"interpreter and must be re-run where the pinned pair is "
            f"installed.")

    _, alea, epis = decompose_forest(forest, x, y, x[:30])
    assert alea.std() > 0 and epis.std() > 0
    assert np.all(alea >= 0) and np.all(epis >= 0)


def test_forest_refuses_mismatched_labels():
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(6)
    x = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    forest = RandomForestRegressor(n_estimators=5, random_state=0).fit(x, y)
    raises(DecompositionError,
           lambda: decompose_forest(forest, x, y[:40], x),
           'training labels')


# ---------------------------------------------------------------------------
# The Gaussian processes
# ---------------------------------------------------------------------------

def test_gp_aleatoric_is_one_number_and_says_so():
    alea, epis, total = decompose_gp(np.array([0.1, 0.5, 2.0]), 0.25)
    assert np.allclose(alea, 0.25)
    assert np.allclose(epis, [0.1, 0.5, 2.0])
    assert np.allclose(total, [0.35, 0.75, 2.25])
    assert SUPPORT['gauche_rbf'][0] == CONSTANT
    assert SUPPORT['GP'][0] == CONSTANT


def test_hetero_gp_gives_both_per_molecule():
    alea, epis, total = decompose_hetero_gp([0.1, 0.5], [0.2, 0.9])
    assert np.allclose(total, [0.3, 1.4])
    assert SUPPORT['heteroscedastic_gp'] == (PER_MOLECULE, PER_MOLECULE)


def test_gp_rejects_a_negative_noise():
    raises(DecompositionError, lambda: decompose_gp([1.0], -0.5))


# ---------------------------------------------------------------------------
# The seed ensemble
# ---------------------------------------------------------------------------

def test_seed_ensemble_needs_more_than_one_seed():
    raises(DecompositionError,
           lambda: decompose_seed_ensemble(np.zeros((1, 4)), np.ones((1, 4))),
           'exactly zero')


def test_seed_ensemble_splits_correctly():
    means = np.array([[1.0, 2.0], [3.0, 2.0]])
    variances = np.array([[0.4, 0.6], [0.6, 0.4]])
    alea, epis, total = decompose_seed_ensemble(means, variances)
    assert np.allclose(alea, [0.5, 0.5])
    assert np.allclose(epis, [1.0, 0.0])
    assert np.allclose(total, [1.5, 0.5])


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------

def test_variances_scale_by_the_square_of_the_spread():
    """The trap in RERUN_PLAN.md 5.5, part 3."""
    v = np.array([1.0, 4.0])
    assert np.allclose(to_raw_variance(v, 3.0), [9.0, 36.0])
    assert not np.allclose(to_raw_variance(v, 3.0), v * 3.0), (
        "variances were scaled by the spread rather than its square")


def test_scale_refuses_a_guessed_spread():
    raises(DecompositionError, lambda: to_raw_variance([1.0], 0.0), 'recorded')
    raises(DecompositionError, lambda: to_raw_variance([1.0], float('nan')))


def test_std_conversion_is_the_only_route_to_a_deviation():
    assert np.allclose(variance_to_std(np.array([4.0, 9.0])), [2.0, 3.0])
    assert variance_to_std(None) is None


# ---------------------------------------------------------------------------
# The guard that ties a model's output to what is claimed for it
# ---------------------------------------------------------------------------

def test_guard_catches_a_constant_sold_as_per_molecule():
    raises(DecompositionError,
           lambda: assert_matches_support(
               'QRF', np.full(10, 0.3), np.linspace(0.1, 0.5, 10)),
           'cannot correlate')


def test_guard_catches_a_varying_term_declared_constant():
    raises(DecompositionError,
           lambda: assert_matches_support(
               'GP', np.linspace(0.1, 0.5, 10), np.linspace(0.1, 0.5, 10)),
           'one number per fit')


def test_guard_catches_a_missing_component():
    raises(DecompositionError,
           lambda: assert_matches_support('QRF', None, np.linspace(0, 1, 5)),
           'nothing was produced')


def test_guard_accepts_the_honest_cases():
    assert assert_matches_support('GP', np.full(5, 0.2), np.linspace(1, 2, 5))
    assert assert_matches_support('QRF', np.linspace(0.1, 0.2, 5),
                                  np.linspace(1, 2, 5))
    assert assert_matches_support('svm', None, None)


def test_unknown_model_is_refused_by_name():
    raises(DecompositionError, lambda: support('some_new_model'), 'SUPPORT')


def test_support_covers_every_queued_model():
    """A model can be added to a job generator and reach the writer without an
    entry here. That would write an unlabelled column, so it fails instead."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    missing = []

    qm9 = os.path.join(root, 'slurm_scripts_qm9_rerun', 'generate_scripts.py')
    lab = os.path.join(root, 'slurm_scripts_uncertainty_rerun',
                       'generate_scripts.py')
    for path in (qm9, lab):
        if not os.path.isfile(path):
            continue
        import ast
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, 'id', '') == 'MODELS'
                            for t in node.targets)
                    and isinstance(node.value, ast.Dict)):
                for k in node.value.keys:
                    name = getattr(k, 'value', None)
                    if isinstance(name, str) and name not in SUPPORT:
                        missing.append(f"{os.path.basename(path)}: {name}")

    assert not missing, (
        "queued models with no SUPPORT entry: " + ", ".join(missing))


def test_no_model_claims_a_split_it_cannot_have():
    """A model with no epistemic axis must not claim one, and vice versa."""
    for name, (alea, epis) in SUPPORT.items():
        assert alea in (PER_MOLECULE, CONSTANT, NONE), name
        assert epis in (PER_MOLECULE, CONSTANT, NONE), name
        if alea == NONE and epis == NONE:
            continue
        assert not (alea == CONSTANT and epis == CONSTANT), (
            f"{name}: both terms constant, so the split says nothing about any "
            f"molecule and should not be written at all")


if __name__ == '__main__':
    print("Shared aleatoric/epistemic split -- gates\n")
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        check(name, fn)

    passed = len(tests) - len(FAILURES) - len(BLOCKED)
    print(f"\n{passed}/{len(tests)} passed")
    if BLOCKED:
        print(f"{len(BLOCKED)} gate(s) UNVERIFIED on this interpreter -- they "
              f"are not passes: " + ", ".join(BLOCKED))
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        sys.exit(1)
    sys.exit(0)
