"""One definition of the aleatoric/epistemic split, for both pipelines.

QM9 trains through `models/models.py`; the three laboratory datasets train
through `KIRBy/tests/alternative_data_noise_robustness.py`. Before this module
the two had no shared definition of the split -- QM9 had four different ones and
the laboratory runner had none at all -- which is failure mode 10 in
`RERUN_PLAN.md` 0.6: two implementations of one specification drifting apart.
Both sides import this file and neither computes the arithmetic itself.

EVERYTHING IN AND OUT OF THIS MODULE IS A VARIANCE
==================================================
Not a standard deviation. The repository has mixed the two repeatedly, and the
mixture is not visible in the output: a stacked figure that adds two standard
deviations overstates the total by up to 41% because it is only correct when the
two sources are perfectly correlated, and they are independent by construction
(`RERUN_PLAN.md` 5.5, the standardisation trap). Variances add; standard
deviations do not. Convert once, at the point of writing, with
`variance_to_std`.

THE DEFINITION, AND WHERE IT COMES FROM
=======================================
Three papers held in `research_archive/f692d614/` perform this split and all
three perform it identically:

    Kendall & Gal, NeurIPS 2017, eq. 9        -- kg.pdf
    Ryu, Kwon & Kim, Chem. Sci. 2019, eq. 9   -- ryu2.pdf
    Scalia et al., J. Chem. Inf. Model. 2020, eq. 4 -- scalia.pdf

    aleatoric = mean over stochastic passes of the variance the network
                predicted for THAT MOLECULE
    epistemic = variance over stochastic passes of the predicted means
    total     = aleatoric + epistemic

Their working implementations are on disk beside them and were read line by line
before this was written: `ryu_train_cep.py:128-133` and
`scalia_predict.py:107-118`. Scalia's inverse transform is
`inverse_transform_variance`, which is the same scaling rule used here.

NO PAPER ON DISK FITS A SINGLE SCALAR NOISE VALUE AND CALLS IT ALEATORIC.
A Gaussian process with one likelihood noise term, and a variational last layer
with one learned observation noise, each produce a number that is identical for
every molecule. That is still the correct aleatoric term for those models, but
it cannot correlate with anything measured per molecule -- its correlation with
per-molecule injected noise is zero however good the model is. Such a term is
reported here with `per_molecule=False` and the writer refuses to let it be
described as anything else.

THE FOREST SPLIT
================
`decompose_forest` is the law of total variance over the mixture of per-tree
leaf distributions:

    Var[y|x] = E_t[Var(y | leaf_t(x))] + Var_t[E(y | leaf_t(x))]
               \_____ aleatoric _____/   \_____ epistemic _____/

It is computed after the fact from trees that are already fitted, so it costs no
retraining. Adapted from `research_archive/f692d614/forest_ae.py`. Its source
paper is NOT on disk -- flagged rather than hidden.
"""

import numpy as np

# How each component of a model's uncertainty behaves. Read by the writer, which
# fails the run when a model's actual output disagrees with what is claimed here.
#
#   'per_molecule' -- the value changes from one molecule to the next
#   'constant'     -- one value per fitted model, copied onto every row
#   'none'         -- the model cannot produce this component at all
#
# Names are the ones the job generators use, so a roster change and a support
# change cannot drift apart. QM9 names come from
# `slurm_scripts_qm9_rerun/generate_scripts.py`; the laboratory names from
# `slurm_scripts_uncertainty_rerun/generate_scripts.py`.
PER_MOLECULE = 'per_molecule'
CONSTANT = 'constant'
NONE = 'none'

# CORRECTED 2026-08-28, when the module was wired into the two runners. Three
# families claimed a term the queued configuration does not produce, and
# `assert_matches_support` would have stopped every one of their runs. The table
# now says what the pipelines ACTUALLY write; where that is less than the plan
# wants, the gap is recorded in `RERUN_PLAN.md` 5.5g rather than asserted away.
#
#   NGBoost      -- one fit, so there is no ensemble to disagree with itself and
#                   no model-uncertainty axis at all. `decompose_seed_ensemble`
#                   gives it one, at the price of as many fits as seeds; nobody
#                   has agreed to pay that, so the queued model has NONE.
#   BNN-alpha,
#   BNN-beta     -- a plain Bayesian network predicts a mean and nothing else,
#                   so it has no observation-noise term to report. Kendall &
#                   Gal's epistemic-only model.
#   rf, qrf      -- both terms per molecule, from `decompose_forest`, and only
#                   because `min_samples_leaf` is now 5 (RERUN_PLAN.md 5.5c).
SUPPORT = {
    # QM9 roster
    'rf':                        (PER_MOLECULE, PER_MOLECULE),
    'qrf':                       (PER_MOLECULE, PER_MOLECULE),
    # NGBoost has NO model-uncertainty term, and the seed ensemble does not give
    # it one. MEASURED on 3,000 real QM9 molecules, PDV, out of fold on scaffold
    # groups, three seeds, at four settings of the two parameters the seed
    # actually reaches (RERUN_PLAN.md 2.30):
    #
    #   setting          aleatoric   epistemic   how many times smaller
    #   as shipped         x9.1        x1.9              3,581
    #   half the rows      x8.8        x2.7                150
    #   80% of rows        x9.1        x2.2                243
    #   80% of columns     x9.1        x1.6                506
    #
    # `random_state` feeds only `minibatch_frac` and `col_sample`, and the spec
    # sets both to 1.0, so the members are the same fit and their spread is
    # floating-point dust. Subsampling makes it 24x larger and it is still two to
    # three orders of magnitude below the aleatoric term -- and it RISES with the
    # noise level, which is the behaviour that disqualifies a model-uncertainty
    # term rather than the behaviour that qualifies one.
    #
    # So the term is marked ABSENT, which is what it is: a single distributional
    # fit predicts a variance per molecule and has nothing to disagree with. Its
    # aleatoric term is real and is the second best measured (x9.1). This entry
    # said (PER_MOLECULE, PER_MOLECULE) for one day, and while it did,
    # `assert_matches_support` refused every NGBoost uncertainty row on both
    # pipelines -- it would have stopped every NGBoost task on the cluster.
    'ngboost':                   (PER_MOLECULE, NONE),
    'dnn':                       (NONE, NONE),
    'mlp':                       (NONE, NONE),
    'dnn_bnn_full':              (NONE, PER_MOLECULE),
    'mlp_bnn_full':              (NONE, PER_MOLECULE),
    'dnn_bnn_full_variational':  (CONSTANT, PER_MOLECULE),
    'mlp_bnn_full_variational':  (CONSTANT, PER_MOLECULE),
    # The variational layer with `heteroscedastic=True`, which predicts the
    # observation noise FROM THE INPUT (RERUN_PLAN.md 5.5f). Same base networks,
    # different aleatoric term, so it is a different row name.
    'dnn_bnn_full_variational_hetero': (PER_MOLECULE, PER_MOLECULE),
    'mlp_bnn_full_variational_hetero': (PER_MOLECULE, PER_MOLECULE),
    'gauche':                    (CONSTANT, PER_MOLECULE),
    'gauche_rbf':                (CONSTANT, PER_MOLECULE),
    'heteroscedastic_gp':        (PER_MOLECULE, PER_MOLECULE),
    # NGBoost under several seeds: the outer ensemble is the only
    # model-uncertainty axis a single-fit distributional model has (Duan et al.,
    # via ALEA_EPIS_LITERATURE.md). One fit per seed.
    'ngboost_ensemble':          (PER_MOLECULE, PER_MOLECULE),
    # A Bayesian network with a variance output head -- Kendall & Gal eq. 6, the
    # case the literature actually holds up. Both halves per molecule, from two
    # different mechanisms: the head predicts the noise, the weight samples give
    # the model term.
    'dnn_bnn_full_mve':          (PER_MOLECULE, PER_MOLECULE),
    'mlp_bnn_full_mve':          (PER_MOLECULE, PER_MOLECULE),
    # The names the heteroscedastic Gaussian process actually WRITES, which
    # carry the kernel. The roster label above is what the job generator emits.
    'het_gp_rbf':                (PER_MOLECULE, PER_MOLECULE),
    'het_gp_tanimoto':           (PER_MOLECULE, PER_MOLECULE),
    'het_gp_matern':             (PER_MOLECULE, PER_MOLECULE),
    'xgboost':                   (NONE, NONE),
    'lgb':                       (NONE, NONE),
    'svm':                       (NONE, NONE),
    # Laboratory roster
    'RF':                        (PER_MOLECULE, PER_MOLECULE),
    'QRF':                       (PER_MOLECULE, PER_MOLECULE),
    # Same as `ngboost` above: no epistemic term, measured (RERUN_PLAN.md 2.30).
    'NGBoost':                   (PER_MOLECULE, NONE),
    'GP':                        (CONSTANT, PER_MOLECULE),
    'GP-Tanimoto':               (CONSTANT, PER_MOLECULE),
    # The laboratory runner decorates a Gaussian process variant with a suffix,
    # so its heteroscedastic form is 'GP-Hetero' and 'GP-Tanimoto-Hetero'.
    'GP-Hetero':                 (PER_MOLECULE, PER_MOLECULE),
    'GP-Tanimoto-Hetero':        (PER_MOLECULE, PER_MOLECULE),
    'BNN-Full':                  (NONE, PER_MOLECULE),
    'MLP-BNN-Full':              (NONE, PER_MOLECULE),
    'VBLL-Full':                 (CONSTANT, PER_MOLECULE),
    'MLP-VBLL-Full':             (CONSTANT, PER_MOLECULE),
    'VBLL-Full-Hetero':          (PER_MOLECULE, PER_MOLECULE),
    'MLP-VBLL-Full-Hetero':      (PER_MOLECULE, PER_MOLECULE),
    'DNN':                       (NONE, NONE),
    'MLP':                       (NONE, NONE),
    'XGBoost':                   (NONE, NONE),
    'LightGBM':                  (NONE, NONE),
    'SVM':                       (NONE, NONE),
    # ---------------------------------------------------------------------
    # NOT IN EITHER ROSTER. Present so that a model nobody queued still writes
    # a labelled column instead of stopping the run at the writer, and so that
    # what it produces is on the record. Every one of these is reached only by
    # naming it on the command line.
    # ---------------------------------------------------------------------
    'dnn_bnn_last':              (NONE, PER_MOLECULE),
    'mlp_bnn_last':              (NONE, PER_MOLECULE),
    'dnn_bnn_variational':       (CONSTANT, PER_MOLECULE),
    'mlp_bnn_variational':       (CONSTANT, PER_MOLECULE),
    'flexible_dnn':              (NONE, PER_MOLECULE),
    'graph_gp':                  (CONSTANT, PER_MOLECULE),
    'gnn':                       (NONE, PER_MOLECULE),
    # The two evidential models write both terms per molecule from the
    # Normal-Inverse-Gamma parameters.
    'evidential_kernel':         (PER_MOLECULE, PER_MOLECULE),
    'conformal_hetero':          (PER_MOLECULE, PER_MOLECULE),
    # It writes a single total from the neural-tangent-kernel variance and
    # performs no split.
    'ntk_gnn':                   (NONE, NONE),
    # Refused by name at the top of process_and_train.py (RERUN_PLAN.md 2.22),
    # and it never computed a split.
    'conformal':                 (NONE, NONE),
    # The mitigation experiments write a total and no split.
    'noise_mitigation_gauche':   (NONE, NONE),
}

# The name a row is WRITTEN under is not always the name the job generator uses.
# The neural trainers write 'bnn_full' and the file name carries the base
# network, so the roster's 'dnn_bnn_full' never appears in the model column.
# Mapping them here keeps one entry per model rather than two that can disagree.
SUPPORT_ALIASES = {
    'bnn_full':                     'dnn_bnn_full',
    'bnn_last':                     'dnn_bnn_last',
    'bnn_variational':              'dnn_bnn_variational',
    'bnn_full_variational':         'dnn_bnn_full_variational',
    'bnn_full_variational_hetero':  'dnn_bnn_full_variational_hetero',
}

# Two losses give a network a second output that predicts the observation noise
# for each molecule, so a model whose aleatoric term is absent or constant under
# the default loss has a PER-MOLECULE one under either of them. No queued job
# passes `--loss` today (RERUN_PLAN.md 5.5a point 1), but a run that did would
# otherwise be stopped by the guard for producing BETTER data than the table
# claims.
LOSSES_WITH_A_VARIANCE_HEAD = ('heteroscedastic', 'evidential')


class DecompositionError(RuntimeError):
    """A split that cannot be trusted. Never caught and turned into a blank row."""


def support(model_name, loss_name=None):
    """(aleatoric behaviour, epistemic behaviour) for a queued model name.

    `loss_name` is the loss the model was actually FITTED with. Two of them give
    the network a second output that predicts the observation noise per
    molecule, which turns an absent or constant aleatoric term into a
    per-molecule one -- see LOSSES_WITH_A_VARIANCE_HEAD. Passing it keeps the
    guard live in that configuration instead of stopping a run for producing a
    better term than the table claims.
    """
    try:
        alea, epis = SUPPORT[SUPPORT_ALIASES.get(model_name, model_name)]
    except KeyError:
        raise DecompositionError(
            f"{model_name!r} has no entry in SUPPORT, so nothing knows whether "
            f"its aleatoric term varies per molecule or is one number per fit. "
            f"Add it rather than letting the row be written unlabelled.")
    if loss_name in LOSSES_WITH_A_VARIANCE_HEAD and epis != NONE:
        alea = PER_MOLECULE
    return alea, epis


def _as_1d(a, name):
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        raise DecompositionError(f"{name} is empty")
    return a


def _check_nonneg(v, name):
    if np.any(v < -1e-9):
        raise DecompositionError(
            f"{name} contains a negative variance (min {float(np.min(v)):.6g}). "
            f"A variance cannot be negative; this is an arithmetic error "
            f"upstream, not a small number to clip away.")
    return np.maximum(v, 0.0)


# ---------------------------------------------------------------------------
# The four ways a model in either roster produces a split
# ---------------------------------------------------------------------------

def decompose_sampling(mean_passes, var_passes=None):
    """Stochastic passes over a network. Kendall & Gal eq. 9.

    `mean_passes` is (n_passes, n_molecules) of predicted means.
    `var_passes` is (n_passes, n_molecules) of the variance the network
    predicted for each molecule on each pass, or None when the network has no
    such output -- in which case there is NO aleatoric term, and the total is
    the epistemic term alone. That is Kendall & Gal's epistemic-only model, and
    it is what every Bayesian network in this project produced before the second
    output was added.

    Returns (aleatoric_var, epistemic_var, total_var); aleatoric is None when
    the network predicts no variance.
    """
    m = np.asarray(mean_passes, dtype=float)
    if m.ndim != 2:
        raise DecompositionError(
            f"mean_passes must be (n_passes, n_molecules); got shape {m.shape}. "
            f"A one-dimensional array here means the passes were averaged away "
            f"before the split, which destroys the epistemic term.")
    if m.shape[0] < 2:
        raise DecompositionError(
            f"the epistemic term is the variance across passes and there "
            f"{'is' if m.shape[0] == 1 else 'are'} {m.shape[0]} pass(es). It "
            f"would be exactly zero, which reads as a confident model rather "
            f"than as a missing measurement.")

    epistemic = _check_nonneg(m.var(axis=0), 'epistemic')
    if var_passes is None:
        return None, epistemic, epistemic

    v = np.asarray(var_passes, dtype=float)
    if v.shape != m.shape:
        raise DecompositionError(
            f"var_passes {v.shape} does not match mean_passes {m.shape}")
    aleatoric = _check_nonneg(v.mean(axis=0), 'aleatoric')
    return aleatoric, epistemic, aleatoric + epistemic


def decompose_forest(forest, x_train, y_train, x_score):
    """Law of total variance over a fitted forest's per-tree leaf distributions.

    Costs no retraining: the trees are already fitted and this only walks them.
    Works for `sklearn.ensemble.RandomForestRegressor` and for
    `quantile_forest.RandomForestQuantileRegressor`, which subclasses it.

    Returns (mean, aleatoric_var, epistemic_var), each per molecule.
    """
    from sklearn.ensemble._forest import (_generate_sample_indices,
                                          _get_n_samples_bootstrap)

    x_train = np.asarray(x_train)
    y_train = _as_1d(y_train, 'y_train')
    n_train = x_train.shape[0]
    if y_train.shape[0] != n_train:
        raise DecompositionError(
            f"{n_train} training rows but {y_train.shape[0]} training labels. "
            f"The split reproduces each tree's in-bag rows by index, so a "
            f"mismatch here silently pairs a molecule with another's label.")

    train_leaves = forest.apply(x_train)
    score_leaves = forest.apply(x_score)

    n_boot = (_get_n_samples_bootstrap(n_train, forest.max_samples)
              if forest.bootstrap else None)

    n_score = score_leaves.shape[0]
    tree_means = np.empty((n_score, forest.n_estimators))
    tree_vars = np.empty_like(tree_means)

    for i, est in enumerate(forest.estimators_):
        # The rows THIS tree was grown on, with repeats, reproduced exactly.
        # Using all training rows instead would attribute to a tree's leaves
        # labels the tree never saw, which inflates the aleatoric term.
        if forest.bootstrap:
            idx = _generate_sample_indices(est.random_state, n_train, n_boot)
        else:
            idx = np.arange(n_train)

        leaf_of = train_leaves[idx, i]
        y_of = y_train[idx]

        n_nodes = est.tree_.node_count
        cnt = np.bincount(leaf_of, minlength=n_nodes).astype(float)
        s1 = np.bincount(leaf_of, weights=y_of, minlength=n_nodes)
        s2 = np.bincount(leaf_of, weights=y_of ** 2, minlength=n_nodes)

        safe = np.where(cnt > 0, cnt, 1.0)
        leaf_mean = s1 / safe
        leaf_var = np.maximum(s2 / safe - leaf_mean ** 2, 0.0)

        tree_means[:, i] = leaf_mean[score_leaves[:, i]]
        tree_vars[:, i] = leaf_var[score_leaves[:, i]]

    mean = tree_means.mean(axis=1)
    aleatoric = _check_nonneg(tree_vars.mean(axis=1), 'aleatoric')
    epistemic = _check_nonneg(tree_means.var(axis=1), 'epistemic')

    # A leaf holding one training molecule has no within-leaf spread, so the
    # whole predictive variance lands in the epistemic term and the aleatoric
    # term is identically zero. That is not a property of the data and it is not
    # a finding -- it is `min_samples_leaf`. Both forests in
    # `models/model_defaults.py` were set to 1 when this was written, so it fired
    # by default; they were settled at 5 on 2026-08-28 (RERUN_PLAN.md 5.5c), so it
    # should not fire on a queued run any more. It stays because a hand-run fit,
    # or a tuned setting that puts the leaf back to 1, would otherwise write a
    # column of zeros that reads as "this model sees no label noise".
    if float(np.max(aleatoric)) <= 1e-12:
        leaf = getattr(forest, 'min_samples_leaf', None)
        raise DecompositionError(
            f"every leaf holds one molecule, so the aleatoric term is exactly "
            f"zero and the split says nothing (min_samples_leaf={leaf!r}). "
            f"Measured on real QM9: leaf 1 gives an aleatoric share of 0.000, "
            f"leaf 5 gives 0.373 and costs 0.019 of clean R2 while GAINING "
            f"0.018 at one label spread of injected noise. Set a larger leaf or "
            f"record this model as having no aleatoric term -- do not write "
            f"zeros.")

    return mean, aleatoric, epistemic


def decompose_gp(latent_variance, likelihood_noise):
    """Gaussian process. The epistemic term varies per molecule; the aleatoric
    term is ONE NUMBER for the whole fit, because a homoscedastic likelihood has
    exactly one noise parameter.

    `latent_variance` must be the variance of the LATENT function -- what
    gpytorch calls `model(x).variance`, not `likelihood(model(x)).variance`.
    The second already includes the noise, so passing it here counts the noise
    twice. And it must be a VARIANCE: passing a standard deviation returns the
    fourth root of the quantity wanted, which is what
    `models/models.py:4148` did (`RERUN_PLAN.md` 5.5).
    """
    epistemic = _check_nonneg(_as_1d(latent_variance, 'latent_variance'),
                              'epistemic')
    noise = float(likelihood_noise)
    if noise < 0:
        raise DecompositionError(f"likelihood noise is negative: {noise}")
    aleatoric = np.full_like(epistemic, noise)
    return aleatoric, epistemic, aleatoric + epistemic


def decompose_hetero_gp(latent_variance, predicted_noise_variance):
    """Gaussian process whose observation noise is predicted per molecule by a
    second model. Both terms vary per molecule, so this is the only Gaussian
    process in either roster that can answer a per-molecule question."""
    epistemic = _check_nonneg(_as_1d(latent_variance, 'latent_variance'),
                              'epistemic')
    aleatoric = _check_nonneg(
        _as_1d(predicted_noise_variance, 'predicted_noise_variance'),
        'aleatoric')
    if aleatoric.shape != epistemic.shape:
        raise DecompositionError(
            f"noise {aleatoric.shape} and latent variance {epistemic.shape} "
            f"cover different numbers of molecules")
    return aleatoric, epistemic, aleatoric + epistemic


def decompose_single_distribution(predicted_variance):
    """A model that predicts a distribution per molecule from ONE fit -- NGBoost.

    There is no second fit to disagree with, so there is no model-uncertainty
    axis and the epistemic term is absent rather than zero. Zero would read as a
    model that is certain about its own parameters; None reads as a model that
    was never asked.

    `predicted_variance` is a VARIANCE. NGBoost's `pred_dist().scale` is a
    standard deviation and must be squared before it reaches here -- squaring it
    at the call site is where the repository's earlier
    `decompose_uncertainty_distributional` went wrong, because it returned the
    scale untouched and every consumer added it to a variance.

    `decompose_seed_ensemble` gives the same model an epistemic term, at the
    price of one fit per seed. Nothing in either roster pays that today.

    Returns (aleatoric_var, None, total_var).
    """
    aleatoric = _check_nonneg(_as_1d(predicted_variance, 'predicted_variance'),
                              'aleatoric')
    return aleatoric, None, aleatoric


def decompose_seed_ensemble(mean_by_seed, var_by_seed):
    """A model with a per-molecule predicted variance but no internal spread --
    NGBoost -- fitted under several seeds.

    The epistemic term is the disagreement between the seeds' means, which is
    the only model-uncertainty axis such a model has. It requires as many fits
    as seeds, so it is not free.

    Both arguments are (n_seeds, n_molecules); `var_by_seed` holds the predicted
    variance, NOT the predicted scale. NGBoost's `pred_dist().scale` is a
    standard deviation and must be squared before it reaches here.
    """
    m = np.asarray(mean_by_seed, dtype=float)
    v = np.asarray(var_by_seed, dtype=float)
    if m.ndim != 2 or v.shape != m.shape:
        raise DecompositionError(
            f"mean_by_seed {m.shape} and var_by_seed {v.shape} must both be "
            f"(n_seeds, n_molecules)")
    if m.shape[0] < 2:
        raise DecompositionError(
            f"the epistemic term is the spread across seeds and there "
            f"{'is' if m.shape[0] == 1 else 'are'} {m.shape[0]} seed(s), so it "
            f"would be exactly zero. Fit more than one seed or record the "
            f"aleatoric term alone and mark the epistemic term absent.")
    aleatoric = _check_nonneg(v.mean(axis=0), 'aleatoric')
    epistemic = _check_nonneg(m.var(axis=0), 'epistemic')
    return aleatoric, epistemic, aleatoric + epistemic


# ---------------------------------------------------------------------------
# Scale, and the guard that goes with it
# ---------------------------------------------------------------------------

def to_raw_variance(variance, label_spread):
    """Undo label standardisation on a VARIANCE.

    Both pipelines fit on a standardised label, so every component above is in
    standardised units. Converting back multiplies by the SQUARE of the spread.
    Multiplying by the spread itself -- which is right for a standard deviation
    and wrong for a variance -- is the error `scalia_predict.py` avoids by
    keeping a separate `inverse_transform_variance`.

    The offset does not appear: shifting every label by a constant does not
    change a variance.
    """
    if variance is None:
        return None
    s = float(label_spread)
    if not np.isfinite(s) or s <= 0:
        raise DecompositionError(
            f"label spread {s!r} cannot scale a variance. The spread must be "
            f"recorded at injection and carried on the row; guessing it back "
            f"by fitting a line is what this replaces.")
    return np.asarray(variance, dtype=float) * (s ** 2)


def variance_to_std(variance):
    """The single place a variance becomes a standard deviation."""
    if variance is None:
        return None
    return np.sqrt(_check_nonneg(np.asarray(variance, dtype=float), 'variance'))


def assert_matches_support(model_name, aleatoric, epistemic, n_molecules=None,
                           loss_name=None, blocks=None):
    """Fail the run when what a model produced disagrees with what SUPPORT says.

    This is the guard, not a comment. It catches the failure that has cost this
    project most: a term declared per molecule that is in fact one number copied
    onto every row, whose correlation with anything per-molecule is zero however
    good the model is.

    `blocks` names which FIT each value came from, one entry per molecule. It
    matters for the out-of-fold rows: those come from several fits, so a term
    that is one number PER FIT -- a homoscedastic likelihood noise, a variational
    layer's single observation noise -- takes a different value in each fold and
    is not constant down the column. It is still `constant`, and it still cannot
    answer a per-molecule question: what varies is the fold, not the molecule.
    Without this the guard would stop a correct out-of-fold pass; with it, the
    check becomes constant WITHIN each fit, which is the real claim.

    BOTH checks are made within a fit. The per-molecule one was made across the
    whole column until 2026-08-30, and out of fold that column is several fits
    stacked: a Gaussian process that collapsed in all five folds writes five
    different constants, the column has a spread, and the check passed on a term
    that varies by fold and not by molecule. The one-fold smoke test was the only
    configuration that caught it (RERUN_PLAN.md 5.5i).
    """
    alea_kind, epis_kind = support(model_name, loss_name)

    for kind, values, name in ((alea_kind, aleatoric, 'aleatoric'),
                               (epis_kind, epistemic, 'epistemic')):
        if kind == NONE:
            if values is not None:
                raise DecompositionError(
                    f"{model_name}: SUPPORT says it has no {name} term, but "
                    f"one was produced. Either the model changed or the table "
                    f"is stale; do not write the row until they agree.")
            continue

        if values is None:
            raise DecompositionError(
                f"{model_name}: SUPPORT says its {name} term is {kind!r}, but "
                f"nothing was produced. A blank column here is indistinguishable "
                f"from a model that cannot do it.")

        v = _as_1d(values, name)
        if n_molecules is not None and v.size != n_molecules:
            raise DecompositionError(
                f"{model_name}: {v.size} {name} values for {n_molecules} "
                f"molecules")

        parts = _fits(v, blocks, model_name, name)

        if kind == CONSTANT:
            within = 0.0
            for _label, part in parts:
                within = max(within, float(np.nanmax(part) - np.nanmin(part)))
            if within > 1e-12:
                raise DecompositionError(
                    f"{model_name}: SUPPORT calls its {name} term one number per "
                    f"fit, but it varies by {within:.6g} within a single fit. "
                    f"Update SUPPORT -- a term that really does vary is worth "
                    f"more, not less.")

        if kind == PER_MOLECULE:
            # WITHIN EACH FIT, for the same reason the constant check above is.
            # This used to look at the whole column at once, and out of fold the
            # column is several fits stacked: a process that collapsed in every
            # one of five folds writes five different constants, the column has a
            # spread, and the check passed on a term that answers nothing per
            # molecule. Only the one-fold configuration -- the cheap smoke test --
            # caught it. Found 2026-08-30 (RERUN_PLAN.md 5.5i).
            flat = [(label, part) for label, part in parts if part.size > 1
                    and float(np.nanmax(part) - np.nanmin(part)) <= 1e-12]
            if flat:
                label, part = flat[0]
                value = float(part[np.isfinite(part)][0])
                where = ('' if blocks is None
                         else f" in fit {label} ({len(flat)} of {len(parts)} "
                              f"fits, {part.size} molecules)")
                raise DecompositionError(
                    f"{model_name}: SUPPORT calls its {name} term per molecule, but "
                    f"every molecule{where} got {value:.6g}. A constant cannot "
                    f"correlate with per-molecule noise, so this would be reported "
                    f"as a null result that is an artefact of the model, not a "
                    f"finding.")
    return True


def _fits(v, blocks, model_name, name):
    """Split a column into the FITS it came from: [(label, values), ...].

    Out-of-fold rows come from several fits, so both checks above have to be
    made inside one fit rather than across the stack. Blocks with nothing finite
    in them -- the molecules a fold that was not run left as NaN -- are dropped,
    because a block of blanks says nothing either way.
    """
    if blocks is None:
        return [(None, v)] if np.isfinite(v).any() else []
    b = np.asarray(blocks).ravel()
    if b.size != v.size:
        raise DecompositionError(
            f"{model_name}: {b.size} block labels for {v.size} {name} values")
    out = []
    for label in np.unique(b):
        part = v[b == label]
        if part.size and np.isfinite(part).any():
            out.append((label, part))
    return out
