"""THE single source of truth for every default both pipelines share.

WHY THIS FILE EXISTS
--------------------
The project trains the same named models in two places:

    QM9                    qsar_qm_models/models/models.py
    LogD / Caco-2 / hERG   KIRBy/tests/alternative_data_noise_robustness.py

They are meant to be the same models. On 2026-08-25 they were not: the boosting
model was training at three times the step size on one side, the quantile forest
had a third of the trees, the NN-beta network had sixteen times fewer hidden
parameters, and the Gaussian process was capped at 2,000 molecules on one side
and uncapped on the other. All of that came from one pipeline pinning a value
and the other leaving it unset, so it fell through to a library default.

The first attempt at a guard was a table in a document, then a script that
restated both pipelines' parameters as hand-typed literals. Both go stale. This
file does not, because both pipelines READ IT rather than restating it. Change a
number here and it changes in both studies at once.

RULES
-----
1. Literals only. No imports beyond the standard library, so this can be loaded
   from any interpreter, including one that has none of the model packages.
2. If a value is a library default today, PIN IT ANYWAY. That is the whole
   point: a library upgrade must not be able to move a result silently.
3. Anything that differs between the two pipelines ON PURPOSE does NOT belong
   here. This file is the agreed set.
4. Bump SPEC_VERSION on every change, and say what changed in the log at the
   bottom. Every results row records the version and the hash, so a CSV can
   always be traced to the numbers that produced it.
5. `scripts/audit_pipeline_parity.py --strict` builds every model from this file
   with the installed libraries and fails if the effective parameters have moved.
   Run it in the preflight, before anything is submitted.
"""
from __future__ import annotations

import hashlib
import json

SPEC_VERSION = '1.6.0'


# ---------------------------------------------------------------------------
# Tree, kernel and boosting models
#
# Every key is passed straight to the estimator constructor. `random_state` is
# NOT here: QM9 varies it per repetition and the experimental side pins it, and
# that difference is deliberate (RERUN_PLAN.md section 4, decision 3b).
# ---------------------------------------------------------------------------

SKLEARN_DEFAULTS = {
    # sklearn.ensemble.RandomForestRegressor.
    # max_features 0.3 -- MEASURED, not chosen. QM9 used 'sqrt' and the
    # experimental side used the library's 1.0 by omission, so this had to be
    # resolved in one direction. scripts/parity_test_forest.py settled it on
    # 10,000 QM9 molecules, scaffold split, 3 seeds, clean and at half a label
    # spread of noise, with the decision rule fixed before the run:
    #   Morgan       0.3 beats 'sqrt' by +0.040 clean and +0.032 noised, 3/3
    #                seeds each. 45 of 2048 mostly-zero bits rarely lands on
    #                anything informative.
    #   descriptors  0.3 beats 'sqrt' by +0.0055 clean and +0.0060 noised.
    #   1.0          the experimental side's silent default is the WORST option
    #                under noise on descriptors, losing every seed.
    # Cost: about 3.5x 'sqrt', about 3x cheaper than 1.0.
    # min_samples_leaf 5, not 1, changed 2026-08-28 (RERUN_PLAN.md 5.5c).
    # A leaf holding one molecule has no spread inside it, so the aleatoric half
    # of this model's uncertainty is EXACTLY ZERO at leaf 1 and the split cannot
    # be reported at all. Measured on 4,000 real QM9 molecules at the settings in
    # this block, 3 replicates, against 2,000 held out:
    #        leaf   R2 clean   R2 at 1.0 noise   aleatoric share
    #           1     0.6019            0.5339            0.0000
    #           2     0.5990            0.5333            0.1783
    #           5     0.5771            0.5402            0.4586
    #          10     0.5497            0.5236            0.6214
    #          20     0.5131            0.5006            0.7319
    # Five costs 0.025 of clean R2 and GAINS 0.006 at one label spread of
    # injected noise, which is the level the paper reports at. Bigger leaves
    # average over more molecules, which is what damping label noise looks like.
    'rf': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_leaf': 5,
        'min_samples_split': 2,
        'max_features': 0.3,
        'bootstrap': True,
    },
    # quantile_forest.RandomForestQuantileRegressor.
    # 300 trees, not 100: the quantile estimate IS this model's deliverable and
    # 100 trees give a noisy one. Everything else matches the ordinary forest.
    # 0.3 also gives this model much better-calibrated intervals on clean
    # descriptors -- coverage at 1 sd is 0.688 against a nominal 0.683, where
    # 'sqrt' is far too wide at 0.754 and 1.0 too narrow at 0.648.
    # min_samples_leaf 5 for the same reason as the ordinary forest above, and
    # measured separately on this model because it has 300 trees rather than 100
    # and its reported uncertainty comes from quantiles rather than from leaves.
    # It behaves the same: aleatoric share 0.0000 at leaf 1 and 0.4578 at leaf 5,
    # within 0.001 of the ordinary forest at every setting. The accuracy trade is
    # BETTER here -- five costs 0.024 of clean R2 and gains 0.023 at one label
    # spread of noise (0.5051 to 0.5284).
    # NOTE for the coverage line below: it was measured at leaf 1 and has not
    # been re-measured at leaf 5 (RERUN_PLAN.md 5.5c).
    'qrf': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 5,
        'min_samples_split': 2,
        'max_features': 0.3,
        'bootstrap': True,
    },
    # xgboost.XGBRegressor. learning_rate is the one that bit: unset does NOT
    # give the 0.1 the scikit-learn wrapper documents, it gives the booster's
    # own 0.3. Measured cost of 0.3 under injected noise: up to 0.05 R-squared.
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'min_child_weight': 1,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
    },
    # lightgbm.LGBMRegressor
    'lightgbm': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
    },
    # ngboost.NGBRegressor. Dist and Score are class objects, not literals, so
    # they are named here and resolved by the caller; the audit asserts that the
    # names still point where they used to.
    # How many independent NGBoost fits make the ensemble that gives it a
    # model-uncertainty term. A SINGLE fit has none: there is nothing for it to
    # disagree with. The literature's route is an outer ensemble and the
    # epistemic term is the variance of the members' means -- chemprop's
    # EnsemblePredictor computes exactly that
    # (research_archive/f692d614/chemprop_v1_uncertainty_predictor.py).
    #
    # SETTLED 2026-08-30 by the author, on the evidence in RERUN_PLAN.md 5.5i:
    # of seven models tested, only the Gaussian process and NGBoost-over-seeds
    # decompose correctly, so those two are the ones the study reports.
    #
    # 🔴 IT COSTS ONE FIT PER SEED. NGBoost is already the slowest tree model in
    # the roster. Three seeds is three times its wall clock, and nothing else
    # changes.
    'ngboost_ensemble_seeds': 3,

    'ngboost': {
        # EVERY VALUE HERE IS THE ONE NGBOOST'S OWN AUTHORS USED. Duan, Avati,
        # Ding, Thai, Basu, Ng and Schuler, "NGBoost: Natural Gradient Boosting
        # for Probabilistic Prediction", ICML 2020 (PMLR v119), section 4:
        #
        #   "For all experiments, NGBoost was configured with the Normal
        #    distribution, decision tree base learner with a maximum depth of
        #    three levels, and log scoring rule. [...] the rest of the datasets
        #    were fit with a learning rate of 0.01. In general we recommend
        #    small learning rates, subject to computational feasibility. [...]
        #    for all other datasets we use 100%."
        #
        # They match ngboost 0.3.12's own defaults today. They are pinned anyway,
        # under rule 2: a library upgrade must not be able to move a result
        # silently, and the base learner's depth is exactly the kind of thing an
        # upgrade moves.
        'n_estimators': 500,
        'learning_rate': 0.01,
        'natural_gradient': True,
        'dist': 'Normal',        # ngboost.distns.Normal
        'score': 'MLE',          # ngboost.scores.MLE -- MLE IS LogScore, the
                                 # same class object, verified in 0.3.12. So this
                                 # is the paper's "log scoring rule".
        'minibatch_frac': 1.0,   # the paper's 100%
        'col_sample': 1.0,
        # The base learner. Not a literal in the library -- it is a constructed
        # DecisionTreeRegressor -- so it is named here in parts and built by the
        # caller, the same way Dist and Score are.
        'base_max_depth': 3,             # the paper's "maximum depth of three levels"
        'base_criterion': 'friedman_mse',
        # HOW MANY STAGES, and this is the one the paper does NOT fix. Section 4
        # again: a validation set selects the M giving the best log-likelihood,
        # and the model is then refitted at that M. So `n_estimators` above is a
        # CAP, not the answer, and the answer is read off the validation curve.
        #
        # `early_stopping_rounds` is our operationalisation of that, not the
        # paper's number -- the paper evaluates the whole grid of M. 50 stages of
        # patience against a 500 cap is the computational shortcut that makes the
        # selection affordable; NGBoost is the most expensive model on the
        # roster by a wide margin.
        #
        # `use_best_iteration` is NOT optional and is the trap. ngboost stops at
        # (best + patience) and `predict(X)` then uses EVERY stage it fitted, not
        # the best one -- measured: best_val_loss_itr 249, 300 fitted, and the
        # two predictions differ. Fifty stages past the validation optimum, on
        # noisy labels, is the same defect the neural models had when they
        # returned the last epoch instead of the best one.
        'early_stopping_rounds': 50,
        'use_best_iteration': True,
    },
    # sklearn.svm.SVR. RBF on EVERY representation, on both sides -- no per-rep
    # kernel switching, so the model is free of a kernel/representation
    # confound. (paper.tex:197 claims a Tanimoto kernel for the SVM; that is
    # wrong and is a paper fix, not a code one.)
    'svm': {
        'C': 1.0,
        'gamma': 'scale',
        'kernel': 'rbf',
    },
}


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------

GP_DEFAULTS = {
    # gpytorch ExactGP: ConstantMean + ScaleKernel(kernel)
    'kernel': 'rbf',                 # valid on every representation
    'likelihood_noise': 1e-3,
    'outputscale': 1.0,
    # QM9 computed an outputscale and then never applied it to the model, so its
    # ScaleKernel actually started at gpytorch's default of softplus(0) ~ 0.693
    # while the experimental side set 1.0 "to match". Both now apply it.
    'apply_outputscale': True,
    # None = fit on the whole training set. A 2,000-molecule cap made this a
    # DIFFERENT model on the experimental side, not the same model on less data.
    'max_train_n': None,
    # Fallback Adam loop, used only when the botorch fitter refuses a plain
    # gpytorch ExactGP. Recorded in the results as gp_fit_method, never silent.
    'fallback_adam_lr': 0.1,
    'fallback_adam_iters': 100,
    # Fit the Gaussian process on ONE thread, and only the Gaussian process.
    #
    # The two gradient-boosting libraries each load their own copy of the
    # threading machinery and the neural-network library loads a third. Once all
    # three are in one process -- which is every job, since both pipelines import
    # all of them at module level -- a threaded matrix operation inside the
    # Gaussian process makes the copies collide and the operating system kills
    # the process outright. No Python error, no traceback, nothing in the log:
    # the task simply stops, which is indistinguishable from a task that
    # finished quietly. Measured on this machine: it crashes both with no thread
    # count set (how the QM9 jobs run) and with both set to 4 (how the
    # experimental jobs run).
    #
    # Limiting threads for the WHOLE job also cures it, but taxes every tree fit
    # across the entire grid. This limits only the Gaussian-process fit and puts
    # the setting back afterwards, so nothing else pays.
    #
    # SET FALSE 2026-08-28, on the condition this comment named. The condition
    # is met and was measured on an ARC compute node, not argued from a laptop:
    #
    #   threading runtimes (what a job would load)      one, libomp.so
    #   /proc/self/maps after importing every backend   one runtime mapped
    #   LightGBM fits, no thread count / both at 4      OK / OK
    #   GP fits after lightgbm+xgboost, none / both 4   OK / OK
    #
    # `python scripts/check_environment.py --deep --validation` exited 0 in
    # /data/stat-cadd/scat9264/conda_envs/env_test (RERUN_PLAN.md 2.8i). The
    # workaround cost 11% on every Gaussian-process fit and was the net under a
    # failure that no longer exists.
    #
    # Put it back to True if that check ever reports more than one runtime
    # again: this setting is the net, not the fix.
    'single_thread_fit': False,
    # Start the RBF lengthscale at the median distance between training
    # molecules instead of gpytorch's default of ~0.69.
    #
    # Measured 2026-08-26: typical distances between molecules run from 17 (PDV)
    # to 1,100 (the learned embeddings). At a lengthscale of 0.69 every molecule
    # is effectively infinitely far from every other, the kernel matrix is the
    # identity, the marginal likelihood is flat and there is no gradient to
    # follow. The fit returns a constant and scores whatever that constant
    # happens to score -- which is what produced R2 = -0.0158 for MHG-GNN and
    # +0.0087 for mol2vec in results/gp_kernel_harvest/qm9/. Started from the
    # median distance, those same features reach 0.68 to 0.76.
    'init_lengthscale_from_data': True,
    # Molecules sampled to estimate that median. The median is stable well below
    # the full training set and the pairwise distance matrix is O(n^2).
    'lengthscale_probe_n': 500,
    # A fit whose predictions vary by less than this fraction of the training
    # label spread has collapsed to a constant. It is recorded, not silently
    # scored (RERUN_PLAN.md 0.6, failure mode 9).
    'collapse_fraction': 0.05,
}


# ---------------------------------------------------------------------------
# Neural networks
#
# NN-alpha ('dnn') is the DNN base used by BNN-Full and VBLL-Full.
# NN-beta  ('mlp') is the MLP base used by MLP-BNN-Full and MLP-VBLL-Full.
# ---------------------------------------------------------------------------

NEURAL_DEFAULTS = {
    'dnn': {
        'hidden_sizes': [128, 64],
        'dropout_rate': 0.2,
        'activation': 'relu',
    },
    'mlp': {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'dropout_rate': 0.2,
        'activation': 'relu',
    },
    'training': {
        'optimizer': 'adam',
        'lr': 1e-3,
        'weight_decay': 0.0,
        'batch_size': 32,          # QM9's value; the experimental side used 64
        'epochs': 100,
        # Early stopping. Both sides now roll back to the best epoch: QM9 used
        # to count patience and return the LAST epoch, up to 20 past the
        # validation optimum, which under injected noise is 20 epochs spent
        # memorising corrupted labels -- a procedural reason for QM9's neural
        # degradation curves to look steeper. Author's decision, 2026-08-26.
        'patience': 10,
        'restore_best_weights': True,
        # The improvement test, stated once so the two cannot diverge again.
        # QM9 required a 0.01 absolute drop in a SUMMED validation loss, so its
        # threshold silently scaled with the number of validation batches; the
        # experimental side used strict improvement on a mean. Now: mean loss,
        # strict improvement, no tolerance.
        'val_loss_reduction': 'mean',
        'improvement_tolerance': 0.0,
        # Stochastic forward passes for every Bayesian and variational model.
        'mc_passes': 100,
        # Targets are standardised for neural models only, fitted on TRAINING
        # labels alone.
        'standardise_targets': True,
    },
}


# ---------------------------------------------------------------------------
# Bayesian and variational layers
# ---------------------------------------------------------------------------

BAYESIAN_DEFAULTS = {
    # torchbnn BayesLinear, via torchhk transform_model
    'bnn_prior_mu': 0.0,
    'bnn_prior_sigma': 0.1,
    # How heavily the KL term counts against the fit.
    #
    # Settled 2026-08-27 by the author. Until then there was NO KL term anywhere:
    # BNN-alpha and BNN-beta were trained with plain MSE, so nothing pulled the
    # variational posterior toward the prior and prior_sigma was only an
    # initialisation. torchbnn samples weights in train AND eval mode, so the
    # posterior width received MSE gradients, which drive it toward zero. What
    # was reported as "epistemic uncertainty" was whatever residual weight noise
    # the fit happened to leave. The VBLL variants DID carry a KL term, so the
    # two families were not comparable (RERUN_PLAN.md 2.12).
    #
    # 'elbo' means the ELBO scaling, 1 / n_train: the objective is
    # sum_i NLL_i + KL, and the criterion here is the MEAN over the batch, so
    # dividing the KL by the number of training molecules puts the two on the
    # same footing. A float instead of 'elbo' is used verbatim.
    'bnn_kl_weight': 'elbo',
    # VBLLLayer (Harrison 2024) -- models.py:1131
    'vbll_prior_mu': 0.0,
    'vbll_prior_sigma': 1.0,
    'vbll_init_log_sigma': -3.0,
    'vbll_init_log_noise_var': -2.0,
}


# ---------------------------------------------------------------------------
# Feature standardisation -- WHICH representations get scaled
#
# This is not a model parameter, so no parameter diff could ever have caught it,
# and the two pipelines had silently diverged:
#
#   QM9           standardised only the continuous representations and fed
#                 fingerprints to the models as raw 0/1 bits
#   experimental  standardised EVERY representation unconditionally, in both its
#                 tree path and its neural path
#
# For trees that is irrelevant -- they are scale-invariant. For the support
# vector machine and the Gaussian process, both of which measure distances with
# a radial kernel, it changes the model completely.
#
# MEASURED TWICE, and the size differs a great deal between the two, so quote
# whichever matches the dataset you are talking about.
#
#   QM9, 2,000 molecules, scaffold split, Morgan fingerprint, RBF SVM:
#       raw 0/1 bits    R2 = +0.815, +0.809, +0.832   (seeds 0,1,2)
#       standardised    R2 = +0.320, +0.124, +0.182
#       cost of standardising   -0.50, -0.69, -0.65
#
#   hERG, all 1,415 molecules, 5-fold scaffold split, ECFP4, same SVM:
#       raw 0/1 bits    R2 = +0.5335
#       standardised    R2 = +0.4570
#       cost of standardising   -0.077, and every one of the five folds agrees
#
# The DIRECTION is the same on both and on every fold. The SIZE is not: about
# 0.6 on QM9 against about 0.08 on hERG. Do not carry the QM9 figure over to the
# experimental datasets -- QM9's gap is far more predictable to begin with, so
# there is more to lose, and its molecules are small enough that the fingerprints
# are much sparser than drug-like ones.
#
# Standardising a sparse binary fingerprint gives the rare bits huge magnitudes
# and lets them dominate every distance, which is why it hurts kernel methods.
# The QM9 behaviour is the correct one and the experimental side aligns to it.
#
# To reverse this decision, change this one set.
# ---------------------------------------------------------------------------

# THE RULE IS KEYED ON THE FEATURES, NOT ON THE NAME.
#
# A name-keyed list cannot work here, and trying one was a mistake worth
# recording: `pdv` on QM9 is the BINARISED descriptor vector, while `PDV` on the
# experimental side is the CONTINUOUS one. Same name, opposite answer. That is
# the same trap as the fingerprint being the wrong function -- names in this
# project do not identify features.
#
# So the decision is made from the matrix itself:
#     binary (every value 0 or 1)        ->  never standardised
#     sparse counts (small non-negative
#       integers, mostly zero)           ->  never standardised
#     anything else                      ->  standardised, train-only
#
# This is self-correcting. When the binarised descriptor vector is replaced by
# the continuous one, and when the substructure counts stop being flattened to
# presence bits, the rule follows the data without anyone remembering to edit a
# list.
STANDARDISE_BINARY_FEATURES = False
STANDARDISE_SPARSE_COUNTS = False

# A matrix denser than this is not the sparse-fingerprint case the exemption is
# about, whatever its values look like. Measured: the substructure counts run at
# 1.3% density on QM9 and 4.7% on hERG, while the descriptor vector -- the only
# other candidate, since half its columns hold non-negative integers -- runs at
# 39% and is not all-integer anyway. The threshold sits an order of magnitude
# above the fingerprints and well below the descriptors, so nothing is close to
# it in either direction.
SPARSE_COUNT_MAX_DENSITY = 0.25


def _sample(X):
    import numpy as np
    X = np.asarray(X)
    return X[:min(len(X), 512)]


def is_binary_matrix(X):
    """True if every value is 0 or 1. Sampled -- a full scan of a large
    fingerprint matrix is wasteful and the first rows settle it."""
    import numpy as np
    sample = _sample(X)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    return bool(np.isin(finite, (0, 1)).all())


def is_sparse_count_matrix(X):
    """True for a substructure-count fingerprint: small non-negative integers,
    almost all of them zero, and at least one above 1.

    The last clause matters -- without it this would also claim every binary
    fingerprint, and the two exemptions could then disagree. A matrix of 0s and
    1s is the binary case and is handled there.
    """
    import numpy as np
    sample = _sample(X)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0 or finite.size != sample.size:
        return False
    if not bool((finite >= 0).all()):
        return False
    if not bool(np.all(np.equal(np.mod(finite, 1), 0))):
        return False
    if float((sample != 0).mean()) > SPARSE_COUNT_MAX_DENSITY:
        return False
    return bool((finite > 1).any())


def should_standardise(X, rep_name=None):
    """Should these features be standardised before the model sees them?

    MEASURED on 2,000 QM9 molecules, scaffold split, Morgan fingerprint, support
    vector machine with a radial kernel:
        raw 0/1 bits   R2 = +0.815, +0.809, +0.832   (seeds 0, 1, 2)
        standardised   R2 = +0.320, +0.124, +0.182
    Standardising a sparse binary fingerprint gives its rare bits huge
    magnitudes and lets them dominate every distance, which wrecks anything
    kernel-based. Trees are unaffected either way.

    SUBSTRUCTURE COUNTS: also left raw, MEASURED 2026-08-27 by
    scripts/parity_test_count_scaling.py -- Sort & Slice counts, scaffold split,
    five seeds, the same radial-kernel support vector machine, against a decision
    rule fixed before the run. Standardised counts minus raw counts:
        QM9   -0.070, -0.061, -0.107, -0.039, -0.089   mean -0.073, spread 0.026
        hERG  +0.009, -0.022, -0.061, +0.015, +0.084   mean +0.005, spread 0.053
    QM9 loses on all five seeds by nearly three times the seed-to-seed spread;
    hERG is a coin flip. So the harm is the same harm as the binary case -- rare
    features blown up until they dominate the distances -- and it does not go
    away when the presence bit becomes a count.

    This is why the exemption is written as a rule about sparse features rather
    than about binary ones: the storage fix that restores real counts would
    otherwise have silently started standardising the fingerprint, and cost QM9
    0.07 R2 on every kernel model, with nothing recording that it had happened.
    """
    if is_binary_matrix(X) and not STANDARDISE_BINARY_FEATURES:
        return False
    if is_sparse_count_matrix(X) and not STANDARDISE_SPARSE_COUNTS:
        return False
    return True


# ---------------------------------------------------------------------------
# Uncertainty reporting
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# THE REPORTING LEVEL — the ONE noise level a table quotes accuracy at.
#
# This is the only place it may live. It is here, in the spec both pipelines
# import, because between 2026-08-20 and 2026-08-28 it was stated in more than
# thirty places across two documents, two scripts and paper.tex, on two different
# scales, with four different values live at once. The author's words on
# 2026-08-27: "there's clearly multiple sources of information on it and you need
# to find all of them and rectify it. I'm really sick of this coming up."
#
# SCALE. A reporting level is a fraction of that fold's CLEAN TRAINING LABEL
# SPREAD, and nothing else. It is a point on the ladder in NOISE_DESIGN.md 6.4.
# It is NOT in log units. The old results files are in log units and a number
# read off one of them is a different amount of noise -- that mistake has already
# been made once, in this very decision.
#
# NOT REPORTING LEVELS, and each has been mistaken for one:
#   0.15 / 0.35 / 0.48 / 0.54 / 0.68  published assay error, in log units. These
#                                     are evidence for where the LADDER's points
#                                     sit (NOISE_DESIGN.md 4, 4c).
#   0.6                               an accuracy cutoff in the figure script for
#                                     discarding fits that failed. Not a level.
#   0.0 0.2 0.3 0.5 0.75 1.0 1.5      the ladder. Every one of these RUNS. The
#                                     reporting level is which ONE gets quoted.
#
# NOT SETTLED. The author gave numbers on 2026-08-27 and withdrew them the same
# evening -- "I don't think I did settle them" -- because the tables he read them
# off were printed on two different scales in one message. Nothing may report at a
# level until this dict holds a number for that dataset. Read it through
# reporting_level(), which raises rather than guessing.
REPORTING_LEVELS = {
    # SET 2026-08-28 by the author: "RIGHT okay 1.0/1.0/0.2".
    'qm9': 1.0,
    'logd': 1.0,
    # Caco-2 reports lower than the rest ON PURPOSE. Its clean R2 is only about
    # 0.5, so it has less to lose than the others: at 1.0 the quantile forest is
    # down to 0.256, and 0.75 gives the SAME two rank flips out of four with every
    # model 0.04-0.11 higher. That is a fact about the assay, not a fudge, and it
    # needs one sentence in the caption.
    'caco2': 0.75,
    # hERG's label spread is measured, 0.9143 over 1,415 molecules, so one unit of
    # published assay error is 0.60 of it and twice that is 1.21 -- 1.0 sits just
    # under the design's own "run to about twice real error" rule. It was tested:
    # 7 models x 4 representations survive in results/paper_figures_v2/, but only
    # as auc_norm and a clean baseline. The per-level R2 is not on this machine, so
    # 1.0 is chosen on the anchor rather than on a rank-flip table like the others.
    'herg': 1.0,
}
REPORTING_LEVEL_SCALE = 'fraction_of_clean_training_label_spread'


def reporting_level(dataset):
    """The level a table quotes for this dataset. Raises while it is unset.

    Raising is the point. Every previous default silently became the answer: the
    figure script's `sigma_value=0.3` is in paper.tex today as "R2 at sigma = 0.3",
    on a scale that no longer exists, and nobody chose it.
    """
    key = str(dataset).strip().lower().replace('-', '').replace('_', '')
    alias = {'qm9': 'qm9', 'logd': 'logd', 'caco2': 'caco2',
             'hergki': 'herg', 'herg': 'herg', 'openadmetlogd': 'logd',
             'openadmetcaco2': 'caco2'}
    key = alias.get(key, key)
    if key not in REPORTING_LEVELS:
        raise KeyError(
            f"no reporting level is defined for {dataset!r}. Known: "
            f"{sorted(REPORTING_LEVELS)}")
    value = REPORTING_LEVELS[key]
    if value is None:
        raise ValueError(
            f"the reporting level for {key} is NOT SET, so no table may quote one.\n"
            f"It is a fraction of the clean training label spread, a point on the "
            f"ladder in NOISE_DESIGN.md 6.4.\n"
            f"Set it in models/model_defaults.py REPORTING_LEVELS -- that is the only "
            f"place it may live -- and record the author's decision in "
            f"NOISE_DESIGN.md 6.4 at the same time.")
    return value


UNCERTAINTY_DEFAULTS = {
    # Which column the analysis treats as "the model's uncertainty".
    #
    # 'raw' -- MEASURED, by scripts/parity_test_calibration.py, 36 fits.
    # Calibration does work in the narrow sense: it moved test coverage closer
    # to nominal in all twelve model x noise-level cells. But the multiplier is
    # REFITTED at each noise level, so calibrated coverage is nominal at each
    # level by construction, and the span of coverage ACROSS noise levels
    # collapses:
    #     NGBoost   raw 0.546 -> 0.876 -> 0.978 (span 0.432)
    #               cal 0.637 -> 0.658 -> 0.686 (span 0.049)
    #     QRF       raw 0.799 -> 0.929 -> 0.956 (span 0.157)
    #               cal 0.740 -> 0.733 -> 0.737 (span 0.006)
    # The raw column says these models' intervals become far too wide as noise
    # rises. The calibrated column says nothing, because it was fitted not to.
    # Anything the paper claims about how uncertainty RESPONDS to noise has to
    # be read off raw, or it is circular.
    #
    # Both rank-based uncertainty questions are unaffected either way: the
    # maximum difference in Spearman(uncertainty, |error|) across all 36 fits
    # was exactly 0.0.
    #
    # Honest caveat that belongs in the paper: raw coverage IS poor. The
    # Gaussian process sits at 0.86 against a nominal 0.68, the quantile forest
    # reaches 0.96. That is the finding, not a failure to report.
    'primary_column': 'raw',
    # Temperature calibration: one multiplier fitted by Gaussian negative log
    # likelihood on a carve-out of validation (scripts/utils.py:323).
    'calibration_bounds': [0.1, 10.0],
    'calibration_fraction_of_val': 0.5,
    'coverage_levels': [1, 2],
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def gp_fit_threads():
    """Context manager limiting the thread count for a Gaussian-process fit.

    torch is imported lazily so this module stays loadable in an interpreter
    that has none of the model packages -- the parity audit relies on that.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        if not GP_DEFAULTS.get('single_thread_fit', True):
            yield
            return
        try:
            import torch
        except Exception:
            yield
            return
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(previous)
    return _ctx()


def gp_fit_collapsed(y_pred, y_reference):
    """Did a Gaussian-process fit return one number for every molecule it scored?

    ONE definition, read by both pipelines. QM9 had it in `models/models.py` and
    the laboratory runner wrote the same arithmetic out again inline; two copies
    of one rule is failure mode 10 of RERUN_PLAN.md 0.6, and this one decides
    whether a row reaches the results, so drift in it is not cosmetic.

    A process that cannot use its features returns its prior: the mean
    everywhere, and the prior variance everywhere. That still produces a number
    and the number reads as a weak representation rather than as a fit that
    answered nothing -- which is how the two learned embeddings were written off
    (RERUN_PLAN.md 2.8f).

    It also happens on a HEALTHY fit, one fold at a time. The inner split is
    scaffold-grouped, so a held-out fold is whole scaffold families the fit set
    does not contain; when the fitted length scale is short against the distance
    to them, every kernel value is nil and the process returns its prior across
    that entire fold. That is the process answering honestly -- it knows nothing
    about those molecules -- and its model-uncertainty term is then one number
    for the fold, which cannot answer a per-molecule question (RERUN_PLAN.md
    3.1d, 5.5i).

    `y_pred` are the predictions for the molecules being scored; `y_reference`
    are the labels the model was FITTED on. Returns (collapsed, spread,
    threshold), all three so a caller can report the numbers rather than a bare
    flag.
    """
    import numpy as _np
    spread = float(_np.std(_np.asarray(y_pred, dtype=float)))
    reference = float(_np.std(_np.asarray(y_reference, dtype=float)))
    threshold = GP_DEFAULTS['collapse_fraction'] * reference
    return spread < threshold, spread, threshold


def spec_dict():
    """Everything above, as one plain dictionary."""
    return {
        'spec_version': SPEC_VERSION,
        'sklearn': SKLEARN_DEFAULTS,
        'gp': GP_DEFAULTS,
        'neural': NEURAL_DEFAULTS,
        'bayesian': BAYESIAN_DEFAULTS,
        'uncertainty': UNCERTAINTY_DEFAULTS,
        'standardise_binary_features': STANDARDISE_BINARY_FEATURES,
    }


def spec_hash():
    """Short content hash. Written into every results row, so a CSV can be
    traced to the exact parameters that produced it."""
    blob = json.dumps(spec_dict(), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def bnn_kl_weight(n_train):
    """The multiplier on the KL term for a torchbnn network, from the spec."""
    setting = BAYESIAN_DEFAULTS['bnn_kl_weight']
    if setting == 'elbo':
        if not n_train:
            raise ValueError('bnn_kl_weight: n_train is required for the ELBO '
                             'scaling and was %r' % (n_train,))
        return 1.0 / float(n_train)
    return float(setting)


def sklearn_params(model_key, **overrides):
    """Constructor keywords for one model. Raises on an unknown name rather than
    quietly returning an empty dict."""
    if model_key not in SKLEARN_DEFAULTS:
        raise KeyError(f'no shared defaults for {model_key!r}; '
                       f'known models: {sorted(SKLEARN_DEFAULTS)}')
    params = dict(SKLEARN_DEFAULTS[model_key])
    params.update(overrides)
    return params


def provenance_columns():
    """The columns every results row must carry (RERUN_PLAN.md section 5.2)."""
    return {'spec_version': SPEC_VERSION, 'spec_hash': spec_hash()}


if __name__ == '__main__':
    print(json.dumps(spec_dict(), indent=2))
    print(f'\nspec_version {SPEC_VERSION}   spec_hash {spec_hash()}')


# ---------------------------------------------------------------------------
# CHANGE LOG
# ---------------------------------------------------------------------------
# 1.5.0  2026-08-28  NGBoost pinned to the settings its own paper used, read
#                    from Duan et al., ICML 2020 section 4: base learner depth 3,
#                    friedman_mse, minibatch fraction 1.0, column sample 1.0. All
#                    four were ngboost 0.3.12 defaults and match the paper, so no
#                    fitted model moves -- this closes the door on a library
#                    upgrade moving them. Added with them: the paper's stage
#                    selection, which reads M off a validation curve rather than
#                    fixing it, and the flag that makes prediction use the best
#                    iteration rather than every fitted one.
# 1.3.0  2026-08-27  BNN-alpha and BNN-beta gained the KL term they never had.
#                    Decided by the author after being shown that no KL, ELBO or
#                    BKLLoss existed anywhere in either pipeline while the VBLL
#                    variants carried one. INVALIDATES every BNN-alpha and
#                    BNN-beta number on both pipelines.
# 1.2.0  2026-08-27  The scaling exemption widened from binary features to
#                    SPARSE COUNTS, decided by
#                    scripts/parity_test_count_scaling.py against a rule fixed
#                    before the run. No model parameter changes; what changes is
#                    what the substructure fingerprint looks like to a kernel
#                    once its counts are no longer flattened to presence bits.
#                    Without this, that storage fix would have cost QM9 0.073 R2
#                    on the radial-kernel support vector machine.
# 1.1.0  2026-08-26  Forest max_features 'sqrt' -> 0.3, on BOTH pipelines,
#                    decided by scripts/parity_test_forest.py against a rule
#                    fixed before the run. INVALIDATES every existing forest and
#                    quantile-forest result, including QM9's -- everything is
#                    being re-run, so the cost is zero, but it is a change to the
#                    main study and not only an alignment.
# 1.0.0  2026-08-26  Created (Chat E, cross-pipeline parity). Values transcribed
#                    from the QM9 pipeline's `params_source == 'default'`
#                    branches, which is the path every job in
#                    slurm_scripts_qm9_rerun/ takes -- none of them passes
#                    --use_best_params. Carries the six alignments already
#                    applied on the experimental side (NN-beta width, XGBoost
#                    parameters, quantile-forest trees, stochastic passes,
#                    Gaussian-process cap, hard failure on an unbuildable
#                    model), plus batch size, the early-stopping definition, and
#                    the Gaussian process outputscale.
