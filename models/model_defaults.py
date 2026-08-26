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

SPEC_VERSION = '1.1.0'


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
    'rf': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_leaf': 1,
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
    'qrf': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 1,
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
    'ngboost': {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'natural_gradient': True,
        'dist': 'Normal',        # ngboost.distns.Normal
        'score': 'MLE',          # ngboost.scores.MLE
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
    # VBLLLayer (Harrison 2024) -- models.py:1131
    'vbll_prior_mu': 0.0,
    'vbll_prior_sigma': 1.0,
    'vbll_init_log_sigma': -3.0,
    'vbll_init_log_noise_var': -2.0,
}


# ---------------------------------------------------------------------------
# Uncertainty reporting
# ---------------------------------------------------------------------------

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

def spec_dict():
    """Everything above, as one plain dictionary."""
    return {
        'spec_version': SPEC_VERSION,
        'sklearn': SKLEARN_DEFAULTS,
        'gp': GP_DEFAULTS,
        'neural': NEURAL_DEFAULTS,
        'bayesian': BAYESIAN_DEFAULTS,
        'uncertainty': UNCERTAINTY_DEFAULTS,
    }


def spec_hash():
    """Short content hash. Written into every results row, so a CSV can be
    traced to the exact parameters that produced it."""
    blob = json.dumps(spec_dict(), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


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
