"""Which model-representation pairings exist, and what each one is called.

WHY THIS FILE EXISTS
--------------------
Three places name the same models and the same representations, in three
different spellings:

    QM9 job generator     slurm_scripts_qm9_rerun/generate_scripts.py
    QM9 tuned-params file results/master_tuned_hyperparameters.json
    experimental pipeline KIRBy/tests/alternative_data_noise_robustness.py

Nothing here retypes the first one. `MODELS`, `ALL_REPS` and `FP_REPS` are
imported out of the generator, so the pairing list is whatever the generator
would submit and cannot drift from it.

The other two spellings ARE written down here, because neither is available as
an importable constant: the tuned-params key is the literal string each builder
passes to `load_best_hyperparameters`, buried in a function body, and the
experimental names are built inside `run_dataset_experiment`. Both are checked
against their sources by scripts/test_tuning_rosters.py, which fails if a name
here stops existing there.

THE TUNED KEY IS NOT THE MODEL LABEL
------------------------------------
`load_best_hyperparameters(model_type, rep)` is called with a hard-coded string,
and for SEVEN of the fourteen models that string is not the model's own name:

    qrf                                     ->  'rf'    (models.py:1631 -- the
                                                         literal is 'rf' for both
                                                         branches of the one
                                                         function that builds the
                                                         plain forest AND the
                                                         quantile forest)
    dnn_bnn_full, dnn_bnn_full_variational  ->  'dnn'   (models.py:2450)
    mlp_bnn_full, mlp_bnn_full_variational  ->  'mlp'   (models.py:3200, via
                                                         model_type, which is
                                                         'mlp' for every variant)
    gauche_rbf, gauche                      ->  'gauche' (models.py:2129)

So only FOUR of the fourteen models -- svm, xgboost, lgb and ngboost -- can be
handed a tuned setting that reaches them and nothing else. The quantile forest
reads the plain forest's entry; four Bayesian networks read whatever their
deterministic base reads; and the RBF and Tanimoto Gaussian processes share one
entry per representation, which on ecfp4 and sns is the SAME entry and carries
`kernel_name`, so a tuned entry there would force one kernel on both and delete
the RBF-vs-Tanimoto comparison the roster exists to make.

`TUNED_KEY` records that collapse rather than hiding it; `collapsed_models()`
names the collisions; `writable_keys()` is what may safely be written to the two
JSON files as things stand. Widening it is one changed literal per call site in
models.py, and that is a decision for the author, not a silent edit here.
"""
from __future__ import annotations

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

GENERATOR_PATH = os.path.join(
    _ROOT, 'slurm_scripts_qm9_rerun', 'generate_scripts.py')
KIRBY_PIPELINE_PATH = os.path.expanduser(
    '~/repos/KIRBy/tests/alternative_data_noise_robustness.py')


def _generator():
    """The QM9 job generator, loaded as a module. It is not on a package path
    and its directory has no __init__, so it is loaded from its file."""
    spec = importlib.util.spec_from_file_location(
        'qm9_job_generator', GENERATOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GEN = _generator()

# Read, never retyped. generate_scripts.py:205 and :92.
MODELS = _GEN.MODELS
ALL_REPS = list(_GEN.ALL_REPS)
FP_REPS = list(_GEN.FP_REPS)


def qm9_pairings(include_excluded=False):
    """Every (model label, representation) the QM9 grid would submit.

    The fifth field of each MODELS entry is that model's representation list, so
    the Tanimoto Gaussian process comes out on the binary fingerprints alone
    without this file knowing why.
    """
    pool = dict(MODELS)
    if include_excluded:
        pool.update(_GEN.EXCLUDED_MODELS)
    return [(label, rep)
            for label, entry in pool.items()
            for rep in entry[4]]


# ---------------------------------------------------------------------------
# The key each model's builder passes to load_best_hyperparameters
# ---------------------------------------------------------------------------
# Verified by scripts/test_tuning_rosters.py against the literal call sites.
TUNED_KEY = {
    'rf':                       'rf',
    'qrf':                      'rf',
    'svm':                      'svm',
    'xgboost':                  'xgboost',
    'lgb':                      'lgb',
    'ngboost':                  'ngboost',
    'gauche_rbf':               'gauche',
    'gauche':                   'gauche',
    'dnn':                      'dnn',
    'dnn_bnn_full':             'dnn',
    'dnn_bnn_full_variational': 'dnn',
    'mlp':                      'mlp',
    'mlp_bnn_full':             'mlp',
    'mlp_bnn_full_variational': 'mlp',
}

# Model labels whose tuned entry is shared with another model. Writing one of
# these into the master file changes a model the tuning run never scored.
def collapsed_models():
    """Model labels that cannot be addressed on their own, and what they collide
    with. Returns {label: [other labels sharing its key]}."""
    by_key = {}
    for label, key in TUNED_KEY.items():
        by_key.setdefault(key, []).append(label)
    return {label: sorted(set(by_key[key]) - {label})
            for label, key in TUNED_KEY.items()
            if len(by_key[key]) > 1}


def writable_keys():
    """Tuned-file keys that address exactly one model, so a value written under
    them reaches the model it was measured on and nothing else."""
    collapsed = collapsed_models()
    return sorted({TUNED_KEY[m] for m in TUNED_KEY if not collapsed.get(m)})


# ---------------------------------------------------------------------------
# QM9 <-> experimental names
# ---------------------------------------------------------------------------
# The experimental pipeline builds its roster in run_dataset_experiment
# (alternative_data_noise_robustness.py:2080) and its representations in
# ALL_REPS (:1760). Its display names are the ones the figure script's
# val_model_map already normalises into the QM9 namespace, so these two maps say
# the same thing that mapping does, in the direction the tuner needs.
#
# The GP is the awkward one: the experimental side names it by kernel, 'GP' for
# RBF and 'GP-Tanimoto' otherwise, so it is the one model whose experimental
# name distinguishes what the QM9 tuned-params key cannot.
MODEL_NAME_MAP = {
    'rf':                       'RF',
    'qrf':                      'QRF',
    'xgboost':                  'XGBoost',
    'ngboost':                  'NGBoost',
    'lgb':                      'LightGBM',
    'svm':                      'SVM',
    'gauche_rbf':               'GP',
    'gauche':                   'GP-Tanimoto',
    'dnn':                      'DNN',
    'dnn_bnn_full':             'BNN-Full',
    'dnn_bnn_full_variational': 'VBLL-Full',
    'mlp':                      'MLP',
    'mlp_bnn_full':             'MLP-BNN-Full',
    'mlp_bnn_full_variational': 'MLP-VBLL-Full',
}

REP_NAME_MAP = {
    'ecfp4':          'ECFP4',
    'pdv': 'PDV',
    'sns':            'SNS',
    'mhggnn':         'MHG-GNN-pretrained',
    'avalon':         'Avalon',
    'chemberta':      'ChemBERTa',
}


def to_experimental(model_label, rep):
    """QM9 names -> the experimental pipeline's names for the same pairing."""
    return MODEL_NAME_MAP[model_label], REP_NAME_MAP[rep]
