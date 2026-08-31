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
    # EVERY MODEL HAS ITS OWN KEY (author's decision, 2026-08-31).
    #
    # Until today thirteen of these shared a key with another model: four
    # networks read 'dnn', four read 'mlp', the quantile forest read 'rf' and
    # both Gaussian processes read 'gauche'. `write_master` skipped every shared
    # key -- correctly, because one entry would have changed a model the tuning
    # run never scored -- so only four of the seventeen could receive a tuned
    # value, and the sweep's largest measured gain (dnn_bnn_full on PDV, +1.34)
    # could not reach the cluster at all.
    #
    # The call sites now derive the label from the run's own arguments through
    # roster_label() above, so the key, the job script name and the results file
    # all name the same model.
    'rf':                       'rf',
    'qrf':                      'qrf',
    'svm':                      'svm',
    'xgboost':                  'xgboost',
    'lgb':                      'lgb',
    'ngboost':                  'ngboost',
    'gauche_rbf':               'gauche_rbf',
    'gauche':                   'gauche',
    'dnn':                      'dnn',
    'dnn_bnn_full':             'dnn_bnn_full',
    'dnn_bnn_full_variational': 'dnn_bnn_full_variational',
    'mlp':                      'mlp',
    'mlp_bnn_full':             'mlp_bnn_full',
    'mlp_bnn_full_variational': 'mlp_bnn_full_variational',
    'dnn_bnn_full_variational_hetero': 'dnn_bnn_full_variational_hetero',
    'mlp_bnn_full_variational_hetero': 'mlp_bnn_full_variational_hetero',
    # The heteroscedastic Gaussian process still has NO tuned path:
    # `train_heteroscedastic_gp` (models/models.py) never calls
    # load_best_hyperparameters, so no entry in either JSON file can reach it.
    # None records that as a fact about the model rather than an oversight.
    # It also produced no scored row in any tuning sweep, so there is nothing
    # to give it yet.
    'heteroscedastic_gp':       None,
}

def roster_label(model_type, args):
    """The roster label for the model this run is actually training.

    WHY THIS EXISTS. Until 2026-08-31 every call site passed a hard-coded family
    name, so four different networks asked for the tuned settings of 'dnn', four
    more asked for 'mlp', the quantile forest asked for 'rf' and both Gaussian
    processes asked for 'gauche'. Writing any of those into the master file would
    have changed a model the tuning run never scored, so `write_master` skipped
    them all -- correctly -- and only four of the seventeen models could receive
    a tuned value at all. The largest measured gain in the whole sweep,
    dnn_bnn_full on PDV, was in the unreachable set.

    The label is not a free choice: it is the same string the job generator uses
    for the script name and the results file, so a tuned entry, a job script and
    a results row all name the same model. `scripts/test_tuning_rosters.py`
    checks this function against the generator's own roster.

    Author's decision, 2026-08-31: every model gets its own key.
    """
    bt = getattr(args, 'bayesian_transformation', None)
    hetero = bool(getattr(args, 'heteroscedastic_vbll', False))

    if model_type in ('rf', 'qrf'):
        return model_type
    if model_type == 'gauche':
        # --kernel defaults to tanimoto, so the RBF process must be named by the
        # flag rather than assumed.
        return 'gauche_rbf' if getattr(args, 'kernel', None) == 'rbf' else 'gauche'
    if model_type == 'het_gp':
        return 'heteroscedastic_gp'
    if model_type in ('dnn', 'mlp'):
        if not bt or bt == 'none':
            return model_type
        suffix = {'last_layer': '_bnn_last',
                  'variational': '_bnn_variational',
                  'full': '_bnn_full',
                  'full_variational': '_bnn_full_variational'}.get(bt)
        if suffix is None:
            # An unknown transformation must not silently borrow the plain
            # network's tuned settings.
            raise ValueError(
                f"roster_label does not know the bayesian transformation "
                f"'{bt}'. Add it here rather than letting this run read another "
                f"model's tuned hyperparameters.")
        label = model_type + suffix
        if hetero:
            label += '_hetero'
        return label
    return model_type


# Model labels whose tuned entry is shared with another model. Writing one of
# these into the master file changes a model the tuning run never scored.
def collapsed_models():
    """Model labels that cannot be addressed on their own, and what they collide
    with. Returns {label: [other labels sharing its key]}."""
    by_key = {}
    for label, key in TUNED_KEY.items():
        if key is None:
            continue
        by_key.setdefault(key, []).append(label)
    return {label: sorted(set(by_key[key]) - {label})
            for label, key in TUNED_KEY.items()
            if key is not None and len(by_key[key]) > 1}


def writable_keys():
    """Tuned-file keys that address exactly one model, so a value written under
    them reaches the model it was measured on and nothing else."""
    collapsed = collapsed_models()
    return sorted({TUNED_KEY[m] for m in TUNED_KEY
                   if TUNED_KEY[m] is not None and not collapsed.get(m)})


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
# READ from model_names.json, not retyped. That file is the settled
# correspondence between the two pipelines (2026-08-28, RERUN_PLAN.md 3.4b) and
# the figure script and scripts/uncertainty_stats.py already read it. This map
# was a SECOND hand-typed copy of the same fact, and it went stale exactly the
# way a second copy does: the three models added to the roster on 2026-08-28 --
# heteroscedastic_gp and the two variational networks with a noise head -- were
# added to model_names.json and not here, so the roster gate failed and the
# tuner could not name them.
#
# The one step model_names.json does NOT carry is roster label -> written name.
# The generator submits `heteroscedastic_gp` and the model writes itself as
# `het_gp_rbf` (it is `-m het_gp --kernel rbf`), so that hop is stated here, on
# its own, beside the flags it comes from.
_ROSTER_LABEL_TO_WRITTEN = {
    # roster label in generate_scripts.py -> the name the model writes on a row
    'heteroscedastic_gp':               'het_gp_rbf',
    'dnn_bnn_full_variational_hetero':  'dnn_bnn_full_variational_hetero',
    'mlp_bnn_full_variational_hetero':  'mlp_bnn_full_variational_hetero',
}


def _model_name_map():
    """{QM9 roster label: the experimental pipeline's display name}.

    Built by going QM9 label -> canonical (model_names.json 'qm9') -> the
    laboratory display name (the same file's 'validation', inverted). A roster
    label with no route through raises rather than being dropped: a model that
    silently has no experimental name is how the tuner came to measure a model
    it could not transfer.
    """
    import json
    with open(os.path.join(_ROOT, 'model_names.json')) as fh:
        spec = json.load(fh)
    qm9_to_canonical = spec['qm9']
    canonical_to_validation = {}
    for display, canonical in spec['validation'].items():
        # Several display names map to one canonical name (LGBM/LightGBM). Keep
        # the first, which is the spelling the runner builds its list with.
        canonical_to_validation.setdefault(canonical, display)

    out = {}
    missing = []
    for label in MODELS:
        written = _ROSTER_LABEL_TO_WRITTEN.get(label, label)
        canonical = qm9_to_canonical.get(written)
        display = canonical_to_validation.get(canonical) if canonical else None
        if display is None:
            missing.append(label)
        else:
            out[label] = display
    if missing:
        raise RuntimeError(
            f"model_names.json has no experimental name for roster model(s) "
            f"{sorted(missing)}. Add them there -- it is the one place the two "
            f"pipelines' names are written down -- rather than here.")
    return out


MODEL_NAME_MAP = _model_name_map()

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
