#!/usr/bin/env python3
"""Local hyperparameter tuning for the noise-robustness study.

    export OMP_NUM_THREADS=1
    python scripts/tune_hyperparameters.py --time            # cost first
    python scripts/tune_hyperparameters.py --sweep --settings 12
    python scripts/tune_hyperparameters.py --confirm         # winner vs default
    python scripts/tune_hyperparameters.py --write-master --margin 0.01

    then, before anything is adopted:
    python scripts/confirm_tuned_on_validation_datasets.py --time

WHAT THIS IS
------------
One scaffold split -- train, validation, test -- and no folds. Every model and
representation pairing on the QM9 roster is tuned on its own: a small random
search over that model's parameters, each setting scored by R-squared on the
VALIDATION split, and the best setting written out. The test split is never
touched here; it belongs to the experiment, not to the search.

It replaces the Optuna path, which no job ever used: `--tuning` appears in no
script slurm_scripts_qm9_rerun/generate_scripts.py writes, and four of its
suggested values never reached a model at all (models.py:1660 quantile, :4255
alpha, :4256 predictor_type, :2561 the loss parameters).

A WINNER IS NOT A RESULT UNTIL IT HAS BEATEN THE DEFAULT SOMEWHERE ELSE
-----------------------------------------------------------------------
The search picks the best of N candidates ON THE VALIDATION SPLIT. That winning
score is the maximum of N noisy numbers on the very split that chose it, so it
is biased upward by construction, and the bias GROWS with N -- a bigger search
looks better while being no better. This is where the Optuna path failed: it
wrote the winner to results/master_tuned_hyperparameters.json and nothing ever
compared it with the default on data neither had seen.

So there are two stages after the sweep, and neither is optional:

  --confirm   refits the winner AND the shared default and scores both on the
              QM9 TEST split, which the search never touched. --write-master
              applies --margin to THAT difference, refuses to run if the
              confirmation file is absent, and records how much of the search's
              apparent gain did not survive the move.
  scripts/confirm_tuned_on_validation_datasets.py
              runs the same head-to-head on LogD, Caco-2 and hERG under their
              own scaffold cross-validation. A setting tuned on QM9 that costs
              accuracy on the datasets the paper's validation section rests on
              is not an improvement, whatever QM9 says.

THE CANDIDATE REACHES THE MODEL THE WAY THE CLUSTER WILL DELIVER IT
-------------------------------------------------------------------
This does not rebuild the models. It calls the pipeline's own
`train_<family>_model` functions, with `load_best_hyperparameters` replaced by a
stub that hands back the candidate. That is the same branch, in the same
function, that fires on the cluster with `--use-best-params`. So a setting that
scores well here is a setting the pipeline can actually build -- a parameter
name the builder ignores cannot silently score well and then do nothing, which
is the failure the old master file already contains (`use_default_max_depth`,
which is not an argument of RandomForestRegressor at all).

The validation split is passed where the function expects its test split, so
the R-squared it returns is the validation R-squared. Results rows go to a
scratch CSV under results/tuning_local/, never to a figure input.

WHAT IS TUNED, AND WHAT IS DELIBERATELY NOT
-------------------------------------------
Only keys that (a) the builder actually reads out of the tuned dict and (b)
already exist in models/model_defaults.py. Tuning moves the shared defaults; it
does not introduce parameters that the default path leaves unset, because a
value present only when tuned makes the tuned and default runs differ in two
ways at once.

Consequences, all forced by the builders rather than chosen here:
  * SVM       the kernel is NOT tuned. It is pinned to RBF on every
              representation precisely so the model is free of a
              kernel-representation confound (model_defaults.py SKLEARN_DEFAULTS).
  * GP        the kernel is NOT tuned, for the same reason and one more: the
              tuned entry is keyed 'gauche' for both the RBF and the Tanimoto
              Gaussian process, so a tuned kernel_name on ecfp4 or sns would
              force one kernel on both and delete the head-to-head.
  * dnn       only hidden_size1, hidden_size2 and activation are read from the
              tuned dict (models.py:2444). Its dropout and learning rate are not
              reachable that way.
  * ngboost   only n_estimators, learning_rate and natural_gradient are read
              (models.py:1839); dist and score are resolved by name from the spec.

LOCAL LIMITS, ALL MEASURED ON THIS LAPTOP
-----------------------------------------
  * OMP_NUM_THREADS=1 or torch.randperm segfaults after the import stack. This
    script sets it itself, before torch is imported, and refuses to continue if
    it was already imported at another value.
  * quantile_forest 1.4.1 against scikit-learn 1.3.2 raises
    "Invalid parameter 'monotonic_cst' for estimator DecisionTreeRegressor", so
    qrf cannot be fitted here at all. Its pairings are recorded as blocked, with
    the reason, rather than dropped quietly.
  * The Gaussian process is fitted on one thread via GP_DEFAULTS
    single_thread_fit, which models.py already honours.
  * QM9 is 130k molecules. --sample-size defaults to 10000, which is what the
    QM9 jobs themselves run (generate_scripts.py --sample-size default), so the
    tuned setting is tuned at the size it will be used at. The number is written
    into every output file.
"""
from __future__ import annotations

# Before torch arrives anywhere in the import stack. Measured: with more than
# one OpenMP thread, torch.randperm segfaults on this machine once xgboost,
# lightgbm and torch are all loaded into one process.
import os
import sys

_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for _v in _THREAD_VARS:
    os.environ.setdefault(_v, '1')
# LightGBM segfaults on this laptop even at one thread: it loads its own OpenMP
# runtime while MKL has already loaded another, and the two collide inside the
# fit. Exit 139, no traceback -- the same failure mode GP_DEFAULTS
# single_thread_fit documents for the Gaussian process, by a different route.
# Measured 2026-08-27: lgb x ecfp4 on 400 training molecules dies without this
# and takes 0.1s with it.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
# The danger is not that torch is already loaded -- the check scripts import it
# on the way in, having set the same variables themselves. It is torch loaded
# with a thread count this module could no longer change, because setdefault
# leaves an existing value alone.
# LightGBM must be imported BEFORE torch and tensorflow arrive, and this is the
# first chance. Measured 2026-08-27, three runs, same machine, same environment:
#
#   lightgbm imported first, fitted with its default thread count   works
#   lightgbm imported after process_and_train, fitted with n_jobs=1 works
#   lightgbm imported after process_and_train, default thread count SEGFAULT
#
# Exit 139, no traceback, nothing in the log -- the same threading-runtime
# collision as RERUN_PLAN.md 2.8e, reached by a different route. models.py does
# `import lightgbm as lgb` inside train_lgb_model, so in the pipeline it is
# always the late one; importing it here puts its runtime in first.
import lightgbm as _lightgbm_first  # noqa: F401  (imported for its side effect)

if 'torch' in sys.modules and os.environ.get('OMP_NUM_THREADS') != '1':
    raise RuntimeError(
        f"torch is already imported with OMP_NUM_THREADS="
        f"{os.environ.get('OMP_NUM_THREADS')!r}. It has to be 1 before torch "
        f"loads or torch.randperm segfaults on this machine, and it is too "
        f"late to set it here.")

import argparse
import contextlib
import copy
import csv
import io
import json
import random
import time
import traceback
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')
FEATURE_CACHE = os.path.join(OUT_DIR, 'features')

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# The search spaces
#
# One entry per tuned-params key, not per model label, because the key is what
# load_best_hyperparameters is called with. Every key listed here is read by the
# builder out of the tuned dict AND exists in models/model_defaults.py.
#
# Each value is a callable taking a random.Random. Independent draws, no
# structure between them: this is a random search, and a random search with 12
# settings covers a parameter's useful range better than a grid of 12 does once
# more than two parameters matter (Bergstra & Bengio 2012).
# ---------------------------------------------------------------------------
def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


SEARCH_SPACES = {
    'rf': {
        'n_estimators':     lambda r: r.choice([100, 200, 300, 500, 800]),
        'max_depth':        lambda r: r.choice([None, 10, 20, 40, 80]),
        'min_samples_leaf': lambda r: r.choice([1, 2, 4, 8]),
        'min_samples_split': lambda r: r.choice([2, 5, 10, 20]),
        'max_features':     lambda r: r.choice(['sqrt', 0.1, 0.3, 0.5, 1.0]),
        'bootstrap':        lambda r: r.choice([True, False]),
    },
    # There is no 'qrf' space. train_rf_model looks up the literal 'rf' for the
    # quantile forest too (models.py:1631), so the quantile forest is tuned
    # through the forest's entry or not at all.
    'xgboost': {
        'n_estimators':      lambda r: r.choice([100, 300, 500, 1000]),
        'max_depth':         lambda r: r.choice([3, 4, 6, 8, 10]),
        'learning_rate':     lambda r: _log_uniform(r, 0.01, 0.3),
        'subsample':         lambda r: round(r.uniform(0.5, 1.0), 3),
        'colsample_bytree':  lambda r: round(r.uniform(0.5, 1.0), 3),
        'colsample_bylevel': lambda r: round(r.uniform(0.5, 1.0), 3),
        'min_child_weight':  lambda r: r.choice([1, 2, 5, 10]),
        'gamma':             lambda r: round(r.uniform(0.0, 2.0), 3),
        'reg_alpha':         lambda r: round(r.uniform(0.0, 1.0), 3),
        'reg_lambda':        lambda r: round(r.uniform(0.0, 2.0), 3),
    },
    'lgb': {
        'n_estimators':      lambda r: r.choice([100, 300, 500, 1000]),
        'learning_rate':     lambda r: _log_uniform(r, 0.01, 0.3),
        'num_leaves':        lambda r: r.choice([15, 31, 63, 127]),
        'max_depth':         lambda r: r.choice([-1, 4, 6, 8, 12]),
        'subsample':         lambda r: round(r.uniform(0.5, 1.0), 3),
        'colsample_bytree':  lambda r: round(r.uniform(0.5, 1.0), 3),
        'min_child_samples': lambda r: r.choice([5, 10, 20, 40]),
        'reg_alpha':         lambda r: round(r.uniform(0.0, 1.0), 3),
        'reg_lambda':        lambda r: round(r.uniform(0.0, 2.0), 3),
    },
    'ngboost': {
        'n_estimators':     lambda r: r.choice([200, 500, 800, 1200]),
        'learning_rate':    lambda r: _log_uniform(r, 0.003, 0.1),
        'natural_gradient': lambda r: r.choice([True, False]),
    },
    'svm': {
        # kernel is pinned; see the module docstring.
        'C':     lambda r: _log_uniform(r, 0.1, 100.0),
        'gamma': lambda r: r.choice(['scale', 'auto']) if r.random() < 0.4
                 else _log_uniform(r, 1e-4, 1.0),
        'kernel': lambda r: 'rbf',
    },
    'gauche': {
        # kernel_name is PINNED, not searched -- see pin_kernel() below. It has
        # to be present: once the dict comes from the tuned file the builder
        # reads params['kernel_name'] with no fallback to --kernel, so an entry
        # without it raises KeyError instead of using the CLI kernel.
        'outputscale':      lambda r: _log_uniform(r, 0.1, 10.0),
        'likelihood_noise': lambda r: _log_uniform(r, 1e-4, 0.1),
    },
    'dnn': {
        'hidden_size1': lambda r: r.choice([64, 128, 256, 512, 1024]),
        'hidden_size2': lambda r: r.choice([32, 64, 128, 256, 512]),
        'activation':   lambda r: r.choice(['relu', 'tanh']),
    },
    'mlp': {
        'hidden_size':       lambda r: r.choice([64, 128, 256, 512]),
        'num_hidden_layers': lambda r: r.choice([1, 2, 3, 4]),
        'dropout_rate':      lambda r: round(r.uniform(0.05, 0.5), 3),
        'lr':                lambda r: _log_uniform(r, 1e-4, 1e-2),
    },
}


# The Gaussian process is named by kernel on the experimental side and by flag
# here; these are the two spellings train_gauche_model's kernel_map accepts.
#
# heteroscedastic_gp is deliberately absent. Its tuned-file key is 'gauche', so
# it would take the pinned-kernel branch of sample_setting() below and look
# itself up here -- but its builder never calls load_best_hyperparameters at all
# (models.py:8186-8418, zero call sites), so no searched value can reach it and
# there is nothing for a kernel pin to pin. It is refused by name in run()
# instead, from models/tuning_rosters.py's READS_NO_TUNED_PARAMS, so the reason
# is recorded once and read here rather than restated.
GP_KERNEL_FOR = {'gauche_rbf': 'RBF', 'gauche': 'Tanimoto'}


def search_family(model_label, rosters=None):
    """Which SEARCH_SPACES entry this model is searched under.

    NOT the same question as which tuned-file key it reads. Several models share
    a search space because they share an architecture -- the quantile forest is
    searched as a forest, the Bayesian and variational networks as their
    deterministic base, both Gaussian processes as one. Which tuned FILE key each
    one reads is a separate decision, recorded in models/tuning_rosters.py, and
    it changed on 2026-08-31 when every model was given its own key. Reading the
    search space off that map broke the moment it did, because the map then named
    keys no search space has.
    """
    if rosters is not None:
        key = getattr(rosters, 'TUNED_KEY', {}).get(model_label)
        if key in SEARCH_SPACES:
            return key
    if model_label in SEARCH_SPACES:
        return model_label
    for suffix in ('_bnn_full_variational_hetero', '_bnn_full_variational',
                   '_bnn_full'):
        if model_label.endswith(suffix):
            base = model_label[:-len(suffix)]
            if base in SEARCH_SPACES:
                return base
    return {'qrf': 'rf', 'gauche_rbf': 'gauche'}.get(model_label)


def sample_setting(key, rng, model_label=None):
    params = {name: draw(rng) for name, draw in SEARCH_SPACES[key].items()}
    if key == 'gauche':
        # Pinned to the model's own kernel, so the search moves the two scale
        # parameters and nothing else. Searching it would both reintroduce a
        # kernel-representation confound and, because gauche_rbf and gauche share
        # one tuned entry, force one kernel on both.
        params['kernel_name'] = GP_KERNEL_FOR[model_label]
    return params


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _import_pipeline():
    """The real pipeline, imported once. Heavy -- deepchem drags tensorflow in."""
    import process_and_train as pat
    import models as M
    if unblock_quantile_forest():
        print('  [qrf] scikit-learn is older than quantile_forest expects; the '
              'keyword it cannot take is removed, which changes no fitted model.',
              flush=True)
    return pat, M


VALIDATION_DATASETS = ('logd', 'caco2', 'herg')


def build_validation_split(dataset, sample_size, seed, openadmet_csv=None):
    """One scaffold split of LogD, Caco-2 or hERG, through their own loaders.

    Hyperparameters are dataset-specific. A setting chosen on 4,000 small
    molecules with a computed label has no claim on 1,400 drug-like molecules
    with real assay error, so each of the three is tuned on its own data rather
    than inheriting QM9's answer.

    The molecules, the representations and the scaffold grouping all come from
    the experimental pipeline itself. Rebuilding any of them here would tune on
    features that pipeline does not use.
    """
    import importlib.util

    import tuning_rosters as rosters
    from sklearn.model_selection import GroupShuffleSplit

    spec = importlib.util.spec_from_file_location(
        'kirby_validation_pipeline', rosters.KIRBY_PIPELINE_PATH)
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)

    if dataset == 'herg':
        smiles, labels = K.load_chembl_herg()
    else:
        df = K.download_openadmet(csv_path=openadmet_csv)
        if dataset == 'logd':
            col = next(c for c in df.columns if 'LogD' in c)
            smiles, labels = K.load_openadmet_endpoint(df, col,
                                                       log_transform=False)
        else:
            col = next(c for c in df.columns if 'Caco' in c and 'Efflux' in c)
            smiles, labels = K.load_openadmet_endpoint(df, col,
                                                       log_transform=True)

    smiles = list(smiles)
    y = np.asarray(labels, dtype=np.float64)
    if sample_size and sample_size < len(smiles):
        smiles, y = smiles[:sample_size], y[:sample_size]

    # 80/10/10 by scaffold, the same shape as QM9's split. Grouped, so no
    # scaffold appears in two parts.
    groups, _ = K.assign_scaffold_groups(smiles)
    idx = np.arange(len(smiles))
    tr, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                      random_state=seed).split(idx, y, groups))
    sub = next(GroupShuffleSplit(n_splits=1, test_size=0.5,
                                 random_state=seed).split(
        rest, y[rest], groups[rest]))
    val, test = rest[sub[0]], rest[sub[1]]
    return smiles, y, list(tr), list(val), list(test), K


def build_split(pat, sample_size, seed):
    """One scaffold split of QM9: train / validation / test, no folds.

    The molecules and the split rule are the pipeline's own -- load_qm9 for the
    dataset and its RDKit-valid filter, scaffold_split_indices at 80/10/10, the
    same call split_qm9 makes.
    """
    import torch

    # Loading QM9 through torch_geometric reads a 340 MB tensor file and then
    # decodes one molecule at a time, which costs minutes before any model is
    # fitted. The split is a pure function of (sample_size, seed), so it is kept.
    os.makedirs(OUT_DIR, exist_ok=True)
    cached = os.path.join(OUT_DIR, f'split_n{sample_size}_seed{seed}.json')
    if os.path.exists(cached):
        with open(cached) as fh:
            blob = json.load(fh)
        return (blob['smiles'], np.array(blob['y'], dtype=np.float64),
                blob['train_idx'], blob['val_idx'], blob['test_idx'])

    torch.manual_seed(seed)
    np.random.seed(seed)

    qm9 = pat.load_qm9('homo_lumo_gap')
    idx = torch.randperm(len(qm9), generator=torch.Generator().manual_seed(seed))
    qm9 = qm9.index_select(idx)

    n = min(sample_size, len(qm9))
    smiles = [qm9[i].smiles for i in range(n)]
    y = np.array([float(qm9[i].y) for i in range(n)], dtype=np.float64)

    train_idx, val_idx, test_idx = pat.scaffold_split_indices(
        smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    train_idx, val_idx, test_idx = (list(map(int, train_idx)),
                                    list(map(int, val_idx)),
                                    list(map(int, test_idx)))
    tmp = f'{cached}.{os.getpid()}.tmp'
    with open(tmp, 'w') as fh:
        json.dump({'smiles': smiles, 'y': y.tolist(), 'train_idx': train_idx,
                   'val_idx': val_idx, 'test_idx': test_idx}, fh)
    os.replace(tmp, cached)   # atomic; concurrent array tasks share this file
    return smiles, y, train_idx, val_idx, test_idx


def featurise(pat, rep, smiles, train_idx):
    """The pipeline's featurisers, in memory.

    Sort & Slice is fitted on the TRAINING molecules only, as split_qm9 does.
    Everything else is per molecule.
    """
    from rdkit import Chem

    if rep == 'ecfp4':
        rows = [np.unpackbits(pat.ecfp4_fingerprint(s), bitorder='little')
                for s in smiles]
    elif rep == 'avalon':
        rows = [np.unpackbits(pat.avalon_fingerprint(s), bitorder='little')
                for s in smiles]
    elif rep == 'pdv':
        rows = [pat.rdkit_mol_descriptors_from_smiles(s) for s in smiles]
    elif rep == 'chemberta':
        rows = [pat.chemberta_fingerprint(s) for s in smiles]
    elif rep == 'mhggnn':
        rows = [pat.mhggnn_fingerprint(s) for s in smiles]
    elif rep == 'sns':
        train = set(train_idx)
        mols_train = [Chem.MolFromSmiles(s)
                      for i, s in enumerate(smiles) if i in train]
        featuriser = pat.create_sort_and_slice_ecfp_featuriser(
            mols_train=mols_train, max_radius=2, pharm_atom_invs=False,
            bond_invs=True, chirality=False, sub_counts=True,
            vec_dimension=pat.SNS_DIM, print_train_set_info=False)
        rows = [featuriser(Chem.MolFromSmiles(s)) for s in smiles]
    else:
        raise KeyError(f'no featuriser here for {rep!r}')
    return np.asarray(rows, dtype=np.float32)


def validation_features(K, rep, smiles, cache_key):
    """Representations for a validation dataset, built by that pipeline itself.

    Cached on disk: the transformer and graph encoders are slow, and every model
    tuned on the same dataset wants the same matrix.
    """
    import tuning_rosters as rosters

    os.makedirs(FEATURE_CACHE, exist_ok=True)
    path = os.path.join(FEATURE_CACHE, f'{cache_key}_{rep}.npy')
    if os.path.exists(path):
        return np.load(path)
    exp_rep = rosters.REP_NAME_MAP[rep]
    built = K.generate_representations(list(smiles), rep_filter=[exp_rep])
    if exp_rep not in built:
        raise RuntimeError(f'{exp_rep} could not be built for this dataset')
    x = np.asarray(built[exp_rep], dtype=np.float32)
    tmp = f'{path}.{os.getpid()}.tmp'
    np.save(tmp, x)
    os.replace(tmp if tmp.endswith('.npy') else tmp + '.npy', path)
    return x


def features_for(pat, rep, smiles, train_idx, sample_size, seed, cache=True):
    """Featurise once and keep it. Re-featurising ChemBERTa and MHG-GNN for
    every setting would cost more than the fits."""
    os.makedirs(FEATURE_CACHE, exist_ok=True)
    path = os.path.join(FEATURE_CACHE, f'{rep}_n{sample_size}_seed{seed}.npy')
    if cache and os.path.exists(path):
        return np.load(path)
    x = featurise(pat, rep, smiles, train_idx)
    if cache:
        # Write to a private name and rename into place. On the cluster every
        # array task featurises the same representation at the same time, and a
        # half-written .npy that another task then reads is the config-race
        # defect class again (RERUN_PLAN.md 2.8a). rename is atomic within a
        # filesystem, so a reader sees either the old file or a complete one.
        tmp = f'{path}.{os.getpid()}.tmp'
        np.save(tmp, x)
        os.replace(tmp if tmp.endswith('.npy') else tmp + '.npy', path)
    return x


def standardise(x_train, x_val, x_test, rep):
    """Per-feature standardisation on the TRAINING split alone, or not at all --
    the decision is made from the matrix by the shared spec, exactly as
    process_and_train.py:2876 makes it."""
    from model_defaults import should_standardise

    if not should_standardise(x_train, rep):
        return x_train, x_val, x_test, False

    clean = lambda a: np.nan_to_num(np.asarray(a, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
    x_train, x_val, x_test = clean(x_train), clean(x_val), clean(x_test)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (((x_train - mean) / std).astype(np.float32),
            ((x_val - mean) / std).astype(np.float32),
            ((x_test - mean) / std).astype(np.float32),
            True)


# ---------------------------------------------------------------------------
# Running one fit
# ---------------------------------------------------------------------------
def make_args(pat, model_label, rep, sample_size, scratch_csv, rosters):
    """An args object built by the pipeline's own parser, from the job
    generator's own flags for this model. Nothing about the model is retyped."""
    flags = rosters.MODELS[model_label][0].split()
    argv = ['process_and_train.py', '-d', 'QM9', '-t', 'homo_lumo_gap',
            '-r', rep, '-n', str(sample_size), '-s', 'scaffold',
            '--normalize', 'True', '-f', scratch_csv] + flags
    saved = sys.argv
    try:
        sys.argv = argv
        args = pat.parse_arguments()
    finally:
        sys.argv = saved

    # Tuning scores R-squared. The uncertainty outputs are not scored, and
    # producing them would multiply the cost of every Bayesian fit by the
    # hundred stochastic passes and the out-of-fold pass.
    args.uncertainty = False
    args.oof_folds = 0
    args.tuning = False
    args.use_best_params = False
    args.save_per_epoch_metrics = False
    args.logging = False
    return args


@contextlib.contextmanager
def candidate(M, params):
    """Deliver `params` through the branch the cluster uses.

    models.py reads tuned parameters in exactly one place. Replacing that one
    function is what makes this a test of the delivery path and not a private
    reimplementation of it.
    """
    original = M.load_best_hyperparameters
    M.load_best_hyperparameters = lambda model_type, rep, results_dir='results': (
        dict(params) if params is not None else None)
    try:
        yield
    finally:
        M.load_best_hyperparameters = original


def fit_once(pat, M, model_label, rep, data, params, sample_size, scratch_csv,
             rosters, quiet=True, score_on='val'):
    """Fit one pairing at one setting. Returns (R-squared, seconds).

    `score_on` says which split the returned R-squared is measured on. The split
    asked for is handed in where the function expects its test split, so the
    number that comes back is that split's score.

      'val'   the search. Every candidate is scored here.
      'test'  --confirm, and nothing else. The search never sees this split, so
              it is the only place a tuned setting can be compared with the
              default without the comparison being decided by the same numbers
              that chose the winner.
    """
    x_train, y_train = data['x_train'], data['y_train']
    # The split being SCORED, and separately the split the neural models early-stop
    # on. They must not be the same object in --confirm: passing test as the
    # early-stopping split would choose the stopping epoch on the split the
    # comparison is decided on, which is the leak this whole stage exists to close.
    x_score, y_score = data[f'x_{score_on}'], data[f'y_{score_on}']
    x_stop, y_stop = data['x_val'], data['y_val']
    args = make_args(pat, model_label, rep, sample_size, scratch_csv, rosters)
    model_type = args.models[0]

    use_tuned = params is not None
    args.use_best_params = use_tuned

    sink = io.StringIO() if quiet else sys.stdout
    started = time.perf_counter()
    with candidate(M, params if use_tuned else None), \
            contextlib.redirect_stdout(sink):
        # The scored split goes in where the function expects its test split;
        # the validation argument keeps the real validation split, which is what
        # the neural models early-stop on and what any calibration would use.
        common = dict(x_train=x_train, y_train=y_train,
                      x_test=x_score, y_test=y_score,
                      x_val=x_stop, y_val=y_stop,
                      args=args, s=0.0, rep=rep, iteration=0,
                      iteration_seed=RANDOM_SEED, train_noise=None)
        if model_type in ('rf', 'qrf'):
            r2 = M.train_rf_model(model_type=model_type, file_no=0,
                                  y_test_original=y_score, **common)
        elif model_type == 'svm':
            r2 = M.train_svm_model(**common)
        elif model_type == 'xgboost':
            r2 = M.train_xgboost_model(**common)
        elif model_type == 'lgb':
            r2 = M.train_lgb_model(**common)
        elif model_type == 'ngboost':
            r2 = M.train_ngboost_model(file_no=0, y_test_original=y_score, **common)
        elif model_type == 'gauche':
            r2 = M.train_gauche_model(file_no=0, y_test_original=y_score, **common)
        elif model_type == 'dnn':
            r2 = M.train_dnn_model(file_no=0, y_test_original=y_score, **common)
        elif model_type == 'mlp':
            r2 = M.train_mlp_variant_model(model_type='mlp', file_no=0,
                                           y_test_original=y_score, **common)
        else:
            raise KeyError(f'no local runner for model type {model_type!r} '
                           f'(label {model_label!r})')
    return float(r2), time.perf_counter() - started


# ---------------------------------------------------------------------------
# The two passes
# ---------------------------------------------------------------------------
BLOCKED = {}


def unblock_quantile_forest():
    """Let the quantile forest fit against this older scikit-learn.

    quantile_forest lists `monotonic_cst` among the settings it hands down to
    each tree. scikit-learn 1.3.2 does not accept that keyword at all, so the
    fit raises "Invalid parameter 'monotonic_cst'" before it starts, and the
    quantile forest could not be tuned on any dataset.

    `monotonic_cst` means "no monotonic constraint" at its default, which is the
    only value anything here would use. Removing it from the list changes NOTHING
    about the model that gets fitted -- verified against a fit that works: the
    same trees, the same quantile predictions.

    Done by patching the class once at import, so every construction is covered:
    the tuner's, the out-of-fold refits inside models.py, and any other caller in
    the same process. The alternative was changing the installed package, which
    would have disturbed other work running on this machine.
    """
    import sklearn
    if sklearn.__version__ >= '1.4':
        return False
    try:
        import quantile_forest as qf
    except ImportError:
        return False
    original = qf.RandomForestQuantileRegressor
    if getattr(original, '_monotonic_cst_stripped', False):
        return True

    # A FACTORY, NOT A SUBCLASS, and not a replaced __init__ either.
    #
    # scikit-learn reads an estimator's parameter names off its __init__
    # signature and REFUSES one that takes *args -- "estimators should always
    # specify their parameters in the signature of their __init__". So wrapping
    # the constructor breaks the estimator in a second way. Building the real
    # class and then correcting the instance leaves the estimator exactly as
    # scikit-learn expects it.
    def build(*a, **kw):
        model = original(*a, **kw)
        params = getattr(model, 'estimator_params', ())
        if 'monotonic_cst' in params:
            model.estimator_params = tuple(
                p for p in params if p != 'monotonic_cst')
        return model

    build._monotonic_cst_stripped = True
    build.__wrapped__ = original
    qf.RandomForestQuantileRegressor = build
    return True
    original = Q.__init__

    def __init__(self, *a, **kw):
        original(self, *a, **kw)
        if 'monotonic_cst' in getattr(self, 'estimator_params', ()):  # older sklearn
            self.estimator_params = tuple(
                p for p in self.estimator_params if p != 'monotonic_cst')

    import sklearn
    if sklearn.__version__ < '1.4':
        Q.__init__ = __init__
        return True
    return False


def prepared_data(pat, rep, smiles, y, train_idx, val_idx, test_idx,
                  sample_size, seed, K=None, cache_key=None):
    """All three splits, standardised on the training split alone.

    The test split is carried but NOT handed to the search. It is used by
    --confirm and by nothing else.
    """
    if K is not None:
        x = validation_features(K, rep, smiles, cache_key)
    else:
        x = features_for(pat, rep, smiles, train_idx, sample_size, seed)
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    x_train, x_val, x_test, scaled = standardise(x_train, x_val, x_test, rep)
    return ({'x_train': x_train, 'y_train': y[train_idx].astype(np.float32),
             'x_val': x_val, 'y_val': y[val_idx].astype(np.float32),
             'x_test': x_test, 'y_test': y[test_idx].astype(np.float32)},
            scaled)


def run(args_cli):
    import tuning_rosters as rosters

    pat, M = _import_pipeline()
    os.makedirs(OUT_DIR, exist_ok=True)
    # Per-run too: two runs sharing one scratch file is how the column-mismatch
    # error appears, and it is the same collision as the timing file.
    scratch_csv = os.path.join(OUT_DIR, f'scratch_rows{args_cli.tag}.csv')

    pairings = rosters.qm9_pairings()
    if args_cli.models:
        pairings = [(m, r) for m, r in pairings if m in args_cli.models]
    if args_cli.reps:
        pairings = [(m, r) for m, r in pairings if r in args_cli.reps]

    # EVERY NAME IS RESOLVED BEFORE ANYTHING IS FITTED.
    #
    # to_experimental() indexes two plain dicts with no default. When the job
    # generator gained heteroscedastic_gp and the two heteroscedastic VBLL
    # models on 2026-08-28 and the maps were not updated with them, the sweep
    # fitted every other pairing first -- the three sit last in the generator's
    # dict -- and then raised KeyError inside the loop. write_best() is called
    # after the loop, so the run ended with hours of trial rows on disk and
    # without best_by_pairing.json, which --confirm and --write-master both
    # refuse to run without. (The rows are recoverable with --merge, but only
    # if somebody notices.)
    #
    # Resolving the names up front costs nothing and turns that into a
    # one-second failure that names the models. It runs before build_split,
    # which reads QM9 off disk and takes minutes.
    unmapped = sorted({m for m, _ in pairings
                       if m not in rosters.MODEL_NAME_MAP
                       or m not in rosters.TUNED_KEY})
    if unmapped:
        raise SystemExit(
            f'these models are on the QM9 job generator\'s roster and missing '
            f'from models/tuning_rosters.py: {unmapped}. Nothing is fitted '
            f'until they are named there; add them and re-run '
            f'scripts/test_tuning_rosters.py.')

    # The same up-front resolution for the pinned Gaussian-process kernel, which
    # is the THIRD table this run indexes by model label with no default. On
    # 2026-08-28 it was the one still missing an entry after the other two were
    # filled: the run got past the pairing loop's first two lookups and died at
    # sample_setting() instead, one line later and just as far into the sweep.
    # Models refused below never reach it, so they are not required to be here.
    refused = set(getattr(rosters, 'READS_NO_TUNED_PARAMS', frozenset()))
    unpinned = sorted({m for m, _ in pairings
                       if rosters.TUNED_KEY[m] == 'gauche'
                       and m not in GP_KERNEL_FOR and m not in refused})
    if unpinned:
        raise SystemExit(
            f'these models read the shared \'gauche\' tuned entry and have no '
            f'pinned kernel in GP_KERNEL_FOR: {unpinned}. Searching them would '
            f'either raise here or force one kernel on both Gaussian '
            f'processes; name the kernel, or list them in '
            f'READS_NO_TUNED_PARAMS if their builder ignores tuned values.')

    # A MODEL THAT CANNOT RECEIVE A SEARCHED VALUE IS NOT SEARCHED.
    #
    # models/tuning_rosters.py records which builders contain no call to
    # load_best_hyperparameters at all. Fitting settings for one of those does
    # not fail -- which is the danger. Every setting would deliver the same
    # default model, so the twelve rows would differ only by fitting noise and
    # one of them would still come out ahead of the 'default' row and read as a
    # win. Recording the refusal keeps the pairing visible in the trials file
    # with its reason, which a silent skip would not.
    blocked = dict(BLOCKED)
    for label in refused:
        blocked.setdefault(
            label,
            f'{label} builds itself from the shared spec and the command line; '
            f'its builder in models/models.py contains no call to '
            f'load_best_hyperparameters, so no searched value can reach it and '
            f'every setting would refit the same default model.')

    K, cache_key = None, None
    if args_cli.dataset != 'qm9':
        print(f'loading {args_cli.dataset} and taking one scaffold split...',
              flush=True)
        smiles, y, train_idx, val_idx, test_idx, K = build_validation_split(
            args_cli.dataset, args_cli.sample_size, args_cli.seed,
            args_cli.openadmet_csv)
        cache_key = f'{args_cli.dataset}_seed{args_cli.seed}'
    else:
        print(f'loading QM9 and taking one scaffold split of '
              f'{args_cli.sample_size} molecules...', flush=True)
        smiles, y, train_idx, val_idx, test_idx = build_split(
            pat, args_cli.sample_size, args_cli.seed)
    print(f'  train {len(train_idx)}  validation {len(val_idx)}  '
          f'test {len(test_idx)}  (test is not used by this script)', flush=True)

    rng = random.Random(args_cli.seed)
    n_settings = 0 if args_cli.time else args_cli.settings

    # The rows are written as they are produced, not collected and dumped at the
    # end. A sweep over the whole roster runs for hours locally, and a run that
    # is interrupted at hour three should leave the three hours of measurements
    # it made, not nothing.
    stem = 'timing' if args_cli.time else 'trials'
    # One file per task. A fixed name would have every array task on the cluster
    # writing the same path, and the last one to finish would be the only one
    # that survived -- silently, with a plausible-looking file left behind.
    out = os.path.join(OUT_DIR, f'{stem}{args_cli.tag}.csv')
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    fields = ['model', 'rep', 'setting', 'r2', 'seconds', 'status', 'detail',
              'sample_size', 'seed', 'written']
    csv_fh = open(out, 'w', newline='')
    writer = csv.DictWriter(csv_fh, fieldnames=fields)
    writer.writeheader()

    def record(row):
        row.update(sample_size=args_cli.sample_size, seed=args_cli.seed,
                   written=stamp)
        writer.writerow(row)
        csv_fh.flush()
        rows.append(row)

    rows = []
    cache = {}
    for i, (model_label, rep) in enumerate(pairings, 1):
        exp_model, exp_rep = rosters.to_experimental(model_label, rep)
        head = (f'[{i}/{len(pairings)}] {model_label} x {rep}  '
                f'(experimental: {exp_model} x {exp_rep})')
        if model_label in blocked:
            print(f'{head}  BLOCKED LOCALLY', flush=True)
            record(dict(model=model_label, rep=rep, setting='blocked',
                        r2='', seconds='', status='blocked',
                        detail=blocked[model_label]))
            continue

        if rep not in cache:
            t0 = time.perf_counter()
            cache[rep] = prepared_data(pat, rep, smiles, y, train_idx, val_idx,
                                       test_idx, args_cli.sample_size,
                                       args_cli.seed, K=K, cache_key=cache_key)
            print(f'  featurised {rep} in {time.perf_counter() - t0:.1f}s '
                  f'(standardised: {cache[rep][1]})', flush=True)
        data, _ = cache[rep]

        key = rosters.TUNED_KEY[model_label]
        candidates = [sample_setting(key, rng, model_label)
                      for _ in range(n_settings)]

        def _fit(params, label, on):
            """One fit, recorded. `on` is the data dict to train against."""
            try:
                r2, secs = fit_once(pat, M, model_label, rep, on, params,
                                    args_cli.sample_size, scratch_csv, rosters)
                status, detail = 'ok', (json.dumps(params) if params else '')
            except Exception as exc:
                r2, secs, status = '', '', 'error'
                detail = f'{type(exc).__name__}: {exc}'
                if args_cli.traceback:
                    traceback.print_exc()
            print(f'{head}  {label}: R2={r2 if r2 == "" else f"{r2:+.4f}"}  '
                  f'{secs if secs == "" else f"{secs:.1f}s"}  {status}'
                  f'{"  " + detail[:120] if status == "error" else ""}',
                  flush=True)
            record(dict(model=model_label, rep=rep, setting=label,
                        r2=r2, seconds=secs, status=status, detail=detail))
            return r2

        # THE DEFAULT IS ALWAYS FITTED AT FULL SIZE. It is the baseline every
        # winner is measured against, so it is never screened and never dropped.
        _fit(None, 'default', data)

        if args_cli.screen_at:
            # SCREEN ON TRAINING-SET SIZE, VALIDATION HELD FIXED.
            #
            # Every candidate is fitted on the first `screen_at` training
            # molecules and scored on the SAME validation molecules the full
            # round uses. Only the training budget changes between the rounds,
            # which is what makes the two sets of scores comparable and the
            # promotion meaningful.
            #
            # Screening against a separate, smaller split instead would change
            # the validation molecules too, and then a candidate could be
            # promoted for having had an easier validation set rather than for
            # being better.
            n_screen = min(args_cli.screen_at, len(data['x_train']))
            small = dict(data)
            small['x_train'] = data['x_train'][:n_screen]
            small['y_train'] = data['y_train'][:n_screen]
            print(f'{head}  screening {len(candidates)} settings on '
                  f'{n_screen} of {len(data["x_train"])} training molecules, '
                  f'promoting {args_cli.promote}', flush=True)

            scored = []
            for n, params in enumerate(candidates, 1):
                r2 = _fit(params, f'screen_{n}', small)
                if r2 != '':
                    scored.append((r2, n, params))
            scored.sort(key=lambda t: -t[0])
            survivors = scored[:args_cli.promote]
            if not survivors:
                print(f'{head}  no setting survived screening — every fit '
                      f'errored, so there is nothing to promote', flush=True)
            for r2_small, n, params in survivors:
                _fit(params, f'final_{n}', data)
        else:
            for n, params in enumerate(candidates, 1):
                _fit(params, f'setting_{n}', data)

    csv_fh.close()
    print(f'\nwrote {out}  ({len(rows)} rows)')

    if args_cli.time:
        report_arithmetic(rows, args_cli)
    else:
        write_best(rows, args_cli)
    return 0


def report_arithmetic(rows, args_cli):
    """One fit each, then the multiplication. The point of doing this first is
    that the sweep is priced before it is sized, not shrunk to fit."""
    ok = [r for r in rows if r['status'] == 'ok']
    total = sum(float(r['seconds']) for r in ok)
    blocked = [r for r in rows if r['status'] == 'blocked']
    errors = [r for r in rows if r['status'] == 'error']

    print('\n' + '=' * 72)
    print(f'ONE FIT AT DEFAULTS, {len(ok)} pairings that ran')
    print('=' * 72)
    for r in sorted(ok, key=lambda r: -float(r['seconds']))[:15]:
        print(f'  {float(r["seconds"]):8.1f}s  {r["model"]} x {r["rep"]}')
    print(f'\n  one pass over every pairing: {total / 60:.1f} min')
    for n in (5, 8, 12, 20, 30):
        print(f'  {n:3d} settings each (+1 default): '
              f'{total * (n + 1) / 3600:.2f} h')
    if blocked:
        print(f'\n  blocked locally: {sorted({r["model"] for r in blocked})}')
    if errors:
        print(f'  errored: {len(errors)} — see the detail column')


def write_best(rows, args_cli):
    """Best validation R-squared per pairing, plus what beating the default is
    worth. Nothing is written into results/ proper here; --write-master does
    that, once the adoption rule is settled."""
    # A SCREENING SCORE IS NOT A RESULT. Where a pairing was screened, its
    # screen_* rows were fitted on a fraction of the training molecules; a lucky
    # one of those could easily beat a real score from the full round. So for any
    # pairing that produced a final_* row, only the final_* rows count.
    screened = {(r['model'], r['rep']) for r in rows
                if r['status'] == 'ok' and r['setting'].startswith('final_')}

    best = {}
    for r in rows:
        if r['status'] != 'ok':
            continue
        pair = (r['model'], r['rep'])
        if r['setting'].startswith('screen_') and pair in screened:
            continue
        rec = best.setdefault(pair, {'default_r2': None, 'best_r2': None,
                                     'best_params': None})
        if r['setting'] == 'default':
            rec['default_r2'] = float(r['r2'])
        elif rec['best_r2'] is None or float(r['r2']) > rec['best_r2']:
            rec['best_r2'] = float(r['r2'])
            rec['best_params'] = json.loads(r['detail']) if r['detail'] else None

    # A TAGGED SWEEP WRITES ITS OWN FILE AND LEAVES THE SHARED ONE ALONE.
    #
    # This is the same collision the trials and timing CSVs were tagged for in
    # main(), one file later. Five sweeps were running side by side on 2026-08-29
    # under --tag cheap/vbll/gp/bnn/ngboost, and every one of them ends by
    # calling this function; the path was fixed, so each finisher replaced the
    # whole file with only its own pairings and the last one to end would have
    # been the only search --confirm and --write-master ever saw. It was
    # measured: a two-pairing probe run replaced a twelve-pairing file that the
    # plain-network sweep had written three hours earlier.
    #
    # Tagged runs now write best_by_pairing<tag>.json. The shared, untagged
    # best_by_pairing.json -- the one --confirm and --write-master read -- is
    # written by an untagged sweep or by --merge, which is the mode that exists
    # to combine tagged runs and is the only thing that can see all of them.
    out = os.path.join(OUT_DIR, f'best_by_pairing{args_cli.tag}.json')
    payload = {
        'sample_size': args_cli.sample_size,
        'seed': args_cli.seed,
        'settings_per_pairing': args_cli.settings,
        'scored_on': 'validation R2, one scaffold split, no folds',
        'labels': 'clean' if not args_cli.noise_level else args_cli.noise_level,
        'written': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'pairings': {f'{m}|{r}': v for (m, r), v in sorted(best.items())},
    }
    with open(out, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'wrote {out}  ({len(best)} pairings)')

    gains = [(v['best_r2'] - v['default_r2'], m, r)
             for (m, r), v in best.items()
             if v['best_r2'] is not None and v['default_r2'] is not None]
    gains.sort(reverse=True)
    print('\n  largest gains over the shared default:')
    for g, m, r in gains[:10]:
        print(f'    {g:+.4f}  {m} x {r}')
    print(f'  pairings where the default won: '
          f'{sum(1 for g, _, _ in gains if g <= 0)} of {len(gains)}')




def merge(args_cli):
    """Read every tagged trials file back and build one best_by_pairing.json.

    On the cluster each array task tunes one pairing and writes its own
    trials_<tag>.csv (--tag). This is the step that turns those back into the
    single file --confirm and --write-master read, and it is where a task that
    died is noticed: a pairing on the roster with no rows is reported by name,
    not quietly absent from the result.
    """
    import glob

    import tuning_rosters as rosters

    paths = sorted(glob.glob(os.path.join(OUT_DIR, 'trials*.csv')))
    if not paths:
        print(f'no trials*.csv under {OUT_DIR} -- nothing to merge.')
        return 1
    rows = []
    for path in paths:
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    print(f'{len(rows)} rows from {len(paths)} file(s)')

    sizes = {r['sample_size'] for r in rows if r.get('sample_size')}
    seeds = {r['seed'] for r in rows if r.get('seed')}
    if len(sizes) > 1 or len(seeds) > 1:
        print(f'REFUSING to merge: the files disagree on the split. '
              f'sample sizes {sorted(sizes)}, seeds {sorted(seeds)}. '
              f'Settings tuned on different splits are not comparable and the '
              f'best-of would be picked across two different experiments.')
        return 1

    # --merge is the mode that produces the SHARED file, whatever tag it was
    # invoked with. write_best now suffixes the tag so that concurrent sweeps
    # stop overwriting each other, and a merge run that inherited a tag would
    # otherwise write best_by_pairing_<tag>.json and leave --confirm with
    # nothing -- the one outcome this mode exists to prevent.
    merged = copy.copy(args_cli)
    merged.tag = ''
    write_best(rows, merged)

    scored = {(r['model'], r['rep']) for r in rows if r['status'] == 'ok'}
    blocked = {(r['model'], r['rep']) for r in rows if r['status'] == 'blocked'}
    missing = [p for p in rosters.qm9_pairings()
               if p not in scored and p not in blocked]
    if missing:
        print(f'\n  {len(missing)} pairing(s) on the roster produced no scored '
              f'row. A task that died leaves no trace but this:')
        for m, r in missing:
            print(f'      {m} x {r}')
    return 0


def confirm(args_cli):
    """Refit the winner and the shared default, and score both on the TEST split.

    WHY THIS STAGE EXISTS
    ---------------------
    This is where the Optuna path failed. It picked a best trial by validation
    score, wrote it to results/master_tuned_hyperparameters.json, and nothing
    ever compared that setting with the default on data neither had seen. The
    winner of a search over N candidates is the maximum of N noisy numbers on the
    split that chose it, so its validation score is biased upward by construction
    -- and the bias grows with N, which means a bigger search looks better while
    being no better.

    So the tuned setting and the shared default are refitted here, on the same
    training split, and both are scored on the TEST split. The search never
    touched it. The neural models still early-stop on validation, not on test.

    The number this writes -- not the search's -- is what --write-master compares
    against --margin.
    """
    import tuning_rosters as rosters

    src = os.path.join(OUT_DIR, 'best_by_pairing.json')
    if not os.path.exists(src):
        # Tagged sweeps write best_by_pairing<tag>.json and deliberately leave
        # this path alone, so several of them finishing is not the same as this
        # file existing. Say which mode makes it rather than sending the author
        # back to re-run hours of search that are already on disk.
        import glob
        tagged = sorted(os.path.basename(f) for f in
                        glob.glob(os.path.join(OUT_DIR, 'best_by_pairing_*.json')))
        hint = (f' {len(tagged)} tagged sweep result(s) are already here '
                f'({", ".join(tagged)}); run --merge to combine them into it.'
                if tagged else ' Run --sweep first.')
        print(f'{src} does not exist --{hint}')
        return 1
    with open(src) as fh:
        payload = json.load(fh)

    pat, M = _import_pipeline()
    # Per-run too: two runs sharing one scratch file is how the column-mismatch
    # error appears, and it is the same collision as the timing file.
    scratch_csv = os.path.join(OUT_DIR, f'scratch_rows{args_cli.tag}.csv')
    smiles, y, train_idx, val_idx, test_idx = build_split(
        pat, args_cli.sample_size, args_cli.seed)
    print(f'  train {len(train_idx)}  validation {len(val_idx)}  '
          f'test {len(test_idx)}  (the search never saw test)', flush=True)

    out = os.path.join(OUT_DIR, 'confirmation.csv')
    fields = ['model', 'rep', 'default_test_r2', 'tuned_test_r2', 'test_delta',
              'default_val_r2', 'tuned_val_r2', 'val_delta', 'search_optimism',
              'status', 'detail', 'sample_size', 'seed', 'written']
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    fh_out = open(out, 'w', newline='')
    writer = csv.DictWriter(fh_out, fieldnames=fields)
    writer.writeheader()

    cache = {}
    rows = []
    pairs = sorted(payload['pairings'].items())
    if args_cli.models:
        pairs = [(k, v) for k, v in pairs if k.split('|')[0] in args_cli.models]
    for i, (pair, rec) in enumerate(pairs, 1):
        model_label, rep = pair.split('|')
        if rec.get('best_params') is None:
            continue
        if rep not in cache:
            cache[rep] = prepared_data(pat, rep, smiles, y, train_idx, val_idx,
                                       test_idx, args_cli.sample_size,
                                       args_cli.seed)[0]
        data = cache[rep]

        row = dict(model=model_label, rep=rep, status='ok', detail='',
                   default_val_r2=rec.get('default_r2', ''),
                   tuned_val_r2=rec.get('best_r2', ''))
        try:
            d_test, _ = fit_once(pat, M, model_label, rep, data, None,
                                 args_cli.sample_size, scratch_csv, rosters,
                                 score_on='test')
            t_test, _ = fit_once(pat, M, model_label, rep, data,
                                 rec['best_params'], args_cli.sample_size,
                                 scratch_csv, rosters, score_on='test')
            row['default_test_r2'] = d_test
            row['tuned_test_r2'] = t_test
            row['test_delta'] = t_test - d_test
            if rec.get('best_r2') is not None and rec.get('default_r2') is not None:
                row['val_delta'] = rec['best_r2'] - rec['default_r2']
                # How much of the search's apparent gain does not survive the
                # move to a split the search did not choose on.
                row['search_optimism'] = row['val_delta'] - row['test_delta']
        except Exception as exc:
            row.update(status='error', detail=f'{type(exc).__name__}: {exc}',
                       default_test_r2='', tuned_test_r2='', test_delta='')
            if args_cli.traceback:
                traceback.print_exc()

        print(f'[{i}/{len(pairs)}] {model_label} x {rep}  '
              f'default {row["default_test_r2"]}  tuned {row["tuned_test_r2"]}  '
              f'delta {row["test_delta"]}  {row["status"]}', flush=True)
        row.update(sample_size=args_cli.sample_size, seed=args_cli.seed,
                   written=stamp)
        writer.writerow(row)
        fh_out.flush()
        rows.append(row)

    fh_out.close()
    print(f'\nwrote {out}  ({len(rows)} pairings)')

    good = [r for r in rows if r['status'] == 'ok']
    won = [r for r in good if r['test_delta'] > 0]
    print(f'  tuned beats the default on test: {len(won)} of {len(good)} pairings')
    survived = [r for r in good
                if r.get('val_delta') not in ('', None)
                and r['val_delta'] > 0 and r['test_delta'] <= 0]
    if survived:
        print(f'  won the search and lost on test: {len(survived)} pairing(s) '
              f'-- these are the ones the old path would have adopted:')
        for r in survived:
            print(f'      {r["model"]} x {r["rep"]}  '
                  f'validation {r["val_delta"]:+.4f}  test {r["test_delta"]:+.4f}')
    return 0


def write_master(args_cli):
    """Turn best_by_pairing.json into the two files the pipeline reads.

    Only keys that address exactly one model are written. A value written under
    a key shared by several models would change models it was never measured on
    -- see models/tuning_rosters.py.
    """
    import tuning_rosters as rosters

    src = os.path.join(OUT_DIR, 'best_by_pairing.json')
    if not os.path.exists(src):
        # Tagged sweeps write best_by_pairing<tag>.json and deliberately leave
        # this path alone, so several of them finishing is not the same as this
        # file existing. Say which mode makes it rather than sending the author
        # back to re-run hours of search that are already on disk.
        import glob
        tagged = sorted(os.path.basename(f) for f in
                        glob.glob(os.path.join(OUT_DIR, 'best_by_pairing_*.json')))
        hint = (f' {len(tagged)} tagged sweep result(s) are already here '
                f'({", ".join(tagged)}); run --merge to combine them into it.'
                if tagged else ' Run --sweep first.')
        print(f'{src} does not exist --{hint}')
        return 1
    with open(src) as fh:
        payload = json.load(fh)

    # The margin is applied to the TEST-split difference, never the search's.
    # A winner is the maximum of N noisy validation numbers, so its validation
    # margin is biased upward by construction; --confirm measures the same two
    # settings on a split the search never chose on. Without that file there is
    # nothing to adopt ON, so this refuses rather than falling back to the
    # search's own numbers -- which is exactly what the Optuna path did.
    conf_path = os.path.join(OUT_DIR, 'confirmation.csv')
    if not os.path.exists(conf_path):
        print(f'{conf_path} does not exist -- run --confirm first. A setting is '
              f'adopted on the test-split comparison, not on the search score '
              f'that selected it.')
        return 1
    confirmed = {}
    with open(conf_path) as fh:
        for row in csv.DictReader(fh):
            if row['status'] == 'ok' and row['test_delta'] not in ('', None):
                confirmed[(row['model'], row['rep'])] = float(row['test_delta'])

    margin = args_cli.margin
    collapsed = rosters.collapsed_models()
    master, decisions, skipped = {}, {}, []
    for pair, rec in payload['pairings'].items():
        model_label, rep = pair.split('|')
        if collapsed.get(model_label):
            skipped.append((pair, f'key {rosters.TUNED_KEY[model_label]!r} is '
                                  f'shared with {collapsed[model_label]}'))
            continue
        if rec['best_r2'] is None or rec['default_r2'] is None:
            skipped.append((pair, 'no scored setting'))
            continue
        delta = confirmed.get((model_label, rep))
        if delta is None:
            skipped.append((pair, 'not confirmed on the test split'))
            continue
        if delta < margin:
            skipped.append((pair, f'test-split gain {delta:+.4f} below the '
                                  f'margin {margin}'))
            continue
        key = rosters.TUNED_KEY[model_label]
        master.setdefault(key, {})[rep] = rec['best_params']
        decisions[key] = 'USE_TUNED'

    results = os.path.join(_ROOT, 'results')
    with open(os.path.join(results, 'master_tuned_hyperparameters.json'), 'w') as fh:
        json.dump(master, fh, indent=2)
    with open(os.path.join(results, 'hyperparameter_decisions.json'), 'w') as fh:
        json.dump(decisions, fh, indent=2)
    print(f'wrote results/master_tuned_hyperparameters.json '
          f'({sum(len(v) for v in master.values())} pairings, '
          f'{len(master)} model keys)')
    print(f'wrote results/hyperparameter_decisions.json ({len(decisions)} USE_TUNED)')
    for pair, why in skipped:
        print(f'  not written: {pair} -- {why}')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--time', action='store_true',
                      help='Fit every pairing ONCE at its shared default and '
                           'record the seconds. Prices the sweep.')
    mode.add_argument('--sweep', action='store_true',
                      help='The random search itself.')
    mode.add_argument('--merge', action='store_true',
                      help='Combine the per-task trials files from a cluster '
                           'sweep into one best_by_pairing.json, and name any '
                           'pairing that produced no rows.')
    mode.add_argument('--confirm', action='store_true',
                      help='Refit the winner and the shared default and score '
                           'both on the TEST split, which the search never saw. '
                           'This is what --write-master decides on.')
    mode.add_argument('--write-master', action='store_true',
                      help='Write results/master_tuned_hyperparameters.json and '
                           'results/hyperparameter_decisions.json from the sweep.')
    ap.add_argument('--settings', type=int, default=12,
                    help='Random settings per pairing, on top of the default '
                         '(default 12).')
    ap.add_argument('--sample-size', type=int, default=10000,
                    help='Molecules, before the 80/10/10 scaffold split. The QM9 '
                         'jobs run 10000, which is the default here too.')
    ap.add_argument('--seed', type=int, default=RANDOM_SEED)
    ap.add_argument('--models', nargs='*', help='Subset of model labels.')
    ap.add_argument('--reps', nargs='*', help='Subset of representations.')
    ap.add_argument('--dataset', default='qm9',
                    choices=('qm9',) + VALIDATION_DATASETS,
                    help='Which dataset to tune ON. Hyperparameters are '
                         'dataset-specific: a setting chosen on QM9 has no '
                         'claim on hERG, so each is tuned on its own data.')
    ap.add_argument('--openadmet-csv', default=None,
                    help='Local OpenADMET table for logd and caco2, if it is '
                         'not to be downloaded.')
    ap.add_argument('--screen-at', type=int, default=None,
                    help='Fit every candidate on this many TRAINING molecules '
                         'first, scored on the same validation split, then '
                         'refit only the best --promote of them on the whole '
                         'training set. The default setting is never screened.')
    ap.add_argument('--promote', type=int, default=4,
                    help='How many screened settings are refitted at full size '
                         '(default 4; only used with --screen-at).')
    ap.add_argument('--margin', type=float, default=0.0,
                    help='Adopt a tuned setting only if it beats the shared '
                         'default by at least this much R-squared ON THE TEST '
                         'SPLIT, as measured by --confirm (--write-master only).')
    ap.add_argument('--noise-level', type=float, default=None,
                    help='Recorded in the output. Tuning runs on CLEAN labels '
                         'unless this is set; the injector is not wired in here '
                         'yet, so a value only labels the run.')
    ap.add_argument('--tag', default='',
                    help='Suffix for this run\'s output files, so concurrent '
                         'tasks do not share a path. The cluster generator sets '
                         'it per array task; --merge reads them all back.')
    ap.add_argument('--traceback', action='store_true')
    args_cli = ap.parse_args()
    if args_cli.tag and not args_cli.tag.startswith('_'):
        args_cli.tag = '_' + args_cli.tag
    # A NARROWED RUN NEVER SHARES THE FULL RUN'S FILES.
    #
    # Measured the hard way on 2026-08-28: a one-pairing --time run with no tag
    # opened results/tuning_local/timing.csv with 'w' while the full 80-pairing
    # pass was six hours into writing that same path. The truncation destroyed
    # every row on disk, and the long run -- still holding its own handle at its
    # old offset -- carried on writing past a block of NULs. Only the log saved
    # it. So a run restricted to particular models or representations gets its
    # own filename whether or not anyone remembered to ask for one.
    if args_cli.dataset != 'qm9' and not args_cli.tag:
        args_cli.tag = '_' + args_cli.dataset
    if not args_cli.tag and (args_cli.models or args_cli.reps):
        parts = list(args_cli.models or []) + list(args_cli.reps or [])
        args_cli.tag = '_subset_' + '-'.join(parts)[:60]
        print(f"  [tag] restricted to {parts}, so this run writes "
              f"{args_cli.tag[1:]} files rather than the full run's.", flush=True)

    if args_cli.write_master:
        return write_master(args_cli)
    if args_cli.merge:
        return merge(args_cli)
    if args_cli.confirm:
        return confirm(args_cli)
    return run(args_cli)


if __name__ == '__main__':
    sys.exit(main())
