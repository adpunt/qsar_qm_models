#!/usr/bin/env python3
"""Prove the two pipelines build the same models, in the environment that will run them.

WHY THIS EXISTS
---------------
The project has two independent training pipelines:

  * QM9                    qsar_qm_models/models/models.py
  * experimental datasets  KIRBy/tests/alternative_data_noise_robustness.py

They are meant to run the same models. On 2026-08-25 they did not. The quantile
forest had three times as many trees on one side; XGBoost was training at three
times the learning rate on the other, because one pipeline pinned the rate and
the other left it unset -- and leaving it unset does NOT give the value the
scikit-learn wrapper documents, it falls through to the booster's own 0.3.

THE FIRST TWO GUARDS BOTH FAILED, AND IT IS WORTH SAYING WHY
------------------------------------------------------------
The first was a table in a document. It went stale.

The second was this script, restating both pipelines' parameters as hand-typed
literals. It went stale inside twenty-four hours: the experimental column still
claimed 100 trees and an unset learning rate after both had been fixed, so
running it reported differences that no longer existed. A checker that restates
what it checks is a third copy to keep in sync.

WHAT IT DOES NOW
----------------
Both pipelines READ their defaults from models/model_defaults.py. This script:

  1. loads that spec through BOTH pipelines -- importing models.py the way QM9
     does, and loading the experimental module from its own file the way the
     preflight does -- and fails if the two do not resolve to the same spec
     hash. That is the parity check, and it cannot go stale, because there is
     only one copy of the numbers.
  2. builds every model from that spec WITH THE INSTALLED LIBRARIES and diffs
     the effective parameters against scripts/model_params_baseline.json. This
     catches library-default drift: an upgrade that moves a default we do not
     pin shows up here rather than as a changed result six months later.
  3. asserts the environment facts that are not estimator parameters -- that
     ngboost's MLE is still LogScore, that natural_gradient still defaults to
     True, and whether botorch's fitter accepts a plain gpytorch ExactGP.
  4. reports every package version, because a run whose environment is missing
     a model family produces a subset of the intended experiments -- or, as
     happened on 2026-08-19, nothing at all with only a warning on page one.
  5. prints the differences no automated diff can see, with the file and line.

USAGE
-----
    python scripts/audit_pipeline_parity.py                  # report
    python scripts/audit_pipeline_parity.py --strict         # exit 1 on any problem
    python scripts/audit_pipeline_parity.py --self-test      # prove --strict bites
    python scripts/audit_pipeline_parity.py --write-baseline # after a deliberate change
    python scripts/audit_pipeline_parity.py --compare-results a.csv b.csv --strict

Run --strict under EVERY interpreter that will run jobs. It is wired into
slurm_scripts_uncertainty_rerun/preflight.sh as check 0.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO / 'scripts' / 'model_params_baseline.json'

# Parameters that may differ without changing the fit.
BENIGN = {'random_state', 'n_jobs', 'verbose', 'verbosity', 'nthread', 'seed',
          'silent', 'device', 'n_threads', 'importance_type'}


def _is_benign(key):
    """A nested key is benign when its LAST component is.

    ngboost 0.3.12 inherits scikit-learn's deep get_params, so it hands back
    `Base__random_state` as ONE flat key. `'Base__random_state' not in BENIGN`
    is true, so the seed this script itself puts on the base tree was recorded
    as a real parameter and reported DRIFTED against a baseline written on a
    machine whose ngboost (0.5.5) does not expand nested keys. Measured
    2026-08-29: two of the five problems blocking preflight check 0 here were
    that seed and a scikit-learn version difference, neither of which changes a
    fit. A benign parameter has to stay benign wherever it appears.
    """
    return key in BENIGN or key.rsplit('__', 1)[-1] in BENIGN


# Every package a model family needs. 'bayesian_torch' used to be listed here
# and is imported by NOTHING in either pipeline -- both use torchbnn+torchhk --
# so it reported a missing package that no job has ever needed.
PACKAGES = ['sklearn', 'xgboost', 'lightgbm', 'ngboost', 'quantile_forest',
            'gpytorch', 'gauche', 'botorch', 'torch', 'torchbnn', 'torchhk',
            'rdkit', 'numpy', 'pandas', 'scipy']

# How to build each shared model. (spec key, module, class, extra kwargs the
# constructor needs that are not parameters we pin.)
BUILDERS = [
    ('rf', 'sklearn.ensemble', 'RandomForestRegressor', {}),
    ('qrf', 'quantile_forest', 'RandomForestQuantileRegressor', {}),
    ('xgboost', 'xgboost', 'XGBRegressor', {}),
    ('lightgbm', 'lightgbm', 'LGBMRegressor', {}),
    ('ngboost', 'ngboost', 'NGBRegressor', {}),
    ('svm', 'sklearn.svm', 'SVR', {}),
]

# Real differences that are not estimator parameters.
#
# WHY EVERY ENTRY NOW CARRIES A PROBE
# -----------------------------------
# This list used to be prose alone, and prose goes stale the moment a fix lands.
# On 2026-08-29 four of its eight entries described defects that had been fixed
# weeks earlier, and preflight check 0 prints this list on EVERY run, so the last
# thing an operator read before submitting was four untrue statements:
#
#   * "ECFP4 is not a circular fingerprint". It is. scripts/process_and_train.py
#     ecfp4_fingerprint() calls GetMorganGenerator(radius=2, fpSize=2048) in
#     Python and the Rust record carries the 256 packed bytes untouched.
#   * "the descriptor vector is binarised at (pdv > 0)". It is 200 float32
#     values in an 800-byte slot; the binary form was DELETED on 2026-08-28.
#   * "Sort & Slice counts are discarded by packbits, and a uint8 cast wraps".
#     The counts are stored as uint16 and the writer refuses anything that will
#     not fit.
#   * "learned embeddings are per-molecule min-max scaled into uint8". They are
#     float32, standardised per feature on the training split.
#
# Each one named the laboratory pipeline as the side that was right, so the
# report told the reader the two pipelines still disagreed about four of the six
# representations. They agree. Worse, the four false entries buried the ones that
# are real.
#
# So an entry is a dict now, and `probe` is a function that reads TODAY'S source
# and answers "is this still true?". It returns (True, evidence) if the
# difference is still there, (False, evidence) if it has been fixed and the entry
# should be deleted, or None if it cannot tell. An entry with no probe prints as
# unchecked rather than as fact. Restating is what went stale; reading does not.
QM9_SOURCE = REPO / 'scripts' / 'process_and_train.py'


def _kirby_file(*parts):
    """The laboratory checkout, or None. Probes must degrade, not crash."""
    root = os.environ.get('KIRBY_ROOT')
    for base in ([Path(root)] if root else []) + [REPO.parent / 'KIRBy']:
        p = base.joinpath(*parts)
        if p.exists():
            return p
    return None


def _probe_chemberta_tokenizer():
    """One encoder, two tokenizers — measured on the laboratory side, which
    refuses a tokenizer that merges two molecules onto one vector."""
    lab = _kirby_file('src', 'kirby', 'representations', 'molecular.py')
    if lab is None:
        return None, 'laboratory checkout not found'
    qm9 = QM9_SOURCE.read_text()
    lab_src = lab.read_text()
    qm9_fast = 'AutoTokenizer.from_pretrained' in qm9
    lab_slow = 'RobertaTokenizer.from_pretrained' in lab_src
    if qm9_fast and lab_slow:
        return True, ('QM9 loads AutoTokenizer (resolves to RobertaTokenizerFast), '
                      'the laboratory loads RobertaTokenizer')
    return False, (f'QM9 AutoTokenizer={qm9_fast}, laboratory '
                   f'RobertaTokenizer={lab_slow} — the two now load the same '
                   f'tokenizer class, so DELETE this entry')


def _probe_stereochemistry():
    """The strings the two pipelines featurise are not the same strings."""
    lab = _kirby_file('tests', 'alternative_data_noise_robustness.py')
    if lab is None:
        return None, 'laboratory checkout not found'
    qm9_strips = 'Chem.MolToSmiles(mol, isomericSmiles=False)' in QM9_SOURCE.read_text()
    lab_keeps = 'return Chem.MolToSmiles(mol, canonical=True)' in lab.read_text()
    if qm9_strips and lab_keeps:
        return True, ('QM9 canonicalises with isomericSmiles=False, '
                      'standardise_smiles keeps RDKit\'s default')
    return False, (f'QM9 strips stereochemistry={qm9_strips}, laboratory keeps '
                   f'it={lab_keeps} — DELETE this entry if both now agree')


MANUAL = [
    # Added 2026-08-29, replacing four entries that had been fixed. Both sides
    # load DeepChem/ChemBERTa-77M-MTR at 384 dimensions -- that half was settled
    # on 2026-08-27 -- but they do not load the same tokenizer, and the
    # laboratory side's own _chemberta_tokenizer_report measures what that costs.
    {'title': 'ChemBERTa tokenizer — the encoder matches, the tokenizer does not',
     'qm9': 'AutoTokenizer, which resolves to the FAST RobertaTokenizerFast; its '
            'tokenizer.json carries no unknown token, so a symbol outside the '
            'vocabulary is DELETED rather than substituted',
     'experimental': 'RobertaTokenizer (slow), which substitutes <unk>',
     'see': 'scripts/process_and_train.py get_chemberta_model  vs  '
            'KIRBy/src/kirby/representations/molecular.py _chemberta_tokenizer_report',
     'matters': 'A ChemBERTa row from one pipeline and a ChemBERTa row from the '
                'other are the same network reading two different token streams. '
                'The laboratory side measured it on 400 hERG molecules: '
                'embeddings differ by up to 3.03, 14.8x the mean spread of a '
                'coordinate, and six pairs of DISTINCT molecules collapsed onto '
                'one byte-identical vector under the QM9 tokenizer. Which side '
                'changes is an author decision.',
     'probe': _probe_chemberta_tokenizer},

    # Added 2026-08-29. Also an author decision: the two candidate fixes change
    # different halves of the study.
    {'title': 'Stereochemistry — the two pipelines featurise different strings',
     'qm9': "Chem.MolToSmiles(mol, isomericSmiles=False), so @/@@ and E/Z are "
            "gone before anything is computed",
     'experimental': "standardise_smiles canonicalises with RDKit's default, so "
                     "stereochemistry survives into every representation",
     'see': 'scripts/process_and_train.py  vs  '
            'KIRBy/tests/alternative_data_noise_robustness.py standardise_smiles',
     'matters': 'The four static representations do not care -- the laboratory '
                'side measured ECFP4, Avalon, Sort & Slice and the descriptor '
                'vector bit-identical either way on 100 stereo-bearing '
                'molecules. MHG-GNN does: max|diff| 40.0 against a feature '
                'spread of 11.6, every row changed. So a cross-pipeline sentence '
                'about a learned embedding compares a stereo-aware '
                'featurisation with a stereo-blind one. 24.8% of the laboratory '
                'SMILES carry stereochemistry.',
     'probe': _probe_stereochemistry},

    # The four entries below predate the probe machinery and are carried
    # unchanged. They are NOT machine-checked: the prose is the 2026-08-26/27
    # reading and nothing here re-reads it, which is exactly the condition that
    # let the representation entries rot. Give each one a probe when someone next
    # verifies it by hand.
    # REMOVED 2026-08-29 (author's instruction). 'Validation split -- merged into
    # training for seven model families' was closed on 2026-08-27: no model stacks
    # validation into its training set on either side now (models/models.py,
    # `x_val_train = x_train` in both branches). Printing it as a live difference
    # told every reader that a fixed defect was still open.
    {'title': 'Repetitions',
     'qm9': '10 independent fits per cell, each with its own seed (-b/--repetitions)',
     'experimental': 'ONE fit per cell, seed pinned to 42',
     'see': 'scripts/process_and_train.py:247 vs KIRBy constructors',
     'matters': "Author's decision 2026-08-26: keep one fit on the experimental "
                'side. The experimental ANOVA therefore has no estimable '
                'residual term and no run-to-run error bar. The paper must say so.',
     'probe': None},
    {'title': 'Label standardisation',
     'qm9': 'all models, in Rust, using the CLEAN training mean and SD since 2026-08-26',
     'experimental': 'neural models only, scaler fitted on the noisy training labels',
     'see': 'rust/src/main.rs vs KIRBy train_neural_regression',
     'matters': 'Tree and kernel models on the experimental side see raw labels. '
                'Scale-invariant for trees; NOT for the SVM or the GP.',
     'probe': None},
    {'title': 'Uncertainty calibration',
     'qm9': 'a temperature fitted on a carve-out of validation',
     'experimental': 'absent',
     'see': 'scripts/utils.py:323 and the models.py call sites',
     'matters': 'Decided by scripts/parity_test_calibration.py; the column the '
                'analysis reads is '
                'model_defaults.UNCERTAINTY_DEFAULTS["primary_column"]. Whatever '
                'it is, the figure script must NAME it — today it silently '
                'prefers the calibrated column wherever one exists (Chat J).',
     'probe': None},
]


# ---------------------------------------------------------------------------
# 1. one spec, reached through both pipelines
# ---------------------------------------------------------------------------

def _load_experimental_module(path=None):
    """Load the experimental pipeline from its own file, the way preflight.sh
    does. Returns (module, path) or (None, reason)."""
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get('KIRBY_ROOT')
    if env:
        candidates.append(Path(env) / 'tests' / 'alternative_data_noise_robustness.py')
    candidates += [
        REPO.parent / 'KIRBy' / 'tests' / 'alternative_data_noise_robustness.py',
        Path('/data/stat-cadd/scat9264/KIRBy/tests/alternative_data_noise_robustness.py'),
        Path('/data/stat-ecr/scat9264/KIRBy/tests/alternative_data_noise_robustness.py'),
        Path.cwd() / 'alternative_data_noise_robustness.py',
    ]
    for c in candidates:
        if c.exists():
            spec = importlib.util.spec_from_file_location('_exp_pipeline', c)
            mod = importlib.util.module_from_spec(spec)
            cwd = Path.cwd()
            try:
                os.chdir(c.parent)      # it reads data_cache/ relative to itself
                spec.loader.exec_module(mod)
            finally:
                os.chdir(cwd)
            return mod, str(c)
    return None, ('experimental pipeline not found; looked in '
                  + ', '.join(str(c) for c in candidates))


def audit_spec_parity(exp_path=None, skip_experimental=False):
    """The parity check proper: both pipelines must resolve to ONE spec."""
    print('=' * 78)
    print('SPEC PARITY — both pipelines must resolve to the same shared spec')
    print('=' * 78)
    problems = 0

    sys.path.insert(0, str(REPO / 'models'))
    import model_defaults as canonical
    print(f'  canonical    {REPO / "models" / "model_defaults.py"}')
    print(f'               version {canonical.SPEC_VERSION}  hash {canonical.spec_hash()}')

    # -- through the QM9 pipeline -------------------------------------------
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        qm9 = importlib.import_module('models')
        qm9_hash = qm9.spec_hash()
        status = 'MATCH' if qm9_hash == canonical.spec_hash() else 'DIFFERS'
        print(f'  QM9          models/models.py -> hash {qm9_hash}   {status}')
        if status != 'MATCH':
            problems += 1
    except Exception as exc:
        print(f'  QM9          COULD NOT IMPORT models/models.py: '
              f'{type(exc).__name__}: {exc}')
        print('               (a missing package here means QM9 jobs will skip '
              'model families)')
        problems += 1

    # -- through the experimental pipeline ----------------------------------
    if skip_experimental:
        print('  experimental SKIPPED (--skip-experimental)')
        return problems

    mod, where = _load_experimental_module(exp_path)
    if mod is None:
        print(f'  experimental NOT CHECKED: {where}')
        print('               set KIRBY_ROOT, or run this from the KIRBy tests '
              'directory')
        problems += 1
    else:
        exp_hash = mod.MODEL_SPEC.spec_hash()
        status = 'MATCH' if exp_hash == canonical.spec_hash() else 'DIFFERS'
        print(f'  experimental {where}')
        print(f'               -> hash {exp_hash}   {status}')
        if status != 'MATCH':
            problems += 1
            print('               ^ the experimental pipeline resolved a '
                  'DIFFERENT copy of model_defaults.py.')
            print('                 Check QSAR_QM_MODELS_ROOT — there is more '
                  'than one checkout on this host.')
    return problems


# ---------------------------------------------------------------------------
# 2. build every model with the installed libraries
# ---------------------------------------------------------------------------

def _constructor_params(est):
    """Every constructor argument the estimator is actually holding.

    sklearn's own get_params() is generated from the __init__ signature, so it
    is complete. A library that hand-writes get_params() can silently omit its
    own settings: ngboost 0.5.5 does exactly that (ngboost/ngboost.py:495), and
    the ten keys it returns leave out `tol`, `validation_fraction`,
    `verbose_eval` and `early_stopping_rounds` -- four settings its fit() reads.
    An audit that trusted get_params() would report NGBoost as fully checked
    while blind to the stage-selection settings the two pipelines disagree
    about. Recover anything the signature names and the instance is holding.
    """
    params = dict(est.get_params()) if hasattr(est, 'get_params') else {}
    try:
        names = inspect.signature(type(est).__init__).parameters
    except (TypeError, ValueError):
        return params
    for name in names:
        if name in ('self', 'args', 'kwargs') or name in params:
            continue
        if hasattr(est, name):
            params[name] = getattr(est, name)
    return params


def _effective_params(mod_name, cls_name, kwargs, fit_time=None):
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return None, f'{cls_name} not found in {mod_name}'
    try:
        est = cls(**kwargs)
        params = _constructor_params(est) or dict(kwargs)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    out = {k: _jsonable(v) for k, v in params.items() if not _is_benign(k)}
    # A nested estimator collapses to '<DecisionTreeRegressor>' under _jsonable,
    # which would hide exactly the drift the spec pins it against: ngboost's
    # get_params() is NOT sklearn-deep, so `Base__max_depth` is not in `params`
    # (measured on ngboost 0.5.5 -- the keys are Base, Dist, Score and the seven
    # scalars, nothing prefixed). Expand one level by hand so a library upgrade
    # moving the base learner's depth still shows up as DRIFTED.
    for name, value in list(params.items()):
        if _is_benign(name) or not hasattr(value, 'get_params'):
            continue
        for sub, sub_value in value.get_params().items():
            if _is_benign(f'{name}__{sub}'):
                continue
            out[f'{name}__{sub}'] = _jsonable(sub_value)
    # Settings the spec pins that are applied at fit()/predict() time rather
    # than passed to the constructor. They are not in get_params() and never
    # will be, so record them here or the spec can move without the audit
    # noticing -- see _build_kwargs for which ones and why.
    for name, value in (fit_time or {}).items():
        out[f'fit__{name}'] = _jsonable(value)
    return out, None


def _jsonable(v):
    # NaN never equals itself, so an unconverted NaN (XGBoost's `missing`)
    # reports as drifted on every single run. Map it to a comparable token.
    if isinstance(v, float) and v != v:
        return '<nan>'
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return f'<{type(v).__name__}>'


def _build_kwargs(spec_key, canonical):
    """Return (constructor kwargs, fit-time settings) for one spec entry."""
    kwargs = dict(canonical.SKLEARN_DEFAULTS[spec_key])
    fit_time = {}
    if spec_key == 'ngboost':
        # Names, not classes, in the spec. Resolve them the way both pipelines do.
        from ngboost.distns import Normal
        from ngboost.scores import MLE
        from sklearn.tree import DecisionTreeRegressor
        kwargs['Dist'] = {'Normal': Normal}[kwargs.pop('dist')]
        kwargs['Score'] = {'MLE': MLE}[kwargs.pop('score')]
        # The base learner gets the same treatment as Dist and Score: the spec
        # names it in PARTS because it is a constructed object rather than a
        # literal, and the caller builds it. Both pipelines do exactly this --
        # models/models.py train_ngboost_model and KIRBy's _ngboost_params --
        # so handing `base_max_depth` straight to NGBRegressor, as this function
        # used to, raised TypeError and reported that every NGBoost job would
        # die on contact. They would not: no pipeline builds it that way.
        kwargs['Base'] = DecisionTreeRegressor(
            max_depth=kwargs.pop('base_max_depth'),
            criterion=kwargs.pop('base_criterion'),
            random_state=42)
        # Stage selection is a fit()/predict() concern, not a constructor
        # argument: `early_stopping_rounds` goes to fit and `use_best_iteration`
        # decides whether predict is capped at the validation optimum. Both
        # pipelines pop them here for the same reason. They stay in the spec so
        # the difference between the two pipelines stays visible -- QM9 applies
        # them, the laboratory pipeline does not (RERUN_PLAN.md 5.7k). Carry
        # them out as fit-time settings rather than dropping them: they are the
        # two the pipelines disagree about, so they are the last two the audit
        # should stop watching.
        for name in ('early_stopping_rounds', 'use_best_iteration'):
            if name in kwargs:
                fit_time[name] = kwargs.pop(name)
    return kwargs, fit_time


def audit_effective_params(write_baseline=False):
    """Build each model with the installed libraries and diff against the
    recorded baseline. This is the library-drift check."""
    print('\n' + '=' * 78)
    print('EFFECTIVE PARAMETERS — built with the libraries installed here')
    print('=' * 78)
    sys.path.insert(0, str(REPO / 'models'))
    import model_defaults as canonical

    baseline = {}
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
    elif not write_baseline:
        print(f'  no baseline at {BASELINE_FILE}')
        print('  run with --write-baseline once, then commit it')

    problems, current = 0, {}
    for spec_key, mod_name, cls_name, _extra in BUILDERS:
        try:
            kwargs, fit_time = _build_kwargs(spec_key, canonical)
        except Exception as exc:
            print(f'\n{spec_key}\n  SKIPPED — {type(exc).__name__}: {exc}')
            continue
        params, err = _effective_params(mod_name, cls_name, kwargs, fit_time)
        print(f'\n{spec_key}  ({mod_name}.{cls_name})')
        if params is None:
            print(f'  CANNOT BUILD — {err}')
            print('  ^ every job requesting this model in this interpreter '
                  'would fail on contact')
            problems += 1
            continue
        current[spec_key] = params
        pinned = {k for k in kwargs if not _is_benign(k)}
        pinned |= {f'fit__{k}' for k in fit_time}
        print(f'  pinned by the spec: {len(pinned)}   '
              f'effective parameters: {len(params)}')

        was = baseline.get(spec_key)
        if was is None:
            print('  (no baseline recorded for this model yet)')
            continue
        diffs = [(k, was.get(k, '<absent>'), params.get(k, '<absent>'))
                 for k in sorted(set(was) | set(params))
                 if was.get(k, '<absent>') != params.get(k, '<absent>')]
        if not diffs:
            print('  MATCHES BASELINE')
        else:
            problems += len(diffs)
            for k, a, b in diffs:
                marker = '  PINNED  ' if k in pinned else '  DRIFTED '
                print(f'{marker}{k:24s} baseline={a!r:>16}   now={b!r}')
            print('  ^ a DRIFTED parameter means the installed library changed a '
                  'default we do not pin.')
            print('    Decide whether to pin it in model_defaults.py, then '
                  '--write-baseline.')

    if write_baseline:
        merged = dict(baseline)
        merged.update(current)
        BASELINE_FILE.write_text(json.dumps(merged, indent=2, sort_keys=True) + '\n')
        print(f'\n  wrote {BASELINE_FILE}')
        return 0
    return problems


# ---------------------------------------------------------------------------
# 3. environment facts that are not estimator parameters
# ---------------------------------------------------------------------------

def audit_assumptions():
    print('\n' + '=' * 78)
    print('ASSUMPTIONS — things the parameter diff cannot see')
    print('=' * 78)
    problems = 0

    def check(label, fn, why, fatal=True):
        """fatal=False reports a fact that is not a parity failure -- it still
        has to be visible, but it must not block a queue on its own."""
        nonlocal problems
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f'{type(exc).__name__}: {exc}'
        tag = 'OK  ' if ok else ('FAIL' if fatal else 'NOTE')
        print(f'  {tag}  {label}: {detail}')
        if not ok:
            if fatal:
                problems += 1
            print(f'        {why}')

    def _ngb_score():
        from ngboost.scores import MLE, LogScore
        return MLE is LogScore, f'MLE is LogScore = {MLE is LogScore}'
    check('ngboost MLE is LogScore', _ngb_score,
          'QM9 passes Score=MLE. If the alias moved, the two pipelines now '
          'optimise different scores under one name.')

    def _ngb_natgrad():
        import inspect
        from ngboost import NGBRegressor
        d = inspect.signature(NGBRegressor.__init__).parameters['natural_gradient'].default
        return d is True, f'default natural_gradient = {d}'
    check('ngboost natural_gradient defaults True', _ngb_natgrad,
          'The shared spec pins it, so this is informational — but a changed '
          'default means older results were fitted differently.')

    def _botorch_fitter():
        import gpytorch
        import numpy as np
        import torch
        try:
            from botorch.fit import fit_gpytorch_mll as fit
        except ImportError:
            from botorch import fit_gpytorch_model as fit

        class _GP(gpytorch.models.ExactGP):
            def __init__(self, x, y, lik):
                super().__init__(x, y, lik)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel())

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x))

        rng = np.random.RandomState(0)
        x = torch.from_numpy(rng.normal(size=(40, 3)))
        y = torch.from_numpy(rng.normal(size=40))
        lik = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
        m = _GP(x, y, lik)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, m)
        try:
            fit(mll)
            return True, 'botorch fits a plain gpytorch ExactGP'
        except Exception as exc:
            return False, (f'botorch REFUSES a plain ExactGP '
                           f'({type(exc).__name__}), so the experimental GP will '
                           f'use its Adam fallback')
    check('botorch fitter accepts the GP class', _botorch_fitter,
          'Not a parity failure: BOTH pipelines now share the same try/except '
          'and both record gp_fit_method, so they agree either way. But this '
          'environment will fit every GP with the Adam fallback rather than '
          'botorch, which is a DIFFERENT optimisation from the one the earlier '
          'GP results used. Check the gp_fit_method column before comparing new '
          'GP rows against old ones.', fatal=False)

    # REMOVED 2026-08-29 (author's instruction). It searched job scripts for
    # `use_best_params`; they are written `--use-best-params`, so it matched
    # nothing, reported OK on every run and always would have. Correcting the
    # spelling was not the fix either: the QM9 generator now passes that flag
    # deliberately whenever the two tuned files are present, so a guard
    # asserting that no job script passes it asserts the opposite of the design.
    # What keeps the screen on shared defaults is the runbook's step of moving
    # the tuned files aside, not a check here.

    def _gp_after_boosting():
        """Does a Gaussian process still fit once the boosting libraries are loaded?

        On this laptop it does not: importing lightgbm or xgboost and then
        fitting a plain gpytorch ExactGP SEGFAULTS (exit 139). Both link their
        own OpenMP runtime and PyTorch links another; loading two and then
        running a threaded kernel kills the process.

        This has to be a subprocess. A segfault cannot be caught in-process --
        which is exactly why it is dangerous: in a SLURM array it looks like a
        task that died with no Python traceback at all.

        Both pipelines import xgboost and lightgbm at module level and both fit
        a gauche/gpytorch GP, so if this fires on the cluster, every Gaussian-
        process task dies silently.
        """
        code = (
            "import lightgbm, xgboost, numpy as np, torch, gpytorch\n"
            "class G(gpytorch.models.ExactGP):\n"
            "    def __init__(s,x,y,l):\n"
            "        super().__init__(x,y,l)\n"
            "        s.m=gpytorch.means.ConstantMean()\n"
            "        s.c=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())\n"
            "    def forward(s,x):\n"
            "        return gpytorch.distributions.MultivariateNormal(s.m(x),s.c(x))\n"
            "r=np.random.RandomState(0)\n"
            "X=r.normal(size=(900,208)); y=X[:,0]*2+r.normal(scale=.3,size=900)\n"
            "xt=torch.from_numpy(X); yt=torch.from_numpy(y)\n"
            "l=gpytorch.likelihoods.GaussianLikelihood(noise=1e-3); m=G(xt,yt,l)\n"
            "mll=gpytorch.mlls.ExactMarginalLogLikelihood(l,m)\n"
            "m.train(); l.train()\n"
            "o=torch.optim.Adam(m.parameters(),lr=0.1)\n"
            "for _ in range(20):\n"
            "    o.zero_grad(); (-mll(m(xt),yt)).backward(); o.step()\n"
            "print('ok')\n")
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        threads = os.environ.get('OMP_NUM_THREADS', 'unset')
        if r.returncode == 0:
            return True, f'fits with OMP_NUM_THREADS={threads}'
        how = ('SEGFAULT' if r.returncode in (-11, 139)
               else f'exit {r.returncode}')
        # What cures it? Try the thread pins together, not just OMP -- the
        # experimental pipeline sets MKL_NUM_THREADS as well, and pinning one
        # while the other stays at 4 does nothing. Reporting "OMP=1 does not
        # fix it" on that basis would send the author the wrong way.
        pins = [
            {'OMP_NUM_THREADS': '1'},
            {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
             'NUMEXPR_NUM_THREADS': '1'},
        ]
        cure = 'nothing tried fixes it — the environment needs one OpenMP runtime'
        for pin in pins:
            env = dict(os.environ, **pin)
            if subprocess.run([sys.executable, '-c', code],
                              capture_output=True, text=True,
                              env=env).returncode == 0:
                cure = 'fixed by ' + ' '.join(f'{k}={v}' for k, v in pin.items())
                break
        return False, (f'{how} with OMP_NUM_THREADS={threads}; {cure}')
    check('Gaussian process fits with boosting loaded', _gp_after_boosting,
          'LAUNCH BLOCKER for every Gaussian-process task on both pipelines. Both '
          'import xgboost and lightgbm at module level and then fit a '
          'gauche/gpytorch GP. A segfault leaves NO Python traceback, so in a '
          'SLURM array it looks like a task that simply died -- which is the '
          'shape of the two GP jobs in RERUN_PLAN.md 2.8d. Fix the environment '
          '(one OpenMP runtime: install xgboost/lightgbm from the same channel '
          'as pytorch) rather than pinning threads to 1, which costs every tree '
          'fit its parallelism.')

    def _rust_binary():
        p = REPO / 'rust' / 'target' / 'release' / 'rust_processor'
        return p.exists(), (str(p) if p.exists() else 'not built')
    check('rust noise processor is built', _rust_binary,
          'QM9 jobs invoke it. Chat A changed its CLI, so also run its '
          '--self-test before submitting.')
    return problems


# ---------------------------------------------------------------------------
# 4 & 5. environment, and the manual checklist
# ---------------------------------------------------------------------------

def audit_environment():
    print('\n' + '=' * 78)
    print(f'ENVIRONMENT — {sys.executable}')
    print(f'python {sys.version.split()[0]}')
    print('=' * 78)
    missing = 0
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f'  {name:18s} {getattr(mod, "__version__", "(no __version__)")}')
        except Exception as exc:
            missing += 1
            print(f'  {name:18s} MISSING ({type(exc).__name__})')
    if missing:
        print(f'\n  {missing} package(s) missing. Both pipelines now RAISE on a '
              'requested-but-unbuildable model')
        print('  rather than warning and producing nothing, but the job still '
              'dies — fix the environment first.')
    return missing


def report_manual():
    """Print the hand-written differences, each one re-checked against the source.

    Returns the number of entries whose probe says the difference is GONE. They
    are printed as out of date rather than as fact, because this section is the
    last thing an operator reads before submitting and it spent weeks describing
    four fixed representation defects as open.
    """
    print('\n' + '=' * 78)
    print('NOT VISIBLE HERE — differences that are not estimator parameters')
    print('=' * 78)
    stale = []
    for item in MANUAL:
        probe = item.get('probe')
        if probe is None:
            status, evidence = 'NOT RE-CHECKED', 'no probe — verify by hand before trusting'
        else:
            try:
                still, evidence = probe()
            except Exception as exc:
                still, evidence = None, f'probe raised {type(exc).__name__}: {exc}'
            if still is True:
                status = 'STILL OPEN'
            elif still is False:
                status = 'OUT OF DATE — FIXED, DELETE THIS ENTRY'
                stale.append(item['title'])
            else:
                status = 'COULD NOT CHECK'
        print(f"\n{item['title']}")
        print(f"  status       {status}")
        print(f"  checked      {evidence}")
        print(f"  QM9          {item['qm9']}")
        print(f"  experimental {item['experimental']}")
        print(f"  see          {item['see']}")
        print(f"  matters      {item['matters']}")
    if stale:
        print(f'\n  {len(stale)} entry(ies) above describe a difference that no '
              f'longer exists:')
        for t in stale:
            print(f'    - {t}')
        print('  Delete them from MANUAL. A checklist that restates a fixed defect '
              'is worse')
        print('  than no checklist: it sends the reader to correct code.')
    return len(stale)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def _signal_lines(out):
    """The audit's own findings, one per line, independent of the exit code.

    The self-test used to compare `returncode` alone. Whenever the audit was red
    for ANY unrelated reason -- a stale baseline, a library that moved, the
    Gaussian-process segfault -- every perturbation row also came back 1, which
    is equally consistent with "the check bit" and "it was already red", and the
    script then printed "SELF-TEST FAILED -- this audit would not catch a real
    drift". Measured 2026-08-29: the audit DID catch the drift, adding a
    `PINNED learning_rate baseline=0.1 now=0.3` line and moving the count from 4
    problems to 5. The exit code could not see it. So compare findings, not codes.
    """
    keep = []
    for line in out.splitlines():
        st = ' '.join(line.strip().split())
        if (st.startswith('PINNED') or st.startswith('DRIFTED')
                or st.startswith('CANNOT BUILD') or st.startswith('FAIL')
                or st.startswith('COULD NOT IMPORT') or 'DIFFERS' in st):
            keep.append(st)
    return keep


def _spec_parity_output(exp_path):
    """Run the cross-pipeline comparison in this process and capture it."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        problems = audit_spec_parity(exp_path=exp_path, skip_experimental=False)
    return problems, buf.getvalue()


def self_test():
    """Prove --strict actually bites. Perturbs the spec on disk, confirms the
    audit REPORTS the change, and restores it. This is the check that fails if
    the fix is removed — the standing rule in RERUN_PLAN.md section 13.2.

    Two things changed on 2026-08-29, both because the old version could not tell
    a working check from a broken environment:

      * it judges the perturbation rows on the FINDINGS the audit prints, not on
        the exit code, so a pre-existing problem no longer makes them vacuous;
      * it exercises the cross-pipeline comparison. Every run used to pass
        --skip-experimental, so the one comparison this whole script exists for
        -- both pipelines resolving the same spec -- was never tested, while
        preflight.sh check 0, the gate that actually protects the laboratory run,
        always runs WITH it.
    """
    print('=' * 78)
    print('SELF-TEST — does --strict report a perturbed spec, and does the')
    print('            cross-pipeline comparison bite?')
    print('=' * 78)
    spec_file = REPO / 'models' / 'model_defaults.py'

    # THIS FUNCTION EDITS A TRACKED SOURCE FILE IN PLACE, so only one copy of it
    # may run at a time. Two interleaved runs corrupt the spec and neither
    # notices: A reads the real file and writes the perturbed one; B then reads
    # the PERTURBED file as its "original", perturbs and restores it to that;
    # A's restore may or may not win. It happened on 2026-08-28 and left the
    # quantile forest on 100 trees where the spec says 300 -- a silent change to
    # every quantile-forest result, in the file whose whole purpose is to stop
    # silent changes. The `finally` restore below cannot help: both runs restore
    # faithfully, to different text.
    lock_path = spec_file.with_suffix('.py.selftest.lock')
    lock_handle = open(lock_path, 'w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_handle.close()
        print(f'  REFUSING TO RUN — another --self-test holds {lock_path.name}.')
        print('  This test rewrites models/model_defaults.py, so two at once')
        print('  corrupt it. Wait for the other one and run again.')
        return 1

    original = spec_file.read_text()
    failures = []

    def run_strict(label):
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--strict',
             '--skip-experimental'],
            capture_output=True, text=True, cwd=str(REPO))
        lines = _signal_lines(r.stdout)
        print(f'  {label:38s} exit {r.returncode}   {len(lines)} finding(s)')
        return r.returncode, lines

    def perturbation(label, before, after, must_mention, baseline):
        """A perturbation passes when the audit reports something NEW that names
        the parameter that moved."""
        perturbed = original.replace(before, after, 1)
        if perturbed == original:
            failures.append(f'could not perturb the spec ({before!r} is not in '
                            f'it) — the self-test is not testing anything')
            return
        spec_file.write_text(perturbed)
        try:
            rc, lines = run_strict(label)
        finally:
            spec_file.write_text(original)
        new = [l for l in lines if l not in baseline]
        named = [l for l in new if must_mention in l]
        for l in named:
            print(f'      reported: {l}')
        if not named:
            failures.append(f'{label}: the audit reported nothing new naming '
                            f'{must_mention!r} — it did not notice the change')
        elif rc == 0:
            failures.append(f'{label}: the audit reported the change but still '
                            f'exited 0')

    try:
        rc_clean, baseline = run_strict('unmodified spec')
        if rc_clean != 0:
            print('  NOTE  the unmodified spec does not pass --strict in this')
            print('        environment. That is a fact about the environment, not')
            print('        about this test: the rows below are judged on the')
            print('        findings the audit ADDS, so they are still meaningful.')
            for l in baseline:
                print(f'        pre-existing: {l}')

        perturbation('XGBoost learning rate 0.1 -> 0.3',
                     "'learning_rate': 0.1,", "'learning_rate': 0.3,",
                     'learning_rate', baseline)
        perturbation('quantile forest 300 -> 100 trees',
                     "'n_estimators': 300,", "'n_estimators': 100,",
                     'n_estimators', baseline)
    finally:
        spec_file.write_text(original)
        assert spec_file.read_text() == original, 'FAILED TO RESTORE THE SPEC'
        print('  spec restored')
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

    # ---- the cross-pipeline half, which nothing used to test ----------------
    print('\n  cross-pipeline comparison (NOT --skip-experimental)')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        stub = Path(tmp) / 'alternative_data_noise_robustness.py'
        stub.write_text(
            '# A stand-in experimental pipeline that resolved a DIFFERENT copy of\n'
            '# model_defaults.py. This is the failure the comparison exists to\n'
            '# catch: two checkouts on one host, one of them stale.\n'
            'class _Spec:\n'
            '    @staticmethod\n'
            '    def spec_hash():\n'
            "        return 'notthehash'\n"
            'MODEL_SPEC = _Spec()\n')
        problems, out = _spec_parity_output(str(stub))
        bit = problems >= 1 and 'DIFFERS' in out
        print(f'  {"a mismatched experimental spec is caught":38s} '
              f'{problems} problem(s)  {"OK" if bit else "NOT CAUGHT"}')
        if not bit:
            failures.append('the cross-pipeline comparison did NOT flag an '
                            'experimental pipeline holding a different spec')

    mod, where = _load_experimental_module(None)
    if mod is None:
        print(f'  {"the real experimental pipeline":38s} NOT PRESENT — '
              f'this run does not prove it matches')
        print(f'        ({where})')
    else:
        problems, out = _spec_parity_output(None)
        agrees = problems == 0 and 'MATCH' in out
        print(f'  {"the real experimental pipeline agrees":38s} '
              f'{problems} problem(s)  {"OK" if agrees else "MISMATCH"}')
        if not agrees:
            failures.append('the real experimental pipeline does not resolve the '
                            'same spec as this one — see --strict for the detail')

    print()
    if failures:
        for f in failures:
            print(f'  FAIL  {f}')
        print('\nSELF-TEST FAILED — one of the checks above did not report a '
              'change it should have')
        return 1
    print('SELF-TEST PASSED — the audit reports a perturbed parameter and a '
          'mismatched pipeline')
    return 0


def compare_results(paths):
    """Do these results files come from the same models?

    The import-time hash check proves the two pipelines READ the same spec. This
    proves the rows on disk were WRITTEN under it -- which is the claim anyone
    pooling QM9 and experimental rows in a figure is actually relying on, and
    the one that breaks quietly when a stale CSV survives a re-run.
    """
    import csv as _csv
    print('=' * 78)
    print('RESULTS PROVENANCE — do these files come from the same spec?')
    print('=' * 78)
    seen, problems = {}, 0
    for path in paths:
        f = Path(path)
        if not f.exists():
            print(f'  MISSING  {path}')
            problems += 1
            continue
        with open(f, newline='') as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            print(f'  EMPTY    {path}')
            problems += 1
            continue
        if 'spec_hash' not in rows[0]:
            print(f'  NO PROVENANCE  {path}')
            print('           written before the shared spec existed — it cannot be '
                  'compared, and must not be pooled with rows that can')
            problems += 1
            continue
        hashes = sorted({r['spec_hash'] for r in rows if r['spec_hash']})
        versions = sorted({r.get('spec_version', '') for r in rows})
        gp = sorted({r.get('gp_fit_method', '') for r in rows if r.get('gp_fit_method')})
        print(f'  {f.name}: {len(rows)} rows   spec {versions} {hashes}'
              + (f'   gp_fit_method {gp}' if gp else ''))
        if len(hashes) > 1:
            print('           ^ this ONE file mixes spec hashes — it holds rows from '
                  'two different sets of models')
            problems += 1
        if len(gp) > 1:
            print('           ^ this file mixes Gaussian-process fitting routines; '
                  'those rows are not comparable')
            problems += 1
        for h in hashes:
            seen.setdefault(h, []).append(f.name)

    if len(seen) > 1:
        problems += 1
        print('\n  MISMATCH — these files were produced under different specs:')
        for h, files in sorted(seen.items()):
            print(f'    {h}  {files}')
        print('  Pooling them in one figure or one ANOVA compares different models.')
    elif seen:
        print(f'\n  MATCH — every file was produced under spec '
              f'{list(seen)[0]}')
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero on any mismatch, missing package or '
                         'broken assumption')
    ap.add_argument('--self-test', action='store_true',
                    help='prove --strict fails when the spec is perturbed')
    ap.add_argument('--write-baseline', action='store_true',
                    help='record the current effective parameters as the baseline')
    ap.add_argument('--experimental-path', default=None,
                    help='path to alternative_data_noise_robustness.py')
    ap.add_argument('--skip-experimental', action='store_true',
                    help='audit this repository only')
    ap.add_argument('--compare-results', nargs='+', metavar='CSV',
                    help='check that two or more results files were written under '
                         'the same spec, and that neither mixes specs internally')
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.compare_results:
        problems = compare_results(args.compare_results)
        print(f'\nSUMMARY: {problems} problem(s)')
        return 1 if (args.strict and problems) else 0

    problems = audit_spec_parity(args.experimental_path, args.skip_experimental)
    problems += audit_effective_params(write_baseline=args.write_baseline)
    problems += audit_assumptions()
    missing = audit_environment()
    stale = report_manual()

    print('\n' + '=' * 78)
    print(f'SUMMARY: {problems} problem(s), {missing} missing package(s)'
          + (f', {stale} out-of-date manual entry(ies)' if stale else ''))
    print('=' * 78)
    if args.strict and (problems or missing):
        print('STRICT: refusing to pass. Do not submit jobs from this '
              'environment until these are cleared.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
