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
     True, whether botorch's fitter accepts a plain gpytorch ExactGP, and that
     no QM9 job script passes --use_best_params (which would silently take the
     pipeline off the defaults this whole audit is about).
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
import importlib
import importlib.util
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

# Real differences that are not estimator parameters. Kept short and true: an
# item leaves this list only when it is actually fixed in both pipelines.
MANUAL = [
    ('ECFP4 identity — THE WORST FINDING, still open',
     'NOT a circular fingerprint. rdk_fingerprint_mol binds RDKFingerprintMol, '
     'a Daylight-style topological PATH fingerprint',
     'a genuine Morgan radius-2 fingerprint',
     'rust/src/main.rs:22,:822  vs  KIRBy/src/kirby/representations/molecular.py',
     'paper.tex:203 describes circular substructures at radius 2, which is false '
     'for QM9. ~254 job invocations use it. The crate ships morgan_fingerprint_mol '
     'but hard-coded at radius 3 (ECFP6), so it is not a drop-in fix. No parameter '
     'diff could ever have caught this — it took reading the binding.'),
    ('PDV identity — still open',
     "the representation named pdv is BINARISED at (pdv > 0); pdv is "
     "the real one",
     'PDV is continuous and standardised',
     'scripts/process_and_train.py:377 vs :385',
     'The merge layer must map the experimental PDV to QM9 pdv. That '
     'pair is the ONLY genuinely comparable representation across the two studies.'),
    ('SNS identity — still open',
     'counts computed then discarded by packbits; also casts to uint8 BEFORE '
     'packing, so a count that is an exact multiple of 256 records as ABSENT',
     'raw counts, then standardised',
     'scripts/process_and_train.py:366-371',
     'Binary presence versus standardised counts under one name, plus a silent '
     'data-loss path. Measured on 20,000 QM9 molecules: 21.8% of information-'
     'bearing entries wrong.'),
    ('Learned-embedding storage — Chat C',
     'per-molecule min-max to uint8, never standardised',
     'n/a — representations built independently',
     'scripts/process_and_train.py:971-975, :828-831',
     'Destroys cross-molecule comparability. RERUN_PLAN.md 2.8c.'),
    ('Validation split',
     'merged into training for seven model families',
     'kept separate throughout',
     'models/models.py train_rf/svm/ngboost/xgboost/lgb/gauche/conformal',
     'Roughly a 29% difference in effective training-set size between studies.'),
    ('Repetitions',
     '10 independent fits per cell, each with its own seed (-b/--repetitions)',
     'ONE fit per cell, seed pinned to 42',
     'scripts/process_and_train.py:247 vs KIRBy constructors',
     "Author's decision 2026-08-26: keep one fit on the experimental side. The "
     'experimental ANOVA therefore has no estimable residual term and no '
     'run-to-run error bar. The paper must say so.'),
    ('Label standardisation',
     'all models, in Rust, using the CLEAN training mean and SD since 2026-08-26',
     'neural models only, scaler fitted on the noisy training labels',
     'rust/src/main.rs vs KIRBy train_neural_regression',
     'Tree and kernel models on the experimental side see raw labels. Scale-'
     'invariant for trees; NOT for the SVM or the GP.'),
    ('Uncertainty calibration',
     'a temperature fitted on a carve-out of validation',
     'absent',
     'scripts/utils.py:323 and the models.py call sites',
     'Decided by scripts/parity_test_calibration.py; the column the analysis '
     'reads is model_defaults.UNCERTAINTY_DEFAULTS["primary_column"]. Whatever '
     'it is, the figure script must NAME it — today it silently prefers the '
     'calibrated column wherever one exists (Chat J).'),
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

def _effective_params(mod_name, cls_name, kwargs):
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    cls = getattr(mod, cls_name, None)
    if cls is None:
        return None, f'{cls_name} not found in {mod_name}'
    try:
        est = cls(**kwargs)
        params = est.get_params() if hasattr(est, 'get_params') else dict(kwargs)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    return {k: _jsonable(v) for k, v in params.items() if k not in BENIGN}, None


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
    kwargs = dict(canonical.SKLEARN_DEFAULTS[spec_key])
    if spec_key == 'ngboost':
        # Names, not classes, in the spec. Resolve them the way both pipelines do.
        from ngboost.distns import Normal
        from ngboost.scores import MLE
        kwargs['Dist'] = {'Normal': Normal}[kwargs.pop('dist')]
        kwargs['Score'] = {'MLE': MLE}[kwargs.pop('score')]
    return kwargs


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
            kwargs = _build_kwargs(spec_key, canonical)
        except Exception as exc:
            print(f'\n{spec_key}\n  SKIPPED — {type(exc).__name__}: {exc}')
            continue
        params, err = _effective_params(mod_name, cls_name, kwargs)
        print(f'\n{spec_key}  ({mod_name}.{cls_name})')
        if params is None:
            print(f'  CANNOT BUILD — {err}')
            print('  ^ every job requesting this model in this interpreter '
                  'would fail on contact')
            problems += 1
            continue
        current[spec_key] = params
        pinned = {k for k in kwargs if k not in BENIGN}
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

    def _no_tuned_params():
        hits = []
        for d in ('slurm_scripts_qm9_rerun', 'slurm_scripts_uncertainty_rerun'):
            p = REPO / d
            if not p.exists():
                continue
            for f in sorted(p.glob('*.sh')):
                if 'use_best_params' in f.read_text():
                    hits.append(f.name)
        return not hits, (f'{len(hits)} job script(s) pass --use_best_params: '
                          f'{hits}' if hits else 'no job script passes it')
    check('no job script passes --use_best_params', _no_tuned_params,
          'That flag loads hyperparameters from a JSON, which the experimental '
          'pipeline cannot do at all. This whole audit compares the DEFAULT '
          'path; a job on the tuned path is not audited by anything.')

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
    print('\n' + '=' * 78)
    print('NOT VISIBLE HERE — differences that are not estimator parameters')
    print('=' * 78)
    for title, qm9, val, src, why in MANUAL:
        print(f'\n{title}')
        print(f'  QM9          {qm9}')
        print(f'  experimental {val}')
        print(f'  see          {src}')
        print(f'  matters      {why}')


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def self_test():
    """Prove --strict actually bites. Perturbs the spec on disk, confirms the
    audit fails, and restores it. This is the check that fails if the fix is
    removed — the standing rule in RERUN_PLAN.md section 13.2."""
    print('=' * 78)
    print('SELF-TEST — does --strict fail when the spec is perturbed?')
    print('=' * 78)
    spec_file = REPO / 'models' / 'model_defaults.py'
    original = spec_file.read_text()
    failures = []

    def run_strict(label):
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--strict',
             '--skip-experimental'],
            capture_output=True, text=True, cwd=str(REPO))
        print(f'  {label:38s} exit {r.returncode}')
        return r.returncode

    try:
        rc_clean = run_strict('unmodified spec')
        if rc_clean != 0:
            failures.append('the unmodified spec does not pass --strict')

        perturbed = original.replace("'learning_rate': 0.1,", "'learning_rate': 0.3,", 1)
        if perturbed == original:
            failures.append('could not perturb the spec — the self-test is not '
                            'testing anything')
        else:
            spec_file.write_text(perturbed)
            if run_strict('XGBoost learning rate 0.1 -> 0.3') == 0:
                failures.append('--strict PASSED with a changed learning rate')
            spec_file.write_text(original)

        perturbed = original.replace("'n_estimators': 300,", "'n_estimators': 100,", 1)
        if perturbed != original:
            spec_file.write_text(perturbed)
            if run_strict('quantile forest 300 -> 100 trees') == 0:
                failures.append('--strict PASSED with a changed tree count')
            spec_file.write_text(original)
    finally:
        spec_file.write_text(original)
        assert spec_file.read_text() == original, 'FAILED TO RESTORE THE SPEC'
        print('  spec restored')

    print()
    if failures:
        for f in failures:
            print(f'  FAIL  {f}')
        print('\nSELF-TEST FAILED — this audit would not catch a real drift')
        return 1
    print('SELF-TEST PASSED — --strict fails on a changed parameter, '
          'passes when clean')
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
    report_manual()

    print('\n' + '=' * 78)
    print(f'SUMMARY: {problems} problem(s), {missing} missing package(s)')
    print('=' * 78)
    if args.strict and (problems or missing):
        print('STRICT: refusing to pass. Do not submit jobs from this '
              'environment until these are cleared.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
