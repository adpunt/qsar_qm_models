#!/usr/bin/env python3
"""QM9's noise-predicting Gaussian process must start its lengthscale from the data.

WHY THIS EXISTS
---------------
`init_rbf_lengthscale` was written on 2026-08-26 and wired into
`fit_gp_with_fallback`, which was its only caller. `train_heteroscedastic_gp`
does not go through that function: it builds its own model, its own likelihood
and its own joint Adam loop. So on QM9 this one model kept starting at
gpytorch's softplus(0), about 0.69, while the real distance between two
molecules on these representations runs from about 17 on the PDV to about 1,100
on the learned embeddings.

At 0.69 every molecule looks infinitely far from every other one. The kernel
matrix is the identity, the marginal likelihood is flat, and the fit returns the
prior: one number for every molecule. That number is still scored, and the score
reads as a weak representation. On the PLAIN Gaussian process that produced
R2 = -0.0158 for MHG-GNN before the 2026-08-26 fix -- those rows are
gauche_rbf, in results/gp_kernel_harvest/qm9/, not this model. On this path,
measured below, it is smaller and real: at the 100 epochs the cluster runs the
joint Adam loop drags the lengthscale part of the way up on its own, so the fit
does not collapse, it is merely a different fit.

The laboratory runner initialises at BOTH of its Gaussian-process sites (KIRBy
tests/alternative_data_noise_robustness.py, _init_rbf_lengthscale called at
:1614 and :1774 as of 2026-09-12), so until this was fixed a GP-Hetero row from
one pipeline and a GP-Hetero row from the other were different fits under one
name.

WHAT IS MEASURED
----------------
Real QM9 molecules, the PDV built with the pipeline's own descriptor list, the
real HOMO-LUMO gap as the label, standardised per feature on the training split
the way the pipeline standardises it. Then `train_heteroscedastic_gp` is called
-- the function the cluster calls, not a copy of it -- and three things are
checked:

  1. the lengthscale it starts from is the median distance between the training
     molecules, and is nowhere near 0.69;
  2. every inner fit gets it, not only the reported one;
  3. `GP_DEFAULTS['init_lengthscale_from_data']` is honoured -- turning it off
     leaves the kernel at 0.69 -- and the two starts give different fits, so the
     check fails on the defect rather than on a seed.

HOW BIG IT IS, measured here on 2026-09-12 at the 100 epochs the cluster runs:
R2 0.9359 started from the data against 0.8921 started at gpytorch's 0.69, on
100 held-out molecules. It is 0.044 in R2, not a collapse. At 40 epochs the same
check reports 0.913 against 0.233, which is four times the difference the cluster
would see and is an artefact of stopping the joint loop early -- hence EPOCHS
below.

    python scripts/test_het_gp_lengthscale.py
"""
import argparse
import ast
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'models'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

SMILES_CACHE = os.path.join(REPO, '.decomposition_controls_qm9_smiles.txt')
QM9_CSV = os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf.csv')
PIPELINE = os.path.join(REPO, 'scripts', 'process_and_train.py')

# Small enough to fit in seconds on a laptop, large enough that the median
# distance is a real median and not two molecules.
N_MOLECULES = 400
N_TRAIN = 300

# THE NUMBER THE CLUSTER RUNS, and it is load-bearing here rather than a detail.
# scripts/process_and_train.py:372 defaults --epochs to 100,
# NEURAL_DEFAULTS['training']['epochs'] is 100 and
# slurm_scripts_qm9_rerun/generate_scripts.py:646 prices this model on 100 Adam
# epochs. At 40 the joint loop has not climbed far and the gap between the two
# starts looks like a collapse; at 100 it has climbed most of the way on its own
# and the gap is real but small. Running this check at 40 would have reported a
# effect four times the one the cluster would see.
EPOCHS = 100

# gpytorch's own start, softplus(0). The fix is worth having only if the data's
# answer is a long way from it; on the PDV it is about 17.
GPYTORCH_DEFAULT = 0.6931
MIN_RATIO_TO_DEFAULT = 5.0


def descriptor_list():
    """The pipeline's own 200 RDKit descriptor names, read out of its source.

    Read rather than imported: importing scripts/process_and_train.py drags in
    deepchem and tensorflow, which is a minute of start-up for one list. Read
    rather than retyped: a second copy of this list is exactly how the PDV would
    come to mean two different things.
    """
    tree = ast.parse(open(PIPELINE).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'DEFAULT_DESCRIPTOR_LIST':
                    return ast.literal_eval(node.value)
    raise SystemExit(f'DEFAULT_DESCRIPTOR_LIST is not assigned in {PIPELINE}')


def real_qm9_pdv(n):
    """(features, labels) for the first `n` QM9 molecules that featurise.

    The SMILES come from the cache decomposition_controls.py writes out of
    gdb9.sdf, in file order, so row i of the structure file and row i of
    gdb9.sdf.csv are the same molecule and the label is the real gap.
    """
    if not os.path.exists(SMILES_CACHE):
        raise SystemExit(
            f'{SMILES_CACHE} does not exist. It is written by\n'
            f'    python scripts/decomposition_controls.py\n'
            f'and this check reads real molecules rather than inventing them.')
    if not os.path.exists(QM9_CSV):
        raise SystemExit(f'{QM9_CSV} does not exist, so there are no real labels.')

    from rdkit import Chem, RDLogger
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    RDLogger.DisableLog('rdApp.*')

    with open(SMILES_CACHE) as fh:
        smiles = [line.rstrip('\n') for line in fh]

    gaps = []
    with open(QM9_CSV) as fh:
        header = next(fh).rstrip('\n').split(',')
        col = header.index('gap')
        for line in fh:
            gaps.append(float(line.rstrip('\n').split(',')[col]))

    calc = MolecularDescriptorCalculator(descriptor_list())
    x, y = [], []
    for smi, gap in zip(smiles, gaps):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        row = np.array(calc.CalcDescriptors(mol), dtype=np.float64)
        if not np.isfinite(row).all():
            continue
        x.append(row)
        y.append(gap)
        if len(x) >= n:
            break
    if len(x) < n:
        raise SystemExit(f'only {len(x)} molecules featurised, needed {n}')
    return np.array(x), np.array(y)


def standardise(x_train, x_rest):
    """Per feature, on the TRAINING split alone -- the pipeline's rule."""
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    return ((x_train - mean) / sd).astype(np.float32), \
           ((x_rest - mean) / sd).astype(np.float32)


def median_pairwise(x):
    import torch
    with torch.no_grad():
        t = torch.tensor(np.asarray(x), dtype=torch.float32)
        return torch.cdist(t, t).flatten().median().item()


class Args:
    pass


def build_args(scratch, epochs):
    a = Args()
    a.kernel = 'rbf'
    a.epochs = epochs
    a.filepath = scratch
    a.sample_size = N_MOLECULES
    a.uncertainty = False
    a.score_validation = False
    a.tuning = False
    a.use_best_params = False
    a.logging = False
    return a


def run_fit(M, x_tr, y_tr, x_te, y_te, x_val, y_val, scratch, epochs):
    """One call to the function the cluster calls. Returns the recorded
    (model, lengthscale) pairs and the predictions of the reported fit."""
    seen = []
    original = M.init_rbf_lengthscale

    def recorder(model, train_x):
        out = original(model, train_x)
        base = getattr(model.covar_module, 'base_kernel', None)
        applied = None
        if base is not None and getattr(base, 'lengthscale', None) is not None:
            applied = float(base.lengthscale.detach().flatten()[0])
        # The outputscale is set just above the lengthscale in _fit_het_gp, so
        # by the time this fires it is whatever that site left it at, and the
        # optimiser has not run yet.
        scale = getattr(model.covar_module, 'outputscale', None)
        seen.append({'median': out, 'applied': applied,
                     'outputscale': None if scale is None else float(scale),
                     'n_fit': int(np.asarray(train_x).shape[0])})
        return out

    captured = {}
    original_metrics = M.calculate_regression_metrics

    def metric_spy(y_true, y_pred, **kw):
        captured['y_pred'] = np.asarray(y_pred)
        return original_metrics(y_true, y_pred, **kw)

    M.init_rbf_lengthscale = recorder
    M.calculate_regression_metrics = metric_spy
    try:
        M.train_heteroscedastic_gp(
            x_tr, y_tr, x_te, y_te, x_val, y_val,
            build_args(scratch, epochs), 0.0, 'pdv', 0, 0, 0, y_te)
    finally:
        M.init_rbf_lengthscale = original
        M.calculate_regression_metrics = original_metrics
    return seen, captured.get('y_pred')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scratch', default=os.path.join(
        os.environ.get('TMPDIR', '/tmp'), 'test_het_gp_lengthscale.csv'))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    cli = ap.parse_args()
    if os.path.exists(cli.scratch):
        os.remove(cli.scratch)

    import torch
    import models as M
    from model_defaults import GP_DEFAULTS

    x, y = real_qm9_pdv(N_MOLECULES)
    x_train, x_hold = standardise(x[:N_TRAIN], x[N_TRAIN:])
    y_train, y_hold = y[:N_TRAIN].astype(np.float64), y[N_TRAIN:].astype(np.float64)
    half = len(x_hold) // 2
    x_test, y_test = x_hold[:half], y_hold[:half]
    x_val, y_val = x_hold[half:], y_hold[half:]

    expected = median_pairwise(x_train)
    print(f"  {N_TRAIN} real QM9 molecules, {x_train.shape[1]} PDV features")
    print(f"  median distance between training molecules  {expected:8.3f}")
    print(f"  gpytorch's own start                        {GPYTORCH_DEFAULT:8.4f}")

    failures = []

    if expected < GPYTORCH_DEFAULT * MIN_RATIO_TO_DEFAULT:
        failures.append(
            f"the median distance between these molecules is {expected:.3f}, within "
            f"{MIN_RATIO_TO_DEFAULT}x of gpytorch's {GPYTORCH_DEFAULT}, so this check "
            f"cannot tell the two starts apart. Use a representation whose distances "
            f"are the study's.")

    torch.manual_seed(0)
    seen, y_pred = run_fit(M, x_train, y_train, x_test, y_test, x_val, y_val,
                           cli.scratch, cli.epochs)

    # 1. It is called at all, on the fitting features.
    if not seen:
        failures.append(
            "train_heteroscedastic_gp never called init_rbf_lengthscale, so it is "
            "still starting at gpytorch's 0.69. The call belongs inside _fit_het_gp, "
            "on x_fit, before the optimisation loop.")
    else:
        got = seen[0]
        print(f"  lengthscale the fit started from            "
              f"{got['applied'] if got['applied'] is not None else float('nan'):8.3f}")
        if got['n_fit'] != N_TRAIN:
            failures.append(
                f"it was initialised on {got['n_fit']} molecules, not the {N_TRAIN} "
                f"the fit saw. It must read the features the fit is about to use.")
        if got['median'] is None or not np.isclose(got['median'], expected, rtol=1e-3):
            failures.append(
                f"the start is {got['median']}, not the median distance between the "
                f"training molecules ({expected:.4f}).")
        if got['applied'] is None or not np.isclose(got['applied'], expected, rtol=1e-3):
            failures.append(
                f"the median was computed but the kernel was left at "
                f"{got['applied']}. It has to be assigned to the base kernel.")

        # THE SCALE IN FRONT OF THE KERNEL, the other half of the same
        # difference. The plain Gaussian process here applies it and the
        # laboratory runner applies it at both of its sites; this one did not,
        # so its ScaleKernel started at gpytorch's softplus(0) = 0.693 rather
        # than the spec's 1.0.
        want_scale = GP_DEFAULTS['outputscale'] if GP_DEFAULTS['apply_outputscale'] else None
        print(f"  outputscale the fit started from            "
              f"{got['outputscale'] if got['outputscale'] is not None else float('nan'):8.3f}")
        if want_scale is not None and (
                got['outputscale'] is None
                or not np.isclose(got['outputscale'], want_scale, rtol=1e-3)):
            failures.append(
                f"GP_DEFAULTS['apply_outputscale'] is True and the spec's "
                f"outputscale is {want_scale}, but this fit started at "
                f"{got['outputscale']}. gpytorch's own start is 0.693, and the "
                f"plain Gaussian process on this pipeline does apply the spec's.")

    # 2. Every fit, not only the reported one. The out-of-fold pass refits inside
    #    the same function, which is why the call sits there rather than beside it.
    src = open(os.path.join(REPO, 'models', 'models.py')).read()
    body = src[src.index('def train_heteroscedastic_gp'):]
    body = body[:body.index('\ndef ', 1)]
    inner = body[body.index('def _fit_het_gp'):]
    inner = inner[:inner.index('\n    # The validation molecules')]
    if 'init_rbf_lengthscale(' not in inner:
        failures.append(
            "the call is outside _fit_het_gp, so the inner out-of-fold folds refit "
            "at gpytorch's default while the reported fit does not.")
    else:
        print("  the call is inside _fit_het_gp, so the inner folds get it too")

    # 3. The spec key is honoured, and turning it off reproduces the defect:
    #    the same call, the same seed, the kernel left at 0.69. On this path at
    #    100 epochs that narrows the predictions rather than flattening them,
    #    because the joint Adam loop climbs the lengthscale part of the way on
    #    its own. The check is that the two starts give DIFFERENT fits, which is
    #    what makes the rows already on disk rows from another fit.
    spread_on = float(np.max(y_pred) - np.min(y_pred)) if y_pred is not None else 0.0
    GP_DEFAULTS['init_lengthscale_from_data'] = False
    try:
        torch.manual_seed(0)
        seen_off, y_pred_off = run_fit(M, x_train, y_train, x_test, y_test,
                                       x_val, y_val, cli.scratch, cli.epochs)
    finally:
        GP_DEFAULTS['init_lengthscale_from_data'] = True
    spread_off = float(np.max(y_pred_off) - np.min(y_pred_off)) if y_pred_off is not None else 0.0
    label_spread = float(np.max(y_train) - np.min(y_train))

    def r2(pred):
        if pred is None:
            return float('nan')
        pred = np.asarray(pred).ravel()[:len(y_test)]
        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        return 1.0 - ss_res / ss_tot

    r2_on, r2_off = r2(y_pred), r2(y_pred_off)
    print(f"  predictions vary by, started from the data  {spread_on:8.4f}")
    print(f"  predictions vary by, started at 0.69        {spread_off:8.4f}")
    print(f"  the training labels vary by                 {label_spread:8.4f}")
    print(f"  R2 on held-out, started from the data       {r2_on:8.4f}")
    print(f"  R2 on held-out, started at 0.69             {r2_off:8.4f}")
    print(f"  the difference the fix makes, in R2         {r2_on - r2_off:8.4f}")

    if seen_off and seen_off[0]['median'] is not None:
        failures.append(
            "GP_DEFAULTS['init_lengthscale_from_data'] is False and the lengthscale "
            "was still moved. The switch has to be honoured.")
    if not spread_on > spread_off:
        failures.append(
            f"starting from the data did not widen the predictions "
            f"({spread_on:.4g} against {spread_off:.4g}), so this check is not "
            f"measuring what it claims to measure.")

    if failures:
        print(f"\nFAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: QM9's noise-predicting Gaussian process starts its lengthscale at the "
          "median distance between the training molecules, in every fit it makes")
    return 0


if __name__ == '__main__':
    sys.exit(main())
