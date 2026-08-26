#!/usr/bin/env python3
"""Should the paper report raw or calibrated uncertainty?

THE QUESTION
------------
Every uncertainty model outputs a predicted error bar per molecule. QM9 then
fits ONE number on a carve-out of validation -- the multiplier that makes those
error bars agree on average with the errors actually observed -- and multiplies
every error bar by it (scripts/utils.py:323, a Gaussian negative-log-likelihood
minimisation bounded to [0.1, 10.0]). The three experimental datasets do no such
thing. The figure script then silently prefers the calibrated column wherever it
exists, so the two studies report differently-scaled uncertainties and nothing
says which.

A single positive multiplier cannot reorder molecules, so both rank-based
uncertainty questions are unaffected either way. What it changes is coverage --
the fraction of molecules whose true value falls inside +/-1 error bar.

So the question is not "which is more accurate". It is: does a multiplier fitted
on validation still hold on held-out test data, and does it keep holding as the
injected noise rises? If it does not transfer, calibrated coverage is not good
even on its own terms, and raw is the only defensible primary.

WHAT IS MEASURED, per model, per noise level, per seed
------------------------------------------------------
    temperature            the fitted multiplier
    temperature_at_bound   whether it hit the [0.1, 10] bound and gave up
    cov_raw_1sd/2sd        raw coverage on TEST (nominal 0.6827 / 0.9545)
    cov_cal_1sd/2sd        calibrated coverage on TEST -- does it transfer
    cov_val_cal_1sd        calibrated coverage on the slice it was FITTED to,
                           which is the optimistic number, shown for contrast
    spearman_raw / _cal    uncertainty vs |error|; must be identical
    nll_raw / nll_cal      Gaussian log score, the thing the fit minimises

THE DECISION RULE, fixed before the run
---------------------------------------
Calibrated is defensible as the primary number only if calibrated test coverage
lands closer to nominal than raw for EVERY model at EVERY noise level, and the
fitted multiplier does not drift materially with the noise level (a multiplier
that has to be refitted per noise level is a per-run fudge, not a property of the
model). Otherwise: report raw as primary, keep calibrated as a clearly-labelled
secondary column, and make the analysis name the column it read.

USAGE
-----
    python scripts/parity_test_calibration.py
    python scripts/parity_test_calibration.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_test_common import (  # noqa: E402
    HAS_QRF, QRF_UNAVAILABLE_REASON, build_representations, env_fingerprint,
    forest_quantiles, inject_gaussian, load_qm9, scaffold_split, standardise,
)

warnings.filterwarnings('ignore')
OUT_DIR = Path(__file__).resolve().parent.parent / 'results' / 'parity_tests'
T_BOUNDS = (0.1, 10.0)          # the bounds QM9 uses; hitting them is a finding
GP_MAX_N = 1500                 # this test only, so an exact GP stays tractable


def fit_temperature(y_pred, y_std, y_true):
    """Exactly scripts/utils.py:323 — the function QM9 actually calls."""
    def nll(T):
        s = np.maximum(y_std * T, 1e-6)
        return (0.5 * np.log(2 * np.pi * s ** 2)
                + 0.5 * ((y_true - y_pred) ** 2 / s ** 2)).mean()
    res = minimize_scalar(nll, bounds=T_BOUNDS, method='bounded')
    return float(res.x)


def gaussian_nll(y_pred, y_std, y_true):
    s = np.maximum(y_std, 1e-6)
    return float((0.5 * np.log(2 * np.pi * s ** 2)
                  + 0.5 * ((y_true - y_pred) ** 2 / s ** 2)).mean())


def coverage(y_pred, y_std, y_true, k):
    return float((np.abs(y_true - y_pred) <= k * y_std).mean())


# ---------------------------------------------------------------------------
# the four uncertainty models
# ---------------------------------------------------------------------------

def fit_qrf(X_tr, y_tr, X_list, seed):
    """Quantile forest. Emulated here — see parity_test_common. sigma from the
    16-84 interval half-width, which is what models.py:1388 uses."""
    m = RandomForestRegressor(n_estimators=300, max_features='sqrt', bootstrap=True,
                              random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    out = []
    for X in X_list:
        q16, q50, q84 = forest_quantiles(m, X_tr, y_tr, X).T
        out.append((q50, np.maximum((q84 - q16) / 2.0, 1e-6)))
    return out


def fit_ngboost(X_tr, y_tr, X_list, seed):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    m = NGBRegressor(Dist=Normal, Score=MLE, natural_gradient=True,
                     n_estimators=500, learning_rate=0.01, verbose=False,
                     random_state=seed).fit(X_tr, y_tr)
    out = []
    for X in X_list:
        d = m.pred_dist(X)
        out.append((np.asarray(d.loc, dtype=float),
                    np.maximum(np.asarray(d.scale, dtype=float), 1e-6)))
    return out


def fit_gp(X_tr, y_tr, X_list, seed):
    """gauche/gpytorch ExactGP, RBF, ConstantMean + ScaleKernel, noise 1e-3 —
    the construction both pipelines use."""
    import gpytorch
    import torch

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, x, y, lik):
            super().__init__(x, y, lik)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    rng = np.random.RandomState(seed)
    if len(X_tr) > GP_MAX_N:
        sub = rng.choice(len(X_tr), GP_MAX_N, replace=False)
        X_tr, y_tr = X_tr[sub], y_tr[sub]
    xt = torch.from_numpy(np.asarray(X_tr, dtype=np.float64))
    yt = torch.from_numpy(np.asarray(y_tr, dtype=np.float64))
    lik = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
    model = _GP(xt, yt, lik)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = -mll(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval(); lik.eval()
    out = []
    with torch.no_grad():
        for X in X_list:
            p = lik(model(torch.from_numpy(np.asarray(X, dtype=np.float64))))
            out.append((p.mean.numpy(),
                        np.maximum(np.sqrt(p.variance.numpy()), 1e-6)))
    return out


def fit_bnn(X_tr, y_tr, X_list, seed, passes=100):
    """Full BNN over the QM9 DNN architecture [128, 64], torchbnn priors
    mu=0 sigma=0.1 (models.py:1046-1071), 100 stochastic forward passes."""
    import torch
    import torch.nn as nn
    import torchbnn as bnn
    from torchhk import transform_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cpu')

    net = nn.Sequential(
        nn.Linear(X_tr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    transform_model(net, nn.Linear, bnn.BayesLinear,
                    args={'prior_mu': 0, 'prior_sigma': 0.1,
                          'in_features': '.in_features',
                          'out_features': '.out_features', 'bias': '.bias'},
                    attrs={'weight_mu': '.weight'})
    net = net.to(device)

    y_mu, y_sd = float(y_tr.mean()), float(y_tr.std())
    xt = torch.FloatTensor(X_tr)
    yt = torch.FloatTensor((y_tr - y_mu) / y_sd).view(-1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    for _ in range(100):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            crit(net(bx), by).backward()
            opt.step()

    net.train()      # stochastic passes need the sampling path live
    out = []
    for X in X_list:
        xq = torch.FloatTensor(X)
        with torch.no_grad():
            draws = np.array([net(xq).numpy().ravel() for _ in range(passes)])
        out.append((draws.mean(0) * y_sd + y_mu,
                    np.maximum(draws.std(0) * y_sd, 1e-6)))
    return out


MODELS = {'QRF': fit_qrf, 'NGBoost': fit_ngboost, 'GP': fit_gp, 'BNN': fit_bnn}


def run(n_molecules, seeds, sigmas, models, rep, incremental_path=None):
    smiles, y, scaffolds = load_qm9(n_molecules)
    print(f'molecules {len(smiles)}   label SD {y.std():.4f} eV', flush=True)
    X_all = build_representations(smiles, which=(rep,))
    X = X_all[rep]

    rows = []
    for seed in seeds:
        train, val, test = scaffold_split(scaffolds, seed, train_frac=0.70, val_frac=0.15)
        X_tr, X_va, X_te = X[train], X[val], X[test]
        if rep == 'descriptors':
            X_tr, X_va, X_te = standardise(X_tr, X_va, X_te)
        print(f'\nseed {seed}: train {len(train)}  calibration {len(val)}  '
              f'test {len(test)}', flush=True)

        for sigma in sigmas:
            y_tr, _ = inject_gaussian(y[train], sigma, seed)
            y_va = y[val]      # the calibration slice is held out => CLEAN, as after 9d7db67
            y_te = y[test]
            for name in models:
                t0 = time.time()
                try:
                    (mu_va, sd_va), (mu_te, sd_te) = MODELS[name](
                        X_tr, y_tr, [X_va, X_te], seed)
                except Exception as exc:
                    print(f'  {name:8s} sigma={sigma:.1f}  FAILED '
                          f'{type(exc).__name__}: {exc}', flush=True)
                    rows.append(dict(model=name, seed=seed, sigma=sigma,
                                     failed=f'{type(exc).__name__}: {exc}'))
                    _flush(rows, incremental_path)
                    continue

                T = fit_temperature(mu_va, sd_va, y_va)
                at_bound = bool(abs(T - T_BOUNDS[0]) < 1e-6 or abs(T - T_BOUNDS[1]) < 1e-6)
                sp_raw = spearmanr(sd_te, np.abs(y_te - mu_te)).correlation
                sp_cal = spearmanr(sd_te * T, np.abs(y_te - mu_te)).correlation

                rows.append(dict(
                    model=name, seed=seed, sigma=sigma, rep=rep, failed='',
                    r2=r2_score(y_te, mu_te),
                    temperature=T, temperature_at_bound=at_bound,
                    cov_raw_1sd=coverage(mu_te, sd_te, y_te, 1),
                    cov_raw_2sd=coverage(mu_te, sd_te, y_te, 2),
                    cov_cal_1sd=coverage(mu_te, sd_te * T, y_te, 1),
                    cov_cal_2sd=coverage(mu_te, sd_te * T, y_te, 2),
                    cov_val_cal_1sd=coverage(mu_va, sd_va * T, y_va, 1),
                    spearman_raw=sp_raw, spearman_cal=sp_cal,
                    nll_raw=gaussian_nll(mu_te, sd_te, y_te),
                    nll_cal=gaussian_nll(mu_te, sd_te * T, y_te),
                    mean_sd_raw=float(sd_te.mean()),
                    qrf_emulated=(name == 'QRF' and not HAS_QRF),
                    gp_subsampled=(name == 'GP' and len(train) > GP_MAX_N),
                    fit_secs=time.time() - t0))
                r = rows[-1]
                print(f'  {name:8s} sigma={sigma:.1f}  R2={r["r2"]:+.3f}  T={T:.3f}'
                      f'{" (AT BOUND)" if at_bound else ""}  '
                      f'cov1 raw={r["cov_raw_1sd"]:.3f} cal={r["cov_cal_1sd"]:.3f}  '
                      f'rho raw={sp_raw:+.3f} cal={sp_cal:+.3f}  '
                      f'{r["fit_secs"]:6.1f}s', flush=True)
                # Written after every single fit. This machine is heavily
                # loaded and a run may be interrupted; a partial file that can
                # be read is worth more than a complete one that never lands.
                _flush(rows, incremental_path)
    return pd.DataFrame(rows)


def _flush(rows, path):
    if path:
        pd.DataFrame(rows).to_csv(path, index=False)


NOMINAL_1SD, NOMINAL_2SD = 0.6827, 0.9545


def report(df):
    ok = df[df.failed == ''] if 'failed' in df else df
    if not len(ok):
        print('every fit failed — nothing to report')
        return pd.DataFrame()

    print('\n' + '=' * 78)
    print('DOES THE MULTIPLIER TRANSFER? coverage at 1 sd, nominal 0.6827')
    print('=' * 78)
    piv = ok.pivot_table(index=['model', 'sigma'],
                         values=['cov_raw_1sd', 'cov_cal_1sd', 'cov_val_cal_1sd',
                                 'temperature'], aggfunc='mean')
    piv['|raw-nominal|'] = (piv.cov_raw_1sd - NOMINAL_1SD).abs()
    piv['|cal-nominal|'] = (piv.cov_cal_1sd - NOMINAL_1SD).abs()
    piv['cal_closer'] = piv['|cal-nominal|'] < piv['|raw-nominal|']
    print(piv[['temperature', 'cov_raw_1sd', 'cov_val_cal_1sd', 'cov_cal_1sd',
               '|raw-nominal|', '|cal-nominal|', 'cal_closer']].round(4).to_string())

    print('\n' + '=' * 78)
    print('DOES THE MULTIPLIER DRIFT WITH THE NOISE LEVEL?')
    print('=' * 78)
    for model, g in ok.groupby('model'):
        by_sigma = g.groupby('sigma').temperature.mean()
        seed_sd = g.groupby('sigma').temperature.std().mean()
        drift = by_sigma.max() - by_sigma.min()
        print(f'  {model:8s} T by noise level: '
              + '  '.join(f'{s:.1f}:{t:.2f}' for s, t in by_sigma.items())
              + f'   spread across noise levels {drift:.2f}'
              + f'   typical spread across seeds {seed_sd:.2f}'
              + ('   << DRIFTS' if drift > 3 * max(seed_sd, 1e-6) else ''))
    if ok.temperature_at_bound.any():
        n = int(ok.temperature_at_bound.sum())
        print(f'\n  ⚠ {n} of {len(ok)} fits hit the [{T_BOUNDS[0]}, {T_BOUNDS[1]}] '
              f'bound — calibration gave up there:')
        print(ok[ok.temperature_at_bound][['model', 'sigma', 'seed', 'temperature']]
              .to_string(index=False))

    print('\n' + '=' * 78)
    print('RANKING IS UNTOUCHED — Spearman(uncertainty, |error|), raw vs calibrated')
    print('=' * 78)
    d = (ok.spearman_raw - ok.spearman_cal).abs()
    print(f'  max absolute difference across all {len(ok)} fits: {d.max():.2e}')
    print('  confirms the multiplier cannot change either rank-based answer'
          if d.max() < 1e-9 else '  ⚠ ranking moved — investigate')

    print('\n' + '=' * 78)
    print('VERDICT against the rule fixed before the run')
    print('=' * 78)
    all_closer = bool(piv.cal_closer.all())
    drifts = {m: (g.groupby('sigma').temperature.mean().max()
                  - g.groupby('sigma').temperature.mean().min())
              for m, g in ok.groupby('model')}
    worst_drift = max(drifts.values())
    print(f'  calibrated closer to nominal in every model x noise level: {all_closer}')
    print(f'  largest drift of the multiplier across noise levels: {worst_drift:.2f}')
    if all_closer and worst_drift <= 1.0:
        print('  => CALIBRATED is defensible as the primary number.')
    else:
        print('  => RAW as primary; calibrated kept as a labelled secondary column.')
        if not all_closer:
            bad = piv[~piv.cal_closer]
            print('     it failed to transfer here:')
            print(bad[['cov_raw_1sd', 'cov_cal_1sd']].round(4).to_string())
    return piv.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sized for a loaded laptop. The question here -- does a multiplier fitted
    # on validation still hold on test, and does it drift with the noise level
    # -- is about transfer, not about a precise R2, so it does not need the
    # production molecule count. 3,000 molecules leaves ~450 test molecules per
    # seed, which puts the standard error on a coverage estimate near 2%.
    ap.add_argument('-n', '--n-molecules', type=int, default=3000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--rep', default='descriptors')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.quick:
        args.n_molecules, args.seeds, args.sigmas = 1500, [0], [0.0, 0.6]

    print(json.dumps(env_fingerprint(), indent=2), flush=True)
    if not HAS_QRF:
        print(f'\nNOTE quantile forest is EMULATED here: {QRF_UNAVAILABLE_REASON}\n',
              flush=True)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out or ('calibration' + ('_quick' if args.quick else '')
                        + f'_n{args.n_molecules}')
    out_csv = OUT_DIR / f'{stem}.csv'
    df = run(args.n_molecules, args.seeds, args.sigmas, args.models, args.rep,
             incremental_path=out_csv)
    df.to_csv(out_csv, index=False)
    summary = report(df)
    if len(summary):
        summary.to_csv(OUT_DIR / f'{stem}_summary.csv', index=False)
    (OUT_DIR / f'{stem}_env.json').write_text(json.dumps(env_fingerprint(), indent=2))
    print(f'\nwrote {OUT_DIR / f"{stem}.csv"}')
    print(f'total {time.time() - t0:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
