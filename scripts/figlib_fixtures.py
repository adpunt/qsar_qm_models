#!/usr/bin/env python
"""Synthetic runs in the exact schemas the three producers write.

WHY THIS EXISTS
---------------
Nothing from the re-run has landed on this machine: `results/anova_*.csv` is
empty and there is no validation_rerun or uncertainty_rerun tree. So the
analysis is written against the SCHEMAS and proven here, on data whose answer is
known in advance, before it is ever pointed at the cluster.

Every fixture plants an answer, so a decision that fires can be checked against
what was put in:

  * `chemberta` is deliberately poor for `mlp` only -- one representation
    dragging one model's summary (contingent row 6).
  * `laplace` is drawn to be indistinguishable from `gaussian`, and
    `grouped_shifted` to be clearly different -- so the grid clustering that
    decides F3's panel count has both a pair to merge and a pair to keep
    (contingent row 14).
  * one `gauche_rbf` cell is written with `gp_collapsed` set, so the filter that
    nothing applied before has something to catch (contingent row 11).
  * one `mlp` replicate diverges, so the declared filter has something to drop.
  * one cell retains more than it started with, so the above-1 count is not
    always zero (contingent row 10).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'models'))

import figlib_config as C  # noqa: E402

MODELS = ['rf', 'qrf', 'xgboost', 'svm', 'ngboost', 'gauche_rbf',
          'dnn', 'dnn_bnn_full', 'dnn_bnn_full_mve', 'dnn_vbll',
          'mlp', 'mlp_bnn_full']
REPS = ['ecfp4', 'pdv', 'chemberta']

#: How fast each model loses accuracy per unit of noise. The ORDER is the
#: planted answer for every ranking test.
DECAY = {'rf': 0.10, 'qrf': 0.11, 'xgboost': 0.16, 'svm': 0.13,
         'ngboost': 0.09, 'gauche_rbf': 0.20, 'dnn': 0.24,
         'dnn_bnn_full': 0.18, 'dnn_bnn_full_mve': 0.17, 'dnn_vbll': 0.26,
         'mlp': 0.30, 'mlp_bnn_full': 0.22}

BASE_R2 = {'rf': 0.86, 'qrf': 0.85, 'xgboost': 0.83, 'svm': 0.80,
           'ngboost': 0.84, 'gauche_rbf': 0.78, 'dnn': 0.76,
           'dnn_bnn_full': 0.77, 'dnn_bnn_full_mve': 0.77, 'dnn_vbll': 0.72,
           'mlp': 0.70, 'mlp_bnn_full': 0.73}

#: Every condition damages by the same amount EXCEPT grouped_shifted, which is
#: the one zero-mean condition the design says actually separates.
CONDITION_SEVERITY = {'gaussian': 1.00, 'grouped_wider': 1.02,
                      'grouped_shifted': 1.45, 'student_t_nu5': 1.05,
                      'outlier_p10': 0.95, 'laplace': 1.01}

REP_PENALTY = {'ecfp4': 0.00, 'pdv': 0.02, 'chemberta': 0.01}

RESULT_COLUMNS = [
    'sigma', 'iteration', 'model', 'rep', 'sample_size', 'mae', 'mse', 'rmse',
    'r2', 'pearson_corr', 'params_source', 'loss_function', 'spec_version',
    'spec_hash', 'gp_fit_method', 'gp_collapsed', 'noise_type', 'level_units',
    'delivered_dose', 'file_no', 'standardisation_mean', 'standardisation_sd',
]


def _r2(model, rep, condition, sigma, rng, dataset='qm9'):
    base = BASE_R2[model] - REP_PENALTY.get(rep, 0.0)
    # The planted representation outlier: NN-beta collapses on ChemBERTa and
    # nowhere else. This is the case the averaging rules exist to catch.
    if model == 'mlp' and rep == 'chemberta':
        base -= 0.34
    if dataset != 'qm9':
        base -= 0.25
    decay = DECAY[model] * CONDITION_SEVERITY.get(condition, 1.0)
    # AUC_norm is a retention FRACTION, so lowering a model's clean accuracy
    # does not move it -- the metric divides the baseline out by construction
    # (RERUN_PLAN.md 0.6, guard 4). A representation outlier that D5 can see has
    # to be an outlier in how fast the model DEGRADES, not in where it starts.
    # This is the planted case for contingent row 6.
    if model == 'mlp' and rep == 'chemberta':
        decay *= 2.4
    value = base * max(0.0, 1.0 - decay * float(sigma))
    return float(value + rng.normal(0.0, 0.006))


def write_qm9(directory, models=None, reps=None, conditions=None,
              replicates=10, seed=20260905):
    """The QM9 layout: one CSV per (condition, representation, model)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = list(models or MODELS)
    reps = list(reps or REPS)
    conditions = list(conditions or C.SETTLED_CONDITIONS)
    rng = np.random.default_rng(seed)
    written = []

    for condition in conditions:
        levels = C.expected_levels(condition)
        for rep in reps:
            for model in models:
                rows = []
                for replicate in range(replicates):
                    diverged = (model == 'mlp' and rep == 'pdv'
                                and condition == 'gaussian' and replicate == 3)
                    for sigma in levels:
                        r2 = _r2(model, rep, condition, sigma, rng)
                        if diverged and sigma >= 1.0:
                            r2 = -7.4  # a fit that did not train
                        # One cell that scores better with noise added, so the
                        # above-1 count is not always zero.
                        if (model == 'ngboost' and rep == 'ecfp4'
                                and condition == 'outlier_p10'):
                            # Retains MORE than it started with, well past the
                            # 1.05 line, so the above-1 count is a real test
                            # rather than a threshold nothing reaches.
                            r2 = BASE_R2[model] + 0.10 * float(sigma)
                        collapsed = int(model == 'gauche_rbf'
                                        and rep == 'chemberta')
                        rows.append({
                            'sigma': sigma, 'iteration': replicate,
                            'model': model, 'rep': rep, 'sample_size': 3000,
                            'mae': abs(1 - r2) * 0.20,
                            'mse': abs(1 - r2) * 0.05,
                            'rmse': abs(1 - r2) * 0.22,
                            'r2': r2,
                            'pearson_corr': max(0.0, r2) ** 0.5,
                            'params_source': 'default', 'loss_function': 'mse',
                            'spec_version': C.SPEC_VERSION,
                            'spec_hash': C.spec_hash(),
                            'gp_fit_method': 'rbf' if 'gauche' in model else '',
                            'gp_collapsed': collapsed,
                            'noise_type': condition,
                            'level_units': ('fraction_censored'
                                            if condition.startswith('censoring')
                                            else 'label_sd'),
                            'delivered_dose': float(sigma)
                            * (1.0 + rng.normal(0, 0.004)),
                            'file_no': replicate,
                            'standardisation_mean': 0.0,
                            'standardisation_sd': 1.0,
                        })
                path = directory / f'anova_{condition}_{rep}_{model}.csv'
                pd.DataFrame(rows)[RESULT_COLUMNS].to_csv(path, index=False)
                written.append(path)

    # The three siblings the same run writes off the same base path. Two carry
    # the results columns, so only the NAME rule rejects them.
    stem = f'anova_{conditions[0]}_{reps[0]}_{models[0]}'
    pd.DataFrame([dict.fromkeys(RESULT_COLUMNS, 0) | {
        'model': 'MANIFEST_ROW', 'rep': reps[0], 'sigma': 0.0, 'r2': 0.0}]
    ).to_csv(directory / f'{stem}_noise_manifest.csv', index=False)
    pd.DataFrame([{'epoch': 1, 'train_loss': 0.4, 'val_loss': 0.5}]).to_csv(
        directory / f'{stem}_per_epoch.csv', index=False)
    return written


def write_assay(directory, models=None, reps=None, conditions=None,
                datasets=('logd', 'caco2', 'herg'), folds=5, seed=20260906):
    """The assay layout: one directory per (model, rep, dataset), the runner
    writing `<results_root>/<dataset>/all_results.csv`. One fit per cell, seed
    pinned -- the five folds are a PARTITION, not repeats (RERUN_PLAN.md 3.2b).
    """
    directory = Path(directory)
    models = list(models or MODELS)
    reps = list(reps or REPS)
    conditions = list(conditions or ['gaussian', 'grouped_wider',
                                     'grouped_shifted'])
    rng = np.random.default_rng(seed)
    written = []
    for dataset in datasets:
        for rep in reps:
            for model in models:
                rows = []
                for condition in conditions:
                    for sigma in C.expected_levels(condition):
                        for fold in range(folds):
                            rows.append({
                                'dataset': dataset, 'model': model, 'rep': rep,
                                'noise_type': condition, 'sigma': sigma,
                                'fold': fold,
                                'r2': _r2(model, rep, condition, sigma, rng,
                                          dataset=dataset),
                                'rmse': 0.4, 'mae': 0.3, 'spearman': 0.6,
                                'level_units': 'label_sd',
                            })
                out = (directory / f'{model}_{rep}_{dataset}' / dataset)
                out.mkdir(parents=True, exist_ok=True)
                path = out / 'all_results.csv'
                pd.DataFrame(rows).to_csv(path, index=False)
                written.append(path)
    return written


def write_per_molecule(directory, models=None, reps=None, conditions=None,
                       n_molecules=400, folds=3, seed=20260907,
                       dataset='qm9'):
    """The per-molecule rows, in the QM9 schema.

    Two answers are planted. `gauche_rbf` and `dnn_bnn_full_mve` separate their
    two uncertainty components -- the data-driven half climbs with the noise and
    the model half does not. `qrf` fails to separate them: BOTH climb, because
    one bootstrap causes both, which is what the ordinary forest really does.
    And the uncertainty genuinely tracks the injected amount for `gauche_rbf`
    alone, so exactly one model should fire the Q4 trigger.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = list(models or ['qrf', 'ngboost', 'gauche_rbf',
                             'dnn_bnn_full_mve'])
    reps = list(reps or ['ecfp4', 'pdv'])
    conditions = list(conditions or ['gaussian', 'grouped_shifted'])
    rng = np.random.default_rng(seed)

    # Which components each model may claim, mirroring uncertainty_decomposition
    SUPPORT = {
        'qrf': ('per_molecule', 'per_molecule'),
        'ngboost': ('per_molecule', 'none'),
        'gauche_rbf': ('constant', 'per_molecule'),
        'dnn_bnn_full_mve': ('per_molecule', 'per_molecule'),
    }
    DETECTS = {'gauche_rbf'}

    written = []
    for condition in conditions:
        for rep in reps:
            for model in models:
                alea_support, epis_support = SUPPORT[model]
                rows = []
                for sigma in C.expected_levels(condition):
                    for fold in range(folds):
                        y_clean = rng.normal(0.0, 1.0, n_molecules)
                        injected = rng.normal(0.0, float(sigma), n_molecules)
                        err = rng.normal(0.0, 0.25, n_molecules)
                        y_pred = y_clean + err
                        # The data-driven half rises with the noise for
                        # everyone; the model half rises only where the split
                        # fails.
                        alea = 0.20 + 0.55 * float(sigma)
                        epis = 0.18 + (0.50 * float(sigma)
                                       if model == 'qrf' else 0.0)
                        if alea_support == 'per_molecule':
                            alea_col = alea * np.exp(
                                rng.normal(0, 0.25, n_molecules))
                        else:
                            alea_col = np.full(n_molecules, alea)
                        if epis_support == 'per_molecule':
                            epis_col = epis * np.exp(
                                rng.normal(0, 0.25, n_molecules))
                        elif epis_support == 'constant':
                            epis_col = np.full(n_molecules, epis)
                        else:
                            epis_col = np.full(n_molecules, np.nan)
                        total = np.sqrt(np.nan_to_num(alea_col) ** 2
                                        + np.nan_to_num(epis_col) ** 2)
                        if model in DETECTS:
                            total = total * (1.0 + 0.9 * np.abs(injected))
                        rows.append(pd.DataFrame({
                            'model': model, 'representation': rep,
                            'sigma': sigma, 'iteration': fold,
                            'file_no': fold,
                            'sample_idx': np.arange(n_molecules),
                            'y_pred_mean': y_pred,
                            'y_pred_std_uncalibrated': total,
                            'y_true_original': y_clean,
                            'y_true_noisy': y_clean + injected,
                            'injected_noise': injected,
                            'y_pred_std_calibrated': total * 1.1,
                            'temperature': 1.1,
                            'epistemic_uncertainty': epis_col,
                            'aleatoric_uncertainty': alea_col,
                            'aleatoric_support': alea_support,
                            'epistemic_support': epis_support,
                            'split': 'train_oof',
                            'canonical_smiles': [f'C{i}' for i
                                                 in range(n_molecules)],
                            'noise_scale': float(sigma),
                            'noise_pattern': np.abs(injected)
                            / max(float(sigma), 1e-9),
                            'noise_pattern_pred': np.nan,
                            'oof_folds_ok': True,
                            'standardisation_mean': 0.0,
                            'standardisation_sd': 1.0,
                            'noise_type': condition,
                        }))
                frame = pd.concat(rows, ignore_index=True)
                path = (directory /
                        f'anova_{condition}_{rep}_{model}'
                        f'_uncertainty_values.csv')
                frame.to_csv(path, index=False)
                written.append(path)
    return written


def write_all(root, **kwargs):
    root = Path(root)
    qm9 = root / 'qm9'
    assay = root / 'validation_rerun'
    write_qm9(qm9, **kwargs)
    write_assay(assay)
    write_per_molecule(qm9)
    return {'qm9': qm9, 'assay': assay}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root')
    args = ap.parse_args()
    paths = write_all(args.root)
    for name, path in paths.items():
        n = len(list(Path(path).rglob('*.csv')))
        print(f'{name}: {n} files under {path}')
