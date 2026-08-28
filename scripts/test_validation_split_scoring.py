#!/usr/bin/env python
"""Executable checks for the validation route through the uncertainty writer.

WHAT THIS IS FOR
----------------
A molecule can answer "does the predicted uncertainty find the corrupted
labels?" when two things hold: no model fitted it, and the injector recorded
the noise it received. Out-of-fold scoring buys the first at the price of
`--oof-folds` extra fits per model -- five of every six model fits in the
uncertainty runs were that. Since 2026-08-27 a validation molecule has both for
free: no model stacks validation into its training set, and validation carries
its own independently drawn noise, recorded per molecule in the same provenance
file the training rows come from (RERUN_PLAN.md 13 chat O).

Everything here RUNS the code. Nothing greps the source for a matching line --
this project has already shipped a smoke test that did, and it hid a live bug
for two days. The one check that looks at a function rather than calling it
reads the COMPILED code object's names, so a call commented out or renamed
fails it while a call inside an untaken branch still passes.

Run it:

    python scripts/test_validation_split_scoring.py

Exits non-zero on any failure.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'models'))

from utils import save_uncertainty_values, VALID_SPLITS, CORRUPTED_SPLITS  # noqa: E402

FAILURES = []


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILURES.append(label)


def raises(label, exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type as e:
        check(label, True, f"{type(e).__name__}: {str(e)[:50]}")
    except Exception as e:  # noqa: BLE001
        check(label, False, f"raised {type(e).__name__}, wanted {exc_type.__name__}")
    else:
        check(label, False, "did not raise")


def unc_path(p):
    return p.replace('.csv', '_uncertainty_values.csv')


class Args:
    """The three attributes the scorer reads off the parsed arguments."""

    def __init__(self, filepath, uncertainty=True, score_validation=True):
        self.filepath = filepath
        self.uncertainty = uncertainty
        self.score_validation = score_validation


def write_provenance(tmp, n_train=40, n_val=10, n_test=10, level=1.5):
    """A provenance CSV and manifest in the injector's own format, so the REAL
    TrainingNoiseRecord parses them rather than a stand-in built for the test."""
    rng = np.random.default_rng(20260828)
    rows = []
    smiles = {}
    for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        names = [f'{split[0].upper()}{"C" * (i + 1)}O' for i in range(n)]
        smiles[split] = names
        clean = rng.normal(6.9, 1.3, n)
        pattern = np.full(n, 1.0)
        eps = (rng.normal(0.0, level * 1.3, n) if split != 'test'
               else np.zeros(n))
        scale = (np.full(n, level * 1.3) if split != 'test' else np.zeros(n))
        for i in range(n):
            rows.append({
                'split': split, 'record_index': i, 'canonical_smiles': names[i],
                'y_clean_raw': clean[i], 'epsilon_raw': eps[i],
                'noise_scale_raw': scale[i], 'noise_pattern_raw': pattern[i],
                'y_noisy_raw': clean[i] + eps[i],
                'y_written': (clean[i] + eps[i] - 6.9) / 1.3,
            })
    prov = os.path.join(tmp, 'prov.csv')
    pd.DataFrame(rows).to_csv(prov, index=False)
    man = os.path.join(tmp, 'man.json')
    with open(man, 'w') as f:
        json.dump({'standardisation_mean': 6.9, 'standardisation_sd': 1.3,
                   'noise_targeting': 'uniform', 'noise_type': 'gaussian',
                   'parameters': {}}, f)
    groups = {sm: i % 7 for part in smiles.values() for i, sm in enumerate(part)}
    return prov, man, groups, smiles


def main():
    from process_and_train import TrainingNoiseRecord
    import models as M

    print(__doc__.strip().splitlines()[0])
    print()

    tmp = tempfile.mkdtemp(prefix='valsplit_')

    # ---- the writer accepts the split, and demands the recorded draw --------
    print("the writer")
    check("'validation' is a split the writer accepts",
          'validation' in VALID_SPLITS)
    check("'validation' is a split whose labels were corrupted",
          'validation' in CORRUPTED_SPLITS and 'test' not in CORRUPTED_SPLITS)

    n = 12
    eps = np.linspace(-1.5, 1.5, n)
    y_true = np.linspace(5.0, 8.0, n)
    y_pred = y_true + 0.1
    y_std = np.linspace(0.2, 0.9, n)
    f_val = os.path.join(tmp, 'w.csv')
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_true + eps,
        filepath=f_val, model_name='qrf', rep='pdv',
        sigma_noise=1.5, iteration=0, file_no=1,
        split='validation', injected_noise=eps,
        canonical_smiles=[f'C{"C" * i}O' for i in range(n)],
        noise_scale=np.full(n, 1.95), noise_pattern=np.ones(n),
        noise_pattern_pred=np.ones(n),
        aleatoric_var=np.linspace(0.04, 0.25, n),
        epistemic_var=np.linspace(0.01, 0.09, n))
    df = pd.read_csv(unc_path(f_val), float_precision='round_trip')
    check("a validation row reaches the CSV under split='validation'",
          len(df) == n and bool((df['split'] == 'validation').all()))
    check("its injected_noise is the recorded draw, to the last bit",
          df['injected_noise'].to_numpy(dtype=np.float64).tobytes() == eps.tobytes())
    check("oof_folds_ok is -1: a validation row was not cross-fitted",
          bool((df['oof_folds_ok'].to_numpy() == -1).all()))

    raises("a validation row with no recorded noise raises", ValueError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=os.path.join(tmp, 'bad.csv'), model_name='qrf', rep='pdv',
           sigma_noise=1.5, iteration=0, file_no=1, split='validation')

    # ---- the scorer --------------------------------------------------------
    print("\nthe scorer")
    prov, man, groups, smiles = write_provenance(tmp)
    rec = TrainingNoiseRecord.load(prov, man, groups, level=1.5)
    n_val = len(rec.splits['val']['epsilon_raw'])

    def predict(x):
        # Deterministic, and both components vary per molecule -- which is what
        # SUPPORT says the quantile forest produces, so the writer's own guard
        # has something real to check rather than being stepped around.
        m = np.asarray(x, dtype=float).ravel()
        k = len(m)
        alea = 0.04 + 0.01 * np.arange(k)
        epis = 0.01 + 0.002 * np.arange(k)
        return m, np.sqrt(alea + epis), alea, epis

    f_s = os.path.join(tmp, 's.csv')
    x_val = np.arange(n_val, dtype=float)
    got = M.score_validation_molecules(
        predict, x_val, rec, Args(f_s), 1.5, 'pdv', 0, 1, 'qrf')
    dfs = pd.read_csv(unc_path(f_s), float_precision='round_trip')
    check("the scorer writes one row per validation molecule",
          got == n_val and len(dfs) == n_val, f"{got} rows")
    check("it writes the injector's recorded validation draw, not a reconstruction",
          np.allclose(dfs['injected_noise'].to_numpy(),
                      rec.splits['val']['epsilon_raw']))
    check("it writes the validation molecules, in the provenance's order",
          list(dfs['canonical_smiles']) == list(rec.splits['val']['canonical_smiles']))
    check("the clean label, the draw and the corrupted label are on one scale",
          np.allclose((dfs['y_true_original'] + dfs['injected_noise'] - 6.9) / 1.3,
                      dfs['y_true_noisy'], atol=1e-9))
    check("the condition travels onto the row",
          bool((dfs['noise_type'] == 'gaussian').all()))

    # ---- the refusals ------------------------------------------------------
    print("\nthe refusals")
    f_off = os.path.join(tmp, 'off.csv')
    off = M.score_validation_molecules(
        predict, x_val, rec, Args(f_off, score_validation=False),
        1.5, 'pdv', 0, 1, 'qrf')
    check("without --score-validation it writes nothing",
          off is None and not os.path.exists(unc_path(f_off)))

    f_nou = os.path.join(tmp, 'nou.csv')
    nou = M.score_validation_molecules(
        predict, x_val, rec, Args(f_nou, uncertainty=False),
        1.5, 'pdv', 0, 1, 'qrf')
    check("with uncertainty off it writes nothing",
          nou is None and not os.path.exists(unc_path(f_nou)))

    f_mis = os.path.join(tmp, 'mis.csv')
    raises("scoring a different number of rows than the provenance covers raises",
           RuntimeError, M.score_validation_molecules,
           predict, np.arange(n_val + 3, dtype=float), rec, Args(f_mis),
           1.5, 'pdv', 0, 1, 'qrf')
    check("...and it wrote no file while refusing",
          not os.path.exists(unc_path(f_mis)))

    # ---- the wiring --------------------------------------------------------
    # Read off the COMPILED code object, so a call that is deleted, renamed or
    # commented out fails this while a call in an untaken branch still passes.
    print("\nthe wiring: every trainer that cross-fits also scores validation")
    trainers = [
        ('train_rf_model', 'the two forests'),
        ('train_ngboost_model', 'NGBoost'),
        ('train_gauche_model', 'the Gaussian process'),
        ('train_dnn_model', 'the DNN family'),
        ('train_mlp_variant_model', 'the MLP family'),
        ('train_heteroscedastic_gp', 'the noise-predicting GP'),
    ]
    for name, label in trainers:
        fn = getattr(M, name, None)
        if fn is None:
            check(f"{label} ({name}) exists", False)
            continue
        names = set(fn.__code__.co_names)
        for const in fn.__code__.co_consts:
            if hasattr(const, 'co_names'):
                names |= set(const.co_names)
        cross = 'score_training_molecules_out_of_fold' in names
        val = 'score_validation_molecules' in names
        check(f"{label} scores validation as well as cross-fitting",
              cross and val, f"cross-fit={cross} validation={val}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
