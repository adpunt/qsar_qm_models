#!/usr/bin/env python
"""Executable checks for the QM9 uncertainty writer and its two dead call sites.

Chat F, agent C. Everything here RUNS the code -- nothing greps the source. The
project has already shipped a smoke test whose checks searched the file as text,
and it hid a live bug for two days.

What is under test:

  C1  save_uncertainty_values used to reconstruct the injected noise by regressing
      the noisy label on the clean one and keeping the residuals. Held-out labels
      are no longer noised, so those residuals were identically zero and the column
      was dead. The regression is deleted; the noise is now RECORDED. A test row
      gets exactly 0.0, a training row must be handed the injector's value, and a
      training row without it raises. Six new columns join the schema.

  C3  scripts/noise_mitigation.py called save_uncertainty_values with a keyword
      `y_true=` that is not a parameter, omitted three required ones, and never
      imported the function in the first place -- and the whole call sits inside a
      bare `except Exception` that returns {'r2': -999}, which is why nobody ever
      saw it fail. The check below runs that branch for real and refuses both the
      sentinel and the missing file.

Run it:

    python scripts/test_uncertainty_writer.py

Exits non-zero on any failure.
"""
import os
import sys
import csv
import shutil
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils import save_uncertainty_values, UNCERTAINTY_COLUMNS  # noqa: E402

FAILURES = []


def check(label, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ''))
    if not ok:
        FAILURES.append(label)
    return 0 if ok else 1


def raises(label, exc_type, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc_type as e:
        return check(label, True, f"{type(e).__name__}: {str(e)[:60]}")
    except Exception as e:  # noqa: BLE001 - wrong exception is still a failure
        return check(label, False, f"raised {type(e).__name__} instead of {exc_type.__name__}")
    return check(label, False, "did not raise")


def read_back(path):
    """Read a written uncertainty CSV so float64 survives the round trip exactly."""
    return pd.read_csv(path, float_precision='round_trip')


def unc_path(results_csv):
    return results_csv.replace('.csv', '_uncertainty_values.csv')


# Values chosen to be hostile to any lossy float formatting: full-mantissa
# doubles, a subnormal, a value that is not representable in float32, and a
# negative near-zero.
HOSTILE = np.array([
    0.1234567890123456789,
    -3.141592653589793e-17,
    np.nextafter(1.0, 2.0) - 1.0,
    5e-324,
    1.7976931348623157e+300,
    -0.30000000000000004,
    123456.78901234567,
    np.float64(np.float32(0.7)) + 1e-17,
], dtype=np.float64)


def make_arrays(n):
    rng = np.random.default_rng(0)
    y_true = rng.normal(0.0, 1.0, n)
    y_pred = y_true + rng.normal(0.0, 0.2, n)
    y_std = np.abs(rng.normal(0.5, 0.1, n))
    return y_true, y_pred, y_std


# ---------------------------------------------------------------------------
# C1 -- the writer itself
# ---------------------------------------------------------------------------

def test_writer(tmp):
    print("\nC1  save_uncertainty_values: the noise is recorded, not reconstructed")

    # ---- a test row -------------------------------------------------------
    # y_true_noisy is deliberately an affine function of y_true PLUS a residual
    # that the deleted regression would have recovered. The old code needed ten
    # or more finite points to take its linregress branch, so n is well above
    # that: if the regression ever came back, this frame's injected_noise would
    # be that residual and not zero.
    n = 40
    y_true, y_pred, y_std = make_arrays(n)
    residual = np.random.default_rng(11).normal(0.0, 0.4, n)
    y_noisy_confounded = 2.5 * y_true - 0.75 + residual
    check("the affine confound the old regression would have fitted is non-trivial",
          float(np.abs(residual).max()) > 1e-3,
          f"max |residual| = {np.abs(residual).max():.6g}")

    f_test = os.path.join(tmp, 'res_test.csv')
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_noisy_confounded,
        filepath=f_test, model_name='qrf', rep='pdv',
        sigma_noise=0.6, iteration=0, file_no=1)
    df = read_back(unc_path(f_test))

    check("a test row writes injected_noise as exactly 0.0, not the regression residual",
          bool((df['injected_noise'].to_numpy() == 0.0).all()),
          f"max |injected_noise| = {np.abs(df['injected_noise'].to_numpy()).max():.6g}")
    check("a test row writes oof_folds_ok = -1",
          bool((df['oof_folds_ok'].to_numpy() == -1).all()))
    check("a test row writes split = 'test'",
          bool((df['split'] == 'test').all()))
    check("the file has exactly the declared column set, in order",
          list(df.columns) == UNCERTAINTY_COLUMNS,
          f"{len(df.columns)} columns")
    for col in ('split', 'canonical_smiles', 'noise_scale', 'noise_pattern',
                'noise_pattern_pred', 'oof_folds_ok'):
        check(f"new column '{col}' reaches the CSV", col in df.columns)
    for col in ('canonical_smiles', 'noise_scale', 'noise_pattern', 'noise_pattern_pred'):
        check(f"'{col}' is NaN when it was not supplied", bool(df[col].isna().all()))
    check("existing columns keep their names and values",
          np.array_equal(df['y_true_original'].to_numpy(), y_true) and
          np.array_equal(df['y_pred_mean'].to_numpy(), y_pred))

    # ---- a training row with the injector's recorded value ----------------
    # HOSTILE is the bit-exactness payload; the arrays are resized to match it.
    n = len(HOSTILE)
    y_true, y_pred, y_std = make_arrays(n)
    f_oof = os.path.join(tmp, 'res_oof.csv')
    smiles = [f'C{"C" * i}O' for i in range(n)]
    pattern = np.linspace(0.25, 2.0, n)
    scale = 0.6 * pattern
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_true + HOSTILE,
        filepath=f_oof, model_name='qrf', rep='pdv',
        sigma_noise=0.6, iteration=0, file_no=1,
        split='train_oof', injected_noise=HOSTILE,
        canonical_smiles=smiles, noise_scale=scale,
        noise_pattern=pattern, noise_pattern_pred=pattern * 1.1,
        oof_folds_ok=np.full(n, 5))
    dfo = read_back(unc_path(f_oof))

    got = dfo['injected_noise'].to_numpy(dtype=np.float64)
    bitwise = got.tobytes() == HOSTILE.tobytes()
    check("a training row writes the supplied injected_noise verbatim, to the last bit",
          bitwise,
          "bit-identical" if bitwise else f"max abs diff {np.abs(got - HOSTILE).max():.3g}")
    check("a training row writes split = 'train_oof'", bool((dfo['split'] == 'train_oof').all()))
    check("canonical_smiles round-trips", list(dfo['canonical_smiles']) == smiles)
    check("noise_pattern round-trips bit-for-bit",
          dfo['noise_pattern'].to_numpy(dtype=np.float64).tobytes() == pattern.tobytes())
    check("noise_scale round-trips bit-for-bit",
          dfo['noise_scale'].to_numpy(dtype=np.float64).tobytes() == scale.tobytes())
    check("noise_pattern_pred round-trips bit-for-bit",
          dfo['noise_pattern_pred'].to_numpy(dtype=np.float64).tobytes()
          == (pattern * 1.1).tobytes())
    check("oof_folds_ok round-trips as the supplied fold count",
          bool((dfo['oof_folds_ok'].to_numpy() == 5).all()))

    # ---- the refusals -----------------------------------------------------
    f_bad = os.path.join(tmp, 'res_bad.csv')
    raises("a training row with no recorded noise raises", ValueError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=f_bad, model_name='qrf', rep='pdv',
           sigma_noise=0.6, iteration=0, file_no=1, split='train_oof')
    check("...and it wrote no file while refusing", not os.path.exists(unc_path(f_bad)))

    raises("an unknown split raises", ValueError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=f_bad, model_name='qrf', rep='pdv',
           sigma_noise=0.6, iteration=0, file_no=1, split='train')

    raises("a per-molecule column of the wrong length raises", ValueError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=f_bad, model_name='qrf', rep='pdv',
           sigma_noise=0.6, iteration=0, file_no=1,
           noise_pattern=np.arange(n - 1, dtype=float))

    # ---- appending --------------------------------------------------------
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_true + HOSTILE,
        filepath=f_oof, model_name='qrf', rep='pdv',
        sigma_noise=0.0, iteration=0, file_no=1,
        split='train_oof', injected_noise=np.zeros(n),
        canonical_smiles=smiles, noise_pattern=pattern)
    dfa = read_back(unc_path(f_oof))
    check("a second call appends rather than overwriting, under one header",
          len(dfa) == 2 * n and set(dfa['sigma']) == {0.6, 0.0},
          f"{len(dfa)} rows")
    check("noise_pattern is identical at both noise levels, including zero",
          np.array_equal(
              dfa[dfa['sigma'] == 0.6]['noise_pattern'].to_numpy(),
              dfa[dfa['sigma'] == 0.0]['noise_pattern'].to_numpy()))
    check("the zero level carries exactly zero injected noise",
          bool((dfa[dfa['sigma'] == 0.0]['injected_noise'].to_numpy() == 0.0).all()))

    # A file left over from before the new columns must not be appended to.
    f_stale = os.path.join(tmp, 'res_stale.csv')
    with open(unc_path(f_stale), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(UNCERTAINTY_COLUMNS[:15])   # the pre-2026-08-26 header
        w.writerow([''] * 15)
    raises("appending to a file with the old header is a hard error", RuntimeError,
           save_uncertainty_values,
           y_pred_mean=y_pred, y_pred_std=y_std,
           y_true_original=y_true, y_true_noisy=y_true,
           filepath=f_stale, model_name='qrf', rep='pdv',
           sigma_noise=0.6, iteration=0, file_no=1)

    # ---- a scalar broadcasts ---------------------------------------------
    f_scalar = os.path.join(tmp, 'res_scalar.csv')
    save_uncertainty_values(
        y_pred_mean=y_pred, y_pred_std=y_std,
        y_true_original=y_true, y_true_noisy=y_true,
        filepath=f_scalar, model_name='ngboost', rep='ecfp4',
        sigma_noise=0.3, iteration=0, file_no=2,
        canonical_smiles='CCO', noise_scale=0.3, oof_folds_ok=3)
    dfs = read_back(unc_path(f_scalar))
    check("a scalar per-molecule value broadcasts to every row",
          bool((dfs['canonical_smiles'] == 'CCO').all()
               and (dfs['noise_scale'] == 0.3).all()
               and (dfs['oof_folds_ok'] == 3).all()))


# ---------------------------------------------------------------------------
# C3 -- the call site in noise_mitigation.py, executed for real
# ---------------------------------------------------------------------------

def test_noise_mitigation_call_site(tmp):
    print("\nC3  scripts/noise_mitigation.py: the gauche call site actually runs")

    import noise_mitigation as nm

    check("noise_mitigation imports and now has save_uncertainty_values in scope",
          hasattr(nm, 'save_uncertainty_values'))

    # GAUCHE_AVAILABLE is False on this laptop (one import in its try-block fails)
    # and `fit_gpytorch_model` is not bound in the module at all, so the branch
    # cannot reach the call site unaided. Supply exactly those two and nothing
    # else -- the call site's own lines run unmodified.
    saved_flag = nm.GAUCHE_AVAILABLE
    had_fit = hasattr(nm, 'fit_gpytorch_model')
    nm.GAUCHE_AVAILABLE = True
    nm.fit_gpytorch_model = lambda mll: mll.model.double()
    try:
        rng = np.random.default_rng(7)
        n_train, n_test, d = 40, 12, 16
        X_train = (rng.random((n_train, d)) > 0.5).astype(np.float64)
        X_test = (rng.random((n_test, d)) > 0.5).astype(np.float64)
        y_train = rng.normal(0.0, 1.0, n_train)
        y_test = rng.normal(0.0, 1.0, n_test)

        for label, info in (
            ("with file_no supplied",
             {'filepath': os.path.join(tmp, 'nm_with.csv'), 'rep': 'ecfp4',
              'sigma': 0.6, 'iteration': 0, 'file_no': 4}),
            ("without file_no, falling back to 0",
             {'filepath': os.path.join(tmp, 'nm_without.csv'), 'rep': 'ecfp4',
              'sigma': 0.6, 'iteration': 0}),
        ):
            metrics = nm.train_baseline_model(
                X_train, y_train, X_test, y_test, 'gauche',
                save_uncertainty=True, uncertainty_info=info)

            check(f"train_baseline_model returned real metrics, not the swallowed "
                  f"-999 sentinel ({label})",
                  metrics['r2'] != -999, f"r2 = {metrics['r2']:.4g}")

            out = unc_path(info['filepath'])
            wrote = os.path.exists(out)
            check(f"the call site wrote its uncertainty CSV ({label})", wrote)
            if not wrote:
                continue
            df = read_back(out)
            check(f"one row per test molecule ({label})", len(df) == n_test, f"{len(df)} rows")
            check(f"split is 'test' ({label})", bool((df['split'] == 'test').all()))
            check(f"injected_noise is exactly 0.0 ({label})",
                  bool((df['injected_noise'].to_numpy() == 0.0).all()))
            check(f"oof_folds_ok is -1 ({label})",
                  bool((df['oof_folds_ok'].to_numpy() == -1).all()))
            check(f"y_true_original is the clean test label ({label})",
                  np.array_equal(df['y_true_original'].to_numpy(), y_test))
            check(f"file_no is {info.get('file_no', 0)} ({label})",
                  bool((df['file_no'] == info.get('file_no', 0)).all()))
    finally:
        nm.GAUCHE_AVAILABLE = saved_flag
        if not had_fit:
            del nm.fit_gpytorch_model


def main():
    tmp = tempfile.mkdtemp(prefix='unc_writer_check_')
    try:
        test_writer(tmp)
        test_noise_mitigation_call_site(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED -- {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("the uncertainty writer and both call sites are sound")
    return 0


if __name__ == '__main__':
    sys.exit(main())
