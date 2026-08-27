#!/usr/bin/env python
"""`rmse` and `mae` are in the label's own units, on both pipelines.

Every QM9 label is standardised in the injector and nothing turned it back, so
those two columns came out in label standard deviations -- while the same two
columns on the experimental side, which keeps raw log units throughout, were in
log units. Two quantities under one name, in two files meant to be read together
(RERUN_PLAN.md 2.18, audit entry 78).

The convention is the label's own units: QM9 work reports each target's error in
eV or meV against chemical accuracy at 0.043 eV, and uses a standardised error
only when averaging across the twelve targets, whose units cannot otherwise be
pooled (Godwin et al., arXiv:2106.07971). One target is trained at a time here.

The conversion multiplies afterwards rather than transforming the arrays. That is
exact, not a shortcut, and this is where that is established: the numbers are
compared against metrics computed directly from the raw arrays.

    python3 scripts/test_metric_units.py
"""
import csv
import os
import sys
import tempfile
import traceback

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils import (RESULT_COLUMNS, calculate_regression_metrics,      # noqa: E402
                   save_results, set_current_noise_type)

MEAN, SD = -6.4321, 1.2345          # a HOMO-LUMO gap in eV, roughly
N = 500


def _labels(seed=0):
    """Raw labels and a prediction, and the same pair standardised."""
    rng = np.random.default_rng(seed)
    raw_true = rng.normal(MEAN, SD, N)
    raw_pred = raw_true + rng.normal(0.0, 0.4, N)
    return raw_true, raw_pred, (raw_true - MEAN) / SD, (raw_pred - MEAN) / SD


def the_error_columns_come_back_in_label_units():
    raw_true, raw_pred, std_true, std_pred = _labels()
    try:
        set_current_noise_type('gaussian', level_units='label_sd',
                               standardisation=(MEAN, SD))
        mae, mse, rmse, r2, pearson = calculate_regression_metrics(std_true, std_pred)
    finally:
        set_current_noise_type(None)

    # Against the metrics computed directly from the raw arrays -- the thing the
    # multiplication is standing in for.
    want_mae = mean_absolute_error(raw_true, raw_pred)
    want_mse = mean_squared_error(raw_true, raw_pred)
    assert np.isclose(mae, want_mae), (
        f"mae {mae:.8f} is not the raw-unit mae {want_mae:.8f}")
    assert np.isclose(mse, want_mse), (
        f"mse {mse:.8f} is not the raw-unit mse {want_mse:.8f}")
    assert np.isclose(rmse, np.sqrt(want_mse)), (
        f"rmse {rmse:.8f} is not the raw-unit rmse {np.sqrt(want_mse):.8f}")

    # And it is a real difference: on this data the standardised rmse is smaller
    # by the label spread, so a check that passed either way would prove nothing.
    assert not np.isclose(rmse, np.sqrt(mean_squared_error(std_true, std_pred))), (
        "the standardised and raw rmse are the same on this fixture, so this "
        "check cannot tell them apart")

    # r2 and the correlation are unchanged by an affine map, and must not move.
    assert np.isclose(r2, r2_score(raw_true, raw_pred)), (
        "r2 moved under the conversion; it is scale-free and must not")
    assert np.isclose(r2, r2_score(std_true, std_pred)), (
        "r2 differs between the two scales, which cannot happen")
    assert -1.0 <= pearson <= 1.0

    print(f"    rmse {rmse:.4f} eV against {np.sqrt(mean_squared_error(std_true, std_pred)):.4f} "
          f"in label spreads")


def with_no_standardisation_nothing_is_converted():
    """The experimental pipeline never standardises, and must not be rescaled."""
    raw_true, raw_pred, _, _ = _labels(seed=1)
    try:
        set_current_noise_type('gaussian', level_units='raw_label',
                               standardisation=(None, None))
        mae, mse, rmse, _, _ = calculate_regression_metrics(raw_true, raw_pred)
    finally:
        set_current_noise_type(None)
    assert np.isclose(mae, mean_absolute_error(raw_true, raw_pred)), (
        "labels that were never standardised were rescaled anyway")
    assert np.isclose(rmse, np.sqrt(mean_squared_error(raw_true, raw_pred)))
    print("    unstandardised labels pass through untouched")


def the_row_says_what_it_was_converted_by():
    raw_true, raw_pred, std_true, std_pred = _labels(seed=2)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'results.csv')
        try:
            set_current_noise_type('gaussian', level_units='label_sd',
                                   standardisation=(MEAN, SD), file_no=7)
            metrics = calculate_regression_metrics(std_true, std_pred)
            save_results(path, 0.5, 0, 'dnn', 'ecfp4', N, metrics)
        finally:
            set_current_noise_type(None)
        with open(path) as f:
            rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    for column in ('standardisation_mean', 'standardisation_sd'):
        assert column in RESULT_COLUMNS, f"{column} is not a results column"
        assert row[column] not in ('', 'None'), (
            f"the row carries no {column}, so nothing on it says which scale "
            f"its rmse is on")
    assert np.isclose(float(row['standardisation_sd']), SD)
    assert np.isclose(float(row['rmse']), np.sqrt(mean_squared_error(raw_true, raw_pred))), (
        "the rmse written to the row is not in label units")
    print(f"    row carries mean {float(row['standardisation_mean']):.4f}, "
          f"sd {float(row['standardisation_sd']):.4f}")


def check(name, fn):
    print(f"  {name}")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print("    ok")
    return True


def main():
    print("error metrics are in the label's own units (RERUN_PLAN.md 2.18)")
    results = [
        check("the error columns come back in label units",
              the_error_columns_come_back_in_label_units),
        check("with no standardisation nothing is converted",
              with_no_standardisation_nothing_is_converted),
        check("the row says what it was converted by",
              the_row_says_what_it_was_converted_by),
    ]
    if not all(results):
        print("\nFAIL: rmse and mae are not in the units the row claims")
        return 1
    print("\nOK: rmse and mae are in label units, and the row carries the conversion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
