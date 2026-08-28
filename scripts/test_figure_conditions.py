#!/usr/bin/env python
"""The noise condition a figure-script row belongs to (RERUN_PLAN.md §2.11).

The condition used to survive only in the output FILENAME, and both QM9 loaders
recovered it by matching the stem against six names retired on 2026-08-26.
Three failures under one cause:

  1. a new-scheme file matched nothing, `strategy` stayed blank, and
     `drop_duplicates` treats blanks as equal -- so every condition for one
     (model, rep, level, replicate) collapsed onto whichever file was read last;
  2. `outlier` was in the retired list AND is a prefix of the settled
     `outlier_p10`, so a new contaminated-fraction run pooled with the retired
     value-proportional strategy under one name;
  3. every table that wanted the reference condition wrote
     `frame[frame.strategy == 'legacy'] if 'strategy' in frame else frame`, so a
     frame without the column silently became every condition pooled together.

And separately: the uncertainty column. models/model_defaults.py settles it as
'raw' on 36 measured fits; the figure script picked `y_pred_std_calibrated`
first at four sites, and the raw column's real name was not even a candidate.

Everything here RUNS the loaders on files written to a temporary directory.

Run it directly:  python scripts/test_figure_conditions.py
"""

import os
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_paper_figures_v2 as G  # noqa: E402


RESULT_ROW = dict(
    sigma=0.0, iteration=0, model="rf", rep="ecfp4", sample_size=100,
    mae=0.1, mse=0.02, rmse=0.14, r2=0.9, pearson_corr=0.95,
    params_source="default", loss_function="mse", spec_version="1",
    spec_hash="abc", gp_fit_method="", gp_collapsed="",
)


def write_anova(tmp, name, noise_type=None, r2=0.9):
    row = dict(RESULT_ROW, r2=r2)
    if noise_type is not None:
        row["noise_type"] = noise_type
    pd.DataFrame([row]).to_csv(os.path.join(tmp, name), index=False)


def two_conditions_do_not_collapse_onto_one_row():
    with tempfile.TemporaryDirectory() as tmp:
        write_anova(tmp, "anova_gaussian_ecfp4_rf.csv", "gaussian", r2=0.90)
        write_anova(tmp, "anova_grouped_shifted_ecfp4_rf.csv",
                    "grouped_shifted", r2=0.40)
        df = G.load_anova_data(tmp)
        assert df is not None and len(df) == 2, (
            f"two conditions for one (model, rep, level, replicate) came back "
            f"as {0 if df is None else len(df)} row(s) -- they were "
            f"deduplicated onto one another")
        got = sorted(df["strategy"].tolist())
        assert got == ["gaussian", "grouped_shifted"], got
        print(f"    two conditions, two rows: {got}")


def a_settled_name_is_not_read_as_the_retired_one_it_starts_with():
    # No noise_type column, so the name is all there is -- the case every file
    # written before 2026-08-27 is in.
    with tempfile.TemporaryDirectory() as tmp:
        write_anova(tmp, "anova_outlier_p10_ecfp4_rf.csv", None)
        df = G.load_anova_data(tmp)
        got = df["strategy"].unique().tolist()
        assert got == ["outlier_p10"], (
            f"a contaminated-fraction file was labelled {got}; 'outlier' is the "
            f"RETIRED value-proportional strategy and pooling the two puts "
            f"different mechanisms under one name")
        print(f"    anova_outlier_p10_... -> {got[0]}")


def a_file_naming_no_condition_is_not_left_blank():
    with tempfile.TemporaryDirectory() as tmp:
        write_anova(tmp, "anova_somethingelse_ecfp4_rf.csv", None)
        df = G.load_anova_data(tmp)
        got = df["strategy"].tolist()
        assert all(isinstance(v, str) and v.startswith("unknown_") for v in got), got
        assert df["strategy"].notna().all(), "a blank condition survived"
        print(f"    unnamed file -> {got[0]}")


def the_reference_condition_is_never_the_whole_frame():
    frame = pd.DataFrame({"model": ["rf", "rf"], "auc_norm": [0.5, 0.9]})
    try:
        G.baseline_rows(frame, "test")
    except RuntimeError as exc:
        assert "cannot be selected" in str(exc), str(exc)
        print("    a frame with no condition column is refused, not pooled")
        return
    raise AssertionError(
        "a frame with no condition column was accepted -- every condition "
        "would be pooled under the reference condition's name")


def the_reference_condition_accepts_both_schemes():
    frame = pd.DataFrame({"strategy": ["gaussian", "grouped_shifted", "legacy"],
                          "auc_norm": [1.0, 2.0, 3.0]})
    got = sorted(G.baseline_rows(frame, "test")["strategy"].tolist())
    assert got == ["gaussian", "legacy"], got
    print(f"    reference rows: {got}")


def the_uncertainty_column_is_the_one_model_defaults_settles_on():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "models"))
    from model_defaults import UNCERTAINTY_DEFAULTS
    settled = UNCERTAINTY_DEFAULTS["primary_column"]
    assert G.UNCERTAINTY_PRIMARY == settled, (
        f"the figure script reads {G.UNCERTAINTY_PRIMARY!r} where "
        f"model_defaults settles on {settled!r}")

    both = pd.DataFrame({
        "y_pred_std_uncalibrated": [1.0, 2.0],
        "y_pred_std_calibrated": [10.0, 20.0],
    })
    picked = G.uncertainty_column(both)
    expected = ("y_pred_std_uncalibrated" if settled == "raw"
                else "y_pred_std_calibrated")
    assert picked == expected, (
        f"with both columns present the script reads {picked!r}, not "
        f"{expected!r}")
    print(f"    settled on {settled!r}; with both columns present it reads "
          f"{picked!r}")

    # The experimental side calls it `uncertainty` and writes no calibrated
    # column at all; that must still resolve.
    assert G.uncertainty_column(pd.DataFrame({"uncertainty": [1.0]})) == "uncertainty"


def the_level_axis_is_recorded_and_a_mismatch_is_named():
    """auc_norm is mean retention over each configuration's OWN level range.

    QM9 doses in fractions of its clean label spread and runs to 1.00 of it; the
    experimental grids are raw log units running to 0.84, 1.57 and 1.20 label SD;
    censoring is a fraction of labels clipped. All three were written into one
    column called `sigma` with nothing to tell them apart.
    """
    qm9 = pd.DataFrame({
        "model": ["rf"] * 3, "rep": ["ecfp4"] * 3, "strategy": ["gaussian"] * 3,
        "sigma": [0.0, 0.5, 1.0], "r2": [0.9, 0.7, 0.5],
        "level_units": ["label_sd"] * 3,
    })
    robust, _ = G.calculate_robustness(qm9, baseline_threshold=0.0)
    assert "level_units" in robust.columns, list(robust.columns)
    assert robust["level_units"].tolist() == ["label_sd"], robust["level_units"].tolist()
    assert robust["level_max"].tolist() == [1.0], robust["level_max"].tolist()
    print(f"    QM9 auc row carries {robust['level_units'][0]!r}, top level "
          f"{robust['level_max'][0]:g}")

    val = pd.DataFrame({
        "dataset": ["logd"] * 3, "model": ["rf"] * 3, "rep": ["ecfp4"] * 3,
        "strategy": ["gaussian"] * 3, "sigma": [0.0, 0.5, 1.0],
        "r2": [0.9, 0.7, 0.5], "level_units": ["raw_label"] * 3,
    })
    val_auc = G.calculate_validation_auc(val)
    assert val_auc["level_units"].tolist() == ["raw_label"], val_auc["level_units"].tolist()

    units = G.warn_if_axes_differ([("QM9", robust), ("validation", val_auc)],
                                  "test")
    assert units == {"label_sd", "raw_label"}, units
    print(f"    mixing them is named, not silent: {sorted(units)}")

    same = G.warn_if_axes_differ([("QM9", robust), ("QM9 again", robust)], "test")
    assert same == {"label_sd"}, same


def sibling_files_are_not_read_as_results():
    """`anova_*.csv` matches everything the run writes off the same base path.

    The noise manifest is `<results>_noise_manifest.csv` and the per-epoch
    metrics are `<results>_per_epoch.csv`; both start with `anova_` when the
    results file does. Only `_uncertainty_values` was skipped, so the other two
    were read as results files and concatenated into the results frame
    (RERUN_PLAN.md §2.13).
    """
    with tempfile.TemporaryDirectory() as tmp:
        write_anova(tmp, "anova_gaussian_ecfp4_rf.csv", "gaussian")
        # This manifest carries the results columns too, so ONLY the name rule
        # can reject it. Without that the check would pass on the content rule
        # alone and would not guard the name rule at all -- which is how it was
        # first written, and a revert of the name rule left it green.
        pd.DataFrame([dict(RESULT_ROW, model="manifest_row", iteration=7,
                           file_no=1, noise_level=0.0,
                           noise_type="gaussian")]).to_csv(
            os.path.join(tmp, "anova_gaussian_ecfp4_rf_noise_manifest.csv"),
            index=False)
        pd.DataFrame([{"epoch": 1, "train_loss": 0.4, "val_loss": 0.5}]).to_csv(
            os.path.join(tmp, "anova_gaussian_ecfp4_dnn_per_epoch.csv"),
            index=False)
        pd.DataFrame([{"sigma": 0.0, "something": 1}]).to_csv(
            os.path.join(tmp, "anova_gaussian_ecfp4_odd.csv"), index=False)

        df = G.load_anova_data(tmp)
        assert df is not None and len(df) == 1, (
            f"{0 if df is None else len(df)} rows came back from one results "
            f"file and three siblings")
        assert sorted(df["model"].unique()) == ["rf"], df["model"].unique()
        assert "manifest_row" not in set(df["model"]), (
            "the manifest's row reached the results frame")
        print("    one results file read; the manifest, the per-epoch metrics "
              "and a file without the results columns were skipped")


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
    print("figure-script noise conditions and uncertainty column "
          "(RERUN_PLAN.md §2.11)")
    results = [
        check("two conditions do not deduplicate onto one row",
              two_conditions_do_not_collapse_onto_one_row),
        check("a settled name is not read as the retired name it starts with",
              a_settled_name_is_not_read_as_the_retired_one_it_starts_with),
        check("a file naming no condition is not left blank",
              a_file_naming_no_condition_is_not_left_blank),
        check("the reference condition is never the whole frame",
              the_reference_condition_is_never_the_whole_frame),
        check("the reference condition accepts both schemes",
              the_reference_condition_accepts_both_schemes),
        check("the uncertainty column follows model_defaults",
              the_uncertainty_column_is_the_one_model_defaults_settles_on),
        check("the level axis is recorded and a mismatch is named",
              the_level_axis_is_recorded_and_a_mismatch_is_named),
        check("sibling files are not read as results",
              sibling_files_are_not_read_as_results),
    ]
    if not all(results):
        print("\nFAIL: the figure script can still pool conditions")
        return 1
    print("\nOK: every row carries its condition and the settled column is read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
