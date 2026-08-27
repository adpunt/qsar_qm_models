#!/usr/bin/env python
"""Every QM9 results row carries the condition it was produced under.

Until 2026-08-27 `RESULT_COLUMNS` held no noise-type field, so a row's condition
survived only in the output FILENAME and the figure script had to guess it back
from the stem (RERUN_PLAN.md §2.11). The name is the injector's own -- Rust's
`condition_name` writes it into the manifest, and process_and_train stamps it on
every row of that level rather than composing it a second time from the flags.

Three properties:

  1. the column exists and holds what was set;
  2. a run whose manifest names no condition stops, instead of writing rows that
     cannot be conditioned on;
  3. an old file written without the column is refused on append, not appended to
     ragged.

Run it directly:  python scripts/test_result_row_condition.py
"""

import json
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils  # noqa: E402


METRICS = (0.1, 0.02, 0.14, 0.9, 0.95)


def the_condition_reaches_the_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anova_x.csv")
        utils.set_current_noise_type("grouped_shifted")
        utils.save_results(path, 0.5, 0, "rf", "ecfp4", 100, METRICS)
        utils.set_current_noise_type("censoring_50")
        utils.save_results(path, 0.5, 0, "rf", "ecfp4", 100, METRICS)
        df = pd.read_csv(path)
        assert "noise_type" in df.columns, list(df.columns)
        got = df["noise_type"].tolist()
        assert got == ["grouped_shifted", "censoring_50"], got
        print(f"    two rows, two conditions: {got}")
    utils.set_current_noise_type(None)


def an_explicit_condition_beats_the_stamped_one():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anova_x.csv")
        utils.set_current_noise_type("gaussian")
        utils.save_results(path, 0.0, 0, "rf", "ecfp4", 100, METRICS,
                           noise_type="laplace")
        assert pd.read_csv(path)["noise_type"].tolist() == ["laplace"]
        print("    an explicit noise_type overrides the stamped one")
    utils.set_current_noise_type(None)


def a_manifest_without_a_condition_stops_the_run():
    import process_and_train as P

    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, "noise_manifest.json")
        with open(manifest, "w") as f:
            json.dump({"noise_targeting": "uniform", "parameters": {}}, f)

        class Args:
            dose_units = "spread"
            dataset = "QM9"
            target = "homo_lumo_gap"
            filepath = os.path.join(tmp, "anova_x.csv")

        row = P.record_noise_manifest(Args(), manifest, 0, 1, 0.5)
        assert not (row or {}).get("noise_type"), row

        # The guard process_and_run applies to that row, run for real.
        try:
            P.condition_from_manifest_row(row, manifest, 0.5, 0)
        except RuntimeError as exc:
            assert "no noise_type" in str(exc), str(exc)
            print("    a manifest naming no condition stops the level")
        else:
            raise AssertionError(
                "a manifest with no noise_type was accepted, so the level's "
                "rows would be written with no condition on them")

        # And the same guard passes a manifest that does name one.
        with open(manifest, "w") as f:
            json.dump({"noise_targeting": "uniform", "noise_type": "gaussian",
                       "parameters": {}}, f)
        row = P.record_noise_manifest(Args(), manifest, 1, 2, 0.5)
        assert P.condition_from_manifest_row(row, manifest, 0.5, 1) == "gaussian"
        print("    a manifest that names one is passed through verbatim")


def an_old_file_is_refused_rather_than_appended_ragged():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "anova_old.csv")
        old_header = [c for c in utils.RESULT_COLUMNS if c != "noise_type"]
        with open(path, "w") as f:
            f.write(",".join(old_header) + "\n")
        try:
            utils.save_results(path, 0.0, 0, "rf", "ecfp4", 100, METRICS)
        except RuntimeError as exc:
            assert "noise_type" in str(exc), str(exc)
            print("    a file written without the column is refused on append")
            return
    raise AssertionError("a pre-column file was appended to, producing ragged rows")


def the_manifest_header_covers_every_row():
    """A later level brings parameters the level-zero manifest did not have.

    The writer used to be `csv.DictWriter(f, fieldnames=list(row.keys()))` in
    append mode with the header written once. Fieldnames were recomputed per
    row, so no exception was raised -- the row was written with ITS OWN column
    set. The level loop runs 0.0 first and the injector returns early at level
    zero before inserting lambda, the outlier fraction and the censoring limit,
    so the header came from the narrowest row and every later level appended
    values with no header cells above them (RERUN_PLAN.md §2.11).
    """
    import process_and_train as P

    with tempfile.TemporaryDirectory() as tmp:
        class Args:
            dose_units = "spread"
            dataset = "QM9"
            target = "homo_lumo_gap"
            filepath = os.path.join(tmp, "m.csv")

        narrow = os.path.join(tmp, "m0.json")
        with open(narrow, "w") as f:
            json.dump({"noise_type": "outlier_p10", "noise_level": 0.0,
                       "parameters": {"level": 0.0, "shape": "gaussian"}}, f)
        wide = os.path.join(tmp, "m1.json")
        with open(wide, "w") as f:
            json.dump({"noise_type": "outlier_p10", "noise_level": 0.6,
                       "parameters": {"level": 0.6, "shape": "gaussian",
                                      "lambda": 3.0, "outlier_p": 0.1,
                                      "censor_reference_limit": 9.9}}, f)

        P.record_noise_manifest(Args(), narrow, 0, 1, 0.0)
        P.record_noise_manifest(Args(), wide, 0, 2, 0.6)

        df = pd.read_csv(Args.filepath.replace(".csv", "_noise_manifest.csv"))
        assert len(df) == 2, len(df)
        for col in ("param_lambda", "param_outlier_p",
                    "param_censor_reference_limit"):
            assert col in df.columns, f"{col} has no header cell: {list(df.columns)}"
        row = df[df["noise_level"] == 0.6].iloc[0]
        assert row["param_lambda"] == 3.0, row["param_lambda"]
        assert row["param_outlier_p"] == 0.1, row["param_outlier_p"]
        assert row["param_censor_reference_limit"] == 9.9, row[
            "param_censor_reference_limit"]
        print(f"    {len(df.columns)} columns, and the wide row's values are "
              f"under their own names")


def a_results_row_joins_to_its_noise_manifest():
    """The manifest is keyed on (iteration, file_no, noise_level).

    The results row carried no file_no at all, so it could not be joined to the
    provenance of the noise that produced it. Two more things had to move for the
    join to land: file_no was 64 bits, and a value above 2^63 does not fit in an
    int64, so pandas read the column as float or object on one side; and
    `row.update(manifest)` overwrote noise_level with the injector's value, which
    has been through an f32 -- a level of 0.3 came back as 0.30000001192092896
    (RERUN_PLAN.md §2.11, §2.13).
    """
    import utils as U

    with tempfile.TemporaryDirectory() as tmp:
        class Args:
            dose_units = "spread"
            dataset = "QM9"
            target = "homo_lumo_gap"
            filepath = os.path.join(tmp, "r.csv")

        import process_and_train as P

        for level, file_no in ((0.0, 111), (0.3, 222)):
            manifest = os.path.join(tmp, f"m{file_no}.json")
            with open(manifest, "w") as f:
                # The injector's own f32 round-trip of the level.
                json.dump({"noise_type": "gaussian",
                           "noise_level": float(np.float32(level)),
                           "delivered_dose_in_label_units": level * 1.2,
                           "parameters": {"level": level}}, f)
            row = P.record_noise_manifest(Args(), manifest, 0, file_no, level)
            U.set_current_noise_type(row["noise_type"], level_units="label_sd",
                                     delivered_dose=row.get(
                                         "delivered_dose_in_label_units"),
                                     file_no=file_no)
            U.save_results(Args.filepath, level, 0, "rf", "ecfp4", 100, METRICS)

        results = pd.read_csv(Args.filepath)
        man = pd.read_csv(Args.filepath.replace(".csv", "_noise_manifest.csv"))
        assert str(results["file_no"].dtype) == "int64", results["file_no"].dtype
        assert str(man["file_no"].dtype) == "int64", man["file_no"].dtype
        joined = results.merge(man, left_on=["iteration", "file_no", "sigma"],
                               right_on=["iteration", "file_no", "noise_level"],
                               how="left", suffixes=("", "_m"))
        matched = int(joined["delivered_dose_in_label_units"].notna().sum())
        assert matched == len(results), (
            f"{matched} of {len(results)} results rows joined to their manifest")
        print(f"    {matched} of {len(results)} rows joined on "
              f"(iteration, file_no, sigma)")
    U.set_current_noise_type(None)


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
    print("the QM9 results row carries its condition (RERUN_PLAN.md §2.11)")
    results = [
        check("the condition reaches the row", the_condition_reaches_the_row),
        check("an explicit condition beats the stamped one",
              an_explicit_condition_beats_the_stamped_one),
        check("a manifest without a condition stops the run",
              a_manifest_without_a_condition_stops_the_run),
        check("an old file is refused rather than appended ragged",
              an_old_file_is_refused_rather_than_appended_ragged),
        check("the noise manifest's header covers every row",
              the_manifest_header_covers_every_row),
        check("a results row joins to its noise manifest",
              a_results_row_joins_to_its_noise_manifest),
    ]
    if not all(results):
        print("\nFAIL: a results row can be written with no condition on it")
        return 1
    print("\nOK: every results row names the condition that produced it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
