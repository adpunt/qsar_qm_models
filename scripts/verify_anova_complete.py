#!/usr/bin/env python3
"""
Verify ALL experiments for paper figures are complete.

Usage:
  python verify_anova_complete.py /path/to/results
  python verify_anova_complete.py /path/to/results --validation-dir /path/to/kirby/results/validation

WHAT THIS EXPECTS, AND WHERE IT GETS IT
---------------------------------------
Nothing about the design is written down here. The noise conditions, their level
grids, the representations and the model roster are READ from the QM9 job
generator, which in turn reads noise_conditions.json.

Everything used to be a literal in this file, and every literal was retired
between 2026-08-26 and 2026-08-27 without this file noticing:

  * six condition names (legacy, valprop, quantile, threshold, outlier, hetero)
    that no injector produces any more, so every real file was counted MISSING
  * an eleven-level grid 0.0..1.0 in steps of 0.1, against a settled grid of
    0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5 -- so a complete run reported six missing
    levels per cell, and a genuinely absent 0.75 or 1.5 could not be seen at all
  * `smiles` and `mol2vec` among the representations, both deleted, the same
    mistake the figure script's own gap audit had and had fixed
  * an uncertainty filename that the pipeline has never written

A restated grid is how the two noise injectors drifted apart for the life of the
project. This file does not restate one.
"""

import sys
import argparse
import importlib.util
from pathlib import Path
from collections import defaultdict
import pandas as pd

# =============================================================================
# THE DESIGN, READ FROM THE JOB GENERATOR
# =============================================================================

_GENERATOR = (Path(__file__).resolve().parent.parent
              / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py')


def _load_generator():
    """Import the QM9 job generator as a module, for its settled sets.

    It raises SystemExit at import if its own condition list and
    noise_conditions.json disagree, which is the behaviour wanted here too: an
    audit against a design nobody agreed on is worse than no audit.
    """
    spec = importlib.util.spec_from_file_location('qm9_job_generator', _GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()

# Condition -> the FULL level grid, including the clean level. CONDITION_FLAGS
# carries the full grid; CONDITIONS strips the clean level from every condition
# but the reference one, because the array does not re-run it -- copy_zero_rows.py
# fills it in afterwards. A finished results tree therefore has it everywhere, so
# a missing clean level here means that copy step has not been run yet, which is
# worth being told.
LEVELS_BY_CONDITION = {
    name: [float(v) for v in _GEN.CONDITION_FLAGS[name][1].split()]
    for name in _GEN.CONDITIONS
}

# Conditions the settled file runs on a handful of model-and-representation pairs
# rather than the whole grid -- censoring today. Expecting them everywhere would
# manufacture a gap for every pair that was never meant to run, which is exactly
# the failure this file had with the retired condition names.
PAIR_SUBSET_CONDITIONS = set(_GEN.PAIR_SUBSET_CONDITIONS)
# Depth-only conditions run on a narrower selection too, so they are reported
# but not required.
FULL_GRID_CONDITIONS = [c for c in _GEN.STAGE1_CONDITIONS
                        if c not in PAIR_SUBSET_CONDITIONS]
OTHER_CONDITIONS = [c for c in _GEN.CONDITIONS if c not in FULL_GRID_CONDITIONS]

REPS = list(_GEN.ALL_REPS)
MIN_ITERATIONS = 10

# The roster, and which representations each model runs on -- the Tanimoto GP is
# only defined on binary fingerprints, so it is not "missing" on the other four.
MODEL_REPS = {name: list(spec[4]) for name, spec in _GEN.MODELS.items()}
ALL_MODELS = list(_GEN.MODELS)
# Off by default in the generator, so a file for one of these is not expected.
EXCLUDED_MODELS = list(_GEN.EXCLUDED_MODELS)

# Models asked to emit per-molecule uncertainty, read off the flags the generator
# passes rather than listed again here.
UNCERTAINTY_MODELS = [name for name, spec in _GEN.MODELS.items()
                      if '-u True' in spec[0]]

# Validation datasets: the three directory names the laboratory runner creates
# (KIRBy tests/alternative_data_noise_robustness.py, --results-root / 'logd',
# 'caco2', 'herg').
VALIDATION_DATASETS = ['logd', 'caco2', 'herg']


def check_file(filepath, expected_sigmas):
    """Check if a CSV has all expected sigmas with sufficient iterations."""
    try:
        df = pd.read_csv(filepath)
        if 'sigma' not in df.columns:
            return {'exists': True, 'rows': len(df), 'complete': False, 'error': 'no sigma column'}

        found_sigmas = set(df['sigma'].unique())
        missing = set(expected_sigmas) - found_sigmas
        min_iters = df.groupby('sigma').size().min() if len(df) > 0 else 0

        return {
            'exists': True,
            'rows': len(df),
            'sigmas_found': len(found_sigmas),
            'sigmas_missing': sorted(missing),
            'min_iterations': min_iters,
            'complete': len(missing) == 0 and min_iters >= MIN_ITERATIONS,
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def get_anova_filename(condition, rep, model):
    """The result filename the job scripts write.

    `../results/anova_${cond}_${rep}_${model}.csv` in the generator's template,
    so this is the one spelling both sides agree on.
    """
    return f"anova_{condition}_{rep}_{model}.csv"


def get_uncertainty_filename(condition, rep, model):
    """The per-molecule uncertainty file that sits beside the result file.

    save_uncertainty_values (scripts/utils.py) writes the result path with
    '.csv' replaced by '_uncertainty_values.csv'. This file used to look for
    `uncertainty_<condition>_<rep>_<model>_uncertainty_values.csv` and for
    `<condition>_<rep>_<model>_uncertainty_values.csv`; the pipeline has never
    written either, so every uncertainty cell was reported missing whatever was
    on disk.
    """
    return get_anova_filename(condition, rep, model).replace(
        '.csv', '_uncertainty_values.csv')


def main():
    parser = argparse.ArgumentParser(description="Verify data completeness")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--validation-dir", type=str, default=None,
                        help="Path to KIRBy validation results")
    parser.add_argument("--conditions", nargs='+', default=None,
                        choices=sorted(LEVELS_BY_CONDITION),
                        help="Only audit these noise conditions. Default: the "
                             "ones the main grid runs on every pair.")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Also expect files for the models the job "
                             "generator excludes by default")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    conditions = args.conditions or FULL_GRID_CONDITIONS
    models_to_check = ALL_MODELS + (EXCLUDED_MODELS if args.include_excluded else [])

    # =========================================================================
    # 1. ANOVA DATA
    # =========================================================================
    print("=" * 70)
    print("ANOVA DATA VERIFICATION")
    print("=" * 70)
    print(f"\nDesign read from {_GENERATOR.name} (which reads noise_conditions.json)")
    print(f"  conditions audited: {conditions}")
    for c in conditions:
        print(f"    {c}: levels {LEVELS_BY_CONDITION[c]}")
    print(f"  representations: {REPS}")
    print(f"  models: {len(models_to_check)}")
    if OTHER_CONDITIONS:
        print(f"  NOT audited, because they do not run on every pair: "
              f"{OTHER_CONDITIONS}")

    complete = []
    incomplete = []
    missing = []

    for condition in conditions:
        sigmas = LEVELS_BY_CONDITION[condition]
        for model in models_to_check:
            # The Tanimoto GP is only defined on binary fingerprints, so it is
            # not missing on the other four -- it was never asked for.
            for rep in MODEL_REPS.get(model, REPS):
                filename = get_anova_filename(condition, rep, model)
                filepath = results_dir / filename
                status = check_file(filepath, sigmas)

                entry = (condition, model, rep, filename)
                if not status.get('exists'):
                    missing.append(entry)
                elif not status.get('complete'):
                    incomplete.append((*entry, status))
                else:
                    complete.append(entry)

    total = len(complete) + len(incomplete) + len(missing)
    print(f"\nTotal configurations: {total}")
    print(f"Complete:   {len(complete):4d} ({100*len(complete)/total:.1f}%)")
    print(f"Incomplete: {len(incomplete):4d}")
    print(f"Missing:    {len(missing):4d}")

    if missing:
        print(f"\n--- MISSING ANOVA FILES ({len(missing)}) ---")
        by_model = defaultdict(list)
        for cond, model, rep, fname in missing:
            by_model[model].append((cond, rep))

        for model in models_to_check:
            if model in by_model:
                items = by_model[model]
                conds = sorted(set(c for c, r in items))
                reps = sorted(set(r for c, r in items))
                print(f"  {model}: {len(items)} gaps — conditions={conds}, reps={reps}")

    if incomplete:
        print(f"\n--- INCOMPLETE ANOVA FILES ({len(incomplete)}) ---")
        for cond, model, rep, fname, status in incomplete[:20]:
            ms = status.get('sigmas_missing', [])
            mi = status.get('min_iterations', 0)
            print(f"  {fname}: missing_levels={ms}, min_replicates={mi}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
        print("  (a missing clean level 0.0 on any condition but the reference "
              "one means copy_zero_rows.py has not been run yet)")

    # =========================================================================
    # 2. UNCERTAINTY DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("UNCERTAINTY DATA VERIFICATION")
    print("=" * 70)

    unc_complete = []
    unc_missing = []

    unc_models = [m for m in UNCERTAINTY_MODELS if m in models_to_check]

    for condition in conditions:
        for model in unc_models:
            for rep in MODEL_REPS.get(model, REPS):
                fname = get_uncertainty_filename(condition, rep, model)
                if (results_dir / fname).exists():
                    unc_complete.append((condition, model, rep, fname))
                else:
                    unc_missing.append((condition, model, rep))

    unc_total = len(unc_complete) + len(unc_missing)
    if unc_total > 0:
        print(f"\nTotal uncertainty configs: {unc_total}")
        print(f"Complete:   {len(unc_complete):4d} ({100*len(unc_complete)/unc_total:.1f}%)")
        print(f"Missing:    {len(unc_missing):4d}")

        if unc_missing:
            print(f"\n--- MISSING UNCERTAINTY FILES ({len(unc_missing)}) ---")
            by_model = defaultdict(list)
            for cond, model, rep in unc_missing:
                by_model[model].append((cond, rep))
            for model in unc_models:
                if model in by_model:
                    items = by_model[model]
                    print(f"  {model}: {len(items)} gaps")

    # =========================================================================
    # 3. VALIDATION DATA
    # =========================================================================
    if args.validation_dir:
        val_dir = Path(args.validation_dir)
        print("\n" + "=" * 70)
        print("VALIDATION DATA VERIFICATION")
        print("=" * 70)

        for ds in VALIDATION_DATASETS:
            ds_dir = val_dir / ds
            if ds_dir.exists():
                all_results = ds_dir / 'all_results.csv'
                summary = ds_dir / 'summary.csv'
                if all_results.exists():
                    df = pd.read_csv(all_results)
                    models_found = df['model'].unique() if 'model' in df.columns else []
                    print(f"\n  {ds}/all_results.csv: {len(df)} rows, models={sorted(models_found)}")
                elif summary.exists():
                    df = pd.read_csv(summary)
                    print(f"\n  {ds}/summary.csv: {len(df)} rows (summary only)")
                else:
                    print(f"\n  {ds}/: directory exists but no results files!")
            else:
                print(f"\n  {ds}/: MISSING")

    # =========================================================================
    # 4. WHAT TO DO WITH THE GAPS
    # =========================================================================
    # This file used to write the gap-fill job scripts itself. It emitted
    # `--sigma` and `--noise-strategy value_proportional`, and both flags are now
    # refused by name at the top of process_and_train.py: every script it wrote
    # would have died on its first line. Rebuilding the emitter here would mean
    # two generators for one grid, which is the arrangement that let this file's
    # condition names and level grid go a year out of date unnoticed. The gap
    # list is printed instead, for slurm_scripts_qm9_rerun/generate_scripts.py to
    # be pointed at.
    if missing or incomplete:
        gap_conditions = sorted({c for c, *_ in missing}
                                | {c for c, *_ in incomplete})
        gap_models = sorted({m for _, m, *_ in missing}
                            | {m for _, m, *_ in incomplete})
        gap_reps = sorted({r for _, _, r, *_ in missing}
                          | {r for _, _, r, *_ in incomplete})
        print("\n" + "=" * 70)
        print("TO FILL THESE GAPS")
        print("=" * 70)
        print("\n  python slurm_scripts_qm9_rerun/generate_scripts.py \\")
        print(f"      --conditions {' '.join(gap_conditions)} \\")
        print(f"      --models {' '.join(gap_models)} \\")
        print(f"      --reps {' '.join(gap_reps)}")
        print("\n  then follow slurm_scripts_qm9_rerun/RUNBOOK.md. That generator "
              "owns the\n  flag spelling, the level grid and the array shape; this "
              "file only audits.")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    all_ok = len(missing) == 0 and len(incomplete) == 0
    if all_ok:
        print("ALL ANOVA DATA COMPLETE")
    else:
        gaps = len(missing) + len(incomplete)
        print(f"TOTAL GAPS: {gaps} ANOVA files need filling")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
