#!/usr/bin/env python3
"""
Verify the per-molecule uncertainty files are present and usable.

Usage:
  python verify_uncertainty_complete.py /data/stat-cadd/scat9264/qsar_qm_models/results

WHAT THIS EXPECTS, AND WHERE IT GETS IT
---------------------------------------
Nothing about the design is written down here. The noise conditions, their level
grids, the representations and which models emit an uncertainty at all are READ
from the QM9 job generator, which in turn reads noise_conditions.json.

Four literals used to sit at the top of this file and all four were retired
between 2026-08-26 and 2026-08-27 without it noticing:

  * six condition names -- legacy, valprop, quantile, threshold, outlier, hetero
    -- deleted in noiseInject 1.0.0, so every filename this file looked for was
    one no injector can produce
  * an eleven-level grid 0.0..1.0 in steps of 0.1, against a settled grid of
    0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, so a complete file reported six missing
    levels and a genuinely absent 0.75 or 1.5 could not be seen at all
  * a filename, uncertainty_<condition>_<rep>_<model>.csv, that the pipeline has
    never written: save_uncertainty_values (scripts/utils.py) writes the result
    file's own path with '.csv' replaced by '_uncertainty_values.csv'
  * a required column 'y_true', which does not exist either -- the writer emits
    y_true_original (the clean label) and y_true_noisy (the one the model saw),
    and the distinction is the whole point of the file

Any one of those made every cell fail, so this audit could not have passed on
correct data.
"""

import sys
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
    noise_conditions.json disagree, which is wanted here too: an audit against a
    design nobody agreed on is worse than no audit.
    """
    spec = importlib.util.spec_from_file_location('qm9_job_generator', _GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GEN = _load_generator()

# Condition -> its FULL level grid, the clean level included. CONDITION_FLAGS
# carries the full grid; CONDITIONS strips the clean level from every condition
# but the reference one, because the array does not re-run it and
# copy_zero_rows.py fills it in afterwards.
LEVELS_BY_CONDITION = {
    name: [float(v) for v in _GEN.CONDITION_FLAGS[name][1].split()]
    for name in _GEN.CONDITIONS
}
# Conditions that run on a handful of pairs rather than the whole grid
# (censoring today) would manufacture a gap for every pair never meant to run.
FULL_GRID_CONDITIONS = [c for c in _GEN.STAGE1_CONDITIONS
                        if c not in _GEN.PAIR_SUBSET_CONDITIONS]
OTHER_CONDITIONS = [c for c in _GEN.CONDITIONS if c not in FULL_GRID_CONDITIONS]

REPS = list(_GEN.ALL_REPS)

# Which models were asked for an uncertainty, and on which representations --
# read off the flags the generator passes, not listed again here. The Tanimoto
# GP is only defined on binary fingerprints, so it is not missing on the rest.
UNCERTAINTY_MODEL_REPS = {name: list(spec[4]) for name, spec in _GEN.MODELS.items()
                          if '-u True' in spec[0]}

# The columns the writer emits, scripts/utils.py save_uncertainty_values.
# y_true_original is the CLEAN label; y_true_noisy is the label the model was
# trained on. injected_noise is RECORDED by the injector, never reconstructed.
REQUIRED_COLS = ['sigma', 'y_pred_mean', 'y_true_original', 'y_true_noisy',
                 'injected_noise', 'split']
OPTIONAL_COLS = ['epistemic_uncertainty', 'aleatoric_uncertainty',
                 'y_pred_std_calibrated', 'noise_pattern', 'canonical_smiles']


def uncertainty_filename(condition, rep, model):
    """The file that sits beside anova_<condition>_<rep>_<model>.csv."""
    return f"anova_{condition}_{rep}_{model}_uncertainty_values.csv"


def check_uncertainty_file(filepath, expected_levels):
    """Columns and level coverage for one uncertainty file."""
    try:
        df = pd.read_csv(filepath)

        found = set(df['sigma'].unique()) if 'sigma' in df.columns else set()
        missing_levels = sorted(set(expected_levels) - found)
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]

        return {
            'exists': True,
            'rows': len(df),
            'levels_found': len(found),
            'levels_missing': missing_levels,
            'missing_required_cols': missing_cols,
            'has_epistemic': 'epistemic_uncertainty' in df.columns,
            'has_aleatoric': 'aleatoric_uncertainty' in df.columns,
            'has_injected_noise': 'injected_noise' in df.columns,
            'complete': not missing_levels and not missing_cols,
        }
    except Exception as e:
        return {'exists': False, 'error': str(e)}


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results")

    print("=" * 70)
    print("UNCERTAINTY EXPERIMENT VERIFICATION")
    print("=" * 70)
    print(f"\nDesign read from {_GENERATOR.name} (which reads noise_conditions.json)")
    for c in FULL_GRID_CONDITIONS:
        print(f"  {c}: levels {LEVELS_BY_CONDITION[c]}")
    if OTHER_CONDITIONS:
        print(f"  NOT audited, because they do not run on every pair: "
              f"{OTHER_CONDITIONS}")
    print(f"  models that emit an uncertainty: {sorted(UNCERTAINTY_MODEL_REPS)}")

    all_expected = []
    for condition in FULL_GRID_CONDITIONS:
        for model, model_reps in UNCERTAINTY_MODEL_REPS.items():
            for rep in model_reps:
                all_expected.append(
                    (condition, model, rep,
                     results_dir / uncertainty_filename(condition, rep, model)))

    complete, incomplete, missing = [], [], []
    for condition, model, rep, filepath in all_expected:
        status = check_uncertainty_file(filepath, LEVELS_BY_CONDITION[condition])
        if not status.get('exists'):
            missing.append((condition, model, rep, filepath.name))
        elif not status.get('complete'):
            incomplete.append((condition, model, rep, filepath.name, status))
        else:
            complete.append((condition, model, rep, filepath.name, status))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total = len(all_expected)
    print(f"\nTotal expected: {total}")
    print(f"Complete: {len(complete)} ({100*len(complete)/total:.1f}%)")
    print(f"Incomplete: {len(incomplete)}")
    print(f"Missing: {len(missing)}")

    has_decomp = sum(1 for *_, s in complete
                     if s.get('has_epistemic') and s.get('has_aleatoric'))
    print(f"\nWith both halves of the uncertainty split: {has_decomp}/{len(complete)}")

    has_noise = sum(1 for *_, s in complete if s.get('has_injected_noise'))
    print(f"With the recorded injected noise: {has_noise}/{len(complete)}")

    if missing:
        print("\n" + "=" * 70)
        print("MISSING FILES")
        print("=" * 70)
        by_model = defaultdict(list)
        for cond, model, rep, fname in missing:
            by_model[model].append((cond, rep))
        for model, items in sorted(by_model.items()):
            print(f"\n{model}: {len(items)} missing")
            for cond, rep in items[:10]:
                print(f"  {cond}/{rep}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")

    if incomplete:
        print("\n" + "=" * 70)
        print("INCOMPLETE FILES")
        print("=" * 70)
        for cond, model, rep, fname, status in incomplete[:20]:
            print(f"  {fname}")
            if status.get('levels_missing'):
                print(f"    Levels missing: {status['levels_missing']}")
            if status.get('missing_required_cols'):
                print(f"    Columns missing: {status['missing_required_cols']}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")

    if missing or incomplete:
        print("\nVERIFICATION FAILED — gaps remain")
        return 1
    print("\nALL UNCERTAINTY FILES COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
