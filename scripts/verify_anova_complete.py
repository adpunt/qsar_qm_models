#!/usr/bin/env python3
"""
Verify ALL experiments for paper figures are complete, and generate SLURM scripts for gaps.

Usage:
  python verify_anova_complete.py /path/to/results                     # Just verify
  python verify_anova_complete.py /path/to/results --generate-slurm    # Verify + create scripts
  python verify_anova_complete.py /path/to/results --validation-dir /path/to/kirby/results/alternative_full
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd

# =============================================================================
# ANOVA DESIGN — matches generate_paper_figures.py exactly
# =============================================================================

STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
STRATEGY_FULL = {
    'legacy': 'legacy',
    'valprop': 'value_proportional',
    'quantile': 'quantile',
    'threshold': 'threshold',
    'outlier': 'outlier',
    'hetero': 'heteroscedastic',
}
REPS = ['ecfp4', 'pdv', 'smiles', 'mol2vec', 'mhggnn']
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_ITERATIONS = 10

# Models INCLUDED in ANOVA (12 total)
BASE_MODELS = ['rf', 'xgboost', 'ngboost', 'svm', 'dnn', 'mlp', 'gauche', 'lgb']
BNN_MODELS = ['dnn_bnn_full', 'mlp_bnn_full', 'dnn_bnn_last', 'mlp_bnn_last']
ALL_MODELS = BASE_MODELS + BNN_MODELS

# VBLL models (new, tracked separately)
VBLL_MODELS = ['dnn_vbll', 'mlp_vbll']

# Incompatible combinations
INCOMPATIBLE = {('gauche', 'mhggnn')}

# Models that produce uncertainty data (need -u True)
UNCERTAINTY_MODELS = ['ngboost', 'gauche', 'dnn_bnn_full', 'mlp_bnn_full',
                      'dnn_bnn_last', 'mlp_bnn_last', 'dnn_vbll', 'mlp_vbll']

# BNN model → (base_model, bayesian_transformation) for SLURM script generation
BNN_ARGS = {
    'dnn_bnn_full': ('dnn', '--bayesian-transformation full -u True'),
    'mlp_bnn_full': ('mlp', '--bayesian-transformation full -u True'),
    'dnn_bnn_last': ('dnn', '--bayesian-transformation last_layer -u True'),
    'mlp_bnn_last': ('mlp', '--bayesian-transformation last_layer -u True'),
    'dnn_vbll': ('dnn', '--bayesian-transformation variational -u True'),
    'mlp_vbll': ('mlp', '--bayesian-transformation variational -u True'),
}

# Validation datasets
VALIDATION_DATASETS = ['herg', 'caco2', 'logd']
VALIDATION_EXCLUDE_DIRS = {'herg_fluid', 'openadmet_caco2', 'openadmet_logd'}


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


def get_anova_filename(strategy, rep, model):
    """Get expected ANOVA CSV filename."""
    return f"anova_{strategy}_{rep}_{model}.csv"


def slurm_header(job_name, time="47:59:00", mem="128G"):
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time={time}
#SBATCH --partition=long
#SBATCH --mem={mem}
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

"""


def make_cmd(model, rep, sigmas, strategy, output_file, extra_args=""):
    """Generate process_and_train.py command."""
    sigma_str = " ".join(str(s) for s in sigmas)
    strategy_full = STRATEGY_FULL[strategy]
    uncertainty_flag = "-u True" if model in UNCERTAINTY_MODELS else ""

    # Handle BNN/VBLL models
    if model in BNN_ARGS:
        base_model, bnn_extra = BNN_ARGS[model]
        return (f"python process_and_train.py -d QM9 -t homo_lumo_gap "
                f"-m {base_model} -r {rep} "
                f"--sigma {sigma_str} --noise-strategy {strategy_full} "
                f"-n 10000 -b 10 --normalize True "
                f"{bnn_extra} "
                f"-f {output_file}\n")
    else:
        return (f"python process_and_train.py -d QM9 -t homo_lumo_gap "
                f"-m {model} -r {rep} "
                f"--sigma {sigma_str} --noise-strategy {strategy_full} "
                f"-n 10000 -b 10 --normalize True "
                f"{uncertainty_flag} "
                f"-f {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Verify and fill data gaps")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--validation-dir", type=str, default=None,
                        help="Path to KIRBy validation results")
    parser.add_argument("--generate-slurm", action="store_true",
                        help="Generate SLURM scripts for missing data")
    parser.add_argument("--include-vbll", action="store_true",
                        help="Include VBLL models in verification")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # =========================================================================
    # 1. ANOVA DATA
    # =========================================================================
    print("=" * 70)
    print("ANOVA DATA VERIFICATION")
    print("=" * 70)

    models_to_check = ALL_MODELS[:]
    if args.include_vbll:
        models_to_check += VBLL_MODELS

    complete = []
    incomplete = []
    missing = []

    for strategy in STRATEGIES:
        for model in models_to_check:
            for rep in REPS:
                if (model, rep) in INCOMPATIBLE:
                    continue

                filename = get_anova_filename(strategy, rep, model)
                filepath = results_dir / filename
                status = check_file(filepath, SIGMAS)

                entry = (strategy, model, rep, filename)
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
        for strat, model, rep, fname in missing:
            by_model[model].append((strat, rep))

        for model in models_to_check:
            if model in by_model:
                items = by_model[model]
                strats = sorted(set(s for s, r in items))
                reps = sorted(set(r for s, r in items))
                print(f"  {model}: {len(items)} gaps — strategies={strats}, reps={reps}")

    if incomplete:
        print(f"\n--- INCOMPLETE ANOVA FILES ({len(incomplete)}) ---")
        for strat, model, rep, fname, status in incomplete[:20]:
            ms = status.get('sigmas_missing', [])
            mi = status.get('min_iterations', 0)
            print(f"  {fname}: missing_sigmas={ms}, min_iters={mi}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")

    # =========================================================================
    # 2. UNCERTAINTY DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("UNCERTAINTY DATA VERIFICATION")
    print("=" * 70)

    unc_complete = []
    unc_missing = []

    unc_models = [m for m in UNCERTAINTY_MODELS if m in models_to_check]

    for strategy in STRATEGIES:
        for model in unc_models:
            for rep in REPS:
                if (model, rep) in INCOMPATIBLE:
                    continue

                # Try both filename patterns
                fname1 = f"uncertainty_{strategy}_{rep}_{model}_uncertainty_values.csv"
                fname2 = f"{strategy}_{rep}_{model}_uncertainty_values.csv"
                found = False
                for fname in [fname1, fname2]:
                    if (results_dir / fname).exists():
                        found = True
                        unc_complete.append((strategy, model, rep, fname))
                        break
                if not found:
                    unc_missing.append((strategy, model, rep))

    unc_total = len(unc_complete) + len(unc_missing)
    if unc_total > 0:
        print(f"\nTotal uncertainty configs: {unc_total}")
        print(f"Complete:   {len(unc_complete):4d} ({100*len(unc_complete)/unc_total:.1f}%)")
        print(f"Missing:    {len(unc_missing):4d}")

        if unc_missing:
            print(f"\n--- MISSING UNCERTAINTY FILES ({len(unc_missing)}) ---")
            by_model = defaultdict(list)
            for strat, model, rep in unc_missing:
                by_model[model].append((strat, rep))
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
    # 4. GENERATE SLURM SCRIPTS
    # =========================================================================
    if args.generate_slurm and (missing or incomplete):
        slurm_dir = Path(args.results_dir).parent / "slurm_scripts_gapfill"
        slurm_dir.mkdir(exist_ok=True)

        print("\n" + "=" * 70)
        print(f"GENERATING SLURM SCRIPTS → {slurm_dir}")
        print("=" * 70)

        # Collect all gaps (missing + incomplete treated the same)
        all_gaps = []
        for strat, model, rep, fname in missing:
            all_gaps.append((strat, model, rep, SIGMAS))  # Full sigma range
        for strat, model, rep, fname, status in incomplete:
            ms = status.get('sigmas_missing', [])
            if ms:
                all_gaps.append((strat, model, rep, ms))
            else:
                # Has all sigmas but too few iterations — rerun full
                all_gaps.append((strat, model, rep, SIGMAS))

        # Group by model for efficient batching
        # Separate MHGGNN jobs (must run sequentially, NEVER in parallel)
        mhggnn_gaps = [(s, m, r, sigs) for s, m, r, sigs in all_gaps if r == 'mhggnn']
        other_gaps = [(s, m, r, sigs) for s, m, r, sigs in all_gaps if r != 'mhggnn']

        # Group non-mhggnn by model
        by_model = defaultdict(list)
        for strat, model, rep, sigs in other_gaps:
            by_model[model].append((strat, rep, sigs))

        scripts_created = []

        # Non-MHGGNN scripts: one per model (can run in parallel)
        for model, items in sorted(by_model.items()):
            job_name = f"gap_{model[:8]}"
            content = slurm_header(job_name, time="71:59:00")
            content += f"# Gap-fill for {model} (non-mhggnn reps)\n\n"

            for strat, rep, sigs in items:
                output = f"../results/{get_anova_filename(strat, rep, model)}"
                content += f"echo '=== {model}/{rep}/{strat} at $(date) ==='\n"
                content += make_cmd(model, rep, sigs, strat, output)
                content += "\n"

            content += f"echo 'Done: {model} gap-fill at $(date)'\n"
            script_path = slurm_dir / f"gap_{model}.sh"
            script_path.write_text(content)
            scripts_created.append(script_path.name)

        # MHGGNN script: ONE sequential script (NEVER parallel)
        if mhggnn_gaps:
            content = slurm_header("gap_mhggnn", time="71:59:00")
            content += "# MHGGNN gap-fill — MUST run sequentially, never parallel!\n\n"

            for strat, model, rep, sigs in mhggnn_gaps:
                output = f"../results/{get_anova_filename(strat, rep, model)}"
                content += f"echo '=== {model}/mhggnn/{strat} at $(date) ==='\n"
                content += make_cmd(model, 'mhggnn', sigs, strat, output)
                content += "\n"

            content += "echo 'Done: all MHGGNN gap-fill at $(date)'\n"
            script_path = slurm_dir / "gap_mhggnn_sequential.sh"
            script_path.write_text(content)
            scripts_created.append(script_path.name)

        # Submit script
        submit = slurm_dir / "submit_all.sh"
        content = "#!/bin/bash\n# Submit all gap-fill scripts\n"
        content += "# IMPORTANT: gap_mhggnn_sequential.sh runs alone (no other mhggnn jobs)\n\n"
        for s in sorted(scripts_created):
            if 'mhggnn' in s:
                content += f"# Run separately: sbatch {s}\n"
            else:
                content += f"sbatch {s}\n"
        content += f"\necho 'Submitted {len(scripts_created)} gap-fill jobs'\n"
        content += "echo 'Remember to submit gap_mhggnn_sequential.sh separately!'\n"
        submit.write_text(content)

        print(f"\nCreated {len(scripts_created)} SLURM scripts + submit_all.sh")
        print(f"\nTo deploy:")
        print(f"  scp -r {slurm_dir} scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/")
        print(f"  ssh scat9264@gateway.arc.ox.ac.uk")
        print(f"  cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_gapfill")
        print(f"  bash submit_all.sh")
        print(f"  sbatch gap_mhggnn_sequential.sh  # Submit separately!")

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
        if not args.generate_slurm:
            print("Run with --generate-slurm to create SLURM scripts for all gaps")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
