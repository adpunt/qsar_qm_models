#!/usr/bin/env python3
"""
Gap Analysis and SLURM Script Generator

Compares what exists against the experimental protocol and generates
SLURM scripts for missing experiments.

Usage:
    python gap_analysis_and_slurm.py [results_dir]

Outputs:
    - gap_analysis_report.txt (what's missing)
    - slurm_scripts/ directory with job scripts
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

# =============================================================================
# EXPERIMENTAL PROTOCOL DEFINITION
# =============================================================================

PROTOCOL = {
    # Models to test
    'models': ['rf', 'xgboost', 'qrf', 'ngboost', 'dnn', 'mlp', 'gauche'],

    # Models that need BNN-full variant tested
    'bnn_models': ['dnn', 'mlp'],

    # Representations (excluding randomized_smiles)
    'representations': ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn'],

    # Graph models (separate because they use 'graph' representation)
    'graph_models': ['gin', 'gcn'],

    # Noise strategies
    'noise_strategies': ['legacy', 'value_proportional', 'quantile', 'threshold', 'outlier', 'heteroscedastic'],

    # Sigma values
    'sigmas': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],

    # Iterations
    'iterations': 10,
    'uncertainty_iterations': 1,

    # Sample size
    'sample_size': 10000,

    # Models that produce uncertainty estimates (need uncertainty files)
    'uncertainty_models': ['qrf', 'ngboost', 'gauche', 'dnn_bnn_full', 'mlp_bnn_full'],

    # Conformal models (analyzed separately)
    'conformal_base_models': ['rf', 'qrf', 'dnn'],
}

# Model-representation compatibility
# Some models only work with certain representations
INCOMPATIBLE = {
    'gauche': ['mhggnn'],  # GP might not work well with all reps
    'gin': ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn'],  # GNN needs graph
    'gcn': ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn'],  # GNN needs graph
}

# =============================================================================
# SCAN EXISTING RESULTS
# =============================================================================

def scan_existing_results(results_dir):
    """Scan results directory and catalog what exists."""
    results_dir = Path(results_dir)

    existing = {
        'screening': defaultdict(set),  # (model, rep) -> set of sigmas
        'noise_strategy': defaultdict(lambda: defaultdict(set)),  # strategy -> (model, rep) -> set of sigmas
        'uncertainty': defaultdict(set),  # (model, rep) -> set of sigmas
        'bnn': defaultdict(set),  # (base_model, rep) -> set of sigmas
        'conformal': defaultdict(set),  # (base_model, rep) -> set of sigmas
        'raw_files': [],
    }

    for f in results_dir.glob("*.csv"):
        filename = f.name
        existing['raw_files'].append(filename)

        # Skip uncertainty value files for main analysis
        if '_uncertainty_values' in filename:
            # But track them for uncertainty coverage
            # Try to parse model/rep/sigma
            match = re.search(r'phase\d+[a-z]?_(\w+)_(\w+?)(?:_(\w+))?_uncertainty_values', filename)
            if match:
                rep = match.group(1)
                model = match.group(2)
                # Read file to get sigmas
                try:
                    with open(f, 'r') as csvf:
                        header = csvf.readline()
                        for line in csvf:
                            parts = line.strip().split(',')
                            if len(parts) > 2:
                                try:
                                    sigma = float(parts[2]) if 'sigma' in header.lower() else None
                                    if sigma is not None:
                                        existing['uncertainty'][(model, rep)].add(sigma)
                                except:
                                    pass
                except:
                    pass
            continue

        if '_per_epoch' in filename:
            continue

        # Parse phase0c screening files
        if 'phase0c_screen_' in filename:
            # Extract model from filename
            match = re.search(r'phase0c_screen_(\w+?)(?:_(\w+))?\.csv', filename)
            if match:
                model = match.group(1)
                variant = match.group(2)
                if variant and variant not in ['default', 'tuned']:
                    model = f"{model}_{variant}"

                # Read file to get rep and sigma coverage
                try:
                    with open(f, 'r') as csvf:
                        header = csvf.readline().strip().split(',')
                        sigma_idx = header.index('sigma') if 'sigma' in header else None
                        rep_idx = header.index('rep') if 'rep' in header else None

                        if sigma_idx is not None and rep_idx is not None:
                            for line in csvf:
                                parts = line.strip().split(',')
                                if len(parts) > max(sigma_idx, rep_idx):
                                    try:
                                        sigma = float(parts[sigma_idx])
                                        rep = parts[rep_idx]
                                        existing['screening'][(model, rep)].add(sigma)
                                    except:
                                        pass
                except Exception as e:
                    pass

        # Parse noise strategy files (looking for strategy in filename)
        for strategy in PROTOCOL['noise_strategies']:
            strategy_short = strategy.replace('value_proportional', 'valprop').replace('heteroscedastic', 'hetero')
            if f'_{strategy_short}' in filename or f'_{strategy}' in filename:
                # Try to extract model and rep
                match = re.search(r'(\w+)_(\w+)_' + strategy_short, filename)
                if match:
                    rep = match.group(1)
                    model = match.group(2)
                    # Read to get sigmas
                    try:
                        with open(f, 'r') as csvf:
                            header = csvf.readline().strip().split(',')
                            sigma_idx = header.index('sigma') if 'sigma' in header else None
                            if sigma_idx is not None:
                                for line in csvf:
                                    parts = line.strip().split(',')
                                    if len(parts) > sigma_idx:
                                        try:
                                            sigma = float(parts[sigma_idx])
                                            existing['noise_strategy'][strategy_short][(model, rep)].add(sigma)
                                        except:
                                            pass
                    except:
                        pass
                break

        # Parse BNN files
        if 'bnn_full' in filename.lower():
            match = re.search(r'(\w+)_bnn_full', filename)
            if match:
                base_model = match.group(1)
                # Try to find rep
                for rep in PROTOCOL['representations']:
                    if rep in filename:
                        try:
                            with open(f, 'r') as csvf:
                                header = csvf.readline().strip().split(',')
                                sigma_idx = header.index('sigma') if 'sigma' in header else None
                                if sigma_idx is not None:
                                    for line in csvf:
                                        parts = line.strip().split(',')
                                        if len(parts) > sigma_idx:
                                            try:
                                                sigma = float(parts[sigma_idx])
                                                existing['bnn'][(base_model, rep)].add(sigma)
                                            except:
                                                pass
                        except:
                            pass
                        break

    return existing


def analyze_gaps(existing):
    """Compare existing results against protocol to find gaps."""
    gaps = {
        'screening': [],
        'noise_strategy': defaultdict(list),
        'uncertainty_baseline': [],
        'bnn': [],
        'mhggnn': [],
    }

    # Check screening coverage (legacy noise, all sigmas)
    for model in PROTOCOL['models']:
        for rep in PROTOCOL['representations']:
            # Check compatibility
            if model in INCOMPATIBLE and rep in INCOMPATIBLE[model]:
                continue

            key = (model, rep)
            existing_sigmas = existing['screening'].get(key, set())
            missing_sigmas = set(PROTOCOL['sigmas']) - existing_sigmas

            if missing_sigmas:
                gaps['screening'].append({
                    'model': model,
                    'rep': rep,
                    'missing_sigmas': sorted(missing_sigmas),
                    'existing_sigmas': sorted(existing_sigmas),
                })

    # Check noise strategy coverage
    for strategy in PROTOCOL['noise_strategies']:
        strategy_short = strategy.replace('value_proportional', 'valprop').replace('heteroscedastic', 'hetero')

        for model in PROTOCOL['models']:
            for rep in PROTOCOL['representations']:
                if model in INCOMPATIBLE and rep in INCOMPATIBLE[model]:
                    continue

                key = (model, rep)
                existing_sigmas = existing['noise_strategy'][strategy_short].get(key, set())
                missing_sigmas = set(PROTOCOL['sigmas']) - existing_sigmas

                if missing_sigmas:
                    gaps['noise_strategy'][strategy].append({
                        'model': model,
                        'rep': rep,
                        'missing_sigmas': sorted(missing_sigmas),
                        'existing_sigmas': sorted(existing_sigmas),
                    })

    # Check uncertainty at baseline (sigma=0.0)
    for model in PROTOCOL['uncertainty_models']:
        for rep in PROTOCOL['representations']:
            base_model = model.replace('_bnn_full', '')
            if base_model in INCOMPATIBLE and rep in INCOMPATIBLE[base_model]:
                continue

            key = (model, rep)
            existing_sigmas = existing['uncertainty'].get(key, set())

            if 0.0 not in existing_sigmas:
                gaps['uncertainty_baseline'].append({
                    'model': model,
                    'rep': rep,
                    'has_any_uncertainty': len(existing_sigmas) > 0,
                })

    # Check BNN coverage
    for base_model in PROTOCOL['bnn_models']:
        for rep in PROTOCOL['representations']:
            if base_model in INCOMPATIBLE and rep in INCOMPATIBLE[base_model]:
                continue

            key = (base_model, rep)
            existing_sigmas = existing['bnn'].get(key, set())
            missing_sigmas = set(PROTOCOL['sigmas']) - existing_sigmas

            if missing_sigmas:
                gaps['bnn'].append({
                    'model': f"{base_model}_bnn_full",
                    'rep': rep,
                    'missing_sigmas': sorted(missing_sigmas),
                })

    # Check MHGGNN specifically
    for model in PROTOCOL['models']:
        key = (model, 'mhggnn')
        existing_sigmas = existing['screening'].get(key, set())
        missing_sigmas = set(PROTOCOL['sigmas']) - existing_sigmas

        if missing_sigmas:
            gaps['mhggnn'].append({
                'model': model,
                'missing_sigmas': sorted(missing_sigmas),
                'existing_sigmas': sorted(existing_sigmas),
            })

    return gaps


# =============================================================================
# GENERATE SLURM SCRIPTS
# =============================================================================

def generate_slurm_scripts(gaps, output_dir):
    """Generate SLURM scripts for missing experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    scripts_generated = []

    # Common SLURM header - use absolute paths to avoid pwd issues
    def slurm_header(job_name, time="23:59:00", mem="256G"):
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

    # 1. Generate scripts for each noise strategy
    for strategy, missing_list in gaps['noise_strategy'].items():
        if not missing_list:
            continue

        strategy_short = strategy.replace('value_proportional', 'valprop').replace('heteroscedastic', 'hetero')

        # Group by model to create efficient scripts
        by_model = defaultdict(list)
        for item in missing_list:
            by_model[item['model']].append(item)

        for model, items in by_model.items():
            script_name = f"run_{strategy_short}_{model}.sh"
            script_path = output_dir / script_name

            content = slurm_header(f"{strategy_short}_{model}")
            content += f"# Noise strategy: {strategy}\n"
            content += f"# Model: {model}\n\n"

            for item in items:
                rep = item['rep']
                sigmas = ' '.join(str(s) for s in item['missing_sigmas'])

                # Output file naming (../results because we cd to scripts)
                output_file = f"../results/anova_{strategy_short}_{rep}_{model}.csv"

                content += f"# {rep}: sigmas {sigmas}\n"
                content += f"python process_and_train.py -d QM9 -t homo_lumo_gap \\\n"
                content += f"    -m {model} \\\n"
                content += f"    -r {rep} \\\n"
                content += f"    --sigma {sigmas} \\\n"
                content += f"    --noise-strategy {strategy} \\\n"
                content += f"    -n 10000 \\\n"
                content += f"    -b 10 \\\n"
                content += f"    --normalize True \\\n"
                content += f"    -f {output_file}\n\n"

            with open(script_path, 'w') as f:
                f.write(content)
            scripts_generated.append(script_name)

    # 2. Generate script for uncertainty at baseline
    if gaps['uncertainty_baseline']:
        script_name = "run_uncertainty_baseline.sh"
        script_path = output_dir / script_name

        content = slurm_header("uncertainty_baseline", mem="128G")
        content += "# Uncertainty estimates at sigma=0.0 (baseline)\n\n"

        for item in gaps['uncertainty_baseline']:
            model = item['model']
            rep = item['rep']

            # Determine base model and if BNN
            if '_bnn_full' in model:
                base_model = model.replace('_bnn_full', '')
                bnn_flag = "--bayesian-transformation full"
                model_arg = base_model
            else:
                bnn_flag = ""
                model_arg = model

            output_file = f"../results/uncertainty_baseline_{rep}_{model}.csv"

            content += f"# {model} / {rep}\n"
            content += f"python process_and_train.py -d QM9 -t homo_lumo_gap \\\n"
            content += f"    -m {model_arg} \\\n"
            content += f"    -r {rep} \\\n"
            content += f"    --sigma 0.0 \\\n"
            content += f"    -n 10000 \\\n"
            content += f"    -b 1 \\\n"
            content += f"    -u True \\\n"
            if bnn_flag:
                content += f"    {bnn_flag} \\\n"
            content += f"    --normalize True \\\n"
            content += f"    -f {output_file}\n\n"

        with open(script_path, 'w') as f:
            f.write(content)
        scripts_generated.append(script_name)

    # 3. Generate script for MHGGNN completion
    if gaps['mhggnn']:
        script_name = "run_mhggnn_complete.sh"
        script_path = output_dir / script_name

        content = slurm_header("mhggnn_complete", mem="128G")
        content += "# Complete MHGGNN experiments\n\n"

        for item in gaps['mhggnn']:
            model = item['model']
            sigmas = ' '.join(str(s) for s in item['missing_sigmas'])

            if not sigmas.strip():
                continue

            output_file = f"../results/anova_legacy_mhggnn_{model}.csv"

            content += f"# {model}: sigmas {sigmas}\n"
            content += f"python process_and_train.py -d QM9 -t homo_lumo_gap \\\n"
            content += f"    -m {model} \\\n"
            content += f"    -r mhggnn \\\n"
            content += f"    --sigma {sigmas} \\\n"
            content += f"    -n 10000 \\\n"
            content += f"    -b 10 \\\n"
            content += f"    --normalize True \\\n"
            content += f"    -f {output_file}\n\n"

        with open(script_path, 'w') as f:
            f.write(content)
        scripts_generated.append(script_name)

    # 4. Generate script for BNN experiments
    if gaps['bnn']:
        script_name = "run_bnn_full.sh"
        script_path = output_dir / script_name

        content = slurm_header("bnn_full", mem="128G", time="47:59:00")
        content += "# BNN-full experiments\n\n"

        for item in gaps['bnn']:
            model = item['model']  # e.g., dnn_bnn_full
            base_model = model.replace('_bnn_full', '')
            rep = item['rep']
            sigmas = ' '.join(str(s) for s in item['missing_sigmas'])

            if not sigmas.strip():
                continue

            output_file = f"../results/anova_legacy_{rep}_{model}.csv"

            content += f"# {model} / {rep}: sigmas {sigmas}\n"
            content += f"python process_and_train.py -d QM9 -t homo_lumo_gap \\\n"
            content += f"    -m {base_model} \\\n"
            content += f"    -r {rep} \\\n"
            content += f"    --sigma {sigmas} \\\n"
            content += f"    --bayesian-transformation full \\\n"
            content += f"    -n 10000 \\\n"
            content += f"    -b 10 \\\n"
            content += f"    -u True \\\n"
            content += f"    --normalize True \\\n"
            content += f"    -f {output_file}\n\n"

        with open(script_path, 'w') as f:
            f.write(content)
        scripts_generated.append(script_name)

    # 5. Generate a master submission script
    master_script = output_dir / "submit_all.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Master script to submit all SLURM jobs\n")
        f.write("# Review individual scripts before running!\n\n")
        f.write("cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts\n\n")
        for script in scripts_generated:
            f.write(f"sbatch {script}\n")

    return scripts_generated


# =============================================================================
# GENERATE REPORT
# =============================================================================

def generate_report(existing, gaps, scripts_generated, results_dir):
    """Generate human-readable gap analysis report."""
    lines = []
    lines.append("=" * 80)
    lines.append("GAP ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Protocol summary
    lines.append("\n" + "=" * 80)
    lines.append("EXPERIMENTAL PROTOCOL")
    lines.append("=" * 80)
    lines.append(f"Models: {', '.join(PROTOCOL['models'])}")
    lines.append(f"BNN variants: {', '.join(PROTOCOL['bnn_models'])} with bnn_full")
    lines.append(f"Representations: {', '.join(PROTOCOL['representations'])}")
    lines.append(f"Noise strategies: {', '.join(PROTOCOL['noise_strategies'])}")
    lines.append(f"Sigma values: {PROTOCOL['sigmas']}")
    lines.append(f"Iterations: {PROTOCOL['iterations']} (uncertainty: {PROTOCOL['uncertainty_iterations']})")
    lines.append(f"Sample size: {PROTOCOL['sample_size']}")

    # Calculate total experiments needed
    n_models = len(PROTOCOL['models'])
    n_reps = len(PROTOCOL['representations'])
    n_strategies = len(PROTOCOL['noise_strategies'])
    n_sigmas = len(PROTOCOL['sigmas'])

    total_main = n_models * n_reps * n_strategies * n_sigmas
    lines.append(f"\nTotal experiments needed (main): ~{total_main}")

    # Screening gaps
    lines.append("\n" + "=" * 80)
    lines.append("SCREENING GAPS (Legacy Noise)")
    lines.append("=" * 80)

    if gaps['screening']:
        lines.append(f"Missing {len(gaps['screening'])} model-rep combinations:")
        for item in gaps['screening'][:20]:
            lines.append(f"  {item['model']}/{item['rep']}: missing σ={item['missing_sigmas']}")
        if len(gaps['screening']) > 20:
            lines.append(f"  ... and {len(gaps['screening']) - 20} more")
    else:
        lines.append("✓ Screening coverage complete!")

    # Noise strategy gaps
    lines.append("\n" + "=" * 80)
    lines.append("NOISE STRATEGY GAPS")
    lines.append("=" * 80)

    for strategy in PROTOCOL['noise_strategies']:
        missing = gaps['noise_strategy'].get(strategy, [])
        if missing:
            lines.append(f"\n{strategy}: {len(missing)} gaps")
            for item in missing[:5]:
                lines.append(f"  {item['model']}/{item['rep']}: missing σ={item['missing_sigmas']}")
            if len(missing) > 5:
                lines.append(f"  ... and {len(missing) - 5} more")
        else:
            lines.append(f"\n{strategy}: ✓ Complete")

    # Uncertainty baseline gaps
    lines.append("\n" + "=" * 80)
    lines.append("UNCERTAINTY AT BASELINE (σ=0.0) GAPS")
    lines.append("=" * 80)

    if gaps['uncertainty_baseline']:
        lines.append(f"Missing {len(gaps['uncertainty_baseline'])} uncertainty baselines:")
        for item in gaps['uncertainty_baseline']:
            status = "(has other sigmas)" if item['has_any_uncertainty'] else "(no uncertainty data)"
            lines.append(f"  {item['model']}/{item['rep']} {status}")
    else:
        lines.append("✓ Uncertainty baselines complete!")

    # MHGGNN gaps
    lines.append("\n" + "=" * 80)
    lines.append("MHGGNN GAPS")
    lines.append("=" * 80)

    if gaps['mhggnn']:
        for item in gaps['mhggnn']:
            existing_str = f"has σ={item['existing_sigmas']}" if item['existing_sigmas'] else "no data"
            lines.append(f"  {item['model']}/mhggnn: missing σ={item['missing_sigmas']} ({existing_str})")
    else:
        lines.append("✓ MHGGNN coverage complete!")

    # BNN gaps
    lines.append("\n" + "=" * 80)
    lines.append("BNN-FULL GAPS")
    lines.append("=" * 80)

    if gaps['bnn']:
        for item in gaps['bnn']:
            lines.append(f"  {item['model']}/{item['rep']}: missing σ={item['missing_sigmas']}")
    else:
        lines.append("✓ BNN-full coverage complete!")

    # Generated scripts
    lines.append("\n" + "=" * 80)
    lines.append("SLURM SCRIPTS GENERATED")
    lines.append("=" * 80)

    if scripts_generated:
        lines.append(f"Generated {len(scripts_generated)} scripts in slurm_scripts/:")
        for script in scripts_generated:
            lines.append(f"  - {script}")
        lines.append("\nTo submit all jobs:")
        lines.append("  cd slurm_scripts && bash submit_all.sh")
        lines.append("\nOr submit individually:")
        lines.append("  sbatch slurm_scripts/<script_name>.sh")
    else:
        lines.append("No scripts needed - all experiments complete!")

    # Priority recommendations
    lines.append("\n" + "=" * 80)
    lines.append("PRIORITY RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("""
1. HIGHEST PRIORITY - Uncertainty baseline (σ=0.0):
   Run run_uncertainty_baseline.sh first - needed for uncertainty vs noise analysis

2. HIGH PRIORITY - Complete one noise strategy fully:
   Pick 'legacy' to validate the pipeline, then run others in parallel

3. MEDIUM PRIORITY - MHGGNN completion:
   Run run_mhggnn_complete.sh if you want MHGGNN in the paper

4. LOWER PRIORITY - BNN experiments:
   Run after main models are complete
""")

    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    if not os.path.isdir(results_dir):
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results"
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

    print(f"Scanning {results_dir}...")
    existing = scan_existing_results(results_dir)
    print(f"Found {len(existing['raw_files'])} files")

    print("Analyzing gaps against protocol...")
    gaps = analyze_gaps(existing)

    print("Generating SLURM scripts...")
    slurm_dir = Path(results_dir).parent / "slurm_scripts"
    scripts = generate_slurm_scripts(gaps, slurm_dir)
    print(f"Generated {len(scripts)} SLURM scripts in {slurm_dir}")

    print("Generating report...")
    report = generate_report(existing, gaps, scripts, results_dir)

    report_path = Path(results_dir) / "gap_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    print("\n" + report)


if __name__ == "__main__":
    main()
