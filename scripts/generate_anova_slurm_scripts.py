#!/usr/bin/env python3
"""
Generate SLURM scripts for ANOVA gap-filling experiments.

Setup: QM9, HOMO-LUMO gap, scaffold split, 10k samples, 10 reps
"""

from pathlib import Path

# Output directory
SLURM_DIR = Path(__file__).parent.parent / "slurm_scripts_anova"
SLURM_DIR.mkdir(exist_ok=True)

# Common header
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

# Constants
STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
STRATEGY_FULL = {
    'legacy': 'legacy',
    'valprop': 'value_proportional',
    'quantile': 'quantile',
    'threshold': 'threshold',
    'outlier': 'outlier',
    'hetero': 'heteroscedastic'
}
REPS = ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn']
BASE_MODELS = ['rf', 'xgboost', 'qrf', 'ngboost', 'dnn', 'mlp', 'gauche']
EXTENDED_SIGMAS = "0.7 0.8 0.9 1.0"
ALL_SIGMAS = "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

def cmd(model, rep, sigmas, strategy, output_file, extra_args=""):
    """Generate a single python command."""
    return f"""python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m {model} \\
    -r {rep} \\
    --sigma {sigmas} \\
    --noise-strategy {STRATEGY_FULL[strategy]} \\
    -n 10000 \\
    -b 10 \\
    --normalize True \\
    {extra_args}\\
    -f {output_file}
"""

scripts_generated = []

# =============================================================================
# 1. EXTENDED SIGMAS (0.7-1.0) for existing combinations
# =============================================================================
print("Generating extended sigma scripts...")

for strategy in STRATEGIES:
    content = slurm_header(f"ext_{strategy}", time="47:59:00")
    content += f"# Extended sigmas (0.7-1.0) for {strategy}\n\n"

    for model in BASE_MODELS:
        for rep in REPS:
            # Skip incompatible
            if model == 'gauche' and rep == 'mhggnn':
                continue

            output = f"../results/anova_{strategy}_{rep}_{model}.csv"
            content += f"# {model}/{rep}\n"
            content += cmd(model, rep, EXTENDED_SIGMAS, strategy, output)
            content += "\n"

    # Also BNN-full (already exists for legacy, extend sigmas)
    if strategy == 'legacy':
        for base in ['dnn', 'mlp']:
            for rep in REPS:
                output = f"../results/anova_{strategy}_{rep}_{base}_bnn_full.csv"
                content += f"# {base}_bnn_full/{rep}\n"
                content += cmd(base, rep, EXTENDED_SIGMAS, strategy, output,
                              "--bayesian-transformation full -u True ")
                content += "\n"

    script_path = SLURM_DIR / f"ext_sigma_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 2. LIGHTGBM
# =============================================================================
print("Generating LightGBM scripts...")

for strategy in STRATEGIES:
    content = slurm_header(f"lgb_{strategy}", time="23:59:00")
    content += f"# LightGBM for {strategy}\n\n"

    for rep in REPS:
        output = f"../results/anova_{strategy}_{rep}_lgb.csv"
        content += f"# lgb/{rep}\n"
        content += cmd('lgb', rep, ALL_SIGMAS, strategy, output)
        content += "\n"

    script_path = SLURM_DIR / f"lgb_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 3. RANDOMIZED SMILES (side-note)
# =============================================================================
print("Generating randomized SMILES scripts...")

all_models_for_rsmiles = BASE_MODELS + ['lgb']

for strategy in STRATEGIES:
    content = slurm_header(f"rsmiles_{strategy}", time="47:59:00")
    content += f"# Randomized SMILES for {strategy}\n\n"

    for model in all_models_for_rsmiles:
        output = f"../results/anova_{strategy}_randomized_smiles_{model}.csv"
        content += f"# {model}/randomized_smiles\n"
        content += cmd(model, 'randomized_smiles', ALL_SIGMAS, strategy, output)
        content += "\n"

    script_path = SLURM_DIR / f"rsmiles_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 4. BNN LAST_LAYER
# =============================================================================
print("Generating BNN last_layer scripts...")

for strategy in STRATEGIES:
    content = slurm_header(f"bnn_last_{strategy}", time="47:59:00", mem="128G")
    content += f"# BNN last_layer for {strategy}\n\n"

    for base in ['dnn', 'mlp']:
        for rep in REPS:
            output = f"../results/anova_{strategy}_{rep}_{base}_bnn_last.csv"
            content += f"# {base}_bnn_last/{rep}\n"
            content += cmd(base, rep, ALL_SIGMAS, strategy, output,
                          "--bayesian-transformation last_layer -u True ")
            content += "\n"

    script_path = SLURM_DIR / f"bnn_last_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 5. BNN VARIATIONAL
# =============================================================================
print("Generating BNN variational scripts...")

for strategy in STRATEGIES:
    content = slurm_header(f"bnn_var_{strategy}", time="47:59:00", mem="128G")
    content += f"# BNN variational for {strategy}\n\n"

    for base in ['dnn', 'mlp']:
        for rep in REPS:
            output = f"../results/anova_{strategy}_{rep}_{base}_bnn_variational.csv"
            content += f"# {base}_bnn_variational/{rep}\n"
            content += cmd(base, rep, ALL_SIGMAS, strategy, output,
                          "--bayesian-transformation variational -u True ")
            content += "\n"

    script_path = SLURM_DIR / f"bnn_var_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 6. FLEXIBLE DNN (deeper - uses default [128,64], needs tuned params for deeper)
# =============================================================================
print("Generating flexible DNN scripts...")

for strategy in STRATEGIES:
    content = slurm_header(f"flex_{strategy}", time="47:59:00", mem="128G")
    content += f"# Flexible DNN for {strategy}\n"
    content += "# Note: Uses default [128,64] architecture. For deeper, tune or add CLI arg.\n\n"

    for rep in REPS:
        output = f"../results/anova_{strategy}_{rep}_flexible_dnn.csv"
        content += f"# flexible_dnn/{rep}\n"
        content += cmd('flexible_dnn', rep, ALL_SIGMAS, strategy, output)
        content += "\n"

    script_path = SLURM_DIR / f"flex_dnn_{strategy}.sh"
    script_path.write_text(content)
    scripts_generated.append(script_path.name)

# =============================================================================
# 7. CONFORMAL — SKIPPED (in ANOVA_MODELS_EXCLUDE, already have data)
# =============================================================================

# =============================================================================
# MASTER SUBMIT SCRIPT
# =============================================================================
print("Generating master submit script...")

master = SLURM_DIR / "submit_all.sh"
content = """#!/bin/bash
# Master script to submit all ANOVA gap-filling jobs
# Review individual scripts before running!

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_anova

echo "Submitting extended sigma scripts..."
"""

for s in STRATEGIES:
    content += f"sbatch ext_sigma_{s}.sh\n"

content += "\necho 'Submitting LightGBM scripts...'\n"
for s in STRATEGIES:
    content += f"sbatch lgb_{s}.sh\n"

content += "\necho 'Submitting randomized SMILES scripts...'\n"
for s in STRATEGIES:
    content += f"sbatch rsmiles_{s}.sh\n"

content += "\necho 'Submitting BNN last_layer scripts...'\n"
for s in STRATEGIES:
    content += f"sbatch bnn_last_{s}.sh\n"

content += "\necho 'Submitting BNN variational scripts...'\n"
for s in STRATEGIES:
    content += f"sbatch bnn_var_{s}.sh\n"

content += "\necho 'Submitting flexible DNN scripts...'\n"
for s in STRATEGIES:
    content += f"sbatch flex_dnn_{s}.sh\n"

content += "\necho 'All jobs submitted!'\n"

master.write_text(content)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SLURM SCRIPTS GENERATED")
print("="*60)
print(f"Output directory: {SLURM_DIR}")
print(f"Total scripts: {len(scripts_generated) + 1}")
print("\nScripts by category:")
print(f"  - Extended sigmas: 6 scripts")
print(f"  - LightGBM: 6 scripts")
print(f"  - Randomized SMILES: 6 scripts")
print(f"  - BNN last_layer: 6 scripts")
print(f"  - BNN variational: 6 scripts")
print(f"  - Flexible DNN: 6 scripts")
print(f"  - Conformal: 6 scripts")
print(f"  - Master submit: 1 script")
print("\nTo upload to server:")
print(f"  scp -r {SLURM_DIR} scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/")
print("\nTo submit all:")
print("  cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_anova && bash submit_all.sh")
