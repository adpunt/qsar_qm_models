#!/usr/bin/env python3
"""
Generate SLURM scripts for uncertainty experiments.
Limited to PDV and ECFP4 representations only.

Setup: QM9, HOMO-LUMO gap, scaffold split, 10k samples, 1 iteration (uncertainty)
"""

from pathlib import Path

# Output directory
SLURM_DIR = Path(__file__).parent.parent / "slurm_scripts_uncertainty"
SLURM_DIR.mkdir(exist_ok=True)

def slurm_header(job_name, time="23:59:00", mem="128G"):
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
REPS = ['pdv', 'ecfp4']  # Match analysis: PRIMARY_REP=pdv, SUPPLEMENTARY_REP=ecfp4
ALL_SIGMAS = "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

# Models that produce uncertainty
UNCERTAINTY_MODELS = {
    # Native uncertainty models
    'qrf': {'args': '', 'name': 'qrf'},
    'ngboost': {'args': '', 'name': 'ngboost'},
    'gauche': {'args': '', 'name': 'gauche'},

    # BNN variants - DNN
    'dnn_bnn_full': {'args': '--bayesian-transformation full', 'base': 'dnn', 'name': 'dnn_bnn_full'},
    'dnn_bnn_last': {'args': '--bayesian-transformation last_layer', 'base': 'dnn', 'name': 'dnn_bnn_last'},
    'dnn_bnn_var': {'args': '--bayesian-transformation variational', 'base': 'dnn', 'name': 'dnn_bnn_variational'},

    # BNN variants - MLP
    'mlp_bnn_full': {'args': '--bayesian-transformation full', 'base': 'mlp', 'name': 'mlp_bnn_full'},
    'mlp_bnn_last': {'args': '--bayesian-transformation last_layer', 'base': 'mlp', 'name': 'mlp_bnn_last'},
    'mlp_bnn_var': {'args': '--bayesian-transformation variational', 'base': 'mlp', 'name': 'mlp_bnn_variational'},
}

# Conformal models
CONFORMAL_BASES = ['rf', 'qrf', 'dnn']

scripts_generated = []

# =============================================================================
# 1. NATIVE UNCERTAINTY MODELS (QRF, NGBoost, Gauche)
# =============================================================================
print("Generating native uncertainty model scripts...")

for model_key, model_info in UNCERTAINTY_MODELS.items():
    if 'base' in model_info:
        continue  # Skip BNN variants for now

    for strategy in STRATEGIES:
        content = slurm_header(f"unc_{model_key}_{strategy}", time="23:59:00")
        content += f"# Uncertainty: {model_key} - {strategy}\n\n"

        for rep in REPS:
            output = f"../results/uncertainty_{strategy}_{rep}_{model_info['name']}.csv"

            content += f"# {model_key}/{rep}\n"
            content += f"""python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m {model_key} \\
    -r {rep} \\
    --sigma {ALL_SIGMAS} \\
    --noise-strategy {STRATEGY_FULL[strategy]} \\
    -n 10000 \\
    -b 1 \\
    -u True \\
    --normalize True \\
    -f {output}

"""

        script_path = SLURM_DIR / f"unc_{model_key}_{strategy}.sh"
        script_path.write_text(content)
        scripts_generated.append(script_path.name)

# =============================================================================
# 2. BNN VARIANTS
# =============================================================================
print("Generating BNN uncertainty scripts...")

for model_key, model_info in UNCERTAINTY_MODELS.items():
    if 'base' not in model_info:
        continue  # Only BNN variants

    for strategy in STRATEGIES:
        content = slurm_header(f"unc_{model_key}_{strategy}", time="47:59:00", mem="128G")
        content += f"# Uncertainty: {model_info['name']} - {strategy}\n\n"

        for rep in REPS:
            output = f"../results/uncertainty_{strategy}_{rep}_{model_info['name']}.csv"

            content += f"# {model_info['name']}/{rep}\n"
            content += f"""python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m {model_info['base']} \\
    -r {rep} \\
    --sigma {ALL_SIGMAS} \\
    --noise-strategy {STRATEGY_FULL[strategy]} \\
    -n 10000 \\
    -b 1 \\
    -u True \\
    {model_info['args']} \\
    --normalize True \\
    -f {output}

"""

        script_path = SLURM_DIR / f"unc_{model_key}_{strategy}.sh"
        script_path.write_text(content)
        scripts_generated.append(script_path.name)

# =============================================================================
# 3. CONFORMAL MODELS
# =============================================================================
print("Generating conformal uncertainty scripts...")

for base in CONFORMAL_BASES:
    for strategy in STRATEGIES:
        content = slurm_header(f"unc_conf_{base}_{strategy}", time="23:59:00")
        content += f"# Uncertainty: conformal_{base} - {strategy}\n\n"

        for rep in REPS:
            output = f"../results/uncertainty_{strategy}_{rep}_conformal_{base}.csv"

            content += f"# conformal_{base}/{rep}\n"
            content += f"""python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m conformal \\
    --cp-base-model {base} \\
    -r {rep} \\
    --sigma {ALL_SIGMAS} \\
    --noise-strategy {STRATEGY_FULL[strategy]} \\
    -n 10000 \\
    -b 1 \\
    -u True \\
    --normalize True \\
    -f {output}

"""

        script_path = SLURM_DIR / f"unc_conformal_{base}_{strategy}.sh"
        script_path.write_text(content)
        scripts_generated.append(script_path.name)

# =============================================================================
# MASTER SUBMIT SCRIPT
# =============================================================================
print("Generating master submit script...")

master = SLURM_DIR / "submit_all.sh"
content = """#!/bin/bash
# Master script to submit all uncertainty experiments
# Limited to PDV and ECFP4 representations

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty

echo "Submitting native uncertainty models (QRF, NGBoost, Gauche)..."
"""

for model_key in ['qrf', 'ngboost', 'gauche']:
    for s in STRATEGIES:
        content += f"sbatch unc_{model_key}_{s}.sh\n"

content += "\necho 'Submitting BNN variants...'\n"
for model_key in ['dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_var', 'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_var']:
    for s in STRATEGIES:
        content += f"sbatch unc_{model_key}_{s}.sh\n"

content += "\necho 'Submitting conformal models...'\n"
for base in CONFORMAL_BASES:
    for s in STRATEGIES:
        content += f"sbatch unc_conformal_{base}_{s}.sh\n"

content += "\necho 'All uncertainty jobs submitted!'\n"
content += f"echo 'Total jobs: {len(scripts_generated)}'\n"

master.write_text(content)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("UNCERTAINTY SLURM SCRIPTS GENERATED")
print("=" * 60)
print(f"Output directory: {SLURM_DIR}")
print(f"Total scripts: {len(scripts_generated) + 1}")
print(f"\nModels: {len(UNCERTAINTY_MODELS) + len(CONFORMAL_BASES)} uncertainty-producing models")
print(f"Representations: {REPS}")
print(f"Strategies: {len(STRATEGIES)}")
print(f"\nTo upload to server:")
print(f"  scp -r {SLURM_DIR} scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/")
print("\nTo submit all:")
print("  cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty && bash submit_all.sh")
