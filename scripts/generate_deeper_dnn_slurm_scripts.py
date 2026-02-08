#!/usr/bin/env python3
"""
Generate SLURM scripts for deeper DNN architecture experiments.

Tests whether model capacity affects noise degradation:
- flexible_dnn (128,64) — already running, baseline
- flexible_dnn_256_128_64 (256,128,64) — deeper
- flexible_dnn_512_256 (512,256) — wide

Setup: QM9, HOMO-LUMO gap, scaffold split, 10k samples, 10 reps
"""

from pathlib import Path

# Output directory
SLURM_DIR = Path(__file__).parent.parent / "slurm_scripts_deeper_dnn"
SLURM_DIR.mkdir(exist_ok=True)

# Common header
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
ALL_SIGMAS = "0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

# Architecture variants to test (excluding default [128,64] which already runs)
ARCHITECTURES = {
    'deep': {'hidden_sizes': '256 128 64', 'suffix': '256_128_64'},
    'wide': {'hidden_sizes': '512 256',    'suffix': '512_256'},
}

def cmd(rep, sigmas, strategy, output_file, hidden_sizes):
    """Generate a single python command."""
    return f"""python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m flexible_dnn \\
    -r {rep} \\
    --sigma {sigmas} \\
    --noise-strategy {STRATEGY_FULL[strategy]} \\
    --hidden-sizes {hidden_sizes} \\
    -n 10000 \\
    -b 10 \\
    --normalize True \\
    -f {output_file}
"""

scripts_generated = []

for arch_name, arch_config in ARCHITECTURES.items():
    hidden_sizes = arch_config['hidden_sizes']
    suffix = arch_config['suffix']

    for strategy in STRATEGIES:
        job_name = f"fdnn_{arch_name}_{strategy}"
        content = slurm_header(job_name)
        content += f"# Flexible DNN ({arch_name}: [{hidden_sizes.replace(' ', ', ')}]) for {strategy}\n\n"

        for rep in REPS:
            output = f"../results/anova_{strategy}_{rep}_flexible_dnn_{suffix}.csv"
            content += f"# flexible_dnn_{suffix}/{rep}\n"
            content += cmd(rep, ALL_SIGMAS, strategy, output, hidden_sizes)
            content += "\n"

        script_path = SLURM_DIR / f"fdnn_{arch_name}_{strategy}.sh"
        script_path.write_text(content)
        scripts_generated.append(script_path.name)

# =============================================================================
# MASTER SUBMIT SCRIPT
# =============================================================================
master = SLURM_DIR / "submit_all.sh"
content = """#!/bin/bash
# Submit all deeper DNN architecture experiments
# 2 architectures × 6 strategies = 12 jobs

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_deeper_dnn

"""

for arch_name in ARCHITECTURES:
    content += f"echo 'Submitting {arch_name} DNN scripts...'\n"
    for s in STRATEGIES:
        content += f"sbatch fdnn_{arch_name}_{s}.sh\n"
    content += "\n"

content += "echo 'All deeper DNN jobs submitted!'\n"

master.write_text(content)

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("DEEPER DNN SLURM SCRIPTS GENERATED")
print("=" * 60)
print(f"Output directory: {SLURM_DIR}")
print(f"Total scripts: {len(scripts_generated) + 1}")
print(f"\nArchitectures:")
for name, config in ARCHITECTURES.items():
    print(f"  - {name}: [{config['hidden_sizes'].replace(' ', ', ')}] → model name: flexible_dnn_{config['suffix']}")
print(f"\nEach architecture: 6 strategies × 5 reps = 30 configs per script")
print(f"Total jobs: {len(scripts_generated)} + 1 submit script")
print(f"\nTo upload to server:")
print(f"  scp -r {SLURM_DIR} scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/")
print(f"\nTo submit all:")
print(f"  cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_deeper_dnn && bash submit_all.sh")
