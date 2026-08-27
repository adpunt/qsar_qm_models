#!/usr/bin/env python3
"""
Generate SLURM scripts for missing ANOVA experiments.
One script per model per strategy to avoid slow models blocking fast ones.
"""

from pathlib import Path

# (file_label, rust_name) — file_label for filenames/job names, rust_name for --noise-strategy
STRATEGIES = [
    ('legacy',   'legacy'),
    ('valprop',  'value_proportional'),
    ('quantile', 'quantile'),
    ('threshold','threshold'),
    ('outlier',  'outlier'),
    ('hetero',   'heteroscedastic'),
]
REPS = ['ecfp4', 'pdv', 'sns', 'smiles', 'mhggnn']

SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts
"""

CMD_TEMPLATE = """
# {model_label}/{rep}
python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m {base_model} \\
    -r {rep} \\
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
    --noise-strategy {rust_strategy} \\
    -n 10000 \\
    -b 10 \\
    --normalize True \\
    {extra_flags}\\
    -f ../results/anova_{file_strategy}_{rep}_{model_label}.csv
"""

# Missing models and their configurations
MISSING = {
    # model_label: (base_model, extra_flags, reps)
    'dnn':          ('dnn', '',                                    REPS),
    'mlp':          ('mlp', '',                                    REPS),
    'gauche':       ('gauche', '',                                 ['ecfp4', 'pdv', 'sns', 'smiles']),  # no mhggnn
    'lgb':          ('lgb', '',                                    REPS),
    'dnn_bnn_full': ('dnn', '--bayesian-transformation full -u True ', REPS),
    'mlp_bnn_full': ('mlp', '--bayesian-transformation full -u True ', REPS),
}

output_dir = Path(__file__).parent
submit_lines = ['#!/bin/bash', '# Submit all missing ANOVA jobs', '']

for model_label, (base_model, extra_flags, reps) in MISSING.items():
    for file_strategy, rust_strategy in STRATEGIES:
        job_name = f"{model_label}_{file_strategy}"
        script_name = f"{model_label}_{file_strategy}.sh"

        content = SLURM_HEADER.format(job_name=job_name)

        for rep in reps:
            content += CMD_TEMPLATE.format(
                model_label=model_label,
                base_model=base_model,
                rep=rep,
                rust_strategy=rust_strategy,
                file_strategy=file_strategy,
                extra_flags=extra_flags,
            )

        script_path = output_dir / script_name
        script_path.write_text(content)
        print(f"  {script_name}")

        submit_lines.append(f'sbatch {script_name}')

    submit_lines.append('')  # blank line between model groups

# Write submit_all.sh
submit_path = output_dir / 'submit_all.sh'
submit_path.write_text('\n'.join(submit_lines) + '\n')
print(f"\n  submit_all.sh ({len([l for l in submit_lines if l.startswith('sbatch')])} jobs)")
print("Done!")
