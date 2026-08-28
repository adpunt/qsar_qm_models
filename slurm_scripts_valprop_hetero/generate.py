#!/usr/bin/env python3
"""
Generate SLURM scripts for the missing valprop + hetero strategies.
All scripts self-contained in this directory. ANOVA reps only (no sns/randomized_smiles).
"""

from pathlib import Path

STRATEGIES = [
    ('valprop',  'value_proportional'),
    ('hetero',   'heteroscedastic'),
]

# ANOVA reps only (sns and randomized_smiles excluded from ANOVA)
ANOVA_REPS = ['ecfp4', 'pdv', 'smiles', 'mhggnn']

SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time={time}
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

# ALL 11 models missing valprop + hetero for ANOVA
# model_label: (base_model, extra_flags, time_limit, reps)
MODELS = {
    'rf':                       ('rf',           '',                                              '47:59:00', ANOVA_REPS),
    'xgboost':                  ('xgboost',      '',                                              '47:59:00', ANOVA_REPS),
    'ngboost':                  ('ngboost',      '',                                              '47:59:00', ['ecfp4', 'pdv', 'smiles']),  # mhggnn already OK
    'svm':                      ('svm',          '',                                              '71:59:00', ANOVA_REPS),
    'flexible_dnn':             ('flexible_dnn', '',                                              '47:59:00', ANOVA_REPS),
    'flexible_dnn_256_128_64':  ('flexible_dnn', '--hidden-sizes 256 128 64 ',                    '47:59:00', ANOVA_REPS),
    'flexible_dnn_512_256':     ('flexible_dnn', '--hidden-sizes 512 256 ',                       '47:59:00', ANOVA_REPS),
    'dnn_bnn_last':             ('dnn',          '--bayesian-transformation last_layer -u True ',  '47:59:00', ANOVA_REPS),
    'dnn_bnn_variational':      ('dnn',          '--bayesian-transformation variational -u True ', '47:59:00', ANOVA_REPS),
    'mlp_bnn_last':             ('mlp',          '--bayesian-transformation last_layer -u True ',  '47:59:00', ANOVA_REPS),
    'mlp_bnn_variational':      ('mlp',          '--bayesian-transformation variational -u True ', '47:59:00', ANOVA_REPS),
}

output_dir = Path(__file__).parent
submit_lines = [
    '#!/bin/bash',
    '# Submit all missing valprop + hetero ANOVA jobs',
    f'# {len(MODELS)} models x 2 strategies = {len(MODELS) * 2} jobs',
    '',
]

for model_label, (base_model, extra_flags, time_limit, reps) in MODELS.items():
    for file_strategy, rust_strategy in STRATEGIES:
        job_name = f"{model_label}_{file_strategy}"
        script_name = f"{model_label}_{file_strategy}.sh"

        content = SLURM_HEADER.format(job_name=job_name, time=time_limit)

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

    submit_lines.append('')  # blank between models

# Write submit_all.sh
submit_path = output_dir / 'submit_all.sh'
submit_path.write_text('\n'.join(submit_lines) + '\n')

total = len([l for l in submit_lines if l.startswith('sbatch')])
print(f"\n  submit_all.sh ({total} jobs)")
print("Done!")
