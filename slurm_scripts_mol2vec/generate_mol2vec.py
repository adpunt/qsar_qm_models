#!/usr/bin/env python3
"""
Generate SLURM scripts for mol2vec representation across all ANOVA models.
One script per model, running all 6 strategies sequentially.
"""

from pathlib import Path

STRATEGIES = [
    ('legacy',   'legacy'),
    ('valprop',  'value_proportional'),
    ('quantile', 'quantile'),
    ('threshold','threshold'),
    ('outlier',  'outlier'),
    ('hetero',   'heteroscedastic'),
]

# All ANOVA models and their configurations
# model_label: (base_model, extra_flags, time, mem)
MODELS = {
    # Tree-based (fast)
    'rf':          ('rf',     '',                                    '23:59:00', '64G'),
    'xgboost':     ('xgboost','',                                    '23:59:00', '64G'),
    'lgb':         ('lgb',    '',                                    '23:59:00', '64G'),
    'svm':         ('svm',    '',                                    '23:59:00', '128G'),
    'ngboost':     ('ngboost','',                                    '47:59:00', '128G'),
    # Neural networks (medium)
    'dnn':              ('dnn',          '',                          '47:59:00', '128G'),
    'mlp':              ('mlp',          '',                          '47:59:00', '128G'),
    'flexible_dnn':     ('flexible_dnn', '',                          '47:59:00', '128G'),
    'flexible_dnn_256_128_64': ('flexible_dnn', '--hidden-sizes 256 128 64 ', '47:59:00', '128G'),
    'flexible_dnn_512_256':    ('flexible_dnn', '--hidden-sizes 512 256 ',     '47:59:00', '128G'),
    # BNN variants (slow)
    'dnn_bnn_last':        ('dnn', '--bayesian-transformation last -u True ',        '47:59:00', '128G'),
    'dnn_bnn_variational': ('dnn', '--bayesian-transformation variational -u True ', '47:59:00', '128G'),
    'mlp_bnn_last':        ('mlp', '--bayesian-transformation last -u True ',        '47:59:00', '128G'),
    'mlp_bnn_variational': ('mlp', '--bayesian-transformation variational -u True ', '47:59:00', '128G'),
    'dnn_bnn_full':        ('dnn', '--bayesian-transformation full -u True ',        '47:59:00', '128G'),
    'mlp_bnn_full':        ('mlp', '--bayesian-transformation full -u True ',        '47:59:00', '128G'),
    # GP (slow, but mol2vec is only 300d so should be OK)
    'gauche':      ('gauche', '',                                    '47:59:00', '128G'),
}

SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name=mol2vec_{model_label}
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

CMD_TEMPLATE = """
# {model_label}/mol2vec/{file_strategy}
python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m {base_model} \\
    -r mol2vec \\
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
    --noise-strategy {rust_strategy} \\
    -n 10000 \\
    -b 10 \\
    --normalize True \\
    {extra_flags}\\
    -f ../results/anova_{file_strategy}_mol2vec_{model_label}.csv
"""

output_dir = Path(__file__).parent
submit_lines = ['#!/bin/bash', '# Submit all mol2vec ANOVA jobs', '']

for model_label, (base_model, extra_flags, time, mem) in MODELS.items():
    script_name = f"mol2vec_{model_label}.sh"
    content = SLURM_HEADER.format(model_label=model_label, time=time, mem=mem)

    for file_strategy, rust_strategy in STRATEGIES:
        content += CMD_TEMPLATE.format(
            model_label=model_label,
            base_model=base_model,
            rust_strategy=rust_strategy,
            file_strategy=file_strategy,
            extra_flags=extra_flags,
        )

    script_path = output_dir / script_name
    script_path.write_text(content)
    print(f"  {script_name}")
    submit_lines.append(f'sbatch {script_name}')

submit_lines.append('')
submit_path = output_dir / 'submit_all.sh'
submit_path.write_text('\n'.join(submit_lines) + '\n')
print(f"\n  submit_all.sh ({len([l for l in submit_lines if l.startswith('sbatch')])} jobs)")
print("Done!")
