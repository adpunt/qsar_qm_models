#!/usr/bin/env python3
"""
Generate SLURM scripts for gauche + mhggnn experiments.
Extra memory (256G) and time (72h) for GP on 1024d input.
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

SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name=gauche_mhggnn_{file_strategy}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts
"""

CMD_TEMPLATE = """
# gauche/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \\
    -m gauche \\
    -r mhggnn \\
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
    --noise-strategy {rust_strategy} \\
    -n 10000 \\
    -b 10 \\
    --normalize True \\
    -f ../results/anova_{file_strategy}_mhggnn_gauche.csv
"""

output_dir = Path(__file__).parent
submit_lines = ['#!/bin/bash', '# Submit gauche+mhggnn jobs (extra resources)', '']

for file_strategy, rust_strategy in STRATEGIES:
    script_name = f"gauche_mhggnn_{file_strategy}.sh"
    content = SLURM_HEADER.format(file_strategy=file_strategy)
    content += CMD_TEMPLATE.format(
        rust_strategy=rust_strategy,
        file_strategy=file_strategy,
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
