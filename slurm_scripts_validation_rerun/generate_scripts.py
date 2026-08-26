#!/usr/bin/env python3
"""Generate individual SLURM scripts for validation re-run.

Each script runs ONE model x ONE rep x ONE dataset.
Writes to isolated dir: results/validation_rerun/{rep}_{dataset}/
This avoids race conditions from parallel jobs writing the same file.

After all jobs complete, run merge_results.py to combine into results/validation/.
"""
import os

MODELS_ALL = ['RF', 'QRF', 'XGBoost', 'DNN', 'GP', 'NGBoost', 'SVM', 'LightGBM']
MODELS_NO_GP = ['RF', 'QRF', 'XGBoost', 'DNN', 'NGBoost', 'SVM', 'LightGBM']
NON_PDV_REPS = ['ECFP4', 'SNS', 'MHG-GNN-pretrained']
DATASETS = ['logd', 'caco2', 'herg']

TIME_LIMITS = {
    'RF': '8:00:00',
    'QRF': '8:00:00',
    'XGBoost': '8:00:00',
    'LightGBM': '8:00:00',
    'SVM': '16:00:00',
    'NGBoost': '24:00:00',
    'DNN': '24:00:00',
    'GP': '47:59:00',
}

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=val_{safe_name}
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --time={time_limit}
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

cd /data/stat-cadd/scat9264/KIRBy
. ../qsar_qm_models/setup.sh

# Activation is not optional. micromamba has never worked on this cluster, so
# the `export MAMBA_EXE=...` lines that used to sit above `. setup.sh` pointed
# at a file that does not exist -- and nothing checked. setup.sh falls through
# to its conda branch; if that also fails, the task carries on in the system
# Anaconda at /apps/system/..., which has no gpytorch, no quantile_forest and
# no ngboost. The job then runs, finds nothing to do, and writes no rows. That
# is what happened to 12822693 and 12822694 (RERUN_PLAN.md section 2.8d).
if [ -z "${{CONDA_PREFIX:-}}" ]; then
    echo "ERROR: setup.sh did not activate an environment (CONDA_PREFIX unset)."
    exit 2
fi
PY_PATH="$(command -v python)"
case "$PY_PATH" in
    "$CONDA_PREFIX"/*) : ;;
    *)
        echo "ERROR: python is $PY_PATH, which is not inside the activated"
        echo "       environment ($CONDA_PREFIX). setup.sh did not activate 'env_test'."
        case "$PY_PATH" in
            /apps/system/*)
                echo "       That is the system Anaconda. It has no gpytorch, no"
                echo "       quantile_forest and no ngboost, so this job would run,"
                echo "       find nothing to do, and write no rows." ;;
        esac
        exit 2 ;;
esac
echo "=== interpreter: $PY_PATH  (CONDA_PREFIX=$CONDA_PREFIX)"

export LD_LIBRARY_PATH="${{CONDA_PREFIX:-}}/lib:${{LD_LIBRARY_PATH:-}}"

cd tests

python alternative_data_noise_robustness.py \\
    --datasets {dataset} \\
    --models {model} \\
    --reps {rep} \\
    --results-root results/validation_rerun/{rep_safe}_{dataset}

echo "Done: {model} x {rep} x {dataset}"
"""


def safe_name(rep):
    return rep.replace('-', '_').lower()


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = []

    ALL_REPS = ['ECFP4', 'SNS', 'MHG-GNN-pretrained', 'PDV']
    for model in MODELS_ALL:
        for rep in ALL_REPS:
            # GP is PDV-only in the code
            if model == 'GP' and rep != 'PDV':
                continue
            for dataset in DATASETS:
                sn = f"{model}_{safe_name(rep)}_{dataset}"
                content = SLURM_TEMPLATE.format(
                    safe_name=sn[:30],
                    time_limit=TIME_LIMITS[model],
                    dataset=dataset,
                    model=model,
                    rep=rep,
                    rep_safe=safe_name(rep),
                )
                filename = f"val_{model.lower()}_{safe_name(rep)}_{dataset}.sh"
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(content)
                scripts.append(filename)

    lines = ["#!/bin/bash", "# Submit all validation re-run jobs", "COUNT=0", ""]
    for s in sorted(scripts):
        lines.append(f"sbatch {s}")
        lines.append("COUNT=$((COUNT + 1))")
    lines += ["", f'echo "Submitted $COUNT jobs (expected {len(scripts)})"', ""]

    with open(os.path.join(output_dir, 'submit_all.sh'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated {len(scripts)} SLURM scripts + submit_all.sh")


if __name__ == '__main__':
    main()
