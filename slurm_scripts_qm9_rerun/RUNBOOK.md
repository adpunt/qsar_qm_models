# QM9 re-run — runbook

> ⚠️ **PARTLY SUPERSEDED (2026-08-24). Do not submit these scripts as they stand.**
> The noise scheme is being replaced (`NOISE_DESIGN.md`), so the six noise types and the
> eleven-level ladder below are both out of date, and the completeness check at the end globs
> for names that will no longer exist. Several code changes must also land before any of this
> runs — see `RERUN_PLAN.md` §5.1.
>
> What is still good and should be carried across when the scripts are regenerated: the reasoning
> about which models and representations are in and out, the tier ordering, the one-task-first
> discipline, and the archive step.

The noise map is keyed by **training** index, and `write_data` restarted its
counter for each split — so validation and test molecules were handed the noise
drawn for the *training* molecule at the same position. Two consequences:

1. **Held-out labels were corrupted**, contrary to the Methods. Every R² above
   σ = 0 was scored against a moving target, mixing "the model got worse" with
   "the answer moved".
2. **The corruption was attached to the wrong molecules**, so any correlation
   between a molecule's predicted uncertainty and "its" noise was zero by
   construction.

Fixed in `rust/src/main.rs` — `write_data` takes an `apply_noise` flag, true only
for the training call. **The binary must be rebuilt on the cluster**; the job
scripts refuse to start without it rather than silently regenerating the same
invalid numbers.

**σ = 0 results were never affected** (`process_and_train.py` sets
`'noise': s > 0`, so at zero the noise path is off for all splits). They are
re-run anyway — splicing two runs is more error-prone than re-running one grid,
and σ = 0 is one eleventh of the cost.

## The grid

| | |
|---|---|
| Models | 11 ANOVA models, plus QRF and both Gaussian processes |
| Representations | ECFP4, continuous PDV, SMILES, MHG-GNN, Mol2vec, SNS |
| Strategies | 6 |
| Noise levels | 11 (0.0 to 1.0) |
| Replicates | 10 |

**14 array scripts, 480 tasks**, each 110 training runs — **52,800 training
runs** in total.

### What is in, and what is deliberately out

**In, beyond the ANOVA roster:**

- **SNS.** Excluded from the variance decomposition as redundant with ECFP4
  (ρ = 0.90), but it is still *reported*: `generate_paper_figures_v2.py:2313`
  prints SNS specifically and `:4065` slices on it, because
  `table1_supp_simple_effects_all_reps.csv` is deliberately built with no
  exclusions (`:2251`). Leaving it out would have left that table short.
- **QRF.** Dropped from the ANOVA as redundant with RF for accuracy (ρ = 0.996),
  but it is the strongest error-ranker in the uncertainty results.
- **Both Gaussian processes.** `gauche_rbf` (RBF) on all six representations, so
  one consistent kernel finally spans every representation and the GP can enter
  the cross-representation ANOVA. `gauche` (Tanimoto) on the two **binary
  fingerprint** representations only — Tanimoto is undefined on continuous
  features. Together these are the RBF-versus-Tanimoto comparison, which
  separates "the GP is good" from "the kernel suits this representation".

**Out by default, behind `--include-excluded`:** the conformal wrappers
(ρ > 0.99 with their base models), the last-layer BNNs (no significant gain over
base), the pre-VBLL variational BNNs (identical to last-layer — a bug), and the
flexible-DNN architecture variants. `GLOBAL_MODELS_EXCLUDE` drops all of these
from **every** figure, so re-running them produces files nothing reads. They
survive only in the no-exclusions supplementary table. Turning them on adds 8
scripts and 240 tasks:

```bash
python generate_scripts.py --include-excluded     # 22 scripts, 720 tasks
```

**Out permanently:** binary `pdv` (superseded by continuous PDV, and dropped
after direct comparison), `morgan` (ρ = 0.995 with ECFP4), and
`randomized_smiles` (incomplete coverage). All three are in
`ANOVA_REPS_EXCLUDE`.

One task per (strategy, representation), one script per model. Model and
representation are the two factors of the variance decomposition, so neither can
be cut without gutting the paper's first research question. Replicates can be cut
(see below); representations cannot.

Two arms sit outside the ANOVA roster and are included deliberately:
- **QRF** — dropped from the ANOVA as redundant with RF for accuracy (ρ = 0.996),
  but it is the strongest error-ranker in the uncertainty results.
- **The RBF Gaussian process on every representation** — it has never been run on
  the paper's primary representation at all, so it is a visible hole in two
  figures. Running one consistent kernel across all five representations is also
  what would let the GP finally enter the cross-representation ANOVA.

## 1. Rebuild the binary — nothing works without this

```bash
ssh gateway.arc.ox.ac.uk && ssh arc-login

cd /data/stat-cadd/scat9264/qsar_qm_models
git fetch origin && git checkout additional_reps && git pull --ff-only origin additional_reps

cd rust && cargo build --release && cd ..
ls -l rust/target/release/rust_processor      # must exist and be executable
```

Confirm the fix is in the binary you just built:

```bash
grep -n "apply_noise" rust/src/main.rs | head    # flag + 3 call sites (true/false/false)
```

## 1b. Audit the interpreter — this is a launch blocker

Two Gaussian-process jobs (`12822693`, `12822694`) ran to completion on
2026-08-19 and produced nothing. The cause, confirmed on the cluster on
2026-08-26, is worse than a missing package in one submission:

**`micromamba` has never worked on this cluster.** `setup.sh` has always fallen
through to its `conda` branch. Every job script in this directory used to open
with

```bash
export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"   # does not exist
eval "$("$MAMBA_EXE" shell hook --shell bash)"               # fails
```

and because the scripts run under `set -uo pipefail` with **no `-e`**, that
failure stopped nothing. A task that also failed to activate carried on in
whatever python was on `PATH` — the system Anaconda at
`/apps/system/easybuild/software/Anaconda3/2022.05/bin/python`, which has
**no `gpytorch`, no `quantile_forest` and no `ngboost` at all**. The job then
runs, finds nothing to do, and produces no rows.

The dead `MAMBA_EXE` lines are gone, and the generated scripts now refuse to
start if `CONDA_PREFIX` is unset or the interpreter resolves under
`/apps/system`.

### Confirm the environment before submitting

Activate exactly the way the jobs do — through conda, not micromamba:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list                     # env_test must be here
conda activate env_test

python -c "import sys; print(sys.executable)"     # must NOT be under /apps/system
python -m pip check | grep -i scikit-learn || echo "no scikit-learn conflicts"

cd scripts && python check_environment.py
```

`check_environment.py` names the interpreter, prints every relevant package
version, **constructs** each model rather than importing its package, and fits
the two that construct cleanly and fail on contact. It must end
`OK: everything requested can be constructed`.

If `env_test` is missing, `. setup.sh` builds it from `env.yml` — but note that
setup.sh also runs four network `pip install` calls **on every invocation**, so
every task in a 390-task campaign re-installs packages at startup. Build the
environment once, interactively, before submitting.

### The other interpreter

`/data/stat-cadd/scat9264/py311-kirby` is a separate environment that the
uncertainty work has used. Check it too if you submit anything against it:

```bash
/data/stat-cadd/scat9264/py311-kirby/bin/python \
    /data/stat-cadd/scat9264/qsar_qm_models/scripts/check_environment.py
```

**Jobs are submitted by script, never with `--wrap`.** The two dead jobs were
`--wrap` submissions with no output path: they inherited whatever interpreter
was active and left no log saying which.

## 2. Archive the current results before anything overwrites them

They are the only record of what the paper claims today, and the revision guide
still references them.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
tar czf ~/results_preRerun_$(date +%Y%m%d).tar.gz results/
ls -lh ~/results_preRerun_*.tar.gz
```

## 3. Account and partition

```bash
cd /data/stat-cadd/scat9264/KIRBy
bash tests/slurm_scripts/where_to_submit.sh            # read sections 2, 3, 5
read -r ACCT PART < <(bash tests/slurm_scripts/where_to_submit.sh --emit)
echo "account=$ACCT partition=$PART"
sinfo -o "%.12P %.12l" | grep -E "medium|long"         # confirm >= 48 h wall
```

## 4. One task first — do not submit 390 blind

Pick the cheapest model and the cheapest representation, and let it finish:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_qm9_rerun
sbatch --account=$ACCT --partition=$PART --array=0 qm9_rf.sh
squeue -u $USER
tail -f qm9_rf_*_0.out
```

Then check the output is sane before committing the queue:

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv('../results/anova_legacy_ecfp4_rf.csv')
print(d.shape, sorted(d.sigma.unique()))
print(d.groupby('sigma').r2.mean().round(3))
PY
```

R² should fall monotonically with σ and start near the published clean value.
**Record how long that task took** — it sets the wall times for everything else:

```bash
sacct -j <jobid> --format=JobID,JobName%22,State,Elapsed,MaxRSS
```

## 5. Submit

```bash
# Tier 1 — the ANOVA roster, tree and deterministic models
for s in qm9_rf qm9_xgboost qm9_lgb qm9_svm qm9_ngboost qm9_dnn qm9_mlp; do
    sbatch --account=$ACCT --partition=$PART --array=0-35%5 $s.sh
done

# Tier 2 — the Bayesian networks
for s in qm9_dnn_bnn_full qm9_mlp_bnn_full \
         qm9_dnn_bnn_full_variational qm9_mlp_bnn_full_variational; do
    sbatch --account=$ACCT --partition=$PART --array=0-35%4 $s.sh
done

# Tier 3 — outside the ANOVA: uncertainty and both Gaussian processes
sbatch --account=$ACCT --partition=$PART --array=0-35%5 qm9_qrf.sh
sbatch --account=$ACCT --partition=$PART --array=0-35%4 qm9_gauche_rbf.sh
sbatch --account=$ACCT --partition=$PART --array=0-11%4 qm9_gauche.sh   # 12 tasks: fingerprints only
```

## 6. Monitor and resubmit

```bash
squeue -u $USER -o "%.12i %.22j %.8T %.10M %.10L %R" | head -40
sacct -X -S today --format=JobID%18,JobName%24,State,Elapsed,MaxRSS | grep qm9_
grep -l "exit=[^0]" qm9_*.out            # failed tasks
sbatch --account=$ACCT --partition=$PART --array=7,19 qm9_dnn.sh   # only those
```

Completeness check once things land — this is the same audit the figure script
runs, so use it rather than counting files by hand:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
python -c "
import glob, re, collections
want_reps = ['ecfp4','continuous_pdv','smiles','mhggnn','mol2vec','sns']
want_strat = ['legacy','valprop','quantile','threshold','hetero','outlier']
models = ['rf','xgboost','lgb','svm','ngboost','dnn','mlp','dnn_bnn_full','mlp_bnn_full',
          'dnn_bnn_full_variational','mlp_bnn_full_variational','qrf','gauche_rbf']
fp_only = {'gauche': ['ecfp4','sns']}   # Tanimoto GP: binary fingerprints only
have = set()
for f in glob.glob('../results/anova_*.csv'):
    m = re.match(r'.*anova_(\w+?)_(ecfp4|continuous_pdv|smiles|mhggnn|mol2vec)_(.+)\.csv', f)
    if m: have.add(m.groups())
miss = [(s,r,mo) for s in want_strat for r in want_reps for mo in models if (s,r,mo) not in have]
miss += [(s,r,'gauche') for s in want_strat for r in fp_only['gauche'] if (s,r,'gauche') not in have]
print(f'{len(have)} present, {len(miss)} missing of {len(want_strat)*(len(want_reps)*len(models)+len(fp_only["gauche"]))}')
for x in miss[:20]: print('  missing', x)
"
```

## 7. Regenerate the figures

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_analysis
sbatch --account=$ACCT --partition=$PART run_figures_v2.sh     # NOT run_figures.sh
```

`run_figures.sh` is the retired script — it still computes the old robustness
metric and writes to `results/paper_figures/`. The live one is `run_figures_v2.sh`.

## Cost, and the levers

`--bootstrapping 5` halves everything and still clears every gate in the analysis
(which needs at least 5). It costs precision on the residual term, which is
itself a reported result — so it is a real trade, not a free one.

```bash
python generate_scripts.py --bootstrapping 5
```

**A large free saving that is NOT applied here.** For every noise level and every
replicate the pipeline re-shuffles QM9, redoes the scaffold split, and recomputes
every molecular representation from scratch — then the Rust stage re-reads and
rewrites the whole file. None of that depends on the noise level; only the labels
change. Measured locally, the RDKit descriptor set alone takes **220 seconds per
10,000 molecules** and is recomputed 110 times per output file — about 6.7 hours
of which 6.1 is pure repetition. Caching the prepared split per replicate would
cut the preparation stage by roughly 91%.

It is not applied because it is a change to the QM9 data path that **cannot be
tested on this laptop** — the local Python environment cannot import
`torch_geometric` (two compiled extensions do not match the installed PyTorch),
so `process_and_train.py` will not run here at all. Making an untestable change to
the code that writes the training data, the night before a long run, is how you
get a second invalid grid. Do it when it can be smoke-tested on the cluster.

Also inherent to splitting one strategy per task: the σ = 0 point is recomputed
six times per (model, representation), because at σ = 0 all six strategies are
byte-identical. That is about 9% of the bill.
