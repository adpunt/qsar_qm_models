# QM9 re-run — runbook

> ✅ **BROUGHT UP TO DATE 2026-08-27 (chat M).** The generator now emits the new noise CLI,
> the settled six representations and the staged design; the array sizes, script names, smoke
> test and completeness check below all follow from it. Nothing here has been run on the
> cluster yet — §1b is still a launch blocker, and the wall times in §5 are the generator's
> arithmetic, not measurement.
>
> There are no `.sh` files in this directory to submit. They are not in version control, they
> are rebuilt from `generate_scripts.py`, and §4 below regenerates them. The generator is the
> artefact; the scripts are its output.
>
> The generator has a test, and it is what stops the last failure repeating — the old generator
> emitted `--sigma` and `--noise-strategy` for weeks after the pipeline started refusing both by
> name, because nobody had executed its output:
>
> ```bash
> python slurm_scripts_qm9_rerun/test_generate_scripts.py
> ```

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
and level 0 is one seventh of the cost.

## The grid

| | |
|---|---|
| Models | 11 ANOVA models, plus QRF and both Gaussian processes |
| Representations | ECFP4, PDV (`continuous_pdv`), MHG-GNN, Avalon, ChemBERTa, Sort & Slice |
| Noise conditions | 4 at full grid: gaussian, grouped-wider, grouped-shifted, censoring |
| Noise levels | 7 per condition (`NOISE_DESIGN.md` §6.4) |
| Replicates | 10, as stage 0 (1) plus stage 1 (9) |

**Stage 0: 14 array scripts, 320 tasks**, each 7 training runs — 2,240 training runs.
**Stage 1: the same 320 tasks**, each 63 training runs — 20,160. **22,400 in total.**

Three of the four conditions repeat the level-0 cell, which is bit-identical every
time because level 0 switches the noise path off. That is 11% of the grid, and it is
kept because the figure script anchors retention on the level-0 row *within* each
(model, representation, condition) group, so dropping it would leave three quarters
of the conditions with nothing to normalise against. It also gives four independent
tasks that must agree exactly, which is a real check on the level-0 path. Sharing one
anchor across conditions is a figure-script change, and `RERUN_PLAN.md` §13.1 prices
the grid as though it had already happened.

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

**Out permanently:** binary `pdv` (superseded by PDV as continuous descriptors,
after direct comparison), `morgan` (ρ = 0.995 with ECFP4), one-hot `smiles` and
`randomized_smiles` (dropped 2026-08-26, and refused by name in `parse_mmap` so a
job cannot run them by accident), and `mol2vec`, which is deleted from the code
outright.

One task per (noise condition, representation), one script per model. Model and
representation are the two factors of the variance decomposition, so neither can
be cut without gutting the paper's first research question. Replicates can be cut
(see below); representations cannot.

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

cd scripts && python check_environment.py --deep
```

`--deep` also imports `models/models.py` itself, which is the only check that
proves the training code can start at all. It costs about a minute, which is why
it is not in the per-task guard — the tasks run the fast form.

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

> ✅ **The generator carries the activation guard and the new noise CLI**, so both land in
> whatever is regenerated. Regenerate rather than editing a `.sh` by hand — a hand-edit is lost
> the next time anyone runs the generator, and it is not covered by the generator's test.

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

## 4. Generate the scripts, then run ONE task — do not submit the grid blind

Stage 0 is the screen: every model on every representation, the four stage-1 noise
conditions, one replicate. It is reused as replicate 0 of stage 1 rather than thrown
away, which is why stage 1 starts at replicate 1 and both write to the same files.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_qm9_rerun
python generate_scripts.py --stage 0
ls qm9_s0_*.sh
```

Then the cheapest model on the cheapest representation, and let it finish:

```bash
sbatch --account=$ACCT --partition=$PART --array=0 qm9_s0_rf.sh
squeue -u $USER
tail -f qm90_rf_*_0.out
```

Then check the output is sane before committing the queue:

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv('../results/anova_gaussian_ecfp4_rf.csv')
print(d.shape, sorted(d.sigma.unique()))
print(d.groupby('sigma').r2.mean().round(3))
PY
```

The `sigma` column now carries the noise LEVEL — a fraction of the clean training
label spread, not the old σ. R² should fall monotonically with it, and at level 0
it should match the paper's clean number, because level 0 switches the noise path
off for every split.

**Record how long that task took** — it sets the wall times for everything else,
and the ones the generator prints are arithmetic, not measurement:

```bash
sacct -j <jobid> --format=JobID,JobName%22,State,Elapsed,MaxRSS
```

## 5. Submit

Four conditions x six representations is 24 tasks per model; the Tanimoto Gaussian
process runs on the two binary fingerprints only, so it is 8.

```bash
# Tier 1 — the ANOVA roster, tree and deterministic models
for s in rf xgboost lgb svm ngboost dnn mlp; do
    sbatch --account=$ACCT --partition=$PART --array=0-23%5 qm9_s0_$s.sh
done

# Tier 2 — the Bayesian networks
for s in dnn_bnn_full mlp_bnn_full dnn_bnn_full_variational mlp_bnn_full_variational; do
    sbatch --account=$ACCT --partition=$PART --array=0-23%4 qm9_s0_$s.sh
done

# Tier 3 — outside the ANOVA: uncertainty and both Gaussian processes
sbatch --account=$ACCT --partition=$PART --array=0-23%5 qm9_s0_qrf.sh
sbatch --account=$ACCT --partition=$PART --array=0-23%4 qm9_s0_gauche_rbf.sh
sbatch --account=$ACCT --partition=$PART --array=0-7%4  qm9_s0_gauche.sh   # fingerprints only
```

Stage 1 is the same grid at replicates 1–9, appending to the same files, so it is
submitted the same way once stage 0 has landed and been checked:

```bash
python generate_scripts.py --stage 1
for s in rf xgboost lgb svm ngboost dnn mlp; do
    sbatch --account=$ACCT --partition=$PART --array=0-23%5 qm9_s1_$s.sh
done
# ...and the other two tiers as above, with qm9_s1_ in place of qm9_s0_
```

## 6. Monitor and resubmit

```bash
squeue -u $USER -o "%.12i %.22j %.8T %.10M %.10L %R" | head -40
sacct -X -S today --format=JobID%18,JobName%24,State,Elapsed,MaxRSS | grep qm9_
grep -l "exit=[^0]" qm9_*.out            # failed tasks
sbatch --account=$ACCT --partition=$PART --array=7,19 qm9_dnn.sh   # only those
```

Completeness check once things land. It reads the roster out of the generator
rather than restating it, because a hand-typed list here is how the old check came
to glob for six noise names that no longer exist:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_qm9_rerun
python - <<'PYCHECK'
import glob, os, re, sys
sys.path.insert(0, '.')
import generate_scripts as gen

STAGE = 0                                    # or 1
conds = gen.STAGE_DEFAULTS[STAGE]['conditions']
want = {(c, r, m) for m, (_, _, _, _, reps) in gen.MODELS.items()
        for r in reps for c in conds}

have = set()
for f in glob.glob('../results/anova_*.csv'):
    if '_uncertainty_values' in f:
        continue
    rest = os.path.basename(f)[len('anova_'):-len('.csv')]
    for c in conds:
        if rest.startswith(c + '_'):
            tail = rest[len(c) + 1:]
            for r in sorted(gen.ALL_REPS, key=len, reverse=True):
                if tail.startswith(r + '_'):
                    have.add((c, r, tail[len(r) + 1:]))
                    break
            break

missing = sorted(want - have)
extra = sorted(have - want)
print(f'{len(want) - len(missing)}/{len(want)} present, {len(missing)} missing')
for x in missing[:20]:
    print('  missing', x)
if extra:
    print(f'{len(extra)} file(s) nothing asked for:')
    for x in extra[:10]:
        print('  unexpected', x)
PYCHECK
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
