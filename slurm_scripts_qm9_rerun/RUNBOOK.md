# QM9 re-run — runbook

> ✅ **BROUGHT UP TO DATE 2026-08-27 (chat M).** The generator now emits the new noise CLI,
> the settled six representations and the two-pass design; the array sizes, script names, smoke
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
>
> A second test covers the other half of that failure — the generator being right and the `.sh`
> on the cluster being old. It compares every `.sh` in this directory against a fresh
> generation, byte for byte, and it passes trivially here because there are none:
>
> ```bash
> python slurm_scripts_qm9_rerun/test_generated_scripts_match_generator.py
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
| Noise conditions | **3 in the array**: gaussian, grouped-wider, grouped-shifted. Censoring is the fourth settled condition but runs on a named subset of model-and-representation pairs, so it is generated and submitted separately — §5b |
| Noise levels | 7 per condition (`NOISE_DESIGN.md` §6.4) |
| Replicates | 10 — the screen contributes 1, the main grid the other 9 |

**The screen (`--stage 0`): 17 array scripts, 294 tasks** — 1,862 training runs.
**The main grid (`--stage 1`): the same 294 tasks** at replicates 1–9 — 16,758 training runs.

Those two totals were 2,058 and 18,522 until 2026-08-30, and both were 10.5% too
high. The generator worked the total out as tasks times the LONGEST condition's
level count, and the conditions do not all run the same levels — the clean level
is stripped from every condition except the reference and censoring. The wall
clock each job asks for is still worked out from the longest condition, which is
correct: one request covers a whole array, so it has to fit the worst task in it.
**20,580 between them**, plus censoring, which is generated and submitted separately (§5b).

The roster grew on 2026-08-28 with the noise-predicting Gaussian process and the two variational
networks that predict noise per molecule, which is why these are larger than the figures this file
carried before.

Every figure in this section is printed by the generator. Do not retype one: the numbers
above are checked against its output by
`python slurm_scripts_qm9_rerun/test_runbook_matches_generator.py`, which exists because
this file spent a fortnight describing a run of four conditions and 320 tasks that the
generator had stopped emitting (RERUN_PLAN.md §13.12 A7).

**The clean level runs once, under Gaussian, and is copied into the other two.**
At level 0 the pipeline does not add noise at all, and the replicate seed depends only
on the replicate number, so the clean run is bit-identical whichever condition it is
labelled with — measured on 400 QM9 molecules, random forest on ECFP4, all four
conditions returning R² = 0.7579128047581825 and RMSE = 0.5176004014184159 to the last
digit. Running it once per condition would spend 11% of the grid recomputing a number
already on disk.

It cannot simply be left out, because `auc_norm` divides each condition's curve by that
condition's own clean accuracy, so a condition with no clean row produces nothing. After
the array finishes, run:

```bash
python copy_zero_rows.py --results ../results
```

It refuses to overwrite a clean row a job actually computed — it checks that row against
the reference instead, and stops the whole copy if any of them disagrees. Run it again
after any resubmission; it does not duplicate.

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

**Out permanently, 2026-08-28:** the conformal wrappers. Not merely excluded —
`-m conformal` and `-m conformal_hetero` are refused before any data is read, and
the three entries are commented out of `EXCLUDED_MODELS`, so `--include-excluded`
cannot bring them back (RERUN_PLAN.md §2.22).

**Out by default, behind `--include-excluded`:** the last-layer BNNs (no
significant gain over base), the pre-VBLL variational BNNs (identical to
last-layer — a bug), and the flexible-DNN architecture variants. `GLOBAL_MODELS_EXCLUDE` drops all of these
from **every** figure, so re-running them produces files nothing reads. They
survive only in the no-exclusions supplementary table. Turning them on adds 5
scripts and 90 tasks — it was 8 and 144 before the three conformal entries were
commented out:

```bash
python generate_scripts.py --stage 0 --include-excluded   # 22 scripts, 384 tasks
```

That is 2,432 training runs in the screen alone, on models `GLOBAL_MODELS_EXCLUDE` drops
from every figure.

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
git fetch origin && git checkout additional_reps
bash scripts/pull_safely.sh        # NEVER a bare `git pull` here -- RERUN_PLAN.md, chat H

cd rust && cargo build --release && cd ..
ls -l rust/target/release/rust_processor      # must exist and be executable
```

> 🔴 **Checked on the cluster 2026-08-27: the binary there was built on 27 February** —
> six months old, and months before the noise redesign. Its command-line flags have
> changed since (`--sigma` and `--noise-strategy` are refused by name), so it is not
> merely stale, it cannot be driven by the current pipeline. Rebuilding is not
> optional. Confirm the date afterwards:
>
> ```bash
> ls -l --time-style=full-iso rust/target/release/rust_processor
> ```

Confirm the fix is in the binary you just built:

```bash
grep -n "apply_noise" rust/src/main.rs | head    # flag + 3 call sites (true/false/false)
```

## 1a. Source setup.sh ONCE before the first submission — in an allocation

**Not on a login node** (it caps memory per user) and **not left to the array**.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
```

**Why once is enough for all 294 tasks.** The shell environment `setup.sh` sets — `PATH`,
`CONDA_PREFIX` — is per-shell and per-node, and does not carry; every job script sources the file
itself, which is correct. What carries is on the shared project filesystem: the environment at
`/data/stat-cadd/scat9264/conda_envs/env_test`, and the stamp written inside it,
`$CONDA_PREFIX/.env_test_extras`. Every later task hashes `env.yml` and `pip-constraints.txt`,
compares against that one file, and prints *"Extras already match the recipe; nothing to install."*

**Why it now fails loudly instead of quietly, 2026-08-28.** `setup_reconcile` has always refused to
install from inside an array task. The extras block — a from-source `torchsort` compile plus three
more installs — did not, so on the first launch after `env.yml` changed, all 294 tasks would enter it
at once. It now refuses the same way, and the job scripts **exit 2** when it does, rather than
training in an environment that is not the one `env.yml` describes. Proven three ways: an array task
with a mismatched stamp refuses, an array task with a matching stamp proceeds, and an ordinary shell
with a mismatched stamp installs.

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

python scripts/check_environment.py --deep --validation
```

`check_environment.py` names the interpreter, prints every relevant package
version, **constructs** each model rather than importing its package, and fits
the two that construct cleanly and fail on contact. It must end
`OK: everything requested can be constructed`.

`--deep` adds the checks that cost minutes and belong in a preflight rather than
a per-task guard: it imports `models/models.py` for real, checks that `env.yml`
describes this interpreter, checks that `noiseInject` and `kirby` are importable,
counts the **distinct** OpenMP runtime files a job would load, and runs the two
failures that forced the environment rebuild — a LightGBM fit and a
Gaussian-process fit with the boosting libraries already loaded. `--validation`
adds the KIRBy roster. Run both flags: the environment is shared, and passing one
half at the other's expense is exactly the trap (RERUN_PLAN.md §2.8i).

**The per-task guard, in all three job families.** Every generated script runs a
cheap version of the same probe before it loads any data — seconds, not minutes,
because it constructs only the one model that task runs and does not import
`models/models.py`. The two families use different flags because they use
different model names:

| Family | Flag | Names |
|---|---|---|
| `slurm_scripts_qm9_rerun/` | `--models lgb` | as `process_and_train.py -m` spells them |
| `slurm_scripts_validation_rerun/` | `--validation-models LightGBM` | as KIRBy's `--models` spells them |
| `slurm_scripts_uncertainty_rerun/` | `--validation-models BNN-Full` | same |

`python scripts/check_environment.py --audit-roster` checks that every label all
three generators can emit is known to the probe, so a new model cannot fail its
own guard for the guard's own reason. It imports nothing and takes no time.

If `env_test` is missing or its threading check fails, rebuild it — **before a
launch, never during one**. The copy-paste block is RERUN_PLAN.md §2.8i. Sourcing
`setup.sh` is now cheap on the ordinary path: it installs the extras only when a
hash of `env.yml` + `pip-constraints.txt` has changed, so a task in a 390-task
array can no longer write into the shared environment while another reads it.

### The other interpreter — do not use it

`/data/stat-cadd/scat9264/py311-kirby` is **missing eight of the packages the roster
needs**, measured on the cluster 2026-08-27: `quantile_forest`, `gpytorch`, `gauche`,
`botorch`, `torch`, `torchbnn`, `torchhk` and `torch_geometric`. It cannot build the
neural, Bayesian, Gaussian-process or quantile-forest halves of the roster at all.

This is the environment the two dead Gaussian-process jobs actually used. It is not
worth repairing — nothing should be submitted against it. It is listed here so that
the next person who finds it knows why, and checks rather than assumes:

```bash
/data/stat-cadd/scat9264/py311-kirby/bin/python \
    /data/stat-cadd/scat9264/qsar_qm_models/scripts/check_environment.py
```

**Jobs are submitted by script, never with `--wrap`.** The two dead jobs were
`--wrap` submissions with no output path: they inherited whatever interpreter
was active and left no log saying which.

> ✅ **The generator carries the activation guard, the model-buildability probe and the new noise
> CLI**, so all three land in whatever is regenerated. Regenerate rather than editing a `.sh` by
> hand. A hand-edit is not covered by the generator's test, and it is worse than lost: a
> hand-edited script gets SKIPPED the next time the directory is regenerated, so it keeps whatever
> it had. Three scripts in the validation directory sat for weeks with the dead micromamba lines
> and no activation guard for exactly that reason, looking as sanctioned as their 85 siblings
> (RERUN_PLAN.md §13.2, chat D). If a script needs to differ, change the generator.

## 2. Archive the current results before anything overwrites them

They are the only record of what the paper claims today, and the revision guide
still references them.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
tar czf ~/results_preRerun_$(date +%Y%m%d).tar.gz results/
ls -lh ~/results_preRerun_*.tar.gz
```

## 2b. Clear three caches, not one

The standing instruction is that everything is re-run and the cache cleared, and the third
of these has bitten before.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models

# 1. The memory-mapped intermediates and any settings file left by a killed task.
#    A task writes train_<file_no>.mmap and the binary rewrites it in place.
rm -f train_*.mmap test_*.mmap val_*.mmap config_*.json
rm -f noise_manifest_*.json noise_provenance_*.csv scaffold_groups_*.json
# The jobs `cd scripts` before running, and the intermediates are opened relative
# to the working directory, so they land there and not in the repository root.
rm -f scripts/train_*.mmap scripts/test_*.mmap scripts/val_*.mmap \
      scripts/config_*.json scripts/noise_provenance_*.csv

# 1b. The results of the OLD grid. Section 2 above archives them; nothing deleted
#     them, and every anova_*.csv left in place carries rows produced under the
#     held-out-noise bug and the merged-validation-split bug. Archive first, then:
rm -f results/anova_*.csv

# 2. The processed QM9 directory. ChemBERTa changed encoder on 2026-08-27 -- 768 wide to
#    384 -- and the record layout moved with it, so anything cached before that decodes
#    every field after it at the wrong offset.
rm -rf data/QM9/processed        # see the warning below -- it MUST be rebuilt before you submit

# 3. The tuned hyperparameters. THIS IS THE ONE NOBODY EXPECTS.
ls -l results/master_tuned_hyperparameters.json results/hyperparameter_decisions.json
```

`load_best_hyperparameters` (`models/models.py`) substitutes tuned hyperparameters when
**both** of those files are present, the decisions file says `USE_TUNED`, **and the job was
given `--use-best-params`**. All twelve call sites are behind
`if hasattr(args, 'use_best_params') and args.use_best_params and not args.tuning:` -- checked
in `models/models.py` on 2026-08-28, twelve guards against twelve calls.

⚠️ **CORRECTED 2026-08-28, same day.** An earlier version of this section said no generated
script passes that flag. That was true when it was written at 05:59 and false nine minutes later:
commit `15c012c` (06:08) added `--use-best-params` to the template (`generate_scripts.py:473`), so
**every freshly generated script passes it**. The 28 `.sh` files sitting in this directory are older
than the generator and do NOT, which means the behaviour depends entirely on whether you regenerate
before copying. Regenerate — §4 — and do not submit what is on disk.

**So the tuned path is LIVE, and that creates a race the runbook must state.** The author's plan is
to submit now and let the sweep finish while the jobs sit in the queue for days. Six conditions must
hold for a tuned value to reach a model, and every one of them is evaluated **inside each individual
training run**, not once per job — so a task that starts before the two files exist does not fail.
It prints one line to the job log, trains on the shared defaults, and appends those rows to the same
`anova_<condition>_<rep>_<model>.csv` the tuned rows go to.

**The sharpest form of it.** `load_best_hyperparameters` fires once per (noise level, replicate).
A task already running when the files land produces ONE degradation curve whose early noise levels
were fitted with the defaults and whose later levels were fitted with the tuned setting. `auc_norm`
is a statistic *of that curve*. Different array tasks write different files, so this leaves no
duplicate for the deduplication to reveal, and nothing downstream reads `params_source` — the figure
script prints its value counts and warns on a mixed `spec_hash`, which cannot see this, because the
hash covers only the shared spec.

Only `svm`, `xgboost`, `lgb` and `ngboost` can flip this way — `--write-master` refuses every model
whose tuned key is shared with another — so the blast radius is 24 of 98 pairings. That is small and
it is not zero, and NGBoost is in it.

✅ **THE START-GATE NOW EXISTS, 2026-08-28.** Each generated script decides ONCE, at task
start, whether the two files are present, and passes `--use-best-params` or does not
(`generate_scripts.py`, the `TUNED_FLAG` block). A task is therefore entirely tuned or entirely
default, whichever was true when it started, and `params_source` on every row says which. The
mixed-curve failure above cannot happen any more; what remains is that different tasks in one
array can differ, which `params_source` records and which the level-by-level mixture did not.

**One operational rule the code cannot enforce:** `confirm_tuned_on_validation_datasets.py
--prune` is a SECOND writer of both files and rewrites them after they exist. Do not run it
against a live grid — a task that already decided to use the tuned setting would pick up the
pruned one on its next training run.

**For the screen specifically, the intended state is the shared defaults**, uniformly, because
the sweep is not finished. Moving the two files aside below is what guarantees it, and it makes
the screen's replicate 0 comparable with the main grid only if the main grid also runs on the
defaults. If tuning is adopted later, replicate 0 has to be re-run — one tenth of the grid.

Move them aside:

```bash
for f in master_tuned_hyperparameters hyperparameter_decisions; do
    [ -f results/$f.json ] && mv results/$f.json results/$f.superseded_$(date +%Y%m%d).json
done
ls results/ | grep -i tuned          # nothing named exactly master_tuned_hyperparameters.json
```

Only the two-file pair fires that branch, so renaming either one is enough; renaming both
leaves less to reason about. The local checkout already carries
`master_tuned_hyperparameters.superseded_2026-02.json` for the same reason.

### ⚠️ Deleting `data/QM9/processed` means it has to be rebuilt BEFORE the array

`torch_geometric`'s QM9 builds `data/QM9/processed/data_v3.pt` from the raw files on first
access, and it **takes no lock**. Submit the array cold and all 294 tasks build the same file
into the same path at once: 294 times the work at best, and at worst a task loading a `.pt`
another task is still writing.

It has to go — the ChemBERTa encoder changed on 2026-08-27 and the record layout moved with it,
so anything cached before that decodes every later field at the wrong offset. So rebuild it
**once, in the interactive allocation**, which the smoke task in §6 does as a side effect.

**The job scripts now refuse to start without it** (`generate_scripts.py`, the
`data/QM9/processed/data_v3.pt` guard) and exit 2 naming the remedy, so this cannot be
forgotten rather than merely documented. Same shape as the `setup.sh` extras refusal: the
expensive shared work belongs in one allocation before a launch, never in 294 tasks at once.

## 3. Account and partition

```bash
# The live KIRBy checkout is /data/stat-ecr, NOT stat-cadd. 125 of the 127 job
# scripts in that repository use stat-ecr, and where_to_submit.sh's own header
# says so (RERUN_PLAN.md 2.8b). This block used to `cd` to stat-cadd with no
# `|| exit`, so on a missing directory it ran the next line from the wrong place
# and returned an EMPTY account and partition -- and every sbatch below then
# failed on an empty --account.
cd /data/stat-ecr/scat9264/KIRBy || { echo "KIRBy checkout not found"; exit 1; }
bash tests/slurm_scripts/where_to_submit.sh            # read sections 2, 3, 5

# --emit returns the association with the HIGHEST FAIRSHARE, which is not
# necessarily the one these jobs bill to. QM9 runs under stat-cadd. Take the
# partition from the script and pin the account by hand.
read -r EMIT_ACCT PART < <(bash tests/slurm_scripts/where_to_submit.sh --emit)
ACCT=stat-cadd
echo "account=$ACCT (emit suggested $EMIT_ACCT)  partition=$PART"

# Recorded limits, 2026-08-28: short 12 h, medium 2 days, long 30 days.
# The screen's longest tier asks 23:59, so `medium` holds all of it and a shorter
# requested wall backfills sooner. Use `long` only for the main grid.
sinfo -o "%.12P %.12l" | grep -E "medium|long"
PART=medium
```

## 4. Generate the scripts, then run ONE task — do not submit the grid blind

**The screen** — `--stage 0` on the command line — is every model on every representation,
the main grid's four noise conditions, one replicate. It is kept and reused as replicate 0
of the main grid rather than thrown away, which is why the main grid starts at replicate 1
and the two write to the same files.

(The generator's flag is still spelled `--stage`. The words for the four passes are **the
screen**, **the main grid**, **the deep run** and **the uncertainty runs** — RERUN_PLAN.md
§13. "Stage N" meant four different things across three weeks of this project and is not
used in prose here.)

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_qm9_rerun
rm -f qm9_s0_*.sh
python generate_scripts.py --stage 0
ls qm9_s0_*.sh
python test_generated_scripts_match_generator.py
```

**Delete before regenerating, and run that last line.** The `.sh` are not in git, so what sits
here is whatever was generated on this host last, and the generator overwrites the scripts it
still emits while leaving the rest. The gap between the last set that WAS committed and a
generation from today's source was not cosmetic: `--oof-folds 5` missing from all sixteen
scripts whose model reports an uncertainty (and without it a QM9 uncertainty row is a test row,
whose label is never corrupted — the question "does the uncertainty find the corrupted labels?"
would have had no data behind it), the representation named `continuous_pdv` which the reader
now refuses by name, no tuned-hyperparameter block, neither of the two start-up refusals, three
models with no script at all, and wall clocks four to six times too short for the fits the
scripts now ask for.

**Before the first submission, run the concurrency check — it only works here.**

Two array tasks share a working directory. They used to share one settings file with it, and
the file names which memory-mapped training files the binary opens **and rewrites**, so two
tasks could silently overwrite each other's training data. That is fixed and there is a check
for it, but the substantive half of that check starts two real training tasks side by side, and
it **skips on macOS** — a library the Gaussian process pulls in writes two fixed-name files
under `/tmp` during import, so two simultaneous imports race on a laptop for reasons that have
nothing to do with this. The cluster is the only place the check means anything.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
# `interactive` is the partition for --pty. `short` was advised twice and had
# ZERO idle nodes both times, so the request was valid and had nothing to land
# on (RERUN_PLAN.md 2.8i).
srun --account=stat-cadd --partition=interactive --cpus-per-task=8 --mem=32G \
     --time=01:00:00 --pty python scripts/test_config_isolation.py --end-to-end
```

Three checks must pass. It is also the first end-to-end confirmation that the pipeline runs on
this cluster at all. If it fails, do not raise job concurrency and do not submit the grid.

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

Three conditions x six representations is 18 tasks per model, so `--array=0-17`; the
Tanimoto Gaussian process runs on the two binary fingerprints only, so it is 6 and
`--array=0-5`. An index past the end exits 2 on the generator's own guard, so a range
that is too wide mis-submits and one that is too narrow drops cells silently.

⚠️ **This block covers all SEVENTEEN scripts.** Until 2026-08-28 it listed fourteen: the
three decomposition models added that day — `heteroscedastic_gp` and the two heteroscedastic
variational networks — appeared in no `sbatch` line, so 54 of the 294 tasks would never have
been queued. Those three are the only models in the roster that report both halves of the
uncertainty per molecule, so their absence would have been discovered at analysis time.
Submit by looping over the generator's own output rather than a hand-typed list:

```bash
# Every script the generator wrote, nothing hand-typed. 17 scripts, 294 tasks.
ls qm9_s0_*.sh | wc -l          # must print 17

# Tier 1 — the ANOVA roster, tree and deterministic models
for s in rf xgboost lgb svm ngboost dnn mlp; do
    sbatch --account=$ACCT --partition=$PART --array=0-17%5 qm9_s0_$s.sh
done

# Tier 2 — the Bayesian networks
for s in dnn_bnn_full mlp_bnn_full dnn_bnn_full_variational mlp_bnn_full_variational; do
    sbatch --account=$ACCT --partition=$PART --array=0-17%4 qm9_s0_$s.sh
done

# Tier 3 — outside the ANOVA: uncertainty, both Gaussian processes, and the three
# models that report BOTH uncertainty components per molecule
for s in heteroscedastic_gp dnn_bnn_full_variational_hetero mlp_bnn_full_variational_hetero; do
    sbatch --account=$ACCT --partition=$PART --array=0-17%4 qm9_s0_$s.sh
done
sbatch --account=$ACCT --partition=$PART --array=0-17%5 qm9_s0_qrf.sh
sbatch --account=$ACCT --partition=$PART --array=0-17%4 qm9_s0_gauche_rbf.sh
sbatch --account=$ACCT --partition=$PART --array=0-5%4  qm9_s0_gauche.sh   # fingerprints only
```

The main grid is the same 294 tasks at replicates 1–9, appending to the same files, so it
is submitted the same way once the screen has landed and been checked:

```bash
# THE MAIN GRID GOES TO `long`, AND IT MUST. Nine replicates is 63 training runs
# per task, and every model that emits an uncertainty does six fits per run, so
# ngboost's tier needs 202 hours against medium's 48. The generator REFUSES to
# write a script it cannot honour -- it does not cap the request -- so this needs
# the long partition's ceiling passed explicitly.
python generate_scripts.py --stage 1 --max-hours 720
for s in rf xgboost lgb svm ngboost dnn mlp; do
    sbatch --account=$ACCT --partition=long --array=0-17%5 qm9_s1_$s.sh
done
# ...and the other two tiers as above, with qm9_s1_ in place of qm9_s0_
```

⚠️ **Read the levers below before submitting the main grid.** A 202-hour request cannot
backfill into a short idle gap, which is exactly the reason `where_to_submit.sh` gives for a
job sitting on (Priority) while CPUs are free. `--replicates 4` is the lever, and the screen
supplies replicate 0, so four gives five in total — the minimum the balanced decomposition
accepts.

## 5b. Censoring — separate, and it must be, twice over

Censoring is the fourth settled condition, and it is **not in the array above**. It runs on
a named subset of model-and-representation pairs rather than all 80, because the question there is
how big the effect is and not which model resists it best — 70 training runs per pair against 5,460
for the full grid (`noise_conditions.json`, RERUN_PLAN.md §13.13). Which pairs comes out of the screen, and none are chosen yet;
the same way the deep run's selection does, so this is submitted **after** §5 has landed.

Two things the generator refuses, both found by the close-out audit after they had already
happened once:

- **It will not generate censoring at full breadth.** Ask for it without `--models` and
  `--reps` and it names the choice rather than making one.
- **It will not write censoring into this directory.** Scripts are named by model and pass
  index only, so `qm9_s0_rf.sh` for censoring would overwrite `qm9_s0_rf.sh` for the main
  grid — exit 0, no warning, and the files are untracked so git could not restore them.
  `--out-dir` is required.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_qm9_rerun
python generate_scripts.py --stage 0 --conditions censoring \
    --models <the chosen models> --reps <the chosen representations> \
    --out-dir ../slurm_scripts_qm9_censoring
```

It runs 10 replicates numbered 0–9, not 9 numbered 1–9: the screen supplies replicate 0 for
the other conditions and there is no screen for this one.

Censoring has no clean row of its own — the generator strips level 0 from every condition
but the reference — so `copy_zero_rows.py` supplies it from the Gaussian run for the same
pairs. Run it after these land as well.

## 6. Monitor and resubmit

```bash
squeue -u $USER -o "%.12i %.22j %.8T %.10M %.10L %R" | head -40
sacct -X -S today --format=JobID%18,JobName%24,State,Elapsed,MaxRSS | grep qm9[01]_
grep -l "exit=[^0]" qm90_*.out           # failed tasks -- the screen is qm90_, the main grid qm91_
sbatch --account=$ACCT --partition=$PART --array=7,15 qm9_s0_dnn.sh   # only those
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

STAGE = 0                                    # 0 = the screen, 1 = the main grid
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

Halving the replicate count still clears every gate in the analysis (which needs at
least 5). It costs precision on the residual term, which is itself a reported result —
so it is a real trade, not a free one. The screen supplies replicate 0, so the main
grid's four give five in total:

```bash
python generate_scripts.py --stage 1 --replicates 4 --max-hours 720
```

The generator's flag is `--replicates`. `--bootstrapping` was its old spelling, and
although `process_and_train.py` still accepts it as an alias, the generator does not have
it at all and `test_generate_scripts.py` refuses any script that carries it.

**A large free saving that is NOT applied here.** For every noise level and every
replicate the pipeline re-shuffles QM9, redoes the scaffold split, and recomputes
every molecular representation from scratch — then the Rust step re-reads and
rewrites the whole file. None of that depends on the noise level; only the labels
change. Measured locally, the RDKit descriptor set alone takes **220 seconds per
10,000 molecules** and is recomputed 110 times per output file — about 6.7 hours
of which 6.1 is pure repetition. Caching the prepared split per replicate would
cut the preparation step by roughly 91%.

It is not applied because it is a change to the QM9 data path that **cannot be
tested on this laptop** — the local Python environment cannot import
`torch_geometric` (two compiled extensions do not match the installed PyTorch),
so `process_and_train.py` will not run here at all. Making an untestable change to
the code that writes the training data, the night before a long run, is how you
get a second invalid grid. Do it when it can be smoke-tested on the cluster.

Also inherent to splitting one strategy per task: the σ = 0 point is recomputed
six times per (model, representation), because at σ = 0 all six strategies are
byte-identical. That is about 9% of the bill.
