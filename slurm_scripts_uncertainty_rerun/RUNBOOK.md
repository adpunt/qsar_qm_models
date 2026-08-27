# Uncertainty re-run — runbook

> ✅ **BROUGHT ONTO THE SETTLED NOISE CONDITIONS, 2026-08-27.** The generator listed the six
> deleted strategies and emitted flags the runner no longer has, so every task would have died at
> argument parsing. It now reads `noise_conditions.json` and emits `--conditions` /
> `--unc-conditions`. `scripts/test_uncertainty_job_scripts.py` runs the command line each
> generated script emits through the runner's own argument parser and fails if any of it is
> refused. **Regenerate before submitting** — `python generate_scripts.py`.

**What this produces:** the data needed to answer both uncertainty questions.

- **(A) Do the corrupted molecules come back as the uncertain ones?**
  Measured on **training** molecules, scored **out-of-fold** so no molecule is
  judged by a model that fitted its own corrupted label.
- **(B) Does the model learn *where* the data is unreliable?**
  Measured on **test** molecules against the noise scale their region receives.
  Only some conditions give different molecules different amounts, so only those
  have a pattern to learn. Where every molecule gets the same amount the
  question-B correlation is **undefined, not zero**.

**The five conditions, and which question each serves.** The run takes the four
the main grid runs and adds one — `outlier_p10` — which is the recorded default,
`RERUN_PLAN.md` §13.1 item 2. Under a condition that gives every molecule the
same amount of noise, "which molecules were corrupted" is undefined, not zero, so
only the patterned ones can answer question B at all.

| condition | per-molecule pattern? | what it is for |
|---|---|---|
| `gaussian` | no — every molecule gets the same amount | question A, and the leakage check |
| `grouped_shifted` | no — whole scaffold families pushed one way, by a constant | question A |
| `grouped_wider` | **yes**, keyed to the scaffold | question B |
| `censoring` | **yes**, keyed to the label | question B |
| `outlier_p10` | **yes** — a tenth of the molecules take nearly all of it | question B |

`outlier_p10` is depth-only on the main grid and is added here because it is the
only one of the three depth-only conditions that is not flat by design, and the
only concentrated-noise condition left in the study — which is the case the
question was raised about. It costs 25% of the run. `--main-grid-only` drops back
to the inherited four; `--include-deep-conditions` adds Student-t and Laplace as
well, both flat, so they buy shape coverage for question A and nothing for
question B. Whichever is run, the Methods must say which.

**Both grouped conditions are keyed to something a scaffold split holds out
whole.** On held-out molecules the grouped pattern is flat, truthfully, and the
predicted-label control is degenerate for it — `RERUN_PLAN.md` §3.1d. That is a
Methods sentence, not a defect. Censoring and outlier are keyed to the label and
to the draw, so neither is affected.

Both questions come out of one set of jobs. Question B costs nothing extra; the
only added compute is the out-of-fold folds, and only for the seven models that
emit a per-molecule uncertainty.

### The confound in question B, and the control for it

The amount of noise a molecule receives is a **deterministic function of the
molecule**: under censoring it is a function of the label, under the grouped
conditions a function of the scaffold. A model's predicted uncertainty may
already track those for reasons that have nothing to do with noise — extreme
molecules are simply harder, and rare scaffolds are simply less well covered. So
a raw correlation between uncertainty and the noise amount would be partly
manufactured.

Every row therefore carries **two** noise columns:

| column | meaning |
|---|---|
| `noise_scale` | the noise scale actually applied at this level. Exactly **0** at level 0. |
| `noise_pattern` | the *shape* — which molecules the condition hits hardest — taken at a fixed reference level, so it is **the same column at every level, including 0**. |

The defensible effect for question B is therefore

> ρ(uncertainty at level ℓ, `noise_pattern`) **minus** ρ(uncertainty at level 0, `noise_pattern`)

The level-0 model was trained on completely clean labels but saw the same label
distribution, so its correlation is exactly the confound. Report the difference,
not the raw number.

`gaussian` and `grouped_shifted` give every molecule the same noise scale, so
`noise_pattern` is flat and the question-B correlation is **undefined rather than
zero**. Neither is a control for question B — a control has to produce a number.
Their role is in question A and as the leakage check. Preflight section 4b prints
the per-(dataset, condition) count of distinct noise scales, so a condition that
is flat where it should not be is visible before the queue is spent.

**Scope:** 3 datasets × 7 models × 4 representations × 5 conditions × 6 noise
levels (7 for censoring, which sweeps the clipped fraction instead) × 5 scaffold
folds. 7 array scripts, 60 tasks each, **420 tasks**.

The levels are **not** passed on the command line. The runner anchors them per
dataset to published assay error — logD 0.15, Caco-2 0.35, hERG 0.54 log units —
and one shared ladder would be six different experiments across the three
datasets. `--sigmas` would override that, so the job scripts do not use it.

---

## 0. What changed in the code

| Repo | File | Change |
|---|---|---|
| `NoiseInject` | `noiseInject/core.py` | New `noise_scale()` (per-molecule noise scale, draws no randomness) and `inject_verbose()` (returns the noise it actually drew). Strategies refactored onto one shared scale function. **Verified bit-identical to the old code on 336 checks.** |
| `KIRBy` | `tests/alternative_data_noise_robustness.py` | Runners now return an `extras` dict carrying the true injected noise, the per-molecule noise scale for train *and* test, and out-of-fold training predictions. Uncertainty saved for **every** condition, with `split`/`noise_type`/`fold`/`noise_scale`/`injected_noise` columns. Flags `--conditions`, `--unc-conditions`, `--oof-folds`, `--oof-outer-folds`, `--no-noise-validation`. Three bugs fixed — see below. |
| `qsar_qm_models` | `slurm_scripts_uncertainty_rerun/` | This directory. |

### The nine defects found in adversarial review — all fixed

| # | Defect | Fix |
|---|---|---|
| 1 | The neural cross-fitting consumed the global torch generator, so the **main** neural results at every noise level would silently differ from a run made without `--oof-folds` | Snapshot and restore the generator around the out-of-fold block |
| 2 | Uncertainty was held in memory and written only after all five folds — a wall-clock timeout destroyed everything the run exists to produce | Written after **every** fold, atomically, keyed so re-writing a fold replaces rather than duplicates |
| 3 | Cross-fitting used a **random** split, breaking the project's scaffold-split rule and putting out-of-fold uncertainty on a different scale from the test set | Scaffold groups threaded through; inner split is now `GroupKFold`, with a logged fallback if a fold has too few scaffolds |
| 4 | The Gaussian process is capped at 2,000 training molecules, and its out-of-fold rows were numbered against that subsample — so joining GP to QRF on molecule index paired **different molecules** | The subsample's real row indices are carried through and written as `sample_idx` |
| 5 | A failed inner fold was caught, blanked, and written; the coverage report then marked the cell OK | `_oof_predict` returns how many folds succeeded; all-blank blocks are skipped, `oof_folds_ok` is written on every row, and coverage reports `OOF_ALL_NAN` |
| 6 | The merge step built one DataFrame of ~100 million rows and would have run out of memory **after** the multi-day run finished | Streams in chunks, appends, and accumulates coverage per file; `--parquet` option added |
| 7 | Threshold noise cut the raw label at ±1.0, and every hERG value clears it — a constant column, so question B was undefined there | **Retired with the condition**: threshold was deleted in noiseInject 1.0.0, and with it `--threshold-quantile`. The check survives: the runner warns at run time about any condition whose noise scale is constant, and preflight section 4b prints the exact array indices to skip |
| 8 | Heteroscedastic and value-proportional rank molecules **identically** (Spearman 1.000) — a third of the run was duplicate information | **Retired, because it was the finding that deleted them.** All four value-keyed strategies are gone; preflight section 4c now checks the opposite property, that every remaining condition delivers the amount it was asked for |
| 9 | The neural models early-stopped against **clean** validation labels, so they were explicitly selected not to fit the injected noise — and only the neural half of the roster | Validation now carries the same condition at the same level, drawn from an independent generator so the training corruption is untouched. `--no-noise-validation` restores the old behaviour |

Plus one found while testing the job scripts: adding `set -u` made the unguarded
`$CONDA_PREFIX` reference fatal, which would have killed **every task in under
a second**, before python was ever reached. And a dangling line-continuation that
`bash -n` accepts but which splits the command in two.

### Earlier bugs, also fixed

1. **The uncertainty files only ever contained one fold.** One frame was
   appended per fold, but the writer's filename had no fold in it, so the file
   was rewritten five times and only the last fold survived. It also
   `groupby`-averaged over `sample_idx`, which is a *within-fold* position — so
   it was averaging different molecules together. Every existing
   `*_uncertainty_values.csv` is one fold of five.
2. **`--sigmas` was silently ignored for hERG.** The hERG call omitted
   `sigma_levels`, so it always ran all 11 regardless of the flag.
3. **The cached hERG file killed the run.** The loader hard-coded a `pKi`
   column; the cache here has `pChEMBL`, raising `KeyError`. Now accepts either.

### Verified before shipping

- Patched injector reproduces the old one **exactly** — 336/336 checks.
- Patched pipeline reproduces pre-patch predictions **exactly** (control:
  pre-patch code compared against itself gives the same answer, proving a
  ~2e-15 wobble seen with `n_jobs>1` is the forest library's parallel
  summation, not this change).
- Recorded noise reconstructs the corrupted label exactly, on every condition.
- Level 0 records exactly zero noise — a **true** negative control. The old
  pipeline's `injected_noise` was a regression residual that was non-zero at
  level 0, which is why its zero-noise control showed a *stronger* signal than
  the real noise levels.
- Test-side noise scale uses the **training** distribution's cut-points.
- `noise_pattern` is identical at every level including 0, and non-degenerate
  for the conditions that are supposed to have a pattern.
- ⚠️ The three bullets above were verified against the **six deleted
  strategies**, before the redesign. The properties are the injector's, not the
  strategies', and `scripts/crosscheck_injectors.py` re-establishes them on the
  new conditions across 342 checks — but read them as inherited, not as
  re-measured here.
- Out-of-fold error exceeds in-sample error, i.e. we are not measuring memorisation.
- On synthetic data the analysis recovers the expected answers: signal where the
  noise is concentrated on some molecules, and **nothing** for the Gaussian
  control.
- All nine fixes have their own regression test
  (`tests/smoke/smoke_nine_fixes.py`), including that a scaffold never spans the
  fit/score boundary, that a truncated out-of-fold pass is reported, and that
  re-flushing a fold neither duplicates nor overwrites.
- The generated job scripts were **executed** with a stubbed python: hyphenated
  representation names survive quoting, and all three guards fire (missing
  partition, index out of range, unset `CONDA_PREFIX`). The array dispatch is
  re-checked for **every** index by
  `scripts/test_uncertainty_job_scripts.py`, which also runs the command line
  each script emits through the runner's own argument parser.

---

## 1. Push the code (local)

Three repos changed. All three must be on the server.

```bash
cd ~/repos/NoiseInject
git add noiseInject/core.py
git commit -m "Expose per-molecule noise scale and the noise actually injected

noise_scale() returns sigma_i without drawing (so it can be computed for
molecules that are never corrupted); inject_verbose() returns the epsilon it
drew. Both are needed to ask whether predicted uncertainty tracks corruption.
Verified bit-identical to the previous implementation."
git push

cd ~/repos/KIRBy
git add tests/alternative_data_noise_robustness.py tests/smoke/smoke_uncertainty_patch.py
git commit -m "Record injected noise; save uncertainty for every condition and fold

- runners return the true injected epsilon, the per-molecule noise scale for
  train and test, and optional out-of-fold training predictions
- uncertainty saved for every condition, not just Gaussian, with
  split/noise_type/fold/noise_scale/injected_noise columns
- fix: the writer kept only the LAST fold and averaged across a within-fold
  index, so every existing uncertainty file is one fold of five
- fix: --sigmas was ignored for hERG
- fix: hERG loader hard-coded a pKi column the cached file does not have
- flags: --conditions, --unc-conditions, --oof-folds"
git push

cd ~/repos/qsar_qm_models
git add slurm_scripts_uncertainty_rerun
git commit -m "Add uncertainty re-run job arrays, preflight and merge"
git push
```

## 2. Update the server

**The three commits are on DIFFERENT branches. A bare `git pull` will pull the wrong
one.** Check out explicitly:

| Repo | Branch | Commit |
|---|---|---|
| `NoiseInject` | `main` | **1.0.0 or newer** — the redesigned injector. Preflight section 2 refuses anything older by name |
| `KIRBy` | `similarity-metrics-study` | must carry `--conditions`; the job scripts refuse a checkout that does not |
| `qsar_qm_models` | `additional_reps` | whatever is current |

> The three commit hashes that used to be pinned here (`42b5fac`, `00166dd`, `9d7db67`) all
> predate the noise redesign of 2026-08-26. Checking them out would put the old injector and the
> old six strategies back. Pull the branch tips and let the preflight and the job scripts' own
> guards say whether they are new enough — a hash in a document goes stale silently, a guard does
> not.

```bash
ssh gateway.arc.ox.ac.uk
ssh arc-login

cd /data/stat-cadd/scat9264/NoiseInject
git fetch origin && git checkout main && git pull --ff-only origin main

cd /data/stat-cadd/scat9264/KIRBy
git fetch origin && git checkout similarity-metrics-study \
  && git pull --ff-only origin similarity-metrics-study

cd /data/stat-cadd/scat9264/qsar_qm_models
git fetch origin && git checkout additional_reps \
  && git pull --ff-only origin additional_reps
```

Confirm you have the right commits before going further:

```bash
cd /data/stat-cadd/scat9264/NoiseInject      && git log --oneline -1   # 42b5fac
cd /data/stat-cadd/scat9264/KIRBy            && git log --oneline -1   # 00166dd
cd /data/stat-cadd/scat9264/qsar_qm_models   && git log --oneline -1   # 9d7db67
```

> If `git checkout` refuses because of local modifications, `git stash` first —
> do NOT force. The KIRBy and qsar checkouts on the cluster may carry local edits.

**`NoiseInject` must be installed editable from the checkout you just pulled**, or the
patch has no effect and every task silently runs the old injector:

**micromamba has never worked on this cluster.** Activate through `setup.sh`, which is what the
job scripts do:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models && . setup.sh   # activates env_test via conda
python -c "import noiseInject, inspect; print(inspect.getfile(noiseInject))"
python -c "from noiseInject import CONDITIONS; print(sorted(CONDITIONS))"
# must list gaussian, grouped_wider, grouped_shifted, censoring and the three
# depth conditions. If it lists legacy/quantile/threshold/hetero/valprop, that is
# the pre-1.0.0 injector, and the job scripts will refuse to run against it:
pip install --no-deps -e /data/stat-cadd/scat9264/NoiseInject
```

> There are two KIRBy checkouts on the cluster — `/data/stat-cadd/…/KIRBy` (what these scripts
> use) and `/data/stat-ecr/…/KIRBy` (what 125 of KIRBy's own 127 job scripts use, after the move
> on 2026-05-07 when stat-cadd hit 99.9% of its quota). **The generated scripts no longer take
> this on trust**: each one checks the directory exists and that the runner in it has
> `--conditions`, and exits 2 naming the other checkout if not. If stat-ecr is the live one,
> regenerate with `python generate_scripts.py --kirby-dir /data/stat-ecr/scat9264/KIRBy` rather
> than editing the file.

## 3. Regenerate the scripts, then preflight — do not skip either

The `.sh` files in this directory are generated. Regenerate them so they carry the settled
conditions and the current command line; the checked-in copies are only a record of the last run.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty_rerun
python generate_scripts.py            # add --kirby-dir if stat-ecr is the live checkout
```

On a laptop, prove they will parse before they are copied anywhere — this runs each generated
command line through the runner's own argument parser:

```bash
python ~/repos/qsar_qm_models/scripts/test_uncertainty_job_scripts.py --kirby-dir ~/repos/KIRBy
```

Two of the preflight's checks caught real failures locally.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty_rerun
mkdir -p logs
bash preflight.sh 2>&1 | tee logs/preflight.log
```

It must end with `ALL PREFLIGHT CHECKS PASSED`. In particular it will tell you
whether **QRF is usable** — locally `quantile_forest` and `scikit-learn` are
incompatible and every QRF fit raises `Invalid parameter 'monotonic_cst'`. If
that reproduces on the cluster, fix the environment before submitting `unc_qrf.sh`
(`pip install -U quantile-forest`), or submit the other six scripts first.

## 4. Choose account and partition

```bash
cd /data/stat-cadd/scat9264/KIRBy
bash tests/slurm_scripts/where_to_submit.sh          # full diagnostic — read §2, §3, §5
bash tests/slurm_scripts/where_to_submit.sh --emit   # prints: <account> <partition>
```

Bill to whichever account has the higher fair share. Use `medium`. Then:

```bash
read -r ACCT PART < <(bash /data/stat-cadd/scat9264/KIRBy/tests/slurm_scripts/where_to_submit.sh --emit)
echo "account=$ACCT partition=$PART"
```

## 5. Submit — tier 1 first

Tier 1 is the three models the paper's uncertainty argument rests on. Get these
running and confirm they work before committing the rest of the queue.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty_rerun

sbatch --account=$ACCT --partition=$PART --array=0-59%6 unc_qrf.sh
sbatch --account=$ACCT --partition=$PART --array=0-59%6 unc_ngboost.sh
sbatch --account=$ACCT --partition=$PART --array=0-59%6 unc_gp.sh
```

> `0-59` is 3 datasets × 4 representations × 5 conditions. Each script's own header prints the
> range it was generated for — use that, not this line, if you regenerated with a different
> condition or representation set.

`%6` caps each array at 6 concurrent tasks — 18 running at once across tier 1.
Raise it if `where_to_submit.sh` §3 shows idle capacity, lower it if §5 shows a
backlog.

**Before committing tier 2, let one task finish and check it:**

```bash
squeue -u $USER -o "%.12i %.22j %.8T %.10M %.10L %R"
ls results/../../../KIRBy/tests/results/uncertainty_rerun/    # task dirs appear as they start
tail -40 logs/unc_ngboost_*_0.out
```

## 6. Submit tier 2

```bash
sbatch --account=$ACCT --partition=$PART --array=0-59%4 unc_bnn_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-59%4 unc_vbll_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-59%4 unc_mlp_bnn_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-59%4 unc_mlp_vbll_full.sh
```

## 7. Monitor

```bash
squeue -u $USER -o "%.12i %.22j %.8T %.10M %.10L %.6D %R" | head -40
squeue -h -u $USER -t PENDING -o "%r" | sort | uniq -c        # why anything is stuck
sacct -X -S today --format=JobID%18,JobName%22,State,Elapsed,MaxRSS | grep unc_
sacct -X -S today --format=JobID,State | grep -c COMPLETED
grep -l "exit=[^0]" logs/*.out          # tasks that failed
```

Resubmit only the failed indices:

```bash
sbatch --account=$ACCT --partition=$PART --array=3,17,40 unc_ngboost.sh
```

## 8. Merge and bring the results home

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty_rerun
python merge_results.py --root /data/stat-cadd/scat9264/KIRBy/tests/results/uncertainty_rerun \
    --kirby-dir /data/stat-cadd/scat9264/KIRBy
```

**Read `coverage.csv` first** — it lists all 420 expected cells and marks each
`OK` / `MISSING` / `NO_OOF` / `OOF_ALL_NAN` / `TRUNCATED_OOF` / `PARTIAL_FOLDS` /
`PARTIAL_LEVELS`. Do not analyse anything until you know what is missing.

It takes the condition list from the generated `unc_*.sh` and the expected number of noise levels
from the runner itself, so a run submitted with a different condition set is still checked against
what it actually ran. Without `--kirby-dir` it reports each cell's level count without judging it.

From your laptop:

```bash
scp -r 'gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/KIRBy/tests/results/uncertainty_rerun/_merged' \
    ~/repos/qsar_qm_models/results/uncertainty_rerun_merged
```

---

## How to analyse the output — the plan, so nobody manufactures a result

**Question A — do the corrupted molecules come back as the uncertain ones?**

The statistic is **not** the raw correlation between uncertainty and injected
noise. Under cross-fitting the model scoring a molecule never saw that molecule's
noise, so it cannot know the individual draw — under Gaussian noise, where every
molecule has the same noise *scale*, that correlation is zero by construction and
the design forbids any other answer.

What is informative:
- the **cross-fitted residual**, `|y_true + injected_noise − y_pred|`, which does
  track the injected noise (all three columns are in the output);
- whether **uncertainty adds anything on top of it** — is the residual divided by
  the uncertainty a better detector of a corrupted label than the residual alone;
- uncertainty against the noisy *region* — but note `noise_scale` is exactly
  the level times `noise_pattern`, so at a fixed noise level the two rank
  molecules **identically** (measured: Spearman 1.000). It is question B's
  statistic under another name, and it needs the same level-0 subtraction. Do
  not report it as a separate unsubtracted question-A result.

Report a permutation null with every number, and **state it precisely** — the
naive version fires on clean data. The residual is `(y_true − y_pred) + epsilon`,
so it *contains* the epsilon you are correlating against. Permuting the epsilon
column while leaving the residual as computed therefore compares a residual that
contains the real epsilon against a shuffled one, and declares a leak on a
simulation that has none (measured: observed rho +0.62, that null's 95% band
[−0.04, +0.04]).

The correct null: permute `injected_noise` within
(dataset, model, rep, condition, split, fold, level) **and recompute the residual
from the permuted epsilon**, so observed and null both carry the same additive
term (measured on the same simulation: null mean +0.60, band [+0.58, +0.62],
observed +0.62 sits inside — correctly showing no leakage). This is built:
`scripts/uncertainty_stats.py`, `permutation_null`, 18 tests.

**Question B — does the model learn where the data is unreliable?**

Use `noise_pattern`, and always subtract the level-0 baseline (see above). Only
`grouped_wider`, `censoring` and `outlier_p10` can be asked. Three further
guards:

- **A sham ceiling.** Recompute the pattern from the model's *predicted* label
  instead of the true one. If uncertainty correlates with that just as strongly,
  the model is tracking its own prediction, not the noise. Computable from the
  saved `y_pred`, no extra runs. ⚠️ **It is degenerate for the grouped
  conditions** — a prediction does not change a molecule's scaffold, so the
  pattern recomputed from `y_pred` is the same pattern (`RERUN_PLAN.md` §3.1d).
  It is a real control for `censoring` and `outlier_p10` only.
- **`gaussian` is not the control for B, and neither is `grouped_shifted`.**
  Their noise scale is constant, so the correlation is undefined, not zero — a
  control has to produce a number. The control for B is level 0 *within the same
  condition*. Keep both for question A and the leakage check.
- **A foreign-pattern placebo.** For a run under condition *c*, compute the same
  baselined effect against every *other* condition's pattern. The model was never
  exposed to those, so a real effect should be largest against its own pattern.
  Rows join across tasks on (dataset, fold, sample_idx). Three patterned
  conditions give six such comparisons — enough to catch a model that correlates
  with everything, not enough to be a spectrum.

**Print the cross-condition pattern correlation before any "the conditions
agree" claim.** It is a pure function of the labels and costs nothing.
`grouped_wider` is keyed to the scaffold, `censoring` to the label and
`outlier_p10` to the draw, so all three should be close to independent — but
"should be" on these three datasets is a measurement, not an assumption, and it
differs per dataset. The old six
strategies were the cautionary case: heteroscedastic and value-proportional
ranked molecules identically at Spearman 1.000, which is why they were deleted.

**The sham ceiling is precomputed.** `noise_pattern_pred` is the pattern computed
from the model's *predicted* label, written on every row. It has to be computed
inside the run because censoring needs the training distribution, which is not
recoverable from the output once `--oof-outer-folds 1` leaves folds 1–4 without
training rows.

## Cost and the one thing to decide

Each task is 6 noise levels (7 for censoring) × 5 scaffold folds × (1 + 5) fits
= **180 model fits**, against 30 without cross-fitting. That is 45% less per task
than the old eleven-level ladder, against 25% more tasks. The `--time` requests were set against that ladder and
are left as they are: a task that finishes early costs nothing, one killed at the
wall costs the whole task, and nothing here has been timed on the cluster.

Two levers if the queue is tight, in order of how little they cost you:

| Lever | Saving | What you lose |
|---|---|---|
| `--oof-outer-folds 1` | ~3x on each task | the spread of question A across folds. **Test-side data for all five folds is unaffected** — those folds are trained regardless |
| `--oof-folds 3` | ~40% on each task | each molecule is scored by a model trained on 67% of the data rather than 80% |

**`--main-grid-only` is the third lever, and it is the honest one**: it drops
`outlier_p10` for 20% of the run, leaving two conditions that can answer question
B instead of three. Dropping anything else is not a lever. `gaussian` and
`grouped_shifted` are question A's; `grouped_wider` and `censoring` are the only
other two that can answer question B. `--drop-conditions` exists, and the
generator says out loud which question a drop removes.

The five scaffold folds themselves are **not** a lever: they are trained whatever
you do, because they produce the R² numbers. Saving their uncertainty is free.

**The dose question that used to be open here is closed.** Every condition now
solves for the internal scale that delivers the level it was asked for, so one
unit of level means the same amount of corruption in all of them — measured
spread across conditions at one setting: **0.40% in Python** (`NOISE_DESIGN.md`;
gate 2, `scripts/crosscheck_injectors.py`). The axis will not move under these
runs.
