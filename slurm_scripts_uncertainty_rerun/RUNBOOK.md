# Uncertainty re-run — runbook

**What this produces:** the data needed to answer both uncertainty questions.

- **(A) Do the corrupted molecules come back as the uncertain ones?**
  Measured on **training** molecules, scored **out-of-fold** so no molecule is
  judged by a model that fitted its own corrupted label.
- **(B) Does the model learn *where* the data is unreliable?**
  Measured on **test** molecules against the noise scale their region receives.
  Five of the six strategies corrupt some molecules far more than others, so
  there is a pattern to learn. Gaussian (`legacy`) hits everything equally and
  is the **control** — if a signal appears there too, the analysis is wrong.

Both come out of one set of jobs. Question B costs nothing extra; the only added
compute is the out-of-fold folds, and only for the seven models that emit a
per-molecule uncertainty.

### The confound in question B, and the control for it

The noise scale a molecule receives is a **deterministic function of its label**
(value-proportional scales with |y|, quantile and threshold cut on y, and so on).
A model's predicted uncertainty may already track the label for reasons that have
nothing to do with noise — extreme molecules are simply harder. So a raw
correlation between uncertainty and noise scale would be partly manufactured.

Every row therefore carries **two** noise columns:

| column | meaning |
|---|---|
| `noise_scale` | the noise scale actually applied at this σ. Exactly **0** at σ = 0. |
| `noise_pattern` | the *shape* — which molecules the strategy hits hardest — taken at a fixed reference level, so it is **the same column at every σ, including σ = 0**. |

The defensible effect for question B is therefore

> ρ(uncertainty at σ, `noise_pattern`) **minus** ρ(uncertainty at σ = 0, `noise_pattern`)

The σ = 0 model was trained on completely clean labels but saw the same label
distribution, so its correlation is exactly the confound. Report the difference,
not the raw number.

`legacy` (Gaussian) is the second control: it gives every molecule the same noise
scale, so `noise_pattern` is flat and the correlation is undefined by
construction. If a Gaussian arm ever shows a signal, the analysis is wrong.

**Scope:** 3 datasets × 7 models × 4 representations × 6 strategies × 11 noise
levels × 5 scaffold folds. 7 array scripts, 72 tasks each, **504 tasks**.

---

## 0. What changed in the code

| Repo | File | Change |
|---|---|---|
| `NoiseInject` | `noiseInject/core.py` | New `noise_scale()` (per-molecule noise scale, draws no randomness) and `inject_verbose()` (returns the noise it actually drew). Strategies refactored onto one shared scale function. **Verified bit-identical to the old code on 336 checks.** |
| `KIRBy` | `tests/alternative_data_noise_robustness.py` | Runners now return an `extras` dict carrying the true injected noise, the per-molecule noise scale for train *and* test, and out-of-fold training predictions. Uncertainty saved for **all six** strategies, with `split`/`strategy`/`fold`/`noise_scale`/`injected_noise` columns. New flags `--strategies`, `--unc-strategies`, `--oof-folds`. Three bugs fixed — see below. |
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
| 7 | Threshold noise cuts the raw label at ±1.0, and every hERG value clears it — a constant column, so question B is undefined there | Run-time warning naming every constant arm; preflight prints the exact array indices to skip; `--threshold-quantile` makes the cut-points quantiles of the labels instead |
| 8 | Heteroscedastic and value-proportional rank molecules **identically** (Spearman 1.000) — a third of the run was duplicate information for these questions | Documented as one arm; `--drop-strategies hetero` reclaims 84 tasks with no loss of rank information |
| 9 | The neural models early-stopped against **clean** validation labels, so they were explicitly selected not to fit the injected noise — and only the neural half of the roster | Validation now carries the same strategy at the same level, drawn from an independent generator so the training corruption is untouched. `--no-noise-validation` restores the old behaviour |

Plus one found while testing the job scripts: adding `set -u` made the unguarded
`$CONDA_PREFIX` reference fatal, which would have killed **all 504 tasks in under
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
- Patched pipeline reproduces pre-patch predictions **exactly** on all six
  strategies (control: pre-patch code compared against itself gives the same
  answer, proving a ~2e-15 wobble seen with `n_jobs>1` is the forest library's
  parallel summation, not this change).
- Recorded noise reconstructs the corrupted label exactly, for all six strategies.
- σ = 0 records exactly zero noise — a **true** negative control. The old
  pipeline's `injected_noise` was a regression residual that was non-zero at
  σ = 0, which is why its zero-noise control showed a *stronger* signal than
  the real noise levels.
- Test-side noise scale uses the **training** distribution's cut-points.
- `noise_pattern` is identical at every σ including 0, and non-degenerate for all
  five uneven strategies (flat for Gaussian, as it must be).
- Out-of-fold error exceeds in-sample error, i.e. we are not measuring memorisation.
- On synthetic data the analysis recovers the expected answers: signal for
  concentrated strategies, and **nothing** for the Gaussian control.
- All nine fixes have their own regression test
  (`tests/smoke/smoke_nine_fixes.py`), including that a scaffold never spans the
  fit/score boundary, that a truncated out-of-fold pass is reported, and that
  re-flushing a fold neither duplicates nor overwrites.
- The generated job scripts were **executed** with a stubbed python: the array
  dispatch is right at indices 0, 5, 24, 52 and 71; hyphenated representation
  names survive quoting; and all three guards fire (missing partition, index out
  of range, unset `CONDA_PREFIX`).

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
git commit -m "Record injected noise; save uncertainty for all strategies and folds

- runners return the true injected epsilon, the per-molecule noise scale for
  train and test, and optional out-of-fold training predictions
- uncertainty saved for all six strategies, not just Gaussian, with
  split/strategy/fold/noise_scale/injected_noise columns
- fix: the writer kept only the LAST fold and averaged across a within-fold
  index, so every existing uncertainty file is one fold of five
- fix: --sigmas was ignored for hERG
- fix: hERG loader hard-coded a pKi column the cached file does not have
- new flags: --strategies, --unc-strategies, --oof-folds"
git push

cd ~/repos/qsar_qm_models
git add slurm_scripts_uncertainty_rerun
git commit -m "Add uncertainty re-run job arrays, preflight and merge"
git push
```

## 2. Update the server

```bash
ssh gateway.arc.ox.ac.uk
ssh arc-login

cd /data/stat-cadd/scat9264/NoiseInject   && git pull --ff-only
cd /data/stat-cadd/scat9264/KIRBy         && git pull --ff-only
cd /data/stat-cadd/scat9264/qsar_qm_models && git pull --ff-only
```

**If `NoiseInject` is not installed editable from that path the patch will not
take effect.** Check, and reinstall if needed:

```bash
export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate env_test
python -c "import noiseInject, inspect; print(inspect.getfile(noiseInject))"
python -c "from noiseInject import NoiseInjectorRegression as N; print(hasattr(N, 'inject_verbose'))"
# must print True. If False:
pip install --no-deps -e /data/stat-cadd/scat9264/NoiseInject
```

> There are two KIRBy checkouts on the cluster — `/data/stat-cadd/…/KIRBy`
> (what the qsar job scripts use) and `/data/stat-ecr/…/KIRBy` (what KIRBy's own
> scripts use). These job scripts use **stat-cadd**. Confirm that is the live one
> before submitting; if not, change `KIRBY_DIR` in `generate_scripts.py` and
> regenerate.

## 3. Preflight — do not skip this

Two of its checks caught real failures locally.

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

sbatch --account=$ACCT --partition=$PART --array=0-71%6 unc_qrf.sh
sbatch --account=$ACCT --partition=$PART --array=0-71%6 unc_ngboost.sh
sbatch --account=$ACCT --partition=$PART --array=0-71%6 unc_gp.sh
```

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
sbatch --account=$ACCT --partition=$PART --array=0-71%4 unc_bnn_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-71%4 unc_vbll_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-71%4 unc_mlp_bnn_full.sh
sbatch --account=$ACCT --partition=$PART --array=0-71%4 unc_mlp_vbll_full.sh
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
python merge_results.py --root /data/stat-cadd/scat9264/KIRBy/tests/results/uncertainty_rerun
```

**Read `coverage.csv` first** — it lists all 504 expected cells and marks each
`OK` / `MISSING` / `NO_OOF` / `PARTIAL_FOLDS`. Do not analyse anything until you
know what is missing.

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
- uncertainty against `noise_scale` (the noisy *region*), which the model can
  learn from other molecules — unlike the individual draw.

Report a permutation null with every number: shuffle `injected_noise` within
(fold, sigma, strategy) a few hundred times. Under Gaussian the observed value
must sit inside that null; if it does not, the cross-fitting is leaking.

**Question B — does the model learn where the data is unreliable?**

Use `noise_pattern`, and always subtract the σ = 0 baseline (see above). Two
further guards:

- **A sham ceiling.** Recompute the pattern from the model's *predicted* label
  instead of the true one. If uncertainty correlates with that just as strongly,
  the model is tracking its own prediction, not the noise. Computable from the
  saved `y_pred`, no extra runs.
- **`legacy` is not the control for B.** Its noise scale is constant, so the
  correlation is undefined, not zero — a control has to produce a number. The
  control for B is σ = 0 *within the same strategy*. Keep `legacy` for question A
  and for the leakage check.

**Never treat heteroscedastic and value-proportional as independent** — they rank
molecules identically.

## Cost and the one thing to decide

Each task is 11 noise levels × 5 scaffold folds × (1 + 5) fits = **330 model
fits**, against 55 without cross-fitting. The `--time` requests (24 h for QRF,
36 h for the rest) are deliberately generous — nothing here has been timed on
the cluster, and requesting too little wastes a whole task.

Three levers if the queue is tight, in order of how little they cost you:

| Lever | Saving | What you lose |
|---|---|---|
| `--drop-strategies hetero` | 84 tasks (17%) | nothing for these questions — it ranks molecules identically to value-proportional |
| `--oof-outer-folds 1` | ~3x on each task | the spread of question A across folds. **Test-side data for all five folds is unaffected** — those folds are trained regardless |
| `--oof-folds 3` | ~40% on each task | each molecule is scored by a model trained on 67% of the data rather than 80% |

The five scaffold folds themselves are **not** a lever: they are trained whatever
you do, because they produce the R² numbers. Saving their uncertainty is free.

**Open question that affects nothing here but everything later:** if the noise
dose is later renormalised so one unit of σ means the same corruption in every
strategy, these runs would be measured at different absolute noise levels. Both
analyses are rank-based, so the *conclusions* transfer — but the σ values on the
axis would move. Worth deciding before this is written up.
