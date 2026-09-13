# Handoff — the code defects found 2026-09-12

Read `CLAUDE.md` first. Branch `additional_reps`. Code reaches the cluster only when the author
runs `bash scripts/pull_safely.sh` against a commit that is already pushed. Write answers into
`RERUN_PLAN.md` §13.27.

These nine came out of a second code pass over the paper's Methods on 2026-09-12, which re-checked
302 claims against the pipelines and corrected 74. Most of the 74 were writing problems and are
handled in `PAPER_REVISION_GUIDE_FINAL.md`. **These nine are not. They are things the code or the
run configuration is doing wrong** — not results that have failed to arrive — and every one is
verified at the line numbers given.

They are ordered by what a wrong result costs. **1 and 2 change numbers. 3, 4 and 5 change which
model was fitted. 6 to 9 are provenance — nothing is wrong with the runs, but a claim in the paper
cannot be backed.**

Three things to be clear about before starting.

**None of these nine is a claim that a result is missing. Work is in the queue right now.** The
author's `squeue` on 2026-09-12:

```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 13043774_[2-26%6]      long unc_gp_h scat9264 PD       0:00      1 (Priority)
 13043781_[0-35%6]      long unc_gp_h scat9264 PD       0:00      1 (Priority)
 13043780_[0-35%6]      long unc_mlp_ scat9264 PD       0:00      1 (Priority)
 13043778_[0-35%6]      long unc_vbll scat9264 PD       0:00      1 (Priority)
 13043776_[0-35%6]      long unc_ngbo scat9264 PD       0:00      1 (Priority)
13099373_[0-1,4,6-      long qm92_mlp scat9264 PD       0:00      1 (Priority)
 13111529_[0-17%4]      long val_mlp- scat9264 PD       0:00      1 (Priority)
       12986318_34      long qm92_ngb scat9264  R   22:48:00      1 arc-c149
       12986318_28      long qm92_ngb scat9264  R 1-12:23:40      1 arc-c244
       12986318_22      long qm92_ngb scat9264  R 2-02:57:57      1 arc-c049
12971614_[14-17%4]    medium qm90_mlp scat9264 PD       0:00      1 (Priority)
```

Every pending array says `(Priority)`, which is queue position and nothing else — not a bad resource
request, not an unmet dependency. Six arrays and three running tasks were outstanding when this list
was written, so **`check_runs_landed.py` reporting MISSING for any of them means "has not run yet",
not "was never submitted".** Run `squeue` before you resubmit anything, every time.

The job names decode from the three generators — `slurm_scripts_uncertainty_rerun` writes
`--job-name=unc_{slug}` (`:390`), `slurm_scripts_validation_rerun` writes `val_{name}` (`:546`), and
`slurm_scripts_qm9_rerun` writes `qm9{stage}_{slug}` (`:798`):

| Prefix | What it is | Where it writes |
|---|---|---|
| `unc_*` | the uncertainty runs | `<KIRBy>/tests/results/uncertainty_rerun/` |
| `val_*` | the assay robustness grid | `<KIRBy>/results/validation_rerun/` |
| `qm90_*` | the QM9 screen | `results/` in this checkout |
| `qm92_*` | the QM9 deep run | `results/` in this checkout |

**Not on this list, because it is correct.** Five of the seven QM9 conditions never run noise level
zero, and `copy_zero_rows.py` refuses to copy a clean *uncertainty* row into them. That is the
author's decision of 2026-08-28, with the reasoning written out at
`slurm_scripts_qm9_rerun/generate_scripts.py:468-503`: of the three conditions with a shape,
`grouped_wider` is keyed to the scaffold group that the out-of-fold pass splits on and
`outlier_p10` picks at random, so both are structural nulls, and censoring is the only condition
whose clean level buys anything. **Do not "fix" it.**

**Do not re-fetch hERG.** See item 8.

---

## 1. QM9's heteroscedastic Gaussian process never gets its lengthscale from the data

**What is wrong.** `init_rbf_lengthscale` is defined at `models/models.py:2672` and called at
exactly one place, `:2744`, inside `fit_gp_with_fallback`. The heteroscedastic Gaussian process
does not go through that function: `_fit_het_gp`, from `models/models.py:8449`, builds its own
`HeteroscedasticGPModel`, its own `GaussianLikelihood` and its own Adam optimiser, and never calls
it. So on QM9 that model starts at gpytorch's default lengthscale.

KIRBy does it at both sites — `alternative_data_noise_robustness.py:1602` for the plain process and
`:1762` for the heteroscedastic one — so the two pipelines differ on the one model where it matters
most.

**Why it costs numbers.** The default lengthscale sits near 1 while real pairwise distances on these
representations run far above that, which is the condition that makes a process return its prior and
predict one flat value for every molecule. That is the defect fixed in `574a3f0` on 2026-08-26 and
it is still live on this path. `GP-Hetero` is the model
`slurm_scripts_uncertainty_rerun/generate_scripts.py:186` calls "the cleanest separation measured in
the roster" and "the only kernel model that can be asked the per-molecule question".

**Do.** Call `init_rbf_lengthscale` inside `_fit_het_gp` on `x_fit`, before the optimisation loop,
so the inner out-of-fold folds get it too — `_fit_het_gp` is deliberately one function for exactly
that reason.

Then ask which of three states the QM9 `heteroscedastic_gp` tasks are in, because the work differs:
**not yet submitted** — fix, push, `pull_safely.sh`, then submit, and nothing is wasted;
**queued** — the same, and get the fix onto the cluster before those tasks dispatch, because a
pending task picks up whatever the code says when it starts; **already completed** — those rows were
fitted at gpytorch's default lengthscale, so check `gp_collapsed` and the spread of the predictions
on each and resubmit what collapsed. Say how many rows each state holds.

Note that the `unc_*` arrays in the queue above are the uncertainty runs, which go through KIRBy, and
KIRBy already initialises at both its sites. This defect is QM9's `heteroscedastic_gp` alone.

## 2. Two different lists of which models the uncertainty work runs

**What is wrong.** `uncertainty_pairs.json` names six models. `MODELS` in
`slurm_scripts_uncertainty_rerun/generate_scripts.py:116-186` names seven — the extra one is
`GP-Hetero`, added there on 2026-09-07. Nothing in that generator reads `uncertainty_pairs.json`;
grep it and you get no hits.

**Why it matters, in the file's own words.** `uncertainty_pairs.json` exists because
"Both pipelines answer the same uncertainty questions on the same pairs, so a table can put QM9
beside logD, Caco-2 and hERG row for row. A second list anywhere means the two halves stop being
comparable, which is what happened between 2026-08-28 and 2026-08-29." There is a second list now.
The effect is that the assay datasets can ask GP-Hetero the per-molecule question and QM9 cannot,
because QM9's out-of-fold pass only fires on pairs from that file.

**This is live, not hypothetical.** Two `unc_gp_h` arrays are queued right now — 13043774 and
13043781 — so the assay side is about to produce GP-Hetero uncertainty rows that QM9 has no
counterpart for. Do not read their absence on the QM9 side as a missing run: QM9 has no uncertainty
submission of its own, its uncertainty rows fall out of the main grid, and the out-of-fold pass only
fires on pairs named in `uncertainty_pairs.json`.

**Do.** Put the question to the author: does `GP-Hetero` join the QM9 out-of-fold list, or come out
of the assay one? Then make the assay generator READ `uncertainty_pairs.json` instead of holding its
own copy, so this cannot recur. If GP-Hetero goes in, that is a resubmission of those QM9 tasks —
size it before submitting.

## 3. The tuned setting was ranked without Avalon and then shipped to Avalon

**What is wrong.** `scripts/write_chosen_settings.py:71` ranks over
`REPS = ['pdv', 'chemberta', 'ecfp4', 'mhggnn', 'sns']`, under a comment at `:69-70` reading
"AVALON IS OUT OF THIS STUDY (author, 2026-09-01). Its columns are not shown and no run collects
it." `scripts/ship_tuned_settings.py:58` then uses
`REPS = ['ecfp4', 'pdv', 'mhggnn', 'avalon', 'chemberta', 'sns']` and writes
`master[key] = {rep: setting for rep in REPS}` at `:84`.

**The comment is false about the study.** `ALL_REPS` at
`slurm_scripts_qm9_rerun/generate_scripts.py:112` and at
`slurm_scripts_validation_rerun/generate_scripts.py:49` both contain Avalon, and `CLAUDE.md` lists
it as one of the six. Every run collects it.

**Consequence.** `results/master_tuned_hyperparameters.json` carries an `avalon` entry for all four
Bayesian networks, holding a setting chosen by a ranking Avalon never entered.

**Do.** Ask the author what the 2026-09-01 decision actually was. Then either re-rank including
Avalon, or stop shipping to Avalon and let it run at the shared default. Correct the comment either
way — it is the only record of that decision and it currently contradicts both generators.

## 4. The two base networks are tuned over different parameter sets

**What is wrong.** `models/models.py:3458` builds NN-$\alpha$'s optimiser with
`lr=NEURAL_DEFAULTS['training']['lr']` — the shared default, unconditionally.
`models/models.py:4193` builds NN-$\beta$'s with `lr=params['lr']`, which comes from the tuned
dictionary. Dropout is the same story: `models.py:1050` hard-codes `p=0.2` on the $\alpha$ path
while `models.py:4066` reads it from the tuned dictionary on the $\beta$ path.

**Consequence.** The shipped settings run BNN-$\beta$ at learning rate 4.29e-3 and dropout 0.379 and
VBLL-$\beta$ at 1.19e-3 and 0.357, while BNN-$\alpha$ and VBLL-$\alpha$ cannot leave 1e-3 and 0.2
whatever the sweep found for them. An $\alpha$-versus-$\beta$ comparison is then partly a comparison
of how much tuning each was allowed.

**Do.** Make both paths read the same keys from the same place. Then check whether the $\alpha$
models' tuned entries hold a learning rate or dropout that was never being applied — if they do,
those runs were not at the setting the file says.

## 5. The variance-head and heteroscedastic networks train at defaults beside tuned siblings

**What is wrong.** Nothing hidden — `models/tuning_rosters.py:161-166` states it: the two
variance-head networks "have their own keys, so they read their own entry or none — and no sweep has
ever scored them, so as things stand they train at their defaults. That is a fact about what has
been measured, not a decision to leave them untuned; tuning them needs a sweep that includes
`--loss heteroscedastic`." The same applies to the two heteroscedastic VBLL networks, and
`:170-172` says the heteroscedastic Gaussian process has no tuned path at all.

**Consequence.** Four transformations sit on each base network. Two of them run tuned and two run at
the shared default, so a BNN-versus-MVE comparison on one base network is partly
tuned-versus-untuned.

**Do.** Either run the sweep with `--loss heteroscedastic` so all four are on the same footing, or
get the author's word that the asymmetry stands and it goes in the Methods. Cost the sweep before
proposing it.

## 6. ChemBERTa's collision count has never been produced

`scripts/crosscheck_chemberta.py` gate 4 counts how often two molecules collapse to one embedding.
It has never been run, and the Methods needs the two percentages — QM9 and hERG. Run it, and put the
output where a figure script can read it rather than quoting it into a document.

## 7. The QM9 pool has no generator

`data/valid_qm9_indices.pth` holds 129,428 indices with a maximum of 130,830. Five scripts read it
(`scripts/process_and_train.py:73` and `:1136`, plus `clean_noise.py`, `domain_clustering.py`,
`run_qm_qsar_models.py`, `noise_mitigation.py`). Nothing writes it. It is a binary dated
12 November 2024, and the only account of what the 1,403 missing molecules are is a comment.

The Methods currently says they are the uncharacterised ones plus a set RDKit could not process.
**Do.** Write the script that regenerates the index from PyG's QM9 and prove it returns the same
129,428, or establish that it does not and say what the difference is. Until then that sentence
cannot be written.

## 8. hERG provenance — verify, do NOT regenerate

`fetch_chembl_herg_ki` at `alternative_data_noise_robustness.py:952` returns the cached
`data_cache/chembl_herg_ki.csv` — two columns, SMILES and pChEMBL, 1,415 rows — before reaching the
binding-assay filter at `:1036`, the median collapse at `:1042` or the inter-assay
standard-deviation filter at `:1045`. A live fetch is refused unless `KIRBY_ALLOW_CHEMBL_FETCH=1`.

**The cache-first behaviour is correct and deliberate**, and the refusal message says why: fetching
live pulls whatever release is current today, which is not the dataset any existing result came
from. **A re-fetch would silently change N and invalidate every assay result in the study. Do not
do it on the working path.**

The problem is only that nothing proves which filters produced that file.
`data_cache/chembl_herg_ki.provenance.json` states release 36 and a re-check against 37, and says in
its own text that it was reconstructed on 2026-09-04 rather than written by the fetch.

**Do.** Fetch into a SEPARATE file, with the environment variable set and the output path changed,
and compare it against the cache: how many of the 1,415 survive, what the release actually is, and
whether the standard-deviation filter reproduces the same set. Report the comparison. Do not
overwrite the cache whatever it shows — the answer goes into the Methods sentence, not into the
data.

## 9. Additional file 1 is hand-typed and contradicts the code

`additional_files.tex:38-122` lists hyperparameters for eleven models. The roster is nineteen
configurations. It gives both forests `min_samples_leaf` 1 where `models/model_defaults.py` pins 5
and `max_features` sqrt where it pins 0.3, gives the Gaussian process a Tanimoto kernel and nothing
else, and has no rows at all for any BNN, VBLL or MVE variant, the heteroscedastic Gaussian process,
the epoch cap, the patience or the 100 Monte Carlo passes. `scripts/generate_supp_tables.py` writes
Additional files 2 to 5 and not this one.

The Methods' first sentence points at it.

**Do.** Write the generator, reading `models/model_defaults.py` (SPEC_VERSION 1.7.0) and the tuned
files, and emit the LaTeX. Do not hand-edit the table — that is how it got here. Additional file 12
is the representation-specific SVM kernel table that `paper.tex:197` cites; the claim it supports is
being deleted, so it goes too.

---

## One thing to confirm with the author, because it changes what runs

All nineteen assay scripts pass `--conditions gaussian grouped_wider grouped_shifted`. The
Student-$t$, Laplace and outlier conditions are opt-in on that generator via
`--include-depth-conditions` (`slurm_scripts_validation_rerun/generate_scripts.py:1203-1205`) and
were not asked for, and censoring needs its pairs named. `noise_conditions.json` gives the three
depth-only conditions no `applies_to` scope, so nothing declares that they should run there.

So the assay datasets carry three of the seven conditions, and QM9 carries all seven. **Confirm that
is intended before the Methods says it.** If it is not, it is a submission, not a code fix.

This is about which conditions were *asked for*, not about which have finished: `val_mlp-`
(13111529) was still queued on 2026-09-12, so the assay grid is incomplete on disk for reasons that
have nothing to do with this question. Read the `--conditions` line in the generated `val_*.sh`
scripts, not the results directory.

---

## Every chat owes the same two round trips as before

The rules at the bottom of this file still hold: prove the change took, and prove nothing is
missing. `python scripts/check_runs_landed.py --stage N --verbose`, pointed at the KIRBy checkout for
the laboratory and uncertainty results. No item is finished while its part of that output is not
clean.

**With one addition, from the queue above.** `check_runs_landed.py` compares what is on disk against
what the generators asked for. It cannot see the queue, so anything still pending reads as MISSING.
Pair every run of it with `squeue -u $USER` and separate the two before reporting:

```bash
squeue -u $USER -o "%.20i %.12P %.14j %.2t %.11M %R"
python scripts/check_runs_landed.py --stage 2 --verbose
```

A task that is MISSING **and** in `squeue` is waiting. A task that is MISSING and **not** in `squeue`
needs resubmitting. Saying "34 missing" without that split sends the author to resubmit work that is
already queued, which is worse than saying nothing — it doubles the queue and the second copy wins
the race unpredictably.

---
---

# Handoff — five chats, one thing each (2026-09-07, state in `RERUN_PLAN.md` §13.27)

Read `CLAUDE.md` first.

The author, 2026-09-07: *"All I care about is getting all the results in from the server."*
Figures, the paper and the analysis job are set aside until that is done.

Each chat owns one thing from end to end. That is the point — the previous versions of this
file split each thing across several chats by kind of problem, so no chat could answer a
question about any one of them.

Branch `additional_reps`. Code reaches the cluster only when the author runs
`bash scripts/pull_safely.sh` against a commit already pushed. Write answers into
`RERUN_PLAN.md` section 13.27.

**Read section 13.28 first. It is what the cluster said on 2026-09-07, measured, and it
closes four questions that earlier versions of this file still asked.**

---

## Every chat owes two round trips to the cluster

The author, 2026-09-07: *"By the end of this not only does everything have to be fixed code
wise, every single result has to be queued up correctly."* Fixing the cause is not the job.
The job is the task running.

**The middle one: prove the change took.** After you change a time limit, a memory request or
a generator, and after you resubmit anything, send one block that shows it on the cluster.
Not the command you ran — what the cluster now says. A resubmitted task that fails again in
thirty seconds looks exactly like a resubmitted task that is working, for the first thirty
seconds.

```bash
squeue -u $USER -o "%.12i %.28j %.2t %.11M %.11l %.7m %R" | head -40
sacct -S today -X -n -P --format=JobID,JobName,State,Elapsed,ExitCode | grep -v COMPLETED | head -30
```

**The end one: prove nothing is missing.** This is the only command that answers the author's
question, because it checks what is on disk against what the generators asked for, rather than
against anything anyone typed.

```bash
python scripts/check_runs_landed.py --stage 1 --verbose      # the QM9 main grid
python scripts/check_runs_landed.py --stage 2 --verbose      # the deep run and censoring
python scripts/check_runs_landed.py --stage 0 --verbose      # the screen
```

It exits 0 when everything has landed. Three states, and each needs a different action:
**MISSING** — resubmit that index. **PARTIAL** — the task wrote a file and died part-way
through the noise levels, so squeue showed it finished; resubmit it. **THIN** — it ran but has
fewer replicates than the variance work needs.

For the laboratory and uncertainty results, point it at the KIRBy checkout, because those
jobs change into the `tests` directory and write there. **Confirm the path before trusting an
empty answer** — a wrong directory reports everything missing and looks like a disaster.

```bash
python scripts/check_runs_landed.py --stage 1 --verbose \
    --validation-dir <KIRBy>/results/validation_rerun \
    --uncertainty-dir <KIRBy>/tests/results/uncertainty_rerun
```

**No chat is finished while its part of that output is not clean.** Say so plainly if it is
not, rather than closing on the code being fixed. And say which of MISSING, PARTIAL or THIN
is left, because they are three different amounts of work.

---

## Chat 1 — how long and how much memory every job asks for

> Read `CLAUDE.md`, then `RERUN_PLAN.md` section 13.28.
>
> The author: *"I need far more accurate guesses on how long these will take that won't
> jeopardize its ability to finish in time but also doesn't land it in the queue for
> eternity. This step is vital."*
>
> The measurement is done and it is in section 13.28. `measure_walls.py` found **188 changes
> worth making**, and `--emit-scontrol` prints them. You do not need to re-measure. You need
> to apply it, and to make the generators stop producing the old numbers.
>
> **1. Five jobs may be killed before they finish.** These ask for barely more than their
> longest measured run, and a job stopped at its limit writes nothing and cannot be recovered.
> `scontrol` cannot raise a limit, so each needs the generator changed and those tasks sent
> again.
>
> | job | longest run | asks for | margin | should ask |
> |---|---|---|---|---|
> | `val_svm`, laboratory censoring | 0:54 | 1:00:00 | 1.13× | 2:47:00 |
> | `qm91_qrf`, QM9 main grid | 20:52 | 1-02:59 | 1.29× | 1-18:43 |
> | `qm92_qrf`, QM9 censoring | 20:52 | 1-05:59 | 1.44× | 1-18:43 |
> | `qm92_qrf`, QM9 deep run | 20:52 | 1-05:59 | 1.44× | 1-18:43 |
> | `qm90_qrf`, QM9 screen | 2:25 | 3:59 | 1.66× | 5:49 |
>
> **2. Nothing in this study has ever used more than 4.2 GB.** That is the peak across every
> finished task. The screen jobs ask 128 GB and the deep run asks 96 GB. 64 GB is the author's
> floor and stays; the uncertainty jobs stay at 96 GB. Everything else should come down, and
> a smaller request is what lets the scheduler fit a job into a gap at all.
>
> **3. The long requests are why work sits still.** `qm92_ngboost` asks 22 days against 18
> hours measured. `qm92_mlp_bnn_full_mve` asks 13 days 18 hours against 6 hours. Lowering a
> limit on a job that has not started keeps its place in the queue, so it costs nothing.
>
> **4. Two of the three generators still use times typed by hand.** The QM9 one reads measured
> times from `model_hours.json`. The laboratory one and the uncertainty one do not, so
> anything rebuilt from them carries the old guess. Fix that first, because chats 2 and 3
> both need to resubmit.
>
> Chats 2, 3 and 5 all have tasks to resubmit and all of them are waiting on item 4. Tell
> them the moment it is pushed.
>
> **✅ 2026-09-07, chat 1: item 4 was already done and nobody is waiting on it.** All three
> generators price their walls from measurement — the QM9 one from `model_hours.json`, the
> laboratory one from its own seconds-per-fit table, the uncertainty one by importing that
> table rather than keeping a second copy. It was fixed at `eb08bb8`, before this file was
> written. Checked by generating each submission twice: once from the commit that was live
> when it went out, once from the branch tip. **Chats 2, 3 and 5 can regenerate and resubmit
> now.** Items 1, 2 and 3 are done and are in `RERUN_PLAN.md` §13.27 D1 with the paste block.
> Two things there that those chats need: memory is 64 GB everywhere on both grid pipelines
> now, and the twelve uncertainty arrays are queued with walls too short for five of their six
> models, on a partition that cannot hold the right ones.
>
> Done when section 13.27 lists, per submission, what it asks now, what it should ask, and the
> command to change it — **and `squeue` shows the new limits in place**, not just the
> `scontrol` lines having been printed.

---

## Chat 2 — `gauche_rbf`

> Read `CLAUDE.md`, then `RERUN_PLAN.md` section 13.28.
>
> One model, four separate problems, and until now they were in three different chats so
> nobody owned it. The author added this model for the uncertainty requirement: it separates
> the two halves of uncertainty better than anything else in the study.
>
> **1. Twenty tasks failed and the cause is already fixed.** Eight on the QM9 main grid and
> twelve on the deep run, all at noise level 1.5, replicates 7 to 9, on all three
> representations. The error said the model was fitting 5,000 rows while the noise record
> covered 8,000. Commit `c223ec3` fixed it: an exact Gaussian process is too slow above 5,000
> molecules so it subsamples, and the function returned only the count and threw away which
> molecules it kept. **Nobody has checked whether the twenty failures survive the fix.** Check
> that first; everything else about this model depends on it.
>
> **2. Its screen jobs have never started.** Eighteen tasks, job 12971618, queued since 2
> September, waiting on Priority. It asks 1 day 18 hours and 128 GB. The same model finished
> elsewhere in 1 hour 34 minutes and peaked at 4.2 GB. Cut both and it will be scheduled.
>
> **3. It is in the deep run's model list and the corrected reading takes it out.** See chat
> 4 — do not act on that alone, the two chats have to agree.
>
> **4. Whether it stays at all was raised as a way to finish sooner**, on the grounds that it
> had never run and was worth 17 days of requested time. Both of those are now false: it has
> produced screen results, and 17 days was the wrong request, not the real cost. Re-price it
> before anyone offers dropping it again.
>
> Done when the twenty tasks are either running clean or their remaining cause is written
> down, its screen jobs have started, and `check_runs_landed.py` no longer reports this model
> as MISSING or PARTIAL anywhere.

---

## Chat 3 — the laboratory datasets

> Read `CLAUDE.md`, then `RERUN_PLAN.md` section 13.28.
>
> logD, Caco-2 and hERG. Everything here writes into the KIRBy checkout, not this one:
> the breadth grid into `results/validation_rerun/`, the uncertainty jobs into
> `tests/results/uncertainty_rerun/`, because those jobs change into the `tests` directory
> and pass a relative results path.
>
> **1. Fifty tasks failed and nothing knows why.** Six each from `val_bnn-full`,
> `val_bnn-full-mve`, `val_dnn` and `val_lightgbm`, one from `val_gp-tanimoto`, and the 25
> hERG resubmissions. Their output files are missing. The jobs ran in
> `slurm_scripts_validation_rerun` in this repository, so the tool was looking in the right
> place — the logs are genuinely gone, most likely because the scripts were rebuilt with a
> different output name afterwards. **Get the cause from the cluster another way** before
> resubmitting any of them, because resending an unfixed cause gets the same failure. The 25
> hERG ones were the missing-cache deaths of 2 September and the cache is present now, loading
> 1,415 molecules, so those are probably ready — confirm and send them.
>
> **2. One bug can write results that look fine.** In the KIRBy runner, if one inner fold
> fails while scoring training molecules, it is caught, skipped with a warning, and rows are
> still written for every molecule with empty values. The checks count rows rather than real
> values, so the job reports success. The merge step labels the cell truncated and also
> reports success. **A laboratory uncertainty result cannot be trusted from whether the job
> succeeded.** Fix the check to count real values, run the merge over everything already
> written, and list every truncated cell for deletion.
>
> **3. The 378 uncertainty jobs have never started.** Twelve arrays, 12986390 to 12986401,
> queued since 6 September, all waiting on Priority, all asking 96 GB and one and a half to
> two days. Nothing has ever run in this pipeline so nothing can size them; chat 1 owns the
> request sizes, you own whether they should run at all. Of the four models chosen, only the
> quantile forest reports both halves of its uncertainty per molecule. NGBoost has no
> model-uncertainty half. The Gaussian process and the variational network give one noise
> number for the whole fit. The three models that do give both halves per molecule are on
> neither list. Price the options and put it to the author; do not choose.
>
> **4. Two things are already clean, do not re-open them.** No laboratory task anywhere ran
> under the old noise draw — that was checked on 7 September and the answer was zero. And the
> laboratory depth run repeats three conditions the breadth grid already runs, which wastes
> queue time but produces correct numbers, because the runner replaces its own rows.
>
> Done when the fifty failures have a cause, the truncated cells are listed, the 378 jobs are
> either running or withdrawn on the author's word, and `check_runs_landed.py` pointed at the
> KIRBy checkout reports nothing MISSING or PARTIAL for logD, Caco-2 or hERG.

---

## Chat 4 — the deep run's model list

> Read `CLAUDE.md`, then `RERUN_PLAN.md` section 13.28.
>
> `deep_run_pairs.json` names six models and three representations, eighteen combinations.
> `censoring_pairs.json` names five combinations outright. Both still say provisional. The
> deep run, jobs 12986314 to 12986332, and both censoring runs are executing against them now.
> **Each task reads the file when it starts**, so removing a combination is free at any
> moment, and adding one back means resubmitting those indices.
>
> **1. The list was written from an unfinished screen and a tool with a bug in it.** The
> corrected tool was run on 7 September, at six models, over 20,224 rows. Its reading and the
> queued file agree on three of six and differ on three:
>
> | in the file now | the reading says | why |
> |---|---|---|
> | `ngboost` | `ngboost` | locked; the generator refuses to build without it |
> | `rf` | `rf` | most noise-tolerant in 2 of the screen's combinations |
> | `svm` | `svm` | the only kernel model |
> | `het_gp_rbf` | `dnn_vbll_hetero` | least noise-tolerant in 8 of 9, against 2 of 3 before |
> | `gauche_rbf` | `gauche` | one from the Gaussian process family |
> | `dnn_bnn_full_mve` | `dnn` | one from the plain neural family |
>
> **2. The reading's own list breaks a rule and the queued file does not.** The tool warns
> that only one of its six reports an uncertainty per molecule, and the rule needs at least
> two. The author put `gauche_rbf` and `dnn_bnn_full_mve` in the file for exactly that reason.
> So this is not "the tool is right and the file is wrong" — say that plainly when you put it
> to her.
>
> **3. The censoring list has the same problem, more sharply.** The reading names five
> combinations and warns that only one of them reports an uncertainty per molecule.
>
> **4. The scores this rests on are sound.** The no-noise results the ranking divides by were
> checked on 7 September and agree exactly across the three noise types. See section 13.28.
>
> **This is the author's decision, not yours.** Put the three differences in front of her with
> what each costs, note that removing is free and adding back is not, and let her answer.
>
> Done when both files say what she has decided, no longer say provisional, and
> `check_runs_landed.py --stage 2` accounts for every combination those files name.

---

## Chat 5 — Sort & Slice

> Read `CLAUDE.md`, then `RERUN_PLAN.md` section 13.28.
>
> One representation, 104 failed tasks, and a cause that is already fixed and proved.
>
> **1. What went wrong.** Methane, ammonia and water each carry exactly one Morgan
> substructure, and it occurs in exactly one molecule, so none of them can ever reach a
> top-1024 chosen by how often a substructure appears in training. Their vector comes out all
> zeros, and the guard refuses to train on a molecule with no features. Measured over all
> 132,480 QM9 molecules: **3 molecules, 0.0023 per cent**.
>
> **2. The fix is in and tested.** Commit `62f1fe2`. Those molecules are dropped from **every**
> representation, not just this one, so an ECFP4 job and a Sort & Slice job score the same
> molecules. They are found by the property rather than by a list of three names, and the
> featuriser is built on every run. Proved by `scripts/test_sns_zero_exclusion.py`. The guard
> stays.
>
> **3. It changes nothing already on disk.** This was checked in code on 7 September: the
> exclusion removes molecules from the three split lists *after* the shuffle and the split,
> and never re-splits. So a task whose sample never drew one of the three gives exactly the
> same answer before and after the fix, and the tasks that did draw one crashed and wrote no
> rows. **The screen's Sort & Slice results are comparable with the main grid's. Do not
> re-run the screen for this.**
>
> **4. What is left is the resubmission, and one tool stands in the way.** The 104 tasks are
> indices 5, 11 and 17 of nearly every QM9 main-grid job. `failed_tasks.py --emit-sbatch`
> prints nothing for them, because it refuses to print a resubmission for anything marked
> failed, on the rule that resending an unfixed cause gets the same error. That rule is wrong
> here. Teach it to print when the cause is fixed at a named commit, tied to the commit so it
> cannot just be asserted, and add that case to `scripts/test_slurm_status_tools.py`. Fix it
> once, prove it with the test, and resubmit. Do not ship the tool, run it, find a bug and
> ship it again — that consumed a whole session.
>
> **5. Wait for chat 1 to push the corrected time limits**, rebuild the scripts, then send
> them. Never type a job array range by hand; the generators write the right range for each
> script, and typing one has queued out-of-range tasks three times.
>
> **6. The same tool fix unblocks chats 2 and 3.** Twenty `gauche_rbf` tasks and fifty
> laboratory tasks are in the same position. Tell both chats when it is pushed.
>
> Done when the 104 tasks are running, `failed_tasks.py --emit-sbatch` prints a command for a
> fixed cause, and `check_runs_landed.py --stage 1` reports no Sort & Slice cell MISSING or
> PARTIAL.

---

## The last thing, after all five chats

Whoever gets there last runs the three completeness commands above and posts the output. If
every one exits 0, every result the generators asked for is on disk. If any does not, the
remaining work is named cell by cell and goes into section 13.27 as MISSING, PARTIAL or THIN
with the resubmission line beside it.

**Nobody declares this finished from a code change, a commit, or a job having been submitted.**
Only from that output.
