# Handoff — five chats, one thing each

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
> Done when section 13.27 lists, per submission, what it asks now, what it should ask, and the
> command to change it.

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
> down, and its screen jobs have started.

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
> Done when the fifty failures have a cause, the truncated cells are listed, and the 378 jobs
> are either running or withdrawn on the author's word.

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
> Done when both files say what she has decided and no longer say provisional.

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
> Done when the 104 tasks are running and `failed_tasks.py --emit-sbatch` prints a command for
> a fixed cause.
