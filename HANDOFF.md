# Handoff — three chats

Read `CLAUDE.md` first. Everything here is aimed at one thing the author asked for on
2026-09-07: **every result off the cluster, as fast as possible.** Figures, the paper, the
analysis job's output and the open decisions are set aside until that is done.

Branch `additional_reps`. Code reaches the cluster only when the author runs
`bash scripts/pull_safely.sh` against a commit already pushed.

Each chat writes its answers into `RERUN_PLAN.md` section 13.27 and nowhere else. When all
three have written, that section is the list of commands to run.

Two orderings matter. Chat 1 pushes its time-and-memory changes before chat 2 rebuilds and
resubmits anything. Chat 3 checks the clean results before it decides which models the deep
run keeps.

---

## Chat 1 — how much time and memory each job asks for

> Read `CLAUDE.md` first.
>
> The author, 2026-09-07: *"I need far more accurate guesses on how long these will take that
> won't jeopardize its ability to finish in time but also doesn't land it in the queue for
> eternity. This step is vital."*
>
> **Get real numbers off the cluster before changing anything.** Not from a document, not
> from a previous session. For every model, on every dataset, in every submission: how long
> its longest finished task actually took, and how much memory it actually used. `sacct` will
> give you both. Assume you will need one copy-paste round trip to get them, and send one
> block that covers everything you need rather than asking twice.
>
> Then fix these, in this order.
>
> **1. The quantile forest may be killed before it finishes.** Its longest task anywhere took
> 20 hours 52 minutes. The main grid asks for 1 day 2 hours 59 minutes, which is only 29 per
> cent more than the longest run measured. Laboratory censoring asks 1 hour for a model whose
> longest run took 54 minutes, 13 per cent more. Everything else in the study asks for nearly
> five times its longest run or more. A job stopped at its limit writes no result and the work
> is lost, so this is the only way this study loses finished work permanently. `scontrol`
> cannot raise a limit for its owner, so the request has to change in the script generator and
> the affected jobs have to be sent again.
>
> **2. NGBoost asks for 22 days and never gets scheduled.** Its longest measured run is under
> four days. One of the network jobs asks 13 days 18 hours against 8 hours 38 minutes
> measured. A three-week request almost never fits a gap in the schedule, which is why these
> sit waiting while four-hour jobs overtake them. Lowering the limit on a job that has not
> started keeps its place in the queue, so it costs nothing. This list has never been produced
> since the last fix.
>
> **3. Two of the three generators still use time guesses typed by hand.** The QM9 one now
> reads measured times from `model_hours.json`. The laboratory one and the uncertainty one do
> not, so anything resubmitted from them carries the old guess. Tell chat 2 the moment this is
> pushed, because it cannot resubmit until then.
>
> **4. The memory amounts come from a different study.** The tiers rest on a 61.2 GB figure
> that was not measured on this work. Measure what this study actually uses. 64 GB is the
> author's floor and does not change, and the uncertainty jobs stay at 96 GB. What is open is
> the 96 GB tier for everything else, and the proposal is to lower it only where something has
> actually been measured, keeping 96 GB where nothing has finished yet.
>
> **5. Nobody recorded whether the main grid's memory was ever raised.** Jobs 12980573 to
> 12980591 went in at 32 GB, below the author's floor. One command tells you whether the
> raise was applied.
>
> **Also find out why some jobs have never started at all**, because a request no machine can
> satisfy is the usual reason: the 378 uncertainty jobs since 6 September, the `gauche_rbf`
> screen jobs since 2 September, and two more screen jobs that are part-finished and stalled.
>
> Done when section 13.27 lists, for every submission, what it asks for now, what it should
> ask for, and the exact command to change it.

---

## Chat 2 — the 174 tasks that failed and have never been sent again

> Read `CLAUDE.md` first.
>
> 174 tasks have failed and **not one has been resubmitted.** Read the live count every time;
> it was 48, then 51, then 58, then 104 for one of the causes, because more tasks keep
> reaching the same fault. It is one number growing, not four documents disagreeing.
>
> Of the 174: 104 failed on Sort & Slice, 20 on `gauche_rbf`, and 50 have no known cause at
> all.
>
> **1. The tool refuses to print a resubmission command.** `failed_tasks.py --emit-sbatch`
> printed nothing. It marks every task as failed, and it will not print a resubmission for
> anything marked failed, on the rule that resending an unfixed cause gets the same error.
> That rule is now wrong for 124 of them, because both causes have been fixed in the code:
> Sort & Slice in commit `62f1fe2` and `gauche_rbf` in commit `c223ec3`. Teach the tool to
> print a resubmission when the cause is fixed, tied to the commit so it cannot just be
> asserted, and add that case to `scripts/test_slurm_status_tools.py`. **Fix it once, prove
> it with the test, and go straight to resubmitting.** The previous session was consumed by
> shipping a tool, running it, finding a bug and shipping it again.
>
> **2. Fifty tasks have no cause because the tool looks for their output in the wrong
> repository.** It looks in `slurm_scripts_validation_rerun/`. Laboratory jobs change into
> the `tests` directory of the KIRBy checkout and write there. **The 25 hERG tasks are inside
> those 50 and have been waiting three sessions.** Their cause was a missing cache file on 2
> September; the cache is there now and loads 1,415 molecules, so nothing else is wrong with
> them.
>
> **3. `gauche_rbf`'s 20 failures have never been re-checked against the fix.** Commit
> `c223ec3` fixed the cause. Nobody has confirmed the failures go away. Do that before anyone
> discusses dropping the model. It has since produced results on the screen, so the argument
> that it has never run is no longer true.
>
> **4. Then resubmit.** Wait for chat 1 to push the corrected time limits, rebuild the
> scripts, and send them. Never type a job array range by hand — the generators write the
> right range for each script, and typing one has queued out-of-range tasks three times.
>
> **5. While you are in the status tool:** it labels the two uncertainty submissions
> "uncertainty, the three" and "uncertainty, the four". Those are invented codes, which the
> author has banned, and they appear in every status output she reads. Rename them.
>
> Done when section 13.27 lists, for each script, the exact resubmission command, how many
> tasks it covers, and which cause it clears.

---

## Chat 3 — whether the finished results are correct, and whether the running jobs are

> Read `CLAUDE.md` first.
>
> **1. Start here, before anything in any chat. Two results that should be identical are
> not.** When a run is set to add no noise at all, the answer cannot depend on which kind of
> noise the run was labelled with. For one model on ChemBERTa, it does: the no-noise result
> differs between three of the noise types, on all five accuracy measures. That difference
> has two possible causes and they are worlds apart. If it is around one part in ten million,
> it is neural network training being slightly different on different machines, nothing is
> wrong, and the check that found it just needs to tolerate it. If it is around one part in a
> hundred, then noise is being added to runs that are supposed to have none, and **every
> robustness number in the study is wrong**, because each one is measured against its own
> no-noise result. Print the rows side by side at full precision and you will know. The check
> stops at the first disagreement, so nothing after ChemBERTa was looked at — check the rest
> too, then make it refuse one configuration instead of stopping everything.
>
> **2. A failed scoring pass can write results that look fine.** In the KIRBy runner, if one
> inner fold fails during the pass that scores training molecules, it is caught, skipped with
> a warning, and rows are still written for every molecule with empty values. The checks count
> rows rather than real values, so the job reports success. The only thing that notices is the
> merge step, which labels the cell as truncated and then also reports success. **A result
> here cannot be trusted from whether the job succeeded.** Make the check count real values,
> then run the merge over everything already written and list every truncated cell for
> deletion.
>
> **3. One number in the screen looks impossible.** One network on ChemBERTa scores 0.9958,
> the highest in the whole screen, from a cell that has only two of the seven noise types.
> Check whether that score is being computed from a shorter run than the others. If it is,
> every such number is inflated, and they feed the ranking in item 5.
>
> **4. The progress counts are wrong.** `run_status.py` counts a job that started and
> immediately exited as finished work. The deep run reads 364 finished, and most of those are
> exits that took seconds. Fix it to count them separately, which the time-measuring tool
> already does.
>
> **5. The deep run may be training the wrong models right now.** It reads its model list when
> each task starts, so the file can still be changed. The list was written before the screen
> finished. Now that the screen is readable, two of the six no longer match it. The random
> forest is in the list as the most noise-tolerant model on all three representations; on the
> corrected reading NGBoost is top in seven of the nine combinations and the random forest is
> top in none. A Gaussian process variant is in the list as the least tolerant on two of
> three; on the corrected reading a different network is least in eight of nine. Re-read it
> asking for six models, not four — the tool takes the number from whichever file you point it
> at, so pointing it at a scratch file silently asks for four. **Wait until item 1 is settled,
> because these scores are all measured against the no-noise results.** Then put the change to
> the author. It is her decision, not yours.
>
> **6. The uncertainty jobs may be running models that cannot answer the question.** Of the
> four chosen, only the quantile forest reports both halves of its uncertainty for each
> molecule. NGBoost has no model-uncertainty half at all. The Gaussian process and the
> variational network each give one noise number for the whole fit rather than one per
> molecule. The three models that do give both halves per molecule are on neither list. Price
> the options in jobs and hours and put it to the author. Do not choose yourself.
>
> Done when section 13.27 lists every result that has to be deleted, with the reason, and what
> each queued submission is currently computing against what it should be.

---

## What is set aside until the results are in

On the author's instruction, 2026-09-07. Not closed, not forgotten.

Figures and the rank-versus-level charts. The noise level hardcoded in the figure script. The
three numbers in the paper on the retired scale. The Caco-2 noise anchor and the label spread
it needs. The three levers for finishing sooner. Everything in the KIRBy repository except
the failed scoring pass in chat 3: the preflight step that runs a git pull that cannot
succeed, the script that picks the wrong account, the runbook pointing at the wrong checkout,
the fold check that claims more than it tests, and the model list that disagrees between
files.
