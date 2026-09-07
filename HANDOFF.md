# HANDOFF — four chats, one goal: every result off the server

**Rewritten 2026-09-07, second pass.** The author, in their own words:

> *"All I care about is getting all the results in from the server. Things are not working
> right so I can't do this. I do not care about anything else right now. ALL the results.
> Not just the screen. Not just the ANOVA."*

So: **no figures, no paper text, no analysis, no decision menus.** Four chats. Every one of
them exists to make the cluster produce rows and to say which rows on disk are wrong.

**The deliverable is not a report.** It is one command sheet at `RERUN_PLAN.md` §13.27 that
says, for every submission: what failed and why, what has to be **deleted**, and what has to
be **resubmitted** — with the exact `sbatch` and `rm` lines. Each chat writes its verdict
into §13.27. Nobody writes a new file.

Branch `additional_reps`. A fix that is not pushed does not exist — the cluster's only route
in is the author running `bash scripts/pull_safely.sh`.

---

## A. THE ONE THING THAT MIGHT INVALIDATE RESULTS

Everything else checked out. Two candidates were traced to the code today and **both are
clear**:

- **The QM9 sample is the same molecules across every condition.**
  `torch.manual_seed(iteration_seed)` runs at `scripts/process_and_train.py:3450`, nothing
  consumes torch randomness between there and `split_qm9`'s `torch.randperm` at `:1160`, and
  `iteration_seed` depends only on `--random-seed` and the replicate number. Two conditions
  at the same replicate train on the same molecules.
- **The Sort & Slice exclusion does not move anybody else.** It removes indices from the
  three split lists *after* the shuffle and split (`:1261`–`:1265`); it does not re-shuffle
  and does not re-split. A task whose sample never drew methane, ammonia or water is
  bit-identical before and after `62f1fe2`. The tasks that did draw one crashed and wrote
  nothing. **T08 is closed: the screen's Sort & Slice rows are not invalidated.**

**The one that is still open is T52, and it is the highest-value command in this handoff.**

`copy_zero_rows.py` refused to copy because the clean row for ChemBERTa /
`dnn_bnn_full_mve` under `grouped_wider` and `grouped_shifted` differs from the same
model's clean row under `gaussian`, replicate 1, on all five accuracy columns. Two
explanations, and they are worlds apart:

| if the difference is | then | and |
|---|---|---|
| ~1e-7 | torch is nondeterministic across nodes — nothing in the codebase sets `torch.use_deterministic_algorithms` or `cudnn.deterministic`, and only neural models can show it | **nothing is invalid.** Fix the copy to tolerate it for networks and move on |
| ~1e-2 or larger | noise is being applied at level 0 for the grouped conditions | **every AUC_norm in the study is wrong**, because AUC_norm divides by the clean level. Everything gets re-run |

It is one command to know which. Do it first, in CHAT 3.

---

## B. WHAT THE 378 UNCERTAINTY TASKS ACTUALLY ARE

These are the jobs that test whether a model knows when it is wrong. They have nothing to do
with QM9. They run on the three laboratory datasets only.

There are 378 of them because every combination gets its own job. The combinations are:

- **6 models** — the quantile forest, NGBoost, the Gaussian process, the variational
  network, and the two networks that predict their own error.
- **3 datasets** — logD, Caco-2 and hERG.
- **3 ways of describing a molecule** — ECFP4, PDV and ChemBERTa.
- **7 kinds of noise.**

Six times three times three times seven is 378.

They went to the cluster as two separate submissions because three of the seven kinds of
noise were sent on their own and the other four followed. That is 162 jobs and then 216.

**What one job does.** It takes one model, one dataset, one way of describing a molecule and
one kind of noise. It then trains that model at every noise level on the ladder. At each
level it trains five times over, on five different splits of the molecules. On top of that it
trains extra times so that every training molecule gets a score from a model that never saw
it. It does all of that once — there are no repeat runs, the five splits are the only repeat.

**Some of the 378 do nothing at all.** The Gaussian process is only meant to run on PDV, but
the number of jobs was worked out as though every model ran on all three ways of describing a
molecule. So the Gaussian process jobs for ECFP4 and ChemBERTa start and immediately stop.
Nobody has counted how many of the 378 are those. That matters, because it changes what
"none of them have started" actually means.

## C. THE RULES EVERY CHAT INHERITS

**`CLAUDE.md` in the repository root is the full set and every chat must read it.**
It is loaded automatically, but read it anyway — the plain-English rules at the top of
it are the ones that have been broken most. What follows is the short version.

- **Fixes, not findings.** Never advise stopping, pausing or deferring.
- **Ask the server freely, but batch.** You cannot see the cluster. Send one block covering
  several threads. Prefer raw `sacct` / `squeue` / `ls` / `tail` over anything you just
  wrote. Run `python scripts/test_slurm_status_tools.py` locally before sending anything
  that uses `failed_tasks.py`, `measure_walls.py`, `run_status.py` or `slurm_jobs.py`.
- **Never send a command whose output you cannot act on.**
- **Never delete rows on a hunch.** A delete line goes in §13.27 with the reason and the
  evidence, and the author runs it. Deleting a result is irreversible and the re-run is days.
- **Anything that changes the science is the author's. Anything a measurement can settle is
  yours.**
- **Plain English.** One fact per sentence. Lead with a verdict. Never open a message with a
  correction of your own last one.
- **Never quote a number from memory.** Trace it to a file or a command, or say "not checked".
- **Banned: "arm", "stage".** Say *the screen / the main grid / the deep run / the
  uncertainty runs*. Say **AUC_norm**, say **PDV**.
- **One state document** — `RERUN_PLAN.md`. Never start a new file.
- **`paper.tex` is never edited.** Nothing in this handoff touches the paper anyway.

**Deferred on the author's instruction, 2026-09-07:** figures, the analysis job's output,
paper replacement text, and the six-decision menu — *"I BASICALLY MADE THEM OR DEEMED THEM
IRRELEVANT"*. Threads T04 T05 T19 T20 T21 T42 T43 T44 T49 T50 are parked, not closed. The
only survivor from that group is T13, and only as *stop it writing*, not as *read what it
wrote*.

---

# THE FOUR PROMPTS

---

## PROMPT 1 — WHY NOTHING RUNS

> **Read `CLAUDE.md` first and follow it, especially the plain-English rules — the
> author has to be able to read what you send.** Then `RERUN_PLAN.md` §13.26 and
> §13.27, and `HANDOFF.md` sections A–C.
>
> You own **T06 T07 T15 T18 T23 T26 T27 T28 T35 T36**. Nothing here is scientific. All of it
> is why the cluster is idle while the author waits, and one item of it destroys work.
>
> **The 378 uncertainty tasks have not started a single task since 2026-09-06** (T23),
> `gauche_rbf`'s screen array has not started since 2026-09-02, `qm90_mlp_bnn_full_mve` has
> 16 of 18 still queued and `qm90_dnn_bnn_full_variational` 10 of 18 (T15). **Find out why
> each is not being admitted before you touch anything else** — a wall or memory request no
> node can satisfy is the usual cause, and `unc_*` is its own pipeline so nothing has ever
> measured its walls.
>
> **T26 is the only irreversible failure class in the study.** `qrf`'s longest task anywhere
> is 20:52 against a 1-02:59 request — 1.29× headroom on a measurement that is already the
> worst of fifteen tasks. `val_svm` in laboratory censoring asks 1:00:00 against 0:54, 1.13×.
> Everything else has 4.9× or more. A job killed at its wall writes no row and gets no
> partial credit, and `scontrol` cannot raise a limit for its owner — so this is a generator
> change plus a resubmission of the unrun indices.
>
> **T28/T18, the other direction:** `qm92_ngboost` asks 22 days against under four measured,
> `qm92_mlp_bnn_full_mve` 13 days 18 hours against 8:38. A three-week request fits almost no
> backfill gap, which is exactly why these sit on Priority while four-hour jobs overtake
> them. Cutting a `TimeLimit` on a pending job keeps its submit time. **The corrected
> `scontrol` list has never been emitted or applied.** The previous session called this one
> change "your week".
>
> **T06:** the QM9 generator reads `model_hours.json` at a 2.0× margin since `62f1fe2`. The
> **laboratory and uncertainty generators still use the hand-written `hours_per_110`
> guesses**, so every laboratory resubmission goes out over-asked. Port the rule before
> CHAT 2 resubmits anything, and tell CHAT 2 when it is pushed.
>
> **T27/T07, memory:** the tiers rest on 61.2 GB from a different study. **Re-measure the
> real peak — do not quote 4.1 GB, that came from a previous session.** `measure_walls.py`
> prints a peak GB column per model per submission. 64G is the author's settled floor and the
> uncertainty runs hold 96G; neither changes. What is open is the 96G tier for everything
> else, and the proposal is a **carve-out, not a flat drop**: 64G for everything that has
> measured, 96G kept for anything with no completed task to argue from.
>
> **T35:** `12980573`–`12980591` went in at `--mem=32G`, below the floor, and no document
> records whether the `scontrol MinMemoryNode` lines were ever run. One command.
>
> **T36:** submission 7 (`12986352`–`12986370`) went in without `--conditions student_t_nu5
> outlier_p10 laplace`, so the laboratory depth run repeats three conditions the breadth grid
> already runs. The numbers are right either way — it is 327 tasks of queue time competing
> with work that has never started.
>
> **Write into §13.27:** per submission, what it is asking, what it should ask, and the exact
> `scontrol` / `sbatch` lines.

---

## PROMPT 2 — PUT BACK WHAT FAILED

> **Read `CLAUDE.md` first and follow it, especially the plain-English rules — the
> author has to be able to read what you send.** Then `RERUN_PLAN.md` §13.26 and
> §13.27, and `HANDOFF.md` sections A–C.
>
> You own **T12 T17 T24 T25 T54 T55**. **174 failed tasks and not one resubmission line has
> ever been emitted for any of them.** 104 Sort & Slice, 20 `gauche_rbf`, 50 with no cause
> classified at all. Read the live number every time — it was 48, then 51, then 58, then 104,
> and that is one number growing as more tasks reach the fault, not documents disagreeing.
>
> **T54 is the blocker and it is yours to fix, today.** `failed_tasks.py --emit-sbatch`
> printed nothing, because every task classes as `FAILED` and the tool prints no sbatch for
> `FAILED` on the rule that resubmitting an unfixed cause gets the same exit. **That rule is
> wrong for 124 of them**: Sort & Slice was fixed in `62f1fe2` and `gauche_rbf` in `c223ec3`.
> Give the tool a way to say *"cause fixed at commit X, emit the resubmission"*, tied to the
> commit so it cannot be asserted by hand, and add the case to
> `scripts/test_slurm_status_tools.py`. Fix it once, prove it with the test, resubmit. Do not
> ship it, run it, find a bug, and ship it again — that consumed the previous session.
>
> **T55:** 50 tasks have no cause because the log path is wrong — it looks in
> `slurm_scripts_validation_rerun/`, but laboratory jobs `cd tests` inside the KIRBy
> checkout and write there. **The 25 hERG tasks of T17 are inside those 50 and are still
> unclassified after three sessions.** Fix the path, classify them, put them back. Their
> cause was the missing cache of 2026-09-02; the cache is present and loading 1,415
> molecules, so nothing further is wrong with them.
>
> **T25:** `gauche_rbf`'s 20 failures have a fixed cause (`c223ec3`) and **nobody has
> re-checked whether they survive it.** Do that before anyone argues about dropping the
> model. It has now landed on the screen under gaussian — AUC_norm 0.9387 ECFP4, 0.9399 PDV,
> 0.9503 ChemBERTa — so "it has never run" is no longer true.
>
> **T24:** resubmit the 104 Sort & Slice tasks — but **wait for CHAT 1 to push the wall rule**
> so the regenerated scripts do not carry the old numbers, and regenerate before you submit.
> Never type an array range: all three generators write `submit_all.sh` with each script's own
> range, and `gauche` runs on ECFP4 alone so it holds a different count from the rest. Guard:
> `scripts/test_submit_all_ranges.py`.
>
> **T12:** `slurm_jobs.py` labels the two uncertainty submissions "uncertainty, the three" and
> "uncertainty, the four". Coined codes, banned. Rename — it is in every status output the
> author reads.
>
> `scripts/slurm_jobs.py` knows the submission-to-directory-to-script mapping. The QM9 job
> name is `qm91_rf` and its script is `qm9_s1_rf.sh`; they are different, and getting it wrong
> has queued out-of-range tasks three times.
>
> **Write into §13.27:** the exact `sbatch --array=` line per script, with the count and the
> cause it clears.

---

## PROMPT 3 — WHAT IS ON DISK AND WRONG

> **Read `CLAUDE.md` first and follow it, especially the plain-English rules — the
> author has to be able to read what you send.** Then `RERUN_PLAN.md` §13.26 and
> §13.27, and `HANDOFF.md` sections A–C.
>
> You own **T11 T13 T38 T52 T56**, and you produce the **delete list**. Nothing else in this
> handoff can tell the author which rows already on disk are not to be trusted.
>
> **T52 first, before anything else in any chat.** Read `HANDOFF.md` section A for what is at
> stake: if the clean-row difference is ~1e-7 it is torch nondeterminism and nothing is
> invalid; if it is ~1e-2 then noise is being applied at level 0 and **every AUC_norm in the
> study is wrong.** Print the two rows and the reference row side by side, all five accuracy
> columns, full precision. Then check whether any other model shows it —
> `copy_zero_rows.py` stops at the first disagreement, so it got as far as ChemBERTa and
> nothing past it was examined at all. Only then fix the script: one model's disagreement must
> refuse that one configuration, not stop the copy for all nineteen. **The copy is a
> prerequisite for the deep-run selection**, so tell CHAT 4 the moment it runs clean.
>
> **T38 is how a wrong number gets written with exit 0.** In the KIRBy runner, an inner fold
> that raises during the out-of-fold pass is caught, skipped with a printed warning, and rows
> are still written for every training molecule with NaN values. The integrity gates count
> rows rather than finite values, so the job exits 0. The only detection is
> `merge_results.py`, which classifies the cell `TRUNCATED_OOF` and then returns normally,
> exit 0 as well. **An uncertainty cell cannot be trusted from its exit code.** Fix the gate
> to count finite values, then run the merge coverage table over everything already written
> and put every `TRUNCATED_OOF` cell on the delete list.
>
> **T56:** `mlp_bnn_full_mve` on ChemBERTa reads AUC_norm 0.9958, the highest number in the
> entire screen, from a cell holding 2 of 7 conditions. Check whether AUC_norm is being
> computed over a short level ladder anywhere. If it is, every such cell is inflated and
> feeds the ranking CHAT 4 depends on.
>
> **T11:** `run_status.py` counts a selection-gate skip as done, so "deep run 364 done" is
> mostly seconds-long exits. `measure_walls.py` already separates them. Fix it — the author
> cannot see how far along anything is.
>
> **T13, narrowly:** `13033488` is the analysis job, reading a grid that is missing one
> representation of six in every replicate 1 to 9. **Your only job here is to stop it writing
> over good output** — `check_runs_landed.py` should gate it. Do not read or discuss what it
> produced; the author has deferred all analysis.
>
> **Write into §13.27:** the delete list — one line per file or row range, the reason, and
> the `rm` or rewrite command. The author runs it. Never delete anything yourself.

---

## PROMPT 4 — WHAT THE RUNNING JOBS ARE RUNNING

> **Read `CLAUDE.md` first and follow it, especially the plain-English rules — the
> author has to be able to read what you send.** Then `RERUN_PLAN.md` §13.26 and
> §13.27, and `HANDOFF.md` sections A–C.
>
> You own **T02 T03 T22 T33**. These are the two files that decide what the queued work
> actually computes, and both still say `provisional: true` while the jobs execute against
> them. Every task reads its file **when it starts**, so narrowing is free at any moment and
> widening means resubmitting those indices.
>
> **T02/T33 — the deep run.** The corrected selector has now been run, on 20,224 rows from
> 342 files, and it does not support two of the six queued models. `rf` is in
> `deep_run_pairs.json` as "most noise-tolerant on all three representations"; on the
> corrected reading `ngboost` tops 7 of the 9 representation-and-condition cells and **`rf` is
> most on none.** `het_gp_rbf` is in as "least noise-tolerant on two of three, READ OFF ONE
> CONDITION"; on the corrected reading the least-tolerant model is `dnn_vbll_hetero`, least in
> **8 of 9**. Re-read it at `--n-models 6` — the selector takes its model count from the file
> named by `--out`, so a scratch path silently reads 4 (T53, fixed in CHAT 3 or by you, do not
> both). **Then put the change to the author as a change to confirm or reject. It is their
> decision.** Wait for CHAT 3 to clear T52 first; AUC_norm is retention against the clean
> level, so the ranking is not final until the clean rows are settled.
>
> **T22 — the uncertainty pair list, and this is the one that decides whether 378 tasks are
> worth running at all.** Of the four settled uncertainty pairs, only `qrf` reports both the
> aleatoric and the epistemic half per molecule. NGBoost has no model-uncertainty half — one
> fit, and its seed reaches only `minibatch_frac` and `col_sample`, both pinned at 1.0. The
> Gaussian process and the variational network each report one data-noise number per fit, not
> per molecule. **The three models that have both halves per molecule —
> `heteroscedastic_gp` and the two variational networks with a noise head — are on neither
> uncertainty run's list**, and the QM9 generator gives them `-u True` with no `--oof-folds`,
> so they write test rows whose injected noise is exactly zero. Price each option in tasks and
> hours and put it to the author. **Do not set the list yourself.**
>
> **T03:** three of the four decisions deferred in §13.17 B have had their trigger fire. Read
> them off, with numbers, in the same message.
>
> **Write into §13.27:** what each queued submission is currently computing, and what it
> should be, so the author can see in one place whether anything running is wasted.

---

## D. HOW THE FOUR CHATS CONVERGE

Order matters in exactly three places, and nowhere else:

1. **CHAT 3 clears T52 before CHAT 4 finalises the selection.** AUC_norm divides by the clean
   level; the ranking is not real until the clean rows are.
2. **CHAT 1 pushes the wall rule before CHAT 2 regenerates and resubmits.** Otherwise 124
   tasks go back out with the old walls and CHAT 1's work is wasted.
3. **CHAT 4 settles the two pair files before anything is widened.** Narrowing is free;
   widening costs a resubmission.

Everything else runs in parallel. Each chat appends to `RERUN_PLAN.md` §13.27 — never a new
file, never a summary in chat. When all four have written their rows, §13.27 **is** the
command sheet: what failed and why, what to delete, what to resubmit, in order.
