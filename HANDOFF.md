# HANDOFF — seven chats, thirty-nine open threads, and what closes each one

**Rewritten 2026-09-07** after the first command block was run against ARC at `09ff59e`.
The previous version of this file was a narrative. This one is a set of **prompts**: each
section below is pasted into its own new chat, and that chat is finished only when every
thread ID it owns is ✅ in `RERUN_PLAN.md` §13.26 with a command output or a commit beside it.

**The register is `RERUN_PLAN.md` §13.26.** It has all 51 threads, one owner chat each,
12 already closed. Nothing here restates it.

---

## A. THE RULES EVERY CHAT INHERITS

- **Fixes, not findings.** A documented issue that is still broken is a failure. Never
  advise stopping, pausing or deferring — find a defect, FIX it, prove it, push it.
- **The author has no shell access from your side.** Everything runs on ARC at
  `/data/stat-cadd/scat9264/qsar_qm_models`, reached only by them running
  `bash scripts/pull_safely.sh` against a commit you have pushed. **A fix that is not
  pushed does not exist.** Branch: `additional_reps`.
- **Ask the server freely, but batch.** You cannot see the cluster; guessing job state is
  how the last session produced a launch log with the laboratory runs mislabelled. Send one
  block covering several threads, never one command at a time. Prefer raw
  `sacct` / `squeue` / `ls` / `tail` over anything you just wrote. Run
  `python scripts/test_slurm_status_tools.py` locally before sending anything that uses
  `failed_tasks.py`, `measure_walls.py`, `run_status.py` or `slurm_jobs.py`.
- **Never send a command whose output you cannot act on.** If the answer would be "then I'd
  need to look at something else", fold that something else into the same block.
- **Anything that changes the science is the author's. Anything a measurement can settle is
  yours.** Do not hand back a decision you were told to make; do not take one you were not.
- **Plain English.** One fact per sentence. Lead with one verdict sentence. Never open a
  message with a correction of your own last message. The author must finish the message
  knowing what to DO.
- **Never quote a number from memory.** Trace it to a file or a command, or say "not checked".
- **Banned words: "arm", "stage".** Say "noise type" / "condition"; for the run design say
  *the screen / the main grid / the deep run / the uncertainty runs*.
- **Say AUC_norm**, never "retention area". **Say PDV**, never "descriptor vector".
- **Representation is a FACTOR.** Never collapse the study to one. The set is exactly PDV,
  MHG-GNN, Avalon, ECFP4, ChemBERTa, Sort & Slice. Never Ridge. mol2vec is deleted.
- **One state document.** Add to `RERUN_PLAN.md`. Never start a new status file.
- **`paper.tex` is never edited** — it is a read-only download from Overleaf. Paper changes
  become replacement text in `PAPER_REVISION_GUIDE_FINAL.md`.

## B. HOW A THREAD CLOSES

Tick it in `RERUN_PLAN.md` §13.26 with **a command output or a commit hash**. Not with
"should be fine". Not with "the cause is fixed in code" — *the cause being fixed* and *the
tasks being put back* are two different threads and both must close.

If a chat finishes with an open ID and no new evidence for it, that chat has failed,
whatever else it produced. Say so plainly rather than closing it quietly.

---

# THE SEVEN PROMPTS

---

## PROMPT 1 — C-SELECT: the deep run may be running the wrong models

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T02 T03 T08 T11 T33 T52 T53 T56**. The deep run
> (`12986314`–`12986332`) and both censoring runs are executing **right now** against
> `deep_run_pairs.json` and `censoring_pairs.json`, both of which still say
> `provisional: true`. Every task reads those files when it starts, so narrowing them is
> free at any moment and widening them means resubmitting those indices.
>
> **T52 is the blocker and it is yours to fix.** `copy_zero_rows.py` stopped and copied
> nothing: the clean row for `chemberta / dnn_bnn_full_mve` under `grouped_wider` and
> `grouped_shifted` disagrees with the same model's clean row under `gaussian`, replicate 1,
> on all five accuracy columns. The script's premise is that the clean run is bit-identical
> whichever condition labels it, and that premise was only ever measured on random forest.
> Nothing in `models/models.py` or `scripts/process_and_train.py` sets
> `torch.use_deterministic_algorithms` or `torch.backends.cudnn.deterministic`, so a torch
> model is not bit-identical across two nodes at the same seed. **Get the size of the
> difference before deciding anything** — 1e-7 is nondeterminism and the copy is safe;
> 1e-2 is noise leaking in at level 0 and it is a far bigger problem than the copy.
> Then fix the script so one model's disagreement refuses that one configuration instead of
> stopping the copy for all nineteen.
>
> **T53:** the selector takes its model count from the file named by `--out`, so sending it
> to a scratch path silently read 4 models against a queue of 6. Re-read with
> `--n-models 6`. Fix the fallback so it reads the queued file regardless of `--out`.
>
> **T02/T33:** the corrected reading no longer supports two of the six queued models. `rf`
> is not the most noise-tolerant model on any representation — `ngboost` tops 7 of 9 cells —
> and the least-tolerant slot moves from `het_gp_rbf` to `dnn_vbll_hetero`, which is least
> in 8 of 9. Put that in front of the author as a change to confirm or reject. **It is
> their decision, not yours.**
>
> **T56:** `mlp_bnn_full_mve` on ChemBERTa reads AUC_norm 0.9958, the highest number in the
> screen, from a cell holding 2 of 7 conditions. Check whether AUC_norm is being computed
> over a short level ladder there, because it feeds the ranking.
>
> **T08:** the screen's Sort & Slice rows were produced without the three-molecule exclusion
> and the repaired main-grid rows will carry it. Different molecule sets. Either every table
> says so or the screen's Sort & Slice tasks are re-run — cost both and put it to the author.
>
> **T11:** `run_status.py` counts a skipped selection-gate exit as done, so "deep run 364
> done" is mostly seconds-long skips. Fix it to report them separately, as `measure_walls`
> already does.
>
> **T03:** three of the four decisions deferred in §13.17 B have had their trigger fire.
> Read them off and put them to the author with the numbers beside them.

---

## PROMPT 2 — C-RESUB: 174 failed tasks, none put back

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T12 T15 T17 T24 T25 T54 T55**. As of `09ff59e` there are **174 failed
> tasks and not one resubmission line has ever been emitted for any of them.** The count was
> 48, then 51, then 58, then 104 for Sort & Slice alone — that is one number growing as more
> tasks reach the fault, not four documents disagreeing. Always read the live number.
>
> The breakdown: **104** Sort & Slice (`ValueError: Sort & Slice produced an all-zero count
> vector for N`), **20** `gauche_rbf` (the out-of-fold row-count guard), **50** with no cause
> classified at all.
>
> **T54 is the blocker and it is yours to fix.** `failed_tasks.py --emit-sbatch` printed
> nothing, because every task classes as `FAILED` and the tool prints no sbatch for `FAILED`
> on the rule that resubmitting an unfixed cause gets the same exit. That rule is wrong for
> 124 of them: Sort & Slice was fixed in `62f1fe2` and `gauche_rbf` in `c223ec3`. Give the
> tool a way to say *"cause fixed at commit X, emit the resubmission"*, tied to the commit so
> it cannot be asserted by hand. Add the case to `scripts/test_slurm_status_tools.py`.
>
> **T55:** 50 tasks have no cause because the log path is wrong — it looks in
> `slurm_scripts_validation_rerun/`, but laboratory jobs `cd tests` inside the KIRBy
> checkout. **The 25 hERG tasks of T17 are inside those 50 and are still unclassified.** Fix
> the path, then classify them.
>
> **T25:** `gauche_rbf`'s twelve main-grid failures have a fixed cause (`c223ec3`) and
> **nobody has re-checked whether they survive the fix.** Do that before anyone discusses
> dropping the model. It has now landed on the screen under gaussian — ECFP4 0.9387, PDV
> 0.9399, ChemBERTa 0.9503 — so the "it has never run" argument is dead.
>
> **T17:** the 25 hERG tasks are the missing-cache deaths of 2026-09-02. The cache is present
> and loading 1,415 molecules. Nothing is wrong with them; they need putting back, and have
> needed it across three sessions.
>
> **T24:** Sort & Slice's exclusion is written and tested. Resubmit the 104 — but regenerate
> the scripts first so they carry the new walls from `model_hours.json`, and coordinate with
> C-QUEUE so you are not both regenerating.
>
> **T15:** `qm90_mlp_bnn_full_mve` has 16 of 18 tasks still queued and
> `qm90_dnn_bnn_full_variational` 10 of 18. Neither is a failure and no lever covers them.
> Find out why they are not starting.
>
> **T12:** `slurm_jobs.py` labels the two uncertainty submissions "uncertainty, the three"
> and "uncertainty, the four". Those are coined codes and the author banned them. Rename.
>
> `scripts/slurm_jobs.py` knows the submission-to-directory-to-script mapping — the QM9 job
> name is `qm91_rf` and its script is `qm9_s1_rf.sh`, and they are different.

---

## PROMPT 3 — C-QUEUE: every wall and every memory tier is wrong, in both directions

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T06 T07 T18 T26 T27 T28 T35 T36**. Nothing here is scientific. All of it
> is queue throughput, and one item of it is the study's only irreversible failure class.
>
> **T26 first, because it is the only one that destroys work.** `qrf`'s longest task anywhere
> is 20:52 and the generator asks 1-02:59 on the main grid — 1.29× headroom on a measurement
> that is already the worst of fifteen tasks. `val_svm` in laboratory censoring asks 1:00:00
> against 0:54 measured, 1.13×. Everything else in the study has 4.9× or more. A job killed
> at its wall has no partial credit and writes no row, and `scontrol` cannot raise a limit
> for its owner — so this needs the generator's wall rule changed and the unrun indices
> resubmitted.
>
> **T06:** the QM9 generator now reads `model_hours.json` at a 2.0× margin (`62f1fe2`). **The
> laboratory and uncertainty generators still use the hand-written `hours_per_110` guesses**,
> so every laboratory resubmission still goes out over-asked. Port the rule.
>
> **T28/T18:** `qm92_ngboost` asks 22 days against under four measured and
> `qm92_mlp_bnn_full_mve` 13 days 18 hours against 8:38. A three-week request fits almost no
> backfill gap, which is why they sit on Priority while four-hour jobs run past them. Cutting
> a `TimeLimit` on a pending job keeps its submit time, so it costs no queue position. The
> corrected `scontrol` list has never been emitted or applied — the previous session's last
> technical message told the author not to run the old one and nothing replaced it.
>
> **T27/T07:** the memory tiers rest on **61.2 GB from a different study**. The worst peak
> across ~1,400 completed tasks of *this* study is 4.1 GB — and **re-measure that, do not
> quote it**; `measure_walls.py` prints a peak GB column per model per submission. 64G is the
> author's settled floor and the uncertainty runs hold 96G; neither changes. What is open is
> the 96G tier for everything else, and the proposal is a **carve-out, not a flat drop**:
> move to 64G everything that has measured, but keep `gauche_rbf` and the deep run's
> `ngboost` at 96G because neither had a completed task to argue from. Check whether that is
> still true before repeating it.
>
> **T35:** `12980573`–`12980591` went in at `--mem=32G`, below the author's floor. No document
> records whether the `scontrol MinMemoryNode` lines were ever run. One command answers it.
>
> **T36:** submission 7 (`12986352`–`12986370`) went in without `--conditions student_t_nu5
> outlier_p10 laplace`, so the laboratory depth run repeats three conditions the breadth grid
> already runs. The runner replaces its own rows so the numbers are right either way — it is
> 327 tasks of queue time competing with work that has never started.

---

## PROMPT 4 — C-UNC: 378 tasks queued, none started, and they may be the wrong models

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T22 T23**. This is the whole of the study's uncertainty evidence and
> there is currently nothing in it.
>
> **T23:** twelve arrays, `12986390`–`12986401`, 378 tasks, submitted 2026-09-06, **not one
> task has begun.** `unc_*` is its own pipeline so no other run can size its walls — the wall
> tool reports "this model has run NOWHERE" for all twelve. Find out why nothing starts
> before anything else; a wall or memory request that no node can satisfy is the usual cause.
>
> **T22 is the one that decides whether the run is worth anything.** Of the four settled
> uncertainty pairs, only `qrf` reports both the aleatoric and the epistemic half per
> molecule. NGBoost has no model-uncertainty half at all — one fit, and its seed reaches only
> `minibatch_frac` and `col_sample`, both pinned at 1.0. The Gaussian process and the
> variational network each report one data-noise number per fit, not per molecule.
> **The three models that have both halves per molecule — `heteroscedastic_gp` and the two
> variational networks with a noise head — are on neither uncertainty run's pair list**, and
> the QM9 generator gives them `-u True` with no `--oof-folds`, so they write test rows whose
> injected noise is exactly zero.
>
> Put that to the author as a decision with the cost of each option priced in tasks and
> hours. **Do not set the pair list yourself** — a previous session said it would and the
> author's answer was *"I don't trust you to make unilateral decisions."*
>
> Background you will need: `RERUN_PLAN.md` §5.5g–§5.5i settles which models decompose and
> which do not, §3.1c settles how the out-of-fold pass works, and §3.1d records that a
> scaffold split leaves the grouped conditions' held-out shape flat and truthfully so.

---

## PROMPT 5 — C-KIRBY: the other checkout, where every laboratory result lands

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T10 T37 T38 T39 T40 T41**. The KIRBy checkout is where the laboratory
> grid and the uncertainty runs actually write: `<KIRBy>/results/validation_rerun/` for the
> grid and `<KIRBy>/tests/results/uncertainty_rerun/` for the uncertainty runs, because those
> jobs `cd tests` and pass a relative `--results-root`.
>
> **T38 is the one that can silently corrupt a result.** An inner fold that raises during the
> out-of-fold pass is caught, skipped with a printed warning, and rows are still written for
> every training molecule with NaN values. The runner's integrity gates count rows rather
> than finite values, so the job exits 0. The only detection is `merge_results.py`, which
> classifies the cell `TRUNCATED_OOF` and then returns normally, exit 0 as well. **An
> uncertainty cell cannot be trusted from its exit code.** Fix the gate to count finite
> values.
>
> **T41:** `slurm_scripts_uncertainty_rerun/RUNBOOK.md` names `/data/stat-cadd/scat9264/KIRBy`
> on six lines while the generator it documents defaults to `stat-ecr`, and it tells the
> operator to run a `preflight.sh` that was deleted on 2026-08-28 and that no generator
> writes. These are the operator instructions for the 378 tasks in C-UNC. The runbook lock
> checks conditions, counts and array ranges only, so nothing catches either.
>
> **T39:** `preflight_check.sh` runs `git pull origin main` **inside a submitted job**, into a
> checkout sitting on `similarity-metrics-study`, 261 commits ahead of origin/main and one
> behind, so the fast-forward cannot succeed. In `where_to_submit.sh` the same line is a
> comment and harmless.
>
> **T40:** `where_to_submit.sh --emit` picks the account with `sort -rn | head -1` and no sort
> key, so an exact fairshare tie falls through to a reversed byte comparison and `stat-ecr`
> wins over `stat-cadd`, which this study does not bill to. It also hard-codes
> `part=${EMIT_PARTITION:-medium}`, so "medium" came back as a default and not as a
> measurement. Both are latent only because §13.19 pins account and partition by hand.
>
> **T37:** the fold-independence gate's PASS line overclaims on two axes. Validation labels
> are drawn from a deliberately independent stream — base 42 training, 1337 validation — so a
> molecule that is training in one fold and validation in another carries two different
> corruptions, measured at **791 of hERG's 1,415 molecules, 55.9 per cent**. That separation
> is the settled design. What is wrong is a comment in the runner asserting the opposite of
> what the code does, and a PASS line that reads as proof of something it did not test. 56
> per cent is not a corner case, so it also needs a Methods sentence — draft it for the
> author, into `PAPER_REVISION_GUIDE_FINAL.md`.
>
> **T10:** `model_names.json` holds 24 names across both pipelines while QM9 runs 19. Five —
> `het_gp_tanimoto`, `dnn_bnn_last`, `dnn_bnn_variational`, `mlp_bnn_last`,
> `mlp_bnn_variational` — are models QM9 never runs. Nothing records where they run, or
> whether they are meant to run anywhere. Settle it and write it down, because every future
> "has everything landed" check reads that file.

---

## PROMPT 6 — C-FIG: the analysis is reading an incomplete grid and a retired noise scale

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T13 T21 T42 T43 T44**. Every number here traces to
> `generate_paper_figures_v2.py` or it does not get quoted — add analyses to that script and
> re-run it rather than computing anything on the side. `generate_paper_figures.py` is the
> dead NDS v1 script and `results/paper_figures/` is its stale output; the regeneration
> command is `sbatch run_figures_v2.sh`.
>
> **T42 is the one that makes every ANOVA table wrong.** `run_anova_decomposition(df,
> sigma_value=0.3)` plus two filters at `:2671` and `:2784` hardcode noise level 0.3. **0.3 is
> on the retired raw scale and nobody chose it — it is a default argument.** The settled
> scale is the fraction of the clean training label spread, and the reporting levels are QM9
> 1.0, logD 1.0, hERG 1.0, Caco-2 0.75 (§13.16). It must read `reporting_level()`, and it has
> never been rebuilt for the seven-point ladder at all.
>
> **T44:** `paper.tex` lines 380, 387 and 409 quote R² at sigma = 0.3, taken from that same
> default argument. `paper.tex` is never edited — this becomes replacement text in
> `PAPER_REVISION_GUIDE_FINAL.md`.
>
> **T13:** `13033488` is `slurm_scripts_analysis/run_paper_analysis.sh`, the author's own
> decision report. Sort & Slice is missing from every replicate 1 to 9 of the main grid, so
> one representation of six is absent from everything it reads. Find out whether it has
> already written output and whether that output has to be discarded.
> `check_runs_landed.py` should gate it rather than follow it.
>
> **T43:** the rank-versus-level charts described in §5.4a have an owner and no build. They
> are what answers whether a model's ranking moves with the noise level rather than only its
> score, and hERG's reporting level is supposed to be confirmed against a rank-flip table
> that does not exist.
>
> **T21:** the clean Caco-2 training label standard deviation has never been printed.
> `NOISE_DESIGN.md` §6.4 derives 0.76 of the label spread and §2.12 derives 0.79 and neither
> records the SD both rest on. It is one line of output and it is what C-DECIDE needs to
> settle the Caco-2 anchor. **Print it and hand the number to that chat.**
>
> Two paper insertions are flagged and must not be skipped on the next pass: the dataset-size
> rebuttal to the Fig 8 "smaller datasets" claim, and the ChemBERTa caveat — Methods must
> state that it cannot separate halogens, charges, enantiomers or azole tautomers, with hERG
> 20.1% and QM9 2.71% of molecules sharing a vector. Text is in `RERUN_PLAN.md` §2.8k.

---

## PROMPT 7 — C-DECIDE: six decisions, each in a form the author can answer in one word

> Read `RERUN_PLAN.md` §13.26 and `HANDOFF.md` sections A and B first.
>
> You own threads **T04 T05 T19 T20 T49 T50**. Your job is **not** to make these decisions.
> It is to put each one in front of the author with what each side actually does, the cost in
> tasks and hours, and a recommendation — so that answering takes one word. Spell out what
> each option does before naming any statistic, and where a measurement can settle a
> disagreement, run the measurement instead of arguing.
>
> **T20/T49 — the Caco-2 noise anchor.** This is recorded as both settled and open, which is
> the first thing to fix. §13.18 records the author choosing within-laboratory error, about
> 0.10–0.15 of the label spread, on 2026-09-04. §13.17 A1 and §13.23 C5 both still say the
> choice is not made, and C5 cites the evidence as "§13.20 decision 2" when the numbered
> decisions are in §13.18. **Read the session transcripts at
> `~/.claude/projects/-Users-apunt-repos-qsar-qm-models/*.jsonl` before saying anything was
> never decided** — repo documents lag the conversation, and this has been reopened twice by
> grepping files instead of the chat history. The old anchor was 0.35 log10 from Bentz 2013,
> a between-laboratory figure, which is 0.76–0.79 of the label spread and implies an R²
> ceiling of 0.376 against an observed clean 0.565 — and the extraction has no laboratory,
> source, site, assay or batch column, so between-laboratory variance is not a property of
> this data. **This changes Methods text only. No run changes either way**, because levels
> are set as a fraction of the label spread. Get the printed SD from C-FIG (T21) first.
>
> **T05/T50 — the censoring replicate count.** Two documents price the same cut differently.
> `serverChat.txt` calls cutting censoring from ten replicates to three "the biggest single
> saving on the critical path". §13.14 prices it at **270 runs out of 22,140** and recommends
> keeping ten, because below that censoring carries a different error bar from every other
> condition in the study. **Resolve which is right by measurement, then put one number to the
> author** — they cannot weigh the lever while the two accounts disagree about what it is worth.
>
> **T04 — the three levers for finishing sooner.** They were offered and never chosen between,
> because the author's next message changed the subject. Lever one is the censoring replicate
> cut above. Lever two is dropping `gauche_rbf` from the deep run, worth 17 days off both the
> deep run and censoring — **but its failure cause is fixed in `c223ec3` and it has now landed
> on the screen, so re-price it before offering it.** Lever three is not a lever: `ngboost`
> cannot be dropped, the generator refuses to build a deep run without it.
>
> **T19 — the author's last message is unanswered.** The previous session ended with them
> saying they were overwhelmed, that threads were being created and dropped and mistakes
> fixed all at once, and asking for the plain-language rules to be followed. Nothing replied.
> The register written in response is over a hundred lines of tables and is not an answer
> either. **What closes this thread is one short message: what is being done, by which chat,
> and what they personally need to decide.** Nothing else.

---

## C. WHAT THE FIRST COMMAND BLOCK PROVED, SO NO CHAT RE-ASKS IT

Run at `09ff59e`, 2026-09-07. Full detail in `RERUN_PLAN.md` §13.26.

- **No laboratory task ran under the old noise draw.** 8,384 tasks are on the new one.
  `lab_tasks_on_old_noise.py` was outstanding across three sessions and the answer is zero.
- **174 failed tasks**, not 87: 104 Sort & Slice, 20 `gauche_rbf`, 50 unclassified because
  the log path is wrong.
- **Both variance-head networks are in the screen** with rows on all three representations.
- **`gauche_rbf` has landed on the screen** under gaussian; its failures are confined to
  level 1.5, replicates 7–9.
- **The corrected selector has been run** on 20,224 rows from 342 files — and its reading
  does not support two of the six queued deep-run models.
- **`copy_zero_rows.py` copied nothing**, stopped by a real disagreement in one neural model.
