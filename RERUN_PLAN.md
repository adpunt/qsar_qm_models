# Re-running the paper — the plan

Single working document for rebuilding the NoiseInject results from scratch. Written
2026-08-24 after reading `paper.tex` in full, both noise injectors, both training
pipelines, the figure script's robustness path, and every state document written in the
last week.

**Status of the other documents.** I deleted six of them on 2026-08-24 and restored all six
intact on 2026-08-25 after they turned out to hold decisions this file had missed. Nothing was
lost. They stay until this file demonstrably covers them, and the audit in §11 says where each
one's content went. `immediate_next_steps.md` in particular is **not** superseded — §D4 of it is
the sourced write-up of how the uncertainty decomposition gets built, and it was written to your
instruction.

**What stands beside it.** `NOISE_DESIGN.md` — the sourced specification of the new noise scheme.
Its core rule, that the noise level becomes the amount actually delivered, you approved on
2026-08-21. The condition set was **settled 2026-08-27** (§13.9) and now lives in
`noise_conditions.json` with a gate on each side. **Nothing in it is open.** Laplace was the last
question and the author kept it, in the deep run only, at 720 runs (§4 Decision 4, §13.5).

**Which document owns what.** This file owns what gets *run* and in what order — the staged design,
the replicate counts, the representation set, the job scripts, the analysis, the decisions still
open. `NOISE_DESIGN.md` owns what the noise *is* — the conditions, the algebra, the parameters,
their sources, the level grids, and the checks that are properties of the noise scheme. Neither
restates the other; where a fact is needed on both sides it lives in its owner and the other points
at it. `scripts/check_bib_and_docs.py` fails if the two start restating each other again.

`REVISION_GUIDE.md` is **already gone** — it was scrapped rather than kept, as step 3 of your
process (§0.1) said it would be. Everything that had to survive it is §10b. Do not go looking for
the file; §1927 below describes what was in it.

**Start at §0.6** if you read nothing else — the thirteen failure modes and the assertion that stops each recurring.

**Rules this document obeys.** Nothing is written down without a file, a line, or a run
behind it. Nothing is presented as decided that you have not decided. Where two documents
disagreed, the disagreement is resolved here or flagged as open, never averaged.

---

## 0. The frame, and what is already decided

Rebuilt 2026-08-25 from the session transcripts rather than from the repo documents, after I
twice reopened questions that were already settled. **Read this section before proposing
anything.**

### 0.1 Where this sits in your process

Your five steps, stated 2026-08-20 and repeated since:

1. **Triage everything and re-plan** ← still here
2. Modify and re-run figure generation to match the new plan
3. Write a new revision guide from the plan in (1) and the results from (2)
4. You edit `paper.tex` by hand; I help while you write, and never touch the file
5. Editing

Two consequences I had wrong. `REVISION_GUIDE.md` is **not** a live reference to preserve — you
said it *"has gotten completely out of hand"* and gets scrapped and rewritten at step 3. And
nothing in this plan is a paper edit; those are step 4 and yours.

### 0.2 The argument the paper makes

Your words, confirmed as correct and to be kept in view:

> *"what, if anything, makes a QSAR model robust to noise — and can a model's uncertainty tell
> you when your data is bad?"*

And what you want out of it: *"I want practical results — how do you go about selecting a
representation or model when you suspect your data is noisy? Do you know what type of noise?
What can we say about uncertainty in these contexts?"*

### 0.3 Decisions already made — do not reopen any of these

| Decided | When | The instruction |
|---|---|---|
| **The noise level becomes the noise actually delivered** | 2026-08-21 | *"Fine lets do Noise SD, add it to the next steps doc, make sure to cite kolmar and make sure that ends up in the paper (add a note)"* — this is the dose-matching rule. It is approved, and it must appear in the Methods with the Kolmar & Grulke citation |
| **Censoring is in** | 2026-08-24 | *"OH censoring is fascinating yes definitely include that"* |
| **Keep all six of the old noise types — do not cut them for redundancy** | 2026-08-21 | *"My goal is not to prove these strategies unique. That's not my paper."* Superseded in effect by the redesign, which replaces rather than cuts — but the reasoning matters: redundancy is a *finding*, never a reason to hide data |
| **A skewed-noise experiment on the highlighted models only** | 2026-08-21 | *"let's add a set of experiments — not for all combinations — for models that are being highlighted (and their reps) — try picking from a skewed distribution of values"*. ⚠️ **This row used to say the heavy-tailed types satisfy the request. They do not** — Student-t and Laplace are symmetric, and heavy-tailed is not skewed. The request is met by **censoring** and by the new **grouped-shifted** condition, both one-directional by construction. Corrected 2026-08-26; the full argument is §13.3 |
| **The aleatoric/epistemic split gets BUILT, to industry standard** | 2026-08-20 and 2026-08-21 | *"I'm not dropping it. How do other real papers do it (with evidence, not hallucinated)"* → *"I NEED THESE RESULTS AND I WANT TO IMPLEMENT THEM BY INDUSTRY STANDARDS"* → *"there's some faulty code that must be DELETED and REPLACED. get over it"*. Three separate refusals. See §4 |
| **The uncertainty experiment: three questions, one run** | 2026-08-23 | Approved with *"Okay yes make those changes, smoke test them."* See §3.1 |
| **The three headline numbers, and how they work together** | 2026-08-20 | *"auc_norm needs to take into account baseline performance at higher sigmas. the r2 at sigma=0.6 is a bit of a sanity check on auc_norm. Obviously other noise levels exist, but it catches a few cases."* Not a metric replacement — make the baseline visible beside the retention number |
| **Expected calibration error: removed entirely** | 2026-08-19 | *"add in the revision guide the complete removal of ECE and remove it in the figure generation scripts. Full removal not commenting out"* |
| **Binary physicochemical descriptors: dropped** | 2026-08-14 | *"Let's drop binary PDV"* |
| **The paper is fixed by re-running, never by rewording** | 2026-08-23 | *"There is no rewording, there is only re-running."* |
| **Every table and figure in the paper comes from the figure script** | 2026-08-17 | *"Every single table and figure that is in the paper needs to be derived from the figures script. No exceptions."* |
| **Averaging is permitted only when every level is shown** | 2026-08-14 | *"averaging is fine if all 6 are shown"* — the precise form of the rule |
| **Hold one representation constant, and it is PDV** | 2026-08-19 | Chosen with you, not unilaterally. When comparing models across noise types, one representative representation is held fixed |
| **Do not drop models from the variance decomposition for missing representations** | — | *"Dropping five models from the robustness ANOVA for missing representations is BAD"* — relax the inclusion criterion rather than lose them. And the Bayesian variants were excluded from the **decomposition only**, never from the rest of the results |
| **No new figures** | — | Standing constraint. Changes are panels and columns on existing figures |
| **The ranking table must not average across noise types** | 2026-08-14 | Report the reference type, keep the concordance coefficient for the strategy-independence claim, and highlight any type that behaves differently |
| **The paper is not about choosing a model, and not about choosing a metric** | 2026-08-20 | *"This paper isn't about which metric is the best."* The three numbers assess models from different angles. Watch for drift — §7.3 was edging toward a metric bake-off |
| **The Gaussian process goes into the variance decomposition, on one kernel everywhere** | 2026-08-26 | *"The kernel is not why the Gaussian process was kept out of the variance decomposition. Commit to the radial basis everywhere and it can go in alongside the support vector machine — which is what those jobs were for."* Settled. Both `gauche` and `gauche_rbf` come out of `ANOVA_MODELS_EXCLUDE`; one model, one kernel, every representation. **Conditional on the embedding rescaling fix (§2.8c)** — see §10b.2 |
| **The representation set** | 2026-08-26 | *"Drop SMILES from the list, add Avalon and ChemBERTa."* Final six: PDV, MHG-GNN, Avalon, ECFP4, ChemBERTa, Sort & Slice. Out: mol2vec and one-hot SMILES. Sort & Slice is fixed in place — a colleague's method. See §13.7 |
| **Ten replicates in stage 1** | 2026-08-26 | *"Let's do the 10 reps for stage 1 then."* Takes the whole staged design to 37% of the old one instead of 26% (§13.1) |
| **Both forms of grouped noise are run** | 2026-08-26 | *"For the skewed noise lets go with option C."* Groups get wider errors *and*, as a separate condition, groups get shifted errors. See §13.3 |
| **QM9 leads the Results** | 2026-08-26 | *"qm9 leads the results because it's actually clean data."* It is computed rather than measured, so it has no measurement error of its own to confound the injected noise. The three experimental datasets follow it. A question I asked on 2026-08-26 — whether they should lead instead — was wrong and is closed |
| **The noise conditions, and how many settings of each** | 2026-08-27 | Settled on chat G's measurement (§13.9): Gaussian, both grouped conditions and censoring at full grid; **one** Student-t setting (ν = 5) and **one** Outlier setting (p = 10%) at depth only; Laplace **kept** at depth, settled by the author the same day; ν = 10, ν = 3, p = 1% and p = 5% dropped; the skewed draw never built. Lives in `noise_conditions.json`, enforced by tests on both sides |
| **The between-laboratory variance finding goes in the Background** | 2026-08-24 | Nearly two-thirds of real measurement variance is between laboratories — you asked for this to be written up |

**Three still-open items from the register that are not in §4 and are yours to close:**

- **Is "evaluated by the trapezoidal rule" acceptable in a paper without a citation?** You asked
  for five published examples of the same phrasing, or an admission that it is filler. I have no
  record of that ever being answered.
- **Should Caco-2 stay in the set?** You raised removing it on 2026-08-19. Never resolved. It is
  also the dataset most distorted by the units problem (§10b.4), which cuts both ways — the
  distortion is now explainable rather than mysterious.
- **Are the experimental-dataset accuracies high enough to report at all?** You asked for this to
  be checked against published values for the same endpoints. Not done.

### 0.4 Analyses you asked for that are not yet built

Each was requested explicitly and is still outstanding. They are work items, not questions.

- **A variance decomposition of R² at the reporting noise level**, not only at the level currently hard-coded. *"I may want to add an ANOVA for R2 at sigma=0.6 with gaussian noise."*
- **Rebuilt paired significance tests** — *"the change in all three quantities — clean R², R² at 0.6, retention — per strategy, not pooled."*
- **A side-by-side ranking table** — *"these ranks SHOULD be in the paper, side by side performance and robustness. But it should be one type of noise, and multiple columns representing each model."* One noise type, models as columns, both metrics visible, no single overall winner.
- **A representation-by-noise-type table.** There is still no table anywhere with representation as the aggregation unit, so the representation claims have nothing to check against.
- **The variance decomposition on the experimental datasets**, which has never worked. *"Okay why the fuck do I not have validation ANOVA? Why is it broken? … We cannot proceed until this is finished."*

### 0.5 One thing the redesign changes that you should see coming

You have said twice that you want the noise types to differ: *"I would LOVE to find something
that says different types of noise perform differently in some way."*

Once every type delivers the same amount of noise, the honest answer from the pilots is that
**shape barely matters** — under 0.02 in R² at realistic levels. That is a clean negative result
and it is publishable, but it is not the result you were hoping for.

**What does differ, enormously, is censoring** — twelve times any shape effect, and the only
mechanism that biases labels in one direction instead of scattering them. So the "noise types
behave differently" story survives, but it relocates: it is about censored versus scattered
error, not about the shape of the scatter. That reframing is worth agreeing before the run,
because it changes which figure carries the paper.

### 0.7 Thread register — what is done, built, or open

Rebuilt 2026-08-26 after reading the full session log end to end and re-verifying every
load-bearing claim against the code. **This is the status summary. §13 is the plan that clears it.**

| # | Thread | State | Where it is picked up |
|---|---|---|---|
| 1 | Diagnosis: held-out labels corrupted on QM9 | ✅ found, fixed in code (`9d7db67`), **now guarded by a test that fails if the fix is removed** (chat A), still never re-run | §13 chat H |
| 2 | Diagnosis: six noise types were one type at six strengths | ✅ found, evidenced, **and fixed in Rust 2026-08-26** — the conditions' mean delivered dose now spreads 1.27% on QM9 | §13 chat A ✅ |
| 3 | Noise redesign — specification, literature, local tests | ✅ done, sourced, and the settings **settled 2026-08-27** (§13.9), in `noise_conditions.json` with a gate on each side | §13 chats A, B, G |
| 3a | **Do the uncertainty runs inherit that settled set, or test it?** | ✅ settled by the author 2026-08-27 — inherit the four, add `outlier_p10` (§13.1 item 6) | done, and it is already the generator's default |
| 4 | Assay-error anchors and the blocklist of bad numbers | ✅ done, peer-reviewed, two passes reconciled | — |
| 5 | Gaussian-process kernel question | ✅ **decision stands, its evidence was wrong** — the 0.89 kernel gap was a failed fit, not the features (§2.8f). One kernel everywhere is now better supported, not worse | done |
| 6 | Within-noise-level uncertainty correlation | ✅ **author's fix, and it is implemented** — `within_sigma_unc_noise_rho`, `generate_paper_figures_v2.py:1031-1057`. See §3.5 | §13 chat F |
| 7 | Uncertainty machinery in KIRBy (out-of-fold, recorded noise, confound control) | 🟠 built, 9 defects fixed, **never submitted, never reviewed with the author** | §13 chat F |
| 8 | QM9 job scripts | 🟠 written, superseded by the redesign before they ran | §13 chat G |
| 9 | Uncertainty job scripts | 🟠 written, superseded, and point at a possibly stale checkout | §13 chat G |
| 10 | Parity audit script | 🟠 written; its literals were still being verified when the last session ended | §13 chat E |
| 11 | Noise redesign in the pipelines | ✅ **BOTH DONE 2026-08-26** — Rust (chat A) and Python (chat B). The two are held together by an executable gate: `scripts/crosscheck_injectors.py`, 342 checks on all 133,885 real QM9 labels and real Murcko scaffold groups. Dose spread across conditions: **1.16% in Rust, 0.40% in Python**, against 0.49×–2.00× before | §13 chats A ✅, B ✅ |
| 12b | The pipeline ignored the injector's exit code | ✅ **found and fixed 2026-08-26 (chat D, close-out pass)** — every hard failure in the Rust half wrote to a pipe nobody read, and a run that died partway trained on a noised training split against clean held-out splits. Gate: `scripts/test_failure_propagation.py` (§2.8g) | §13 chat D ✅ |
| 12a | Records could be written short, and an unparseable SMILES crashed the binary | ✅ **fixed 2026-08-26 (chat D)** — all-or-nothing records, a null-pointer check RDKit's binding needed, and the reader now refuses to guess (§2.7) | §13 chat D ✅ |
| 12 | Per-molecule rescaling of learned embeddings | ✅ **fixed 2026-08-26 (chat C)** — storage, widths and standardisation, with a guard that fails if any of the three is removed (§2.8c) | done |
| 13 | Concurrent-task configuration race | ✅ **fixed 2026-08-26 (chat D)** — the configuration file is named per task and the binary has no default path; guarded by `scripts/test_config_isolation.py` (gate 10) and a test in `rust/tests/writer_guards.rs` | §13 chat D ✅ |
| 14 | Aleatoric/epistemic decomposition | 🔴 spec written, not built; 4 further defects found (§5.5) | §13 chat I |
| 15 | The five never-built analyses | 🔴 none built (§0.4) | §13 chat J |
| 16 | Figure script consolidation to one file | 🔴 not started (§5.4) | §13 chat J |
| 17 | Environment: jobs were running in the wrong interpreter | ✅ **found, fixed and confirmed on the cluster 2026-08-26 (chat D)** — `micromamba` has never worked there, so the `MAMBA_EXE` lines in every job script were dead and an unactivated task fell through to the system Anaconda, which has none of the uncertainty packages. Deleted, and the scripts now refuse to start unactivated. `env_test` verified green on a compute node: 21 of 22 model labels build, NGBoost and the quantile forest also fit. The scikit-learn concern was laptop-only. One real failure left, `conformal`, and it blocks nothing (§2.8d) | §13 chat D ✅ |
| 18 | Paper-side fixes needing no compute | 🔴 not started — **deliberately parked**, see §12 | parked |
| 19 | The two documents had drifted apart | ✅ **done 2026-08-26** (chat K). Ownership rule stated, ten disagreements resolved, two of them a document contradicting itself. Guarded by `scripts/check_bib_and_docs.py` | §13 chat K ✅ |
| 20 | The bibliography | ✅ **done 2026-08-26** (chat K) — 25 entries added, a key collision on two different papers split, the rejected-source blocklist made executable. **One line left in `paper.tex`, and it is the author's** (§9.1) | §13 chat K ✅ |
| 21 | Hyperparameter tuning — replacing Optuna | 🟠 **built 2026-08-27, not yet run** — the experiment, the roster/name map and three checks are in; the sweep is priced but waits on decisions 7 and 8. Found and fixed two live defects on the way in: NN-β's Bayesian variants could not run at all (§5.7c), and the tuned-parameter reader would have raised NameError the first time it fired (§5.7g) | §5.7 |

---

### 0.6 The thirteen ways this analysis has gone wrong, and the guard for each

**This is the most important section in the document.** Almost every wrong number in the paper
came from one of thirteen failure modes, not from thirteen unrelated bugs. The findings they produced
are being regenerated and are not worth carrying forward. The failure modes are, because every one
of them will recur silently in the new analysis unless something actively stops it.

Each guard is an **assertion that fails the run**, not a note in a document. A guard nobody
executes is not a guard.

| # | The failure mode | What it produced | The guard |
|---|---|---|---|
| 1 | **Pooling across a dimension that should have been conditioned on** | The whole per-sample uncertainty claim. The correlation pooled every noise level *and* every noise type, so it measured the population trend and reported it as per-molecule detection | Every correlation carries its conditioning set in the output row. Assert that each group is exactly one (dataset, model, representation, noise type, level, replicate) cell before computing anything |
| 2 | **Reconstructing a quantity instead of recording it** | The injected noise was recovered by fitting a line. At zero noise the residuals were floating-point rounding, whose size grows with the label — which is where uncertainty is largest. The zero-noise control therefore showed a *stronger* signal than the real levels | Record ground truth where it is drawn. Assert the zero-noise control is **exactly** zero, not small |
| 3 | **Averaging replicates before computing a derived quantity** | Retention computed on an averaged curve, so no robustness number has a spread. Folds averaged before integration on the experimental sets, which forces the unexplained share to zero | Compute the derived quantity **per replicate**, then aggregate. Assert more than one observation per cell before any variance decomposition |
| 4 | **Printing a ratio without its denominator** | "Robustness is decoupled from accuracy" — which is arithmetic, since the metric divides the baseline out. A model with a weak baseline scores well by having less to lose | No ratio is ever printed without its components beside it. Enforce it in the table writer, not in the caption |
| 5 | **A knob that means different things in different conditions** | Six noise types that were one type at six doses. Every apparent difference in shape was a difference in amount | The delivered amount is measured and written to every row. Assert it is equal across noise types at a given setting, to a stated tolerance |
| 6 | **A cut-point expressed in the wrong units** | The threshold rule fired on raw electronvolts, so it caught 99.99925% of molecules and became Gaussian noise at double strength | Any cut-point is expressed relative to the label distribution. Record the affected fraction on every row and assert it is neither near zero nor near one |
| 7 | **Scoring a molecule with a model that fitted it** | Would have measured memorisation and called it uncertainty — a Gaussian process has zero posterior variance at its own training inputs | Out-of-fold scoring, scaffold-grouped. Assert out-of-fold error exceeds in-sample error |
| 8 | **A filter that is not random with respect to the question** | Whole replicates below an accuracy floor are deleted, and the variance decomposition drops more on top. Both are undeclared, and they bias unstable configurations' retention upward | Declare both in the Methods. Report the headline result with and without, and fail the run if the two disagree in direction |
| 9 | **Silent no-ops** | Uncertainty written only if the zero-noise level is present, and only for one noise type unless a flag is passed. A guard evaluated before the thing it tests for exists, so a placebo column is always blank | Assert expected row counts per condition after every write. A condition that produces no rows fails loudly |
| 10 | **Two implementations of one specification drifting apart** | The two injectors disagreed on a constant and on how cut-points are computed, unnoticed for the life of the project | Cross-check them on the same labels as a launch gate |
| 11 | **Spending compute on a condition that cannot answer the question** | A noise type whose scale is constant on a given dataset makes the "where is the noise" question undefined there | Preflight computes the scale on the real labels and refuses to queue degenerate conditions |
| 12 | **One number, two names** | The same values captioned with two different metrics on facing pages; a retired metric surviving in the Conclusion as *the* headline | One script generates every number and its caption. See §5.4 |
| 13 | **Shared mutable state between concurrent tasks** | Every task read and wrote one configuration file in one directory, including the identifier that selects its data files, so two tasks at once could overwrite each other's inputs silently (§2.8a). ✅ **fixed 2026-08-26** | The path is unique per task and the binary has no default. Gate 10 runs two tasks concurrently and asserts each read its own configuration |

**The pattern underneath nearly all of them:** a quantity was computed at the wrong granularity, or
recovered instead of recorded, and nothing checked. The re-run fixes the specific instances. Only
the assertions stop the next instance.

**Where they live.** Guards 1, 3, 4, 8, 9 and 12 belong in the figure script. Guards 2, 5, 6, 7,
10, 11 and 13 belong in the pipelines and the preflight, and must pass before any cluster time is
spent (§8).

---

## 1. Where things actually stand

| | |
|---|---|
| Every QM9 number above zero noise | **invalid** — held-out labels were corrupted (§2.1) |
| The fix for that | **in the code, committed** (`9d7db67`), **never run** |
| Every derived QM9 table in `results/paper_figures_v2/` | dated **8 July**, six weeks before the fix |
| The three experimental datasets (LogD, Caco-2, hERG) | **clean** — that pipeline never had the bug |
| The noise scheme itself | **being replaced** — `NOISE_DESIGN.md`, not signed off |
| The uncertainty experiments | **built and tested**, nothing submitted |
| The paper text | built on the invalid data, and internally inconsistent besides (§9) |

Two things are worth saying plainly before the detail.

**The paper's spine survives.** The central claim — that the choice of model, not the choice
of molecular representation, decides how well a QSAR model tolerates label noise — was
reproduced on the clean experimental data (model 71.4% of variance, representation 10.8% on
LogD, per `RESULTS_REWORK.md` §3.1). What has to be rebuilt is the evidence, not the argument.

**The re-run is not just a repeat.** Four separate things change what gets injected, what gets
trained, and what gets recorded. Getting any one of them wrong means running twice, so they all
have to land in one commit.

---

## 2. The defects, and what each one costs

Everything in this section was read from the code this session. File and line numbers are
current as of commit `6099659` plus the working tree.

### 2.1 Held-out labels were corrupted — fixed, not yet run

The map of injected noise is keyed by *training* position, `0..train_count`. The writer that
produces the memory-mapped files restarted its counter at zero for each of the three splits,
so validation molecule 7 and test molecule 7 each received the noise drawn for **training**
molecule 7. Held-out labels were corrupted, and the corruption was attached to the wrong
molecules.

Fixed by an `apply_noise` flag on `write_data` (`rust/src/main.rs:623`), passed `true` for
training and `false` for validation and test (`:1028`, `:1055`, `:1082`). The comment at
`:742-748` records the old behaviour.

**Why this one was nastier than an ordinary bug.** Corrupting held-out labels does not add
scatter; it adds a smooth downward bias that looks exactly like a robustness curve. If held-out
labels carry noise of variance *v*, a perfect model still scores R² = 1/(1 + v/Var(y)) — with no
model dependence at all. Fitting that curve to the contaminated results gave 1.288 against a
label standard deviation of 1.293 (verified this session from `data/QM9/raw/gdb9.sdf.csv`:
133,885 molecules, gap in electronvolts, mean 6.833, SD 1.293). Within 0.4% of pure artefact.

The zero-noise results were never affected — `process_and_train.py:1609` sets the noise flag to
`s > 0`, so at zero the whole path is off.

### 2.2 The six noise types are one noise type at six doses

Documented and evidenced in `NOISE_DESIGN.md` §0 and §5. The short version: at the same nominal
setting the six types deliver between 0.49× and 2.00× the same amount of noise, and their
apparent severity ordering is entirely explained by that, with rank correlation ±1.000 for all
eleven models individually.

One of them is outright degenerate. The threshold rule fires when the raw label exceeds 1.0, and
it is applied to electronvolts. **Verified from the raw data: the smallest HOMO–LUMO
gap in QM9 is 0.669 eV, and 99.99925% of the 133,885 molecules clear the cut** — ten molecules
in the whole dataset escape it. (`NOISE_DESIGN.md` §2 owns this figure and carries the same
values; it is repeated here only because the argument of this section depends on it.) On QM9 threshold noise is plain Gaussian noise at double strength.
Had the pipeline stayed in Hartree the same cut would have caught nothing at all. (Two earlier
documents disagreed here, quoting 2.08 eV and 100%; that was the first 10,000 molecules in file
order, and the pipeline samples at random from the whole set. The whole-set figure is the one
that applies.)

**Consequence for the re-run:** the amount of noise delivered must become a *result* that each
noise type solves for, not a knob each type interprets differently. That is the core of
`NOISE_DESIGN.md`.

### 2.3 ✅ CLOSED 2026-08-26 — the gate now exists

**The check is `scripts/crosscheck_injectors.py`** and it runs both implementations on the same
labels, the same target, the same scaffold groups and the same seeds, then compares realised dose,
unit dose, the fraction of labels off by more than three times the dose, the median error, the
worst-hit 5%'s share of the noise energy, and the realised affected-molecule fraction. It exits
non-zero on any failure and is the artefact chat H wires into the preflight (§8 gate 2).

**It cannot be element-wise, and that is deliberate.** Rust's `StdRng` and numpy's generator produce
different streams, so identical draws are impossible; a check written that way would fail for a
reason that does not matter and would then be turned off, which is worse than no check.

It found four real disagreements before it passed, none of which would have been noticed otherwise:

1. **The `effective_n` formula was wrong for grouped-shifted, in BOTH implementations** — it used
   the group *count* where the group term is averaged over *molecules*, so a few large groups
   dominate. On real QM9 scaffolds the effective count is **189 against a count of 30,313**, a factor
   of 160, and the dose tolerance derived from it was 0.79% where it should have been 9.6%. Fixed in
   `NoiseInject/noiseInject/core.py` and `rust/reference/noise_arms.rs`. 🔴 **Still to apply in
   `rust/src/main.rs`** — ✅ applied 2026-08-26, see §2.3a.
2. **Censoring at a fraction of zero reported the largest label as the assay limit.** There is no
   limit at the clean baseline. Fixed in the reference.
3. **Censoring encoded its fraction twice** — once in the condition name, once in the level — so a
   "level 0" run still clipped and the clean baseline was not clean. Now one condition whose level
   *is* the fraction, and the name is derived, matching chat A.
4. The old Python dose solver matched the **first** moment; see §2.3b.

The original point stands and is why the gate is permanent: two independent implementations drifted
apart on a constant and on how cut-points are computed, and nothing noticed for the life of the
project.

### 2.3a ✅ APPLIED 2026-08-26 — the effective group count, in `rust/src/main.rs`

`effective_n` for `GroupedShift` (`rust/src/main.rs:757-767`) divides `rho²` by the number of
scaffold groups. The group-level term is averaged over **molecules**, not over groups, so the right
denominator is `(Σ n_g)² / Σ n_g²` — the effective group count. Measured on the real QM9 scaffold
assignment: 189, against a group count of 30,313.

The consequence is not a wrong number in a results row; it is a **gate that fails runs it should
pass**. `dose_tolerance` derives its tolerance from `effective_n`, so the pipeline currently expects
grouped-shifted's realised dose to land within 0.79% when its true sampling spread is 9.6%.

The corrected form is in `rust/reference/noise_arms.rs` (`effective_n`, the `GroupedShifted` branch)
and in `noiseInject.NoiseInjectorRegression._effective_group_count`.

✅ **Applied to `rust/src/main.rs` 2026-08-26,** and the three implementations now agree. Chat B was
right about the consequence and understated how bad it was: on the 4,000-molecule fixture the old
formula demanded 3.4% where the condition's own spread is 6.85%, so the gate was passing **only by
luck of seed 42** and would have failed intermittently, by seed, once the real run started. That is
the worst way for a gate to be wrong — it looks like a data problem, not a gate problem.

Guarded by `grouped_shift_precision_uses_the_effective_group_count` in `rust/tests/noise_gates.rs`,
on a fixture whose groups are deliberately lopsided so the raw count (20) and the effective count
(10.2) cannot be confused. It asserts the answer matches the effective count, asserts it does *not*
match the raw count, and then runs eight seeds through the flat-dose gate to confirm the tolerance
admits the condition's real spread. Falsified by reverting to the raw count.

### 2.3b The Python dose solver matched the wrong moment

`calibrate_sigma` (`NoiseInject/noiseInject/calibration.py:16`), reached through
`KIRBy/src/kirby/noise_spec.py:124-141`, binary-searched for the scale that made **mean |Δy| / SD**
hit a target — the *first* moment. R² and RMSE are second-moment quantities, so the second moment is
what has to be matched. At identical root-mean-square noise, mean|ε|/RMS is **0.797 for a Gaussian
but 0.642 for Student-t at ν = 3**, so calibrating that way handed the heavy-tailed conditions up to
**24% more actual noise at the same nominal level** — the same confound as §2.2, in the half of the
study nobody had looked at. It also re-used one injector across all twenty of its iterations, so the
objective it searched drew fresh noise every time.

Deleted. The closed-form solver replaces it: exact, deterministic, and identical to the Rust side.

### 2.3c The original finding, kept for the reasoning

**Not a defect to fix.** The two places where the Rust and Python injectors were found to
disagree — the value-proportional factor and the quantile cut-points — are both in noise types the
redesign deletes outright, and no old result survives the re-run, so there is nothing here to
repair.

It is recorded for one reason only: it is the evidence that **two independent implementations
drifted apart and nobody noticed**, over the whole life of the project. The redesign has to be
implemented twice as well (§3), so the same thing will happen again unless agreement between them
is an enforced gate rather than an assumption. That gate is §8 item 2, and it is the only thing
this paragraph is here to justify.

### 2.4 Labels are standardised using the noisy spread

`generate_aggregate_stats` (`rust/src/main.rs:938-981`) adds the noise, then computes the mean
and standard deviation from the **noisy** training labels (`:959-965`, `:973-978`). Every split
is then standardised with those constants (`:759-760`).

**Be precise about what this does and does not break**, because the earlier note overstated it.
R² is close to unaffected: the test labels and the model's predictions both carry the same scale
factor, so it cancels in the ratio. What it does break:

- Every quantity with a fixed scale attached. The Bayesian networks use a weight prior of
  N(0, 0.1²); the Gaussian process fits a likelihood noise term; the variational last layer
  learns an observation variance. All three are being fitted against a target whose scale moves
  with the noise level.
- **Every reported uncertainty magnitude and every coverage number.** The predicted standard
  deviations are in units that shrink as noise rises.
- The paper's own description. `paper.tex:354` says the threshold cut is applied "on normalized
  data". It is not — noise goes onto the raw label and standardisation happens afterwards.

Fix: compute the mean and standard deviation from the **clean** training labels, before
injection, and use those everywhere.

### 2.5 Seven model families train on the validation split — so 11% of their labels are now clean

`y_train = np.hstack((y_train, y_val))` appears in seven training functions:
random forest and quantile forest (`models/models.py:1383`), support vector machine (`:1458`),
NGBoost (`:1507`), XGBoost (`:1623`), LightGBM (`:1673`), the Gaussian process (`:1738`), and the
conformal wrapper (`:3577`).

The split is 80/10/10, so those models train on 9,000 of 10,000 molecules.

- **Before the held-out fix:** validation carried noise — mis-indexed, but noise. All 9,000
  training labels were corrupted.
- **After the fix:** validation is clean. **One in nine of their training labels is now clean**,
  and only for the tree-and-kernel half of the roster. The neural models, which do not merge the
  splits, are unaffected.

Both states are wrong, and the second one silently advantages seven models over the rest. The
correct design is: training noisy, validation noisy with its own independently drawn noise, test
clean. That also fixes early stopping, which currently selects the neural models against clean
labels — an oracle nobody has in practice.

The experimental pipeline has already fixed its half of this (defect 9 in
`slurm_scripts_uncertainty_rerun/RUNBOOK.md`: validation now carries the same noise type at the
same level, drawn from an independent generator). **QM9 has not.** This is decision 2 in §4.

### 2.6 On QM9, the uncertainty question is now unanswerable by construction

Three facts, each read from code, that compound:

1. Uncertainty is only ever saved for **test** molecules. All thirteen `save_uncertainty_values`
   call sites pass test arrays, and `predict(x_train` appears nowhere in `models/models.py`.
2. Corruption only ever enters the **training** split. So the question "does uncertainty flag the
   labels I corrupted" is being asked about molecules that were never corrupted.
3. The noise per molecule was never recorded. `save_uncertainty_values`
   (`scripts/utils.py:218-223`) reconstructs it by regressing the noisy label on the clean one and
   keeping the residuals. **After the held-out fix the test labels are an exact affine function of
   the clean labels, so those residuals are identically zero.** The column is now dead.

Before the fix it was worse than dead: it held another molecule's noise, and at zero noise the
residuals were floating-point rounding whose size grows with the label — which is exactly where
uncertainty is largest. That is why the zero-noise control showed a *stronger* correlation than
the real noise levels did. The control failed in the direction that manufactures a positive result.

**So Table 7 of the paper (`tab:top_unc_noise`) has no support and cannot be rebuilt from the QM9
pipeline as it stands.** The experimental pipeline can answer it, because it was rebuilt to
(§3.2). Whether QM9 gets the same treatment is decision 1 in §4.

### 2.7 ✅ FIXED 2026-08-26 (chat D) — three ways the record stream could go wrong

Two were known and one was not. All three are now closed, and each is held by a check that
fails if the fix is removed: `rust/tests/writer_guards.rs` and
`scripts/test_record_alignment.py` (gate 8).

**What was actually there, re-read in the working tree rather than taken from this document.**
The write order had moved since this section was first written: the label is written *before*
the ECFP4 block, and the fingerprint is the **last** field in the record. The consequence was
unchanged — the two `continue` statements in the ECFP4 block fired after the rest of the record
was on disk, so the record came out 256 bytes short and every molecule after it was read at the
wrong offset.

⚠️ **Items 1 and 2 describe code that has since been deleted.** ECFP4 moved to Python on
2026-08-26 (§2.13b), taking `prepare_ecfp4` and its null check with it. Read them as history:
they say what went wrong and why, not what is in the tree. Where the protection lives **now** is
stated under each. The property both were protecting — every record the same length, or the read
offset of every molecule after it shifts and the file is silently misparsed — is unchanged and
still what gate 8 tests.

1. **Truncated records.** *Historical fix:* decide the fingerprint before any byte of the record
   is written, via a `prepare_ecfp4` helper that returned 256 zero bytes on failure so the record
   stayed full length.

   **Now:** nothing computes a fingerprint in Rust. `ecfp4_fingerprint`
   (`scripts/process_and_train.py`) computes Morgan radius 2 and **raises** on an unparseable
   SMILES, on a wrong bit count, and on an all-zero row; the block is carried through to
   `rust/src/main.rs` as a fixed `[u8; 256]` (`SmilesData::ecfp4_buf`), which cannot be short.
   `write_data` checks the remaining case — an all-zero block, meaning the Python refusal was
   bypassed — records it in `featurisation_failures_{file_no}.csv`, and the run **refuses to
   finish** unless `--allow-featurisation-failures` is passed.

2. **🔴 An unparseable SMILES did not take the error branch — it killed the process.**
   Found while writing the test for (1), and not previously recorded anywhere. RDKit's
   `SmilesToMol` returns a **null pointer** for a SMILES it cannot parse; the binding
   (`rdkit-sys-0.4.12/wrapper/src/ro_mol.cc:18`) wraps that null in a shared pointer and returns
   it as `Ok`. Only a thrown C++ exception becomes `Err`. So the old `Err(_) => continue` branch
   was unreachable for the ordinary bad-SMILES case: `rdk_fingerprint_mol` dereferenced null and
   the process died with **SIGSEGV, no message and no partial output**. Confirmed by feeding
   `this-is-not-a-smiles` to the built binary — exit 139. Not live on QM9, where every SMILES
   comes from the dataset; live on the ADME sets.

   **Now:** this route does not exist. The only RDKit parse is `Chem.MolFromSmiles` in
   `ecfp4_fingerprint`, which returns `None` rather than a null pointer and is checked on the
   line after. Kept here because the resubmit-failed-indices workflow in the runbook needs it: a
   task that died of signal 11 with no message would be resubmitted straight back into the same
   crash, so signal 11 is worth recognising.

3. **Index drift → truncation.** `read_all_target_values` no longer exists. Chat A replaced it,
   and it is now `read_split_labels` (`rust/src/main.rs:2195`) — one reader for all three splits
   rather than a training-only one. The interim version changed the failure from a shifted index
   to a silent `break`: a short label vector, so the noise plan would cover fewer molecules than
   the file holds. Now a hard error naming the record index and the expected count.

4. **The Python reader guessed.** `parse_mmap`'s bare `except: continue` is gone. These records
   have no delimiters, so once a read has gone wrong the offset is unrecoverable and `continue`
   only stops anyone finding out. It now raises, naming the entry and the byte offset. Three
   assertions were added after the loop: the file must be consumed **exactly**, the parsed row
   count must equal the record count, and **every feature row must be the same width**.

   The width check earned its place immediately. Writing the test showed that a short record does
   not always trip the first two — it can produce rows of differing width that reach `np.vstack`,
   which reports a shape error naming no entry and only when the widths happen to differ. Two
   records misaligned to the *same* wrong width would have gone through silently. The width check
   names the first bad entry before any of that.

5. **🔴 A representation the writer emits and the reader has never read.** Found by the new
   consumption assertion. The Rust writer emits 256 bytes for `morgan`; `parse_mmap` has no
   branch for it and never had one, and `-r` takes any string with no `choices` list. Any run
   including `morgan` was misaligned by 256 bytes per record from the label onward. The reader now
   refuses an unknown representation **by name** (`PARSEABLE_REPS`) rather than silently stepping
   over the wrong number of bytes. ✅ `morgan` itself has since been deleted from the Rust writer
   (2026-08-26) — it was a leftover of the `avalon` work: the Python side never wrote it into the
   input file and it was not in `bit_vectors`. The reader's guard stays, because it is what would
   catch the next one.

### 2.8 The analysis throws away the replicate spread

`calculate_robustness` (`scripts/generate_paper_figures_v2.py:1803`) averages across the ten
replicates *before* integrating the retention curve (`group.groupby('sigma')['r2'].mean()`). Every
ranking table and heatmap fed from that frame therefore has one number per cell and no spread.

The same defect, worse, on the experimental side: `calculate_validation_auc` (`:1318`) averages
the five folds before integrating, leaving exactly one observation per cell — which forces the
residual term of the validation variance decomposition to zero arithmetically. That is why
Additional file 10 reports a residual of 0.0.

The QM9 variance decomposition itself is fine — `run_robustness_anova` (`:2068`) groups by model,
representation and replicate, so its residual is real.

### 2.8a ✅ FIXED 2026-08-26 (chat D) — concurrent tasks overwrote each other's configuration

**Found 2026-08-25. Nothing in any document recorded it, and the run instructions actively
triggered it.**

`process_and_train.py` wrote `config.json` with a **relative** path and `rust/src/main.rs` read
`config.json` with a **relative** path. Every job script does `cd scripts` first, and the job
scripts are array jobs the runbook tells you to submit at four to six concurrent tasks. So every
task on a node read and wrote **one shared file**.

What the shared file carries: `sample_size`, `train_count`, `test_count`, `val_count`,
`molecular_representations`, whether noise is on — **and `file_no`**. That last field is the
serious one: the binary uses it to choose which memory-mapped files to open and rewrite. If task
A's binary read task B's configuration, **it opened and overwrote task B's training data**. Not a
wrong number in one cell — two tasks silently corrupting each other's inputs, with no error
raised by either.

It also fits the evidence. 245 whole replicates were deleted by the catastrophic-run filter,
concentrated in specific model and representation pairs — the signature of a record stream read
at the wrong offset, which is what happens when the representation list in the configuration does
not match the file being read.

**The fix.** The configuration is written to `config_{file_no}.json` and the path is passed to
the binary as `--config`. The argument is **required and has no default**: a default would let a
stale caller fall back to the shared file, which is the defect. The file is removed in the same
`finally` block that already removes the memory-mapped files, so 110 invocations per task do not
leave 110 files behind. Everything else the binary touches was already keyed by `file_no` — the
mmaps, the scaffold groups, the noise manifest and the provenance file — so nothing else needed
changing. `run_qm_qsar_models.py` had a second copy of the same pattern; it is superseded and
cannot run against the current binary, but it was given the per-task name anyway.

**The check (gate 10).** `scripts/test_config_isolation.py`. Two binaries are launched
concurrently in one directory, each handed its own configuration naming its own files, its own
representation and its own payload byte; each must come out with its own data intact and no
shared `config.json` may appear. A second, instant half greps the tree for any remaining fixed
`config.json`, so the gate fails even without running anything. `--end-to-end` additionally runs
two real `process_and_train.py` tasks side by side.

**🔴 AND THE FIX WAS INCOMPLETE — corrected 2026-08-26 on the close-out pass.** This section
said the memory-mapped files "are already named `train_{file_no}.mmap` and `file_no` is
effectively unique per task, so *those* do not collide. **Only the configuration file has a fixed
name.**" That was wrong, and renaming the configuration file while leaving `file_no` alone fixed
the smaller half of the defect.

`file_no` was `(iteration_seed ^ int(time.time() * 1e6)) & 0xFFFFFFFF`. Array tasks differ by
representation and strategy, **not by seed** — every one of them passes `--random-seed 42`, so
`iteration_seed` is identical across the whole array. That leaves the microsecond clock as the
only distinguishing term, and it is not fine-grained enough: consecutive calls to
`int(time.time() * 1e6)` on this machine return the same value. Measured, for two back-to-back
calls with the same seed and iteration: **14,391 collisions in 20,000 pairs — 72%.**

Two colliding tasks open the same `train_{file_no}.mmap` and rewrite each other's training data.
That is the corruption this section is about, and it survived the fix.

`file_no` is only ever a filename token — every use across `process_and_train.py` and
`models/models.py` is a filename, never a seed — so it is now `uuid.uuid4().int &
0x7FFFFFFFFFFFFFFF`. **63 bits**, not 64: the Rust side types it `usize` and the Python side puts
it in a filename, and clearing the top bit keeps it positive on every path that might read it as
signed. A full grid draws about
4,000 of them; 32 bits would leave roughly a 1-in-500 birthday collision across the run. Measured
after: **0 collisions in 500,000 draws.** A single-task run before and after produces
bit-identical accuracy (MAE 0.5017960652047769 at level 0), confirming nothing but the filename
changed.

**How it was found.** Not by reading — by running two real pipeline tasks side by side, which
only became possible once the injector's exit code was checked (§2.8g). Before that the resulting
panic went to a pipe nobody read.

**Still worth knowing:** whether it ever fired. It needs two tasks between writing the
configuration and reading it at the same moment — a short window per task, but one that opens
once per noise level per replicate, 110 times per task. The deleted-replicate list is on the
cluster; if those failures cluster in time rather than in configuration, that was this race
rather than a modelling problem.

### 2.8b The uncertainty re-run points at the wrong copy of KIRBy

The generator hard-codes `/data/stat-cadd/scat9264/KIRBy`
(`slurm_scripts_uncertainty_rerun/generate_scripts.py:54`), and the runbook repeats that path
throughout.

**125 of the 127 job scripts in the KIRBy repository itself use `/data/stat-ecr/scat9264/KIRBy`.**
Two use stat-cadd. The recorded reason is a move on 2026-05-07, when stat-cadd hit 99.9% of its
quota.

The runbook already notices there are two checkouts and says to confirm which is live. It then
picks the one the evidence says is stale. **Confirm before submitting**, or 504 tasks run against
an old checkout — which, since the whole point of that run is a patched injector, would produce a
full set of results from the unpatched code.

### 2.8c ✅ FIXED 2026-08-26 (chat C) — the learned embeddings were rescaled per molecule

> ⚠️ **CORRECTION, 2026-08-26.** What follows calls this "the largest single defect in the study" and
> blames it for the learned embeddings scoring near zero. **That is wrong.** Those scores came from a
> different bug — the Gaussian process was never told how far apart is "far", so it could not fit at
> all. See §2.8f. The storage defect below is real, it damages the geometry, and fixing it was right.
> It is not what produced the numbers in the paper.

**Found 2026-08-25 from the harvested clean-data results, then confirmed in the code.** ~~This is
the largest single defect in the study.~~ It is a data-preparation bug and it has never been a
robustness result. It damages the geometry — how far apart two molecules sit stops tracking how
different their properties are — and fixing it was right. **It is not what produced the near-zero
scores; §2.8f is.** Measured, the fix changes prediction by less than the run-to-run wobble.

#### What the defect is, in plain terms

mol2vec gives each molecule 300 numbers. Those numbers are meant to be comparable **between**
molecules — number 7 means the same thing for every molecule, on a shared scale. That
comparability is the entire point of an embedding.

The storage step destroys it. For each molecule it finds **that molecule's own** smallest and
largest value and stretches the vector so its smallest becomes 0 and its largest becomes 255.
Every molecule gets a different stretch factor.

```
vec_min, vec_max = vec.min(), vec.max()          # over ONE molecule's vector
vec_uint8 = ((vec - vec_min)/(vec_max - vec_min) * 255).astype(np.uint8)
```

`process_and_train.py:971-975` for mol2vec, `:828-834` for the graph embedding, and
**`:807-813` for ChemBERTa — a third representation, found 2026-08-26 and not recorded before.**
ChemBERTa is built and wired in (`bit_vectors`, `:102`; the build call at `:579`) but has never
appeared in a reported result, which is consistent with it being unusable for the same reason.
Fixing the storage makes a transformer language-model embedding available at no extra
implementation cost — see §13.4.

So a molecule whose values naturally span a narrow range gets blown up; one that spans a wide
range gets squashed. Two molecules with the same *shape* but different *magnitude* come out
identical. Absolute magnitude is discarded, and **a different amount of it is discarded from each
molecule**.

#### Why one kernel survives it and the other does not

The radial-basis kernel works entirely on straight-line distance between two vectors. If the two
vectors have each been stretched by a different unknown factor, that distance means nothing.

The Tanimoto kernel is a ratio of an overlap to a total, so a whole-vector rescale largely
cancels — which is exactly why it still reaches 0.87.

#### The evidence

Same model, same data, two kernels. Ten seed-matched clean replicates per cell, harvested from the
cluster 2026-08-25.

| Representation | Tanimoto kernel | Radial-basis kernel | Standardised before the model? |
|---|---|---|---|
| ECFP4 | 0.823 | 0.820 | binary, no scaling needed |
| SMILES | 0.806 | 0.803 | binary, no scaling needed |
| **Graph embedding** | **0.872** | **−0.016** | **no** |
| **mol2vec** | **0.868** | **0.009** | **no** |
| PDV | — | 0.889 | **yes** |

The two kernels agree to within 0.004 wherever the features are sane. On the two embeddings the
gap is 0.88. And the one representation that *is* standardised is the best-performing cell in the
table.

#### What has to change — and removing the rescaling is not enough on its own

Three things. The radial basis needs all three.

**1. Stop the per-molecule rescaling.** Non-negotiable, and irreversible if skipped: each
molecule's own minimum and maximum are never saved, so the damage cannot be undone downstream.
Store the embedding values as they come out of the model.

**2. Store as 32-bit floats, not bytes.** PDV already does exactly this. If byte
storage is kept for size, the scaling must be computed **per feature across the training set** and
the constants saved — never per molecule.

**3. Standardise per feature before the model sees it.** This is the part that would still be
missing after fixing 1 and 2. Only PDV is z-scored today
(`process_and_train.py:1800-1809`); the embeddings are handed over raw.

Point 3 matters because of how the kernel is configured. It is `gpytorch.kernels.RBFKernel` with
**no per-dimension lengthscale — one lengthscale shared across all 1024 dimensions**
(`models/models.py:1726`, and the model receives the array unscaled at `:1740`). With raw
embedding dimensions of differing spread, the widest few dominate the distance and the rest are
invisible. Standardising is what makes a single shared lengthscale a reasonable choice.

That also explains the control case cleanly: PDV goes through the radial basis
at 0.889, the best in the study, **because it is the one representation that gets standardised.**

#### What this overturns

On the Tanimoto evidence the embeddings carry the best signal of any representation except the
PDV — **better than the fingerprint**. The paper says the opposite everywhere.

All four Bayesian variants fail on exactly these two representations, which is the entire
48-configuration exclusion list. Every claim that learned embeddings are weak, that they cannot
support uncertainty estimation, or that they break neural networks, is now suspect and has to be
re-tested rather than repeated.

**After the fix, re-check:** the exclusion list, the representation term in the variance
decomposition, every embedding sentence in the Results and Conclusion, and the kernel comparison
itself on those two representations — the kernel answer in §10b.2 is established only where the
features are correctly scaled.

#### ✅ What chat C actually did, 2026-08-26

All three changes landed together, because the radial basis needs all three and because the record
layout is only correct if the writer, the reader and the Rust record move at once.

| Change | Where |
|---|---|
| The per-molecule rescaling is gone; each builder returns the model's own values as 32-bit floats, and so do its failure paths, so a molecule that cannot be embedded still writes a full-width record | `scripts/process_and_train.py` — `chemberta_fingerprint`, `mhggnn_fingerprint` |
| The record widened: ChemBERTa 768→3072 bytes, MHG-GNN 1024→4096, read back as float32 | the writer and reader in `process_and_train.py`, and the buffers in `rust/src/main.rs` |
| **mol2vec deleted outright** and **one-hot SMILES refused by name**, checked at the top of `main()` before a molecule is read so a job cannot print a complaint and exit 0 | `process_and_train.py` (`DROPPED_REPS`), `rust/src/main.rs` |
| Per-feature standardisation, fitted on the training split and applied to validation and test with the training constants. The existing block is reused rather than copied; the representations it covers are read off one list, `CONTINUOUS_REPS` | `process_and_train.py` |
| **Avalon added** — 2048 bits packed to 256 bytes, computed in Python and passed through Rust exactly like the other pass-through fingerprints. Binary, so it needs no float storage and no standardisation | both pipelines |
| **ChemBERTa needed no builder** — it was already implemented and wired on the QM9 side and had simply never produced a usable result. §13.7 read as though both new representations were new code; only Avalon was | — |
| Avalon and ChemBERTa wired into the experimental runner, reusing `create_avalon` and `create_chemberta`, which were already written | `KIRBy/tests/alternative_data_noise_robustness.py`, `ALL_REPS` |

**The guard: `scripts/test_embedding_storage.py`.** Five sections, every one of them executed rather
than matched against the source. It writes two molecules whose vectors differ by a known factor
through the real writer and requires the factor to survive the round trip; it reads the Rust buffer
widths out of the Rust source and requires the Python reader to agree; it requires every builder to
return floats on its failure path; it requires the standardisation constants to come from the
training split, caught by the test split's mean *not* being zero. The fifth section runs the
retired storage through the first section's own assertion and requires that assertion to **fail** —
so the guard cannot be quietly watered down until it can no longer go red.

**The Rust half is in commit `d25bcb0`** — it was swept in with chat A's noise redesign, which was
committed while both chats had the same file open. Nothing was lost, but the message on that commit
does not mention it.


### 2.8d ✅ FIXED 2026-08-26 (chat D) — the two Gaussian-process jobs that produced nothing

Both failed on 2026-08-19 (`12822693`, `12822694`), after eight and six minutes. The output
directories were created and are **empty** for all three datasets.

**What the logs show.** Every one of the five folds printed its train and test sizes, then
`ERROR: No results for <dataset>`. Nothing between. That gap is the diagnosis: the per-experiment
progress line at `alternative_data_noise_robustness.py:1342` (that line number is from August and no longer points there) never printed once, so **the
experiment list was empty**. The folds looped over nothing.

The two tracebacks at the end are downstream noise — an empty summary file gets written and then
read back, crashing at `:1285` and `:1301`. Not the cause.

**Why the list was empty.** The representation names are not the problem — `--gp-reps` was passed
`ECFP4 PDV SNS MHG-GNN-pretrained` and the generator produces exactly those keys. That leaves the
import guard at `:118-132`: if `gpytorch`, `gauche` or `botorch` will not import, `HAS_GP` is set
false, **no Gaussian-process experiment is ever added to the list**, and filtering to that model
leaves nothing.

Supporting evidence: the traceback names the interpreter as
`/data/stat-cadd/scat9264/py311-kirby` — a different environment from the one the uncertainty
scripts activate.

**Confirm it in two lines.** The guard prints a warning on the first page of the log, and the
count is printed at `:1310`:

```bash
head -30 slurm-12822693.out | grep -i "warning\|gauche\|gpytorch\|botorch"
grep -m1 "model-rep configs" slurm-12822693.out      # reads "0 model-rep configs" if this is it
```

**Two defects, both now closed (chat D, 2026-08-26).**

1. ✅ **A missing optional dependency silently produced an empty run.** Already fixed in KIRBy
   before this chat opened — `tests/alternative_data_noise_robustness.py` raises rather than
   running five folds over an empty list (commit `333f005`, "fail loudly when a task produces
   nothing"). This is guard 9.

1b. 🔴 **AND THE REAL CAUSE, found on the cluster 2026-08-26: `micromamba` has never worked
   there.** `setup.sh` has always fallen through to its `conda` branch, so the two
   `export MAMBA_EXE=/data/stat-cadd/scat9264/bin/micromamba` lines that opened **every** job
   script in `slurm_scripts_qm9_rerun/` pointed at a file that does not exist. The hook failed —
   and because those scripts run under `set -uo pipefail` with **no `-e`**, the failure stopped
   nothing. A task that also failed to activate carried on in whatever python was on `PATH`: the
   system Anaconda at `/apps/system/easybuild/software/Anaconda3/2022.05/bin/python`, which has
   **no `gpytorch`, no `quantile_forest` and no `ngboost` installed at all**.

   This was never a version clash. It was an interpreter nobody meant to use. The dead
   `MAMBA_EXE` lines are deleted, and the generated scripts now refuse to start if `CONDA_PREFIX`
   is unset or `command -v python` resolves under `/apps/system`.

2. ✅ **The environment is now asserted, and the interpreter is pinned.**
   `scripts/check_environment.py` names the interpreter it is speaking for, prints every relevant
   package version, **constructs** each requested model rather than merely importing its package,
   and additionally *fits* the two that import cleanly and fail on contact. It is wired into the
   job template (`slurm_scripts_qm9_rerun/generate_scripts.py`) so a task dies in seconds rather
   than after five folds, and §1b of the runbook has the copy-paste block that runs it under both
   cluster interpreters and diffs them. The two dead jobs were `--wrap` submissions with no output
   path, so they inherited whatever interpreter was active and left no log saying which; the
   runbook now states that jobs are submitted by script, never by `--wrap`.

**⚠️ The probe told the author to uninstall working packages, and that was my bug.**
Run on the login node on 2026-08-26 the probe reported sixteen model failures and advised
removing the four torch_geometric companions. Every one of those errors was
`failed to map segment from shared object` — the loader could not *mmap* the library, which is
what a CUDA build of torch does on a login node, where `libtorch_cuda.so` and `libcublasLt.so`
are over a gigabyte between them and memory is capped per user. Nothing was missing: `lgb`, `rf`,
`svm`, `xgboost`, `ngboost` and `qrf` all passed, and the companions match their torch
(`2.3.1+cu121` both sides, installed by `setup.sh`).

The probe now classifies loader errors. `undefined symbol` / `Symbol not found` is an ABI
mismatch and still says to remove the package; `failed to map segment` / `cannot allocate memory`
is reported as **inconclusive**, with the instruction to re-run inside an allocation, and exits 3
rather than 0 — nothing is known to be broken, but nothing is confirmed working either, and a
preflight must never report a pass it did not observe. The job scripts already run it on a
compute node, which is the only place its answer means anything.

**✅ Confirmed green on a compute node, 2026-08-26.** `env_test`
(`/data/stat-cadd/scat9264/conda_envs/env_test`, Python 3.10) builds 21 of the 22 job-generator
model labels, with NGBoost and the quantile forest also *fitting*. Every declared requirement is
satisfied; scikit-learn is 1.6.1. The sixteen login-node failures were the mapping artefact
above and none of them was real.

**🟠 One real failure, and it blocks nothing.** `conformal` fails with a true ABI mismatch —
`torchsort/isotonic_cpu...so: undefined symbol: _ZNK3c105Error4whatEv`. `torchcp` imports
`torchsort`, and `torchsort` was compiled against a different libtorch than the installed
`2.3.1+cu121`. It does not block the re-run: the three conformal variants live in
`EXCLUDED_MODELS` in `generate_scripts.py`, off unless `--include-excluded` is passed, because
`GLOBAL_MODELS_EXCLUDE` drops them from every figure. Note that `models/models.py` catches this
one in a bare `except ImportError` and sets `torchcp = None`, so without the probe a conformal
job would have started, run, and failed later on a null reference. If conformal is ever wanted:
`pip install --force-reinstall --no-cache-dir --no-binary :all: torchsort` rebuilds it against
the installed torch.

**A gap the compute-node run exposed in the guard itself.** Five labels the generator can emit —
`conformal_rf`, `conformal_qrf`, `conformal_dnn`, `dnn_bnn_variational`, `mlp_bnn_variational` —
were unknown to the probe, so an `--include-excluded` task would have been stopped by the guard
for the guard's own reason. Added, and `python scripts/check_environment.py --audit-roster` now
cross-checks the probe's roster against the generator's two model tables so it cannot drift
again. Verified to fail when a label is removed.

**🟠 A second finding, now downgraded.** The probe also checks whether each package's own
declared requirements are satisfied, because pip never re-checks that after the fact. On the
laptop three of them are not — but `env.yml` pins `scikit-learn=1.6.1`, which satisfies all
three, so this is very likely a laptop-only problem and the cluster answer comes from §1b of the
runbook. On the laptop:

| package | declares | installed |
|---|---|---|
| `quantile-forest` 1.4.1 | `scikit-learn>=1.5` | 1.3.2 |
| `ngboost` 0.5.8 | `scikit-learn>=1.6,<2.0` | 1.3.2 |
| `torchcp` 1.2.1 | `scikit-learn>=1.5.0` | 1.3.2 |

The quantile forest is not merely unsupported, it is **broken**: it imports and constructs
perfectly and then `fit()` raises `Invalid parameter 'monotonic_cst'`. So the quantile forest and
the conformal models cannot be verified on this laptop at all, and NGBoost is running outside its
supported range. Three of those are uncertainty models, which is where the paper's second question
lives.

Two ways out, and the choice is yours because it changes numbers, not just tooling:

- **Upgrade scikit-learn to ≥1.6.** One environment, everything supported. But tree defaults have
  moved between 1.3 and 1.6, so every forest and every boosted model would need re-running rather
  than merely re-checking — which the re-run is doing anyway, so the cost may be zero if it
  happens *before* launch and nothing after.
- **Downgrade `quantile-forest` to a 1.3.x that supports scikit-learn 1.3.** Smaller, but it
  changes the quantile estimator, and if only the laptop is downgraded it creates exactly the
  cross-environment divergence §3.4 exists to eliminate.

Run §1b of the runbook on the cluster before deciding — this may be a laptop-only problem, and
whether it is changes the answer.

### 2.8e 🔴 LAUNCH BLOCKER CANDIDATE — the Gaussian process SEGFAULTS once the boosting libraries are loaded

**Found 2026-08-26 (chat E), by accident, when the calibration test died with no traceback.**

Importing `lightgbm` **or** `xgboost` and then fitting a plain gpytorch `ExactGP` kills the process
with a segmentation fault — exit 139, no Python error, no stack. Reproduced on this laptop, both
libraries independently, and it is not data-dependent: it fires on random arrays.

```bash
python -c "
import lightgbm, numpy as np, torch, gpytorch
# ...build any ExactGP on ~900x208 and take a few Adam steps..."
echo $?          # 139
```

**Why it reaches the real runs.** `models/models.py` imports `xgboost` (`:24`) and `lightgbm`
(`:29`) at module level and then fits the gauche/gpytorch GP in `train_gauche_model`. The
experimental pipeline imports both and fits a GP too. **Every Gaussian-process task on both
pipelines therefore runs inside a process where both are loaded.**

**Why it has been invisible.** A segfault produces no Python traceback, so a SLURM task simply
stops. That is the exact signature of the two GP jobs in §2.8d, which "ran to completion and
produced nothing". Those were diagnosed as a missing `gpytorch`, which was true of that
interpreter — but this is a second, independent way for the same jobs to die on a host where the
package IS present, and nothing was checking for it.

**The cause is two OpenMP runtimes.** LightGBM and XGBoost link their own; PyTorch links another.
Loading both and then running a threaded linear-algebra kernel crashes.

**What was measured:**

| Setting | Result |
|---|---|
| unset (what QM9 does) | **SEGFAULT** |
| `OMP_NUM_THREADS=4` — what the experimental pipeline already sets at `:3` | **SEGFAULT** |
| `OMP_NUM_THREADS=2` | **SEGFAULT** |
| `OMP_NUM_THREADS=1`, from a clean environment | ✅ fits |
| `OMP_NUM_THREADS=1` **alone**, when `MKL_NUM_THREADS` is already 4 | **SEGFAULT** |
| `OMP_NUM_THREADS=1` **and** `MKL_NUM_THREADS=1` | ✅ fits |
| `KMP_DUPLICATE_LIB_OK=TRUE` | **SEGFAULT** — the usual macOS workaround does not help |

So the experimental pipeline's existing thread pin does **not** protect it, and QM9 has no pin at
all. Note the third row: the experimental pipeline sets **both** `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` to 4 (`:3-4`), and pinning only the first while the second stays at 4 does
nothing. Anyone reaching for a quick thread pin has to set both, and the audit now says which
combination actually cured it rather than testing one and reporting a misleading failure.

#### ✅ FIXED 2026-08-27 in BOTH pipelines — and the earlier entry above understated it

⚠️ **Two corrections to what this section said before.**

**First, both pipelines crash, not just the experimental one.** The probe simulated "no thread
count set" by setting the count to an EMPTY STRING, which some numerical libraries reject outright,
so the QM9 condition returned a spurious ordinary error instead of the crash. Unset properly:

| condition | result |
|---|---|
| no thread count set — **how the QM9 jobs run** | **CRASHES** |
| both pinned to 4 — how the experimental jobs run | **CRASHES** |

**Second, "the real fix is one runtime in the environment" was wrong.** Three repairs were
measured, and the cheapest is neither of the two considered before:

| approach | works | cost |
|---|---|---|
| limit threads for the whole job | yes | every tree fit loses its parallelism — the reason this was rejected |
| reorder the imports so the neural library loads first | yes | free, but silently reopens if anyone reorders imports |
| **limit threads only around the Gaussian-process fit** | **yes** | **11%, and only the Gaussian process pays it** |

Eleven percent because the operation is limited by memory speed rather than cores: 38.5 seconds
against 34.8 for twenty steps on 2,000 molecules. Nothing else in either study is affected.

**Applied** as `GP_DEFAULTS['single_thread_fit']` in the shared spec, so both pipelines read one
setting and it cannot drift. It wraps the fit, the fallback loop, **and prediction** — prediction
solves against the same kernel matrix and can crash the same way — and restores the previous
thread count afterwards. Set it to `False` once the environment is rebuilt with a single threading
runtime; `scripts/server_audit.sh` will say when that is true.

**Verified** with the boosting libraries imported first, exactly as both pipelines do, under BOTH
job thread settings: the fit completes and the thread count is restored.

**Still worth running on the cluster**, because the crash is environment-specific and the fix
should be confirmed there rather than assumed: `scripts/server_audit.sh`.

### 2.8e-ter 2026-08-27 — the same conflict DEADLOCKS as well as segfaulting, and it hit a real run

Found on the laptop while running the roster screen (`scripts/setting_selection_test.py`), not on the
cluster. **One process fitting the neural models and the boosting models in turn stopped dead**: 40
minutes of wall clock against 2 minutes 55 of CPU, 0% CPU, no error, no output, no crash. It had
completed the clean fits — which is where torch first builds its thread pool — and hung on the first
boosting fit after that.

**This matters because it is a new failure mode for a known defect.** §2.8e and §2.8e-bis describe
the conflict as a segfault. A segfault is loud and the exit code carries it. This is silent: a job
that hangs at 0% CPU looks like a job that is working, and on the cluster it would burn its whole
wall-time allocation and be killed by the scheduler with no diagnostic. **A preflight that only
checks for the segfault will pass a job that then hangs.**

Both failures were reproduced in the same session on the same machine: a separate probe that fitted
NGBoost, then the DNN, then the MLP in one process exited 139 after printing all three results — the
segfault, at interpreter shutdown, results intact.

**The workaround that works, and it is the one the screen now uses: one process per model.** The
screen's per-replicate seeds depend only on the replicate index and the level
(`setting_selection_test.py`, `noise_seed = 60000 + rep * 100 + int(level * 10)`; subsample and split
seeded from the replicate alone), so seven separate processes draw the same molecules, the same
split and the same noise. The paired comparison across models survives the split — confirmed by
XGBoost reporting a clean R² of 0.9054 on replicate 0 both inside the combined process and in its
own.

**For the cluster:** the job scripts already put one model per script, so the queue does not hit
this. What is exposed is anything that fits several model families in one process — the preflight,
the parity audit, and any future combined harness.

---

### 2.8e-bis The threading-runtime conflict, measured — and why no import order fixes it

Chat D reproduced §2.8e's silent Gaussian-process death on 2026-08-27 while running the real
pipeline on real QM9, and it is the same defect as a LightGBM hang found the same day. One root
cause, two symptoms.

**What happened.** A real run (2 models, 2 representations, 3 noise levels, 1,000 molecules) sat
for three hours at 0% processor time having written one result row. A stack trace showed LightGBM
stopped inside a thread barrier that never completes. Three OpenMP threading runtimes were loaded
in the one process: Intel's (via PyTorch), and LLVM's twice (via scikit-learn and via LightGBM).

**The measured matrix.** Which library is imported first decides which model dies:

| imported first | fitting LightGBM | fitting the Gaussian process |
|---|---|---|
| neither | **crash** | works |
| LightGBM | works | **crash** |
| PyTorch | **crash** | works |

**So there is no import order that saves both.** Chat D briefly committed an "import LightGBM
first" fix and reverted it on measuring this: it cures the LightGBM half by causing the
Gaussian-process half, which is the blocker the server audit already reports.

**The mitigations that do NOT work**, all measured, all still exit 139:
`KMP_DUPLICATE_LIB_OK=TRUE` (it only silences the warning), and `OMP_NUM_THREADS=1`. This
confirms the server audit's line that pinning threads does not cure it.

**Why every job is exposed even though each job runs one model.** `models/models.py` imports
every backend at module scope, unguarded — torch, torch_geometric, lightgbm, xgboost, gpytorch,
gauche, quantile_forest, torchbnn. So a LightGBM job still loads the Gaussian-process stack, and
a Gaussian-process job still loads LightGBM. Every process carries every threading runtime
regardless of what it was asked to run.

**Two ways out, and they are not equivalent:**

1. **Rebuild the environment so only one threading runtime is present** — what the server audit
   recommends. Correct, and it fixes every model at once. It is also a rebuild of `env_test`,
   which changes library versions and therefore numbers, so it belongs before launch, not during.
2. **Import each backend only when its model is requested.** Smaller, no version changes, and it
   removes the conflict by construction: a LightGBM job would never load the Gaussian-process
   stack. It touches `models/models.py`, which is not chat D's, and it would need the roster
   dispatch rewritten to import lazily.

**Neither is chat D's to choose.** Both are recorded here because the audit's section 5 and this
section are the same defect, and fixing one fixes the other.

### 2.8f ✅ FIXED 2026-08-26 (chat C) — the Gaussian process could not fit, and it looked like a bad result

**This is the defect that produced the near-zero scores everyone read as "learned embeddings don't
work".**

**What it was.** A Gaussian process decides how similar two molecules are from how far apart they
sit. How far counts as "far" is a number the model must be given, and nothing ever gave it one. It
stayed at the library default of about 0.7 for every representation and every run. `lengthscale`
appeared zero times in `models/models.py`; so did `ard_num_dims`.

Real distances between molecules run from 14 to 1,100 depending on the representation. At 0.7 every
molecule looks infinitely far from every other. There is nothing left to learn from, so the fit gives
up and predicts one flat number for everything. That still produces a score, and the score looks like
a weak representation rather than a failed fit.

**What it cost.** `results/gp_kernel_harvest/qm9/` reports −0.0158 for MHG-GNN and +0.0087 for
mol2vec. Both are failed fits. Nothing recorded that, so they were read as evidence.

**The fix, proved in the real pipeline rather than a test harness.** The width now starts at the
median distance between training molecules, and any fit that still collapses is written to the
results as collapsed instead of scored. `process_and_train.py -m gauche --kernel rbf`, 600 molecules,
zero noise (`results/embedding_storage_retest/gp_fix_check.csv`):

| Representation | Width used | R² | Collapsed | Same cell on the cluster |
|---|---|---|---|---|
| MHG-GNN | 36.99 | **0.590** | no | −0.016 |
| Avalon | 13.89 | **0.536** | no | did not exist |
| PDV | 17.06 | **0.493** | no | 0.889 at 9,000 molecules |
| ChemBERTa | 38.59 | **0.474** | no | never ran |

**Three things follow.**

1. **Every Gaussian-process number in the study is suspect, not only the two learned embeddings.**
   Whether a fit worked depended on how far that representation's distances happened to sit from 0.7,
   and nothing recorded whether it had.
2. **The decision to use one kernel everywhere is unaffected, and better supported than before.** It
   was made because the two kernels agree wherever the features are sane. With a workable width they
   agree everywhere — largest gap 0.040 across twelve paired measurements, against the 0.86–0.89 that
   was read as proof the features were unusable.
3. **The claim that learned embeddings are weak must be re-tested from scratch.** It rests on failed
   fits.

**Settings:** `init_lengthscale_from_data`, `lengthscale_probe_n`, `collapse_fraction` in
`models/model_defaults.py`. **New results column:** `gp_collapsed`.

### 2.8g ✅ FIXED 2026-08-26 (chat D) — every guard in the Rust half was decorative

**Found on the close-out pass, after the author asked for another look. This is the most
consequential thing chat D found, and it invalidates the confidence of everything above it that
was said before it was fixed.**

`process_and_run` ran the injector with `subprocess.Popen`, called `communicate()`, printed
stdout and stderr — and **never looked at the return code**. It then reopened the memory-mapped
files and trained on whatever was on disk.

So every hard failure the redesign added wrote to a pipe nobody read: chat A's dose gates and
molecule-identity assertion, the held-out checks, chat D's featurisation abort and truncated-record
error, the configuration that will not open, and a segmentation fault alike. The binary refused;
the pipeline did not notice.

**It is worse than a lost message.** `preprocess_data` renames the rewritten training file over
the original **before** it processes val and test. A run that dies partway therefore leaves the
training split noised and the held-out splits clean — and the pipeline trains and scores on that
combination without complaint, producing numbers that look entirely reasonable and are not.

**Two more layers underneath.** The noise-level handler printed only `if logging:` — off by
default — and then `continue`. A noise level that failed produced **no rows and no message**.
That is precisely the shape of jobs `12822693` / `12822694` (§2.8d), sitting unnoticed in the QM9
pipeline. The per-(representation, model) handler printed but carried on.

**Fixed.** The return code is checked and raises with the injector's stderr attached. The
noise-level handler always reports, with a traceback, regardless of `--logging`, and records the
cell. `main` ends by listing every `(noise level, replicate)` that produced no rows and exits
non-zero, so a run that lost cells cannot be mistaken for one that did not.

**Gate:** `python scripts/test_failure_propagation.py` runs the real pipeline with an injector
substituted for one that always exits 3, and asserts the run stops, names the exit code, reports
the level without `--logging`, and calls its own results incomplete. Verified 2026-08-26: the
pipeline exits 1 with `the noise injector exited 3 for noise level 0.4, replicate 0`.

**What this means for the other chats.** Any claim of the form "the run refuses to finish if X"
made before this fix was true of the *binary* and false of the *pipeline*. Chat H should assume
no Rust-side guard was ever enforced end to end until this commit.

### 2.8h The close-out audit — what 13 reviewers found in chat D's own work

Run 2026-08-26 at the author's instruction ("look over all the code you touched, look for code
you should have touched but didn't"). Eight agents reviewed each changed area, five hunted for
classes of code the same defect could hide in, and **every finding was then handed to a separate
agent told to refute it**. 106 raised, **34 survived**, 72 refuted.

**Defects chat D introduced or left behind, now fixed:**

1. **`PARSEABLE_REPS` omitted `"graph"`**, so the new unreadable-representation guard rejected
   every graph run. `-r graph` puts it in `molecular_representations` and `run_qm9_graph_model`
   passes that straight to `parse_mmap`. A regression introduced with the guard, found by three
   reviewers independently.
2. **Every reader guard was swallowed one frame up.** The per-(representation, model)
   `except Exception` wraps the parse calls *and* the whole model loop, and only printed — so
   making `parse_mmap` raise instead of `continue` converted a silent skip into a printed line
   and nothing else; the task still exited 0 with an empty results file. Six reviewers.
   Data-integrity errors now propagate; the rest are tallied and surfaced.
3. **`check_environment.py` could report a pass it never observed** — `check_pyg_companions`
   returned `not abi`, so a memory-mapping failure left no trace and the script exited 0. That is
   the exact policy the file exists to enforce.
4. **The interpreter check was near-tautological.** `setup.sh:83` prepends `$CONDA_PREFIX/bin` to
   `PATH`, so after sourcing it `command -v python` resolves inside the prefix whichever
   environment was activated. It still catches an unset `CONDA_PREFIX` — the case that actually
   bit us — but it would have passed a task in the wrong environment. The environment name is
   asserted too now.
5. **The zero-vector invariant covered one representation of six.** ChemBERTa, MHG-GNN and Avalon
   each returned an all-zero vector on any exception, unrecorded. The bad case is not one
   molecule: `get_chemberta_model()` is called *inside* the try, so on a compute node with a cold
   cache **every** molecule takes that branch — a whole matrix of zeros, a run that finishes
   normally, and R² near zero at every noise level, indistinguishable from a real finding that
   the representation is uninformative. Now recorded and refused, with the same
   `--allow-featurisation-failures` escape as the Rust side.

**Handed on, not chat D's:**

- The generators emit **retired CLI flags** (`--sigma`, `--noise-strategy`, `--strategies`,
  `--bayesian-transformation last`) and representations the pipeline now refuses (`smiles` is in
  `DROPPED_REPS`; `mol2vec` no longer exists). Every generated task would die at argparse.
  **This is why the stale `.sh` files were deleted rather than left to look usable** — but it
  also means the generator cannot produce a runnable script today. Chat H, gated on §13.1.
- `ecfp4` is still not ECFP4 (§3.4.1), carried through chat D's refactor unchanged, and there is
  still no gate that stops a run producing figures labelled ECFP4 from path-fingerprint features.
- `extract_smiles_from_mmap` (`scripts/extract_and_cluster_for_domains.py`) is a second reader of
  the record format that steps over only three of the representations, so it misparses whenever
  continuous_pdv, chemberta, mhggnn or avalon is present.
- `load_and_split_polaris` returns unfiltered split indices while skipping molecules it could not
  write, so `config.train_count` can exceed the records in the file — which chat D's new hard
  error in `read_train_labels` now turns from silent truncation into an abort.
### 2.8i 🔴 THE ENVIRONMENT REBUILD — one threading runtime, and the roster completed (2026-08-27)

⚠️ **Since 2026-08-27 this is a hard stop for every job in the study, not a warning.** The
per-task guard runs the threading count, and it is now in all three job families (§13.2 chat D,
D3). If the rebuilt environment still exposes more than one threading runtime, **every task
refuses to start** — QM9, the experimental datasets and the uncertainty runs alike — and nothing
runs at all.

That is the intended behaviour: the alternative is the measured hang, where a task holds its
whole allocation, writes no rows and no error, and looks like a queue problem. But it means one
property of one environment gates the entire re-run, so **confirm it immediately after the
rebuild rather than finding out on launch day**:

```bash
python scripts/check_environment.py --deep --validation ; echo "exit: $?"
```

Exit 0 is the only value that lets anything through.

**This is the fix for §2.8e and §2.8e-bis at once, and for the two launch blockers the
server audit reports.** Section 2.8e-bis showed there is no import order that saves both
LightGBM and the Gaussian process. This removes the conflict instead of choosing a victim.

#### The claim, measured on one machine on 2026-08-27

Both interpreters on this laptop, same probes, same session — the LightGBM fit from the
audit's launch-blocker list, and the Gaussian-process fit from `server_audit.sh` section 5:

| interpreter | distinct OpenMP runtime **files** | LightGBM fits | GP fits after lightgbm+xgboost |
|---|---|---|---|
| the system Anaconda | **4** | **SEGFAULT** (−11, no traceback) | **SEGFAULT** (−11, no traceback) |
| `env_test`, built from conda-forge | **1** | ✅ | ✅ |

The four in the system Anaconda are exactly the ones the audit named: Intel's inside
`torch/lib/libiomp5.dylib`, LLVM's inside `sklearn/.dylibs/libomp.dylib`, a third that
LightGBM and XGBoost link, and a fourth under `functorch`. **Both blockers appear together
and disappear together, and the only thing that changed is the number of runtimes.**

#### Where the extra runtimes actually come from — not conda

A real `linux-64` solve of `env.yml`'s conda list gives `_openmp_mutex 4.5 7_kmp_llvm` and
one `llvm-openmp`, and conda-forge's `llvm-openmp` ships `libgomp.so.1` and `libiomp5.so`
as **symlinks to `libomp.so`** — three names, one file. conda cannot produce the defect.

Every extra runtime arrives on a **PyPI wheel installed over the top of a conda package**:
torch, scikit-learn, lightgbm and xgboost wheels each bundle a private copy. That is not a
theory about this environment, it is its history — the live `env_test` ran a pip wheel of
`torch 2.3.1+cu121` while `env.yml` said conda `pytorch 2.5.1`, and `setup.sh` was patched
*down* to 2.3.1 on 2026-03-03 to match the machine rather than the file being fixed.

**A second, quieter door was open the whole time.** `env.yml` carried a
`channel_priority: strict` line, and conda has never read it: the valid keys of an
environment file are `name`, `dependencies`, `prefix`, `channels`, `variables`, and
everything else is silently ignored. So the `defaults` channel — whose `mkl` pulls a
genuinely separate `intel-openmp` — was never excluded.

#### What changed

| file | change |
|---|---|
| `env.yml` | `nodefaults` channel; CPU torch; every compiled package pinned and sourced from conda-forge; `quantile-forest` moved out of pip; the bogus `channel_priority` line gone; the pip block cut to the five things no channel carries |
| `pip-constraints.txt` | **new.** Pins every conda-installed package by version. `setup.sh` exports `PIP_CONSTRAINT`, so a pip package that wants a different torch/scikit-learn/lightgbm/xgboost now **fails the build loudly** instead of silently swapping in a wheel with its own runtime |
| `setup.sh` | rebuilds on `SETUP_REBUILD=1`; installs the extras only when the recipe hash changes; installs `torchsort` with `--no-build-isolation`; installs `noiseInject` and `kirby` editable; **no longer installs the PyG companion wheels** |
| `scripts/check_environment.py` | the threading check counts **distinct resolved files**, not library names; new checks for `env.yml` truthfulness, `noiseInject`/`kirby`, `/proc/self/maps`, and both blocker probes |
| `scripts/process_and_train.py` | the dead `from gensim.models import word2vec` deleted — the last trace of mol2vec in an import path |

**Why the file-count matters more than it sounds.** A name-based check gets this wrong in
both directions: it fails a healthy conda environment (three names, one file) and it passes
the actual defect (two wheels, two copies, one name). The old check was name-based. The new
one resolved four distinct files in the system Anaconda and one in `env_test`, which is what
made the table above possible.

#### The torch pin, settled 2026-08-27

`env.yml` has claimed `pytorch=2.5.1` since March 2025 and the cluster has run a pip wheel of
`2.3.1+cu121`, so **the file has not been a record of what any result was produced under.**
Settled **upward, to 2.5.1, CPU build**:

- QM9 is being regenerated and the validation sets have to be re-run under the redesigned
  noise conditions regardless, so **no surviving result is invalidated by the move**.
- **CPU** because no SLURM script in either repo has ever requested a GPU — zero `--gres`
  across all of `slurm_scripts_*` and KIRBy. The CUDA build was four gigabytes that only ever
  ran CPU kernels, and its `libtorch_cuda.so` is what made the preflight report
  "inconclusive" on a login node instead of a verdict (§2.8d).
- `gpytorch` moves 1.14 → **1.11** in the file. 1.14 was never installed: `botorch 0.10.0`
  pins `gpytorch==1.11` and pip enforced it every time.

Solves measured the same day: linux-64, 290 packages, 538 MB, one runtime. osx-64, 278
packages, 268 MB, one runtime.

#### Built and gated, not just solved (2026-08-27)

The whole recipe was built from scratch through `setup.sh` — conda list, pip block,
`torchsort` from source, `torchcp`, and both editable installs — and then put through
`check_environment.py --deep --validation`:

| check | result |
|---|---|
| every QM9 roster label constructs | ✅ **26 of 26**, conformal included |
| `qrf` and `ngboost` actually *fit* | ✅ |
| `models/models.py` imports for real | ✅ |
| distinct OpenMP runtime files | ✅ **one** |
| LightGBM fits, no thread count set / both pinned to 4 | ✅ / ✅ |
| GP fits after lightgbm+xgboost, no thread count / both pinned to 4 | ✅ / ✅ |
| `env.yml` describes the interpreter | ✅ all 34 pins |
| `noiseInject` 1.0.0, `kirby` 0.2.0 importable | ✅ |
| KIRBy validation roster, every optional backend | ✅ |

**The conformal models build for the first time.** §2.8d's remaining real failure was
`torchsort/isotonic_cpu...so: undefined symbol: _ZNK3c105Error4whatEv`, and
`--no-build-isolation` closes it: all four `conformal*` labels construct. They stay in
`EXCLUDED_MODELS` — that is a study decision, not a broken package — but the guard no longer
blocks them for the guard's own reason.

One pin was wrong on the first build and the check caught it: conda-forge ships `lightning`
as `2.5.1.post0`, not `2.5.1`. That is the truthfulness check doing its job on its first run.

The other two code bases were run against the same fresh environment on the same day:

| check | result |
|---|---|
| `scripts/test_noise_conditions.py` — the settled conditions resolve on both sides | ✅ 7 conditions, Python injector and job generator agree |
| `scripts/crosscheck_injectors.py` — Rust and Python injectors agree | ✅ **342 of 342**, on all 133,885 real QM9 labels |
| KIRBy `alternative_data_noise_robustness.py` imports and builds its parser | ✅ reads the shared model spec, offers the 11 settled conditions |

⚠️ **All of this was measured on osx-64, on the laptop.** It proves the recipe resolves,
builds, and clears both blockers, and that all three code bases run against it. **It does not
prove the cluster's answer**, and the cluster is where the failure was found: `linux-64` has
been verified only as a *solve*, never as a build, and the `/proc/self/maps` check is
Linux-only and has not run at all. Until `sbatch scripts/rebuild_env.sh` comes back clear,
this section is verified on one platform out of two.

#### Packages the roster was missing, now in the recipe

- **`quantile-forest`** — the audit found it absent from the interpreter it checked, so every
  quantile-forest task died on contact. It is on conda-forge for linux-64/py310 and is now a
  conda package, built against the same scikit-learn it runs with.
- **`kirby`** — *not importable in `env_test` at all* (checked 2026-08-27). The validation
  pipeline does `from kirby.representations.molecular import ...` with no `sys.path` help and
  KIRBy uses a `src/` layout, so **the whole KIRBy half cannot start without an editable
  install.** `setup.sh` now does it.
- **`noiseInject`** — editable from the checkout, so the injector cannot drift from the code
  the cross-check gates test. (Its `.egg-info` currently reports 0.4.0 while the code is
  1.0.0; a fresh editable install clears that.)
- **`torchcp` + `torchsort`** — `torchsort` has no linux wheel and must compile against the
  *installed* torch. With pip's default build isolation it compiles against a fresh torch pip
  downloads into a throwaway environment, which is precisely the
  `undefined symbol: _ZNK3c105Error4whatEv` of §2.8d. `--no-build-isolation` is the fix.
- **`requests`** — imported at module scope by the validation pipeline, never declared.

Dropped as imported nowhere in either repo: `tensorflow`, `jax`, `gensim`, `mol2vec`,
`torchaudio`, and the four PyG companion wheels.

**Expect four harmless lines on every `import deepchem`** now that TensorFlow, JAX and DGL
are gone: *"Skipped loading some Tensorflow models, missing a dependency"* and three like it.
That is deepchem declining to register its own optional model zoo, none of which this study
uses — everything here uses `dc.data.DiskDataset`, `dc.splits.ScaffoldSplitter` and the
`molnet` loaders, all pure numpy. Checked 2026-08-27: scaffold splitting produces the same
splits with TensorFlow absent. Do not read those lines as breakage.

#### It must never happen during a run

`setup.sh` is **sourced by every job script**, and it used to run four `pip install` commands
every time — on a 390-task array that is 390 concurrent writers into one shared
`site-packages`, quietly undoing whatever the last rebuild pinned. The extras now run only
when a hash of `env.yml` + `pip-constraints.txt` changes, recorded in a stamp file inside the
environment. A running task cannot mutate the environment any more.

#### Copy-paste: the rebuild on ARC

**One submission does all of it** — records the old environment, rebuilds in the same prefix,
runs the gate, runs both blockers standalone, and checks the noise injectors and the KIRBy
pipeline. It writes one file, `~/env_rebuild_report.txt`, and ends with a verdict:

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models
git pull
REBUILD_DRY_RUN=1 bash scripts/rebuild_env.sh    # optional: see what it would do
sbatch scripts/rebuild_env.sh
```

It **refuses to start while any of your jobs are queued or running** — a rebuild changes
numbers, and rows written under two different environments in one results file, with nothing
recording which, is the failure this whole section exists to prevent. `FORCE_REBUILD=1`
overrides it deliberately.

It rebuilds **in the prefix env_test already occupies**, read from `conda env list`. Creating
by name would put a multi-gigabyte environment in whichever `envs_dir` comes first, which on
this cluster can be the home quota.

It is a job, not a login-node paste, because a login node caps memory per user — the same cap
that made an earlier audit report sixteen phantom failures (§2.8d).

**Rollback**, if it is ever wanted: step 1 writes
`research_archive/env_test_before_rebuild_<date>.txt`, a `conda list --explicit` of the old
environment. `conda create --prefix <prefix> --file <that file>` puts it back, then rerun
`setup.sh` for the extras.

If you would rather drive it by hand:

```bash
# 0. Keep a record of what the OLD environment was, before it is destroyed.
cd /data/stat-cadd/scat9264/qsar_qm_models
git pull
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_test
conda list --explicit > research_archive/env_test_before_rebuild_2026-08-27.txt
python -c "import torch, sklearn, gpytorch; print(torch.__version__, sklearn.__version__, gpytorch.__version__)" \
    >> research_archive/env_test_before_rebuild_2026-08-27.txt
conda deactivate

# 1. Rebuild INSIDE AN ALLOCATION. A login node caps memory per user, which is what
#    made the last audit report sixteen phantom failures.
srun --account=stat-cadd --partition=short --cpus-per-task=8 --mem=32G \
     --time=02:00:00 --pty bash

cd /data/stat-cadd/scat9264/qsar_qm_models
source "$(conda info --base)/etc/profile.d/conda.sh"
conda config --env --set channel_priority strict
export ENV_TEST_PREFIX=/data/stat-cadd/scat9264/conda_envs/env_test
SETUP_REBUILD=1 . ./setup.sh

# 2. THE GATE. Nothing launches until this exits 0, in this same allocation.
python scripts/check_environment.py --deep --validation ; echo "exit: $?"
```

`--deep --validation` is the whole answer: it constructs every model in both rosters, imports
`models/models.py` for real, checks `env.yml` against what is installed, checks `noiseInject`
and `kirby`, counts the distinct OpenMP runtimes both statically and in
`/proc/self/maps` after importing every backend, and then **runs both blockers in the same
environment** — the LightGBM fit and the Gaussian-process-after-boosting fit, each under both
of the thread settings the two pipelines use (QM9 sets none, the validation module pins both
to 4). Curing one at the other's expense is the trap; this fails if either fails.

The two named probes standalone, if you want them separately:

```bash
python -c "import torch, lightgbm as lgb, numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=400, n_features=512, random_state=0)
lgb.LGBMRegressor(n_estimators=15, verbose=-1).fit(X, y); print('LightGBM OK')"

bash scripts/server_audit.sh          # section 5 is the Gaussian-process probe
```

#### Consequences to carry forward

- **Every number changes.** Torch moves 2.3.1 → 2.5.1 and the BLAS/threading layer changes
  underneath every model. This is why it happens before launch and never during.
- `GP_DEFAULTS['single_thread_fit']` stays **True** until the gate above has passed on the
  cluster. It costs 11% on the Gaussian process alone and it is the net under exactly this
  failure; it comes down after the cluster says one runtime, not before.
- **§2.8e-bis's second option is now done, as far as it can go.** `lightgbm`, `xgboost`,
  `quantile_forest` and `grakel` are imported inside the functions that use them, so
  importing `models/models.py` no longer loads any of the three boosting/tree backends —
  measured 2026-08-27. A Gaussian-process task therefore no longer sits in a process with
  both boosting libraries resident, which was the stated precondition of the §2.8e segfault.

  **It goes no further, and this is why the environment fix is the real one.** `torch`,
  `torch_geometric`, `gpytorch` and `gauche` are needed to *define* the module — classes
  inherit from `nn.Module`, `MessagePassing`, `gpytorch.models.ExactGP` and `SIGP` — so a
  LightGBM job still loads the whole Gaussian-process stack, and no rearrangement of imports
  can change that. `grakel` is the same from the other side: `gauche.kernels.graph_kernels`
  imports it regardless. Import hygiene closes one direction of the conflict; only one
  threading runtime closes both.

#### The first cluster attempt, 2026-08-27 21:47 — what it cost and what it changed

`scripts/rebuild_env.sh` was run on an ARC **login node**. The conda solve was killed part way
through `Collecting package metadata (repodata.json)` — the message is the single word `Killed`,
which is the per-user memory cap, not the recipe. Four defects showed themselves at once, all now
fixed, and every one of them would have fired again on the next attempt:

1. **`setup.sh` deleted the old environment before building the new one.** The solve then died and
   the account was left with no `env_test` at all. It now moves the old prefix to `<prefix>.old`,
   builds at the real path, and puts the old one back if the build fails. Aside-then-build rather
   than build-then-move because a conda environment is not relocatable: every console script in
   `bin/` carries its build path in the shebang.
2. **`setup.sh` carried on after the failed build.** Its activation check asked only whether
   `CONDA_PREFIX` was set, and the system Anaconda's was, so the four extras installed themselves
   into `/apps/.../Anaconda3` and `~/.local/lib/python3.9`. It now requires the activated prefix to
   be the one that was asked for and refuses otherwise.
3. **`rebuild_env.sh` sourced `setup.sh` through a pipe** (`. ./setup.sh | tee`). The left side of a
   pipe is a subshell, so the activation was discarded and every check below it ran against
   whatever python was on `PATH` — python 3.9 from the system Anaconda, which is why the report
   showed sixteen absent packages and six failures that were not tests of `env_test` at all.
   Process substitution now keeps the source in the calling shell, and the script stops with one
   line if the build did not produce the environment.
4. **It warned about the login node and continued.** It now re-executes itself inside an allocation
   (`srun --account=stat-cadd --cpus-per-task=8 --mem=48G --time=03:00:00`) rather than asking for a
   second paste, and puts the conda package cache beside the environment instead of letting it
   default into the home quota.

Two further hardenings came out of the same log. `PYTHONNOUSERSITE=1` is now exported, because
`~/.local/lib/python3.X/site-packages` is read by every interpreter of that version — this account
has a `torch 2.2.2+cu121` sitting in one, a second set of OpenMP runtimes arriving by a route
neither `env.yml` nor `pip-constraints.txt` can see. `PIP_USER=0` is exported with it, so pip fails
rather than silently falling back to `--user` when it cannot write to site-packages.

**Cost:** the old `env_test` at `/data/stat-cadd/scat9264/conda_envs/env_test` is gone. It was
recorded first — `research_archive/env_test_before_rebuild_2026-08-27.txt`, 190 lines,
`conda list --explicit`, enough to rebuild it exactly — and the next run is a first build at that
same prefix rather than a rebuild. Nothing was lost that the recipe does not carry, and the old
environment held two distinct OpenMP runtime files, so it was not an environment worth keeping.

The three-way behaviour was tested against a stubbed conda before this went back to the cluster:
a failed build leaves the old environment in place, a successful one keeps it at `.old` and says
how to delete it, and an activation that lands elsewhere refuses to install anything.

### 2.8j ✅ FIXED 2026-08-27 — the uncertainty jobs asked for six conditions that no longer exist

**The defect.** `slurm_scripts_uncertainty_rerun/generate_scripts.py` still listed the six deleted
strategies — `legacy, outlier, quantile, hetero, threshold, valprop` — and emitted `--strategies`,
`--unc-strategies` and `--threshold-quantile`, none of which the runner has any more. It takes
`--conditions` and `--unc-conditions` now, and `--conditions` carries `choices=`, so every one of
the 504 tasks would have died at argument parsing. The same six were hard-coded a second time in
`merge_results.py`, where they set the expected cells of the coverage report: had the jobs somehow
run, every cell would have read `MISSING` and nothing that did run would have been checked. The
level check beside it expected eleven levels; the ladder is six per dataset, and seven for
censoring, which sweeps the clipped fraction instead.

**What it runs now.** The four conditions the main grid runs — `gaussian`, `grouped_wider`,
`grouped_shifted`, `censoring` — plus `outlier_p10`, which is §13.1 item 2's recorded default. All
five are read from `noise_conditions.json`, not restated. 3 datasets × 4 representations × 5
conditions = **60 tasks per model script, 420 in total**, against 504 before. `--include-deep-conditions` runs all three depth-only
conditions instead of the one (588 tasks). Levels are not passed at all: the runner sweeps one shared grid in
fractions of each dataset's own clean training label spread, the same seven QM9 runs (the author's
decision of 2026-08-27, `KIRBy` `2df1a5c`), with censoring on its own clipped-fraction axis.
**88,200 model fits.**

**The five split cleanly across the two questions**, which is the thing to say in the Methods.
`gaussian` and `grouped_shifted` give every molecule the same amount, so "does uncertainty find
*which* molecules are corrupted" is undefined there, not zero — they answer question A and serve
as the leakage check. `grouped_wider` (keyed to the scaffold), `censoring` (keyed to the label) and
`outlier_p10` (keyed to the draw) are the three with a pattern to find. Both grouped conditions are
keyed to something a scaffold split holds out whole, which is §3.1d: on held-out molecules the
grouped pattern is flat, truthfully, and the predicted-label control is degenerate for it.
Censoring and outlier are not affected by that.

**Three defects found alongside it, all fixed.**

| Where | What |
|---|---|
| `merge_results.py` | The six deleted names hard-coded as the expected cells, and "11 levels" as the expected ladder. Both are now read — conditions from the generated `unc_*.sh`, level counts parsed out of the runner's own source — so neither can drift again. `task_strategy` becomes `task_condition`; `scripts/uncertainty_stats.py` accepts both, so a merge made before today still loads |
| `preflight.sh` | Section 4b still reached for `m.STRATEGIES`, which the runner no longer defines, and read `STRATS=(...)` out of the job scripts. It reads `CONDS=(...)` and `m.NOISE_CONDITIONS`, and warns rather than guessing if it finds a script that predates the redesign. Its advice for a flat condition — regenerate with `--threshold-quantile` — pointed at a deleted flag; a condition that is flat where it should not be is now reported as a fact about the dataset |
| the generated scripts | They hard-coded `/data/stat-cadd/scat9264/KIRBy` and took it on trust (§2.8b: 125 of KIRBy's own 127 job scripts use `/data/stat-ecr`). Each script now checks the directory exists, that the runner in it has `--conditions`, and that the installed injector knows the conditions being asked for — exiting 2 and naming the other checkout rather than producing a full run's worth of results from the old code. `--kirby-dir` regenerates against the other path |

**Two gates, and they check different halves.** Both need a KIRBy checkout (`--kirby-dir`,
`$KIRBY_DIR`, or a sibling directory) and refuse to run without one.

`python scripts/test_uncertainty_job_scripts.py` checks that the command line each script **emits**
is one the runner accepts. It generates real scripts, pulls the command line out of each one,
substitutes the shell variables with values the array dispatch can actually produce, and runs the
result through the runner's **own argument parser**, not a string match against its source. Seven
checks: every emitted command line parses (35 of them, one per script per condition); no deleted
name or flag survives; the conditions match the settled file, and the one condition added to them
is one that is *not* flat by design — which is the only reason to add it; dropping every even
condition or every patterned one is called out by name; the model, representation and dataset
rosters match the runner's; the scripts are valid bash; and all 60 array indices map to 60 distinct
tasks with the header promising the same range.

`bash scripts/smoke_uncertainty_job_scripts.sh` checks the other half — that a script gets far
enough to emit it. It **executes** a real generated script, 60 times, with two things stubbed and
nothing else: a `setup.sh` that activates an environment called `env_test`, and a `python` that
records the argument list instead of running the 47-hour job. Everything between the top of the
file and the python call runs for real, including the injector check, which genuinely imports
`noiseInject`. 12 checks, about 40 seconds: four representative tasks reach the runner with no
retired flag in the command; all 60 indices write 60 distinct results directories covering exactly
the five conditions; and each of six guards stops the job it exists to stop, with the message it
exists to print — an index past the end, no partition, a KIRBy checkout with no `--conditions`, no
KIRBy checkout at all, the wrong environment, and no environment at all. `--qsar-dir` was added to
the generator to make this possible: the path to this repository on the cluster was hard-coded, so
a generated script could not be run anywhere else.

**Still to do here, and it is not this fix's.** Whether the uncertainty runs use four
representations or all six is the same open question as which models and representations go deep
(§13.1 item 4). Four is what the generator has always used and what it still uses; `--reps` changes
it without editing anything.

### 2.9 The Methods figure does not show the experiment

`paper.tex:359` captions it as the QM9 label distribution. The code that draws it
(`generate_paper_figures_v2.py:2541-2562`) uses a synthetic three-component Gaussian mixture and
reimplements two of the noise types differently from the pipeline — threshold as a median split,
value-proportional as additive where the pipeline is multiplicative. Counting the Python injector,
"threshold" therefore has three different definitions in three places.

---

### 2.10b ✅ FIXED 2026-08-27 (chat E) — the QM9 graph models were trained on other molecules' labels

Two separate faults in the same function, both executed and both closed. The QM9 graph roster is
gin, gcn, ginct, gin2d, graph_gp and the conformal graph wrapper.

**The split was void.** `split_qm9` shuffles with `qm9.index_select(randperm(...))`. PyG returns a
NEW dataset from that call, so the name it assigns to is local and nothing outside the function
sees it. Every index the function returns is a position in the shuffled order, and every molecule
it featurises and writes to the mmap is taken from the shuffled order. `main()` kept the indices
and dropped the object, then passed the ORIGINAL dataset to `run_qm9_graph_model`, where the graphs
are read as `qm9[train_idx]`.

Measured, on the real dataset, 200 molecules, scaffold split: **0 of 160 training rows had the same
SMILES on the two sides, and 0 of 160 had the same label.** So each graph was paired with a
different molecule's label, and `qm9[train_idx]` on the unshuffled dataset is an arbitrary subset
rather than a held-out-scaffold partition.

**The noisy labels never reached the model.** `run_qm9_graph_model` did
`qm9[idx].y_noisy = y_train_noisy[i].item()`. Indexing a PyG InMemoryDataset BUILDS a new `Data`
object every time, so that assignment landed on a temporary and was discarded. Executed on the real
dataset: after the loop, `hasattr(qm9[0], 'y_noisy')` is False, and a batch raises
`'GlobalStorage' object has no attribute 'y_noisy'`. The caller's blanket `except Exception` turned
that into a missing result row. **No QM9 graph-model row can have come from this code.**

Both fixed: `split_qm9` returns the shuffled dataset and `main()` passes it on; the graphs are
materialised once into a list and the labels attached to those objects. A third fault surfaced only
when the run got that far — the loader collates the per-graph scalar into a one-element tensor, so
the target array came out `(n, 1)` against an `(n,)` prediction and `pearsonr` refused it. The
targets are read with `float()` now.

**The check:** `scripts/test_qm9_split_alignment.py`. Removing either half gives
*"main() passed a dataset in which 160 of 160 training rows carry another molecule's graph"* and a
non-zero exit. Executed 2026-08-27. A GIN then ran end to end on real QM9 at two noise levels and
wrote rows for the first time.

⚠️ On this laptop `torch.randperm(129428)` segfaults after the full import stack unless
`OMP_NUM_THREADS=1` is set. It is the same shape as §2.8e. Not reproduced on the cluster and not
guarded in the code.

### 2.10c ✅ FIXED 2026-08-27 (chat E) — the job generator asked for a setting the program refuses

`--bayesian-transformation last` matched no branch in models.py, which tests `last_layer`. Already
closed on the program side: argparse now has `choices=` and, executed,
`process_and_train.py --bayesian-transformation last` exits with *"invalid choice"*. The generator
at `slurm_scripts_qm9_rerun/generate_scripts.py` still emitted `last` for `dnn_bnn_last` and
`mlp_bnn_last`; both now say `last_layer`.

**The check:** `scripts/test_generated_job_flags.py` runs all 22 of the generator's model flag
strings through the program's real parser. Restoring `last` gives *"1 of 22 job definitions pass
flags the program refuses"*.

### 2.11 ✅ FIXED 2026-08-27 (chat E) — a results row now names the condition that produced it

The noise condition survived only in the output FILENAME. Both QM9 loaders in the figure script
recovered it by matching the stem against `{legacy, outlier, quantile, threshold, hetero, valprop,
heteroscedastic, value_proportional}` — the six names retired on 2026-08-26. Three failures under
one cause, all three reproduced by executing the loader on files written to a temporary directory:

- A file written under the settled scheme matched nothing, `strategy` stayed blank, and pandas
  treats blanks as equal — so two conditions for one (model, rep, level, replicate) came back as
  **one row**.
- `outlier` is in the retired list AND is a prefix of the settled `outlier_p10`, so
  `anova_outlier_p10_*.csv` was labelled `outlier`: a contaminated-fraction run pooled with the
  retired value-proportional strategy under one name.
- Every table that wanted the reference condition wrote
  `frame[frame.strategy == 'legacy'] if 'strategy' in frame else frame`, so a frame without the
  column silently became every condition pooled together under one condition's name. Ten of those.

What changed. `noise_type` is now a column in `RESULT_COLUMNS`, stamped from the injector's own
manifest — the name `condition_name` in `rust/src/main.rs` produced, never composed a second time
in Python. A level whose manifest names no condition stops the run. The figure script reads that
column, falls back to the filename only for older files, matches settled names before retired ones,
and labels a file it cannot place `unknown_<stem>` so it groups with nothing. `baseline_rows()`
raises instead of returning the whole frame. The experimental results loader now copies `noise_type`
into `strategy` and keeps `fold` in its dedup key — without it, five folds × seven conditions
collapsed to one row per (dataset, model, rep, level).

The uncertainty column moved too. `models/model_defaults.py` settles `primary_column` as `'raw'` on
36 measured fits; the figure script picked `y_pred_std_calibrated` first at four sites and never
imported the spec — and the raw column's real name, `y_pred_std_uncalibrated`, was not among its
four candidates at all. It now reads the column the spec names.

**The checks:** `scripts/test_figure_conditions.py` and `scripts/test_result_row_condition.py`.
Reverting the loader reproduces all three failures by name.

### 2.12 A level means three different things, and what the literature does about it

QM9 doses in fractions of the clean training label spread (`--dose-units spread`) and runs to 1.00
of it. The experimental grids are in RAW LOG UNITS, anchored to published assay error, and reach
0.84 label SD on LogD, 1.57 on Caco-2 and 1.20 on hERG. Censoring's level is the fraction of labels
clipped, which is not an amount of noise at all. All three were written into one column called
`sigma`. `auc_norm` is mean retention over each configuration's own level range, so two `auc_norm`
values are on the same footing only if the axis under them is the same quantity over the same span.

**What the predecessor does.** Kolmar & Grulke, *The effect of noise on the predictive limit of QSAR
models*, J Cheminform 13:92 (2021): the dose is set per dataset from that dataset's own endpoint
range — *"σnoise was determined from the product of the range of endpoint values in the dataset, the
noise level n, and a multiplier"* — and datasets are compared with BOTH axes divided by the
noise-free baseline error: *"the y-axis is RMSE/RMSE0. The x-axis is the standard deviation of the
Gaussian distribution from which the added error was sampled (σ), divided by RMSE0."*

So the shared axis is the delivered dose in raw label units divided by that configuration's own
zero-noise RMSE. Both pipelines now write what the noise DELIVERED (`delivered_dose` on the QM9 row,
`realised_dose_label_units` on the experimental one) and what the level MEASURES (`level_units`:
`label_sd`, `raw_label` or `fraction_censored`). The figure script builds
`dose_over_baseline_rmse` from those and reports `auc_norm_shared` beside `auc_norm`, and
`warn_if_axes_differ` names a mismatch instead of pooling silently. Censoring stays on its own axis:
its level is already dimensionless and it has no dose to rescale.

**✅ SETTLED 2026-08-27 by the author: one shared grid, no rescaling.**

The three experimental grids become the QM9 level grid (`NOISE_DESIGN.md` §6.4), read as
fractions of each fold's CLEAN TRAINING label spread — the grid QM9 already runs,
so QM9 does not move. Each experiment runner multiplies the level by that spread
before the dose reaches the injector, so `sigma` on the row stays the shared
ladder and `level_units` reads `label_sd` on both sides. **Censoring is exempt**:
its level is the fraction of labels clipped, already dimensionless and already
shared, and scaling it by a spread puts it outside [0, 1] — caught by the smoke
suite the moment the change was made.

Measured on hERG, forest + ECFP4, after the change: at level 0.20 the delivered
noise is 0.182 log units, at 0.50 it is 0.440, at 1.00 it is **0.896 against a
clean label spread of 0.896**, and at 1.50 it is 1.295. R² falls 0.514 → 0.159
across that fold.

**What it costs, and it belongs in the caption.** A level is no longer a stated
multiple of assay error. At the top of the grid LogD carries **11.9x** its
within-lab error of 0.15, Caco-2 **1.9x** its 0.35, hERG **2.5x** its 0.54. That
asymmetry cannot be removed: the published assay error is 0.13 of the label
spread on LogD and 0.79 on Caco-2, so whichever axis is held constant, the other
varies sixfold. §6.4 of `NOISE_DESIGN.md` held realism constant; this holds comparability
constant. Both grids and the reason for each are written above
`NOISE_LEVELS_BY_DATASET`.

**Re-runs: all three experimental datasets. QM9 is unaffected.**

`auc_norm_shared` is kept as a diagnostic rather than deleted — one column, and
it is how a future drift between the grids would be noticed.

**Also settled 2026-08-27, and each one invalidates results:**

- **ChemBERTa is one encoder now.** QM9 loaded `seyonec/ChemBERTa-zinc-base-v1` (masked language
  model, ZINC, 768 wide, 6 layers) and the experimental pipeline loaded
  `DeepChem/ChemBERTa-77M-MTR` (multi-task regression, PubChem, 384 wide, 3 layers). Read from the
  two cached configs this session. Both sides now use `DeepChem/ChemBERTa-77M-MTR`. The record slot
  moved 3072 → 1536 bytes in `scripts/process_and_train.py` and `rust/src/main.rs`, and the QM9
  pooling now excludes padding the way the experimental side already did. **Every QM9 ChemBERTa
  cell must be re-featurised and re-run.**
- **No model stacks validation.** Forest, SVM, XGBoost and LightGBM took all of it (90% of the
  data), NGBoost and the Gauche GP took half (85%), the neural models took none (80%) — so the
  ANOVA's model factor confounded model family with training-set size, and nothing on the row
  recorded which regime produced it. Every `vstack((x_train, x_val))` is gone. **Invalidates every
  QM9 tree, kernel, NGBoost and GP number.**
- **The Bayesian networks are fitted on the ELBO.** There was no KL, ELBO or `BKLLoss` anywhere in
  either pipeline: BNN-α and BNN-β trained on plain MSE while the VBLL variants carried a KL term
  all along. torchbnn samples weights in train AND eval mode, so the posterior width received MSE
  gradients, which drive it toward zero; `prior_sigma` was only an initialisation. Measured on 300
  steps of the same data with the same seed: mean posterior width **0.10000 → 0.06028 on plain MSE,
  → 0.06432 on the ELBO**. The weight is `BAYESIAN_DEFAULTS['bnn_kl_weight'] = 'elbo'`, meaning
  1/n_train. Spec bumped to **1.3.0**. **Invalidates every BNN-α and BNN-β number on both
  pipelines.**
- **The early-stopping validation split is carved by scaffold.** It used to be the first fifth of
  each fold's training block in dataset order, which is alphabetical by SMILES. On a
  600-molecule/120-group fixture the old rule put 5 scaffold groups in both training and validation,
  and left **112 of 600 molecules (19%) out of every training set in all five folds**. The new
  `scaffold_validation_carve` gives 0 and 0. **Invalidates every experimental neural number.**

**The checks:** `scripts/test_bnn_kl_term.py`, `scripts/test_figure_conditions.py` (level axis),
`KIRBy tests/smoke/smoke_kirby_splits.py`.

### 2.13 ✅ FIXED 2026-08-27 (chat E) — definitions that were never the ones that ran

`create_gnn_model` was defined **four** times in models.py. Python binds the last, so three
architectures sat in the file looking authoritative and never ran — including the one whose
GCN/GAT/GIN classes anyone reading the file to describe the GNN would describe. Executed:
`create_gnn_model.__code__.co_firstlineno` was 4106, not the 3679 the audit named. `pyg_to_grakel`
was defined twice and `class GCN` twice.

It was not harmless. `train_conformal_graph_model` builds its base network as `GIN(dim_h=dim_h)` —
the signature of the `class GCN` that a later definition shadows. Executed: the live GCN and GIN
take `(num_node_features, hidden_dim, ...)` and `GIN(dim_h=64)` raises `TypeError`. The live classes
also return `(prediction, embedding)`, and `train_epochs` does `out.detach().cpu().numpy()[:, 0]`.
**That path has never constructed a model.** It now refuses by name instead of failing into the
caller's blanket handler as a missing row. It is tier 4 in the job generator and has never produced
a number; whether to repair it or drop it is yours.

The shadowed copies are deleted. An `ast` comparison of every top-level definition before and after
shows **no live definition changed**.

Three more in the same pass:

- `train_conformal_model` built every base estimator from an EMPTY parameter dict — conformal_rf on
  sklearn's `max_features=1.0` instead of the spec's 0.3, conformal_qrf on 100 trees instead of 300,
  conformal_xgboost on the booster's `learning_rate` 0.3 instead of 0.1. It calls
  `sklearn_params(...)` now.
- `train_gnn` had no early stopping, no validation loss inside its loop, and never read
  `args.epochs`. It now stops on the shared `NEURAL_DEFAULTS` rule and restores the best epoch.
  Executed on real QM9: *"stopped at epoch 15, restored epoch 5"*.
- Both graph paths wrote the TEST-set size into `sample_size`. Executed: a `-n 200` run wrote 20.
  They write `args.sample_size` now, and the same run writes 200.

**The check:** `scripts/test_no_shadowed_definitions.py` walks models.py, model_defaults.py,
process_and_train.py, utils.py, generate_paper_figures_v2.py and the experimental runner, and fails
on any repeated top-level definition.

---

### 2.13b ✅ FIXED 2026-08-27 (chat E) — 'ECFP4' was not ECFP4

The QM9 side computed `ecfp4` with `rdk_fingerprint_mol`, which is
`RDKFingerprintMol` — RDKit's **path** fingerprint. The experimental side computes
`rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`, which is what
ECFP4 means.

Measured on the first 1,500 QM9 molecules: **the two agreed on 0 of 1,500.**
Methane, ammonia and water came back **all zero** under the path fingerprint,
because a molecule with one heavy atom has no bond paths — and those rows passed
the featurisation gate, which only refuses a fingerprint that FAILED to compute.

The rdkit-sys binding exposes only `morgan_fingerprint_mol`, hardcoded to radius
3, so there was no route to radius 2 on the Rust side at all. ECFP4 is computed in
Python now and carried through the record, like Avalon; `prepare_ecfp4` and its
binding are deleted. The Python writer refuses an all-zero fingerprint by name,
and an all-zero block reaching Rust is recorded as a featurisation failure.

**The check:** `scripts/test_ecfp4_identity.py`, and a real QM9 run: 240 of 240
dumped training rows match a direct Morgan radius-2 fingerprint, 0 of 240 match
the old one. ⚠️ **Every QM9 ECFP4 result is void.**

### 2.13c ✅ FIXED 2026-08-27 (chat E) — the QM9 split trained on a population it never scored

`dc.splits.ScaffoldSplitter` keys on `MurckoScaffoldSmiles`, which returns the
EMPTY STRING for an acyclic molecule, and does not special-case it — so every
acyclic molecule joined one pseudo-group, and DeepChem fills training from the
largest group first.

Measured on the first 2,000 QM9 molecules: **851 (42.5%) are acyclic, and they
were 53.2% of training and 0.0% of validation and of test.** Half the training
population never appeared in what the models were scored on.

`scaffold_split_indices` replaces it: each acyclic molecule is its own group (as
QM9's own noise grouping and the experimental pipeline's CV grouping already do),
and groups are filled in a seeded random order rather than largest-first —
largest-first leaves the singletons for the held-out splits, which made validation
100% acyclic and test 82%. Same 2,000 molecules now: 39.6 / 65.5 / 43.0% against a
population of 42.5%, and **0 scaffold groups shared between train and test.**

**The check:** `scripts/test_qm9_split_alignment.py`.

### 2.13d ✅ FIXED 2026-08-27 (chat E) — the experimental pipeline's splits and targets

Four faults, each measured, each with a check that carries the old rule as a
control (`KIRBy tests/smoke/smoke_kirby_splits.py`,
`smoke_kirby_target_scaling.py`, `smoke_kirby_merge.py`).

- **The CV scaffold key carried stereochemistry.** `Chem.MolToSmiles` defaults to
  `isomericSmiles=True`, so stereoisomers of one framework landed in different
  groups and could be split between train and test — while `create_ecfp4` leaves
  `includeChirality` False, so those molecules are bit-identical in the
  fingerprint. Test rows whose ECFP4 vector is identical to a training molecule's
  in their own fold: **LogD 263/5,039 = 5.22% → 4/5,039 = 0.08%; Caco-2 72/2,161 =
  3.33% → 2/2,161 = 0.09%; hERG 20/1,415 = 1.41% → 0/1,415 = 0.00%.** QM9 was
  chirality-blind all along, so this is also what makes 'scaffold split' one
  protocol.
- **The early-stopping split was the first fifth of each fold's training block in
  alphabetical SMILES order.** It shared scaffolds with training, and the same
  low-index molecules sit at the front of four folds out of five. On a
  600-molecule/120-group fixture: **5 scaffold groups in both training and
  validation, and 112 of 600 molecules (19%) in no training set in any fold.**
  Both are 0 under `scaffold_validation_carve`.
- **The target was raw log units on the tree and kernel path** while QM9 fits every
  model on a z-scored label, and the shared spec is written for a unit-variance
  target — SVR's `epsilon` 0.1 is a tenth of a standard deviation on QM9 and a
  large dead zone on a label whose spread is a few tenths of a log unit. Both
  sides z-score now, **fitted on the CLEAN training labels**: the noisy target's
  spread grows with the dose (1.0000 at level 0, 1.5984 at 0.6 on the fixture)
  instead of being renormalised back to 1 at every level, which is what made the
  same nominal level a different optimisation problem.
- **`noise_pattern` described an injection that never happened.** It is the
  level-free shape the zero-level subtraction rests on, and WHO gets hit was
  seeded from the injector's seed, which carries the level. Measured on a
  120-molecule fixture: for `outlier_p10` the pattern and the realised scale had
  **Spearman −0.101 and −0.087**, top-15% overlap **3/18 and 4/18**; for
  `grouped_wider`, 0.375 and −0.250. `NoiseInjectorRegression` gained an optional
  `selection_state` (defaulting to `random_state`, so nothing else changes) and
  the runner pins it to (stream, condition) with no level. Now **Spearman 1.000
  and 18/18 at every level.**

**Found by running it, not by reading it.** `quantile_forest` raises
`Invalid parameter 'monotonic_cst'` on this sklearn, so a `--models QRF BNN-Full`
run produced no QRF rows at all — **and exited 0 with a results file.** The
all-failed guard fires only when EVERY model fails; one model failing everywhere
is the likelier case and the more damaging one, because the model is then absent
for that dataset only and every cross-model aggregate is computed on a biased
subset. A requested model or representation that produces no rows now stops the
job. Confirmed on the same command: exit 1, *"['QRF'] were requested and produced
NO rows for ChEMBL-hERG-Ki"*. ⚠️ QRF could not be run on this laptop at all —
verify it on the cluster before submitting QRF jobs.

Also closed here: a `--conditions` job rewrote `all_results.csv` and `summary.csv`
with its own rows alone and destroyed every other condition in the file — the
merge guard knew about `--models` and `--reps` and not about the flag the runbook
says the jobs are split by. The neural stream is seeded per cell, so a
`--models BNN-Full` job reproduces the same cell as a full-roster job. A fit whose
validation loss is never finite raises instead of losing the cell to
`load_state_dict(None)`. A representation that failed to build stops the job, and
so does an all-zero feature row.

### 2.13e ✅ FIXED 2026-08-27 (chat E) — the uncertainty columns were the wrong quantity

- **The Gaussian process reported the LATENT posterior spread**, not the
  predictive one: `model(x)` rather than `likelihood(model(x))`, so the
  observation noise was excluded — while `total = sqrt(posterior_variance +
  likelihood_noise)` was computed on the very next line and thrown away. A
  coverage number is a statement about observations.
- **VBLL reported the spread of its Monte Carlo passes**, which is the epistemic
  term alone, and discarded the layer's learned observation noise. Measured on a
  fixture: total 1.1117 against an MC spread of 0.9926. A plain BNN has no learned
  noise, so nothing moves there.
- **The Graph GP's per-molecule uncertainty was one constant** —
  `np.ones(n) * sqrt(best_noise)`. The posterior variance is computed from the
  Cholesky factor already in scope; 8 distinct values on a fixture where the old
  rule gave 1.
- **Three unpack faults meant no graph-model uncertainty could ever be written:**
  `decompose_uncertainty_gp` and `decompose_uncertainty_sampling` return three
  values and were unpacked into two, and the second was called with one argument
  where it takes two.
- `train_ntk_gnn` takes the three noisy-label arrays and reads none of them —
  it reads `data.y`, so it would be trained and scored on CLEAN labels at every
  level and its flat curve would read as robustness. It refuses by name.

### 2.15 ✅ SETTLED 2026-08-27 — what a spread on auc_norm is, and what it is not

**What was happening.** Every fold writes its own row: 5 folds x 7 levels per
model, representation and condition, all in `all_results.csv` with a `fold`
column. `summary.csv` then averaged the 5 folds at each level into one curve and
computed auc_norm once from that average. The only spread kept anywhere was
`baseline_r2_std` -- the 5 folds at level 0, and nothing else. QM9 writes no
spread at all; `save_results` has no std in it. **So auc_norm had no error bar on
either side.**

**What is there now.** auc_norm is computed on each fold's own curve, and
`summary.csv` carries `auc_norm_fold_mean`, `auc_norm_fold_std` and
`auc_norm_n_folds` beside it. `auc_norm` itself is unchanged -- still the
fold-averaged curve -- so nothing that already reads it changes meaning. No extra
fits: the per-fold rows were already written.

Measured, hERG / forest / ECFP4 / gaussian: auc_norm 0.7988 from the averaged
curve, 0.7942 as the mean of the five per-fold values, **spread 0.0848** across
them, against a baseline R2 of 0.5435 +/- 0.0672. **The fold spread is about 11%
of auc_norm.** A claim that one model is more robust than another on one dataset
needs a gap larger than that.

**The two sides' spreads are NOT the same quantity, and no compute makes them so.**
A QM9 replicate reshuffles all 129,428 molecules and takes the first
`--sample-size`, so it redraws WHICH MOLECULES as well as the split: its spread
contains sampling variance. LogD has 5,039 molecules and that is all of them --
there is no pool to draw from, so the only thing that could vary is the fold
assignment, which gives a split-to-split spread and not a sample-to-sample one.
The folds also partition one fixed dataset and share training molecules, so their
errors move together.

They therefore go in separate columns, and **must not share an axis or a
significance test**. QM9's per-replicate equivalent is computed in the figure
script rather than the pipeline, so it waits for that rewrite.

### 2.17 ✅ FIXED 2026-08-27 — the network's own predicted variance now reaches the file

Audit entry 53. Two losses make the network output how uncertain it is about each
molecule alongside the prediction. Every prediction site kept the prediction and
sliced the second output away — ten places across three functions — so both
models reported the spread over their stochastic passes, which is exactly what an
ordinary network reports, and their aleatoric column was blank. The one helper
that does the split properly had no callers anywhere in the three repositories.

**Kept only when uncertainty was asked for**, which is the author's decision of
2026-08-27. In `train_dnn_model` the whole block runs under `-u` already. In the
MLP path the prediction loop runs either way, because the prediction comes out of
it, so the variance is collected only under `-u`; a run that was not asked for
uncertainty holds nothing extra.

`split_predictive_head` in `scripts/utils.py` is the one place a wide head is
narrowed. Its transforms are the ones in `scripts/loss_functions.py`, because
that is what was fitted: `exp(log_var)` for the heteroscedastic head, and
`beta / (alpha - 1)` under the loss's own softplus for the evidential one.

Found next to it and fixed with it: on the path that is not asked for
uncertainty, `heteroscedastic` was handled and `evidential` was not, so its four
outputs were flattened into four times as many predictions as there are
molecules and the metrics were computed against whatever that lined up with.

`scripts/test_predictive_head.py` checks three things, all executing: the
variance is read back through the loss classes themselves rather than a copy of
their algebra; the aleatoric term differs between molecules; and a real 120-molecule
training run writes it for every test molecule. Restoring the slice fails the
third with "30 of 30 molecules have no aleatoric term".

`flexible_dnn` still drops its predicted variance, and that is deliberate — it is
in `EXCLUDED_MODELS`, writes no decomposition, and there is nothing to write it
to. Neither loss is in either job generator, so no existing result changes.

### 2.16 ✅ FIXED 2026-08-27 — one noise condition, one name

Audit entry 39, the naming half of what was item 4 below. The amount-of-noise
half is entry 34, fixed the same day and written up in §2.14a.

The two injectors named the same condition differently as soon as a shape other
than Gaussian was asked for. QM9 composed `outlier_p05_laplace`; the Python
injector returned `outlier/laplace`. Rows carrying those two strings cannot be
joined, and the join failure is silent — a filter simply returns nothing.

Worse than a mismatch: `outlier/laplace` was also what 1% and 10% contamination
came out as. The contaminated fraction is in no results column — `RESULT_COLUMNS`
carries `noise_type`, `level_units` and `delivered_dose` — so a name that drops
it loses it from the results entirely.

**Settled: QM9's rule, on both sides.** Targeting, then the contaminated
fraction where the targeting has one, then the shape, with the shape left off
when it is Gaussian. The eleven settled conditions keep the names they have
always had; the rule reproduces them. This is also what the label-noise
benchmarks do — corruption type and severity recorded as separate fields — and
the amount of noise is already its own column here.

The expected name for every condition is written down once, in
`condition_names.json`, and `scripts/test_condition_names.py` checks both
injectors against it: the QM9 one through the real binary's `--self-test --json`
mode, so it reads the shipped executable rather than a restatement of it. 19
conditions, both sides, all passing.

Second half of the same entry: `--noise-targeting` took `grouped_wide` while
every results row, manifest and figure said `grouped_wider`, so typing the name
read off a row killed the run. Both spellings are now accepted. The emitted name
is unchanged, and the table spells the targeting the way the rows do, so
removing that fix fails the check.

No results need re-running. Every condition the study runs is one of the seven in
`noise_conditions.json`, and the old and new rules agree on all of them.

### 2.14 The audit of 2026-08-26, closed

All 151 candidates in `research_archive/audit_2026_08_26/` now carry a verdict.
`unverified.json` is empty; `verdicts.json` holds all 111 with the evidence for
each; `confirmed_35.json` and `refuted_5.json` each carry a
`recheck_2026_08_27` field.

| | count |
|---|---|
| real, fixed | 77 |
| real, still open — yours to decide | 12 |
| duplicate of another entry | 14 |
| partly fixed | 4 |
| refuted | 2 |
| not a fault | 2 |

Of the 35 already marked confirmed: 27 real and fixed, 7 plain duplicates (four
entries for the ECFP4 fault alone, two for each graph-model unpack), 1 still open.
Of the 5 marked refuted: three refutations stand, one is half-real (the
validation-stacking half was real and is fixed), and **one is misfiled** — entry 4,
the Graph GP constant, is refuted in name while its own evidence confirms it. It
is real, and it is fixed.

**Every check was broken on purpose to see it go red.**
`scripts/check_fixes_fail_when_removed.py` edits the real file, runs the check,
and puts the file back. Twenty cases now; of the first twelve, ten went red
first time and **two stayed green**, which is the point of running it:

- The QM9 split check asserted each split is more than 10% acyclic. Under
  DeepChem's largest-first ordering WITH singletons, validation comes out 100%
  acyclic and test 82% — which passes a floor. It now asserts each split sits
  within 25 points of the population's 42.5%.
- The sibling-file check was caught by the column rule, not the name rule, so it
  did not guard the name rule at all. Its fixture's manifest now carries the
  results columns.

**The checks.** Fourteen suites, all executing, none matching source text:

| where | what it guards |
|---|---|
| `scripts/test_qm9_split_alignment.py` | the graph models' molecules, labels and split composition |
| `scripts/test_ecfp4_identity.py` | ECFP4 is Morgan radius 2 on both sides |
| `scripts/test_figure_conditions.py` | conditions, the level axis, the uncertainty column, sibling files |
| `scripts/test_result_row_condition.py` | the condition on the row, the manifest header, the manifest join |
| `scripts/test_no_shadowed_definitions.py` | no definition in either pipeline is shadowed |
| `scripts/test_bnn_kl_term.py` | the Bayesian networks are fitted on the ELBO |
| `scripts/test_spec_is_live.py` | changing the spec changes what is built |
| `scripts/test_generated_job_flags.py` | every flag the generator emits is one the program has |
| `scripts/test_uncertainty_writer.py`, `test_record_alignment.py` | as before |
| `rust/tests/` (33) | the injector, and the record writer |
| KIRBy `tests/smoke/smoke_kirby_splits.py` | scaffold key, acyclic groups, the validation carve |
| KIRBy `tests/smoke/smoke_kirby_target_scaling.py` | what the models are fitted on, and the noise pattern |
| KIRBy `tests/smoke/smoke_kirby_merge.py` | a subset run does not destroy what it did not produce, and a model with no rows is named |
| `scripts/test_condition_names.py` | one noise condition has one name on both injectors, against `condition_names.json` |
| `scripts/check_fixes_fail_when_removed.py` | that each check above fails when its fix is removed |
| KIRBy `tests/smoke/smoke_kirby_uncertainty.py` (80) | as before |

**Still yours.** Twelve entries are real and unfixed because fixing them is a
decision, not a repair:

1. ~~Which auc_norm the paper reports~~ — **settled**: one shared grid in
   fractions of the label spread, no rescaling (§2.12).
2. **`--calibration-size`** for the conformal models: honour it, or refuse it by
   name. Nothing uses it today and the whole validation split is the calibration
   set, which is the better estimator now that no model trains on it.
3. **The heteroscedastic and evidential heads.** Their predicted variance is
   sliced off at every prediction site, so they report the same quantity as a
   plain MC-dropout network. Neither loss is in the roster. Making the head's
   variance the reported aleatoric term changes what those models are.
4. ~~`grouped_shifted` off-registry, and the naming of off-registry conditions~~
   — **settled**: the QM9 rule on both sides, checked against
   `condition_names.json` (§2.16), and the amount of noise fixed with it.
5. **Table 4's pooling** across noise levels, and the cross-dataset figure's
   pooling across conditions and representations. The report now says out loud
   what is pooled; computing them per level changes the table's shape.
6. **`rmse` and `mae` in different units** across the two pipelines. Recoverable —
   the standardisation constants are on every QM9 uncertainty row now.
7. **hERG N.** `paper.tex` says 1,482; the cached extract holds 1,415, and so does
   the module docstring. The loader reproduces 1,415 exactly.
8. The remaining spec literals in models.py (batch size, the MC pass count, the
   flexible/conformal hidden sizes), the retired-scheme methods figure, and the
   latent `randomized_smiles` routes — all listed in `verdicts.json`.

`paper.tex` was not touched.

### 2.14a ✅ FIXED 2026-08-27 — the shifted grouped condition delivered the wrong amount under every shape but Gaussian, in BOTH injectors

Audit entry 34, the last of the noise-amount entries, now closed. It was written
up as latent. It is not latent in the sense the entry assumed: both injectors were
wrong, in opposite directions, and each was wrong on its own terms.

Every other condition draws at the shape's own spread and multiplies by the scale
the dose solver returned. The shifted grouped condition bypassed that in each
implementation, and the two bypassed it differently:

| | what it multiplied by | delivered, Gaussian | Laplace | Student-t nu=5 |
|---|---|---|---|---|
| Python, `noiseInject/core.py` | the target, on draws at the shape's own spread | target | 1.51x | 1.19x |
| Rust, `rust/src/main.rs` | the solved scale, on draws divided down to unit variance | target | 0.71x | 0.77x |
| both, now | the solved scale, on draws at the shape's own spread | target | target | target |

A Gaussian draw has spread 1, so the three conventions coincide there, and the
condition roster is Gaussian at every entry on both sides. That is the whole
reason neither was ever seen. It is also why **no results need re-running**: every
run in the study is Gaussian, where the old code and the new agree exactly, bit
for bit — the seed stream is unchanged, because both halves already called the
same draw and only the constant in front of it moved.

The Rust half is worth stating plainly because the entry recorded the opposite.
`rust/src/main.rs` says in its own comment that the two components are unit
variance "hence G = 1 by construction", but the solver computes G generically as
the shape's spread, so the two disagreed with each other inside one function. On
a real run the delivered-dose gate would have caught it — it runs on all three
splits at line 3323 — so the Rust side would have aborted rather than written a
wrong number. The Python side had no such check and would have written one.

**The Python injector now checks what it delivered**, the same way and against the
same band: `dose_tolerance`, derived from the draw's own fourth moment and its
effective size, is the twin of the Rust function and until now only the tests
called it. Censoring is exempt — it is swept on the fraction clipped and has no
target amount.

**One thing for you, and it is a number rather than an opinion.** That band is a
flat 15% for Student-t at nu <= 4, because the sample kurtosis it is otherwise
derived from is itself meaningless there. Flat means it does not shrink with the
dataset, and the experimental sets are two orders of magnitude smaller than QM9.
Measured over 200 draws of `student_t_nu3` at k=0.5:

| n | single draws outside the 15% band | worst |
|---|---|---|
| 1,415 (hERG) | 7.0% | 98% |
| 4,000 | 2.5% | 47% |
| 20,000 | 1.0% | 20% |
| 133,885 (QM9) | 0.0% | 11% |

So the check is safe where Rust runs it and will stop about one hERG
`student_t_nu3` run in fourteen. Those runs did deliver twice what was asked, so
refusing them is defensible — but whether a heavy-tailed draw on a small set
should abort the run or be recorded and kept is your call, not a repair. Nothing
is red today: all 342 checks in `crosscheck_injectors.py` pass on the real QM9
column, all 28 Rust gates pass, and all 17 conditions agree between the reference
and the pipeline.

**What now compares the two.** `scripts/test_noise_arms.py` ends in an assertion
rather than a table: it drives both injectors on the shifted grouped condition
under Gaussian, Laplace and Student-t at nu=5 and requires that they deliver the
target and agree with each other. Reaching the Rust one needed a way in, because
the roster is Gaussian throughout — `--self-test --json` now runs a single named
shape-and-targeting pair when both flags are given, and the roster path is
untouched. No noise algebra changed for that.
`rust/reference/noise_arms.rs` cannot take part at all: it fuses the shape and
the targeting into one type, so its shifted grouped entry has no shape to set.

Three new cases in `scripts/check_fixes_fail_when_removed.py`, each confirmed red
with its fix removed: the Python scale, the Python delivered-amount check, and
the agreement between the two injectors.

---

## 3. Two pipelines, one design

This is the structural fact that organises the whole re-run, and it was never written down
clearly.

**Pipeline one — QM9.** `scripts/process_and_train.py` prepares the data, shells out to the Rust
binary for representation building and noise injection, and reads the memory-mapped files back.
Models live in `models/models.py`. Output: one `results/anova_{type}_{rep}_{model}.csv` per cell.
Ten replicates, each a fresh random 10,000-molecule subset with its own scaffold split.

**Pipeline two — the experimental datasets.** `KIRBy/tests/alternative_data_noise_robustness.py`,
with noise from the `NoiseInject` Python package. Five-fold scaffold cross-validation.

**Label standardisation there is split by model type, and an earlier note in this file got it
wrong.** The tree and Gaussian-process models never standardise the label, so for them the noise
level is directly in log units — which is what makes anchoring it to published assay error
possible. The neural models *do* standardise it (`alternative_data_noise_robustness.py:698-704`),
and the scaler is fitted on the already-noisy training labels, which is the same confound as §2.4.
It is mitigated rather than absent: predictions and predicted uncertainties are inverted back to
raw units before anything is scored (`:768-781`). Say this explicitly rather than describing that
pipeline as unstandardised.

Consequences that follow:

- The noise redesign must be implemented **twice**, and the two implementations cross-checked
  against each other on the same labels. There is already a working Rust reference at
  `rust/reference/noise_arms.rs` that agrees with an independent Python version to within half a
  percent (`NOISE_DESIGN.md` §5.1b).
- ✅ **The noise design now covers both injectors.** It used to name `rust/src/main.rs` and nothing
  else, which left the half that produces the three experimental datasets *and* every uncertainty
  number unspecified — the same omission that let the two implementations drift in the first place.
  Closed 2026-08-26 (`8de0eed`): `NOISE_DESIGN.md` §6.0a names both implementations and their
  callers, §6.1 puts the six superseded Python strategies and `calibrate_sigma` on the delete list,
  §6.2 step 6 specifies the Python build, and §6.3 items 6–8 make agreement between the two a gate.
- The standardisation defect (§2.4) is a QM9-only problem.
- The validation-noise defect (§2.5) exists in both, and is already fixed in one.
- The uncertainty machinery — out-of-fold scoring, recorded noise, the confound controls — exists
  **only** in pipeline two.

### 3.1 What the uncertainty work already does

Built and regression-tested, not submitted. From
`slurm_scripts_uncertainty_rerun/RUNBOOK.md` and the code:

- **Question A — do the corrupted molecules come back as the uncertain ones?** Measured on
  *training* molecules, scored out-of-fold, so no molecule is judged by a model that fitted its
  own bad label. Without that, a Gaussian process has zero posterior variance at its own training
  inputs and a forest has memorised the row; you would be measuring memorisation.
- **Question B — does the model learn where the data is unreliable?** Measured on test molecules
  against the noise scale their region receives.

**There is a third question, it was agreed with the others, and it fell out of the run plan.**
On 2026-08-23 the set was settled as three, not two, and the third is the one most uncertainty
papers actually ask:

| Question | Measured on | What it is useful for |
|---|---|---|
| Which of my measurements are bad? | training molecules, scored out-of-fold | cleaning your data |
| Does noisy training data make the model less sure about new molecules? | test molecules, uncertainty against the noise level | knowing the damage |
| **When the training data is noisy, does uncertainty still tell you which predictions to trust?** | **test molecules, uncertainty against test error** | **using the model** |

The third one costs nothing — the numbers are already produced by every run — and it is the
standard calibration question, so it is the one a referee will expect. It is in neither the run
plan nor the analysis section of the runbook. **Add it.** One caveat carried over from the earlier
review: measure the error against the *clean* label, not the noisy one, or the error you are
ranking contains the noise you are correlating against.
- **The confound in B is controlled properly.** The noise scale is a deterministic function of
  the label, and uncertainty may already track the label because extreme molecules are simply
  harder. So every row carries a second column — the *shape* of the noise at a fixed reference
  level, identical at every noise level including zero — and the reported effect is the
  correlation at a given level **minus** the correlation at zero. The zero-noise model saw the
  same label distribution but no corruption, so its correlation is exactly the confound.
- Nine defects were found in adversarial review and fixed, the most serious being that the
  out-of-fold split was **random** rather than scaffold-based, which broke the project's own
  splitting rule and put out-of-fold uncertainty in an interpolation regime while the test set
  was in an extrapolation regime. Now `GroupKFold` on Murcko scaffolds, with a logged fallback.
- The recorded noise reconstructs the corrupted label exactly, and at zero noise it is exactly
  zero — a real negative control, which the old reconstruction never was.

**This machinery is sound and should be kept.** What changes is the set of noise types it runs and
the levels it runs them at — plus one live bug, below.

### 3.1a ✅ FIXED 2026-08-27 (chat F) — the check on training rows that never fired

The check recomputes the noise shape from the model's *predicted* label rather than the true one.
If uncertainty correlates with that just as strongly, the model is tracking its own prediction and
not the noise. Every training row the pipeline has ever written left that column empty.

The guard `if sigma in extras['oof_mean']:` sat thirteen lines above the line that writes the
current noise level into `extras['oof_mean']`, inside the same pass of the same loop. The
identical dead block sat in the neural runner, thirty-two lines above its write. Both blocks now
sit below the block that fills those values in.

**The check that catches it if it comes back:** `tests/smoke/smoke_kirby_uncertainty.py`. Removing
the fix and re-running gives five failures, among them *"noise_pattern_pred is populated and
varying on training rows — 0 distinct values, 288 blank"*, and a non-zero exit. Executed
2026-08-27.

Why nothing caught it before: the regression test read the pipeline as text and searched for a
matching string (`smoke_nine_fixes.py:79`, `src = open(PIPE).read()`). A string match passes
whether or not the matched line ever runs. Of that file's 25 checks, 11 were string matches, two
were tautologies — the one for the random-number generator reduced to `torch.equal(mid, mid)` and
never touched the pipeline — two confirmed a defect rather than a fix, and one of the nine had no
check at all while the banner printed `ALL NINE-FIX CHECKS PASSED`. It has been replaced.

**The probes that DID execute the code exist and were archived, not shipped.** `d1_oof_rng.py`,
`d1_state.py`, `t3.py`, `t3b.py`, `t4.py`, `t4b.py`, `t5.py`, `d78.py`, `d78b.py`, `d7b.py` and
`d9_val.py` are in `research_archive/e1d07839/`, timestamped between the review and the commit.
Each hardcodes a session scratchpad path, so none was runnable anywhere else. What they assert has
been ported into the replacement.

### 3.1b ✅ FIXED 2026-08-27 (chat F) — the silent no-ops, and three more like them

- Per-molecule uncertainty was written only if the zero noise level was in the run
  (`if save_this and uncertainties.get(0.0) is not None`). The out-of-fold block was nested inside
  that guard, so a level grid without zero wrote **no uncertainty at all, test rows included**,
  after paying for every cross-fitting fit — exit status 0, no warning, and a stale file from an
  earlier run left in place looking current. One expression was doing two unrelated jobs. Split
  apart: whether the model emits a per-molecule uncertainty is now tested across all levels, and a
  missing zero level raises with the reason (the zero level is the negative control the whole
  subtraction rests on).
- `--unc-strategies` defaulted to `legacy`, so the only condition written was plain Gaussian —
  whose noise scale is constant, which is the one condition where "does uncertainty find where the
  noise is" has no answer. Renamed `--unc-conditions` and defaulted to `all`. Passing a single
  non-Gaussian condition without it used to write nothing at all, silently.
- `--oof-folds 1` was silently ignored (`if oof_folds and oof_folds > 1`). Now refused, in both
  pipelines.
- `UNCERTAINTY_MODELS` was matched by exact equality, so `GP-Tanimoto` got no training-side
  scoring and nothing said so. Matched by prefix, and a model that emits an uncertainty but is not
  matched now raises.
- The noise generator was built once outside the level loop, so each level consumed a draw in grid
  order: `--sigmas 0.0 0.6` and `--sigmas 0.0 0.3 0.6` corrupted differently at 0.6, and any
  gap-filling resubmission drew noise unrelated to the original. The seed is now derived from the
  level itself, through `zlib.crc32` rather than `hash()` — Python randomises string hashing per
  process, so `hash()` would have made the noise differ on every run. ⚠️ **This changes the
  realised noise relative to any run made before it. Files from the two are not mergeable.**

Row counts are now asserted after every write, so a condition that produces no rows stops the run.

### 3.1c ✅ 2026-08-27 — QM9 can now answer the uncertainty question, and what it cost

§2.6 said QM9 could not answer it at all: uncertainty was saved for held-out molecules only,
corruption entered training only, and the noise was reconstructed rather than recorded. The author
settled the scope on 2026-08-26 — *"QM9 is the core results of the paper because it is the only
confirmed clean dataset. Make the change in QM9 that is non-negotiable."* All three are now closed.

- **The noise is recorded, not reconstructed.** The injector already wrote a per-molecule
  provenance file; it now also writes the amount applied to each molecule and the level-free shape
  of the condition. Python reads it, keys it by canonical SMILES, and asserts the molecule on the
  way out is the one the noise was drawn for.
- **Training molecules are scored by models that never fitted them.** One shared routine,
  `GroupKFold` over Murcko scaffold groups, wired into the quantile forest, NGBoost, the Gaussian
  process and both neural families. The torch generator is snapshotted around the extra fits, so
  the main result does not move.
- **Validation labels carry their own noise**, drawn from a separate seed and dosed against the
  clean TRAINING spread rather than validation's own. Measured on a real run: validation received
  0.760882 label units against training's 0.766289, both anchored on a training spread of 1.258461
  where validation's own is 1.331802. A gate compares the two and stops the run if they diverge.
- **Held-out molecules stay clean**, and their recorded noise is exactly `0.0`, not a small number.

**Every result row now carries its condition's name.** Without it two noise types land in one column
with nothing to separate them, and every statistic computed over the file pools across a dimension
it has to condition on. The analysis module's list of known conditions was the pre-redesign one, so
a real run's file matched nothing; it now reads the registry from the injector itself.

### 3.1d 🔴 FOR THE METHODS — a scaffold split makes one question unanswerable on held-out molecules

**Owner: chat J** (assigned 2026-08-27). It was written up and marked "for the Methods" with no chat
attached, which is how it would have been missed — chat J is the only chat that touches what gets
written up.

Found by running the pipeline, not by reading it. The grouped conditions are keyed to the scaffold
group, and the splitter holds whole scaffold groups out. So a held-out molecule is in a group the
training selection never marked, and its level-free shape is **flat**.

Flat is the truthful answer there — the condition never reached that region — so the shape must not
be redrawn for held-out molecules, or the file would record an injection that did not happen. Two
consequences, and both belong in the paper rather than in a comment:

1. **For the grouped conditions, "does the model become less certain where the data is unreliable"
   is answerable on the out-of-fold TRAINING rows and undefined on held-out molecules.** It stays
   answerable on both splits for censoring, which is keyed to the label and so is defined anywhere.
2. **Validation is different**, because validation receives noise. Handing it a selection that
   reaches none of its molecules would give it plain Gaussian noise under the name of a grouped
   condition. It draws its own selection at the same molecule fraction.

Why every test passed while the first real run died: the injector's test fixture put held-out
molecules in the **same** scaffold groups as training, which is a random split. Every experiment in
this project uses a scaffold split. A fixture with disjoint groups is now in
`rust/tests/noise_gates.rs`, and the test fails if either half of the rule above is removed.

**A second property to state rather than discover later.** The check that recomputes the shape from
the model's own predicted label is a real control for a *label-keyed* condition and is degenerate
for a *group-keyed* one: a predicted label does not change a molecule's scaffold, so the recomputed
shape is identical to the true one. Measured on both datasets — the two correlations agree to every
digit. Report it for censoring; do not report it as a control for the grouped conditions.

### 3.1e ✅ FIXED 2026-08-27 — the out-of-fold pass asked for validation rows no model fits

**Found by running NGBoost on QM9 rather than by reading the code.** It died at the guard:
*the model fits 4000 rows but the recorded noise covers 4250*.

When the author settled on 2026-08-27 that **no model stacks validation into its training set**
(§2.12), the fitting code changed and the provenance slices did not. Three of the five callers in
`models/models.py` went on describing the old regime: the forest took the `slice(None)` default,
and NGBoost and the Gauche GP each passed the first half of validation. Every one of them asked
the recorded noise for molecules its model no longer sees.

**The out-of-fold pass was therefore dead for all three tree and kernel uncertainty models** from
the moment the merge was removed, and it would have taken the deep run with it. The guard refused
rather than attributing one molecule's noise to another — which is the original QM9 defect, and
exactly what the guard is for.

`val_slice` now defaults to `None`, so forgetting is the safe case; a family that genuinely fits
part of validation has to say so. Proved on QM9 at 5,000 molecules, ECFP4, level 1.5, three
cross-fit folds: NGBoost and the quantile forest each wrote 4,000 out-of-fold rows against 4,000
fitted, 3 of 3 inner folds, R² 0.695 and 0.713.

✅ **The experimental pipeline does not have it.** `KIRBy`
`tests/alternative_data_noise_robustness.py` never merges validation into training — there is no
`vstack` or `concatenate` in the file — and both out-of-fold calls (`:1542` tree, `:1716` neural)
pass `X_train` with the `y_noisy` drawn for that same split. Checked, not assumed.

⚠️ **Two things this says about the QM9 uncertainty results that already exist.** Any QM9 run made
with cross-fitting since the merge was removed wrote **no training rows at all** for the quantile
forest, NGBoost or the Gaussian process — the question about which molecules were corrupted has
never been answerable from them. And the local environment hid it: every QRF fit on this laptop
raises `Invalid parameter 'monotonic_cst'` (scikit-learn 1.3.2 against a 1.6.1 pin), so the one
model most likely to have surfaced this never ran here.

### 3.2 What the noise redesign does to the uncertainty questions

Nobody has written this down, and it is the one place where the two workstreams collide.

The redesign drops the four label-keyed noise types. Under the replacement set, question B —
which needs a *learnable pattern* of unreliability — has a much cleaner structure than before:

| New noise type | Is there a pattern to learn? |
|---|---|
| Gaussian, Student-t, Laplace | **No, by construction.** Every molecule gets the same scale. The correlation is undefined, not zero. These are question A's conditions and the leakage check. |
| Grouped by scaffold | **Yes, and it is predictable from structure.** This is the positive case, and the only one where a model could genuinely spot bad data from the features it has. |
| Outlier, random selection | **No — a true null.** Victims are chosen at random, so nothing in the features predicts them. This is the honest negative control that the old design lacked. |
| Censoring | **Yes, and keyed to the label.** Which molecules get clipped is a deterministic function of the label, so the zero-noise subtraction is doing real work here. |

That is a better design than the six it replaces, and it answers the open question in
`NOISE_DESIGN.md` §7.2: **censoring is the deliberately label-keyed condition**, so a separate
artificial positive control is not needed.

### 3.3 Parity — what each pipeline actually has

Established 2026-08-25 by reading all three repositories, with every row cross-checked against the
cited file. This is the spine of the re-run: almost every remaining task is closing one of these
gaps.

The short version is that **the two pipelines are near-mirror images**. The experimental pipeline
has the whole uncertainty apparatus and no repeats. QM9 has real repeats and none of the
apparatus.

| Capability | QM9 | Experimental | What has to happen |
|---|---|---|---|
| Independent repeats of the whole experiment | **10** | **none** — 5 folds only, every model pinned to one seed | The experimental variance decomposition has no true replicate term |
| Out-of-fold scoring of training molecules | no | yes | QM9 cannot ask the uncertainty question at all (§2.6) |
| Inner split grouped by scaffold | no — validation is halved by position, and `scaffold` appears **zero** times in `models/models.py` | yes, `GroupKFold`, with a logged fall back to random when a fold has too few scaffolds | QM9's calibration set sits on a different split geometry from its test set |
| The injected noise recorded per molecule | ✅ recorded (chat A) | ✅ recorded, and now with the full provenance beside it (chat B) | Done on both sides. Every result row carries `unit_dose_g`, `solved_scale`, `target_dose_label_units`, `realised_dose_label_units`, `realised_dose_fraction_of_spread`, `mean_epsilon`, `affected_molecule_fraction`, `effective_n` and the clean standardisation constants — under **the same column names in both pipelines**, checked by the cross-check |
| Per-molecule noise scale recorded | no — computed then discarded (`main.rs:309-317`) | yes | Same serialisation change |
| Held-out noise scale computed against the *training* cut-points | no | yes | Follows once QM9 records a scale at all |
| The level-invariant noise pattern (the confound control) | no — appears nowhere | ✅ yes, and **it is now genuinely level-invariant** | Chat B found it was not. The scale map drew from the same generator as the noise, so the "pattern" moved every time it was recomputed and the zero-noise subtraction compared two different patterns. The selection is now drawn from a separately seeded generator, re-seeded per call, so it is a deterministic function of (seed, groups, parameters) and identical at every level including zero |
| The placebo check on training rows | no | **blank — see §3.1a** | Two-line fix in the experimental runner |
| Validation labels carry their own noise | no | yes, on by default | The headline asymmetry, and one day old rather than architectural — before the held-out fix QM9 noised *both* held-out splits |
| Validation kept out of the training set | no — seven model families merge it | yes | §2.5 |
| Best weights kept after early stopping | **no — see below** | yes, snapshot and restore | Real defect, listed below |
| A stable molecule identifier | no — a row position within a split | yes, added 2026-08-24 | No per-molecule analysis across models or conditions is possible on QM9 without it |
| Zero noise is a recorded zero, not an estimate | no | yes | §2.6 |
| A warning when a noise type's scale is constant on a dataset | no | yes | Belongs in the shared package so both pipelines get it |
| Folds or replicates surviving into the reported numbers | partial — the variance decomposition keeps them, the ranking tables do not | partial — the fold column is written, then dropped by the consumer at `generate_paper_figures_v2.py:1148-1152` | §5.4 items 1 and 2 |

**A defect this turned up that belongs in §2.** QM9's neural training never keeps the best
weights. It tracks the best validation loss as a number, counts epochs without improvement, and
breaks (`models/models.py:1872-1880`, patience 20). There is no snapshot and no restore anywhere
in that function. So validation controls *when* training stops, not *which* model you end up with
— the weights kept are the last epoch's, twenty past the best. Any statement that the QM9 neural
models are selected on validation performance is wrong as the code stands. The experimental
pipeline does snapshot and restore (`:759`, `:765`).

**And one more, cheap to fix.** The two last-layer Bayesian job scripts pass
`--bayesian-transformation last`, which is not a value the code recognises — it accepts `full`,
`last_layer`, `variational` and `full_variational`. Those models therefore train as ordinary
networks while still being flagged as Bayesian.

### 3.5 ✅ The within-noise-level fix — the author's, and it is already in the code

**Recorded 2026-08-26, after the author twice asked where this had gone.** It was never lost. It
was previously written down only as a failure mode (guard 1, §0.6), which buried the fact that it
is a *fix the author identified and that produced a real positive result*.

**The fix.** The uncertainty-versus-noise correlation must be computed **within each noise level**,
never pooled across levels. Pooling stacks the levels into a staircase: mean uncertainty rises with
the noise level and so does the mean size of the injected noise, so the correlation reports that
shared ramp — a population trend — and it gets read as per-molecule detection.

**Where it is.** `scripts/generate_paper_figures_v2.py:1031-1057`, `within_sigma_unc_noise_rho`,
with `WITHIN_SIGMA_LEVELS` at `:1028`. Its own docstring closes with *"Fully disaggregated — the
caller must NOT average across σ (or across strategies)."* It is called at `:3433` and `:3564` and
its output is written per level, one column each.

**What it produced.** Recomputed within each level, the paper's uncertainty finding reversed. The
Gaussian process — the paper's example of a model with an explicit observation-noise term — came out
at approximately zero at every level, because a single global noise term *cannot* be per-molecule.
The genuine detector was a Bayesian network on PDV under subset-targeting noise.
That is a mechanism-consistent result, not a rescue.

**Three things that must carry into the re-run.**

1. **The pooled version still exists beside it** (`:3428-3430`, written to the same table). Keep one
   or label both unmistakably; two numbers for the same thing under similar names is guard 12.
2. **The zero-noise control must be clean.** One model's apparent signal rode on a zero-noise
   correlation of about 0.22 — a label-magnitude confound, not detection. The KIRBy rebuild handles
   this properly by subtracting the zero-noise correlation (§3.1); the QM9 path never did.
3. ✅ **FED 2026-08-27 (chat F). On QM9 the fix was correct but starved.**
   `save_uncertainty_values` did not record the injected noise — it *reconstructed* it by
   regressing the noisy label on the clean one and keeping the residuals. After the held-out fix
   the test labels are an exact affine function of the clean labels, so those residuals were
   identically zero and there was nothing left to correlate against. The regression is deleted. The
   injector's recorded draw is read from the provenance file and written verbatim, and a training
   row that arrives without its recorded noise now raises rather than being filled in.

**Status 2026-08-27:** the fix is correct, implemented, load-bearing, and now fed on both
pipelines. What it is fed on QM9 is the recorded draw, not an estimate of it, verified on a real
run (§3.6).

---

### 3.3a 🔴 FOR YOU — the experimental pipeline is not fold-independent, and never was

Measured 2026-08-26 (chat B), because chat B's own prompt names this as a property that must be
preserved.

`KIRBy/src/kirby/noise_spec.py` exists to guarantee two things, and its docstring says so: a fresh
generator per call, and **one realisation drawn over the whole label column, then subset per fold,
so a molecule gets the same corruption whichever fold it lands in**. Both are intact there — checked.

**The experimental pipeline does not go through that module.** It imports the injector directly and
draws over *each fold's* training labels. Measured on a 600-molecule five-fold scaffold split:
**1,688 of 2,400 molecule-fold pairs received different noise from the same molecule in an earlier
fold.** The delivered dose is stable (0.27% spread across folds), so the robustness curves are
unaffected — what changes is what a *molecule* means.

**This is a design choice, not obviously a defect, which is why it is yours.**

| | Draw once over the full column | Draw per fold (what it does now) |
|---|---|---|
| A molecule's corruption | a property of the molecule, so rows join on `mol_idx` across folds | differs by fold; joining across folds mixes five realisations |
| Noise-realisation variance | none — one draw for the whole study at each level | the five folds give five draws, which is real protection against a freak realisation |
| Matches `noise_spec`'s stated guarantee | yes | no |

**I have not changed it**, because it would alter every experimental number and the argument is
genuinely two-sided. My recommendation is to **keep the per-fold draw** — with only one replicate
planned on the uncertainty side (§13.1 stage 3), five independent realisations are worth more than
cross-fold joinability — and to state it in the Methods in one sentence.

**The one thing that must happen either way:** the per-molecule uncertainty rows carry `fold`, and
chat J must not average `injected_noise` across folds for a molecule. Under the per-fold draw that
is averaging five different corruptions. This is failure mode 3 in §0.6 wearing a new hat.

### 3.4 Cross-pipeline audit — the same name is not the same thing

Audited 2026-08-25 by reading both codebases, with every claim independently re-checked. **29
confirmed differences: 17 that invalidate a finding, 9 that change numbers, 3 cosmetic. 17
confirmed matches.** The five worst are representation-identity errors, and one of them means the
paper's Methods are wrong about its most-used representation.

I verified the five highest-severity findings myself, in the source, before writing them here.

#### 3.4.1 Four representations share a name and are not the same features

| Name | QM9 | Experimental | Verdict |
|---|---|---|---|
| **ECFP4** | **not a circular fingerprint at all** — `rdk_fingerprint_mol` (`rust/src/main.rs:22`, called at `:822`) binds `RDKFingerprintMol`, RDKit's Daylight-style **topological path** fingerprint | a genuine Morgan radius-2 fingerprint via `GetMorganGenerator(radius=2, fpSize=2048)` | ❌ **different family** |
| **PDV** | `pdv` is **binarised** — `(pdv > 0).astype(np.uint8)` then packed (`process_and_train.py:377`). The real one is the separate `continuous_pdv` | continuous descriptors, standardised | ❌ the merge layer must map to `continuous_pdv` |
| **SNS** | ~~counts computed (`sub_counts=True`) then **thrown away** by `packbits`~~ ✅ **FIXED 2026-08-27 (chat M)** — 1,024 counts as 16-bit integers, and neither side standardises them (§3.4.3a) | raw counts, and no longer standardised | ✅ **now agrees** |
| **MHG-GNN / ChemBERTa** | ~~per-molecule min-max to 8 bits, never standardised~~ ✅ **FIXED 2026-08-26** — 32-bit floats, standardised per feature on the training split, so both pipelines now build the same features (§2.8c). mol2vec had the same defect and has been deleted | full precision, no quantisation, standardised | ✅ **now agrees** |

**The fingerprint one is the most serious thing found in this whole audit.** `paper.tex:203`
describes ECFP4 as *"circular substructures with radius r=2"*. On QM9 that is false — it is a
path-based fingerprint, a different substructure family with different bit density and different
similarity behaviour. Around 254 job invocations pass `-r ecfp4`.

It is a wrong-function bug rather than a naming slip: the same crate file ships
`morgan_fingerprint_mol` and it is never imported. Note that binding is **radius 3**, so it is
ECFP6 — switching to it would still not give ECFP4, and a new binding is needed.

**A second, quieter data-loss bug in the same area — ✅ FIXED with the first.** Because the
substructure counts were cast to `uint8` *before* being packed, a count that was an exact multiple
of 256 wrapped to zero and the substructure recorded as **absent**. Rare, silent, wrong. The record
now holds 16-bit counts and the writer **refuses** a value that will not fit rather than wrapping
it. Guarded in `scripts/test_embedding_storage.py` §6 and §7, which store a count of exactly 256 and
require the old path to fail on it.

**Only one representation pair is genuinely comparable across the two studies:** QM9's
`continuous_pdv` against the experimental `PDV`. Both end as train-only per-feature standardised
descriptors. Every other cross-pipeline representation claim in the paper rests on a name, not on
the features.

#### 3.4.2 Several models share a name and are not the same model

| | QM9 | Experimental | Consequence |
|---|---|---|---|
| **NN-β width** | **[128, 128]** — passed explicitly | **[32, 32]** — the call passes no width and takes the class default | ❌ Sixteen times fewer hidden parameters under one name. The class docstring on the experimental side *claims* it matches QM9; it matches the class default instead |
| **Forest feature sampling** | `max_features='sqrt'` | not passed → library default **1.0**, every feature at every split | ❌ A fundamentally different forest. With 2048-bit inputs that is 45 features per split against 2048 |
| **XGBoost learning rate** | 0.1, pinned | not passed → library default **0.3** | ❌ Three times the step at the same round count. This is the model the paper says collapses on the experimental data |
| **Quantile forest trees** | 300 | 100 | Changes the quantile estimates, which are that model's deliverable |
| **Stochastic passes** | **100** | **30** | The paper says 100 |
| **Gaussian process** | default kernel Tanimoto; uncapped | RBF; capped at 2,000 molecules | Different kernel *and* different training-set size |

#### 2.10 🔴 FOUND AND FIXED 2026-08-26 — Sort & Slice on QM9 was trained on PERMUTED features

**The most damaging thing found in the whole audit, and a previous session had checked it and
concluded the opposite.**

The substructure fingerprint's training molecules were queued by iterating the training indices.
The record writer iterates positions in **ascending** order. On reaching a training molecule it
popped the next molecule off that queue, overwrote the current one with it, and computed the
fingerprint from the popped molecule — while the label, the canonical SMILES and every other
representation in that record came from the row's own position.

**The scaffold splitter does not return its indices in ascending order.** I checked by running it
rather than reading it. So the two orders diverge almost immediately:

| molecules | training rows | rows holding ANOTHER molecule's fingerprint |
|---|---|---|
| 500 | 400 | **383 (95.8%)** |
| 2,000 | 1,600 | **1,584 (99.0%)** |

**This is not bias, it is destruction.** Features and labels no longer correspond, so the model is
fitting noise. Every Sort & Slice result on QM9 to date is meaningless — not degraded, meaningless.

**Scope, checked rather than assumed.** The corruption is confined to this one representation: the
molecule variable is set to `None` immediately afterwards, and the descriptor vector, the
fingerprint and the learned embeddings are all built from the row's own canonical SMILES. Training
rows only.

**The fix is a deletion.** The queue was only ever needed to FIT the substructure vocabulary, which
happens before the writing loop; the featuriser is an ordinary function of one molecule. Both
writers now featurise the row's own molecule.

**Guarded.** `scripts/audit_representation_identity.py` refits the featuriser on the very molecules
the pipeline used and compares row by row, so a return of this fails loudly rather than silently.

**Why it survived.** An earlier session examined exactly this and recorded that "the QM9 path
queues and consumes in the same ascending order and is correct". That conclusion was reached by
reading the code. It is wrong, and one line of execution — printing whether the splitter's indices
are sorted — settles it. The note never reached this document, so nothing here needed correcting,
but the lesson is the same one this project keeps paying for: **reading is not checking.**

**✅ THE THREE EXPERIMENTAL DATASETS ARE NOT AFFECTED — tested, not read.** The author asked
directly. That side builds each row from its own molecule and has no queue to drain, so the fault
cannot occur there; but since reading is what got this wrong the first time, it was measured
instead. Shuffle the input molecules, recompute, and check the rows follow the shuffle:

| representation | rows tracking their own molecule |
|---|---|
| ECFP4 | 100.0% |
| PDV | 100.0% |
| SNS | 100.0% |

So the damage is bounded to QM9 and to that one representation. Nothing on the experimental side
needs regenerating for this reason. Re-checkable at any time, and it needs no argument about what
the code says:

```bash
python scripts/audit_representation_identity.py --experimental-alignment --strict
```

One limit worth stating: it tests featurisation, not the later fold splitting — though that splits
by index arrays, so rows and labels move together by construction.

#### 3.4.3a 🔴 FOUND AND FIXED 2026-08-26 — the two sides scaled features differently

**The largest thing the second parity pass found, and no settings comparison could ever have seen
it**, because it is not a model setting. It is a preprocessing step applied to one class of
representation and not another.

| | which representations were scaled |
|---|---|
| QM9 | only the continuous ones. Fingerprints reached the models as raw 0/1 bits |
| experimental | **everything**, unconditionally, in both the tree path and the neural path |

Irrelevant for trees, which are scale-invariant. It changes the support vector machine and the
Gaussian process completely: both measure distances with a radial kernel, and standardising a
sparse binary fingerprint hands its rare bits enormous magnitudes so they dominate every distance.

**Measured on both, because the size differs a great deal:**

| | raw 0/1 bits | standardised | cost |
|---|---|---|---|
| QM9, 2,000 molecules, Morgan, three seeds | +0.815 / +0.809 / +0.832 | +0.320 / +0.124 / +0.182 | **−0.50 / −0.69 / −0.65** |
| hERG, all 1,415 molecules, ECFP4, five folds | **+0.5335** | +0.4570 | **−0.077**, every fold agreeing |

⚠️ **The direction is identical everywhere; the size is not.** About 0.6 on QM9 against about 0.08
on hERG. Do not carry the QM9 figure across — I did exactly that in a summary and it was wrong.
QM9's target is far more predictable to begin with, so there is more to lose, and its molecules are
small enough that their fingerprints are much sparser than drug-like ones.

**What it costs the paper.** Any statement that the support vector machine or the Gaussian process
does worse on the experimental datasets than on QM9 now has a candidate explanation with nothing to
do with the data or the noise. Re-check those after the re-run.

**The fix, and the mistake inside it worth recording.** The rule is now in the shared spec and both
pipelines read it — but it is keyed on **the matrix, not the name**. My first attempt used a name
list and was wrong within minutes: `pdv` on QM9 is the BINARISED vector while `PDV` on the
experimental side is the CONTINUOUS one. One name, opposite correct answers — the same trap as
§3.4.1's fingerprint. Binary means never scale; anything else means scale, fitted on training rows
only. That follows the data when the binarised vector and the flattened counts are fixed, with no
list for anyone to forget.

✅ **MEASURED AND SETTLED 2026-08-27 (chat M) — sparse counts are exempt too.**
`scripts/parity_test_count_scaling.py`, results in `results/parity_tests/count_scaling.csv`. Sort &
Slice counts through the pipeline's own featuriser, scaffold split, five seeds, the same
radial-kernel support vector machine, against a decision rule fixed before the run. Standardised
counts minus raw counts:

| | per seed | mean | seed-to-seed spread | standardising wins |
|---|---|---|---|---|
| QM9, 4,000 molecules | −0.070, −0.061, −0.107, −0.039, −0.089 | **−0.073** | 0.026 | 0/5 |
| hERG Ki, 1,415 compounds | +0.009, −0.022, −0.061, +0.015, +0.084 | +0.005 | 0.053 | 3/5 |

QM9 loses on every seed by nearly three times its own run-to-run spread; hERG is a coin flip. So it
is the same harm as the binary case — rare features blown up until they dominate the distances — and
it does not go away when a presence bit becomes a count. **The exemption is now written as a rule
about sparse features rather than binary ones** (`is_sparse_count_matrix`: non-negative integers,
under 25% dense, at least one value above 1). The forest control moved by at most 0.005, as it
should: standardising is monotone per feature and trees split on thresholds.

**Why the threshold is where it is.** The substructure counts run at 1.3% density on QM9 and 4.7% on
hERG. The descriptor vector is the only other candidate — half its columns hold non-negative
integers — and it runs at 39% density and is not all-integer anyway. Nothing is near the threshold
from either side.

Guard: `python scripts/parity_test_count_scaling.py --self-test`, which fails if the rule is taken
back out, in a second and without data.

**A second defect found while repairing a bug I introduced here.** The record reader chose its
storage type from that same name list, so any representation not on it was cast to an 8-bit
integer — and a value above 255 would **wrap silently**, a count of 256 reading back as absent.
That is the known wrap risk, but on the *reader* side, where nobody had looked. The narrow type is
now used only when it provably cannot lose anything.

#### 3.4.3 Protocol differences

- **Early stopping.** QM9 **never restores the best weights** — it counts patience and breaks,
  returning the last epoch, up to twenty past the validation optimum. The experimental side
  snapshots and reloads. Under injected noise the gap widens with the noise level, because those
  extra epochs memorise corrupted labels. **QM9's neural degradation curves are therefore steeper
  for a procedural reason**, and the stochastic uncertainty passes are drawn from the overfitted
  weights.
- **Validation labels.** Clean on QM9 since the held-out fix; noised by default on the
  experimental side. Opposite directions (§2.5).
- **Validation split.** Folded back into training for seven QM9 model families, discarded on the
  experimental side — roughly a 29% difference in training-set size.
- **Uncertainty calibration.** A temperature fit exists only on QM9, **and the figure script
  silently prefers the calibrated column when it is present.** So one study's uncertainties are
  calibrated and the other's are not, and the analysis does not say which it read.

#### 3.4.4 Confirmed identical — do not re-audit these

LightGBM on all nine hyperparameters. The support vector machine, keyword for keyword. The NGBoost
constructor. The quantile-forest uncertainty definition. The NN-α architecture. The full-Bayesian
transformation and all its priors. The variational layer, loss and constants, byte for byte. The
optimiser and learning rate. The epoch budget. The noise-level grid, and the fact that on both
sides noise is added to raw labels before any standardisation. The accuracy estimators themselves.
The Gaussian-process model class. The Sort & Slice featuriser settings. And
`continuous_pdv`/`PDV`, the one honest representation pair.

#### 3.4.4b Which settings are actually better — measured, not assumed

Two benchmarks were run. **Where they disagree, trust the production-scale one.**

**Random forest feature sampling, at QM9 production scale** (10,000 molecules, 80/10/10 scaffold,
3 seeds, clean and at half a label spread of noise, 36 fits), paired against the current `sqrt`:

| Representation | Noise | all features | 30% features |
|---|---|---|---|
| descriptors | clean | −0.001 | +0.002 |
| descriptors | noised | **−0.011 (loses all 3)** | −0.000 |
| Morgan | clean | +0.034 | **+0.046 (wins all 3)** |
| Morgan | noised | +0.016 | **+0.027 (wins all 3)** |

The answer depends on the representation. On the descriptor vector, `sqrt` is fine and using every
feature is *worse* under noise — 208 descriptors means the square root is about 14 per split, which
is enough. On the fingerprint `sqrt` is clearly bad: 45 bits sampled from 2048 mostly-zero bits
rarely lands on anything informative.

**A smaller earlier benchmark at 3,182 training molecules found every feature winning everywhere,
and that reversed at production scale.** More training data is what lets a narrow feature sample
find the signal. Recorded because it is a good reminder that a hyperparameter conclusion drawn at a
convenient scale need not hold at the real one.

**Recommendation: 30% on both pipelines** — the only setting that wins or ties in all four cells,
about three times faster than every feature, and free on the primary representation.

**Other settings, from the 4,000-molecule benchmark** (3 seeds, 2 feature sets, clean and noised):

- **XGBoost: the pinned 0.1 beats the unset 0.3, and the gap widens under noise** — up to −0.050 R².
  That is the fast rate fitting the injected corruption. Only the rate actually differed; the other
  nine pinned values already equalled the library defaults.
- **LightGBM: bit-identical.** Same parameters, zero R² difference in all twelve runs.
- **Quantile forest tree count: +0.003 R², inside the seed spread.** Holding the feature setting
  fixed, tripling the trees buys almost nothing. The gap between the pipelines was the feature
  setting all along.
- **Two proposed alternatives are not wins.** Wider LightGBM leaves are worse, and worse under
  noise. A slower XGBoost with more rounds is inside the seed spread at three times the cost.

#### 3.4.4c What has already been changed

Applied to the experimental pipeline on 2026-08-26, so the parity table above is now partly
historical:

| Change | Effect |
|---|---|
| NN-β width 32 → 128 | Matches QM9 and the paper. ~38% more per epoch. Invalidates every row for that model family |
| XGBoost: all ten parameters pinned | Baseline accuracy will **fall** — a slower rate at a fixed round count is a less-fitted model. Correct, not a regression |
| Quantile forest 100 → 300 trees | Three times the fit cost, and it is cross-fitted, so check the wall-clock headroom |
| Stochastic passes 30 → 100 | Under 1.5% runtime. Every Bayesian and variational uncertainty column shifts and must be regenerated |
| Gaussian-process cap removed | It was also keyed on the model name being exactly `GP`, so a Tanimoto variant escaped it. Now matches any variant |
| **A requested-but-unavailable model is now a hard failure** | Guard 9. This is what would have stopped the two jobs that ran five folds over an empty roster |

**Applied 2026-08-26 in the second parity session (chat E), on BOTH sides via the shared spec:**

| Change | Where it was wrong | Effect |
|---|---|---|
| **Neural batch size 64 → 32** | not in the hand audit at all | The experimental side trained on 64, QM9 on 32. Every neural row moves |
| **Early stopping rolls back to the best epoch on QM9** | QM9 returned the last epoch, up to twenty past the optimum | Removes a procedural reason for QM9's neural degradation curves to look steeper. Author's decision |
| **One early-stopping criterion** | QM9 needed a 0.01 absolute drop in a **summed** validation loss, so its threshold scaled silently with the number of validation batches; the other side used strict improvement on a mean | Now mean loss, strict improvement, patience 10, on both |
| **The Gaussian process outputscale is applied** | QM9 set `params['outputscale'] = 1.0` and **never passed it to the model**, so its scale kernel actually started at gpytorch's `softplus(0) ≈ 0.693` while the experimental side set 1.0 "to match". The Optuna path searched a dimension that did nothing | Both now start at 1.0 |
| **Which optimiser fitted the Gaussian process is recorded** | the experimental side fell back to an Adam loop whenever botorch refused a plain gpytorch `ExactGP`, and left no trace outside one attribute. **QM9 had no fallback at all**, so the same environment that quietly changed one study would kill every GP job in the other | Both share one try/except and both write a `gp_fit_method` column. ⚠️ On this laptop botorch 0.16.1 **does** refuse it, so every local GP fit uses the fallback — check that column before comparing new GP rows to old ones |
| **NGBoost's distribution and score are named in the spec** | QM9 passed `Dist=Normal, Score=MLE`; the experimental side passed neither, so the two agreed only while ngboost's defaults happened to match | Both resolve the same names, and the audit asserts `MLE is LogScore` |
| **`-b/--bootstrapping` renamed to `--repetitions`** | it is not bootstrapping — nothing is resampled with replacement; each repetition refits on a freshly seeded split | Author's instruction. Old spellings kept as aliases so the thirteen existing job arrays still run |
| **Provenance columns on every results row** | a CSV could not be traced to the parameters behind it | `spec_version`, `spec_hash`, `gp_fit_method` on both sides. Appending to a file with the old header is now a hard error rather than ragged rows |
| **Stale job-script headers corrected** | `unc_gp.sh` still advertised the 2,000-molecule cap and four scripts still said 30 stochastic passes | Six files |

#### ✅ 3.4.4e SETTLED 2026-08-26 — the forest feature setting is 0.3, on both pipelines

Re-measured this session at production scale by `scripts/parity_test_forest.py`, because the
author's recollection was that the earlier recommendation had talked them out of `sqrt` rather
than measured them out of it. **The decision rule was fixed in the script's docstring before the
run**, so the answer could not be rationalised afterwards: take the setting that wins or ties on
the descriptor vector *under noise* and does not lose on Morgan; ties inside the seed spread go to
the faster one.

10,000 QM9 molecules, 80/20 scaffold split (which is what the tree models effectively get, since
they stack validation back into training), 3 seeds, clean and at half a label spread, ordinary
forest and quantile forest, 72 fits. Paired against `sqrt`, from
`results/parity_tests/forest_max_features_n10000_paired.csv`:

| | 0.3 vs `sqrt`, clean | 0.3 vs `sqrt`, noised | 1.0 vs `sqrt`, clean | 1.0 vs `sqrt`, noised |
|---|---|---|---|---|
| forest, Morgan | **+0.040 (3/3)** | **+0.032 (3/3)** | +0.025 (3/3) | +0.015 (3/3) |
| forest, descriptors | +0.0055 (2/3) | +0.0060 (2/3) | −0.0002 (2/3) | −0.0019 (1/3) |
| quantile forest, Morgan | **+0.018 (3/3)** | **+0.016 (3/3)** | −0.0032 (1/3) | −0.0044 (1/3) |
| quantile forest, descriptors | +0.0037 (3/3) | +0.0015 (2/3) | −0.0021 (1/3) | **−0.0074 (0/3)** |

**`0.3` passes both clauses of the rule. `1.0` fails both.** So neither pipeline was right: `sqrt`
is QM9's value and loses badly on the fingerprint, and `1.0` — what the experimental side has been
taking silently by omission — is the *worst* option under noise on the descriptor vector, the
paper's primary representation and the regime the paper is about.

**A second result, not asked for and worth having.** `0.3` also gives the quantile forest much
better-calibrated intervals, and intervals are that model's whole deliverable. Coverage at one
standard deviation on clean descriptors, against a nominal 0.683:

| setting | descriptors | Morgan |
|---|---|---|
| `sqrt` | 0.754 — far too wide | 0.772 |
| **0.3** | **0.688** | 0.601 |
| 1.0 | 0.648 — too narrow | 0.520 |

**Cost.** Per fit, about 3.5× `sqrt` and about 3× cheaper than `1.0` (forest, mean seconds:
descriptors 3.1 / 11.1 / 35.4; Morgan 5.3 / 25.2 / 68.7 for sqrt / 0.3 / 1.0).

**Difference from the earlier benchmark.** That one put `0.3` and `sqrt` level on descriptors
(+0.002); this one has `0.3` ahead by +0.0055 clean and +0.0060 noised. Same conclusion, and the
two runs differ in that this one drops the molecules QM9's authors flag as uncharacterised and uses
the 80/20 the tree models actually see. Neither difference is large enough to matter to the choice.

**Applied** in `models/model_defaults.py` v1.1.0 (`67f1b96564b9`), which both pipelines import, so
it lands on QM9 and the experimental datasets at once. ⚠️ **It invalidates every existing forest
and quantile-forest result, QM9's included.** Everything is being re-run, so the cost is zero, but
this is a change to the main study and not only an alignment.

#### ✅ 3.4.4f SETTLED 2026-08-26 — the paper reports RAW uncertainty

The author could not choose from argument — *"I don't know the right thing to do. Can you test
which works the best?"* — so `scripts/parity_test_calibration.py` measured it. 3,000 QM9 molecules,
70/15/15 scaffold split, 3 seeds, four uncertainty models, noise levels 0 / 0.5 / 1.0, 36 fits,
with the rule fixed in the docstring before the run. Numbers from
`results/parity_tests/calibration_n3000.csv`.

**Calibration works in the narrow sense.** The multiplier moved test coverage closer to nominal in
**all twelve** model × noise-level cells, sometimes dramatically — the Gaussian process goes from
0.864 to 0.682 against a nominal 0.683. No fit hit the `[0.1, 10]` bound.

**And it is still the wrong number to lead with, for a reason that is not a matter of taste.** The
multiplier is refitted at each noise level, so calibrated coverage is nominal at each level *by
construction*. The span of coverage **across** noise levels collapses:

| | raw, σ = 0 / 0.5 / 1.0 | span | calibrated | span |
|---|---|---|---|---|
| NGBoost | 0.546 → 0.876 → 0.978 | **0.432** | 0.637 → 0.658 → 0.686 | 0.049 |
| quantile forest | 0.799 → 0.929 → 0.956 | **0.157** | 0.740 → 0.733 → 0.737 | **0.006** |
| Gaussian process | 0.864 → 0.857 → 0.840 | 0.025 | 0.682 → 0.668 → 0.643 | 0.039 |
| Bayesian network | 0.628 → 0.642 → 0.603 | 0.039 | 0.681 → 0.714 → 0.723 | 0.043 |

The raw column says something: NGBoost's and the quantile forest's intervals become far too wide as
noise rises. The calibrated column says nothing, because it was fitted not to. **Any claim about
how uncertainty *responds* to noise must be read off raw, or it is circular.**

**The multiplier is not a property of the model either.** For NGBoost it goes 1.20 → 0.56 → 0.37
across noise levels — a spread of 0.83 against a seed spread of 0.07, twelve times over — and every
one of the three seeds shows the same monotone collapse, so it is systematic rather than noise. The
quantile forest is five times its seed spread. A correction that has to be refitted per condition
is a per-run fudge.

**Both rank-based uncertainty questions are unaffected**, confirmed rather than assumed: the maximum
difference in Spearman(uncertainty, |error|) across all 36 fits was exactly **0.0**.

**The honest caveat, which belongs in the paper.** Raw coverage *is* poor — the Gaussian process
sits at 0.86 against a nominal 0.68, the quantile forest reaches 0.96. That is the finding, not a
failure to report. Say it, and report the calibrated column beside it as what a single post-hoc
multiplier would buy.

⚠️ **A defect in the instrument, found after the run and worth recording.** The script carried two
operationalisations of "does the multiplier drift materially" and **they disagreed on this data**:
a relative test (drift greater than three times the seed spread) flagged NGBoost and the quantile
forest, while an absolute ceiling of 1.0 — a number written into the code without justification —
passed everything. Neither was chosen silently. The script now prints both and takes the tie-break
from the flatness table above, which does not depend on either threshold. A second defect in the
same file: re-running the report against a saved CSV reported "every fit failed", because an empty
`failed` cell reads back as `NaN` rather than `''`. Both fixed.

#### 3.4.4d The quantile forest cannot be fitted in the local environment

`quantile_forest` 1.4.1 and scikit-learn 1.3.2 disagree on a constructor parameter, and every fit
raises `Invalid parameter 'monotonic_cst'`. **This is live on this machine.** It is the exact
incompatibility the uncertainty preflight checks for. If it is live on the cluster it takes out
every task of that run. Check before submitting.

Consequence for the benchmark above: the quantile forest could not be tested. The mechanism should
carry over from the ordinary forest, but that is an inference and should be labelled as one.

#### 3.4.5 ✅ 2026-08-26 — parity is now structural, not checked

The first guard was the table above. It went stale.

The second was `scripts/audit_pipeline_parity.py` in its original form, which restated both
pipelines' parameters as hand-typed literals. **It went stale inside twenty-four hours**: its
experimental column still said the quantile forest had 100 trees and that XGBoost left the
learning rate unset, hours after both were fixed. A checker that restates what it checks is a
third copy to keep in sync.

**What was built instead.** `models/model_defaults.py` is now the single source of truth for every
default the two pipelines share — the six tree, kernel and boosting models, the Gaussian process,
both neural architectures, the training and early-stopping constants, and the Bayesian and
variational priors. **Both pipelines import it.** `models/models.py` reads it in each of its
`params_source == 'default'` branches; `alternative_data_noise_robustness.py` resolves the
`qsar_qm_models` checkout from `$QSAR_QM_MODELS_ROOT`, a sibling directory, or one of the two known
cluster paths, and **raises if it cannot find one** — no vendored copy and no silent fallback,
because a second copy going quietly stale is the defect, not the fix. Every uncertainty job script
already runs `. <qsar_qm_models>/setup.sh`, so the checkout is present by construction.

Drift is now impossible rather than detectable: there is one copy of each number.

**What the audit script does now.**

```bash
python scripts/audit_pipeline_parity.py --strict      # preflight check 0
python scripts/audit_pipeline_parity.py --self-test   # proves --strict bites
```

1. Loads the spec **through both pipelines** — importing `models.py` the way QM9 does, and loading
   the experimental module from its own file the way `preflight.sh` does — and fails unless both
   report the same content hash.
2. Builds every model from that spec **with the installed libraries** and diffs the effective
   parameters against `scripts/model_params_baseline.json`. This is the library-drift check: an
   upgrade that moves a default we do not pin appears here rather than as a changed result months
   later.
3. Asserts the facts a parameter diff cannot see — that ngboost's `MLE` is still `LogScore`, that
   `natural_gradient` still defaults to `True`, which optimiser will actually fit the Gaussian
   process, and that **no job script passes `--use_best_params`** (that flag loads hyperparameters
   from a JSON, which the experimental side cannot do at all, so a job on that path is audited by
   nothing).
4. Reports every package version, and closes with the manual checklist.

**Executed 2026-08-26**, on this laptop, against both checkouts:

```
canonical    models/model_defaults.py           version 1.0.0  hash aa0368c86eb6
QM9          models/models.py                -> hash aa0368c86eb6   MATCH
experimental alternative_data_noise_robustness.py -> hash aa0368c86eb6   MATCH
SUMMARY: 0 problem(s), 0 missing package(s)
```

`--self-test` perturbs the spec on disk — the XGBoost learning rate back to 0.3, the quantile
forest back to 100 trees — confirms `--strict` fails on each, and restores the file. That is the
check that fails if the fix is removed.

**What it still cannot see, and therefore still needs a human:** representation identity. No
parameter diff would ever have caught the fingerprint being the wrong function — that took reading
the binding in the crate source. §3.4.1 is the manual half, it is printed at the end of every audit
run, and it has to be re-checked by hand whenever the featurisers change.


## 4. Decisions I need from you

These are the ones where different answers mean materially different work. Everything else in
this document I can settle from the code or the evidence.

**Decision 1 — does the uncertainty question run on QM9, or only on the three experimental
datasets?**
Running it on the experimental datasets only: the machinery exists, it is tested, it is where
measurement error is real, and it is where censoring is a real mechanism rather than an imposed
one. Adding QM9 means building out-of-fold scoring, training-molecule uncertainty saving, and
noise recording inside `process_and_train.py` and `models/models.py` — a substantial build in the
pipeline that cannot currently be run on this laptop at all. My recommendation is experimental
datasets only, with one sentence in the Methods saying why. But it moves the paper's uncertainty
section off QM9 entirely, which is a visible change, so it is your call.

**Decision 2 — do validation labels carry their own noise?**
§2.5. It changes every fitted model in both pipelines. The experimental pipeline has already made
this change; QM9 has not, so right now the two pipelines disagree with each other as well as with
the Methods. My recommendation is yes — training noisy, validation noisy from an independent
draw, test clean — because the alternative is that seven models get a tenth of their labels free
and the neural models early-stop against an oracle.

**Decision 7 — the tuning sweep: how many settings, on what labels, and when a tuned value wins.**
§5.7. Three answers, and one of them changes what the sweep costs.

- **(a) How many settings per pairing?** The cost is linear in this number and the arithmetic is
  in §5.7f, priced from one fit of every pairing at its shared default. My recommendation is 12:
  a random search of 12 covers each parameter's range better than a grid of 12 once more than two
  parameters matter, and the models here have three to ten.
- **(b) Clean labels, or repeated per noise level?** Clean is one sweep. Repeating it at every
  level of the main grid multiplies it by seven, and makes "the tuned setting" a different setting
  at every level, which the two JSON files cannot express — they are keyed by model and
  representation, with no level. My recommendation is clean only, with a Methods sentence saying
  the hyperparameters are fixed across the noise axis so that the axis is the only thing moving.
- **(c) Does a tuned value have to beat the shared default by a margin?** `--write-master
  --margin X` adopts a setting only when it beats `models/model_defaults.py` by X of R² **on the
  QM9 test split** — the split the search never chose on (§5.7i) — and it refuses to run at all
  until `--confirm` has measured that. A margin of 0 adopts anything that wins by any amount, which
  at that size is partly luck. My recommendation is 0.01, and that pairings below it keep the shared
  default — which is the file both pipelines already read, so the fallback is the parity-checked
  value, not a library one. Whatever survives that then goes through the three validation datasets
  and is pruned there.

**Decision 8 — do the other ten models get their own tuned entry?** §5.7a. As the code stands,
only svm, xgboost, lgb and ngboost can be handed a setting that reaches them alone. Giving the
quantile forest, the four Bayesian networks and the two Gaussian processes their own entries is
one changed literal per call site in `models/models.py`. Cost: about half an hour of edits and a
re-run of the two checks. Not doing it means the sweep measures 80 pairings and can deliver 24.

**Decision 3 — the aleatoric/epistemic split — NOT A DECISION. You settled it on 2026-08-21.
It gets BUILT, to industry standard, by deleting the broken code and replacing it.**

Your words, 00:58: *"so should I be modifying my code to do epistemic/aleatoric decomposition
differently? … I NEED THESE RESULTS AND I WANT TO IMPLEMENT THEM BY INDUSTRY STANDARDS HOW ARE YOU
NOT GETTING THIS"*. And at 01:12, after I hedged: *"there's some faulty code that must be DELETED
AND REPLACED. get over it"*.

That is why the relevant section of `immediate_next_steps.md` is headed "The code to delete and
replace" rather than "options". **Do not treat this as open, and do not propose dropping it.**

What it means concretely — all of it already written up and sourced at the time:

| Step | Where |
|---|---|
| Delete the stub that hard-codes the data-driven component to nothing, and all four of its call sites | `scripts/utils.py:62-85`; `models/models.py:2121`, `:2724`, `:3133`, import at `:3077` |
| Give the Bayesian networks a two-output head — a mean and a log-variance | `models/models.py` ~`:1015`. A head already exists at `:2023-2034` behind a loss flag no job script passes, and its variance column is sliced off again at `:2086` and `:2102` |
| Train with a Gaussian likelihood instead of squared error | the exact expression already sits inside the variational loss at ~`:1226` |
| Route to the helper that does the split correctly, collecting variances as well as means | `scripts/utils.py:88-110` — correct, and never called by anything in any of the three repositories |
| Quantile forest: compute it after the fact from the trees already fitted, no retraining | working function saved at `scratchpad/forest_ae.py` |

Two traps recorded at the time. When undoing label standardisation, scale the **variances**, not
the standard deviations. And one existing call passes one argument to a function that takes two,
so that path raises on contact and has never run.

**Why I got this wrong, so it does not happen again.** There is a separate, earlier decision from
2026-08-14 (`REVISION_GUIDE.md:73`, `:208`) removing the aleatoric/epistemic *frame* from the
paper's Scientific Contribution. That is about a claim in the abstract — the per-sample detector
that turned out to be a pooling artefact. It says nothing about whether the decomposition is
computed, it is older than the 21 August instruction, and it was superseded on 23 August by the
ruling that the uncertainty question gets re-run rather than reworded. I read the older note,
matched it on keywords, and inverted a live instruction.

**Decision 3b — SUPERSEDED, mostly by action.** Of the three questions raised here on 2026-08-25,
two have been applied and one is answered by measurement:

- The quantile-forest tree count and the Gaussian-process cap are **done** (§3.4.4c). Note the
  tree count buys +0.003 R², inside the seed spread — the feature setting was doing the work.
- The forest feature setting moved to decision 3c below, because the benchmark contradicted the
  original instruction.
- ✅ **Repeat fits on the experimental side — SETTLED 2026-08-26: keep one fit per cell**, seed
  pinned at 42. The author's call, against my recommendation of three, and it is the cheap
  direction. **What the paper now has to say, because it is no longer optional:** the experimental
  variance decomposition has **no estimable residual term**, and no experimental model-vs-model
  comparison carries a run-to-run error bar. The five folds are a partition, not repeats, so their
  spread mixes randomness with scaffold difficulty and cannot stand in for one (§3.3). QM9 keeps
  its ten repetitions, so the two studies are asymmetric here on purpose. Recorded in the audit
  script's manual checklist so it cannot be forgotten at writing time.


**Decision 3c — four alignment calls the audit and the benchmark opened.**

| # | Question | Context | My recommendation |
|---|---|---|---|
| ~~**Forest feature sampling**~~ | ✅ **SETTLED 2026-08-26 by measurement — `0.3` on both.** Re-run at the author's request against a decision rule fixed before the run; see §3.4.4e. Applied in the shared spec | — |
| ~~**Early stopping**~~ ✅ **SETTLED 2026-08-26 — roll back, both sides.** Applied in `train_nn` and in the experimental trainer, both reading `NEURAL_DEFAULTS['training']`. Proven on a real run: the DNN stopped at epoch 23 and restored epoch 13. The improvement test was aligned at the same time — QM9 required a 0.01 absolute drop in a **summed** validation loss, so its threshold scaled with the batch count | Keep the last epoch, or roll back to the best? | QM9 counts twenty epochs of no improvement and then returns *that* epoch's weights. The experimental side snapshots and restores the best. Those twenty extra epochs are spent memorising injected corruption, and more of it at higher noise — so **QM9's neural degradation curves are steeper for a procedural reason, pointing the same way as the finding** | **Roll back, both sides.** It is what almost everyone means by early stopping. Caveat: it means selecting on validation labels, which under decision 2 would be noisy — but that is correct, since nobody gets clean labels when deciding when to stop |
| ~~**Uncertainty calibration**~~ ✅ **SETTLED 2026-08-26 by measurement — RAW as primary**, calibrated kept as a labelled secondary. See §3.4.4f. Set in the shared spec as `UNCERTAINTY_DEFAULTS['primary_column']`; the figure script must now name the column it reads (chat J) | Report calibrated or raw? | A single multiplier fitted after training so predicted uncertainties match observed errors. QM9 does it, the experimental side does not, and the figure script silently prefers the calibrated column where it exists. Because it is one positive multiplier it **cannot change the order** of molecules — so both uncertainty-tracking questions are unaffected either way. It moves coverage and calibration-error numbers only, which are exactly what it is fitted to fix | **Raw as primary**, calibrated as a clearly-labelled secondary if wanted. Reporting coverage after calibrating is close to circular. Either way the analysis must state which column it read — it does not today. Free: aligning down needs no re-run |
| **Embedding standardisation** | Standardise the learned embeddings per feature, or leave them raw? | Separate from the storage fix in §2.8c, which is not optional. Without it, a kernel with one shared lengthscale across a thousand dimensions is dominated by whichever dimensions are widest | ✅ **SETTLED 2026-08-26 — standardise.** Approved as part of chat C's plan and implemented: `CONTINUOUS_REPS` in `process_and_train.py` now covers PDV and all three learned embeddings, fitted on the training split. It changes every embedding number, which is why it is recorded here rather than left implicit |

**Decision 4 — sign off the noise design.** ✅ **CLOSED 2026-08-27.** Laplace is **kept, at depth**
(*"Keep laplace"*) — out of the full grid on measurement, in the depth stage for the citation, 720
runs. `noise_conditions.json` records it without the `optional` marker so the file states the
decision rather than deferring it. `NOISE_DESIGN.md` §7 now has nothing open. The record of what the
decision rested on:
The condition set is settled (§13.9) and enforced by `noise_conditions.json` plus a gate on each
side. The positive-control question is answered by censoring (§3.2). **What is left is one narrow
yes/no: is Laplace queued at depth at all?** It is out of the full grid — measured indistinguishable
from Gaussian on both roster models at both levels (largest 0.0136 R², 0.24 of the wobble) — so the question is no longer worth
4,680 runs but **720**. My view is unchanged: include it. It is the only distribution actually
*fitted* to real bioactivity error, so it buys a citation for a claim the paper wants to make, and
at that price the citation is cheap. Saying no costs nothing but the citation.

**Decision 5 — do the experimental datasets lead the Results?**
They are the only data that has never been contaminated, they are where measurement error is
real, and after the re-run they would be the only place the uncertainty question is answered.
Currently they are a validation section at the end. Moving them to the front is a structural
change to the paper, not just the Results.

**Decision 6 — scope.** You have already said fewer folds or levels on the uncertainty side is
acceptable. The specific levers, in the order they cost you least, are in §6.4. I need to know the
ceiling you want to spend before I size the job scripts.

---

## 5. What gets deleted, built, and fixed

Grouped by whether it changes what gets trained. Everything in the first group must land in one
commit, because each of them changes what a noise level means.

### 5.1 Changes what gets trained — all of these, or none

| # | Change | Where |
|---|---|---|
| 1 | ✅ **DONE 2026-08-26** — deleted the five superseded noise types and the six unreachable distribution variants, plus the four functions that served them | `rust/src/main.rs`, per `NOISE_DESIGN.md` §6.1 |
| 2 | ✅ **DONE 2026-08-26** — dose solver, three shapes, five targeting rules (the shifted grouped condition of §13.3 included) | `rust/src/main.rs`, per `NOISE_DESIGN.md` §6.2, §6.2a |
| 3 | Implement the identical specification in Python and cross-check it against Rust on the same labels — **the specification is `NOISE_DESIGN.md` §6.2 step 6, the deletions are its §6.1, and the cross-check is its §6.3 items 6–8**; do not work from a paraphrase | `NoiseInject/noiseInject/core.py` |
| 4 | ✅ **DONE 2026-08-26** — standardisation constants come from the clean training labels; `generate_aggregate_stats` no longer takes the noise at all | `rust/src/main.rs`, `generate_aggregate_stats` |
| 5 | Validation split gets its own independently drawn noise (decision 2) — **still the author's call (§13.5)**. The code is now shaped for it: `write_data` takes an `apply_noise` flag and a `NoisePlan`, so it is one extra plan built over the validation labels | `rust/src/main.rs`, `preprocess_data` |
| 6 | ✅ **DONE 2026-08-26** — recorded per molecule where it is drawn, never reconstructed. See §5.2 and `NOISE_DESIGN.md` §6.2a | `rust/src/main.rs`, `write_noise_manifest` + the provenance writer in `preprocess_data`; `scripts/process_and_train.py`, `record_noise_manifest` |
| 7 | Guard the two truncation and index-drift risks | `rust/src/main.rs` ECFP4 block, `:173-191` |
| 8 | Two-output head on the Bayesian networks (decision 3) | `models/models.py:2031`, `:2088` — note the head **already exists and is disabled**, and two models already compute a per-molecule aleatoric term and discard it (`:6835-6851`, `:6963-6977`) |
| 9 | Out-of-fold uncertainty on QM9 training molecules (decision 1) | `process_and_train.py`, `models/models.py`, `scripts/utils.py` |

### 5.2 What every result row must carry from now on

The single reason the confound in §2.2 went unnoticed for so long is that nothing recorded how
much noise was actually delivered. Every row, both pipelines:

`split` · `record_index` · `canonical_smiles` · `y_clean_raw` · `epsilon_raw` ·
`noise_type` · `unit_dose` · `delivered_dose_in_label_units` · `delivered_dose_as_fraction_of_label_spread` ·
`affected_molecule_fraction` · `standardisation_mean` · `standardisation_sd` · `seed`

✅ **`gp_collapsed` added 2026-08-26 (chat C).** A Gaussian process that cannot use its features
returns one constant for every molecule. That still writes a score, and the score reads as a weak
representation rather than a failed fit — which is how the two learned embeddings were written off
(§2.8f). Every Gaussian-process row now says whether its fit collapsed. `gp_fit_method`, added by
chat E, says which optimiser produced it.

The molecule identifier matters as much as the dose columns: the current `sample_idx` is a row
position, so rows cannot be linked to molecules or matched across replicates — which is
separately why the Gaussian process's out-of-fold rows had to be re-indexed in the uncertainty
patch.

✅ **The Rust half is done, 2026-08-26.** Two files come out of every run
(`NOISE_DESIGN.md` §6.2a): `noise_provenance_{file_no}.csv` carries
`split, record_index, canonical_smiles, y_clean_raw, epsilon_raw, y_noisy_raw, y_written` for
**every split**, and `noise_manifest_{file_no}.json` carries the run-level dose columns, which
`process_and_train.py` appends to `<results>_noise_manifest.csv`. Held-out rows carry
`epsilon_raw = 0` exactly, so the provenance file is itself the evidence for gate 3.

Two things still to wire, and neither is chat A's: the Python injector writes the same columns
(chat B), and the figure script joins the manifest onto the results rows so no figure can be
un-traceable to the dose that produced it (chat J).

### 5.3 Free savings, no statistical cost

| Saving | Size | What it costs |
|---|---|---|
| Cache the prepared split and features per replicate instead of per noise level | ~91% of the preparation stage | nothing. Measured: the RDKit descriptor set takes 220 s per 10,000 molecules and is recomputed 110 times per output file — about 6.7 hours of which 6.1 is pure repetition. **Cannot be tested on this laptop** (the local Python environment cannot import `torch_geometric`), so it must be smoke-tested on the cluster before the long run |
| Run the zero-noise point once per model and representation instead of once per noise type | ~9% | nothing — the runs are byte-identical. It also fixes a real statistical error: every zero-noise standard error and p-value is currently inflated sixfold by counting the same run six times |
| Rebuild one deduplicated set of job scripts from the design | up to ~3× on specified work | nothing. The thirty existing script directories overlap heavily and specify roughly three times the clean design |

### 5.4 The figure script — one script, and the full change map

**First: there is only one script from now on.** `generate_paper_figures.py` is the retired
version — 216 mentions of the old slope metric, zero of the current one — and
`run_figures.sh` still invokes it, so anyone who runs it silently regenerates retired numbers into
`results/paper_figures/`. Delete all three. Rename `generate_paper_figures_v2.py` to
`generate_paper_figures.py`, and `run_figures_v2.sh` to `run_figures.sh`. No versioning, one path
to every number. Fold `deep_analysis.py` in and delete it too — it is a second entry point whose
numbers were never certified against the pipeline.

Line numbers below are against `generate_paper_figures_v2.py` as it stands.

**Structural changes — these implement the guards in §0.6**

| Where | What it does now | What it must do | Guard |
|---|---|---|---|
| `calculate_robustness`, `:1803` (averaging at `:1818-1820`) | Averages replicates, *then* integrates the retention curve. One value per cell, no spread | Integrate **per replicate**, then aggregate. Emit the full accuracy-versus-level curve, the clean baseline, the value at the reporting level, and the drop | 3 |
| `calculate_validation_auc`, `:1318` | Averages the five folds before integrating, leaving one observation per cell | Keep the fold axis all the way through. This alone un-saturates the experimental variance decomposition | 3 |
| validation loader dedup, `:1148-1152` | Deduplicates on dataset, model, representation, noise type and level — **no fold** — keeping the first row and silently discarding four fifths of the data | Add `fold` to the key. The QM9 loader at `:641` already includes the replicate; this is the same fix | 3 |
| `fix_injected_noise`, `:980-1021`, called at `:3997` | Recovers the injected noise by regression, grouped without the noise type | **Delete both.** The real value is recorded at source now | 2 |
| every correlation site | Groups without stating the conditioning set | Carry the conditioning set into the output row; assert one cell per group | 1 |
| the two gates, `:423` and `:429` | 0.6 governs the simple-effects table, 0.3 governs the variance decomposition — two thresholds under one headline | Pick one, state it once, apply everywhere | — |
| `filter_catastrophic_iterations`, `:821`, threshold `:437`; plus extra drops at `:1946`, `:2147`, `:2260` | Two undeclared filters, one stacked on the other | Declare both. Emit the headline result with and without, and fail if they disagree in direction | 8 |
| the methods figure, `:2541-2562` | Draws a synthetic three-component mixture and reimplements two noise types differently from the pipeline | Redraw from real labels through the real injector, at matched dose | 12 |
| every table writer | Prints retention alone in several places | No ratio without its components in adjacent columns | 4 |
| every caption | Metric names hard-coded, and two captions disagree about what the same numbers are | Generate the metric name from the same constant the column comes from | 12 |

**New analyses — the five you asked for (§0.4), with what already exists**

Three of the five are smaller than they look, because the machinery is already there.

| What | State of play | The change |
|---|---|---|
| Variance decomposition of accuracy at the reporting level | **Nearly free.** The function already takes the level as a parameter and already runs once per noise type. The parameter defaults to 0.3 and is **never passed** (`:1935`, call site `:2802`) | Add a module constant beside the primary-representation one and pass it. Warn if the roster surviving the minimum-replicate gate differs between the two levels — a different roster makes the two decompositions non-comparable |
| Paired significance tests rebuilt | **Worse than not built.** The function takes a representation and a noise type argument and **ignores both**, pooling across every shared condition to manufacture enough pairs (`:1888-1903`) | New signature taking the quantity to test. Filter to one representation and one noise type at the top, pair on the replicate, and emit six rows per model pair. Note the floor: a two-sided signed-rank test on five pairs cannot go below p = 0.0625, so the replicate count decides whether anything can be significant at all |
| Side-by-side ranking table | **Two partial versions exist and both are wrong-shaped.** One ranks robustness across noise types, one ranks accuracy across levels. Both are transposed relative to what you asked for, and **both are sorted by exactly the mean-rank column you rejected** on 2026-08-14 | New function: one noise type, models as columns, the three quantities as rows, no overall winner column |
| Representation by noise type | **Does not exist in the figure script at all.** The nearest thing is in the standalone script being folded in, and it averages over a model roster that is not held constant across representations — so it compares different rosters | New function in the tables section, with the roster held constant and stated |
| Experimental-dataset variance decomposition | **The machinery is correct and is not the problem.** It uses sequential sums of squares from nested fits specifically so the residual is pure within-cell variance. It is being fed **one observation per cell**, so the saturated fit reproduces the data exactly and the residual is arithmetically zero | Do not rewrite it. Fix the two places that discard folds (`:1368-1372`, and the dedup at `:1148`) and it starts working |

#### 5.4a Rank-versus-noise-level line charts — the author's spec, 2026-08-27

**Superseded my first write-up of this, which was more complicated than what was asked for.** What
follows is the author's own specification. Do not add to it.

**The chart.** Hold one noise type and one representation fixed. Plot every model. The x-axis is the
noise level; the y-axis is where that model ranks against the others. Each model is one line, and
the lines cross as the noise rises.

**Then repeat it** for each noise type — one chart per noise type, not panels within one chart.

**Then the mirror.** Hold one noise type and one **model** fixed, and plot every representation the
same way.

**Which one is held fixed is chosen from the results, not now.** The author's examples were PDV for
the representation charts and XGBoost for the model charts, **given explicitly as examples**. The
real choice follows §13.1 item 4.

**When a model does not produce a usable result, the author's rule, in three cases:**

| Case | What to do |
|---|---|
| It broke — the run failed for a technical reason | **Re-run it.** It is not a data point |
| It worked and then fell off a cliff as noise rose | **Cut its line short** at the level where it drops below the R² cutoff. The line ends; it does not plunge to last place |
| It never worked at any level | **Leave it out of the chart entirely** |

**The axis stays the same across all the charts** regardless of who is dropped, so the charts can be
read against each other.

**Print the actual R² beside the ranks.** The author's instruction, and it is guard 4 — a rank hides
whether two models are a thousandth apart or a fifth apart.

**The one thing this figure is for:** every other ranking table has to pick a single noise level.
These charts do not. They show the whole ladder, so a reader sees where the order changes rather
than being told what it is at one point.

**Two implementation notes, not additions to the spec.** Rank within each replicate and then
aggregate, because ranking an averaged score answers a different question (guard 3). And do not order
the legend by average rank — that ordering was rejected on 2026-08-14 and two existing ranking
functions still use it; order by rank on clean labels so each line starts where that model starts.

**Two roster questions the script encodes and the paper never states.** The robustness
decomposition runs on seven models while the accuracy one runs on eleven, because four Bayesian
variants do not train on two of the representations. That one belongs in the Methods.

**The second is now settled and the script must change.** `ANOVA_MODELS_EXCLUDE`
(`generate_paper_figures_v2.py:129-133`) currently drops both Gaussian processes, giving the kernel
as the reason: *"Tanimoto kernel incompatible with continuous PDV"* and *"RBF GP for PDV only;
excluded from cross-rep ANOVA"*. **Neither is the real reason and the author has ruled on it
(§0.3):** the kernel comparison is answered (§10b.2), the radial basis is the kernel everywhere, and
the Gaussian process enters the decomposition alongside the support vector machine.

Three changes follow, and they must land together or the model will look catastrophically bad for a
reason that has nothing to do with the model:

1. **Fix the embedding rescaling first (§2.8c).** On the two learned embeddings the radial basis
   scores −0.016 and 0.009 against Tanimoto's 0.872 and 0.868. That gap is the per-molecule
   rescaling defect, not a kernel property. Committing to one kernel before fixing it would put two
   near-zero cells into the decomposition and let them drive the representation term.
2. **Run the radial-basis Gaussian process on every representation** in the QM9 re-run. It has never
   been run on SNS at all, and it is the only model with a representation-shaped hole.
3. **Then delete both entries from `ANOVA_MODELS_EXCLUDE`, and `PDV_ONLY_MODELS` with them** —
   `gauche_rbf` stops being descriptor-only the moment it has full coverage. Keep the Tanimoto
   version as the reported head-to-head on the fingerprints; it is evidence, not a second model.

---


### 5.5 The uncertainty decomposition — the build spec

You settled this on 2026-08-21: build it, to industry standard, by deleting the broken code and
replacing it (§0.3, §4). This is that spec. Every line reference was checked this session.

The literature behind it is recovered and sits in `research_archive/f692d614/` — a 36 KB evidence
review, the implementation notes, and the reference sources from Chemprop and from Ryu et al.

**Four defects found while writing this spec that nobody had recorded.**

- **The two components are swapped, in at least seven places.** For the evidential
  parameterisation the data-driven term is `beta/(alpha−1)` and the model term is
  `beta/(v·(alpha−1))`. The code has them the other way round — `models.py:4297`, `:4818`,
  `:5047`, `:5217`, `:5428`, `:5976`, `:6079` and more. Because `v` is always greater than one,
  **the repository has been reporting the smaller quantity as the data-driven one throughout.**
  The *total* is unaffected, since addition is symmetric — only the split is wrong, which is
  exactly the thing the paper wanted to report.
- **Two paths crash on contact.** `models.py:3133` and `:3162` call a two-argument function with
  one argument and unpack two of its three returns. `models.py:3375` unpacks two values from a
  three-value return *and* passes a standard deviation where a variance is required. Any graph
  model with a Bayesian transformation dies before writing a row.
- **One call site is fed the wrong quantity.** The Gaussian-process split is passed a variance at
  `models.py:1775` and a standard deviation at `:3375`. The second gives the fourth root of the
  noise.
- **The variational split is a scalar broadcast to every molecule.** That model has one noise
  parameter, so its per-molecule data-driven term is a constant by construction, and its
  correlation with per-molecule injected noise is **guaranteed to be zero however good the model
  is**. That is a mechanism, not a result — and it plausibly explains the coverage anomaly that
  has been open since June.

**Delete**

| What | Where | Why |
|---|---|---|
| The stub that hard-codes the data-driven term to nothing and sets the total equal to the model term | `scripts/utils.py:62-85` | It cannot decompose anything, and every downstream coverage number computed from its "total" silently omits observation noise |
| The distributional variant that hard-codes the model term to nothing | `scripts/utils.py:138-166` | Its premise is false for the quantile forest — a forest *is* an ensemble and its spread across trees is a legitimate model-uncertainty estimate |
| The first of two definitions of the loss factory | `scripts/loss_functions.py:170-203` | Defined twice; only the second is live. The dead one is missing a key, so anyone reading it draws the wrong conclusion |

**Build**

| What | Where | Note |
|---|---|---|
| A two-output head under a constructor flag | `models.py:1012-1017` | `nn.Linear(hidden_size2, 2 if heteroscedastic else 1)`. **Delete the post-construction patch at `:2031`** — patching the head after the fact is what lets the mismatch below slip through |
| The likelihood loss | **already exists**, `scripts/loss_functions.py:47-62` | It already takes a two-column output and does the right thing. Lift it; do not write a new one |
| Guard the loss mismatch | `models.py:1226-1234` | With a two-column output and a one-column target the subtraction **broadcasts instead of raising**, so both heads get regressed onto the label. The variational loss must slice the mean column explicitly |
| The quantile-forest split | `models.py:1387-1397` | Replace the interquartile heuristic: total from the pooled conditional distribution, model term from the spread across per-tree leaf means, data term as the remainder |

**Wire — four trainers, and they must move together**

The two neural trainers carry byte-identical copies of the same broken block (`models.py:2112-2125`
and `:2720-2728`). That duplication is why fixes drift. **Extract one shared routine** rather than
patching both. In each, the Monte Carlo loop currently discards the variance column outright; that
slice comes out and both columns get collected. A third trainer computes no split at all and
writes blanks for every architecture-sweep run. The Gaussian process needs its calibration fitted
against the predictive variance rather than the latent one.

**Two that already compute a real per-molecule split and throw it away**

`models.py:6830-6839` and `:6962-6965`. Both are a two-keyword change at the point where results
are saved. The first is a genuinely input-dependent estimate — better than anything the Bayesian
network roster produces — and it currently dies at the file boundary.

**The standardisation trap, in three parts**

1. The divisor is computed from **noisy** labels, so it moves with the noise level (§2.4). A model
   whose data-driven term perfectly recovered the injected noise would report a value that
   saturates and compresses at high noise.
2. **There is no inverse transform anywhere.** The scale is guessed back afterwards by fitting a
   line, in two different files. Record the mean and the spread at injection and carry them
   through (§5.2).
3. When converting back, **scale variances, not standard deviations.** The repository already
   gets this wrong once — a stacked figure adds two standard deviations, which is only correct if
   the two sources are perfectly correlated. They are independent by construction, so it
   overstates by up to 41%.

**What the output must carry.** Both components as **variances**, on both the standardised and the
raw scale, with the scale and offset on every row so anything can be converted without guessing.

**Say honestly which models can support the split.** The Gaussian process can, exactly. The
Bayesian networks can, once they have the head. The quantile forest can, post hoc. The variational
model cannot give a per-molecule data term at all, and the boosting model has no model-uncertainty
axis without bagging over seeds. That table belongs in the paper.

---

### 5.6 The two representation repairs

Both change the record layout, so they land together with the embedding storage fix (§2.8c) and
before any cluster time.

#### The fingerprint

**How different are the two, actually?** Measured on a seeded random sample of 497 QM9 molecules:
the path fingerprint and a genuine Morgan radius-2 agree at a **mean Tanimoto of 0.0091**, and
differ on 497 of 497 molecules. Mean bits set: 20 for Morgan against 206 for the path fingerprint.
They are effectively unrelated representations, whatever their downstream accuracy turns out to be.

An earlier measurement claiming a much smaller difference was taken from the first 200 records of
the file, which are methane, ammonia, water, acetylene and hydrogen cyanide — too small to have
radius-3 environments at all. The random-sample figure is the right one.

**Do not patch the Rust crate.** Its Morgan wrapper is hard-coded to radius 3, and the bridge
exposes no radius argument, so there is no drop-in fix inside it.

**Compute it in Python instead — and note this already existed.** A Python-computed `morgan`
representation was added in commit `636ef8f` (2026-02-23) and reverted in `46256be` (2026-02-25).
The pattern for handing a Python-computed representation to the writer is already established for
several others, so this is following an existing path rather than inventing one.

#### The descriptor vector

The two names encode the same 200 descriptors from the same list, and **share the same source
array** — one line assigns one from the other. The binary one is a presence bit per descriptor
packed into 25 bytes; the continuous one is 800 bytes of 32-bit floats.

Target state: the name `pdv` *is* the continuous, standardised vector, and the binary one ceases
to exist.

**The experimental side needs no change at all** — its `pdv` is already the continuous vector, and
the string `continuous_pdv` does not appear anywhere in that repository.

**On the QM9 side the edits are in lockstep and the compiler catches most of them:** the record
struct collapses two buffers into one, the two read branches collapse to one, the struct literal
loses a field, and the two write branches collapse to one — which must stay in the same position
in the record between the neighbouring representations. Then the Python reader, the z-scoring
branch, every job script, and the figure script's representation lists and labels.

⚠️ **Watch for substring matches when renaming**: `pdv` occurs inside `continuous_pdv`, so a naive
find-and-replace will corrupt one while fixing the other.

### 5.7 🟠 BUILT 2026-08-27, NOT YET RUN — the local hyperparameter tuning experiment

**It replaces Optuna, which no job ever used.** `--tuning` appears in no script
`slurm_scripts_qm9_rerun/generate_scripts.py` writes, and four of its suggested values never
reached a model at all: the quantile at `models/models.py:1660`, `alpha` and `predictor_type` at
`:4255-4256`, and the loss parameters at `:2561`.

**Shape.** One scaffold split of QM9 — train, validation, test — and no folds. Every pairing tuned
on its own by a small random search, each setting scored on **validation** R², the best setting
written out. The test split is never touched by the search. Clean labels; the noise level is a
question for you, below.

**What was built.**

| file | what it is |
|---|---|
| `scripts/tune_hyperparameters.py` | the experiment: `--time` prices it, `--sweep` runs it, `--write-master` writes the two files the pipeline reads |
| `models/tuning_rosters.py` | the pairing list, imported from `generate_scripts.py` and never retyped, plus the QM9 ↔ experimental name map |
| `scripts/test_tuned_params_reach_models.py` | a tuned value in the two JSON files must change the model that gets built |
| `scripts/test_tuning_rosters.py` | the names in those files must be the roster's names |
| `scripts/confirm_tuned_on_validation_datasets.py` | the same head-to-head on LogD, Caco-2 and hERG, with `--prune` (§5.7i) |
| `scripts/test_bnn_criterion_order.py` | NN-β must train on the loss its Bayesian transformation wrapped (§5.7c) |

All three checks are cases in `scripts/check_fixes_fail_when_removed.py`.

**A candidate reaches the model the way the cluster will deliver it.** The script does not rebuild
the models. It calls `train_<family>_model` with `load_best_hyperparameters` replaced by a stub
returning the candidate — the same branch, in the same function, that `--use-best-params` fires.
So a parameter name the builder ignores cannot score well here and then do nothing on the cluster.

⚠️ **The flag in the runbook is wrong.** `--use_best_params True` is not accepted: the option is
declared `--use-best-params` with `action='store_true'` (`process_and_train.py:379`), so it takes
no value and the underscored spelling is not an option at all. `--tuning False` is accepted but is
already the default.

#### 5.7i 🔴 A WINNER IS NOT A RESULT — THE TWO CONFIRMATION STAGES

**This is where Optuna failed, and it is not a detail.** The search picks the best of N candidates
**on the validation split**. That winning score is the maximum of N noisy numbers on the very split
that chose it, so it is biased upward by construction — and **the bias grows with N**, which means
a bigger search looks better while being no better. The old path wrote that winner straight into
`results/master_tuned_hyperparameters.json`, and nothing ever compared it with the default on data
neither had seen. `models/consolidate_tuned_params.py` has a paired t-test that would have done it,
and it reads two results files that were never produced.

So there are two stages after the sweep, and neither is optional.

**Stage 1 — the QM9 test split.** `--confirm` refits the winner AND the shared default on the same
training split and scores both on the **test** split, which the search never touched. The neural
models still early-stop on validation, never on test — passing test as the early-stopping split
would choose the stopping epoch on the split the comparison is decided on, which is the same leak
one level down. `--write-master` applies `--margin` to **that** difference, **refuses to run at all
if the confirmation file is absent**, and records `search_optimism` — how much of the search's
apparent gain did not survive the move.

**Stage 2 — the three validation datasets.** A setting tuned on QM9 is tuned on 10,000 small
molecules with a computed label and essentially no measurement error. LogD, Caco-2 and hERG are
drug-like, an order of magnitude smaller, and carry real assay error. There is no reason a setting
that helps the first must help the others, and one that HURTS them is not an improvement — those
three are what the validation section rests on. `scripts/confirm_tuned_on_validation_datasets.py`
refits each adopted setting and the shared default on **every fold** of the experimental pipeline's
own 5-fold scaffold CV, on clean labels, reading that pipeline's loaders, representations and
grouping rather than restating them. `--prune` then drops from the master file every pairing the
default beats, and says which and why.

**Every fold's number is written out and the summary is a COUNT of folds won** — never a mean or a
median over folds, which would hide a setting that wins big on one fold and loses on the other four.

**What stage 2 cannot cover yet:** only the families the experimental side builds from
`sklearn_params(...)` — the forest, the quantile forest, XGBoost, LightGBM, NGBoost and the SVM. Its
neural models go through `train_neural_regression` and its Gaussian process through
`GaussianProcessGauche`, neither of which takes a parameter dict. That is the same gap as §5.7d and
it overlaps exactly with the four writable keys, so nothing adoptable today is uncovered — but if
decision 8 widens the writable set, those families need a route here before anything of theirs is
adopted.

#### 5.7a 🔴 THE ROSTER IS 80 PAIRINGS, AND ONLY 24 OF THEM CAN BE DELIVERED TODAY

Counted from the generator, not from memory: thirteen models on all six representations plus the
Tanimoto Gaussian process on the two binary fingerprints is **80**, not 74. (74 is 80 minus the
quantile forest's six, which cannot be fitted on this laptop at all — see 5.7b.)

`load_best_hyperparameters(model_type, rep)` is called with a **hard-coded string**, and for seven
of the fourteen models that string is not the model's own name:

| model | reads the entry keyed | at |
|---|---|---|
| `qrf` | `rf` | `models.py:1631` — one function builds both forests and the literal is `'rf'` |
| `dnn_bnn_full`, `dnn_bnn_full_variational` | `dnn` | `models.py:2450` |
| `mlp_bnn_full`, `mlp_bnn_full_variational` | `mlp` | `models.py:3200`, via `model_type`, which is `'mlp'` for every variant |
| `gauche_rbf`, `gauche` | `gauche` | `models.py:2129` |

So **only svm, xgboost, lgb and ngboost — 24 of the 80 pairings — can be handed a tuned setting
that reaches them and nothing else.** The quantile forest would silently take the plain forest's
setting. The four Bayesian networks would take their deterministic base's. And the two Gaussian
processes share one entry per representation, which on `ecfp4` and `sns` is the *same* entry and
carries `kernel_name` — so a tuned entry there would force one kernel on both and delete the
RBF-versus-Tanimoto comparison the roster exists to make.

`models/tuning_rosters.py` records the collapse rather than hiding it, and `--write-master`
refuses to write a shared key. **Widening it is one changed literal per call site.** It is a
decision for you, not a silent edit, because it changes which models the cluster can be told
about.

#### 5.7h Two more things the builders force, found by running them

- **A tuned Gaussian-process entry must carry `kernel_name`.** Once the dict comes from the tuned
  file, `train_gauche_model` reads `params['kernel_name']` with **no fallback to `--kernel`**, so an
  entry without it raises `KeyError` rather than using the CLI kernel. The tuner pins it to the
  model's own kernel and never searches it.
- **LightGBM segfaults if it is IMPORTED after torch and tensorflow and then fits with its default
  thread count.** Exit 139, no traceback, nothing in the log. Measured 2026-08-27, three runs on
  the same machine and environment:

  | when lightgbm is imported | thread count at the fit | result |
  |---|---|---|
  | first, before the pipeline stack | its default | works |
  | after `process_and_train` | `n_jobs=1` | works |
  | after `process_and_train` | its default | **SEGFAULT** |

  So it is the import order, not the fit order — reordering which model fits first does nothing.
  Same collision as §2.8e by a different route. `scripts/tune_hyperparameters.py` and both checks
  import lightgbm in their first lines, before anything else.

  ⚠️ **`models.py` does `import lightgbm as lgb` INSIDE `train_lgb_model`**, so in the pipeline it
  is always the late one. That is only survivable because the cluster environment was rebuilt to a
  single threading runtime (§2.8i). Worth knowing before anyone runs a QM9 job on a laptop.

#### 5.7b Local limits, all measured on this laptop 2026-08-27

- `OMP_NUM_THREADS=1`, or `torch.randperm` segfaults after the import stack. The script sets it
  itself, before torch is imported.
- **LightGBM segfaults even at one thread** — exit 139, no traceback — because it loads its own
  OpenMP runtime alongside MKL's. `KMP_DUPLICATE_LIB_OK=TRUE` cures it; the script sets that too.
  Same class of failure as §2.8e, by a different route.
- `quantile_forest` 1.4.1 against scikit-learn 1.3.2 raises `Invalid parameter 'monotonic_cst' for
  estimator DecisionTreeRegressor`. **qrf cannot be fitted here at all**; its six pairings are
  written to the output as `blocked` with the reason, not dropped.
- The Gaussian process runs on one thread through `GP_DEFAULTS['single_thread_fit']`, which
  `models.py` already honours.
- QM9 is 130k molecules. The sample is **10,000**, which is what the QM9 jobs themselves run, so a
  setting is tuned at the size it will be used at. It is recorded in every output row.

#### 5.7c ✅ FIXED 2026-08-27 — NN-β's Bayesian variants could not run, and would have had no KL term

Found by the first smoke run of the tuner. `train_mlp_variant_model` built its base loss **after**
the Bayesian transformation and then overwrote it:

- `criterion = bnn_elbo_criterion(criterion, model, len(x_train))` read `criterion` before it
  existed, so **MLP-BNN-Full raised `UnboundLocalError` and produced no rows at all**;
- and had it got past that, the unconditional `criterion = get_loss_function(...)` on the next line
  would have thrown the ELBO wrapper away and trained NN-β's Bayesian variants on plain MSE with
  **no KL term** — the exact defect the KL term was added on 2026-08-27 to fix (`model_defaults.py`
  1.3.0), reintroduced by ordering.

`train_dnn_model` has always built the loss first. NN-β now does the same, and
`scripts/test_bnn_criterion_order.py` asserts the loss that reaches the trainer, for both bases and
every transformation.

#### 5.7d The experimental pipeline has no reader — RAISED, NOT BUILT

`alternative_data_noise_robustness.py` builds every model from `sklearn_params(...)` and
`NEURAL_DEFAULTS`, so a tuned value cannot reach LogD, Caco-2 or hERG today. Adding one is a
second change and is not made here. The name map in `models/tuning_rosters.py` is checked against
that pipeline's own display names so it is ready if you want it.

#### 5.7g ✅ FIXED 2026-08-27 — the tuned-parameter path would have crashed on the cluster

`load_best_hyperparameters` calls `json.load` twice, and **`models.py` never imported `json`**.
Every other use of json in that file is a function-local `import json`; this one is not. `os` was
no better — it arrived only through one of the star-imports.

So the moment both files existed and a job ran `--use-best-params`, the tuned branch would have
raised `NameError: name 'json' is not defined`. It has never been reached, for one reason only:
`results/hyperparameter_decisions.json` has never existed, so the function returned early every
time. **The entire tuned path has been dead code, not merely unused.** Both imports are now at the
top of the file, and `scripts/test_tuned_params_reach_models.py` reads the two files through the
real reader, so this cannot come back unnoticed.

#### 5.7e The old file on disk was a landmine

`results/master_tuned_hyperparameters.json`, dated 28 February, was keyed by `pdv`, `smiles`,
`randomized_smiles` and `graph` and carried `gcn`, `gin`, `residual_mlp`, `factorization_mlp`,
`mtl`, `flexible_dnn` and `conformal` — a representation set and a model set from before the
roster was settled. Its `rf` entry contained `use_default_max_depth`, an Optuna bookkeeping flag
that is **not an argument of `RandomForestRegressor`**: under `--use-best-params` the forest would
have raised `TypeError` on construction. It never fired only because
`results/hyperparameter_decisions.json` has never existed.

Renamed to `results/master_tuned_hyperparameters.superseded_2026-02.json` so nothing can read it.
`scripts/test_tuning_rosters.py` now fails on any of those names.

---

## 6. The re-run design

### 6.1 Noise types and levels

From `NOISE_DESIGN.md` §2 and §6.4, with the levels set by the range-finding run rather than
chosen in advance.

**✅ The condition set was settled on 2026-08-27 and now lives in `noise_conditions.json`, which
tests on both sides read** (§8). What runs, and why, is §13.9; what it is is here.

| | Conditions | |
|---|---|---|
| **Full grid** | Gaussian · Grouped — wider · Grouped — shifted | three zero-mean types, all delivering the same amount of noise |
| **Full grid, own axis** | Censoring | not zero-mean, cannot be dose-matched, and the largest effect in the study |
| **Depth only** | Student-t at ν = 5 · Outlier at p = 10% | one setting each, not three: the three settings of each came within 0.006 R² of Gaussian and of one another over twelve replicates |
| **Depth only** | Laplace | indistinguishable from Gaussian; its value is citational. **Kept by the author 2026-08-27** at 720 runs — not optional, and no longer a question |
| **Dropped** | Student-t at ν = 10 and ν = 3 · Outlier at p = 1% and 5% | measured redundant, §13.9 |
| **Never built** | a skewed draw | tested in the local screen and rejected; the asymmetry story is carried by censoring and by grouped-shifted, which are mechanisms with sources rather than a chosen distribution |

**Both grouped conditions run at full grid.** They differ *only* in whether the family's error is
centred, and that single difference is worth 0.10–0.31 R² — the largest zero-mean effect anywhere in
this study. The pair is the claim; neither half is.

| Dataset | Axis | Levels |
|---|---|---|
| QM9 | fraction of the label spread — there is no assay error to anchor to | **`NOISE_DESIGN.md` §6.4** |
| LogD, Caco-2, hERG | log units, each anchored to that endpoint's published assay error | **`NOISE_DESIGN.md` §6.4** |
| Censoring, all datasets | fraction of labels clipped | **`NOISE_DESIGN.md` §6.4** |

**The numbers are deliberately not repeated here.** They were, and the two documents disagreed
about them for a fortnight. §6.4 of the design owns every level grid; this section owns what the
grids are *for*. Seven levels on QM9 and six on each experimental dataset is what the cost
arithmetic in §13.1 assumes.

Each dataset's list of noise levels includes the assay error published for that endpoint and runs to
about twice it. Report the fraction-of-spread alongside, because that published error is 0.13 of the
label spread on LogD but 0.76 on Caco-2 — a factor of six.

**This supersedes the old eleven-point ladder.** Your choice of 0.6 as the reporting level
survives on the experimental datasets, where it becomes 0.6 log units — still one unit of
published assay error, still a sanity check on retention rather than a replacement for it.

⚠️ **It does not survive on QM9, and I wrote that carelessly above.** The QM9 axis is a fraction
of the label spread, and the grid set by the range-finding run has no 0.6 point. **You need to
pick the QM9 reporting level explicitly**, from its own grid. My suggestion is 0.5 — it is on the
grid, it sits mid-range, and 0.5 of the label spread is close to the 0.57 that 0.6 electronvolts
would have been under the old scale, so continuity with everything discussed so far is preserved.
But it is a choice and it has to be made, because every table that reports accuracy at one level
depends on it.

### 6.2 The main QM9 grid

Unchanged in shape from `slurm_scripts_qm9_rerun/RUNBOOK.md`, which is sound apart from the noise
types and levels: eleven models for the variance decomposition plus the quantile forest and both
Gaussian processes; six representations; ten replicates. The runbook's reasoning about what is in
and what is out is good and should be carried across when the scripts are regenerated.

Two things in it need correcting before it is used again:
- The noise types and levels are the old ones.
- The completeness check at the end globs for the old type names.

### 6.3 The experimental datasets and the uncertainty runs

One set of jobs produces both the robustness numbers and the uncertainty numbers, because the
out-of-fold pass is the only added cost and the five scaffold folds are trained regardless.

Scope, updated for the settled condition set (§13.9): 3 datasets × 7 models that emit a per-molecule
uncertainty × 4 representations × **4 noise conditions** × 6 levels × 5 folds. Two changes from what
the built scripts assume, and they compound: six levels instead of eleven, and four conditions
instead of six.

⚠️ **The four are the full-grid four, and one of them behaves differently here.** Gaussian, both
grouped conditions and censoring. Grouped-shifted matters more on the experimental datasets than
anywhere, because it is the condition that mimics a laboratory offset and these are the datasets
where that is a real mechanism rather than a simulation. The single-setting conditions —
Student-t at ν = 5 and Outlier at p = 10% — belong to the depth stage, so they enter here only if
the depth stage is run on the experimental side.

**One thing to check before pricing this, not to assume — and it is §13.1 open item 6, not just a
note here.** §13.9 measured redundancy **on QM9**, one representation and three tree/linear models.
The experimental datasets are smaller and noisier, and the uncertainty question is about which
*molecules* are corrupted rather than how much accuracy is lost — §5.3 of `NOISE_DESIGN.md` notes a
model can lose the same accuracy while being much better or worse at spotting corruption. So the
condition set is settled for the **accuracy** grid and is **not** established for corruption
detection. Inheriting it is the honest default and costs nothing; testing it costs two extra
conditions across the whole uncertainty grid. Either way the Methods must say which was done.

### 6.4 The levers, in the order they cost you least

| Lever | Saving | What you lose |
|---|---|---|
| Six levels instead of eleven | ~45% on the uncertainty runs | nothing — the eleven-point ladder was never justified, and the range-finding run shows the curves do not move below 0.2 of the label spread |
| Cross-fit only the first of the five scaffold folds | ~3× on each uncertainty task | the spread of question A across folds. Test-side data for all five folds is unaffected |
| Three cross-fitting folds instead of five | ~40% on each uncertainty task | each molecule is scored by a model trained on 67% of the data rather than 80% |
| Five replicates instead of ten on QM9 | ~50% | real but bounded — every gate in the analysis needs at least five. It costs precision on the residual term, which is itself a reported result |
| Fewer representations | ~60% | **guts the first research question. Do not.** |

---

## 7. The three headline results, and what each one needs

### 7.0 The questions, and the statistic that answers each

Written 2026-08-26 at the author's request: *"It would be nice to briefly plan out what questions
I'm trying to answer and how? With what specific statistics?"* **Every entry names the statistic,
what it is computed over, and what the design must supply for it to be computable.** If a row's
requirement is not met, that question cannot be answered and the run should not claim it.

| # | Question | Statistic | Computed over | What the design must supply |
|---|---|---|---|---|
| **Q1** | Is noise robustness decided by the model, the representation, or their pairing? | Two-way variance decomposition, model × representation, reported as the share of variance each term explains, with the residual shown | One value per (model, representation, replicate) cell, **separately for each noise type** | ≥2 replicates per cell for the residual to be real; **≥6 for any paired follow-up test** (§13.1); one roster and one exclusion rule applied everywhere |
| **Q2** | Does the *kind* of noise matter, or only the amount? | (a) Spread in accuracy across noise types at matched delivered amount; (b) paired signed-rank test, each type against Gaussian, per model, **not pooled** | Paired on the replicate, within one representation and one noise level | Delivered amount verified flat across types (§8 gate 1). Six replicates minimum, or no result can reach significance |
| **Q3** | What does model choice actually buy you at a realistic amount of error? | Accuracy at the anchored noise level; best-minus-worst across models at each level; retention area **printed beside its clean baseline, never alone** | Per (model, representation, noise type, level) | The anchored level chosen per dataset. ✅ **All four settled 2026-08-27** — QM9 1.0, logD 1.0, Caco-2 0.2, hERG still to set (§13.11) |
| **Q4** | Can a model's uncertainty tell you which labels are bad? | **Two numbers, settled 2026-08-26, reported side by side under names that cannot be confused.** (a) The plain Spearman correlation between predicted uncertainty and the size of the injected noise, **within each noise level**, scored **out-of-fold** — near zero by design, because the scoring model never saw that molecule's draw and under an even condition every molecule gets the same amount. (b) The answer: take the out-of-fold error `\|y_clean + injected − y_pred\|`, which does track the noise, and ask whether dividing it by the predicted uncertainty ranks corrupted labels better than the error alone. Both with a permutation null | Per (dataset, model, representation, noise type, level) — never pooled | The noise **recorded**, not reconstructed (§5.2); out-of-fold scoring on scaffold groups; a zero-noise run of the same type to subtract. ✅ **All three now exist on both pipelines** (§3.1c). Built in `scripts/uncertainty_stats.py` |
| **Q5** | Does noisy training data make a model less sure about new molecules? | Mean predicted uncertainty against noise level — a **population-level** statement, and it must be labelled as one | Per (model, representation, noise type), across levels | Uncertainty magnitudes on a fixed scale — needs the standardisation fix (§2.4), which currently makes them shrink as noise rises |
| **Q6** | With noisy training data, does uncertainty still rank which predictions to trust? | Spearman correlation between predicted uncertainty and absolute error **against the clean label** | Per (model, representation, noise type, level) | Clean test labels retained alongside noisy ones. Free — every run already produces both |

**Two things this table settles.**

Q4, Q5 and Q6 are three different questions and the paper has repeatedly fused them. Q5 is the
easy one and it was already known; Q4 is the hard one and is where the pooled correlation went
wrong; Q6 is the one a referee expects and it is currently in neither run plan.

And Q1's requirement is the reason the replicate count is not a free parameter: **the residual term
of the decomposition and the paired test in Q2 are both replicate-limited**, and a two-sided
signed-rank test on *n* replicates cannot return a p-value below 2/2ⁿ whatever the effect size —
0.0625 at five replicates, 0.03125 at six. Five replicates makes Q2 unanswerable by arithmetic.

### 7.0a ✅ 2026-08-27 — the analysis exists now, in one module

Until this chat nothing computed anything from an uncertainty run. `merge_results.py`
concatenates and stops, and no other file in any of the three repositories read the output. Every
statistic in §7.0 is now in `scripts/uncertainty_stats.py`, with one loader that reads both
producers' schemas, and `generate_paper_figures_v2.py` calls it rather than recomputing anything.

**The permutation null is built the way `slurm_scripts_uncertainty_rerun/RUNBOOK.md` specifies**,
and its test reproduces the runbook's measured numbers independently: permuting the injected noise
while leaving the error as computed gives a band of [-0.040, +0.034] against an observed +0.616, so
it fires on simulated data with no leakage at all; recomputing the error from the permuted value
gives [+0.601, +0.634] with the observed value inside it and p = 0.86. Eighteen tests, all
executing, all passing.

**Two guards are assertions rather than notes.** Before any correlation, the module checks the
group is exactly one (dataset, model, representation, condition, level, split) cell — pooling
across a dimension that should have been conditioned on is what produced the paper's per-molecule
claim. And the pooled-across-levels correlation in the figure script no longer shares a name with
the within-level ones: it is written as a population trend and deliberately does not begin with the
same prefix, so no column glob can sweep it in among them.

**The experimental-dataset loader was discarding most of the run.** It de-duplicated on
`dataset, model, rep, sigma, sample_idx`, and the rebuilt runner writes two splits × six conditions
× five folds per molecule. Measured on a file with two splits, three conditions and two folds:
**880 of 960 rows discarded, 91.7%**. On the real grid it would be 98.3%. The key now carries
split, condition and fold, and the same file loads losslessly.

### 7.1 Variance decomposition, per noise type

Model against representation against their interaction, computed separately for each noise type,
for both predictive performance and robustness.

What it needs: replicates inside each cell (QM9 has ten, the experimental sets have five folds);
the fold axis preserved through the integration (§5.4 item 2); and the same roster and exclusion
rule stated once and applied everywhere.

**Expect this result to change shape, and that is the point.** The current finding — that the
residual dominates under outlier and heteroscedastic noise, at 83.6% and 77.4% — is a dose effect.
Those two types inject the least noise, so they separate the models least, and run-to-run wobble
is most of what is left to explain. Once every type delivers the same amount, that asymmetry
should disappear. If it does not, that is a genuine finding about noise shape.

The clean-data half of the decomposition is worth stating carefully too: on QM9 the interaction
term was the largest on all six types, but on clean LogD the model term led at every level
including zero. Only the re-run settles which is real.

### 7.2 Does uncertainty track per-sample noise

The two questions, the confound control, and the guards are in §3.1 and in
`slurm_scripts_uncertainty_rerun/RUNBOOK.md`. What has to be added to that plan:

- The set of conditions changes as in §3.2. Grouped is the positive case, random-selection outlier
  is the honest null, censoring is the label-keyed case, and the unstructured shapes are
  question A's conditions.
- Report a permutation null with every number, and construct it correctly: permute the injected
  noise *and recompute the residual from the permuted value*, so the observed and null statistics
  both carry the same additive term. The naive version fires on clean simulated data.
- Recompute the uncertainty-versus-error correlation against the **clean** label. It is currently
  measured against the noisy label, so the error being ranked contains the noise itself.

**One finding worth re-testing rather than re-deriving.** Across QM9 and all three experimental
datasets, the quantile forest was consistently the best at ranking a model's own errors and
NGBoost consistently the best calibrated. Two different things, two different winners, replicating
across four datasets. The paper currently dismisses the quantile forest. If it holds on the new
data it is a better result than the one it replaces.

### 7.3 Robustness and retention

Three numbers, always printed together, never one alone:

- **Clean R²** — can this pairing learn the property at all. A precondition for the other two
  meaning anything.
- **R² at one stated noise level** — what you actually get when labels carry about one unit of
  real assay error.
- **Normalised area under the retention curve** — the shape of the decline, each model divided by
  its own clean score.

That third one is what you decided to keep, and the reason to keep it is that it measures
something the other two do not. But it must never be printed without the baseline beside it,
because a model that starts weak has less to lose.

**The mechanism, which survives the re-run; the numbers, which do not.** On the old data the two
metrics named the same best model in barely a quarter of cells, and one model topped delivered
accuracy on a dataset while topping retention nowhere. Do not carry those counts forward — they
are being recomputed. Carry the mechanism: **retention divides the baseline out, so it is
independent of it by construction, and a model can score near the top by having had little to
lose.** That is arithmetic, not a finding, and it is guard 4.

**One property of the metric that is not about any particular number, and will recur.** Retention
is a ratio with a noisy denominator. On the experimental data a sixth of the ratios exceeded one —
the model scoring *better* with noise added — and some integrated areas came out negative. Its
spread was several times worse for weak configurations than strong ones, which is exactly where
robustness matters most. **This is structural**: it follows from dividing by a small noisy number
and cannot be patched by raising the exclusion gate. It will reappear on the new data unless
fewer, better-chosen levels and a visible baseline are both in place. Check it explicitly on the
regenerated output rather than assuming the redesign fixed it.

**A correction the paper must carry, and this one is about the sentence, not the number.** The
claim that robustness is decoupled from clean-data accuracy is the artefact rather than the
finding — the metric removes the baseline by construction. Whatever the new data says, the
sentence has to change, because as written it presents an arithmetic identity as a result. The
honest form is a question the new data can actually answer: does *delivered* accuracy under noise
track the clean baseline, and does the answer differ by dataset? On the old data it did differ,
and even flipped sign between datasets — worth re-testing, worth nothing as a quoted number.

---

## 8. Verification gates — none of this costs cluster time

No job is submitted until all of these pass locally.

**Status, chat A 2026-08-26.** Gates 1, 3, 4, 5 and 7 are now executable checks that fail rather
than notes that reassure. Two commands run them:

```
# the noise conditions, on the real label column — this is the preflight gate
./rust/target/release/rust_processor --self-test <labels.csv> --scaffold-file <groups.json>

# the pipeline, over real mmap files — 33 gates (28 noise gates + 5 writer guards)
cd rust && cargo test --release
```

**Gate 2 is now executable too — chat B, 2026-08-26.** One more command, and it needs no cluster
time either:

```
# the two injectors agree: 342 checks on all 133,885 real QM9 labels
python scripts/crosscheck_injectors.py
```

It also covers gates 1, 4 and 5 on the Python side, and it found four real disagreements before it
passed — including an `effective_n` formula that was wrong in **both** implementations (§2.3, §2.3a).
Its dose-matching check was verified to fail, on 8 of 10 conditions, when the solver is removed.

Gates 6 and 9 need a training run and are chat H's. Gates 8, 10 and 11 are chat D's. Each of chat
A's gates was checked by removing the fix and confirming the gate fails.

**One more gate, and it costs a second — chat G, 2026-08-27.** It guards the settled condition set:

```
python3 scripts/test_noise_conditions.py
```

`noise_conditions.json` at the repository root says what the study runs — four conditions at full
grid, two at depth, one optional, four dropped and one never to be built, each with the measurement
behind it (§13.9). The file is **read by tests, not merely documented**, in three places:

| Checked | By | Against |
|---|---|---|
| The Python injector | `scripts/test_noise_conditions.py` | every name resolves in `noiseInject.CONDITIONS` with the settled parameters |
| The Rust injector | `rust/tests/noise_gates.rs` | the self-test covers every condition that runs, the dropped settings stay dropped, and the command-line defaults are the settled settings |
| The QM9 job generator | `scripts/test_noise_conditions.py` | its full-grid set and its spelled-out `--nu` and `--outlier-p` match |
| **Every condition, run for real** | `smoke_every_settled_condition_runs_end_to_end` in `rust/tests/noise_gates.rs` | each one executes end to end, the manifest names it, it delivers what it asked for, held-out labels are untouched, and the recorded noise reconstructs the label per molecule |

All four were verified to fail: put a dropped setting back and the Python gate names it; change a
settled parameter and the Rust gate names both the number and the number it should be; move a
condition between stages and the generator check says which side is about to run and which is about
to be believed; add a condition nothing exercises and the smoke test refuses it by name, because a
condition the study runs but nothing runs end to end is a condition that ships unverified.

**The smoke test is the one that answers "will the grid actually run".** The others each prove one
property of one path; that one runs all seven settled conditions in turn through the real binary.
It exists because "the unit tests pass" and "the grid will run" are different claims, and this
project has already queued jobs that ran five folds over nothing (§2.8d).

---

### 8a. 🔴 The preflight exited 1, and now does not — 2026-08-27

**The command every operator is pointed at failed on a 4,000-molecule column.** Deterministic, and
reproduced independently. Exactly one gate was responsible: gate 11, *"validation is dosed against
the clean training spread, not its own"*, on Student-t ν = 3 — validation delivered **+36.54%**
against a band of 21.21%, and the run stopped there.

**The cause is not the anchoring rule. It is the statistic used to test it.** Gate 11 built one
draw on training and one on validation, at `seed: 42`, and compared them. Student-t ν = 3 is the
heaviest-tailed shape on the grid, and the same self-test measures its per-run delivered dose
spread at **14.5%** on training — the validation part is a fifth of an already small column, so its
own spread is larger still. The ratio of two single heavy-tailed draws is not a statistic about the
anchoring rule; it is mostly the draws.

**This is the third gate to need the same fix and the first that stops a run.** The flat-dose gate
went from 20 seeds to 200 and the ν→∞ gate from one draw to 50, both on 2026-08-27, for exactly
this reason — a launch gate that fails at random is worse than no gate, because the next person
turns it off. Gate 11 was missed. Rule 3 of `NOISE_DESIGN.md` §2a says it in one line: gate on the
population, never on one realisation.

**The fix.** `VALIDATION_SEEDS = 100` in `rust/src/main.rs`. Both sides are built at each of 100
seeds and the **ratio of the two means** is compared against three standard errors of that ratio,
floored at 0.5% — the same rule the flat-dose gate above uses. Each side's per-run spread is printed
beside it, so the reader can see which conditions are noisy (guard 4).

⚠️ **The first version of this fix used the mean of the per-seed ratios, and that was wrong twice.**
Caught on review the same day, before it went further than this document. It **contradicted its own
printed components** — on the 4,000-molecule column it read +0.39% for ν = 3 while the train and val
means printed beside it give −0.99%, the opposite sign, which is precisely what guard 4 exists to
stop. And the two draws are independent, so E[V/T] ≈ (E[V]/E[T])(1 + CV²): ν = 3's per-draw spread of
about 14% biases that statistic **upward by roughly 2%**, worst on the heaviest tail, which is the
shape the gate is already weakest on. The ratio of means has neither problem.

**It also made the gate much stronger, which is the point.** The old bands ran 8.6%–28.7%; the new
ones run 0.92%–5.84%. It now detects an error five to twenty times smaller than before while no
longer failing by chance.

**Verified four ways, all re-run after the statistic was corrected.**

| Check | Result |
|---|---|
| The 4,000-molecule column that failed | ✅ `EXIT=0`, all gates pass. ν = 3 now reads **−0.99%** against a band of 5.84%, and 0.647348/0.653810 reproduces it exactly |
| The full 132,480-label column | ✅ `EXIT=0`, all gates pass. Validation spread is 3.37× training's and every condition lands within 0.12% |
| **The fix removed** — validation dosed against its own spread | ✅ **All ten conditions FAIL**, at **+178.67% to +183.38%** against bands of 2.58%–16.42%. The gate still catches what it exists to catch, by an order of magnitude and more |
| 1,000 / 2,000 / 4,000 / 8,000 / 20,000 molecules | ✅ `EXIT=0` at every size. The verdict no longer depends on how much data the gate is handed |

**The last row is the part worth keeping.** The gate passed on the full column and failed on a
subsample — so it was *right for the wrong reason*, and its verdict tracked sample size rather than
the property under test. Any gate whose answer depends on how much data it is given will eventually
be given the wrong amount.

**Why a file and not a note.** `scripts/noise_strategy_params.json` was a settings file that nothing
read: it was never passed to the binary, so for the life of the project it silently meant nothing
while everyone believed it was in force (§2.2). A file that describes the run and a run that ignores
the file is worse than no file.

**One more gate, and it costs a second — chat K, 2026-08-26.** It guards the documents and the
bibliography rather than the noise, which is why it sits outside the numbered list:

```
python3 scripts/check_bib_and_docs.py
```

It fails if a cited key is undefined, if two entries collide on one key, if a source named in
`NOISE_DESIGN.md` § Sources has no bibliography entry, if one of the five rejected sources in
§4a reappears, or if a single-owned fact — the level grids, the noise-type count, the
threshold-degeneracy figure — starts appearing in both documents again. It reports the outstanding
`\bibliography` line in `paper.tex` (§9.1) as a pending manuscript edit rather than a failure.

1. ✅ **The dose is flat across noise types.** At one target, on the real training labels, every
   type must deliver the same measured amount. This is the single check that proves the confound
   is gone. If it fails, the entire re-run is confounded and worthless.
   **Two halves, per `NOISE_DESIGN.md` §2a rule 3.** The construction — unit dose times solved
   scale equals the target — is asserted *exactly* on every single run. What one realisation
   delivered is asserted against a band the injector works out from the draw itself: the relative
   standard error of a second moment is `√((kurtosis − 1) / (4·n_eff))`, which reproduces the
   0.19% spread §5.1b measured across 40 seeds at n = 133,885. A flat half-percent band would fail
   correct code on a small dataset and pass a broken solver on a large one.
   **Measured on 4,000 real QM9 molecules with real Murcko groups:** mean delivered dose over 20
   seeds within 0.74% of target for every condition, spread between conditions **1.27%**.
2. ✅ **The two injectors agree.** Same labels, same target, same scaffold groups, same seeds.
   **Executable: `python scripts/crosscheck_injectors.py`** — 342 checks on all 133,885 real QM9
   labels, exits non-zero on any failure. It compares statistics rather than individual draws,
   because the two languages' generators produce different streams and a check written as an
   element-wise comparison would fail for a reason that does not matter and would then be turned
   off. Tolerances are **derived per condition** from its fourth moment and its effective number of
   independent contributions, not kept as a list of "unstable conditions" that would need editing
   whenever a condition is added.
3. ✅ **Held-out labels are untouched.** The clean-label column must be bit-identical across every
   noise level. This is the check that caught the original bug.
   `held_out_labels_are_bit_identical_across_levels` compares the written held-out labels bit for
   bit across every condition and every level, and fails if noise reaches one. Re-applying noise
   to the held-out splits makes it fail.
4. ✅ **The recorded noise reconstructs the label exactly.** `y_clean + epsilon == y_noisy`, every
   type, every level. `recorded_noise_reconstructs_the_label`, asserted with `==` on f32, not a
   tolerance — the noise is recorded where it is drawn, so there is nothing to round.
5. ✅ **Zero noise records exactly zero.** Not a small number — zero. This is the negative control
   the old reconstruction never had. `zero_level_records_exactly_zero`, every condition.
6. **Nothing changed at zero noise.** The clean-label R² must reproduce the existing zero-noise
   numbers, because nothing about that path has changed.
7. ✅ **Student-t reduces to Gaussian in the limit.** At 200 degrees of freedom the two must be
   indistinguishable — checked on both the delivered dose and the tail fraction, inside
   `--self-test`.
8. ✅ **A short record no longer desyncs the file.** Implemented both sides (chat D).

   Writer — `rust/tests/writer_guards.rs`, five tests. It does **not** feed the binary an
   unparseable molecule: it cannot any more, because nothing in Rust parses SMILES since ECFP4
   moved to Python (§2.7 item 2). It plants an **all-zero fingerprint block** instead, which is
   what a molecule that could not be featurised now looks like by the time the binary sees it,
   and asserts that the record is still the length the reader expects, that the file is the exact
   sum of its records' expected lengths, that the molecule is reported as a failure, that a full
   fingerprint is *not* reported as one, and that the configuration path has no default.

   Reader — `scripts/test_record_alignment.py`, five checks: a short record raises rather than
   returning silently wrong features; a well-formed file is consumed exactly; a representation
   with no reader is refused by name; and every representation the reader accepts decodes to a
   fixed width — the assumption `parse_mmap`'s uniform-width check rests on. That last check is
   the honest replacement for a test of the width guard itself: it is unreachable today, because
   every accepted representation reads a fixed number of bytes and the two whose rows could
   differ in width are refused before the loop starts. The test fails the day someone adds a
   variable-width representation, which is when the guard needs revisiting.

   `cargo test --release --test writer_guards` and `python scripts/test_record_alignment.py`.
9. **Every new column is populated** in a smallest-possible end-to-end run.
9c. ✅ **A failure in the injector stops the run.** `python scripts/test_failure_propagation.py`.
    Until 2026-08-26 the pipeline never inspected the injector's return code, so every gate in
    this list that the Rust half enforces was reported to a pipe nobody read (§2.8g). **No
    Rust-side gate was enforced end to end before that commit.**
9b. ✅ **The interpreter can build what the job asks for.** Wired into **all three** job
    templates (2026-08-27): `--models <label>` for QM9, `--validation-models <label>` for the
    validation and uncertainty families, which use KIRBy's model names rather than
    `process_and_train.py`'s. `--audit-roster` checks that every label all three generators can
    emit is known to the probe, so no job can fail its own guard for the guard's own reason.
    The runbook no longer diffs two cluster interpreters — `py311-kirby` is missing eight of the
    roster's packages and the runbook now says not to use it, so there is one (§2.8d).
10. ✅ **Two tasks running at once do not corrupt each other.** Implemented (chat D):
    `python scripts/test_config_isolation.py` launches two binaries concurrently in one directory
    with different representations and asserts each keeps its own data, plus an instant static
    half that fails if a fixed `config.json` reappears anywhere in the tree. `--end-to-end` runs
    two real pipeline tasks side by side (§2.8a). **All three passed on 2026-08-26**, including
    the two real tasks — which is also the first end-to-end confirmation that the QM9 pipeline
    runs on the laptop again, not merely that it imports.
11. **The checkout being used is the live one.** Confirm which copy of KIRBy the cluster actually
    updates before submitting anything against it (§2.8b).

---

## 9. Paper defects that need no compute

**Split by whether the re-run kills them.** Most of what an earlier pass listed here was
"the table says X, the file says Y" — and every one of those evaporates the moment the numbers are
regenerated. Listing them is busywork. What matters is the half that survives, because a
regenerated number does not fix a wrong sentence about the method, a retired metric in the
Conclusion, or a broken bibliography.

### Survives regeneration — these are wrong about the *method*, not the values

| Where | Problem |
|---|---|
| `:197`, Additional file 12 | Claims a Tanimoto kernel for the support vector machine on binary representations. **The code uses a radial basis function throughout**, and `:262` already says so. Wrong in two places, and self-contradictory |
| `:354` | "on normalized data" — noise goes onto the raw label and standardisation happens afterwards (§2.4) |
| `:354` | "the difficulty scaling controlled by σ is consistent" across types. False as the code stands (§2.2). Under the redesign it becomes true, so the sentence should say *how* it was made true rather than being deleted |
| `:313` | "Validation and test data remain free of noise" — false for QM9 when the results were generated, and about to become deliberately false for validation (decision 2) |
| `:186` | The Heid et al. citation misrepresents its source. They did not *show* that noise is structure-dependent; they **imposed** structure-dependent noise and showed a method could recover what they injected. Verbatim quote in `NOISE_DESIGN.md` §3.6 |
| `:193` | "Experiments that involved tracking uncertainty values were only run once." Most were run ten times |
| `:571-573`, `:598`, `:660-669`, and throughout | The retired slope metric survives everywhere, including the Conclusion, where it is defined as *the* robustness metric of the paper |
| `:234-238`, `:300-302`, `:503-528`, `:587`, `:667` | The expected calibration error is defined, tabulated and abbreviated, but you removed it from all three scripts. Those cells cannot be filled |
| `:462`, `:493` vs `:567` | PDV is called both the most and the least noise-robust representation in the same section. Whichever the new data says, **one of these sentences is arguing against the other** and the structure has to change, not just the number |
| `:464` | Reports that only the variational configurations were excluded on the two embeddings. The fully Bayesian networks failed on exactly the same representations. That omission is load-bearing, because the paper's story is that the full transformation is the robust one |
| Structure | Results then Conclusion. Journal of Cheminformatics expects a combined Results and discussion, plus Conclusions |
| `:203` | Describes the representation set. **Rewrite for the new six** — mol2vec and one-hot SMILES are out, Avalon and ChemBERTa are in (§13.7) |
| `:493` | "For SMILES, model choice explains over 91% of robustness variance, while on PDV, model choice explains 72%." SMILES has been dropped, so this illustration has no representation behind it. The underlying claim survives; the example must be rebuilt from whichever representation now sits at the extreme |
| `:466` | "SMILES receiving the largest benefits from Bayesian transformations." Same reason — the representation is gone |
| Bibliography | Broken outright — see §9.1 |

### Evaporates on regeneration — do not spend time on these now

Recorded only so nobody re-discovers them and thinks they matter: the variance-decomposition table
disagreeing with its own source file, the concordance coefficient being printed at a stale value,
a significance verdict that flips, the exclusion threshold quoted inconsistently, and the "no
neural configuration in the top ten" claim. **Every one is a number that is being recomputed.**

The only thing to carry from them is that the paper contradicted its own supporting file in at
least three places, which is what guard 12 exists to prevent — one script generating every number
*and* its caption, so prose and table cannot drift apart again.


### 9.1 ✅ The bibliography — fixed on the repository side, one line left for you

**Rebuilt 2026-08-26 (chat K).** The state recorded here before was measured weeks earlier and had
drifted in both directions: the manuscript was in better shape than it said, and `citations.bib`
was in worse shape.

#### 🔴 The one thing left, and it is yours because it is in `paper.tex`

`paper.tex:694` reads `\bibliography{sn-bibliography}`. **There is no `sn-bibliography.bib`
anywhere in the repository** — the bibliography is `citations.bib`. Every citation in the master
build is unresolved as a result: `_build_paper/paper.aux:128` carries `\bibdata{sn-bibliography}`
and `_build_paper/paper.log` warns on all 51 cited keys.

```
paper.tex:694    \bibliography{sn-bibliography}   ->   \bibliography{citations}
```

⚠️ **Do not make the same change to `paper_inline_bbl.tex`.** That file is the submission build and
deliberately inlines the compiled `.bbl` instead of running BibTeX; its own comment at `:695-696`
says so, and its `paper_inline_bbl.bbl` already resolves all 51 keys. Only the master `paper.tex`
build is broken.

#### What the earlier note got wrong

- **Not seven undefined keys — one, and it was a case mismatch.** `Fang2022`, `jorner2021`,
  `Mustapha2016`, `Song2022` and `Wolpert1997` had all been added to `citations.bib` since this
  section was written. `Islam2019` was never missing: its entry at `citations.bib:589` opens
  `@article {Islam2019,` with a space, which a naive key scan misses and BibTeX accepts. Only
  `Rogers2010` was nominally unresolved — the entry key was `rogers2010`. Traditional BibTeX
  matches keys case-insensitively so it resolved; biber would not have. **Renamed to `Rogers2010`.**
- **A collision nothing had recorded.** `Xu2019` was defined twice, on two *different* papers — the
  L_DMI noise-robust loss function and "How Powerful are Graph Neural Networks?". BibTeX silently
  keeps the first, so a future `\citep{Xu2019}` for the graph-network paper would have cited the
  loss-function paper instead, with no warning. **Split into `Xu2019dmi` and `Xu2019gin`**, leaving
  no bare `Xu2019`, so any such citation now fails loudly instead of resolving to the wrong source.

#### The guard

`scripts/check_bib_and_docs.py` asserts all of this and fails the run if any of it regresses —
including the one line above, which it reports as an outstanding manuscript edit rather than a
hard failure so it turns green on its own once you make the change.

### 9.2 The writing principles — keep these, they are not about numbers

Derived from nine Nature Machine Intelligence papers during an earlier pass. They survive the
re-run completely, and the revision guide is the only place they exist.

- **Numbers never appear in the Abstract or the Scientific Contribution.** Only in Results, tables
  and captions.
- **A number rides behind a plain adjective, in parentheses, with its full statistics** — never a
  bare number, and never the word "significant" without a test attached.
- **Results subheadings state the finding as a claim**, with the mechanism as the grammatical
  subject.
- **Reversals are stated plainly and owned.** Narrow results carry an explicit scope caveat.
- **The metric is simply defined and used.** No justifying a change, no naming a predecessor
  metric, no mention of a distribution family that was dropped.

And the name map every table depends on, which exists nowhere else:
NN-α = `dnn`, NN-β = `mlp`, BNN-α = `dnn_bnn_full`, BNN-β = `mlp_bnn_full`,
VBLL-α = `dnn_vbll`, VBLL-β = `mlp_vbll`.

### 9.3 The claim-by-claim triage table

The revision guide carries an eighteen-row table mapping every headline claim in the paper to its
fate — strengthens, breaks, reframe — with the location in `paper.tex` for each. **Its numbers are
being regenerated and are dead. Its structure is not**, and rebuilding that mapping from scratch
after the re-run would be a day's work. The rows whose *verdict* does not depend on the new
numbers:

- The per-sample uncertainty claim was withdrawn. **Do not carry the number forward** — it came
  from an analysis that was wrong in three ways at once, so it measures nothing. What survives is
  the reason: the correlation pooled two dimensions it should have conditioned on, the injected
  noise was reconstructed rather than recorded, and the molecules being scored were never
  corrupted. All three are now guarded (§0.6, guards 1, 2 and 7), and the new design answers a
  differently-shaped question (§3.1).
- The support-vector-machine kernel claim in the Methods is wrong and contradicts the paper's own
  later sentence.
- The exclusion gate is stated inconsistently.
- The retired slope metric appears throughout, including in the Conclusion where it is defined as
  *the* robustness metric.
- The "high baseline has more to lose" apology, which appears twice, is an artefact of the retired
  metric and should simply go.

---

## 10. Ordered work list

Nothing here is started. Steps 1 and 2 are the only ones that need you.

| # | Step | Blocked on |
|---|---|---|
| 1 | Settle the five open decisions in §4 (the sixth was withdrawn — already decided 2026-08-14) | you |
| 2 | ✅ **CLOSED 2026-08-27 — `NOISE_DESIGN.md` §7 has nothing open.** Laplace is kept at depth, the dose-matching rule was approved 2026-08-21, the positive-control question and the level grids were closed 2026-08-26, and the condition set was settled 2026-08-27 | — |
| 3 | **Do NOT blanket-cancel the Gaussian-process jobs.** I previously said to kill job range 12822669–12822694. That was wrong: you submitted them deliberately on 2026-08-19 to settle a live question — *"Unsure if I should do tanimoto or switch to rbf. Or do both … It would be nice to include it in the anova and the kernel difference is holding me back."* Their zero-noise rows answer that question whatever happens to the noise scheme, because no noise is drawn there. **Let the zero-noise point land, harvest the kernel comparison, then cancel the rest.** Check state first: `sacct -j 12822669-12822694 --format=JobID,JobName%24,State,Elapsed` | you |
| 4 | Archive the current results before anything overwrites them — they are the only record of what the paper claims today | — |
| 5 | Delete and build the noise scheme in Rust (§5.1 items 1–4) | 2 |
| 6 | Implement the same specification in Python and cross-check (§5.1 item 3) | 5 |
| 7 | The remaining code changes that alter what is trained (§5.1 items 5–9) | 1 |
| 8 | The three free savings (§5.3) | — |
| 9 | Every verification gate in §8, locally, at 4,000 molecules | 5, 6, 7 |
| 10 | Regenerate one deduplicated set of job scripts from the design | 9 |
| 11 | Rebuild the release binary on the cluster and run one task end to end before submitting the rest | 10 |
| 12 | QM9 re-run | 11 |
| 13 | Experimental datasets and uncertainty re-run | 11 |
| 14 | Rebuild the analysis against the new columns (§5.4) — most of it can start while 12 and 13 queue | 12, 13 |

**Two environment problems that will block step 9 and step 11 if they are not dealt with first.**
The local Python environment cannot import `torch_geometric` — two compiled extensions do not
match the installed PyTorch — so the QM9 pipeline cannot be run on this laptop at all. And the
quantile forest cannot be fitted locally either, because `quantile_forest` and `scikit-learn`
disagree on a parameter name. The uncertainty preflight script checks the second one explicitly;
the first needs fixing or the verification has to move to the cluster.

---

## 10b. Carried forward from the revision guide

Salvaged 2026-08-25 by reading `REVISION_GUIDE.md` directly, before it was scrapped at step 3 of
your process. **The file no longer exists**, so this section is the only surviving copy of what is
below. The verified literature quotes went into
`NOISE_DESIGN.md` §4b, and the two literature passes are reconciled in §4a there.

### 10b.1 Six claims that were asserted and then withdrawn — and why that keeps happening

The claims themselves are dead: every number under them is being regenerated. What transfers is
the shape of the mistake, because it recurred six times in four days.

In five of the six cases a pattern was found in a **raw, ungated** computation and written down
before being checked against the gated pipeline output. One model looked like it collapsed because
a single divergent cell dragged its mean down. A cross-dataset agreement figure of +0.09 came from
the finest granularity under the noise types that barely degrade anything; the grounded value is
+0.79 to +0.93. One claim — that a noise type had never been run on the experimental sets — was
simply false.

The sixth is different and worth keeping in mind: two models genuinely do sit at opposite ends of
the retention-versus-delivered-accuracy trade, and which one is the flattered case was itself
revised once. That is guard 4 — never print a ratio without its components — and it is why the
guard exists rather than a note saying "be careful".

**The rule this implies for the new analysis:** no pattern is written down until it has been
computed through the shipped code path, with the declared gates applied. A pattern found in a
scratch computation is a hypothesis, and it is labelled as one.

### 10b.2 The kernel facts, and the re-run they justify

Established from code, and they settle a paper error and a design question together.

**There is one Gaussian process, not two.** The kernel is a parameter. The results label is set by
the kernel alone, so the two names in the results directory are the same model with different
kernels. Tanimoto is mathematically defined only on fingerprint vectors, so it cannot run on the
descriptor representation at all. **Any comparison of that model across representations is also a
comparison across kernels, and the paper must say so.**

**The support vector machine uses a radial basis function on every representation, in both
pipelines.** That makes its cross-representation comparison fair, and it is why it belongs in the
variance decomposition. It also means `paper.tex:197` and Additional file 12 are wrong, and
contradict `paper.tex:262`, which already says the right thing.

#### ✅ ANSWERED, 2026-08-25 — the two kernels agree where the features are sane

Harvested from the clean-data rows of the jobs on the cluster, ten seed-matched replicates each:
ECFP4 gives 0.823 with Tanimoto against 0.820 with the radial basis; SMILES gives 0.806 against
0.803. **A difference of 0.003.** On that evidence the kernel is not the reason the Gaussian
process was excluded, and committing to the radial basis everywhere is defensible — which lets
that model finally enter the variance decomposition alongside the support vector machine.

**One condition.** That conclusion holds only where the features are correctly scaled. On the two
learned embeddings the same comparison gives 0.87 against −0.02, and that gap is the data defect
in §2.8c, not a kernel property. **Fix the embedding scaling first, then re-check the kernel on
those two representations** before treating the answer as general.

**The replay check passed.** The six noise-type files agree at zero noise to six decimal places or
better, so the configuration race in §2.8a was not firing on these runs. They are not bit-identical
— the residual spread is ordinary numerical non-determinism in the fit — so deduplicate by taking
one file or the mean, and do not assert exact equality.

#### ✅ DECIDED 2026-08-26 — one kernel, and the Gaussian process enters the decomposition

*"The kernel is not why the Gaussian process was kept out of the variance decomposition. Commit to
the radial basis everywhere and it can go in alongside the support vector machine — which is what
those jobs were for."*

**Re-verified independently from the harvested files, 2026-08-26** — 76 files in
`results/gp_kernel_harvest/qm9/`, zero-noise rows only, deduplicated across the six noise-type
replays and paired on the replicate seed:

| Representation | Tanimoto | Radial basis | Paired difference | n |
|---|---|---|---|---|
| ECFP4 | 0.8225 | 0.8201 | **+0.0024** (sd 0.0039) | 10 |
| SMILES | 0.8062 | 0.8025 | **+0.0037** (sd 0.0015) | 10 |
| MHG-GNN | 0.8724 | −0.0158 | +0.8882 | 10 |
| mol2vec | 0.8677 | 0.0087 | +0.8590 | 10 |
| PDV | *(Tanimoto undefined)* | 0.8890 | — | 10 |

Where the features are binary the two kernels are indistinguishable.

> ⚠️ **CORRECTED 2026-08-26 (chat C).** This section read the radial basis collapsing on the two
> learned embeddings as the rescaling defect in §2.8c. **It was not.** Those two cells are failed
> fits: the kernel was never told how far apart counts as far, so on features whose distances run to
> a thousand it had nothing to learn from and returned a constant (§2.8f). Measured after the fix,
> the two kernels agree on the *damaged* features too — largest gap 0.040 across twelve paired
> measurements. The decision below is unaffected and is better supported than it was: the two kernels
> now agree everywhere, not only on binary features.

**What this means for the design:**

- Run the radial-basis Gaussian process on **every** representation in both studies. It has never
  been run on SNS at all.
- **Both entries come out of `ANOVA_MODELS_EXCLUDE`, and `PDV_ONLY_MODELS` goes with them**
  (§5.4). One model, one kernel, full coverage, in the decomposition beside the support vector
  machine.
- Keep Tanimoto on the two fingerprint representations. It is now *evidence* — the head-to-head
  that justifies committing to one kernel — not a second model in the roster.
- Remove the 2,000-molecule cap on the experimental side (§4, decision 3b) so it is the same model
  in both studies.

**The one hard dependency.** The embedding fix (§2.8c) must land *before* the re-run, not after.
Commit to the radial basis with the rescaling still in place and two of six representation cells
come back at essentially zero, which would drive the representation term of the very decomposition
this change exists to enable.

#### Harvesting them — what survives, and the four things that constrain the analysis

**What survives: the zero-noise rows, and nothing above them.** At zero the pipeline switches the
noise path off entirely before anything is drawn, so the held-out-label defect cannot fire and the
standardisation is computed from clean labels. Every row above zero carries both confounds and is
being regenerated anyway.

**The harvest works even for the job still running.** Results are appended a row at a time, and
the noise level is the outer loop with zero first. Any job that got past its first level already
has all ten of its clean replicates on disk.

Four constraints on what you can conclude:

1. **The six noise-type files replay the same run at zero noise.** The seed is fixed and the
   strategy is inert there, so per representation you have **ten seed-matched clean replicates,
   not sixty.** Treating the six files as independent would inflate the sample sixfold and breach
   the no-averaging rule. This doubles as a free integrity check — if the six files *disagree* at
   zero noise, that is the configuration race (§2.8a) showing itself, and it tells you which file
   to distrust.
2. **Decide the kernel on QM9, not on the experimental datasets.** There, the radial-basis model
   is capped at 2,000 training molecules and the Tanimoto one is not, so any gap between them is
   kernel *and* training-set size. Use the experimental run for coverage, not for the kernel call.
3. **The experimental jobs may have written to the other filesystem** — see §2.8b. The commands
   try `stat-ecr` first and fall back.
4. **Whether the descriptor representation was ever run is unsettled.** The submit script asserts
   it already existed; a memory note from June says the radial-basis model had no data anywhere.
   The file check settles it.

Commands: `research_archive/salvage_20260825/08_*.json`, steps 1 through 9 — check job state,
verify completeness, confirm the kernel label, read the logs, check the experimental side, copy
both kernels, then the paired analysis on the zero-noise rows.

### 10b.3 Patterns to re-test, not to re-derive

These came out of the August analysis. Every number behind them is being regenerated, so they are
hypotheses for the new data — but they are *specific* hypotheses, and finding them again is much
cheaper than finding them the first time.

- **Model family interacts with noise type.** The two harshest noise types preferentially killed
  boosting and the quantile forest, while kernel methods shrugged them off. Gaussian noise hit the
  networks and the support vector machine hardest.
- **Fragility depends on dataset size.** Boosting was fragile only on the small experimental sets
  and fine on large clean QM9, where the networks lagged instead. The one-line version: model
  choice matters far more on small real data.
- **Floor and ceiling profiles.** Random forest and LightGBM are high-floor safe defaults. The
  support vector machine is high-ceiling and low-floor — top only with the right representation.
- **The three-tier severity ordering** — two noise types that barely degrade anything, two mild,
  two that gut performance by destroying the high-magnitude tail where most of the signal lives.
  **Expect dose-matching to flatten this**, because it was largely a dose effect. If a tier
  structure survives matched dose, that is a real finding.
- **Clean accuracy did not predict accuracy under noise, and the direction reversed between
  datasets.** On the small experimental sets the neural models held up best while a top boosting
  model collapsed; on large clean QM9 it was the other way round. Re-test the reversal; do not
  quote the old cells.

### 10b.4 The units problem — and why Caco-2 always looked worst

Noise is injected in the label's own units, so a fixed level is **not** a fixed noise condition.
Back-computed label spreads: LogD 1.191, hERG 0.905, Caco-2 0.434, QM9 1.051.

At any given level, **Caco-2 receives about 2.7 times the noise-to-signal that LogD does.** That is
the mechanism behind Caco-2 supplying most of the negative-accuracy cells. It is partly an artefact
of the scale, not purely model fragility, and it needs one honest sentence in the Methods.

Two further facts from that work:

- **One noise type never separates models at any level** — flat from zero to maximum. Its column in
  any robustness table is approximately the clean ranking in disguise. That is a property of the
  noise type, not of the level chosen, and it needs a footnote wherever it appears.
- **LogD reaches no failure threshold at all.** Nearly nine in ten configurations still clear the
  accuracy gate at maximum noise. LogD is insensitive to the entire range tested. That is a
  reportable finding, not a defect to hide.

### 10b.5 The figure changes already specified

Worked out against the figure script after you settled the metric question. No new figures, no
metric removed.

| Change | Where |
|---|---|
| Add a clean-accuracy column to the model-by-noise-type heatmap, visually separated so it cannot be read as another noise type | the main overview figure's second panel |
| Draw a vertical line at the reporting noise level on the panel beside it — the sanity check is already plotted, just visually disconnected | same figure, first panel |
| Colour the retention-versus-baseline scatter by delivered accuracy. A point high on the retention axis but pale is a model that retains well and delivers little | the decoupling figure |
| **Decide the axis zoom.** It currently tightens the range so a small spread fills the panel, with the stated rationale that the flatness is the point. That flatness was the retired claim. A zoomed axis magnifying it now argues visually for something being withdrawn | same figure |
| Add a third panel: delivered accuracy against retention, one point per model. The bottom-right quadrant is the flattered one | same figure |
| Apply the same baseline treatment to the experimental-dataset figures | validation figures |

What does **not** change: the variance-decomposition figure stays on retention, because it asks
about robustness and adding baseline would change the question. The ranking-stability work is
unaffected.

---

### 10b.6 The other thousand lines — read, and deliberately not carried

`REVISION_GUIDE.md` is 1,772 lines. Roughly a thousand of them are the paper walked section by
section, as Replace/With pairs. **Nearly all of it is dead**, and for exactly the reason you gave:
it is built on a metric change and a set of numbers that the re-run supersedes. Half the pairs
exist only to swap one metric name for another; the rest paste in values that are being
recomputed. Carrying them would mean re-editing the same sentences twice.

Three things in that half are worth keeping, and nothing else is.

**The per-section discipline.** Every section closes with the same four-part ledger: what was
kept, what was removed, what was declined and why, then a numbers check and a citations check,
each traced to a file. That is the format the new guide should use at step 3. It is why the
citation audit exists at all.

**The citation audit itself**, which is how the broken bibliography surfaced (§9.1).

**One decision that becomes load-bearing under the decomposition build.** The guide explicitly
*declined* to delete the paragraph at `paper.tex:232` describing how the uncertainty components
are computed, on the grounds that it is mechanics rather than a finding. That was the right call
and it matters now: once the decomposition is built properly, that paragraph stops being
aspirational and becomes an accurate description of what the code does. It is the one piece of the
uncertainty Methods that does not need rewriting.

---

## 11. What was audited and dropped

Recorded so nothing is silently lost. Six documents were read in full and deleted; what follows
is where each one's live content went.

| Document | Disposition |
|---|---|
| `RESULTS_REWORK.md` | The blocking defect is §2.1 here; the verified paper defects are §9; the retention-metric problem is §7.3; the figure and table proposals are superseded by the re-run and will be redrawn from the new columns |
| `DISCUSSION_REWORK.md` | The retention mirage is §7.3; the two-different-failure-modes finding is in §5.4 item 7 and §9; the uncertainty findings are §7.2. Its recommendation to drop the conformal-prediction condition stands — `results/calibration_grid/` contains no intervals, no coverage and no widths, and nothing reads it |
| `DISCUSSION_TRACKER.md` | The metric decision — one reporting level plus retention plus baseline — is §7.3, translated into the new units. The active-job list became step 3. The eleven script-change decisions are superseded: the figure script is being rebuilt against different columns |
| `REVISION_STATUS.md` | The saturated experimental variance decomposition is §5.4 item 2. The cross-study uncertainty finding is §7.2. The rest was status bookkeeping for numbers that are being regenerated |
| `immediate_next_steps.md` | Its code-level findings are §2.4 through §2.8, its cost measurements are §5.3, and its uncertainty protocol is §3.1. Its noise-level decision is superseded by `NOISE_DESIGN.md`, which reaches the same conclusion with sourced evidence |
| `UNCERTAINTY_METRIC_FIX_PLAN.md` | Fully superseded. Its diagnosis — that the correlation pooled all noise levels and therefore measured the population trend, not per-sample tracking — was confirmed and acted on; the fix that was built goes further than the plan asked |

Two contradictions between those documents are resolved here rather than carried:

- **Which experimental results directory is canonical.** The tracker asserted both. `validation_rerun`
  is the one with thirteen models, four representations, all noise types, per-noise-level rows and
  folds preserved; the figure pipeline was pointed at the seven-model directory. The re-run makes
  this moot — but the analysis must be pointed at whatever the new runs write, once, explicitly.
- **Whether the variational last layer is "broken".** It was called a units artefact in one place
  and a real failure in another. Both are partly right and the resolution is §2.4: its coverage
  numbers are computed on a scale that moves with the noise level, so they were never comparable
  across levels. Fixing the standardisation order settles it, and it will need re-checking on the
  new data rather than arguing about the old.

---

### 11.1 🔴 OPEN — seventy-nine possible faults nobody has checked

The full audit of both pipelines on 2026-08-26 produced **151 candidate faults** across nine
areas — model settings, preprocessing, feature handling, noise, uncertainty, and what reaches
disk. Each non-cosmetic one was to be handed to a separate reviewer told to disprove it.

**The verification was capped at 40. It was my cap, and it means 111 candidates were never
examined at all — 79 of them non-cosmetic, and 8 of those rated at the HIGHEST severity.** They are
unchecked, not dismissed. All 151 are saved in `research_archive/audit_2026_08_26/`:
`unverified.json` (the 111), `confirmed_35.json`, `refuted_5.json`, `synthesis.md`.

**One of the eight unchecked top-severity candidates is another scrambling fault**, of the same
class as §2.10: the QM9 graph models are said to index the unshuffled dataset with indices computed
on the shuffled one, which would mean every graph model has been trained on molecules that do not
match their labels. That is unverified and it is the first thing to check.

Caveat on the count: candidates were matched by title, and a few titles differ only in wording
between a dimension report and the cross-pipeline pass, so a handful of the 111 may already be
covered by the 40 that were checked. Two are known duplicates of faults already fixed (the
network-setting one, and the early-stopping-label one). Of the 40 that were checked, 35 survived and
5 were disproved — but that survival rate should be treated with suspicion: every reviewer returned
"certain", with no gradation, and the surviving list contains obvious duplicates (three entries for
the same fingerprint fault, two for the same graph-model unpacking error). The genuine count is
smaller than 35 and the genuine total is unknown.

Three of the confirmed ones were re-checked by hand against the source and acted on: the scrambled
fingerprint (§2.10), the neural-network setting that matched no branch, and the feature-scaling
divergence (§3.4.3a). **The rest are unactioned.**

One artefact to know about when reading that audit's output: it ran against a tree that was being
changed underneath it, so at least one finding was marked "disproved" only because the fix had
landed while the reviewer was reading. The feature-scaling fault is real despite being marked
disproved.

To resume it, re-run the workflow with a higher cap. The candidate list and every reviewer verdict
are in the run's own journal file, so nothing has to be regenerated.

## 12. What is deliberately not in scope

- Rewording any research question to fit the data. The instruction stands: the paper is fixed by
  re-running.
- Editing `paper.tex`. §9 is a list for you, not a set of edits I will make.
- The conformal wrapper — **with one caveat I had wrong.** Its output directory holds no
  intervals, no coverage and no widths, so there is nothing to analyse today. But you asked that
  it be excluded from the main roster *and flagged if it turns out to be good at per-molecule
  noise tracking*. If the re-run produces usable intervals, it goes back on the table.
- The classification half of the framework. Untested in this study and unchanged by any of this.

## 13. THE PLAN — chat by chat

Written 2026-08-26. **Scope: code only.** Paper edits are parked by the author's instruction
(*"Let's hold off on the paper now and just focus on the code. We've diverged too far at this point
and I need to see the results."*). §9 stays in this document as a list for later; nothing in §13
touches `paper.tex`.

**Every item marked 🔴 TODO needs the author before the chat it sits in can finish.** A chat may
start with open TODOs; it may not produce a launchable script with one still open.

### 13.1 🟢 MOSTLY SETTLED — the run design: replicates, and what runs at full grid

**Settled:** the staged shape (2026-08-26), ten replicates in stage 1 (2026-08-26), and
which noise types run at full grid (2026-08-27, §13.9). **Items 2, 4, 5 and 6 were all
deferred on 2026-08-27 with a default recorded for each — see §13.10.** Nothing on this
list blocks anything.

**This is the largest open decision and it gates the job scripts.** Both parts are the author's.

**What is already true and constrains it.**

- The replicate seed is `(random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF`
  (`process_and_train.py:1871`). It depends **only on the replicate index**. Replicate 3 always draws
  the same molecules, the same scaffold split and the same noise realisation, whenever it is run.
- `--start-iteration` (`:239`) exists precisely to run replicates 5–9 later and append them to
  replicates 0–4. **Staging is already supported and is genuinely exchangeable, not approximately.**
- `MIN_CELL_ITERS = 5` (`generate_paper_figures_v2.py:437`) gates entry to the balanced
  decomposition. At exactly five replicates, one run lost to the catastrophic-run filter drops the
  whole cell.
- A paired signed-rank test on *n* replicates floors at p = 2/2ⁿ. Six replicates is the smallest
  number at which Q2 (§7.0) can return anything significant.

**The author's proposal, recorded verbatim:** *"Maybe I have representatives. Or I do an initial run
of 1 replicate (that I can reuse, also ensure that the replication is easy to handle) and then I can
decide what to look further at with the full set of noise strategies? Like run everything with
gaussian but select some models/reps to look at deeper with uncertainty and the full set of noise
strategies."*

This is directly supported by the seeding above. The sketch below is **a proposal for discussion,
not a decision**:

#### ✅ AGREED 2026-08-26 — the staged design

The author agreed the staged shape. **This table is the reference; it belongs in the paper's
Methods**, because a staged design has to be described as one.

| Stage | What runs | Replicates | Answers |
|---|---|---|---|
| **0 — screen** | every model × every representation, **the same noise types stage 1 runs**, full level grid | 1 | Choose what stage 2 goes deep on. **Reused as replicate 0 of stage 1, not thrown away** |
| **1 — breadth** | every model × every representation, a chosen subset of noise types, full level grid | **10** ✅ | Q1, Q3 |
| **2 — depth** | chosen models × chosen representations, **all** noise types | 10 | Q2 — the small effects, where the precision is actually needed |
| **3 — uncertainty** | the models that emit a per-molecule uncertainty, experimental datasets | 1 + a permutation null | Q4, Q5, Q6 |

**Why the replicates are not spread evenly.** Replicates buy precision, and precision only matters
where the effect is small. The model-versus-representation split is large — roughly 70% against 10%
— and six replicates resolves it. The difference between noise types at matched amount is small,
under 0.02 in accuracy from the pilots, which is close to the run-to-run wobble itself. That is
where the replicates have to go, and stage 2 is where they go.

**Cost, in training runs on QM9** (13 models × 6 representations, 7 levels):

| | runs | share of the old design |
|---|---|---|
| Old design (6 noise types, 11 levels, 10 replicates) | 51,480 | 100% |
| Stage 0 screen | 1,482 | 3% |
| *(that 1,482 is 78 × 19 × 1 — the same nineteen level-conditions as stage 1, which is what makes the reuse work. An earlier version of this row said "Gaussian and censoring", which is thirteen conditions and 1,014 runs; the pricing was right and the description was wrong. Chat G, §13.9)* | | |
| Stage 1 at **6** replicates | 8,892 | 17% |
| Stage 1 at **10** replicates | 14,820 | 29% |
| Stage 2 (10 replicates, 4 models × 3 representations) | 4,440 | 9% |
| **Total, stage 1 at 6** | **13,332** | **26%** |
| **Total, stage 1 at 10** | **19,260** | **37%** |

**Answering the author's question directly: ten replicates in stage 1 does not make things a lot
worse.** It adds 5,928 runs, which is 67% more than stage 1 at six, but the whole staged design
still lands at 37% of the old one rather than 26%. Both are far cheaper than what was planned
before. Ten is the safer choice: it keeps headroom above the five-replicate gate, and it drops the
floor on the paired test from 0.031 to 0.002.

**✅ Settled 2026-08-26: ten replicates in stage 1.** **🔴 Still open:** which noise types go into stage 1's full grid; and which
models and representations go deep in stage 2, which cannot be chosen until stage 0 has run.

**Open, and each needs an answer:**

1. ✅ **SETTLED 2026-08-26 — ten replicates in stage 1.** Kept here only because the arithmetic
   is worth not re-deriving: six is the floor if Q2 is to be answerable and the five-replicate gate
   is to have headroom; ten costs 67% more than six and buys precision, not new answers. Author's
   note: *"I believe I was doing 10 and holding off replicates for uncertainty."* **This item read
   🔴 open for a day after it was settled three lines above, and that contradiction cost a session:
   chat D twice told the author the replicate count was the thing blocking launch.** It is not.
2. ✅ **SETTLED 2026-08-27 by the author — one replicate for the uncertainty runs, plus a
   permutation null.** *"Yes one replicate for uncertainty runs."* Do not reopen it.
   **What this commits the paper to saying**, because it is no longer optional: the uncertainty
   results have **no run-to-run error bar**. The five scaffold folds are a partition, not repeats,
   so their spread mixes randomness with scaffold difficulty and cannot stand in for one (§3.3).
   The permutation null is what tells the reader whether an observed correlation is distinguishable
   from chance; it is not a substitute for a repeat, and the Methods must say so in those words.
   **Where it is enforced:** the runner has no replicate axis at all
   (`slurm_scripts_uncertainty_rerun/generate_scripts.py:77-80`), and the null is built and tested
   at `scripts/uncertainty_stats.py:760` (`permutation_null`), which permutes the noise within one
   cell and fold and returns the observed value against a 2.5–97.5 band. Recorded in the audit
   script's manual checklist so it cannot be forgotten at writing time. **Original reasoning, kept
   because it is the answer to "why is one enough":** the uncertainty statistics are correlations
   over thousands of molecules, so their precision comes from the molecule count, not the replicate
   count. The uncertainty statistics are
   correlations over thousands of molecules, so their precision comes from the molecule count, not
   the replicate count — one replicate is defensible **provided** a permutation null is reported so
   the reader has a reference distribution. Without repeats there is no run-to-run error bar at all.
3. ✅ **SETTLED 2026-08-27 — which noise types run at full grid in stage 1 (§13.9).**
   Q1 asks for a decomposition *per noise type*, so every type that needs its own decomposition must
   run at full grid. The structurally distinct ones are **four, not three**: Gaussian (even),
   Grouped — wider (structure-keyed, centred), Grouped — shifted (structure-keyed, one-directional)
   and Censoring (one-directional across the whole dataset). The two grouped conditions differ only
   in whether the group's error is centred, and that single difference is worth 0.10–0.31 R² — the
   largest zero-mean effect measured anywhere in this study — so they cannot share a cell. The
   heavy-tailed and sparse-contamination types were what stage 2 was expected to show behave like
   Gaussian, and they do: every setting within 0.006 R² at the reporting level. **Stage 1 at four
   types is 25 level-conditions, 19,500 runs, and the staged total becomes 48% of the old design.**

   ✅ **The 25 is now what actually runs, 2026-08-27 (chat M).** The clean level is trained ONCE, under
   Gaussian, and copied into the other three conditions afterwards by
   `slurm_scripts_qm9_rerun/copy_zero_rows.py`. At level 0 the pipeline does not add noise at all and
   the replicate seed depends only on the replicate number, so the clean run is bit-identical
   whichever condition it is labelled with — measured on 400 QM9 molecules, random forest on ECFP4,
   all four conditions returning R² = 0.7579128047581825 and RMSE = 0.5176004014184159 to the last
   digit. It cannot simply be left out, because `auc_norm` divides each condition's curve by that
   condition's own clean accuracy, so a condition with no clean row produces nothing at all.

   The copy refuses to overwrite a clean row a job actually computed: it checks that row against the
   reference instead and stops the whole copy if any disagrees, which turns every resubmitted clean
   task into a free four-way agreement test on production data. Proven both ways — three rows copied
   into empty files, then a corrupted row detected and the copy refused, exit 1.
4. ✅ **RULED 2026-08-27 by the author — it is chosen from the screen's results, and that is
   deliberate.** *"That depends on the results from the first runs. That needs to be clearly
   documented."* A suggestion that it be pre-registered instead was put and is **withdrawn**.
   **What this means in practice.** The sequence is: run the screen, look at it, choose, then
   generate the deep run. The generator already refuses `--stage 2` without `--models` and `--reps`
   rather than inventing a default (`slurm_scripts_qm9_rerun/generate_scripts.py:467-469`), so
   nothing can run ahead of the choice.
   **What the paper has to say, because the choice is made on the outcome.** A referee will ask why
   those models and not others. The answer must be in the Methods, in these terms: the screen ran
   every model and every representation at one replicate; the deep run added the remaining noise
   conditions on a subset; here is the rule by which the subset was chosen. **Fix that rule before
   the screen is read, not after** — otherwise the selection and the justification are written from
   the same look at the data. Proposed rule, for the author to accept or replace *before* the screen
   lands: take the widest spread of behaviour the screen shows — the most and least noise-tolerant
   model, plus one from each remaining family — and the representations that span fingerprint,
   descriptor and learned embedding. **Which models and representations go deep in stage 2?** **Yours, and it cannot be asked yet** —
   it is chosen from what stage 0 shows, so the sequence is: chat H runs stage 0, brings you the
   screen, you choose. The generator already refuses `--stage 2` without `--models` and `--reps`
   rather than inventing a default, so nothing can run ahead of the decision.
5. ⏸️ **DEFERRED 2026-08-27 until the main grid has run — at which point it stops being a guess.** Only 0.5 and 1.5 have ever been measured (`results/setting_selection_test.csv`, levels column holds exactly those two). The main grid sweeps all seven levels, so after it runs the level can be picked from data rather than argued. Nothing between now and then needs it: no code contains a reporting level, and the figure script has not been rebuilt for the new grid anyway (chat J). **If a level must be named before then, use 1.5** — it is the only one measured where the study's largest zero-mean result is visible. **The QM9 reporting level** (§6.1). **Yours, and it can be asked now** — nothing is blocked on
   it, but every table that reports accuracy at one level needs it.

   ⚠️ **The standing suggestion of 0.5 is withdrawn, 2026-08-27.** Chat D re-read the same twelve
   replicates by COUNTING them rather than averaging, and 0.5 and 1.5 do not agree. Number of the
   twelve replicates in which grouped-shifted is worse than Gaussian by more than 0.05 R²:

   | | LightGBM | Random forest | Ridge |
   |---|---|---|---|
   | level 0.5 | 1 | 1 | 2 |
   | level 1.5 | 8 | 8 | 11 |

   **Reporting at 0.5 would hide the study's largest zero-mean result.** What the two levels DO agree
   on is that the heavy-tailed and outlier conditions are indistinguishable from Gaussian, which is
   what the condition set rested on, so §13.9's verdict stands unchanged.

   **Recommendation: 1.0 or 1.5**, both real points on the grid. 1.5 is where the effect is clearest;
   1.0 is the easier one to defend to a referee, being one unit of label spread. One caveat carried
   from chat D: Ridge replicate 2 is a broken run, R² between −868 and −0.03 in every condition, and
   it is excluded and named rather than averaged in.
6. ✅ **SETTLED 2026-08-27 by the author — inherit the main grid's four, and add `outlier_p10`.**
   *"Yes add it."* Do not reopen it.
   **Why only that one.** The uncertainty runs ask whether a model's uncertainty rises on the
   molecules whose labels were corrupted. That has an answer only where the corruption hits some
   molecules harder than others. Four of the seven conditions spread it evenly over every molecule
   — gaussian, laplace, student_t, grouped_shifted — so the question is undefined there, not
   negative. Three do not: `grouped_wider`, `censoring` and `outlier_p10`. The first two are already
   in the inherited four; `outlier_p10` was the only one missing, and it is the most concentrated
   case, so it is where a difference is likeliest to show.
   **Cost: +20% on the uncertainty runs** (one condition on top of four).
   **Already in the code**: `slurm_scripts_uncertainty_rerun/generate_scripts.py:143`
   (`ADDED_FOR_QUESTION_B = ['outlier_p10']`) makes it the default, with no flag to turn it off —
   the author confirmed it on 2026-08-27. `--include-deep-conditions` adds all three depth-only
   conditions, two of which are flat by design and buy nothing here. **Yours to decide,
   and chat H is where it gets asked** — it queues the uncertainty runs, so it is the last point at
   which the question can be put before compute is spent. Chat F closed on 2026-08-27 and everything
   it found is in the tree, so nothing is waiting on it. The set was settled on chat G's
   measurement of **accuracy on QM9** — one representation, three tree and linear models (§13.9).
   The uncertainty question is a different one: `NOISE_DESIGN.md` §5.3 notes that a model can lose
   the same accuracy while being much better or worse at spotting *which* labels were corrupted, and
   concentrated noise — the heavy tails and sparse contamination this screen found redundant — is
   exactly where that was expected to show. So "shape does not matter" is established for accuracy
   and is **not** established for corruption detection.
   **The options and what they cost:** inherit the four full-grid conditions (free, and the honest
   default) — or add the two single-setting conditions to the uncertainty grid to test it, which is
   two conditions × 3 datasets × 7 models × 4 representations × 6 levels × 5 folds. Recommend
   inheriting, and saying so in the Methods rather than implying it was tested. Raised by chat G
   2026-08-27; the scope of the uncertainty runs is already §4 Decision 1.

**One caution to hold on to.** A staged design is only honest if the reduced set in stage 1 is
justified by what stage 0 and stage 2 show, and the paper says so. "We ran everything under Gaussian
and a subset under the rest" is defensible; presenting it as a full factorial is not.

### 13.10 ⏸️ CLOSED 2026-08-27 — chats D, E and G, and the four items they handed back

**Why this section exists.** Three chats ended by handing the author a list of open decisions.
The lists were nearly the same list. Read one after another they looked like nine or ten
outstanding questions; they are four, and none of them blocks anything. The author asked for them
to be deferred. They are, each with a default recorded so that no one has to come back and ask.

**The vocabulary, once, because it is where the confusion came from.** This document says
"stage 0/1/2/3". In conversation those are **the screen** (one replicate, everything, choose what
to look at closely), **the main grid** (ten replicates, four noise conditions, every model and
representation), **the deep run** (all noise conditions, a chosen few models and representations),
and **the uncertainty runs** (the three experimental datasets, the models that emit a per-molecule
uncertainty).

#### The four items, and where each one actually came from

| # | The question | Raised in | Restated in | Blocks |
|---|---|---|---|---|
| 1 | The QM9 reporting level | G (suggested 0.5) | D (withdrew 0.5, measured 1.5), E | nothing |
| 2 | Do the uncertainty runs inherit the settled noise conditions, or test them? | G | D | nothing |
| 3 | Is one replicate right for the uncertainty runs? | earlier, §13.1 item 2 | D | nothing |
| 4 | Which models and representations go deep | earlier, §13.1 item 4 | D, E, G | nothing, and it is not askable yet |
| ~~5~~ | ~~**How many noise conditions the three experimental datasets run**~~ | D, 2026-08-27 | — | ✅ **SETTLED by the author 2026-08-27** — match QM9: three conditions at full breadth, censoring on a named pair subset |

**Item 5, raised and settled 2026-08-27.** Those jobs never stated their conditions, so they
inherited the runner's own list of seven — one of which, `outlier_p05`, the author had already
retired. Naming them raised the real question: censoring runs on **about five
model-and-representation pairs** on QM9, not the full grid, and nothing said what it should do on
the experimental datasets.

✅ **The author's ruling: match QM9.** So the experimental robustness runs are **gaussian,
grouped_wider and grouped_shifted at full breadth**, plus **censoring on a named pair subset**.
`noise_conditions.json` carries the widened scope — `applies_to: ["qm9_grid",
"validation_robustness"]` — and the validation generator now refuses to put censoring in a script
unless the models and representations are named, exactly as the QM9 generator does. Which pairs
comes from the screen (§13.13).

⚠️ **The uncertainty runs are the exception and keep censoring at full breadth**, recorded in the
same entry as `full_breadth_in: ["uncertainty_runs"]`. It is one of only two conditions there that
can answer which molecules were damaged, so restricting it would remove the question rather than
cheapen it. `scripts/test_noise_conditions.py` asserts all three behaviours.

**Item 1 was one question that changed answer, not two questions.** Chat G proposed 0.5 while
measuring only accuracy differences between noise conditions, where 0.5 and 1.5 agree. Chat D
re-read the same twelve replicates by counting them and found the two levels disagree about
grouped-shifted. That is a correction to G, not a second open item. **The 0.48 that came up
alongside it is unrelated** — it is the published repeat error for logD, the evidence for where the
logD sweep's points sit, and it is not a reporting level anywhere in the code.

#### What the code says about whether any of this blocks

Checked 2026-08-27, by reading the files rather than the notes:

- **No reporting level exists in any script.** `scripts/generate_paper_figures_v2.py` still carries
  the old eleven-point 0–1.0 grid (`EXPECTED_SIGMAS`, `:885`, `:980`) and has no reporting-level
  constant at all. It has to be rebuilt for the new seven-point grid regardless (chat J), and that
  rebuild is where a reporting level would be set.
- **The QM9 job generator does not have the concept.** `slurm_scripts_qm9_rerun/generate_scripts.py`
  reads `noise_conditions.json` and sweeps the QM9 level grid from `NOISE_DESIGN.md` §6.4 (`:101`).
  Every level runs. Choosing one to report changes nothing about what is queued.
- ✅ **The uncertainty job generator was stale; it was rewritten on 2026-08-27 (§2.8j).** It
  listed the six deleted strategies and emitted flags the runner no longer has, so every task
  would have died at argument parsing. It now reads `noise_conditions.json`. **Items 2 and 3 were
  settled by their recorded defaults in the rewrite**: the main grid's four conditions plus
  `outlier_p10` (item 2), and one replicate with the permutation null computed afterwards by
  `scripts/uncertainty_stats.py` (item 3). Either can be changed with a flag; neither is now
  blocking anything.
- **The deep run cannot be built without item 4 and refuses to try** —
  `generate_scripts.py:467-469` errors on `--stage 2` without `--models` and `--reps`.

#### The recorded defaults

| # | Default, in force unless the author says otherwise | Owner |
|---|---|---|
| 1 | ✅ **SETTLED by the author 2026-08-27, per dataset: QM9 1.0, logD 1.0, Caco-2 0.2.** All three on the settled scale (fraction of the clean training label spread, §2.12); `0.25` is not on the ladder. See §13.11 for the one consequence on Caco-2 | chat J |
| 2 | **Inherit the four, and add `outlier_p10`** — the only one of the three depth-only conditions that is not flat by design, so the only one that can answer the question. +25% on the uncertainty runs | ✅ **CONFIRMED by the author 2026-08-27** and built, §2.8j. Not optional: there is no flag to run fewer than the five |
| 3 | **One replicate, plus a permutation null.** Without the null there is no reference distribution and no error bar of any kind | ✅ built 2026-08-27, §2.8j. The runner has no replicate axis; the null is `permutation_null` in `scripts/uncertainty_stats.py` |
| 4 | ✅ **RULED by the author 2026-08-27** — chosen from the screen's results, and documented as such. The open piece is the *rule* for choosing, which should be fixed before the screen is read (§13.1 item 4) | before the screen lands |

#### Item 1 — the level changes which model wins, so it is not a presentational choice

⚠️ **Corrected 2026-08-27. Ridge is not in the study.** `scripts/setting_selection_test.py:75`
screens `['LGBM', 'RF', 'Ridge']`, and its own docstring at `:384` says why: *"Ridge has no entry
there; it is the cheap linear reference the pilots used, kept at alpha = 1.0 for continuity with
them."* It is a scratch model in the local screening harness and it is **not** in the thirteen-model
roster (`slurm_scripts_qm9_rerun/generate_scripts.py:205-219`). No conclusion may rest on it. The
first version of this section led with a boosting-versus-linear story built on it; that is withdrawn.
LightGBM and the random forest **are** in the roster, and both take their settings from
`models/model_defaults.py`, the file both pipelines read.

Recomputed on those two only, from `results/setting_selection_test.csv` — QM9, PDV, plain Gaussian,
twelve replicates, counted per replicate, nothing averaged:

| Level | LightGBM ahead | Random forest ahead |
|---|---|---|
| clean | **11 of 12** | 1 of 12 |
| 0.5 | 8 of 12 | 4 of 12 |
| 1.5 | 1 of 12 | **11 of 12** |

**PDV. Median R² over the twelve replicates, with the range beside it:**

| Model | clean | 0.5 | 1.5 |
|---|---|---|---|
| LightGBM | 0.902 `[0.882–0.925]` | 0.868 `[0.842–0.895]` | 0.622 `[0.550–0.736]` |
| Random forest | 0.891 `[0.862–0.920]` | 0.865 `[0.836–0.888]` | 0.682 `[0.639–0.731]` |

The order reverses completely between the clean labels and level 1.5, and both models are still
fitting there. The boosting model is more accurate on clean labels and loses more; the forest is
marginally less accurate on clean labels and loses less. **Level 0.5 does not show it** — both sit
within 0.03 of their clean score and the clean order still holds in 8 of 12.

Grouped-shifted, counted the same way (replicates in which it is worse than Gaussian by more than
0.05 R²), roster models only:

| | LightGBM | Random forest |
|---|---|---|
| level 0.5 | 1 / 12 | 1 / 12 |
| level 1.5 | 8 / 12 | 8 / 12 |

**The file contains only levels 0.5 and 1.5.** 1.0 has never been measured on QM9, so the standing
recommendation of "1.0 or 1.5" was half a recommendation for an unmeasured point.

**Independent corroboration, on roster models and real data.**
`results/validation_full/openadmet_logd/all_results.csv`, plain Gaussian, seventeen
model-and-representation pairs across five roster models (DNN, GP, QRF, RF, XGBoost) and four
representations. Rank agreement with the clean ranking: **+0.94 at level 0.5, +0.56 at level 1.0**.
Individual pairs move a long way — the neural network on ECFP4 goes 3rd → 8th → 12th, the forest on
the PDV goes 10th → 7th → 2nd. One fit per cell, no replicates, so this is corroboration rather than
a measurement — but no part of it involves ridge.

**⚠️ That recommendation was made on QM9 alone and is WITHDRAWN — 2026-08-27. Report at 1.0.**

The one-shared-grid decision (§2.12) means one reporting level now has to work on all four datasets,
not just QM9. Checked on the experimental data, plain Gaussian, converting each dataset's raw log-unit
grid onto the shared scale by its label spread:

**Caco-2** (`results/validation_full/openadmet_caco2/`, ECFP4, one fit per cell, label spread ≈0.44
so its 0.2 / 0.4 / 0.7 log-unit points are shared levels 0.5 / 1.0 / 1.5):

| Model | clean | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| DNN | 0.565 | 0.462 | 0.375 | 0.303 |
| Random forest | 0.481 | 0.442 | 0.363 | 0.111 |
| XGBoost | 0.494 | 0.453 | 0.326 | **−0.129** |
| Quantile forest | 0.469 | 0.395 | 0.256 | **−0.142** |

**At level 1.5 two of the four models on Caco-2 are below zero — worse than predicting the mean.**
A model that scores below zero carries no information, and a table of such numbers cannot be read.
QM9 survives 1.5 because its clean R² is ~0.90 and its label spread is wide; Caco-2 has the narrowest
spread in the study and it does not.

**At level 1.0 every model on every dataset is alive**, and QM9 still shows the result: at 1.0 the
best model per replicate is MLP 4 of 9, DNN 2, forest 2 — boosting does not win, which is the whole
contrast. The reversal happens between 0.5 and 1.0, not at 1.5, so 1.0 is where it first shows rather
than where it is largest.

**What 1.0 costs, stated rather than hidden.** Grouped-shifted at level 1.0 is 9 of 9 replicates for
both neural models and 1 of 9 for the random forest and XGBoost. At 1.5 it is 7–9 of 9 for all six.
So at 1.0 the grouped-noise result is model-dependent. **That is a finding, not a weakness**: neural
models are far more damaged by systematic family-level bias than trees are, and it is visible from
level 1.0 upward. It has to be written that way rather than as a single pooled claim.

**And 1.0 is the citable one.** `NOISE_DESIGN.md` §4c: twice the published assay error — the rule the
experimental grids were already built on — is 1.21 on hERG and 1.58 on Caco-2. Level 1.0 sits just
below the hERG figure. Level 1.5 is only reachable via Caco-2's inter-laboratory upper bound, on a
dataset where 1.5 kills half the models.

**Caveats on the experimental evidence above, none of which change the direction:** one fit per cell
and no replicates (the experimental side is pinned to seed 42, §4 decision 3b); Caco-2 has only ECFP4
and four models, 4 of 20 cells; and these runs used the retired noise strategies, of which `legacy`
is the plain Gaussian one that carries over. logD is not shown because its old grid reached only 0.84
of its label spread — under the shared grid it will reach 1.5, and this constraint disappears at the
re-run.

#### Item 2 — ✅ settled: inherit the four, add one more

The uncertainty runs ask one thing: **when a molecule's label has been corrupted, does the model say
it is more unsure about that molecule?**

That only has an answer where the corruption hits some molecules harder than others. Where every
molecule gets the same amount, there is no "which molecules" to find — the question is undefined,
not answered in the negative.

| Noise condition | Hits some molecules harder? | In the uncertainty runs? |
|---|---|---|
| gaussian | no — even across molecules | yes (it answers the other question) |
| grouped_shifted | no — even across molecules | yes |
| laplace, student_t_nu5 | no — even across molecules | no |
| **grouped_wider** | **yes** | yes |
| **censoring** | **yes** | yes |
| **outlier_p10** | **yes** | **yes — the author added it, 2026-08-27** |

The four inherited from the main grid already contained two of the three that can answer the
question. `outlier_p10` was the third, and it is the most concentrated case — a few molecules thrown
a long way off — so it is where a difference is likeliest to show. **Cost: +20%.** Adding the other
two depth-only conditions would have cost 60% more and bought nothing for this question, because
both are even across molecules.

Where it lives: `slurm_scripts_uncertainty_rerun/generate_scripts.py:143`, as the default. The
author confirmed the five on 2026-08-27 and there is no flag to run fewer.

#### Item 3 — ✅ settled: one replicate plus the permutation null

The author's call, 2026-08-27. §13.1 item 2 carries what the Methods must therefore say: the
uncertainty results have **no run-to-run error bar**, and the permutation null is a test against
chance rather than a substitute for a repeat.

#### Item 4 — ✅ ruled: chosen from the screen, and the rule fixed before the screen is read

The author's call, 2026-08-27: which models and representations go deep **depends on the screen's
results**, and the document must say so plainly. A proposal to pre-register the set instead was put
and is withdrawn.

Sequence: run the screen → look at it → choose → generate the deep run. The generator refuses to
build the deep run without explicit choices, so nothing can run ahead of the decision.

**The one thing that should not wait for the screen is the rule.** Choosing the subset from the
results and then justifying the choice from the same results is one look at the data doing two jobs,
and it is the question a referee asks. Fixing the rule beforehand costs nothing and answers it.
§13.1 item 4 carries a proposed rule for the author to accept or replace.

---

### 13.11 ✅ SETTLED 2026-08-27 — the reporting levels, per dataset

**QM9 1.0, logD 1.0, Caco-2 0.2.** The author's call. All three read on the settled scale — a
fraction of that fold's clean training label spread (§2.12). `0.25` was also considered and is **not
on the ladder** (`NOISE_DESIGN.md` §6.4), so it cannot be reported.

**Why they are not the same number.** The experimental labels already carry measurement error and
QM9's do not, so one nominal level is not one amount of noise (`NOISE_DESIGN.md` §4d). The caption
must state each dataset's level and say why they differ.

#### One consequence on Caco-2, recorded rather than argued

The Caco-2 pick was made while reading a table printed in **raw log units** — the units the old
results files use, not the settled scale. That was an error on the reporting side and the tables
were regenerated. On the settled scale, the old Caco-2 run maps like this:

| Ladder level | Old run's point | R², four models on ECFP4 |
|---|---|---|
| 0.2 | 0.1 log units | 0.458 – 0.539 |
| 0.5 | 0.2 log units | 0.395 – 0.462 |
| 1.0 | 0.4 log units | 0.256 – 0.375 |
| 1.5 | 0.7 log units | −0.142 – 0.303 |

Clean is 0.469 – 0.565. **At level 0.2 the models sit inside their clean range**, so a Caco-2 table
reported there shows no noise effect at all. The point whose R² values were on screen when 0.2 was
chosen is called **0.5** on the settled scale. Both are legitimate choices — 0.2 says "even a little
noise costs nothing here", 0.5 says "here is where it starts to bite" — but they are different
claims and the level was picked against the second table's numbers.

⚠️ The ladder's 0.2 and 0.3 columns collapse onto the same old point (0.1 log units) because the old
grid was coarser. That is a limit of the old data, not a result, and it disappears at the re-run.

#### logD has no data at its chosen level yet

The old logD run reached only 0.84 of the label spread, so levels 1.0 and 1.5 were never run there.
This is not a gap in the design — the re-run sweeps the full ladder on all four datasets. It means
only that the logD choice cannot be checked against existing numbers before the re-run.

#### 🔴 The validation jobs were running a retired noise condition

**Found 2026-08-27 on a review pass over the regenerated scripts, not by any check.** The 87
validation job scripts passed **no `--conditions` at all**, so every one of them inherited the
runner's own `NOISE_CONDITIONS` literal (`alternative_data_noise_robustness.py:168):

```
gaussian, student_t_nu5, laplace, grouped_wider, grouped_shifted, outlier_p05, censoring
```

`outlier_p05` is listed under `not_run` in `noise_conditions.json` — retired on 2026-08-27 in
favour of `outlier_p10`, on the evidence that every step of the 1% → 5% → 10% ladder is under
0.0049 R². So the whole experimental-dataset robustness family would have run a setting the study
had dropped, and not run the one it kept, **silently**: nothing in a result file makes anyone read
a condition name.

It is the drift `noise_conditions.json` exists to prevent, and its own comment names the rule —
*"READ BY TESTS ON BOTH SIDES"*. The runner restates the set as a Python literal instead, and
`scripts/test_noise_conditions.py` checks that condition names resolve in the injector, not that
the runner's default matches the settled file. Nothing connected the two.

**Fixed.** The generator now reads `noise_conditions.json` and states the conditions on the
command line. Default: the full grid's four (`gaussian`, `grouped_wider`, `grouped_shifted`,
`censoring`), which is what §6.3 specifies for the experimental datasets;
`--include-depth-conditions` adds the depth three. It asserts no retired name can reach a script.

⚠️ **This changes what the validation family runs** — from an implicit seven, one of them retired,
to an explicit four. Adding the depth three back is one flag. The author should confirm the four
is what is wanted before launch.

**The runner's literal is still wrong and is in the other repository.** Anyone invoking
`alternative_data_noise_robustness.py` by hand still gets `outlier_p05`. The live job families no
longer can: QM9 has its own command line, the uncertainty family has stated its conditions since
2026-08-27, and the validation family now does too.

#### 🔴 The validation jobs had no injector-version guard

Same review pass. The runner does `from noiseInject import CONDITIONS` at module scope, so a stale
checkout does not fail — it runs the pre-1.0.0 scheme, where a level meant something else, and
writes results that look exactly like the new ones. The uncertainty jobs have refused a stale
injector by name since 2026-08-27. The validation jobs, which use the **same runner and the same
injector**, did not. Fixed: the same check, generated into all 88 scripts.

#### 🔴 The merge step discarded three quarters of the validation results

**Found on the same review pass, and it is the worst of the three because it destroys data that
the jobs correctly produced.** `slurm_scripts_validation_rerun/merge_results.py` deduplicated on
`['model', 'rep', 'strategy', 'sigma', 'fold']`, filtered to the columns present. The runner
renamed that column to `noise_type`; the filter **drops a missing name silently rather than
raising**, so the key became `(model, rep, sigma, fold)` and every noise condition on a cell
deduplicated against every other, `keep='last'`.

Measured, on a frame built from the runner's real columns: **four conditions in, one row out** —
only `censoring` survived. Every job would have run correctly, written correctly, and then had
three quarters of its output thrown away at merge time with nothing printed.

The overlap mask had the same shape: it dropped every existing row matching a re-run
`(model, rep)`, so conditions the re-run did not produce were discarded along with the ones it
replaced. That matters more now the default is four of seven.

**Fixed.** The condition column is resolved once, `noise_type` or `strategy`, and **asserted to
exist** rather than silently skipped; the overlap mask matches on the condition too; the merge
prints the key it used, how many rows it dropped and which conditions it kept. Old files written
with `strategy` still merge. Guarded by `the_merge_keeps_every_condition` in the new test, which
fails at "collapsed 4 conditions to 1" when the fix is removed.

#### The validation family had no job-script test at all

Which is why both of the above, and the `--datasets herg` defect, survived. QM9 has
`scripts/test_generated_job_flags.py`; the uncertainty family has
`scripts/test_uncertainty_job_scripts.py`. **`scripts/test_validation_job_scripts.py`** is the
missing third: it generates real scripts into a temporary directory and puts every command line
through the runner's own parser, then checks the conditions are stated and settled, that hERG's
two names are both right, that all three guards are present, and that every model name is one the
environment probe knows. Eight checks. Each was confirmed to FAIL with its fix removed.

The generator also gained `--out-dir`, without which the only way to see what it emits is to run
it — which overwrites the committed scripts, and is how all 87 were silently rewritten on
2026-08-27.

#### hERG was never cut

It is in both live generators — `slurm_scripts_uncertainty_rerun/generate_scripts.py:120` and
`slurm_scripts_validation_rerun/generate_scripts.py:34`. There are simply no old noise-sweep results
on disk for it, so it has no table here.

⚠️ **hERG is spelled two different ways and both spellings are load-bearing.** On the command line
it is `herg_ki` — `alternative_data_noise_robustness.py`'s `--datasets` carries
`choices=['logd', 'caco2', 'herg_ki', 'all']`, so `herg` is rejected by argparse and the task dies
before it loads anything. In every path it is `herg` — the runner writes to
`Path(results_root) / 'herg'`, and `merge_results.py` matches result directories by the
`_{dataset}` suffix. Collapsing them to one name breaks one end or the other. The validation
generator now carries both as a two-column table; before 2026-08-27 it emitted `herg` on the
command line and 28 of its scripts would have died at argument parsing (§13.2, chat D). Its reporting level is still to be set, and unlike the other
three it has a **measured** label spread to set it against: 0.9143 over all 1,415 molecules
(`NOISE_DESIGN.md` §4c).

---

### 13.12 🔴 HANDOFF — what the close-out audit found, for whoever picks up D and G

27 agents ran every gate both chats claim and traced every file, line and commit either one cites.
Each finding was then given to a separate agent told to refute it; only what survived is here.
**Two are already fixed (`7652694`) and struck through. The rest are open.**

**What was proved working, so nobody re-audits it:** the per-task configuration file with no default
path; all-or-nothing record writing; the reader that errors instead of guessing; `morgan` gone and
refused by name; the injector's exit code stopping the run; all five writer guards; all 28 noise
gates over four consecutive runs; both raised seed counts (20 → 200, one draw → 50); the settled
condition set read and enforced by five independent readers; and the singleton-scaffold rule on the
split in both the pipeline and the harness.

#### For chat D

| # | Kind | What | Fix |
|---|---|---|---|
| ~~D1~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — Avalon returned an all-zero fingerprint for an unparseable molecule and for any exception. Nothing caught it downstream | Raises now, plus the all-zero case RDKit produces from `''`. Guard: `scripts/test_avalon_failure.py` |
| ~~D2~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — all three regenerated, and the smoke test is now emitted by the generator instead of kept by hand. Regenerating also uncovered a live defect the hand edits were masking: the generator passed `--datasets herg`, which the runner rejects at argument parsing, so all 28 other hERG scripts would have died there. The CLI name (`herg_ki`) and the path name (`herg`) are now a two-column table in the generator | ✅ Done — §13.2 chat D |
| ~~D3~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — both now call it. They could not use `--models`: KIRBy says `LightGBM` and `BNN-Full` where QM9 says `lgb` and `dnn_bnn_full`, so a QM9 label list would have failed them for the guard's own reason. `check_environment.py` gained `VALIDATION_MODELS` and `--validation-models`, sourced from the runner's own optional-import blocks, and `--audit-roster` now checks all three generators' labels | ✅ Done |
| ~~D4~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — retargeted to `write_data`: it now states the all-or-nothing record property, says the fingerprint is computed in Python and carried through as a fixed `[u8; 256]`, and says what `write_data` itself checks. A mangled failure message on the same path (a broken line continuation leaving 34 spaces mid-sentence) was fixed with it | ✅ Done |
| ~~D5~~ | ~~code~~ | ✅ **RESOLVED 2026-08-27, and the suggested fix was not possible.** The check is **unreachable**: every representation `parse_mmap` accepts reads a FIXED number of bytes, so its rows are all the same width by construction, and a truncated block is swallowed by the fixed-size read and surfaces as the next field failing — the per-entry guard, tested already. The two representations whose rows could genuinely differ (one-hot and randomized SMILES) are refused by name before the loop. Kept as a backstop; the alignment test now asserts the assumption it rests on, per representation, so it fails loudly the day a variable-width representation is added | ✅ Done — §8 gate 8 |
| ~~D6~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — §2.7 now opens with a warning that items 1 and 2 are history, each says where the protection lives now (`ecfp4_fingerprint` in Python, the fixed `[u8; 256]` carried through, `write_data`'s all-zero check), and item 3 names `read_split_labels` | ✅ Done |
| ~~D7~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — rewritten: five writer tests and five reader checks, what each actually plants, and why the unparseable-molecule route no longer exists | ✅ Done |
| ~~D8~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — all four. The mask is 63 bits and the document now says why the top bit is cleared; `setup.sh:124` corrected in the document and in all three generators and the preflight (it was in four files, not one); the two-interpreter sentence replaced with one interpreter; the stale-scripts warning replaced with what is actually there | ✅ Done |
| ~~D9~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — corrected, `smoke_test.sh` disclosed, and the consequence traced through to the defect it was hiding (D2) | ✅ Done |
| **D10** | cluster | The substantive half of gate 10 skips on this laptop, so §8's claim that all three halves passed cannot be re-confirmed off the cluster | ▶️ **Owned by chat H, and now step 4 of the QM9 runbook** — `python scripts/test_config_isolation.py --end-to-end` under `env_test`, before the first submission |
| **D11** | cluster | ▶️ **Owned by chat L, the environment rebuild.** `check_environment.py` exits 1 on the laptop interpreter — requirement conflicts, a quantile-forest fit failure, four OpenMP runtimes reachable. **The gate is behaving correctly**; this is the threading conflict and the environment rebuild, explicitly not chat D's | Run under `env_test`; nothing in chat D's scope changes |
| ~~D12~~ | ~~cluster~~ | ✅ **VERIFIED 2026-08-27** — checked against the KIRBy checkout at `~/repos/KIRBy`. Commit `333f005` resolves, its message is exactly as quoted, and the guard it added (refusing to exit 0 when every model-and-condition combination raised) is in the file. The one stale part is the line number: `:1342` no longer points at the progress line, because that file has moved on since August. The claim is sound; the citation is a moment in time | ✅ Done |

#### For chat G

| # | Kind | What | Fix |
|---|---|---|---|
| ~~G1~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — gate 11 now averages the delivered dose over **100 seeds** on each side instead of comparing one draw against one draw, the same rule the flat-dose and ν→∞ gates already use. Verified below. Original report: 🔴 **The documented launch preflight EXITS 1 on 4,000 real QM9 molecules.** Gate 11 — *"validation is dosed against the clean training spread, not its own"* — fails on Student-t ν=3: train 0.665, validation 0.909, +36.5% against a 21.2% band. Deterministic, reproduced by two verifiers independently, with and without the scaffold file. **It passes on the full 133,885 column.** The cause is the exact pattern chat G fixed twice and missed here: a **single draw**, `seed: 42` at `rust/src/main.rs:1943`, on the heaviest-tailed shape — whose own per-run dose spread the same self-test measures at 14.5%. Gate 11 is chat F's code, but it is the gate an operator is pointed at | ✅ Done — §8a |
| ~~G2~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — the evidence file could not be re-analysed by its own script | The analysis reads its roster from the file, drops non-roster models by name, and says so |
| ~~G3~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — the verdict table now carries the replicate count and the model beside the detectability figure, and prints them, so the label cannot drift from the arithmetic. With Ridge dropped by the roster filter every figure is on twelve replicates, which the printout states. Original report: **Guard 8 quietly makes Ridge an eleven-replicate result** — replicate 2's clean R² of −16.99 is dropped by the accuracy floor, removing 28 rows. The row labelled "what twelve replicates could have detected" has an upper end computed on eleven | ✅ Done |
| ~~G4~~ | ~~code~~ | ✅ **FIXED 2026-08-27** — the docstring now says the corrected ids are used for both the noise grouping and the split, and records the R² = −0.40 collapse that found the omission. Original report: **A docstring states the opposite of what the code does.** `scripts/setting_selection_test.py:430-431` says the split keeps raw scaffolds and only the noise grouping is corrected. Both call sites hand the corrected ids to the splitter, and the real pipeline corrects the split too — so both halves of the sentence are false, and they contradict the fix chat G claims | ✅ Done |
| ~~G5~~ | ~~doc~~ | ✅ **FULLY FIXED 2026-08-27** — `NOISE_DESIGN.md` §5.8 was rewritten on the roster models throughout and carries the correction notice; every remaining `0.0058` quote is gone from both documents and from `noise_conditions.json`. Earlier that day: **PARTLY FIXED** — §13.9's headline numbers were Ridge rows; the correction table is now at the head of §13.9 and `noise_conditions.json`'s Laplace entry no longer quotes a Ridge number | ✅ Closed |
| ~~G6~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — recommendation 4 now quotes **0.0136 R²**, LightGBM at level 1.5, 0.24 of the wobble, paired *t* p = 0.350, with the reporting-level figure of 0.0024 beside it. The audit's own replacement, 0.0362, was also a Ridge row; the correction table's 0.0087 was an exact-dose row and the primary analysis is the algebraic dose. **The recommendation survives on all three numbers** |
| ~~G7~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — both document headers, both condition tables, `NOISE_DESIGN.md` §7 and this document's decision register now say Laplace is kept at depth. Original report: 🔴 **Laplace reads as still open in six places** although the author settled it on 2026-08-27 and the conditions file records it as kept with no optional marker. Two of them are the **first lines a reader lands on** in each state document, so the next session is told a settled decision is still waiting on the author. One of them asserts the conditions file marks it optional, which is false. No code disagrees | ✅ Done |
| ~~G8~~ | ~~doc~~ | ✅ **FIXED 2026-08-27** — all four corrected to **28 noise gates + 5 writer guards = 33**, counted from `#[test]` in `rust/tests/`, and both figures reproduced by `cargo test --release` |

**One thing the audit is emphatic about, and it is the reason G5 and G6 matter:** the condition set
itself is **not** overturned by removing Ridge. Without it the heavy-tailed and contamination ladders
stay flat at 0.0027, and grouped-shifted still separates at level 1.5 on both tree models at
p ≤ 0.0023. What is wrong is every number quoted for those verdicts, in a file that other files cite
as their evidence.

---

### 13.13 ✅ SETTLED 2026-08-27 — censoring runs on about five pairs, QM9 only

**The author's instruction:** *"A SUPER SMALL set of model/rep pairs to test this with on QM9. Like
5. Just to see the individual affects no need to run the full suite."*

**Censoring comes off the full grid.** It runs on **about five model-and-representation pairs, on
QM9, at all seven of its levels, 10 replicates.**

| | Runs |
|---|---|
| Censoring as a full-grid condition (78 pairs) | 5,460 |
| **Censoring on 5 pairs** | **350** |
| Saved | 5,110 |

**Which five pairs is chosen from the screen** (§13.1 item 4), like the deep run's selection.

**No code change is needed.** The generator already takes the flags:
`--conditions censoring --models <...> --reps <...>`.

✅ **APPLIED IN THE FILES 2026-08-27.**

- `noise_conditions.json` carries a **`qm9_scope`** block on censoring — `mode: pair_subset`,
  `n_pairs: 5`, with the reason and the sentence the paper owes. Censoring **stays in the
  full-grid group**, because moving it would also remove it from the uncertainty runs, where it is
  one of only two conditions that can answer the which-molecules question. The scope is QM9 only and
  the file says so.
- The QM9 generator reads that block rather than hardcoding it. **The screen and the main grid now
  default to Gaussian, grouped-wider and grouped-shifted** — verified: both report
  `conditions: gaussian grouped_wider grouped_shifted`. Asking for censoring without `--models` and
  `--reps` is refused with a message naming this section. Asking for it on more than twice `n_pairs`
  is also refused, and says to change the file rather than the flag.
- Guard: `scripts/test_noise_conditions.py` check 8 runs the real generator and asserts all four
  behaviours. **Proved to catch a regression** — reverting the default made it fail.
- §13.1 item 3's arithmetic prices the main grid at four noise types on all 78 pairs. With
  censoring off that list it becomes three, plus censoring's own small run. Current totals are in
  §13.14, read off the generator.

**What the paper has to say, because this is a reduced design and reduced designs get asked about:**
censoring was run on a named subset to measure the size of its effect, not to compare models and
representations under it. Any claim about *which* model resists censoring best rests on those five
pairs and must say so. The claim the reduced run can still carry is the one that matters — how much
damage censoring does compared with ordinary noise at the same delivered amount.

### 13.13a Censoring — what it costs, and the caveat that was withdrawn

#### It is not a second study. It is one condition of four, at the same price as the others.

| | Runs across the screen and the main grid |
|---|---|
| Any one ordinary noise condition, 7 levels | 5,460 |
| **Censoring, 7 levels** | **5,460** |
| The main grid, all four conditions | 21,840 |

**Censoring is 25% of the main grid** — exactly its share as one of four conditions. It reads like
extra work because its levels measure a different quantity and so have to be described separately,
but it costs no more than Gaussian does.

**If it is cut anyway**, the levels are the only lever: 4 levels saves 2,340 runs, 3 levels saves
3,120. The levels were set by the range-finding run, which found no knee — damage accelerates
smoothly, becomes serious around 20% and severe past 30% — so a reduced grid should keep the
resolution where the acceleration is and drop from the flat bottom: **0, 20, 30, 50%** rather than
thinning evenly. `NOISE_DESIGN.md` §6.4 owns the levels; change them there.

#### ⚠️ The "censoring on QM9 is imposed" caveat is withdrawn, 2026-08-27

`NOISE_DESIGN.md` §5.3b and §5.5 both carried it, and it was wrong. The author's words: *"All the
noise in qm9 is imposed, how is this any different? It's actually the best case of censoring."*

**Every condition on QM9 is imposed.** QM9 labels come from a calculation — nothing is measured, so
nothing is corrupted until this study corrupts it. Singling censoring out said nothing.

**And QM9 is the best place to measure what censoring costs**, for the same reason it is the best
place to measure anything else here: the true value of every clipped label is known, so the damage
can be measured exactly. On a real assay dataset any censoring that survived curation is already in
the labels with no record of the true value, so its cost can be observed but not measured.

**One caveat does survive, and it is about the mechanism rather than about QM9:** clipping the top
removes label range as well as corrupting individual values. That is what censoring does to a real
dataset too, so it is a property of the mechanism and not an artefact — but a reader will ask, and
the Methods should say it.

---

### 13.15 ✅ The roster level screen — the evidence the QM9 reporting level rests on

#### 🔴 What this data IS and IS NOT, before anything is quoted from it

**It was run to answer one design question: at what noise level should the tables report?** That is
Q3's outstanding requirement (§7.0). It answered it — level 1.0 (§13.11).

**It is not a result and nothing in the paper may cite it.** It is a screening harness, not the
pipeline:

| | The screen | The real run |
|---|---|---|
| molecules per replicate | 4,000 | 10,000 |
| representations | PDV only | six |
| models | 7 | 13 |
| noise conditions | 2 | 3 at all 78 pairs, 6 in the deep run |
| target | the HOMO–LUMO gap | the same, but through `process_and_train.py` |
| code path | `scripts/setting_selection_test.py` | the training pipeline |

**So the NGBoost finding below is a LEAD, not an answer.** It bears on Q1 (is robustness decided by
the model or the representation) and Q3 (what model choice buys you), and both of those are answered
by the real run on the full grid or not at all. What it earns is a place on the shortlist for the
deep run (§13.1 item 4) — it was not in the earlier three-model screen at all.

`results/roster_level_screen.csv`, 420 rows, run 2026-08-27. QM9, PDV, 4,000 molecules per
replicate drawn fresh, real Murcko scaffold split, noise on training labels only, scored on clean
test labels. **Seven models, ten replicates, levels 0.5, 1.0 and 1.5 plus the clean baseline.**

**Replicate 2 is excluded and named.** Both neural models diverged on it and so did Ridge in the
earlier screen — it is a scaffold split those model families cannot fit, not a model fault. Nine
replicates everywhere below.

**Not in the screen, and why:** the quantile forest cannot be built in this environment
(scikit-learn 1.3.2 rejects `monotonic_cst`); the Gaussian process segfaults in a process that has
already loaded the boosting libraries (§2.8e); the four Bayesian variants are variants of the DNN and
MLP. **Ridge is not in the study and is not here.**

⚠️ **Each model was run in its own process.** Fitting the neural models and the boosting models in
one process **hangs** — see §2.8e-ter, found doing exactly this. The replicate seeds depend only on
the replicate index and the level, so the separate processes draw the same molecules, the same split
and the same noise; XGBoost gives the same clean R² to four decimals either way.

#### PDV — plain Gaussian. Median R² over the nine replicates, with the range beside it

**Every table in this section is one representation, PDV, and says so in its title.** That is the
rule: replicates are repeats of the same experiment and a median over them is what a median is for;
representations are different features and are never pooled.

| Model | clean | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|
| NGBoost | 0.871 `[0.846–0.892]` | 0.860 `[0.829–0.878]` | 0.836 `[0.799–0.854]` | 0.798 `[0.751–0.828]` |
| DNN | 0.907 `[0.857–0.922]` | 0.866 `[0.828–0.892]` | 0.799 `[0.719–0.834]` | 0.775 `[0.653–0.809]` |
| MLP | 0.902 `[0.840–0.921]` | 0.854 `[0.831–0.880]` | 0.791 `[0.774–0.860]` | 0.772 `[0.686–0.831]` |
| SVM | 0.874 `[0.830–0.901]` | 0.849 `[0.790–0.872]` | 0.795 `[0.743–0.825]` | 0.722 `[0.691–0.770]` |
| Random forest | 0.895 `[0.862–0.920]` | 0.867 `[0.836–0.888]` | 0.798 `[0.745–0.822]` | 0.681 `[0.625–0.733]` |
| XGBoost | 0.903 `[0.878–0.925]` | 0.872 `[0.833–0.883]` | 0.783 `[0.697–0.800]` | 0.639 `[0.555–0.739]` |
| LightGBM | 0.902 `[0.882–0.925]` | 0.871 `[0.843–0.895]` | 0.784 `[0.723–0.805]` | 0.611 `[0.558–0.730]` |

**NGBoost has the narrowest range at every level and the highest median at 1.0 and 1.5.** At level
1.5 its worst replicate, 0.751, beats every other model's best except the two neural ones — and its
best clean replicate, 0.892, is below every other model's best. It starts lowest and ends highest.

#### PDV — grouped-shifted noise. Median R² over the nine replicates, with the range

| Model | 0.5 | 1.0 | 1.5 |
|---|---|---|---|
| NGBoost | 0.853 `[0.812–0.876]` | 0.794 `[0.753–0.835]` | 0.671 `[0.647–0.783]` |
| DNN | 0.833 `[0.748–0.888]` | 0.682 `[0.585–0.721]` | 0.363 `[0.122–0.734]` |
| MLP | 0.835 `[0.771–0.881]` | 0.691 `[0.591–0.779]` | 0.397 `[0.120–0.630]` |
| SVM | 0.826 `[0.772–0.861]` | 0.744 `[0.692–0.802]` | 0.597 `[0.469–0.694]` |
| Random forest | 0.857 `[0.816–0.883]` | 0.765 `[0.742–0.801]` | 0.589 `[0.492–0.723]` |
| XGBoost | 0.861 `[0.823–0.893]` | 0.725 `[0.701–0.779]` | 0.535 `[0.425–0.646]` |
| LightGBM | 0.857 `[0.823–0.884]` | 0.726 `[0.691–0.778]` | 0.502 `[0.384–0.629]` |

⚠️ **Look at the ranges on the neural models at level 1.5.** The DNN spans 0.122 to 0.734 and the MLP
0.120 to 0.630 — wider than the entire difference between models. **How much systematic family-level
bias costs a neural network depends enormously on which scaffold split it draws.** NGBoost's range at
the same level is 0.647–0.783, a sixth as wide. The median alone would not show this, which is why
the range is beside it.

#### Which model is best, counted per replicate

| Level | Winner |
|---|---|
| clean | LightGBM 3/9, DNN 2, XGBoost 2, MLP 2 |
| 0.5 | LightGBM 4/9, MLP 2, forest 1, XGBoost 1, DNN 1 |
| **1.0** | **NGBoost 8/9**, MLP 1 |
| **1.5** | **NGBoost 7/9**, MLP 1, DNN 1 |

#### 🔴 The result, and it is the study's argument in one line

**The least accurate model on clean labels is by far the most robust.** NGBoost is last on clean
data at 0.871 and first by a clear margin at both 1.0 and 1.5, winning 8 of 9 and 7 of 9 replicates.
LightGBM is first on clean data and loses the most of anyone — 0.902 down to 0.611.

**This is exactly what the paper set out to test**, and it is not visible at level 0.5, where the
ordering is still the clean ordering and every model sits within 0.03 of its clean score.

#### Why the QM9 reporting level is 1.0 rather than 1.5

| | at 0.5 | at 1.0 | at 1.5 |
|---|---|---|---|
| Is the robustness result visible? | no | **yes, 8/9** | yes, 7/9 |
| Grouped-shifted visible? | 0–3 of 9 | 9/9 neural, 1–4 of 9 trees | 6–9 of 9, every model |
| Every model still fitting? | yes | yes | yes on QM9, **no on Caco-2** |

1.5 shows the grouped-shifted result in every model rather than only the neural ones, which is its
one advantage. It is outweighed by Caco-2, where two of four models fall below zero at 1.5 and the
reporting level has to work on every dataset (§13.11).

**What the paper must therefore say about grouped-shifted at level 1.0:** it damages both neural
models in 9 of 9 replicates and the random forest and XGBoost in 1 of 9. That is a real finding —
neural models are far more damaged by systematic family-level bias than trees are — and it has to be
written as that, not as a single pooled claim.

---

### 13.14 The run design — what runs, and what each part costs

**Read off `slurm_scripts_qm9_rerun/generate_scripts.py` on 2026-08-27, not derived here.** Rebuild
this table by re-running the generator, never by editing the numbers.

QM9 has 13 models × 6 representations = **78 pairs**.

| Part | Noise types | Pairs | Replicates | Training runs |
|---|---|---|---|---|
| **The screen** | Gaussian, grouped-wider, grouped-shifted | 78 | 1 | **1,680** |
| **The main grid** | the same three | 78 | 9 more | **15,120** |
| **Censoring** | censoring | **5** | 10 | **300** |
| **The deep run** | the three above **plus** Student-t, outlier, Laplace | 12 (example: 4 models × 3 reps) | 10 | **5,040** |
| **QM9 total** | | | | **22,140** |

The screen is replicate 0 of the main grid and is reused, so the main grid adds nine more rather
than ten.

**Why Gaussian, grouped-wider and grouped-shifted are the three that run on all 78 pairs, and the
other three do not.** Every one of the six needs its own accuracy-versus-level curve to enter the
per-noise-type decomposition, and running one on all 78 pairs costs about 5,000 runs. The three
above are structurally different from each other — noise spread evenly, noise concentrated on whole
scaffold families and centred, and the same families pushed one direction — so each needs its own
curve. Student-t, the outlier condition and Laplace were **measured** to be indistinguishable from
Gaussian in accuracy (§13.9), so paying 5,000 runs each for a curve that would lie on top of
Gaussian's buys nothing. They run in the deep run, on a dozen pairs, where the question is whether a
small difference exists at all rather than how it varies across every model.

**Only Gaussian runs the clean level.** At zero noise the pipeline adds no noise, and the replicate
seed depends only on the replicate number, so the clean run is **bit-identical whichever condition it
is labelled with** — measured to the last digit on all four conditions. Running it once per condition
would cost 11% of the grid to recompute a number already on disk. `copy_zero_rows.py` fills the rest
in afterwards, and refuses to overwrite a clean row a job actually computed — it checks it against
the reference instead, which is a free four-way agreement test on production runs.

#### Censoring can use Gaussian's clean rows. It just has to run second.

An earlier note here implied it could not. It can. `copy_zero_rows.py` indexes **every** results file
in the directory by its configuration, so once the main grid has produced Gaussian's clean rows for
all 78 pairs, censoring's five pairs are already covered.

**The only constraint is ordering: run the copy step after the main grid, not on a censoring-only
results directory.** The generator warns when it builds a condition set with no Gaussian in it,
because at that moment there is nothing to copy from — not because the two runs cannot share.

#### Do the censoring runs need ten replicates?

| Replicates | Runs | What it gives |
|---|---|---|
| 1 | 30 | The size of the effect, no spread at all |
| 3 | 90 | A range. Not enough for a significance test — a paired test on three pairs floors at p = 0.25 |
| **10** | **300** | Same footing as everything else. A paired test floors at p = 0.002 |

**Recommendation: keep 10.** The saving is 270 runs out of 22,140 — **1.2%** — and at any smaller
number censoring's results carry a different error bar from the rest of the study, which then needs
explaining in the caption. The effect is large enough (about twelve times anything else) that
precision is not the reason to keep them; consistency is. **Cut to 3 only if the run is being
squeezed for time**, in which case say in the Methods that censoring carries a smaller spread.

#### The uncertainty runs, which are not on this grid at all

Three experimental datasets × 4 representations × 7 models, on the four main-grid conditions plus the
outlier one (§13.1 item 6), one replicate plus a permutation null (§13.1 item 2). **336 jobs.**

**Why the training-run count is not given here.** Each job trains its model more than once. To ask
"is the model unsure about the molecules whose labels were corrupted", every training molecule needs
a score from a model that never saw it — so the training set is split into folds and the model is
refitted once per fold. How many times depends on a setting in that generator, so the total lives
there rather than here.

#### What this replaces

§13.1's cost table prices the design at 22,400 runs across the screen and main grid, on four noise
types at 78 pairs. That predates the censoring decision. The table above is current; §13.1 is kept
for its reasoning about replicate counts, not for its totals.

---

### 13.2 The chats

Letters, not numbers, so inserting one does not renumber the rest. **A, C, D, E and G are
independent of each other and can run in parallel.**

#### The standing rule for every chat on this list

**Each chat FINISHES its issue. It does not re-plan it.**

Investigation and reading come first — every prompt says so, and none of this work should be done
from a summary. But the deliverable is working code with a check that proves it works, not a better
description of the problem. A chat that ends with a revised plan and no committed change has failed,
whatever it found on the way.

Specifically, in every chat:

- **Read the code before changing it.** The plan's file and line references are a starting point,
  not a substitute. They were correct when written and the tree moves.
- **Do the work.** If the plan turns out to be wrong about something, fix the plan *and then do the
  corrected work in the same session*. Do not hand back a correction as the deliverable.
- **Prove it.** Every fix ships with a check that fails if the fix is removed. A note in a document
  is not a check. Several defects in this project survived because a test searched the source for a
  matching line rather than running it.
- **Commit.** Uncommitted work is lost work — this project has already had to recover research from
  a temporary directory.
- **Update `RERUN_PLAN.md` and `NOISE_DESIGN.md`** to match what was actually done, and close out
  the items the chat covered rather than leaving them open.
- **Stop at the scope line.** Each chat says what it must not touch. `paper.tex` is out of scope for
  all of them.
- **If a decision is genuinely the author's, ask once, state a recommendation, and carry on with
  everything that does not depend on the answer.** Do not stop the whole chat on one open question.

| | Chat | Depends on | Can start now? |
|---|---|---|---|
| **A** | Noise redesign in Rust | — | ✅ **DONE 2026-08-26** |
| **B** | Noise redesign in the Python injector, and the cross-check | A, for the spec | ✅ **yes** — the specification is settled, so it can be written alongside A |
| **C** | Embedding storage fix, and the Gaussian-process re-test | — | ✅ **DONE 2026-08-26** — and it found the real cause of the near-zero embedding scores (§2.8f) |
| **D** | Infrastructure: settings race, writer guards, environment | — | ✅ **DONE 2026-08-26** |
| **E** | Cross-pipeline parity | — | ✅ **DONE 2026-08-26** |
| **F** | Uncertainty machinery: audit, fix the clear bugs, report the rest | — | ✅ **yes** — it has real work in it, and produces the material for the 1:1 |
| **G** | Local test: which noise settings earn their place | A | ✅ **DONE 2026-08-27** — §13.9; the set is in `noise_conditions.json`, gated on both sides |
| **H** | Job scripts, preflight, gates, launch | A ✅ D ✅ G ✅ + B C E + L + §13.1 | ❌ blocked — but not on G. Read the conditions from `noise_conditions.json` rather than restating them, and put `scripts/test_noise_conditions.py` in the preflight. **Also owns the one chat D check that cannot run off the cluster**: `python scripts/test_config_isolation.py --end-to-end` under `env_test`, now step 4 of the QM9 runbook (§13.2, D10) |
| **I** | The uncertainty decomposition build | F | ❌ blocked on F's findings |
| **J** | One figure script, and the five analyses | 1:1 on details, then the new columns | ❌ blocked |
| **K** | Sync the two documents, fix the bibliography | — | ✅ **yes** — smallest, entirely self-contained |
| **L** | **The environment rebuild** — one threading runtime, the roster completed | — | ▶️ **running** (opened by the author 2026-08-27). The procedure is §2.8i. **Nothing launches until this passes**: the per-task guard refuses every job in all three families while more than one threading runtime is present |

**Eight can start immediately and run unattended: A, B, C, D, E, F, G, K.** L is running.

**Two things are recorded but were owned by nobody until 2026-08-27**, which is how a documented
step becomes a step that never happens. The environment rebuild is now chat L. The concurrency
check that only works on the cluster is now chat H's, and — more to the point — it is now a
numbered step in the runbook an operator actually follows, not only a row in an audit table.
 They touch
different files. The one overlap to watch: A and B both change what the noise means, and C changes
how features are stored — if two of them land at once, the person merging needs to re-run the checks
in §8 rather than trusting either in isolation.

---

#### Chat A — Noise redesign in Rust ✅ DONE 2026-08-26

**What landed.** `rust/src/main.rs` no longer contains the old scheme. The six unreachable
distribution variants and the five superseded targeting rules are deleted outright, along with
`generate_value_based_noise_map`, `generate_adaptive_noise`, `generate_noise_by_indices` and
`sample_from_distribution`. In their place: three shapes (Gaussian, Student-t, Laplace), five
targeting rules (uniform, grouped-wider, grouped-shifted, outlier, censoring), the dose solver, and
the provenance. `scripts/noise_strategy_params.json` is gone with its dead argument.

The four things the prompt required, each demonstrated by a check that fails the build rather than
by a claim:

| Required | Where the check is | What it measures |
|---|---|---|
| The delivered amount is identical across conditions at a given setting | `gate_one_dose_is_flat_across_types`, and `--self-test` on the real column | mean over 20 seeds within 0.74% of target for every condition; **spread between conditions 1.27%** on 4,000 real QM9 molecules |
| Labels standardised with the **clean** training mean and spread, computed before injection | `standardisation_uses_the_clean_training_spread` | the constants do not move across the level grid, and the written value is the noisy label standardised by them |
| Held-out labels bit-identical across every level | `held_out_labels_are_bit_identical_across_levels` | bit-for-bit, every condition × every level |
| The injected value written per molecule, reconstructing the noisy label exactly, exactly zero at zero | `recorded_noise_reconstructs_the_label`, `zero_level_records_exactly_zero` | `y_clean + epsilon == y_noisy` on f32 equality; `epsilon == 0.0` at level 0 |

Each was checked by removing the fix and confirming the check fails — including removing the dose
solver, which reproduces the original confound on demand at 1.00–1.70× the Gaussian dose.

**The third leg, added on audit.** The scheme lives in three places and only two were tied
together: chat B's `scripts/crosscheck_injectors.py` ties the reference to the Python injector,
and nothing tied either to the pipeline that actually noises QM9.
`scripts/crosscheck_pipeline_reference.py` closes it, so the chain is now
**Python injector ↔ reference ↔ pipeline**. On 4,000 real QM9 molecules at k = 0.25, 0.5 and 1.0
the unit doses are identical where algebra fixes them, censoring agrees to six decimal places, and
the 20-seed mean doses agree within 0.82% (2.53% at ν = 3, whose fourth moment is infinite).
Details and what it found: `NOISE_DESIGN.md` §5.1d.

✅ **One thing it found, now settled.** `affected_molecule_fraction` meant 1.0 in the pipeline and
0.0 in the reference for the conditions with no selection rule. Settled on **1.0** across all three
implementations on 2026-08-26 — every molecule is perturbed, and the 0.0 reading left the reference
disagreeing with itself, since grouped-shifted already recorded 1.0. Written up in
`NOISE_DESIGN.md` §5.1d finding 2 (chat B), with a pointer at §5.1c because that is where the code
comments in the other two implementations send you. Both cross-checks now compare the column and
pin it at 1.0, so it cannot drift back.

One knock-on: failure mode 6's guard below says to assert the affected fraction "is neither near
zero nor near one". That belongs to conditions that **select** — a condition with no selection rule
is legitimately at one.

**One defect chat B's cross-check found in this work, now fixed (§2.3a):** the shifted grouped
condition's `effective_n` divided by the raw scaffold-group count instead of the effective one. It
put no wrong number in a results row — it made the flat-dose gate demand a precision the condition
cannot deliver, so the gate was passing only by luck of seed and would have failed intermittently
once the real run started. Applied, and guarded by a regression test that asserts the answer matches
the effective count and does *not* match the raw one.

**Eleven more gates** came out of doing the work: the ν ≤ 2 refusal, a mismatched scaffold file
refused rather than silently degraded to uniform noise, a short record stream stopped rather than
shifting every molecule's noise by one, censoring's direction, the affected-molecule fraction
measured rather than assumed, manifest completeness, and seed reproducibility.

**Reconciled with chat B**, which landed `NOISE_DESIGN.md` §2a and rewrote
`rust/reference/noise_arms.rs` while this was in flight: the group-selection rule, the numpy
quantile for the censoring limit, f64 accumulation in every statistic, and the condition names
(`gaussian`, `student_t_nu5`, `grouped_wider`, `grouped_shifted`, `outlier_p05`, `censoring_25`)
now match the reference exactly, so the cross-check can join the two implementations on the name.

**One difference from the reference, deliberate.** The reference flattens shape and targeting into
one condition list; the pipeline keeps them separately selectable, as `NOISE_DESIGN.md` §6.0
prescribes. The two agree exactly on every condition that is actually queued, because those all use
the Gaussian shape where the normalisation is the identity.

**Also changed, because the injector cannot be called otherwise:** `scripts/process_and_train.py`
now writes the scaffold-group file, passes the new arguments, records the manifest beside the
results, and **refuses** `--sigma`, `--distribution`, `--noise-strategy` and `--strategy-params` by
name. A job script written against the old scheme must not run silently under the new one, where
the level means something different. **The thirty existing SLURM script directories all use the old
flags and will now fail loudly** — they are rebuilt in chat H (§5.3).

**How to re-run everything chat A claims, in three commands.** None of it needs the cluster, and
none of it needs the Python training stack.

```
cd rust && cargo test --release --test noise_gates          # 28 gates over real mmap files
./rust/target/release/rust_processor --self-test <labels.csv> --scaffold-file <groups.json>
python scripts/crosscheck_pipeline_reference.py --labels <labels.csv> --groups <groups.json>
python scripts/test_injector_wiring.py                      # the Python driver's helpers
```

Last run **2026-08-27** against the tree at this commit:

| Command | Result |
|---|---|
| `cargo test --release` | ✅ **33 pass** — 28 noise gates + 5 writer guards |
| `rust_processor --self-test` on all 132,480 QM9 labels | ✅ `EXIT=0`, all gates pass |
| the same on 1,000 / 2,000 / 4,000 / 8,000 / 20,000 molecules with real Murcko groups | ✅ `EXIT=0` at every size (§8a) |
| `scripts/crosscheck_injectors.py` | ✅ **342 checks**, both injectors agree |
| `scripts/crosscheck_pipeline_reference.py` | ✅ all **17** conditions at k = 0.25, 0.5 and 1.0 |
| `scripts/test_injector_wiring.py` | ✅ passes |
| `scripts/test_noise_conditions.py` | ✅ **7 conditions** resolve on both sides with the settled parameters, and the job generator agrees |
| `scripts/check_bib_and_docs.py` | ✅ OK — one pending item, the author's own `\bibliography` line |

Every one was falsified by removing the fix it guards; §8a records the falsification for the gate
that was fixed on 2026-08-27.

**The Python driver is covered without the training stack.** `process_and_train.py` cannot be
imported on a machine without `torch_geometric`, so the three helpers chat A added to it were only
ever read. `scripts/test_injector_wiring.py` lifts them out with `ast` and runs them — the
acyclic-singleton rule (`NOISE_DESIGN.md` §2a rule 2), the manifest CSV and its §5.2 columns, and
the retired-flag refusal. Falsified by collapsing the acyclic molecules back into one group, which
is what the rule exists to prevent.

**Still open, and not chat A's:** whether the validation split gets its own independently drawn
noise (§5.1 item 5, §13.5 — the author's), the ECFP4 truncation and index-drift guards (§5.1 item
7, chat D), and whether Laplace is queued (`NOISE_DESIGN.md` §7 — built either way).

**Not covered locally, and only chat H can cover it:** the injector has never run inside a real
training job. Gate 6 (the clean-label run reproduces the old zero-noise numbers) and gate 9 (every
new column populated end to end) need one, and they are chat H's first act — not a gap in chat A,
but the reason chat A's green does not by itself license a launch.

**⚠️ A note for whoever merges.** Commit `d25bcb0`, which carries this work, also contains chat D's
`--config` change (the configuration race, §2.8a) — it was uncommitted in the working tree when
chat A staged `rust/src/main.rs`, and the commit message does not mention it. Nothing is lost and
nothing is wrong in the code, but chat D should not expect to commit that change again, and the
history attributes it to chat A.

---

**Does:** deletes the five superseded noise types and the six unreachable distribution variants;
builds the dose solver, the three shapes and the four targeting rules; fixes the standardisation
order; emits the provenance columns. Verifies locally against `rust/reference/noise_arms.rs`.

**Does not:** touch the Python injector, the models, the figure script, or `paper.tex`.

**Spec:** `NOISE_DESIGN.md` §6.1 (delete), §6.2 (build), §6.3 (verify). `RERUN_PLAN.md` §5.1
items 1, 2, 4, 6, §5.2 (the columns), §8 gates 1, 3, 4, 5, 7.

**✅ Both TODOs answered by chat G, §13.9:** one Student-t setting (suggest ν = 5) and one Outlier
setting (suggest p = 10%), because all three of each are within 0.006 R² of Gaussian and of each
other; Laplace is indistinguishable too and belongs in stage 2 if it is wanted for the citation at
all. The skewed draw was tested and does not earn implementation.

> **Prompt.** Implement the redesigned noise injection in `rust/src/main.rs`. The specification is
> `NOISE_DESIGN.md` §6.1–6.3 and `RERUN_PLAN.md` §5.1 items 1, 2, 4 and 6 — read both in full and
> read the code before changing it; do not work from the summaries. There is a working reference at
> `rust/reference/noise_arms.rs` that already builds and hits every target amount within half a
> percent: match it rather than reinventing it. Delete the old noise types outright — no
> deprecation, no flag guards, git history is the archive. Four things must be true when you are
> done and each must be demonstrated by a check that fails the run, not by a claim: the delivered
> amount of noise is identical across every noise type at a given setting; the labels are
> standardised using the **clean** training mean and spread, computed before injection; held-out
> labels are bit-identical across every noise level; and the injected value is written per molecule
> and reconstructs the noisy label exactly, with exactly zero at zero noise. Verify locally on QM9 at
> 4,000 molecules. Do not touch the Python injector, the models, the figure script or `paper.tex`.
> Update `RERUN_PLAN.md` and `NOISE_DESIGN.md` before you finish.

---

#### Chat B — Noise redesign in the Python injector, and the cross-check

**Does:** implements the identical specification in `NoiseInject/noiseInject/core.py`, and builds
the cross-check that both implementations deliver the same thing on the same labels.

**Why it is separate and why it matters:** the two injectors already drifted apart once and nothing
noticed (§2.3). `NOISE_DESIGN.md` names the Rust file only — **the Python half is unspecified**, and
it is what produces all three experimental datasets and every uncertainty number.

**Head start nobody has recorded:** `KIRBy/src/kirby/noise_spec.py` already solves the same problem.
It expresses the noise level as a target fraction of the label spread and finds the raw scale by
binary search on the actual labels — the dose solver, in the author's own code, since 13 July. Its
docstring also records two properties worth preserving: a fresh generator per call, and the
realisation drawn once over the whole label column then subset per fold, so a molecule gets the same
corruption whichever fold it lands in.

> **Prompt.** Implement the redesigned noise injection in the Python injector
> (`NoiseInject/noiseInject/core.py`) to the same specification as the Rust side — `NOISE_DESIGN.md`
> §6.1–6.3. Read `KIRBy/src/kirby/noise_spec.py` first: it already implements a dose solver by binary
> search on the real labels and documents two properties (a fresh generator per call, one realisation
> drawn over the whole label column then subset per fold) that must be preserved. Then extend
> `NOISE_DESIGN.md` so it specifies the Python injector explicitly — as written it names
> `rust/src/main.rs` and nothing else, which is the same omission that let the two implementations
> drift apart in the first place. Finally, build the cross-check: same labels, same seed, same target,
> both implementations, agreement within the tolerance in `NOISE_DESIGN.md` §5.1b. It must be a test
> that fails, wired into the preflight, not a note. Do not touch `paper.tex`.

---

#### ✅ Chat B — DONE 2026-08-26

**What landed.**

| Where | What |
|---|---|
| `NoiseInject` **1.0.0** | The six regression strategies and both regression calibrators deleted. Built: three shapes, five targeting rules, the closed-form dose solver, a `CONDITIONS` registry, and `InjectionResult` carrying the provenance every results row needs. 58 tests pass; the dose-matching check was verified to **fail on 8 of 10 conditions** when the solver is removed. Examples, both notebooks and the README move with it |
| `NOISE_DESIGN.md` | §6.0a names both implementations and their callers; §6.1–6.3 cover the Python half; §2a gives the grouped-shifted algebra and the two rules the real scaffolds force |
| `scripts/crosscheck_injectors.py` | Gate 2. 342 checks on all 133,885 real QM9 labels and real Murcko scaffold groups |
| `slurm_scripts_uncertainty_rerun/preflight.sh` | Gate 2 **wired in as section 2b**, so it runs before anything is submitted rather than sitting in a document. Sections 1, 2, 4b, 4c and 5 also moved off the deleted conditions — 4c tested that heteroscedastic and value-proportional ranked molecules identically, which was the *finding* that justified deleting them, so it now tests that every condition delivers the amount it was asked for |
| `rust/reference/` | Now buildable by anyone — it had no `Cargo.toml`, so gate 2 was unrunnable. Censoring and grouped-shifted added; `--json`, `--groups` and `--seeds` added |
| `KIRBy` | `noise_spec.py` and `alternative_data_noise_robustness.py` moved to the new conditions, with the provenance columns written per level; both smoke tests updated |

**The headline number.** Dose spread across every condition at one setting: **1.16% in Rust, 0.40%
in Python.** Before the redesign it was 0.49× to 2.00×, i.e. 308%.

**Four defects the cross-check found**, none of which any single implementation would have shown —
the `effective_n` formula wrong in *both* (§2.3a, applied in all three 2026-08-26), censoring
reporting a limit at its clean baseline, censoring encoding its fraction twice so a "level 0" run
still clipped, and the Python solver matching the first moment (§2.3b).

**A fifth, found on the close-out pass, and it is the worst of them.** `noise_scale` re-ran the
group selection over whatever group array it was handed, so the pattern a *held-out* molecule was
scored against described a different set of groups from the one training was actually corrupted in
— measured on a 40-group split, two of eight corrupted groups went unmarked. That is question B,
for exactly the conditions §3.2 identifies as the only ones with a pattern to find. Fixed by a
`reference_groups` argument, wired through both runners, and pinned by a test that fails if it is
dropped. Also on that pass: `grouped_shifted` returned a constant per-molecule scale while
reporting `scale_is_degenerate` as False — the flag now agrees with the array.

**One more, found while reading the pipeline rather than by the gate.** The confound control's
"level-invariant noise pattern" was not level-invariant: the scale map drew from the same generator
as the noise, so recomputing it gave a different pattern each time and the zero-noise subtraction
compared two different patterns. Fixed in `noiseInject` — the selection now draws from a separately
seeded generator (§3.3).

**Verification, all run locally on the close-out pass 2026-08-26.**

| Check | Result |
|---|---|
| `pytest tests/` in `NoiseInject` | **55 pass** |
| `python scripts/crosscheck_injectors.py` | **342/342** at three levels, re-confirmed **154/154 at one level, exit code 0**, after the close-out fixes. All 133,885 real QM9 labels, 30,313 real Murcko scaffold groups |
| The gate actually fails | sabotaging the dose solver → **40 of 154 fail, exit code 1**; a condition present in one implementation but not the other → fails **by name**, not silently skipped |
| `tests/smoke/smoke_nine_fixes.py` | all pass |
| `tests/smoke/smoke_uncertainty_patch.py` | all pass, including out-of-fold error exceeding in-sample by 59% and censoring's confound-subtracted effect of −0.656 |
| `tests/experiments/noise/_smoke_noisy_search.py` | sections 1–5 pass. **Section 6 fails for an unrelated reason** — it asserts on a SQLite `result_db` that is not written at all any more, including for its own *clean* run, and `~/.cache/kirby/` has no `result_db` either. `find_best_hybrid.py` and `hybrid.py` are being changed by another chat. Not noise |
| Held-out labels bit-identical across every level | ✅ every condition; the caller's training array is not mutated either |
| Determinism across processes | ✅ identical seeds and identical draws under three `PYTHONHASHSEED` values — `zlib.crc32`, not `hash()`, which python randomises per process |
| The solved scale hits the target exactly | worst relative error **2.2e-16** across levels 0.01–3.0 × the label spread |
| The shapes are the distributions they claim | Kolmogorov–Smirnov against the theoretical law: gaussian p = 0.56, Student-t ν = 10/5/3 p = 0.51/0.45/0.52. Laplace read p = 0.002 on one seed and was checked over twelve — 1 of 12 below 0.05, exactly chance, with mean, spread, skew and kurtosis all on theory. Not a defect |
| Hostile inputs | a constant label column, a single label, one single group, a negative dose, a censored fraction of 1.0 — each either handled and recorded, or refused |

**`noise_spec.py`'s two guaranteed properties, which chat B's prompt named** — a fresh generator per
call, and one realisation over the whole label column subset per fold — **are both intact**, checked
directly. The pipeline does not go through that module and never had the second; that is measured
and handed to you in §3.3a rather than changed unilaterally.

**🔴 One more for chat F, found in `noiseInject/uncertainty.py` on the close-out pass.** The comment
beside the pooled uncertainty-noise correlation read *"this pooled value is the paper's
noise-tracking metric"* — which is failure mode 1 in §0.6, endorsed in the package's own source. The
false claim is gone and the two values are now labelled so neither can be reported under the other's
description, but **which one the paper reports is chat F's and chat J's**, not chat B's. The
per-level `unc_noise_rho` already exists in `per_sigma_df` and is the one §3.5 established.

**🔴 Left broken deliberately, and this is chat B's one incomplete edge.** The breaking change
raises `ValueError` on the old strategy names wherever they are still used. Inside this study,
everything is updated. Outside it, **ten KIRBy scripts belonging to other studies still name them**:
`cox2_noise_robustness.py`, `drd2_noise_robustness.py`, `esol_noise_robustness.py`,
`hybrid_noise_robustness.py`, `qm9_graphs_noise_robustness.py`, `noise_mitigation_advanced.py`,
`noise_mitigation_extended.py`, `complexity_theory_experiments.py`, `phase3_analysis.py`,
`phase4_analysis.py`, plus `tests/experiments/complexity/c10_noise_robustness.py`. They fail loudly
rather than silently producing a wrong number, which is the right failure mode, but they are not
this paper's and were not fixed. **Your call whether they matter.**

---

#### Chat C — Embedding storage, and the Gaussian-process re-test ✅ DONE 2026-08-26

**What landed** is in §2.8c. Beyond the three storage changes: Avalon was added to both pipelines,
Avalon and ChemBERTa were wired into the experimental runner, the guard is
`scripts/test_embedding_storage.py`, and the measurement is
`scripts/retest_embedding_kernels.py`. **The noise scheme was not touched** — no change to
`NOISE_DESIGN.md` was needed or made.

#### ✅ Measured 2026-08-26 — does distance between molecules still mean anything?

**This half needs no model, so nothing an optimiser does can reach it.** For each pair of molecules,
how far apart are they in the representation, and how different are their properties? If storage has
destroyed the shared scale, the two should stop tracking each other. Reported as a rank correlation:
zero means distance tells you nothing. Every replicate is shown; nothing is averaged.

600 QM9 molecules per replicate, scaffold split, about 480 to learn from
(`results/embedding_storage_retest/qm9_kernel_retest.csv`). **mol2vec appears in these tables
because it was measured before it was deleted on 2026-08-26. It is out of the study and out of the
code; the rows are kept because they are the only ones that can be checked against the cluster's
pre-fix numbers.**

| Representation | Old storage | New storage | Typical distance, old → new |
|---|---|---|---|
| **MHG-GNN** | 0.023, 0.059, 0.029 | **0.118, 0.146, 0.109** | 1060 → 37 |
| ChemBERTa | 0.101, 0.154, 0.101 | 0.170, 0.192, 0.159 | 1140 → 38 |
| PDV | 0.130, 0.144, 0.123 | 0.175, 0.207, 0.146 | 167 → 17 |
| mol2vec | 0.140, 0.170, 0.102 | 0.128, 0.168, 0.092 | 638 → 24 |

**MHG-GNN is the representation this defect was destroying.** Under the old storage, how far apart
two molecules sat told you essentially nothing about how their properties differed — 0.02 to 0.06,
against 0.11 to 0.15 once stored properly. Every replicate moves the same way.

⚠️ **The two learned embeddings were NOT damaged equally, and §2.8c assumes they were.** mol2vec's
three replicates are 0.140, 0.170, 0.102 before and 0.128, 0.168, 0.092 after — the same numbers.
The per-molecule stretch destroyed MHG-GNN's geometry and left mol2vec's alone. What ruined both
their scores on the cluster was the failed fit in §2.8f, which is common to both and has nothing to
do with storage.

#### ✅ Measured 2026-08-26 — the kernel gap was the FIT, not the features

**This overturns §10b.2 and it downgrades §2.8c. Read it before repeating either. The cause is
written up as §2.8f and the fix is in the pipeline.**

36 fits, none collapsed, 600 QM9 molecules per replicate, about 480 to learn from, scaffold split,
three replicates. Same molecules and same split within every row, so the only thing that differs
across a row is the storage.

**1. The two ways of measuring similarity now agree — on the DAMAGED features.**

The cluster reported them 0.86 to 0.89 apart on the learned embeddings, and §10b.2 read that gap as
proof the features were unusable. Give the fit a workable starting point and the gap disappears.
Twelve pairings, all on the old storage:

| Representation | Straight-line distance | Overlap fraction | Apart by |
|---|---|---|---|
| MHG-GNN | 0.680, 0.686, 0.755 | 0.713, 0.706, 0.756 | 0.033, 0.020, 0.001 |
| ChemBERTa | 0.562, 0.362, 0.630 | 0.585, 0.331, 0.627 | 0.023, 0.031, 0.002 |
| mol2vec | 0.683, 0.590, 0.754 | 0.661, 0.601, 0.743 | 0.022, 0.011, 0.011 |
| PDV | 0.709, 0.680, 0.753 | 0.726, 0.696, 0.712 | 0.017, 0.016, 0.040 |

**Largest gap anywhere: 0.040.** The cluster's 0.86–0.89 was the model failing to fit.

**2. The storage fix does not measurably change how well the model predicts.**

Difference between the new storage and the old, per replicate:

| Representation | Replicate 0 | Replicate 1 | Replicate 2 | Reading |
|---|---|---|---|---|
| **ChemBERTa** | **+0.076** | **+0.122** | **+0.014** | the only consistent gain |
| MHG-GNN | +0.053 | −0.060 | +0.039 | sign flips |
| mol2vec | −0.064 | −0.091 | +0.022 | sign flips |
| PDV *(reference)* | −0.001 | −0.013 | +0.050 | barely moves, as it should |

Run-to-run variation at this size is around ±0.06, so only ChemBERTa's gain stands clear of it, and
only just. **PDV is a reference row, not a control**: it was always stored as decimals and
standardised, so its "old storage" cell is a hypothetical that never ran in the pipeline.

**3. What IS damaged is the geometry, and the fix repairs it** — the model-free measurement above.
MHG-GNN went from 0.02 to 0.12. Both measurements are sound; they simply answer different questions.
The per-molecule stretch does destroy comparability between molecules. A Gaussian process with a
well-chosen width can still recover most of what it needs from what is left.

#### 🔴 NEW DEFECT — the pipeline never sets the width of its similarity function

**Verified 2026-08-26: the word `lengthscale` does not appear anywhere in `models/models.py`.**
Neither does `ard_num_dims`.

The width stays at the library default of about 0.7, for every representation, every dataset, every
run. Typical distances between molecules run from 17 to 1,100 depending on the representation. At a
width of 0.7 every molecule is effectively infinitely far from every other, the similarity matrix
becomes the identity, the likelihood surface is flat, and there is no gradient to follow. The fit
returns a single constant and scores whatever that constant happens to score.

That is what produced −0.0158 for MHG-GNN and +0.0087 for mol2vec in
`results/gp_kernel_harvest/qm9/`. Started from each dataset's own median distance instead, the same
damaged features reach 0.68 to 0.76.

**Three consequences.**

1. ✅ **Fixed, and the Gaussian process can now enter the variance decomposition.** §2.8f has the
   fix and the proof in the real pipeline. §10b.2's decision — one kernel, every representation,
   beside the support vector machine — is sound and better supported than before, because with a
   workable width the two kernels agree everywhere rather than only on binary features.
2. 🔴 **Every Gaussian-process number in the study is suspect, not only the two embeddings.** Whether
   a fit converged depended on how far that representation's typical distances sat from 0.7, and
   nothing recorded whether one had collapsed. **This is a re-run, not a fix**, and the re-run is
   what §13 is for.
3. ✅ **`gp_collapsed` is now written on every Gaussian-process row** (§5.2), so this cannot recur
   silently.

✅ **§2.8c has been corrected.** It called the storage defect "the largest single defect in the
study" and attributed the whole 0.89 gap to it. The storage defect is real, it damages the geometry,
and fixing it was right. It is not what produced those numbers.

**The broken first attempt is kept** at
`results/embedding_storage_retest/qm9_kernel_retest_BROKEN_FIT.csv`. Three scores — the ones you get
for predicting a single constant — came back sixteen times across four representations and both
storage schemes. That repetition across cells with nothing in common is the only reason it was
caught, which is failure mode 9 in §0.6 arriving for the third time in this project.

🔴 **Superseded — the model-fitting half was redone.** The first attempt was measuring its
own optimiser: three scores — the ones you get for predicting a single constant — came back sixteen
times across four different representations and both storage schemes. The width of the similarity
function starts near 0.7 in gpytorch while distances here run from 17 to 1,100, so every molecule
looked infinitely far from every other and there was no gradient to follow. Now started from each
dataset's own median distance, with the property values scaled for fitting, and with any fit whose
predictions are near-constant reported as collapsed rather than as a score. The broken run is kept at
`results/embedding_storage_retest/qm9_kernel_retest_BROKEN_FIT.csv` as the evidence for that
diagnosis.

**This is the third time in this project that a fitting or scoring routine failed quietly and
produced a plausible number** (§0.6 failure mode 9). It failed loudly enough to catch only because
the same value repeated across cells that had nothing in common.

**Confirming it at full size on the cluster.** The local measurement is at a few thousand molecules;
the harvest is at ten thousand, so only the paired difference transfers. These reproduce the harvest
cell for cell, post-fix. Rebuild the binary first — the record widened, so an old binary reads every
field after the embedding at the wrong offset.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models     # confirm the live checkout first (§2.8b)
git pull
cd rust && cargo build --release && cd ../scripts

for rep in mhggnn chemberta avalon continuous_pdv ecfp4 sns; do
  sbatch --account=stat-cadd --job-name=gp_$rep --time=47:00:00 \
    --output=../logs/gp_postfix_$rep.out --wrap="
      cd \$SLURM_SUBMIT_DIR && python -u process_and_train.py -d QM9 -t homo_lumo_gap \
        -m gauche --kernel rbf -u True -r $rep \
        -n 10000 -b 10 -s scaffold --normalize True \
        --noise-level 0.0 --noise-shape gaussian --noise-targeting uniform \
        -f ../results/gp_kernel_postfix/anova_gaussian_${rep}_gauche_rbf.csv"
done
```

Then compare against `results/gp_kernel_harvest/qm9/`, zero-noise rows only, paired on the replicate
index. **Do not treat the six noise-type files there as six independent samples** — at zero noise
they replay the same run (§10b.2, constraint 1).

**Two things this chat found that belong to other chats.**

- ✅ **`morgan` is deleted.** It was wired in Rust — a buffer, a read and a write — and absent from
  Python, so asking for it read every later field at the wrong offset. **The author's instruction,
  2026-08-26: *"don't trust the morgan from rust - in fact get rid of it"*.** Removed from
  `rust/src/main.rs` entirely; the binary rebuilds clean and chat D's record-alignment gate passes.
  This changes what §5.6 recommends: that section proposed reinstating a Python-computed Morgan
  fingerprint to fix the ECFP4 misnaming, and the Rust buffer it would have fed no longer exists.
  **The misnaming itself is still open** — QM9's `ecfp4` is a path-based fingerprint and
  `paper.tex:203` calls it circular — and so is the other half of §5.6, collapsing the two
  descriptor-vector names. Neither is assigned to a chat.
- **This botorch cannot fit the pipeline's Gaussian process.** `fit_gpytorch_mll` refuses a bare
  gpytorch model in botorch 0.16 (*'Gauche' object has no attribute 'transform_inputs'*), which is
  what `models/models.py` calls. Whether the cluster interpreter has an older botorch decides
  whether that model can run at all — an environment question, chat D (§2.8d).

**Does:** removes the per-molecule rescaling from all three learned embeddings, stores them as
32-bit floats, adds per-feature standardisation before the model, then re-tests the radial-basis
Gaussian process on the embeddings.

**The defect, verified in source 2026-08-26.** Each molecule's embedding is stretched so that
*that molecule's own* smallest value becomes 0 and its largest becomes 255. Every molecule gets a
different stretch factor, so distance between two molecules is meaningless — which is exactly what
the radial-basis kernel uses. `process_and_train.py:971-975` (mol2vec), `:828-831` (MHG-GNN), and
**`:807-813` (ChemBERTa) — a third representation, not previously recorded.**

**Three changes, and the radial basis needs all three** (§2.8c): stop the per-molecule rescaling;
store float32; standardise per feature across the training set. Only PDV is
standardised today (`:1800-1809`), and it is the best-performing cell in the study.

**Then re-test.** Whether the rescaling defect is what breaks the radial basis on the two learned
embeddings has to be measured after the fix, not assumed. That is the deliverable of this chat.

✅ **It was measured, and the answer was no** — see the top of this chat's entry and §2.8f. The
prompt below is kept as the record of what was asked for; **do not re-run it.** It names mol2vec,
which no longer exists, and it assumes the storage defect is the cause, which it is not.

> **Prompt (historical — superseded).** Fix the learned-embedding storage defect in `scripts/process_and_train.py` and re-test
> the Gaussian process afterwards. The defect is documented in `RERUN_PLAN.md` §2.8c: each molecule's
> embedding vector is min-max rescaled using that molecule's own minimum and maximum before being
> stored as bytes, which destroys comparability between molecules. It affects three representations,
> not two — mol2vec at `:971-975`, MHG-GNN at `:828-834`, and ChemBERTa at `:807-813`. Read all three
> and the storage path they feed before changing anything. Three things must change together: stop
> the per-molecule rescaling, store 32-bit floats rather than bytes, and standardise per feature
> across the training set before the model sees it — the last one matters because the kernel uses a
> single shared lengthscale across all dimensions (`models/models.py:1726`), so unstandardised
> features let the widest few dominate. Then re-run the kernel comparison locally on the two
> embeddings and report whether the radial basis recovers. `results/gp_kernel_harvest/qm9/` holds the
> pre-fix numbers to compare against; the paired zero-noise values are in `RERUN_PLAN.md` §10b.2. Do
> not touch `paper.tex`.

---

#### Chat D ✅ DONE 2026-08-26 — Infrastructure: the configuration race, the writer guards, the environment

⏸️ **The decisions this chat handed back are deferred with defaults — §13.10.** Nothing from it is waiting on the author.

All three defects are fixed and each is held by a check that fails if the fix is removed. Details
in §2.7, §2.8a and §2.8d; the gates are §8 items 8, 9b and 10.

**What was delivered**

1. **The configuration race (§2.8a).** `config_{file_no}.json`, passed to the binary as a
   **required** `--config` with no default, and removed with the memory-mapped files.
   Gate: `scripts/test_config_isolation.py`.
2. **The writer guards (§2.7).** All-or-nothing records; a failed fingerprint is written as 256
   zero bytes, listed in `featurisation_failures_{file_no}.csv`, and stops the run unless
   explicitly allowed. `read_train_labels` errors instead of truncating. `parse_mmap` raises with
   the entry and byte offset instead of guessing, and asserts the file was consumed exactly.
   Gates: `rust/tests/writer_guards.rs`, `scripts/test_record_alignment.py`.
3. **The environment (§2.8d).** `scripts/check_environment.py`, wired into the job template of
   **all three** job families (QM9 via `--models`, validation and uncertainty via
   `--validation-models`, which is the same probe over KIRBy's model names) and into runbook §1b.
   The runbook no longer diffs two cluster interpreters: `py311-kirby` is missing eight packages
   and the runbook now says not to use it, so there is one interpreter, `env_test`. The local
   `torch_geometric` import is fixed, so the QM9 pipeline runs on the laptop and every other chat
   can verify locally.

**Two things found that were not in the brief, and both were worse than what was**

- **An unparseable SMILES killed the process rather than taking the error branch.** RDKit's
  binding returns a null pointer as `Ok`, so the fingerprint call dereferenced null: SIGSEGV, no
  message, no partial output. The old `Err(_) => continue` branch was unreachable for the ordinary
  bad-SMILES case. Verified at exit 139 and fixed (§2.7 item 2).
- **`morgan` was written by the Rust writer and has never been read by the Python reader.** Any run
  including it was misaligned by 256 bytes per record. The reader now refuses an unknown
  representation by name, and `morgan` has since been deleted from the writer — it was a leftover
  of the `avalon` work (§2.7 item 5).

**Nothing is left with you.** The scikit-learn concern raised during this chat turned out to be
laptop-only — `env.yml` pins `scikit-learn=1.6.1` and the cluster has it. `env_test` was verified
on a compute node on 2026-08-26: 21 of the 22 job-generator model labels build, and NGBoost and
the quantile forest also fit. The one real failure is `conformal` (a `torchsort` ABI mismatch),
and it blocks nothing — the conformal variants are in `EXCLUDED_MODELS`, off by default, and
dropped from every figure (§2.8d).

**Scope note, corrected on the close-out pass.** The dead activation lines were **not** confined
to the QM9 generator — the same two lines open every job-script family in the repository. The
three live generators and the uncertainty preflight are all fixed:
`slurm_scripts_qm9_rerun/generate_scripts.py`,
`slurm_scripts_uncertainty_rerun/generate_scripts.py`,
`slurm_scripts_validation_rerun/generate_scripts.py`, and
`slurm_scripts_uncertainty_rerun/preflight.sh`. The historical directories
(`slurm_scripts_mol2vec`, `slurm_scripts_vbll`, `slurm_scripts_gauche_rbf`, and about twenty
more) still carry them and are deliberately left alone: they are superseded runs, and nothing in
them should be submitted again.

Two of those historical scripts also carry the wrong hERG dataset name —
`slurm_scripts_fixup/val_ngboost_herg.sh:22` and `slurm_scripts_remaining/val_ngboost_herg.sh:22`
pass `--datasets herg`, which the runner now rejects at argument parsing. Left as they are, for
the same reason: they are superseded. Noted so that anyone tempted to resubmit from those
directories knows they will exit at the first line, and knows the live copies are in
`slurm_scripts_validation_rerun/`.

✅ **Corrected 2026-08-27 — this warning is out of date and the directory it names has no `.sh`
files.** The 22 stale scripts under `slurm_scripts_qm9_rerun/` were deleted on the close-out pass
(recoverable at `d791ece`) and the generator has since been rewritten: it emits `--noise-level`,
`--noise-shape` and `--noise-targeting`, reads its level grid and its conditions from
`noise_conditions.json`, and carries the activation guard and the model-buildability probe. It
regenerates 14 working scripts. The original warning stands only as history: the scripts that
were there did predate the fix and did carry the retired CLI.

**One side effect, disclosed — and its consequence, found and fixed on 2026-08-27.**
`slurm_scripts_validation_rerun/generate_scripts.py` ignores `--help` and writes its scripts
immediately, into its own directory. Probing it therefore regenerated them.

The original note here said only two of those files were tracked in git and the other 85 never
had been. **That was wrong: all of them are tracked** (commit `ea54082`, "Commit the working
tree"). Believing otherwise is what made the next sentence look harmless. Three scripts were left
un-regenerated to preserve hand edits — `val_svm_pdv_herg.sh`, `val_svm_sns_herg.sh` and
`smoke_test.sh` — and all three kept the dead `MAMBA_EXE` lines with no activation guard, while
looking as sanctioned as their 85 siblings.

Fixed 2026-08-27. The hand edit those two scripts were protecting turned out to be **load-bearing
and the right side of a real defect**: they passed `--datasets herg_ki`, and the generator emitted
`--datasets herg`, which `alternative_data_noise_robustness.py` rejects at argument parsing
(`choices=['logd', 'caco2', 'herg_ki', 'all']`). Every one of the 28 other hERG scripts would have
died at argparse. The generator now carries a two-column table: `herg_ki` on the command line,
`herg` in every path, because the runner writes its output to `Path(results_root) / 'herg'` and
`merge_results.py` matches on that suffix. The smoke test is generated from the same template
rather than kept by hand — a hand-written file does not get regenerated, which is how it kept the
dead hook. All 88 generated scripts now pass `bash -n`, carry the activation guard, carry the
model-buildability probe, and none contains a live `MAMBA_EXE` line.

**The guard checks the right thing, not a hardcoded path.** The first version keyed on
`/apps/system/*`, which is brittle — it would have passed a task that activated the *wrong*
environment. It now asserts that `command -v python` resolves **inside `$CONDA_PREFIX`**, and
keeps the system-Anaconda text as an extra hint when that is where it landed. Driven through four
cases (no activation, system Anaconda, a different environment, correct activation) and it exits
2, 2, 2, 0.

---

#### Chat E — Cross-pipeline parity ✅ DONE 2026-08-26

⏸️ **The decisions this chat handed back are deferred with defaults — §13.10.**

**Delivered.** Parity is now structural rather than checked: `models/model_defaults.py` is the one
copy of every shared default and **both pipelines import it** (§3.4.5). One edit moves both studies
— demonstrated by the forest change, after which both reported the new hash `67f1b96564b9` from a
single line.

| | Executed check | Result |
|---|---|---|
| parity | both pipelines resolve one spec | ✅ same hash, at import **and** in the results files they write |
| the fix cannot be silently removed | `--self-test` | ✅ exits 1 on a changed learning rate and on a changed tree count, restores the file |
| library drift | effective parameters vs a recorded baseline | ✅ 0 differences |
| QM9 end to end | forest and neural, provenance columns | ✅ |
| experimental end to end | hERG, 10 rows | ✅ same spec hash as QM9 |
| neural change | a real run | ✅ stopped at epoch 23, **restored epoch 13** |

**Six differences fixed that the hand audit never listed** (§3.4.4c): batch size 64 vs 32; QM9
returning the last epoch; an improvement test comparing a *summed* validation loss against an
absolute tolerance; a Gaussian-process `outputscale` computed and never applied; NGBoost's
distribution and score passed on one side only; and a Gaussian-process fallback that existed on one
side and not the other, recorded nowhere.

**Two settings decided by measurement, not argument**, each against a rule fixed before the run:
forest feature sampling → `0.3` on both (§3.4.4e), and raw uncertainty as the primary column
(§3.4.4f). Both contradicted a prior recommendation; both are quoted from the CSVs in
`results/parity_tests/`.

**Found on the way, and the most consequential thing in this chat: §2.8e**, a segfault that kills
every Gaussian-process task on both pipelines wherever the boosting libraries share a process with
PyTorch. Not fixed in the pipelines — the right fix depends on the cluster — but it is now
detected by the preflight instead of being invisible.

**Handed to chat J:** name the uncertainty column the analysis reads, never pool rows with
different `gp_fit_method`, and carry `spec_hash` into captions.

**Still open, and not this chat's:** the three representation-identity defects in §3.4.1. No
parameter diff can see them; they are printed at the end of every audit run so they cannot be
forgotten.

<details><summary>Original brief</summary>

🔴 **The author reports this is being worked on in another session. Re-assess its state before
starting; do not duplicate it.**

**Does:** brings the two pipelines into agreement where they silently differ, and makes the
agreement enforced rather than assumed.

**What was found (§3.4):** XGBoost pins its learning rate to 0.1 on QM9 and leaves it unset on the
experimental side, where it falls through to 0.3 — a threefold difference in step size at the same
number of rounds. The quantile forest is 300 trees on one side and 100 on the other. The Gaussian
process is capped at 2,000 training molecules on one side (`GP_MAX_N`,
`alternative_data_noise_robustness.py:90`) and uncapped on the other. Every model on the
experimental side is pinned to `random_state=42` with no repeat loop.

**Why it is not cosmetic:** the paper's "boosting is fragile on small datasets" finding, and the
"model choice matters more on small real data" story built on it, are both candidates for being an
artefact of that learning rate. That has to be ruled out before either is repeated.

**Already written:** `scripts/audit_pipeline_parity.py`, which builds each model both ways against
the installed libraries and diffs the resulting parameters. ⚠️ Its literal values were transcribed
by hand and were still being verified when the last session ended.

**🔴 TODO:** the three alignment choices in §4 decision 3b — quantile forest tree count, whether to
remove the Gaussian-process cap, and repeat seeds on the experimental side.

> **Prompt.** Bring the two training pipelines into agreement, and make the agreement enforced rather
> than assumed. The differences found so far are in `RERUN_PLAN.md` §3.4 — read that, then read both
> pipelines yourself, because that list was built by hand and is not guaranteed complete. **Check
> first whether another session has already done part of this work** and continue it rather than
> duplicating it. There is a script at `scripts/audit_pipeline_parity.py` that builds each model both
> ways against the installed libraries and diffs the resulting parameters; its literal values were
> transcribed by hand and were never verified, so verify them against both sources before trusting
> it. Fix the differences that are plainly wrong: pin the boosting model's learning rate explicitly
> on the experimental side rather than letting it fall through to a library default three times
> larger, and make a requested model that cannot be constructed a hard failure rather than a skipped
> warning. Three alignment choices are the author's and are listed in §4 decision 3b — ask once,
> recommend, and do everything else meanwhile. Finish by wiring the audit into the preflight so a
> drifted parameter or a missing package stops the queue. Do not touch `paper.tex`.

</details>

---

#### Chat L — Work through all 151 audit candidates ✅ DONE 2026-08-27 (see §2.10b–§2.14)

**Done.** All 151 carry a verdict: 75 real and fixed, 14 real and still yours to decide, 14
duplicates, 4 partly fixed, 2 refuted, 2 not faults. `unverified.json` is empty and
`verdicts.json` holds the evidence for each. The tally, the fourteen checks and the fourteen
open decisions are in §2.14. The findings themselves are §2.10b–§2.13e.

**Where everything is.** `research_archive/audit_2026_08_26/`:

| file | what it holds |
|---|---|
| `unverified.json` | **111 candidates nobody has looked at.** 79 non-cosmetic, of which **8 are top severity**; 32 cosmetic |
| `confirmed_35.json` | the 40 that were checked, minus the 5 disproved — with each reviewer's reasoning |
| `refuted_5.json` | the 5 disproved, kept so they are not re-raised |
| `synthesis.md` | the run's own summary |

**Three are already fixed and need no further work:** the scrambled substructure fingerprint
(§2.10), the neural-network setting that matched no branch, and the feature-scaling divergence
(§3.4.3a).

**Start here.** One of the eight unchecked top-severity candidates is *another scrambling fault* of
the same class as §2.10 — the QM9 graph models are said to index the unshuffled dataset with
indices computed on the shuffled one. If true, every graph model has been trained on molecules that
do not match their labels. Check that one first.

**Read the existing verdicts sceptically.** Every reviewer returned "certain", with no gradation at
all, which is not what honest checking looks like. The surviving list contains plain duplicates —
three entries for one fingerprint fault, two for one graph-model error — so the true count is lower
than 35. And the audit ran against a tree that was changing underneath it, so at least one finding
was marked disproved only because its fix had landed mid-run; the feature-scaling fault is real
despite being marked disproved.

> **Prompt.** Finish the audit of both training pipelines. Every candidate in
> `research_archive/audit_2026_08_26/unverified.json` needs a verdict — all 111, including the 32
> marked cosmetic, because the author has asked for all of them. `confirmed_35.json` and
> `refuted_5.json` hold the ones already judged; re-check those too rather than trusting them,
> because every reviewer claimed certainty, the list contains duplicates, and one finding was marked
> disproved only because its fix landed while the reviewer was reading.
>
> Check each one by **reading the line and then running something that proves it**. This project has
> repeatedly recorded faults as verified-correct on the strength of a reading, and been wrong: the
> scrambled fingerprint had been examined and passed by an earlier session while 99% of its training
> rows carried another molecule's features. One line of execution settled it. Prefer a check that
> executes over an argument about what the code says.
>
> Start with the eight rated top severity, and within those with the claim that the QM9 graph models
> index the unshuffled dataset with shuffled indices — that is the same class of fault as the one
> already found and would void every graph-model result.
>
> Fix what is real, in both pipelines where it applies, and give each fix a check that fails if the
> fix is removed. Anything that is a genuine decision for the author: ask once, recommend, and get on
> with everything else. Record what you find in `RERUN_PLAN.md` and delete the entries you have
> closed from `unverified.json` so the file always shows what is left. Do not touch `paper.tex`.

---

#### Chat M — Three loose ends: the cluster, sparse counts, and the QM9 job scripts

✅ **TWO OF THREE DONE 2026-08-27. The third needs you, and it is one command.**

**Sparse counts — measured, answered, and the storage defect it was waiting on is fixed too.**
The question was what happens to the substructure fingerprint when its counts stop being flattened
to presence bits. The answer is that it must NOT be standardised (§3.4.3a: QM9 loses 0.073 R² on
every one of five seeds), and the rule in `models/model_defaults.py` now says so as a rule about
sparse features rather than binary ones. Spec version 1.2.0.

While measuring it, the thing it was waiting for turned out to be small enough to finish here, so
it is finished: the record holds **1,024 counts as 16-bit integers** instead of 128 bytes of packed
bits, on both sides of the record — `write_to_mmap`, `parse_mmap` and the Rust reader's buffer
width, which are the three that have to move together. That closes both halves of the §3.4.1 SNS
row, including the wrap where a count of exactly 256 recorded as *absent*; the writer now refuses a
count that will not fit rather than wrapping it. Proven on a real run: 320 training molecules,
counts up to 7, 15.9% of present substructures appearing more than once, and the scaling rule
correctly declining to standardise them. Guards: `scripts/test_embedding_storage.py` §6–§7 and
`python scripts/parity_test_count_scaling.py --self-test`, both of which run the retired storage
through the assertion and require it to fail.

**The QM9 job generator — rebuilt, and it now has a test that executes its output.** It emitted
`--sigma`, `--noise-strategy`, `-b`, and two representations refused by name, so every script it
wrote would have died at argument parsing. What it emits now: the new noise CLI, the settled six
representations, the staged design (§13.1), and `--bayesian-transformation last_layer` rather than
the `last` that falls through every branch and files a plain network as a Bayesian one.

It no longer decides which conditions exist — it **reads `noise_conditions.json`** and only
translates a condition name into flags, which is the one job that genuinely belongs to it. A name
there that it cannot produce, or one it can produce that nobody decided to run, stops it at import.

`slurm_scripts_qm9_rerun/test_generate_scripts.py` is what stops this repeating. It generates every
script for stages 0, 1 and 2, **runs each array task** against a stub checkout that satisfies every
guard, and feeds the command line each one builds through `process_and_train.py`'s own parser —
1,185 task runs and 1,180 command lines. It also fires each guard in turn (no partition, index out
of range, no environment, wrong environment, missing binary) and requires each to refuse, and
`--end-to-end` runs one real training task at 400 molecules. The runbook and the completeness check
follow from the generator rather than restating it.

**🔴 The cluster — submitted 2026-08-27 as job `12914447`, waiting on the report.** A login node
cannot answer it: the audit imports torch, and the per-user memory cap there makes the
model-construction checks come back inconclusive. Submitted as a batch job instead — 4 cores, 16 GB,
25 minutes, `--account=stat-cadd --partition=short` — because an interactive session was not being
scheduled. Check it with:

```bash
squeue -j 12914447
sacct -j 12914447 --format=JobID,JobName%16,State,Elapsed,ExitCode
cat ~/server_audit_report.txt
```

**When that report exists, everything below is answered and this item closes.** The original
one-command form, for reference:

```bash
ssh <arc>
cd /data/stat-cadd/scat9264/qsar_qm_models && git pull
bash scripts/server_audit.sh
# then send back ~/server_audit_report.txt
```

It answers, in one file: which checkout is live, whether the two interpreters agree, whether every
model can be constructed, whether the quantile forest fits, whether the Gaussian process still
crashes with the boosting libraries loaded, whether the two pipelines report the same spec hash,
whether the Rust binary is built and carries the new CLI, and — added here — whether every command
line the job generator emits is accepted by the pipeline **as installed on the cluster**. That last
one is the check that would have caught the stale generator months ago.

Until that file comes back, §2.8e stays open: on this laptop the Gaussian process still segfaults
once the boosting libraries are loaded, and whether the cluster's environment has the same single
OpenMP runtime problem cannot be established from here.

<details>
<summary>The original brief for this chat</summary>


Three unrelated items, grouped because each is small and none belongs to another chat.

**1. The cluster checks — needs the author, one command.** Nothing here can be answered from a
laptop. `scripts/server_audit.sh` asks all of it in one run and writes one file.

**2. Sparse counts under scaling — unmeasured, and it goes live the moment the count storage is
fixed.** Whether features are scaled is decided from the matrix itself: anything whose values are
all 0 or 1 is left alone, everything else is scaled (`models/model_defaults.py`,
`should_standardise`). The substructure fingerprint is stored as presence bits today, so it reads
as binary and is left alone. When chat A restores real counts it stops being binary and this rule
will begin scaling it — and sparse counts may suffer the same harm as sparse bits, which was
measured at 0.077 on hERG and about 0.6 on QM9 for a support vector machine with a radial kernel.
Nobody has measured the counts case.

**3. The QM9 job scripts cannot start.** The generated scripts are not kept in version control, and
the local copies have been deleted. **The generator that makes them is still stale:**
`slurm_scripts_qm9_rerun/generate_scripts.py` emits `--sigma` and `--noise-strategy` at `:245-246`,
both of which `scripts/process_and_train.py:421-427` refuses by name, and it still lists the old six
noise names at `:90` when the program now accepts uniform, grouped_wide, grouped_shift, outlier and
censoring. Regenerating from it reproduces the breakage.

> **Prompt.** Three loose ends, none of which belongs to another chat.
>
> First, the cluster. Ask the author to run `scripts/server_audit.sh` on the cluster and send back
> `~/server_audit_report.txt` — it takes about five minutes and answers every question that cannot be
> answered from a laptop: which checkout is live, whether the two interpreters match, whether every
> model can be built, whether the quantile forest fits, whether the Gaussian process still crashes
> now that the thread guard is in, and whether the two pipelines agree. Read the file, act on what it
> says, and record the answers in `RERUN_PLAN.md` so those questions are closed rather than open.
>
> Second, measure whether sparse counts should be scaled. Today the substructure fingerprint is
> stored as presence bits, so it reads as binary and is left unscaled; when chat A restores real
> counts the rule in `models/model_defaults.py` will start scaling it, and nobody knows whether that
> helps or hurts. Measure it the same way the binary case was measured — a support vector machine
> with a radial kernel, on hERG and on QM9, scaled against unscaled, with the decision rule written
> down before the run. Then set the rule accordingly and say what it cost.
>
> Third, make the QM9 job scripts runnable. The generator at
> `slurm_scripts_qm9_rerun/generate_scripts.py` still emits settings the program refuses by name and
> still lists retired noise names. Fix the generator, not the generated files — they are not kept in
> version control and are regenerated from it. Then prove it: generate one script and run a single
> task end to end at a small molecule count. A generator that produces a script nobody has executed
> is not finished.
>
> Do not touch `paper.tex`.

</details>

---

#### Chat F — Uncertainty machinery, reviewed with the author

✅ **DONE 2026-08-27.** The four decisions were settled by the author on 2026-08-26 and the work is
in the tree. What was fixed and what it is checked by is in §3.1a, §3.1b and §3.6. The decisions:
report both question-4 numbers under names that cannot be confused; run the uncertainty question on
QM9 as well as the three experimental sets (*"QM9 is the core results of the paper because it is
the only confirmed clean dataset. Make the change in QM9 that is non-negotiable"*); validation
labels carry their own noise, drawn independently; and the figure script may be modified.

The original request this chat answered: *"I am not confident that uncertainty machinery is built
and reviewed — I've had repeated issues with this and we need to go over it 1:1 before the plan is
finalized."*

**What to go over, in order:**

1. **The within-noise-level fix (§3.5).** It is the author's, it is implemented, and it works. Start
   here so the review begins from what is right.
2. **What actually feeds it.** On QM9 the injected noise is reconstructed rather than recorded, and
   after the held-out fix that reconstruction is identically zero (§2.6). The fix is correct and its
   input is dead.
3. **The three questions (§7.0, Q4–Q6)** — agreed 2026-08-23, and the third is in neither the run
   plan nor the runbook's analysis section.
4. **The nine defects the KIRBy review found and fixed** — walk them, because the author has not
   seen them and the regression test for several of them works by searching the source for a
   matching string, which passes whether or not the line ever runs.
5. **One live bug (§3.1a):** the placebo check never fires. Its guard at
   `alternative_data_noise_robustness.py:941` tests a dictionary key that is not written until
   `:954`, thirteen lines later, so it is false on every pass and every training row writes a blank.
6. **Two silent no-ops (§3.1b).**
7. 🔴 **Decision: does the uncertainty question run on QM9, or only on the three experimental
   datasets?** (§4 decision 1.)
8. 🔴 **Decision: do validation labels carry their own noise?** (§4 decision 2, restated in §13.5.)

**This chat has real work in it, not just discussion.** Items 5 and 6 are unambiguous bugs and get
fixed in the same session. Items 1–4 are the material to bring to the author. Items 7 and 8 are the
author's calls.

> **Prompt.** Audit the uncertainty machinery end to end and fix what is unambiguously broken, then
> report the rest for a decision. The author has had repeated trouble with this part of the project
> and does not trust its current state, so **read the code rather than the summaries** — in this
> repository and in the KIRBy repository, which is where the machinery actually lives.
>
> Start from what is right, because it matters that it is not re-litigated: the uncertainty-versus-
> noise correlation is computed within each noise level rather than pooled across levels, that was
> the author's own fix, and it works. `RERUN_PLAN.md` §3.5 says where it is. Verify it still does what
> its docstring claims.
>
> Then establish, from the code, four things and write each into `RERUN_PLAN.md`: what actually feeds
> that correlation on each of the two pipelines and whether it is real data or a reconstruction; what
> the three uncertainty questions are and whether the run plan and the runbook can answer all three
> (§7.0, Q4–Q6 — one of them is currently in neither); whether each of the nine defects the earlier
> review claims to have fixed is genuinely fixed *in behaviour*, not just present as a matching line
> in the source, because the regression test for several of them works by searching the file rather
> than running it; and whether the confound control does what it is meant to.
>
> Fix these two outright, with a check that fails if the fix is removed. First, the placebo check on
> training rows never fires — its guard tests a dictionary key that is not written until thirteen
> lines later, so it is false on every pass and every training row writes a blank. It appears twice,
> in both runners. Second, the two silent no-ops in the same writer, described in §3.1b: a run that
> omits the zero-noise level writes no uncertainty at all without complaining, and only one noise
> condition is written unless a flag is passed.
>
> Two decisions belong to the author and are in §4, decisions 1 and 2. Ask once, give a
> recommendation, and do everything that does not depend on the answer meanwhile. Do not touch
> `paper.tex`.

---

#### Chat G — Local test: which noise settings earn their place

**Does:** a short local run, on real QM9 at reduced size, to decide how many settings of each noise
type are worth cluster time — and whether a skewed condition is needed (§13.3).

**Why:** `NOISE_DESIGN.md` §2 lists Student-t at three tail weights and Outlier at three
contamination fractions. Run every setting and there are nine conditions, not six — the largest
single multiplier on the grid, larger than the replicate count.

⏸️ **The decisions this chat handed back are deferred with defaults — §13.10.** Its 0.5 reporting-level suggestion was withdrawn by chat D and the level is now decided from the main grid.

**✅ CLOSED 2026-08-27.** The harness is `scripts/setting_selection_test.py`, the run is
`results/setting_selection_test.csv`, and the answer with the settled set is §13.9. Everything below
is what the test found on the way.

##### What this test does that the earlier pilots did not

- **The replicate spread is the real one.** `scripts/pilot_noise_arms.py` and
  `scripts/noise_range_finding.py` both hold the molecule subsample, the scaffold split and the
  model seeds fixed across replicates, so their only source of variation is the noise draw. The real
  pipeline re-derives all of it per replicate — `iteration_seed` (`process_and_train.py:1871`) seeds
  a `torch.randperm` shuffle inside `split_qm9` (`:609-631`). The decision rule here is "the effect
  must clear the replicate spread", so it was being run against the wrong denominator. Every
  replicate now draws a fresh subsample and a fresh split.
- **It screens the models the cluster will run.** The two tree models take their settings from
  `models/model_defaults.py`, the file both pipelines now read — 100 trees at `max_features = 0.3`,
  not the pilots' 150 trees at scikit-learn's every-feature default.
- **It reports what it could have detected.** Every contrast carries the smallest true difference it
  would find at 80% power. The differences at stake are known to be under 0.02 R², so a null result
  without the test's own resolution beside it says nothing.
- **The accuracy floor is declared, not silent** — guard 8 of §0.6. Every verdict is computed with
  and without it, and the run says so if the two disagree.

##### 🔴 Two launch gates in the Rust injector failed at random — found here, fixed

Running the new harness's labels through `rust_processor --self-test` (3,200 real QM9 training
labels, real scaffold groups) failed two of chat A's gates on data where the injector is in fact
correct. Both were under-powered rather than wrong, and both are now fixed and committed:

| Gate | Why it failed | Fix |
|---|---|---|
| Flat dose across noise types | Averaged **20 seeds**. Per-run dose spread is 1.3% for Gaussian but **3.9% for grouped-shifted and 6.9% for Student-t ν = 3**, so their means wander by ±0.9% and ±1.5% and the 3% spread criterion is breached by sampling noise alone. It reported 3.39% and failed, on labels where 400 seeds put grouped-shifted at **+0.03% ± 0.19%** — exactly on target | 200 seeds. Same labels now give a 1.29% spread, and every condition passes |
| Student-t nests Gaussian at ν → ∞ | Compared **one** Gaussian draw against **one** t draw. Two independent draws wobble by ~1.8% against a 2% threshold, so it fails about a quarter of the time; it reported +2.21% | Averaged over 50 seeds: mean dose gap +0.38%, tail fractions 0.29% vs 0.28% |

**A launch gate that fails at random is worse than no gate, because the next person turns it off.**
Both fixes are in `rust/src/main.rs`; `cargo test --release` passes 28 + 5 tests.

##### 🔴 Rule 2 of `NOISE_DESIGN.md` §2a applies to the SPLIT, not only to the noise

The empty Murcko scaffold is a third of QM9. The rule that acyclic molecules become singleton groups
was written for the noise; it matters just as much for the split, because a splitter that treats them
as one group can put that whole chemical class on one side. Measured here, on the same subsample:
clean R² of **0.83 / 0.79 / −0.40** (LightGBM / random forest / ridge) against the **0.92 / 0.92 /
0.90** the range-finding run reports. Splitting on the singleton-corrected grouping restores it and
holds the largest group under 4% of molecules.

⚠️ **And a related finding the real run has to expect.** Even with that corrected, one replicate in
twelve handed ridge a scaffold split it could not fit at all — **clean R² = −16.99**. That is not a
bug; it is what a linear model does on a genuinely disjoint scaffold split. It is also exactly the
situation guard 8 exists for, and it is a live risk for the catastrophic-run filter in the real
analysis: a filter that silently deletes such replicates biases unstable configurations upward.

##### An inconsistency in §13.1's cost table, found while pricing the options

The stage 0 row is priced at **1,482 runs**, which is 78 × 19 × 1 — the *same nineteen*
level-conditions as stage 1, i.e. three noise types plus the clean level. Its description says
"Gaussian and censoring", which is thirteen conditions and **1,014 runs**. The pricing is the
coherent one, because stage 0 is reused as replicate 0 of stage 1 and that reuse only works if it
runs the same conditions; the description is what is wrong. This bears directly on open item 3 —
which noise types run at full grid in stage 1.

The 78 combinations are 13 models × 6 representations. `slurm_scripts_qm9_rerun/RUNBOOK.md` says 14
models (11 for the decomposition, plus the quantile forest and *two* Gaussian processes) — that
count predates §10b.2, which keeps Tanimoto as **evidence on two fingerprint representations rather
than a second model in the roster**. 13 is right; the runbook line is stale.

> **Prompt.** Run a short local test on real QM9 to decide how many settings of each noise type earn
> cluster time. The candidate settings are in `NOISE_DESIGN.md` §2: Student-t at ν = 10, 5 and 3, and
> Outlier at 1%, 5% and 10%. Existing local scripts to build on: `scripts/pilot_noise_arms.py`,
> `scripts/test_noise_arms.py`, `scripts/noise_range_finding.py`. Report, per setting: whether it is
> distinguishable from Gaussian in accuracy at matched delivered amount, and by how much relative to
> the replicate-to-replicate spread — a difference smaller than the run-to-run wobble is not a
> setting worth running. Also test whether a **skewed** noise condition is distinguishable from a
> symmetric one at matched amount; see `RERUN_PLAN.md` §13.3 for why this is open. Report the compute
> each option costs against the table in §13.1. Recommend, do not decide. Do not touch `paper.tex`.

---

### 13.9 ✅ ANSWERED 2026-08-26 — which settings earn cluster time

`scripts/setting_selection_test.py`, results in `results/setting_selection_test.csv` and
`results/setting_selection_test_contrasts.csv`. Real QM9, 4,000 molecules per replicate drawn fresh,
PDV descriptors, real Murcko scaffold split, noise on training labels only, scored on clean test
labels. **Twelve replicates**, two levels — 0.5 (the proposed QM9 reporting level) and 1.5 (the top
of the grid, and per `NOISE_DESIGN.md` §5.5 the only place the noise types separate at all). Three
models on the pipeline's own defaults. 792 rows. Every condition delivers the same amount of noise:
realised 0.490–0.505 at level 0.5 and 1.475–1.522 at level 1.5, every one within three standard
errors of target.

⚠️ **CORRECTED 2026-08-27 — every headline number below was computed WITH Ridge, and Ridge is not
in the study.** Found by the chat D/G close-out audit. §4 of this document got the Ridge correction
on 2026-08-27; this section did not, so the two disagreed. Recomputed from
`results/setting_selection_test.csv` with the declared accuracy floor applied:

| Quantity as stated below | With Ridge (as written) | Roster models only |
|---|---|---|
| Largest \|mean ΔR²\| against Gaussian at the reporting level, excluding grouped-shifted | 0.0058 — a **Laplace / Ridge** row | **0.0047** — Outlier p=0.01 / random forest |
| Same across both levels | 0.0475 — Student-t ν=3 / **Ridge** | **0.0445** — skewed draw / random forest |
| Laplace, largest \|mean ΔR²\| | 0.0362 — **Ridge** | **0.0136** — LightGBM at level 1.5 |
| Grouped-shifted against grouped-wider at level 1.5 | 0.096 – 0.314, up to 7.4× the wobble | **0.0963 – 0.1419**, 2.52× and 2.71× the wobble |

⚠️ **Two of the right-hand figures were themselves wrong until 2026-08-27 and are corrected above.**
The skewed-draw row read 0.0408 and Laplace's read 0.0087; both were **exact-dose** rows, and the
primary analysis is the algebraic dose because that is what the pipeline will use
(`scripts/setting_selection_test.py` takes `dose_mode == 'algebraic'` as primary). Recomputed from
`results/setting_selection_test_contrasts.csv`, which is written on the primary analysis alone.

**Every verdict in this section stands and several are strengthened**, because removing Ridge makes
the differences *smaller*: the case that shape and contamination do not earn separate settings gets
better, and Laplace becomes more indistinguishable from Gaussian, not less. **But `noise_conditions.json`
cites this section as its evidence, so the numbers it points at had to be right.** The `why` strings
in that file quoting "largest 0.0058" and Laplace's "0.0058" were quoting Ridge rows. ✅ **All of
them were rewritten on 2026-08-27** from `results/setting_selection_test_contrasts.csv`, which is
written on the roster models and the primary dose alone.

✅ **The analysis script now drops non-roster models by name rather than silently including them**
(`scripts/setting_selection_test.py`, `models_in()`). `--analyse-only` prints
*"dropping Ridge — not in the study roster"* before any number. It also no longer dies when the
roster changes, which it did between the file being written and the roster being updated.

#### The answer, in one line

**Shape and contamination do not earn separate settings. Direction does.**

#### At the reporting level, everything except one condition is indistinguishable from Gaussian

Across all ten non-Gaussian conditions except grouped-shifted, on the two roster models, at twelve
replicates each (**corrected 2026-08-27** — the figures here were computed with Ridge, which is not
in the study; every one of them got smaller):

| | |
|---|---|
| Largest \|mean ΔR²\| against Gaussian | **0.0047** — Outlier p=0.01 / random forest |
| Largest ratio to the replicate-to-replicate wobble | **0.29** — the same row |
| Smallest paired *p* | 0.178 |
| What twelve replicates could have detected, **for the model carrying each effect** | 0.0068 – 0.0120 |

⚠️ **That last row is paired, and it was not before 2026-08-27.** It used to take the largest effect
across models and the smallest detection threshold across models — two different rows presented as
one comparison, on six of the ten conditions, and always in the flattering direction. Student-t
ν = 10 read *"largest 0.0017, could have seen 0.0064"* when the model carrying that 0.0017 could
only have seen **0.0119**, nearly double. Each effect is now quoted against its own model's
resolution; `scripts/setting_selection_test.py` reports the best-case threshold separately and says
so. **No verdict moves** — every effect is well under even its own model's threshold.

Every one of them is under a third of the run-to-run wobble. **This is not "we could not see it" —
the test's own resolution is stated beside every number, and the effects sit at or below it.**

**The ladders are flat.** ν = 10 → 5 → 3: every step under 0.0061 R², *p* ≥ 0.29. Outlier
1% → 5% → 10%: every step under 0.0049.

⚠️ **One statistically significant nothing, reported because it is the kind of number that gets
misread.** Outlier 5% versus 1% reaches *p* = 0.0024 (LightGBM) and *p* = 0.0014 (ridge) — on
differences of **+0.0049 and +0.0031 R²**, which are 0.36 and 0.14 of the wobble. Common random
numbers across conditions make the paired difference very precise, so a trivial difference can be
significant. It is also the *wrong sign* for a dose response — more contamination did marginally
better — so it is precision around zero, not an ordering.

#### Grouped-shifted is the exception, and it is large

| Contrast | Level | LightGBM | Random forest | Ridge |
|---|---|---|---|---|
| Grouped-shifted − Gaussian | 0.5 | −0.0100 (0.58×) | −0.0077 (0.47×) | **−0.0311 (1.55×, p = 0.017)** |
| Grouped-shifted − Gaussian | 1.5 | **−0.1274 (2.22×, p = 0.0008)** | **−0.1012 (2.98×, p = 0.0022)** | **−0.3300 (4.64×, p = 0.0002)** |
| **Grouped-shifted − Grouped-wider** | 1.5 | **−0.1419 (2.71×, p < 0.001)** | **−0.0963 (2.52×, p = 0.003)** | **−0.3138 (7.44×, p < 0.001)** |

The bracketed figure is the difference as a multiple of the replicate-to-replicate wobble.

**The third row is the one that matters.** The two grouped conditions differ *only* in whether the
group's error is centred — same amount of noise, same group structure, same targeting. So that
contrast isolates **direction**, and direction is worth 0.10 to 0.31 R² where every difference of
*shape* is worth under 0.006. **This is the study's censoring result reproduced by a second,
independent mechanism at the level of a chemical family rather than the whole dataset** — which is
precisely the argument §13.3 made for running both forms, now with evidence behind it.

#### The dose wobble was not driving any of it

Student-t ν = 3 has a per-run delivered-dose spread of 17% at level 1.5, which raises the obvious
question of whether the flat result is really a smeared one. A second pass at level 1.5 rescaled each
draw to *exactly* the target amount, removing that nuisance entirely, and the picture is unchanged:

| Condition (exact dose, level 1.5) | LightGBM | Random forest | Ridge |
|---|---|---|---|
| Student-t ν = 5 | +0.001 | −0.038 | −0.040 |
| Student-t ν = 3 | −0.010 | −0.023 | −0.048 |
| Laplace | −0.009 | −0.006 | −0.036 |
| Outlier p = 10% | −0.010 | −0.023 | −0.037 |
| **Grouped-shifted** | **−0.111 (2.13×, p = 0.002)** | **−0.094 (2.58×, p = 0.002)** | **−0.312 (4.38×, p < 0.001)** |

Every ratio to the wobble outside the last row is at or below 1.04, and every *p* is ≥ 0.074.

#### The skewed draw does not earn implementation — §13.3's rejection holds

Nothing at the reporting level (largest 0.0055, ratio 0.27). At level 1.5 it reaches −0.0445 for the
random forest alone (1.31× the wobble, *p* = 0.007) and nothing for the other two. A condition that
moves one model out of three, only at the top of the grid, does not repay being built into two
injectors and carried through the whole design. **Recommendation: do not implement it.** The
asymmetry story is carried by censoring and by grouped-shifted, both of which are mechanisms with
sources behind them rather than a chosen distribution.

#### Guard 8: the declared filter, and where it matters

One cell of thirty-six was excluded — replicate 2, ridge, **clean R² = −16.99**, a scaffold split a
linear model cannot fit at all. It changed exactly one verdict, and only through ridge: with the
filter, grouped-shifted earned full grid on the strength of a ridge row at the reporting level.
**Ridge is not in the study, so that verdict was never available anyway** — on the two roster models
grouped-shifted separates only above the reporting level, with or without the filter, and
`--analyse-only` now prints *"the declared filter changes no verdict"*. The recommendation is
unaffected either way, because grouped-shifted's case rests on level 1.5, where **both roster
models** agree at *p* ≤ 0.0031.

#### ✅ SETTLED 2026-08-27 — the author approved all five

These were put as recommendations and were approved as they stand. They are now in
`noise_conditions.json`, which tests on both sides read (§8), and in the two injectors' defaults.

| # | Recommendation | Why |
|---|---|---|
| 1 | **One Student-t setting, not three: ν = 5.** | The three are within 0.0036 R² of each other and of Gaussian, against a test that could have seen 0.0087 – 0.0120 on the model carrying each effect. ν = 10 is nearly Gaussian by construction; ν = 3's per-run delivered dose has a 17% spread at level 1.5, which makes it the worst-behaved thing on the grid to report. ν = 5 is mid-ladder and well-behaved |
| 2 | **One Outlier setting, not three: p = 10%.** | Same evidence. p = 10% is the top of Hampel's published range and the strongest contamination, so if anything is ever going to show, it shows there |
| 3 | **Both grouped conditions at full grid in stage 1** | The only zero-mean condition that separates, and its comparator is what makes it interpretable. The pair is a claim; neither half is |
| 4 | **Laplace: not in the main grid — the deep run only.** ✅ The author settled it there on 2026-08-27 | Indistinguishable from Gaussian on both roster models at both levels. The largest difference anywhere is **0.0136 R²** — LightGBM at level 1.5, against a replicate-to-replicate wobble of 0.0574, so **0.24 of the wobble**, paired *t* p = 0.350. At the reporting level it is 0.0024. The earlier "largest 0.0058" was a Ridge row. Its stated value in `NOISE_DESIGN.md` §2 is citational, not empirical, and the citation costs 720 runs |
| 5 | **Do not build the skewed draw** | See above. It also does not exist in either injector yet, so this is a saving rather than a deletion |

**This closes chat A's open TODO** — *"how many Student-t and Outlier settings"* — at one each.

**What implementing it touched.** `noise_conditions.json` (new, the settled set with its evidence);
the Rust injector's contaminated-fraction default, 0.05 → 0.10, and the same default in
`process_and_train.py`; three new tests in `rust/tests/noise_gates.rs` and a new
`scripts/test_noise_conditions.py`, all four of which were verified to fail when the set or a
parameter drifts. Job-script generation reads the settled set rather than restating it — that is
chat H's step, and §13.9 is what it reads.

#### What it costs, priced against §13.1

One extra setting at full grid in stage 1 is 78 combinations × 6 non-zero levels × 10 replicates =
**4,680 training runs, 9.1% of the old design**. In stage 2 only it is 12 × 6 × 10 = **720 runs,
1.4%**.

| Option | Stage 1 | Staged total | Share of the old design |
|---|---|---|---|
| §13.1 as agreed — 3 noise types (19 level-conditions) | 14,820 | 19,260 | 37% |
| **Recommended — 4 types: Gaussian, grouped-wider, grouped-shifted, censoring (25 level-conditions)** | **19,500** | **24,660** | **48%** |
| Every Student-t and Outlier setting at full grid as well (+4 settings) | 33,540 | 37,980 | 74% |

The recommendation costs 5,340 runs more than the currently agreed stage 1 and **13,320 fewer** than
carrying all nine settings. It buys the one contrast in the whole zero-mean set that produces an
effect, and it spends nothing on eight settings that produce none.

**🔴 This changes §13.1 open item 3.** The three structurally distinct types were named as Gaussian,
Grouped and Censoring. Grouped is **two** conditions, and the difference between them is the largest
zero-mean effect measured anywhere in this study — so it is four, not three.

#### The QM9 reporting level is still open (§13.1 item 5), and this does not reopen the answer

Everything above is stated "at the reporting level", and that level is **assumed to be 0.5** — the
standing suggestion in §6.1, not yet a decision. The QM9 grid has seven points
(`NOISE_DESIGN.md` §6.4 owns them), so a different choice is possible.

**It would not change the recommendation, and here is why rather than an assurance.** The test
measured 0.5 and 1.5, which **bracket every other point on the grid**, and both ends give the same
answer: grouped-shifted separates and nothing else does. The intermediate levels are therefore
bracketed rather than measured, which is weaker than measured and is said so here.

**One exception at the top end, so it is not overstated.** At level 1.5 the skewed draw reaches
−0.045 R² for the random forest at *p* = 0.007 — one model of three, at the highest level on the
grid, and nothing at all at 0.5. It does not survive as a reason to build the condition, but if the
reporting level were moved to 1.5 it would be worth re-reading that row before repeating "nothing but
direction matters".

#### What this test cannot settle, stated rather than buried

One representation and three cheap models on 4,000 molecules. A setting that matters for a neural
network or a Gaussian process would be missed. And it measures **accuracy only**: `NOISE_DESIGN.md`
§5.3 already notes that a model may lose the same accuracy while being much better or worse at
spotting *which* labels were corrupted, and that is where concentrated noise was expected to earn its
place. That question belongs to the uncertainty runs, and on QM9 it cannot be answered in the main
pipeline at all (§2.6).

---

#### Chat N — The uncertainty screen: which models and which representations

🔴 **OPEN. Nothing has run.** A first attempt on 2026-08-27 was cancelled by the author because it
was built to answer the wrong question. Read the next paragraph before anything else.

**THE ONLY THING THIS CHAT DECIDES.** Two lists at the top of
`slurm_scripts_uncertainty_rerun/generate_scripts.py`:

    MODELS   currently 7   QRF, NGBoost, GP, BNN-Full, VBLL-Full, MLP-BNN-Full, MLP-VBLL-Full
    REPS     currently 4   ECFP4, PDV, SNS, MHG-GNN   — Avalon and ChemBERTa are NOT in it

The uncertainty runs are 420 jobs. Those two lists are what makes them 420. A model whose predicted
uncertainty does not track injected noise is a job that produces a row nobody can use, and the
author has asked repeatedly which models and which representations are worth running. **That is the
question. It is not the noise types.**

**THE NOISE TYPES ARE SETTLED AND ARE NOT UNDER TEST.** `noise_conditions.json`, settled 2026-08-27,
gated by tests on three sides. The cancelled attempt ran five of them and reported per-noise-type
findings; that multiplied the cost fivefold and answered nothing that was open. **Use ONE noise
type — plain Gaussian — and spend everything on the two lists.**

**THE METRIC IS THE ONE THE PAPER ALREADY REPORTS.** Spearman correlation between predicted
per-molecule uncertainty and injected noise magnitude, computed **within one noise level**, scored
**out of fold**, per (model, representation). That is `paper.tex` `tab:top_unc_noise` — "Strongest
and weakest model–representation combinations for uncertainty–noise correlation". Alongside it the
paper reports the uncertainty-versus-absolute-error correlation and coverage at 1σ and 2σ. Report
those. Do not invent a statistic; `scripts/uncertainty_stats.py` already has every one of them.

**THE RUN.**

    QM9, -n 5000, 1 replicate, --oof-folds 3, -u True
    one noise type: --noise-shape gaussian --noise-targeting uniform
    two levels: --noise-level 0.0 1.5
    all six representations: ecfp4 continuous_pdv mhggnn avalon chemberta sns
    all seven uncertainty models

Run it locally in `env_test`, NOT the base Anaconda — the laptop's base has scikit-learn 1.3.2
against a 1.6.1 pin and every quantile-forest fit raises `Invalid parameter 'monotonic_cst'`.
`env_test` already exists on the laptop with the pinned versions. **Do not pip install anything
into base**: §2.8i is explicit that a PyPI wheel over a conda package is what puts four OpenMP
runtimes in one interpreter and segfaults LightGBM and the Gaussian process.

Anything large goes on `/Volumes/seagate`, the author's external drive.

**THE DELIVERABLE IS AN EDIT, NOT A FINDING.** One table — uncertainty-versus-noise correlation per
model and representation, one table per representation, nothing pooled, nothing averaged — and then
the two lists in the generator edited to match, committed, with the job count before and after.
**If a number does not change a line of that file, do not report it.**

**What the cancelled attempt got wrong, so it is not repeated:** it tested the settled noise types
instead of the open lists; it reported per-noise-type results the author had no decision to make
about; it dropped NGBoost — one of the author's most noise-robust models — without asking; and it
never once connected a number to a line of the generator. The author's words: *"Your refusal to
connect results to implementation is downright offensive."*

**Two defects it did find, both fixed and both still in force** — §3.1e (the out-of-fold pass asked
for validation rows no model fits, which had silently disabled training-molecule scoring for the
quantile forest, NGBoost and the Gaussian process) and the censoring name normalisation in
`scripts/uncertainty_stats.py`.

#### Chat H — Job scripts, preflight, gates, launch

**Blocked** on the run design in §13.1. A, B, C, D, E and G are all done, so that is the only
remaining block, and it applies to the QM9 half only.

🟡 **The uncertainty half is already done, 2026-08-27 (§2.8j).**
`slurm_scripts_uncertainty_rerun/` — generator, merge step, preflight and runbook — is off the six
deleted noise types and onto the settled set, gated by
`scripts/test_uncertainty_job_scripts.py`. What chat H still owns there is submitting it, and it
should not be submitted before chat N has chosen the two lists.

**Does:** regenerates one deduplicated set of job scripts from the settled design; wires every gate
in §8 into a preflight that must pass; clears the caches; launches one task, then the grid.

**Caches to clear — this is not just `results/`.** The author's standing instruction is that
everything is re-run and the cache cleared. Three items, and the third has not been recorded before:
the memory-mapped intermediates `train_*.mmap`, `test_*.mmap`, `val_*.mmap` and any stale
`config*.json`; `data/QM9/processed`; and
**`results/master_tuned_hyperparameters.json` together with `results/hyperparameter_decisions.json`**
— `load_best_hyperparameters` (`models/models.py`) silently substitutes tuned hyperparameters from
February whenever both files are present, so leaving them in place means the re-run does not use the
hyperparameters anyone thinks it uses.

**🔴 TODO:** the compute ceiling, once §13.1 is settled.

> **Prompt.** Generate one deduplicated set of cluster job scripts from the settled design, wire the
> verification gates into a preflight, clear the caches, and launch. **Two things chat G left you
> specifically.** The noise conditions are settled and live in `noise_conditions.json` — read them
> rather than restating them, and put `scripts/test_noise_conditions.py` in the preflight beside the
> other gates; it already checks that the QM9 generator agrees with the file, and it will fail if the
> two drift. And §13.1 item 6 is a decision the author has to make **before** the uncertainty jobs
> are queued, not after: do those runs inherit the settled condition set, or test it? **You are where
> that question gets asked**, because you are what turns it into compute — chat F closed on
> 2026-08-27 and its material is in the tree, so nothing is waiting on it. Put it to the author with
> both options priced and do not choose quietly. Why it is open: the set was settled on **accuracy**
> on QM9, and a model can lose the same accuracy while being much better or worse at spotting which
> labels were corrupted, which is exactly what the uncertainty runs measure. The design is `RERUN_PLAN.md`
> §13.1 (the four stages and the replicate counts) and §6 (the noise types and levels); the gates are
> §8. Read `slurm_scripts_qm9_rerun/RUNBOOK.md` and
> `slurm_scripts_uncertainty_rerun/RUNBOOK.md` first — their reasoning about what is in and out, the
> tier ordering, the one-task-first discipline and the archive step are all still good, and only the
> noise types and levels are out of date. There are thirty existing script directories that overlap
> heavily; produce one set and delete the rest. **Clear three caches, not one:** the memory-mapped
> intermediate files and any stale settings files, the processed QM9 directory, and the tuned
> hyperparameter files in `results/` — the training code silently substitutes tuned hyperparameters
> from February whenever both of those files are present, so leaving them means the re-run does not
> use the hyperparameters anyone thinks it uses. Confirm which copy of the KIRBy checkout the cluster
> actually updates before submitting anything against it (§2.8b). No job is submitted until every
> gate in §8 passes locally, and one task runs end to end before the rest. Do not touch `paper.tex`.

---

#### Chat I — The uncertainty decomposition build

🔴 **UNBLOCKED 2026-08-27 — chat F is done, and nothing was updated here to say so.** This is now
the largest open build in the plan and nothing is in front of it. The author's assessment: *"seems like this needs a massive look. We really
need to discuss the objectives and what's going wrong. Another chat may have to handle that it feels
big."* Agreed — it is the largest single build in the plan.

**Settled, and not to be reopened (§0.3, §4 decision 3):** it gets built, to industry standard, by
deleting the broken code and replacing it.

**The spec is §5.5**, including four defects found while writing it: the two components are swapped
in at least seven places; two paths crash on contact; one call site is passed a standard deviation
where a variance is required; and one model's data-driven term is a single scalar broadcast to every
molecule, so its correlation with per-molecule noise is zero by construction however good the model
is. The sourced literature review is in `research_archive/f692d614/`.

> **Prompt.** Build the uncertainty decomposition — the split of a model's predicted uncertainty into
> the part that comes from noise in the data and the part that comes from the model not knowing.
> This was settled by the author on 2026-08-21 and is not open: it gets built to the standard other
> papers use, by deleting the broken code and replacing it. Do not propose dropping it. The
> specification is `RERUN_PLAN.md` §5.5 — the delete list, the build list, the four trainers to
> rewire, and the three-part trap around undoing label standardisation. The sourced literature review
> that justifies the method is in `research_archive/f692d614/`, along with working reference code.
> Read all of it before changing anything. Four defects were found while writing that specification
> and every one must be fixed, not just noted: the two components are swapped in at least seven
> places, two paths crash the moment they are reached, one call site is handed a spread where a
> variance is required, and one model's data-noise term is a single number copied to every molecule
> so its correlation with per-molecule noise is zero however good the model is. Each fix ships with a
> check that fails if the fix is removed. Do not touch `paper.tex`.

---

#### Chat J — Figure script consolidation and the five analyses

**Also owns one Methods sentence, assigned 2026-08-27: §3.1d.** A scaffold split holds whole
scaffold groups out, so on held-out molecules the grouped conditions' level-free shape is flat —
truthfully, not as a defect — and the predicted-label control is degenerate for them, because a
prediction does not change a molecule's scaffold. Both facts have to be stated where the grouped
results are reported, or the figures show a null that reads as a finding. It had no owner until now.

🔴 **Blocked: the author has asked for a 1:1 on the details of the analyses first** — *"All of those
analyses will have to be built, that belongs in the plan. But we're going to need to go over the
details of them 1:1."*

**Does:** collapses the figure script to one file with no versioning (§5.4); implements the guards
that belong in the analysis (§0.6 guards 1, 3, 4, 8, 9, 12); builds the five analyses in §0.4; and
removes both Gaussian processes from `ANOVA_MODELS_EXCLUDE` per §0.3.

**Three of the five are smaller than they look.** The decomposition already takes the noise level as
a parameter and it is simply never passed. The experimental-dataset decomposition is not broken — it
is correct code being fed one observation per cell because folds are averaged away before
integration. The paired significance test is *worse* than not built: it takes a representation and a
noise type as arguments and ignores both.

**Cannot start before the new columns exist**, because it is being rebuilt against them.

**Handed over from chat E — three requirements the parity work created, not optional:**

1. **Name the uncertainty column you read.** The script currently prefers the calibrated column
   wherever one exists and says nothing, so QM9 reports calibrated uncertainties and the three
   experimental datasets report raw ones under one heading. Read
   `models/model_defaults.py` → `UNCERTAINTY_DEFAULTS['primary_column']`, and print which column
   every figure used.
2. **Read `gp_fit_method` before comparing any Gaussian-process row.** Both pipelines now record
   whether botorch or the Adam fallback actually fitted the model. Those are different
   optimisations, and on at least one interpreter here botorch refuses the model class outright, so
   a mixed set of rows is possible. Rows fitted differently must not be pooled silently.
3. **Carry `spec_version` and `spec_hash` through to every figure caption or table footnote.** They
   are on every results row now. A figure that mixes two spec hashes is mixing two different models
   and must say so or refuse.

> **Prompt.** Collapse the figure and table generation to a single script and build the five analyses
> that were asked for and never written. The change map is `RERUN_PLAN.md` §5.4 and the five analyses
> are §0.4; the guards this work implements are §0.6, numbers 1, 3, 4, 8, 9 and 12. Read the existing
> script in full first — three of the five analyses are much smaller than they look, because the
> machinery is already there and is either never called or called with its arguments ignored. There
> is to be no versioning: delete the retired script, its launcher and its stale output directory,
> rename the current one, and fold the standalone analysis script in so there is one path to every
> number. Also remove both Gaussian processes from the analysis exclusion list per §0.3, and the
> descriptor-only marker with them. Every guard must be an assertion that fails the run, not a
> comment. Do not touch `paper.tex` — record what it needs instead.

#### Chat K — ✅ DONE 2026-08-26 — sync the two documents, and fix the bibliography

**Both jobs are finished.** What follows is the record; the prompt that ran it is kept at the end.

**1. The two documents now have an owner per fact.** The rule is stated in the opening of this file
and in `NOISE_DESIGN.md`'s header: **the design owns what the noise IS** — conditions, algebra,
parameters, sources, level grids, and the checks that are properties of the noise scheme — and
**this plan owns what gets RUN and in what order**. Ten disagreements were resolved; two of them
were a document contradicting *itself*:

| Was | Now |
|---|---|
| §1 of the design proposed level ladders that its own §6.4 superseded, and this plan restated the grids a third time | §6.4 owns every grid. §1 keeps the axis rule; §6.1 of this plan points |
| §0.3 of this plan said the heavy-tailed types satisfy the skewed-noise request; §13.3 said plainly that they do not | §0.3 corrected — the request is met by censoring and grouped-shifted, both one-directional |
| The design's opening said six state documents had been deleted; all six are on disk | Corrected, and it now points at §11. `REVISION_GUIDE.md` is the one that is genuinely gone |
| §7 of the design had three open items; two were answerable | Positive-control question closed (censoring *is* the label-keyed condition, §3.2); grids closed (the range-finding run set them). **Laplace was the last open item and the author closed it on 2026-08-27** — out of the full grid on measurement, kept in the deep run for the citation at 720 runs (§13.9). §7 now has nothing open |
| The design's status said nothing was implemented | Chat A built the Rust half; the header and §6.5 now say what is built and what is not |
| The design's delete list reached into the figure script | Moved out — §5.4 of this plan owns those, chat J executes them |
| §6.5 of the design was a second, different ordering of the whole re-run | Retitled to the noise-scheme build order; §10 of this plan owns the run order |
| The threshold-degeneracy figure was quoted at two precisions | One value in both: 99.99925% and 0.669 eV |
| The design did not cover the Python injector | Already closed by `8de0eed`; §3 of this plan updated to stop describing it as open |
| §13.3's grouped-condition table restated the design's algebra | Replaced by a pointer to §2 and §2a |

**2. The bibliography is fixed on the repository side.** 25 entries added, a key collision split, the
rejected-source blocklist made executable. Details and the corrections to what this section used to
claim are in §13.8; the one remaining line, which is yours because it is in `paper.tex`, is §9.1.

**The guard.** `scripts/check_bib_and_docs.py` — it fails if a cited key goes undefined, if two
entries collide on a key, if a source in the design's Sources list has no entry, if a rejected
source reappears, or if the two documents start restating each other's facts. Each check was
confirmed to fail by removing the fix it guards.

**Verified by a real build**, not only by the script: `paper.tex` copied to a scratch directory with
the one-line change applied, then `pdflatex → bibtex → pdflatex → pdflatex`. **51 of 51 citations
resolved, zero BibTeX warnings**, against 92 undefined-citation warnings in the repository's own
`_build_paper/paper.log`.

> **Prompt (as issued).** Two housekeeping jobs. First, bring `NOISE_DESIGN.md` and `RERUN_PLAN.md`
> into agreement. They were written at different times and have drifted — the noise type count and
> structure, the level grids, the Python injector, the Gaussian-process decision and the staged run
> design. Read both in full first. The design document is the specification of what the noise *is*;
> the plan is the process of what gets *run*. Where they disagree, decide which one owns the fact,
> put it there once, and have the other point at it rather than restating it. Do not average two
> disagreeing statements — resolve them or mark them open. Second, fix the bibliography: the
> manuscript's `\bibliography` line names a file that does not exist, seven cited keys are undefined,
> and twenty-two sources the noise redesign relies on are missing. `RERUN_PLAN.md` §13.8 lists them
> and `NOISE_DESIGN.md` §4a–4b has the verbatim quotes and access routes. Add the entries; do not
> edit `paper.tex` itself — record the one-line change it needs instead.

### 13.3 ✅ SETTLED — skewed noise, and the count of noise types

**Two things the author raised that the design does not currently answer honestly.**

**1. "6 strategies? I thought it was 4?"** Both are right; the count drifted and was never restated.
The design has **four core zero-mean types** — Gaussian, Student-t, Grouped, Outlier — which is the
four the author remembers. **Censoring** was then promoted from optional to essential by the
range-finding run, because it does twelve times more damage than any difference between the four.
**Laplace** was added as a sixth. So: four core, plus censoring confirmed, plus Laplace.

✅ **Superseded 2026-08-27 by measurement (§13.9).** The set is no longer described by this count at
all — it is described by what runs where, in `noise_conditions.json`: four conditions at full grid
(Gaussian, both grouped conditions, censoring), two at depth (one Student-t setting, one Outlier
setting), Laplace kept at depth, four settings dropped and the skewed draw never built. The
count drifted in the first place because nothing executable owned it; something does now.

**2. Skewed distributions — this is a genuine gap and §0.3 currently overstates it.** The author
asked on 2026-08-21 for *"a set of experiments — not for all combinations — for models that are
being highlighted (and their reps) — try picking from a skewed distribution of values"*. §0.3 records
that the redesign's heavy-tailed types satisfy this. **They do not.** Student-t and Laplace are both
symmetric; so are Grouped and Outlier. Heavy-tailed is not the same as skewed — a heavy tail makes
large errors more likely in *both* directions, while skew makes them more likely in one.

The only asymmetric condition in the design is **censoring**, which is one-directional by
construction, and it is the one condition where the effect is large. So the request is *partly*
satisfied, by the strongest condition in the study — but by a mechanism, not by a skewed draw.

#### ✅ The decision, 2026-08-26 — run both forms of grouped noise

*"For the skewed noise lets go with option C."*

**What was wrong with the grouped condition as specified.** It gives some scaffold groups *wider*
errors, and everything stays centred on the true value. But the evidence it leans on is that 62% of
real measurement variance sits between laboratories — and that describes laboratory *averages*
differing from one another. That is an offset, not a widening. The condition and the evidence for it
did not match.

**So there are two mechanisms, both real, both separately sourced, and they are now two conditions**
— one where the affected scaffold groups get a *wider* error still centred on the truth, and one
where they get a *shifted* error. Both are dose-matched, so they deliver the same total amount of
noise and can be compared directly.

**The two conditions, their algebra and their parameters live in `NOISE_DESIGN.md` §2 and §2a**,
which owns what the noise is. They were written there on 2026-08-26 (`8de0eed`) along with the two
rules the real Murcko scaffolds force — select groups by molecule fraction, and do not treat the
empty scaffold as a group. This section is the record of *why* the second condition was added and
is not the place to look up how it works.

**Why this is worth one extra condition.** The study's emerging result is that error pushing in one
direction hurts far more than error that scatters. Censoring shows that at the level of the whole
dataset. Grouped-shifted would show it at the level of a chemical family, through a different
mechanism, at matched amount. Two independent demonstrations of one effect is a much stronger claim
than one, and the comparison is direct — the two grouped conditions differ *only* in whether the
group's error is centred.

**It also closes a gap in the specification.** The design currently says the affected group fraction
has no published number, so choose 0.2 and say so. Under the shifted version there is nothing to
choose: Bentz's decomposition states how much of the total variance the group-level term carries,
so the parameter comes from the source rather than from a judgement.

**Not doing, and why.** A skewed *draw* — an asymmetric bell curve — was considered and rejected for
the three experimental datasets. Potency data is spread multiplicatively in raw units, and taking
logarithms converts that to ordinary symmetric variation (Srinivasan & Lloyd 2025, already cited in
`NOISE_DESIGN.md` §3.2). Since the models are fitted on the log scale, a skewed error draw there is
poorly supported and a referee who knows that argument would ask why. QM9 is different — it is
computed rather than measured, so nothing is being simulated and no assay has to justify the shape —
but with two grouped conditions and censoring already testing asymmetry, it would not earn its cost.

✅ **CLOSED 2026-08-26.** Chat B wrote the algebra into `NOISE_DESIGN.md` §2a; chat A implemented
it in `rust/src/main.rs` as `NoiseTargeting::GroupedShift { group_variance_share }`, ρ = 0.62 from
Bentz Table 7, offsets deliberately not centred. Its per-run delivered dose varies by 6.9% on
4,000 QM9 molecules — that is the group count talking, not an error, and it is why gate 1 is on the
construction and on the 20-seed mean rather than on one realisation.

The original wording, for the record: the shifted condition needs its algebra written down in
`NOISE_DESIGN.md` alongside the other five, in the same form. The natural construction is the one
Bentz's analysis itself uses — a group-level term plus a within-molecule term, with the two
variances summing to the target amount and the split between them taken from the paper.

### 13.4 — an additional embedding

Superseded by §13.7, which gives the evidence in tables rather than reasoning about it.

### 13.5 The two decisions restated plainly

The author asked for one of these to be rephrased. Both are restated without shorthand.

**Do the labels the model uses for early stopping also get noise?**

The data is split three ways: one part to train on, one to decide when to stop training, one to
score on. Right now the middle part is left clean on QM9. Two things follow.

Seven model families paste that middle part into the training set (`models/models.py:1383`, `:1458`,
`:1507`, `:1623`, `:1673`, `:1738`, `:3577`). Since the held-out fix, **one in nine of their training
labels is clean** — a free advantage no other model gets, and only for the tree-and-kernel half of
the roster.

The neural models use the middle part to decide when to stop. Stopping against clean labels means
they are being told when to stop by data that has none of the corruption they are supposed to be
suffering from. Nobody has that in practice.

The proposal is that the middle part carries the same kind and amount of noise as the training part,
drawn independently, and the scoring part stays clean. 🔴 The author's call.

**Is Laplace worth one extra condition?** The author's answer was *"I don't know the context well
enough"*, so the context: Laplace is a specific bell-shaped curve, more sharply peaked in the middle
than a normal curve with tails that fall away more slowly. Statistically it is nearly the same shape
as Student-t at about six degrees of freedom, which the design already covers — so **it buys almost
no new behaviour**. What it buys is a sentence: it is the only distribution that has actually been
*fitted* to real bioactivity measurement disagreements, with a formal test rejecting a normal curve
behind it (Anderson-Darling, p < 2×10⁻¹⁶, Krüger & Overington 2012). With it you can write "we
tested the error distribution observed in bioactivity data"; without it, "we tested a heavy-tailed
distribution". One extra condition on QM9 only. 🔴 The author's call.

### 13.6 The rescaling defect is QM9 only — the experimental pipeline is clean

Checked 2026-08-26 in response to the author's question, *"Is there a rescaling defect in KIRBy too
for the validation data?"*

**No. The experimental pipeline does both things correctly and is the reference for the fix.**

| | QM9 pipeline | Experimental pipeline |
|---|---|---|
| Embedding values as the model produces them | ✅ **fixed 2026-08-26** (was: each molecule rescaled to fill 0–255 using its own smallest and largest value, then stored as bytes) | ✅ returned as ordinary decimal numbers, no rescaling, no quantisation (`KIRBy/src/kirby/representations/molecular.py` — mol2vec `:1565`, MHG-GNN `:2074`, ChemBERTa `:2201`) |
| Features put on a common scale before the model | ✅ **fixed 2026-08-26** — `CONTINUOUS_REPS` covers PDV, MHG-GNN and ChemBERTa (was: PDV only) | ✅ every representation, fitted on the training fold and applied to the rest (`alternative_data_noise_robustness.py:882-884` for the tree and kernel models, `:967-970` for the neural ones) |

**Three consequences.**

1. **The paper's claim that learned embeddings are weak comes from QM9 and cannot stand.** But the
   reason is not the one this section gives: the QM9 numbers behind it are failed Gaussian-process
   fits (§2.8f), not badly stored features. Both are now fixed and the claim needs re-testing from
   scratch.
2. ✅ **The fix was a port, not a design** — what the experimental pipeline already did. Done.
3. ✅ **Measured, and the attribution was wrong.** The 0.87-against-−0.02 gap was *attributed* to
   this defect. It is a failed fit (§2.8f). Chat C measured it — `scripts/retest_embedding_kernels.py`, which scores the same molecules, the same
   scaffold split and the same seeds under both storage schemes so the difference is the storage and
   nothing else. Result in §2.8c.

### 13.7 🔴 TODO — an additional embedding, on the evidence

Replaces the reasoning previously offered here, at the author's request. **This is KIRBy's own
representation-pruning evidence**, which scores every representation on its own and then compares
how similar their predictions are.

**How to read it.** *Solo score* is how well that representation alone predicts the property —
rank correlation between predicted and true values, higher is better. *Average similarity* is how
close its predictions are to every other representation's, averaged — high means it carries the same
signal as the rest. *Nearest twin* is the single representation it most duplicates.

#### hERG Ki (1,412 molecules), from the stored evidence

| Representation | Solo score | Average similarity | Nearest twin | Similarity to twin |
|---|---|---|---|---|
| Graph kernel | 0.745 | 0.814 | PDV | 0.844 |
| PDV | 0.729 | 0.823 | Mordred descriptors | 0.889 |
| Atom pair | 0.727 | 0.820 | Mordred descriptors | 0.845 |
| ECFP4 | 0.722 | 0.807 | Avalon | 0.839 |
| Mordred descriptors | 0.713 | 0.825 | PDV | 0.889 |
| Avalon | 0.704 | 0.808 | ECFP4 | 0.839 |
| ChemBERT | 0.696 | 0.797 | SMI-TED | 0.831 |
| GROVER | 0.693 | 0.793 | PDV | 0.825 |
| SMI-TED | 0.671 | 0.801 | ChemBERT | 0.831 |

The full run on the cluster covered fifteen representations and ranked them:
graph kernel .745 · MHG-GNN .744 · ECFP4 .728 · PDV .727 · atom pair .724 ·
Mordred .721 · Avalon .714 · mol2vec .713 · topological torsion .700 · ChemBERT .696 ·
GROVER .692 · SMI-TED .690 · ChemBERTa .689 · MolFormer .687 · GraphMVP .655.
**No pair exceeded 0.9 similarity** — on this dataset the representations genuinely disagree.

#### logD (5,028 molecules)

Mordred .894 · MHG-GNN .889 · PDV .882 · GROVER .868 · Avalon .865 · ECFP4 .860 ·
graph kernel .857 · mol2vec .850 · atom pair .848 · SMI-TED .844 · MolFormer .835 ·
topological torsion .830 · ChemBERT .809 · ChemBERTa .786 · GraphMVP .782.

**Heavily redundant, unlike hERG.** The most similar pair is Mordred and PDV at
0.967, and the top nine are all above 0.90 with each other. The final pool cut the descriptor
vector, mol2vec, ChemBERTa, the graph kernel and atom pair.

#### Caco-2

Only one pair above 0.90 — PDV and Mordred at 0.907. Everything else genuinely
distinct. Mordred and mol2vec were cut.

#### QM9

Final pool: Mordred, Avalon, SMI-TED, ECFP4, SELFormer, ChemBERTa, mol2vec, ChemBERT, GraphMVP,
MolFormer. Cut: PDV, GROVER, MHG-GNN, RDKit fingerprint, atom pair, MACCS.
The recorded reason is that the redundancy sits in the strong fingerprints, in two tight clusters,
**not in the pretrained embeddings** — so the cut removed duplicate fingerprints and kept one
representative of each learned family.

#### QM9 itself — 9,978 molecules, five-fold scaffold-group CV, best of five models

The most directly comparable evidence there is: same dataset, same size, same target, same splitting
rule as the paper. Scored as fraction of variance explained, higher is better.

| Representation | Score | ± | Family |
|---|---|---|---|
| Mordred descriptors | 0.912 | 0.010 | descriptor |
| **PDV** | **0.903** | 0.011 | descriptor |
| GROVER | 0.899 | 0.010 | learned, graph |
| **MHG-GNN** | **0.889** | 0.033 | learned, graph |
| **Avalon** | **0.884** | 0.013 | fingerprint |
| RDKit fingerprint | 0.859 | 0.014 | fingerprint |
| Atom pair | 0.850 | 0.009 | fingerprint |
| SMI-TED | 0.826 | 0.019 | learned, transformer |
| SELFormer | 0.818 | 0.013 | learned, transformer |
| **ECFP4** | **0.816** | 0.063 | fingerprint |
| ChemBERTa | 0.812 | 0.012 | learned, transformer |
| **mol2vec** | **0.803** | 0.028 | learned, shallow |
| MACCS | 0.795 | 0.031 | fingerprint |
| ChemBERT | 0.788 | 0.062 | learned, transformer |
| GraphMVP | 0.778 | 0.021 | learned, graph |
| MolFormer | 0.772 | 0.029 | learned, transformer |
| Topological torsion | 0.680 | 0.047 | fingerprint |
| MolCLR | 0.620 | 0.026 | learned, graph |
| Coulomb matrix | 0.500 | 0.038 | 3D |
| USRCAT | 0.417 | 0.055 | 3D |
| Graph kernel | 0.149 | 0.031 | graph |
| Uni-Mol v2 | −0.007 | 0.007 | learned, 3D |

⚠️ **The stored file reports error, not variance explained**, and error runs the other way. An
earlier reading of it here had the ranking upside down and is corrected above.

#### What this evidence actually says

- **The graph kernel is dataset-specific and must not be added.** It is the *best* representation on
  hERG Ki at 0.745 and near-useless on QM9 at 0.149.
- **ChemBERT is the weakest of the transformers.** On QM9 it scores 0.788, *below* mol2vec's 0.803.
  If a transformer is wanted, SMI-TED (0.826), SELFormer (0.818) and ChemBERTa (0.812) all beat it.
- **Mordred is the strongest single representation on QM9 but adds little as a factor level.** It is
  the same family as PDV already in use, and on logD the two agree at 0.967 — the
  most redundant pair in that whole study.
- **Avalon is the strongest genuinely-new option.** At 0.884 it beats ECFP4 (0.816) on QM9, it is a
  different kind of fingerprint, and on hERG Ki its closest match is ECFP4 at 0.839 — below the
  redundancy line. It is also an ordinary RDKit fingerprint, so implementing it is trivial.
- **MHG-GNN is not weak.** 0.889 on QM9, second of fifteen on logD, second of fifteen on hERG Ki.
  The paper's claim that learned embeddings are weak is a QM9 claim, from the pipeline with the
  storage defect (§13.6).
- **mol2vec is the weakest learned representation** and was cut from three of the four pools.

#### ✅ SETTLED 2026-08-26 — the representation set, and ✅ IMPLEMENTED the same day

*"Drop SMILES from the list, add Avalon and ChemBERTa."*

**State of the code, 2026-08-26 (chat C).** Avalon builds and runs in both pipelines. ChemBERTa was
already implemented on the QM9 side and unusable because of the storage defect; it now produces
results for the first time in the study. mol2vec is **deleted** — Python pipeline, Rust record,
storage guard, hybrid source list. One-hot and randomized SMILES still build, because removing the
tokenizer means editing the record layout and the vocabulary handling, so they are **refused by
name** at the top of `main()` instead: a job asking for one exits non-zero before a molecule is
read. Verified both ways.

**The six representations for the re-run, both studies:**

| Representation | QM9 score | Family | Status |
|---|---|---|---|
| PDV | 0.903 | physicochemical descriptors | unchanged — the primary representation |
| MHG-GNN | 0.889 | learned, graph | unchanged |
| **Avalon** | **0.884** | fingerprint, path and feature based | **NEW** |
| ECFP4 | 0.816 | fingerprint, circular | unchanged |
| **ChemBERTa** | **0.812** | learned, sequence | **NEW** |
| Sort & Slice | — | fingerprint, collision-free circular | unchanged — **fixed in place**, it is a colleague's method and the paper describes it as the collision-free counterpart to ECFP4 |

**Out:** mol2vec (0.803, the weakest learned representation, cut from three of four pruning pools)
and one-hot SMILES. Both are now enforced in the code, not just recorded here.

Scores are from KIRBy's own survey on 9,978 QM9 molecules under five-fold scaffold-group CV, best of
five models. They rank representations; they do not transfer as numbers to this study.

**Why this set.** Four distinct families instead of two overlapping fingerprint families plus a raw
string. Every member is individually competent on QM9 — the weakest is 0.812, against 0.149 for the
graph kernel and −0.007 for Uni-Mol, both of which were considered and rejected. Count is unchanged
at six, so the grid does not grow.

**Two things that follow, and neither is optional.**

**1. Both new representations depend on chat C.** ✅ **Done 2026-08-26.** One correction to what
follows: both are new to the representation *set*, but only Avalon was new to the *code*. ChemBERTa
was already built and wired on the QM9 side and needed the storage fix and nothing else; Avalon was
written for the QM9 pipeline and wired into the experimental one, where `create_avalon` already
existed. ChemBERTa is already implemented in this pipeline
but carries the same per-molecule rescaling defect as the other learned representations, so it is
unusable until that is fixed. Avalon is an ordinary RDKit fingerprint and needs adding to both
pipelines; it is binary, so it needs no rescaling and no standardisation.

**2. Every claim the paper makes about SMILES comes out.** There are two, both in §9's
"survives regeneration" list now:
- `paper.tex:493` — model choice explains over 91% of robustness variance on SMILES against 72% on
  PDV. **This was the paper's sharpest single illustration of its central claim**, and there is no
  longer a representation at that end of the scale. The claim itself survives; the illustration has
  to be rebuilt from whichever representation now sits at the extreme, which the re-run decides.
- `paper.tex:466` — SMILES gains most from making a neural network Bayesian.

`paper.tex:203` describes the representations and must be rewritten for the new set.

### 13.8 ✅ DONE — citations for the paper are reachable

The author asked that the sources be easy to use when the paper edits happen. **Done 2026-08-26
(chat K): 25 entries added to `citations.bib`, and `NOISE_DESIGN.md`'s Sources list now carries the
BibTeX key beside every source**, so a paper pass can cite straight from the evidence document
without looking anything up.

**The count in the earlier version of this section was wrong in a way worth recording**, because it
is the same failure mode as guard 12 — a name matched, so the source was assumed present. Three of
the ten sources listed as "already in `citations.bib`" had no entry at all; the name matched a
co-author or an unrelated paper:

| Listed as present | Actually |
|---|---|
| **Kalliokoski** | Appeared only as a co-author on `Kramer2012`. This is the pIC50 = 0.68 anchor and the √2 pairing correction |
| **Niu** | Absent. This is §4a's recommended logD source, and the one that removes the dependence on a figure that failed verification |
| **Zhao** | Absent. The only Zhao entry was `zhao2025`, an unrelated preprint. Zhao **2017** is the published precedent for the fraction-of-spread axis in `NOISE_DESIGN.md` §1 |

Genuinely present and reused rather than duplicated: `Heid2023`, `Kolmar2021`, `Kramer2012`,
`Song2022`, `huber1964robust`, `landrum2024`.

**What was added.** All 25 carry metadata taken from Crossref DOI content negotiation, or from the
publisher record where no DOI exists. The full list with keys is `NOISE_DESIGN.md` § Sources.

Four had no usable metadata anywhere in the repository and were looked up:

- **Svensson et al. 2025** → *Artificial Intelligence in the Life Sciences* 7:100128,
  doi:10.1016/j.ailsci.2025.100128. Peer-reviewed and open access. **Reading it corrected a number
  in the design**: the claim was "25–63% censored in ten of fifteen assays"; Table 1 says eight of
  fifteen in that band and thirteen of fifteen with any censoring. Fixed in `NOISE_DESIGN.md` §3.5,
  §5.3b and §5.5.
- **Tukey** → Tukey JW (1960), *A survey of sampling from contaminated distributions*, in Olkin (ed.),
  *Contributions to Probability and Statistics*, Stanford University Press, 448–485. This is the
  source of the conventional contamination scale factor of 3.
- **Hampel 2001** → ETH Zürich Seminar für Statistik Research Report 94,
  doi:10.3929/ethz-a-004158597. **An institutional research report, not a peer-reviewed article.**
  Added with that status stated, because it is the source of the 1–10% contamination fraction and
  the document's rule is peer-reviewed primary sources. No peer-reviewed version carries the
  sentence verbatim.
- **Assay Guidance Manual** → the *Assay Operations for SAR Support* chapter, NCBI Bookshelf
  NBK91994, PMID 22553866.

**Three author-attribution errors were found and fixed while building the entries**, all in
`NOISE_DESIGN.md`'s Sources list: Zhao 2017 was credited to "Zhao Y, Wang J" rather than Zhao Linlin
and Wang Wenyi; Sato 2018 was given six authors in one place and eleven in another when it has four;
and Bruneau & McElroy was dated by its online-first year rather than its 2006 issue.

**The blocklist is now enforced, not just written down.** `NOISE_DESIGN.md` §4a lists five sources
that were traced to source and rejected — Matsson 2019 (no such paper), Pham-The 2013 (no
reproducibility experiment), Lanevskij & Didziapetris 2019 as a Caco-2 log-unit source (unit error),
Lee 2017 as a Caco-2 standard-deviation source, and the Fagerholm preprint. Nothing stopped one
being re-added by a later pass. `scripts/check_bib_and_docs.py` now fails the run if any of them
appears in `citations.bib`, and the same names are repeated in a comment block at the head of the
added entries.

🔴 **Still outstanding, and it is the author's:** the one-line `\bibliography` change in §9.1.
