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
2026-08-21; **one item is open there and it is whether Laplace is queued** (§4 Decision 4, §13.5).

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
| 3 | Noise redesign — specification, literature, local tests | ✅ done and sourced | §13 chats A, B |
| 4 | Assay-error anchors and the blocklist of bad numbers | ✅ done, peer-reviewed, two passes reconciled | — |
| 5 | Gaussian-process kernel question | ✅ **answered and decided 2026-08-26** (§10b.2) | §13 chat C |
| 6 | Within-noise-level uncertainty correlation | ✅ **author's fix, and it is implemented** — `within_sigma_unc_noise_rho`, `generate_paper_figures_v2.py:1031-1057`. See §3.5 | §13 chat F |
| 7 | Uncertainty machinery in KIRBy (out-of-fold, recorded noise, confound control) | 🟠 built, 9 defects fixed, **never submitted, never reviewed with the author** | §13 chat F |
| 8 | QM9 job scripts | 🟠 written, superseded by the redesign before they ran | §13 chat G |
| 9 | Uncertainty job scripts | 🟠 written, superseded, and point at a possibly stale checkout | §13 chat G |
| 10 | Parity audit script | 🟠 written; its literals were still being verified when the last session ended | §13 chat E |
| 11 | Noise redesign in the pipelines | 🟢 **Rust done 2026-08-26** (chat A) — deleted, built, 14 gates passing, verified on 4,000 real QM9 molecules. Python is chat B's | §13 chats A ✅, B |
| 12a | Records could be written short, and an unparseable SMILES crashed the binary | ✅ **fixed 2026-08-26 (chat D)** — all-or-nothing records, a null-pointer check RDKit's binding needed, and the reader now refuses to guess (§2.7) | §13 chat D ✅ |
| 12 | Per-molecule rescaling of learned embeddings | ✅ **fixed 2026-08-26 (chat C)** — storage, widths and standardisation, with a guard that fails if any of the three is removed (§2.8c) | done |
| 13 | Concurrent-task configuration race | ✅ **fixed 2026-08-26 (chat D)** — the configuration file is named per task and the binary has no default path; guarded by `scripts/test_config_isolation.py` (gate 10) and a test in `rust/tests/writer_guards.rs` | §13 chat D ✅ |
| 14 | Aleatoric/epistemic decomposition | 🔴 spec written, not built; 4 further defects found (§5.5) | §13 chat I |
| 15 | The five never-built analyses | 🔴 none built (§0.4) | §13 chat J |
| 16 | Figure script consolidation to one file | 🔴 not started (§5.4) | §13 chat J |
| 17 | Environment missing three model families | 🟢 **mostly done 2026-08-26 (chat D)** — the loud-failure half was already in KIRBy (`333f005`); `scripts/check_environment.py` now probes any interpreter and is wired into the job template; the local `torch_geometric` import is fixed. **One open finding for you: three packages need a newer scikit-learn than is installed** (§2.8d) | §13 chat D ✅ |
| 18 | Paper-side fixes needing no compute | 🔴 not started — **deliberately parked**, see §12 | parked |
| 19 | The two documents had drifted apart | ✅ **done 2026-08-26** (chat K). Ownership rule stated, ten disagreements resolved, two of them a document contradicting itself. Guarded by `scripts/check_bib_and_docs.py` | §13 chat K ✅ |
| 20 | The bibliography | ✅ **done 2026-08-26** (chat K) — 25 entries added, a key collision on two different papers split, the rejected-source blocklist made executable. **One line left in `paper.tex`, and it is the author's** (§9.1) | §13 chat K ✅ |

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
| 13 | **Shared mutable state between concurrent tasks** | Every task reads and writes one configuration file in one directory, including the identifier that selects its data files. Two tasks at once can overwrite each other's inputs, silently (§2.8a) | Make the path unique per task, or give each task its own directory. Gate: run two tasks concurrently and assert each read its own configuration |

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

### 2.3 Nothing was ever checking that the two injectors agreed

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

1. **Truncated records.** Fixed by deciding the fingerprint *before* any byte of the record is
   written. `prepare_ecfp4` returns the 256-byte block and, on failure, the reason; the failure
   path writes 256 zero bytes so the record stays full length, records the molecule in
   `featurisation_failures_{file_no}.csv`, and the run then **refuses to finish** unless
   `--allow-featurisation-failures` is passed. Alignment is preserved and the condition is loud —
   a zero fingerprint can never reach a model as though it were real features.

2. **🔴 An unparseable SMILES did not take the error branch — it killed the process.**
   Found while writing the test for (1), and not previously recorded anywhere. RDKit's
   `SmilesToMol` returns a **null pointer** for a SMILES it cannot parse; the binding
   (`rdkit-sys-0.4.12/wrapper/src/ro_mol.cc:18`) wraps that null in a shared pointer and returns
   it as `Ok`. Only a thrown C++ exception becomes `Err`. So the old `Err(_) => continue` branch
   was unreachable for the ordinary bad-SMILES case: `rdk_fingerprint_mol` dereferenced null and
   the process died with **SIGSEGV, no message and no partial output**. Confirmed by feeding
   `this-is-not-a-smiles` to the built binary — exit 139. Fixed with a null check in
   `prepare_ecfp4`. Not live on QM9, where every SMILES comes from the dataset; live on the ADME
   sets. Worth knowing for the resubmit-failed-indices workflow in the runbook: a task that hit
   this would die with signal 11 and be resubmitted straight back into the same crash.

3. **Index drift → truncation.** `read_all_target_values` no longer exists; chat A replaced it
   with `read_train_labels` (`rust/src/main.rs`), which changed the failure from a shifted index
   to a silent `break` — a short label vector, so the noise plan would cover fewer molecules than
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

**Found 2026-08-25 from the harvested clean-data results, then confirmed in the code.** This is the
largest single defect in the study. It is a data-preparation bug, it has never been a robustness
result, and it invalidates a family of the paper's conclusions.

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
| The per-molecule rescaling is gone; each builder returns the model's own values as 32-bit floats, and so do its failure paths, so a molecule that cannot be embedded still writes a full-width record | `scripts/process_and_train.py` — `chemberta_fingerprint`, `mhggnn_fingerprint`, `mol2vec_fingerprint` |
| The record widened: mol2vec 300→1200 bytes, ChemBERTa 768→3072, MHG-GNN 1024→4096, read back as float32 | the writer and reader in `process_and_train.py`, and the buffers in `rust/src/main.rs` |
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


### 2.8d ✅ MOSTLY FIXED 2026-08-26 (chat D) — the two Gaussian-process jobs that produced nothing

Both failed on 2026-08-19 (`12822693`, `12822694`), after eight and six minutes. The output
directories were created and are **empty** for all three datasets.

**What the logs show.** Every one of the five folds printed its train and test sizes, then
`ERROR: No results for <dataset>`. Nothing between. That gap is the diagnosis: the per-experiment
progress line at `alternative_data_noise_robustness.py:1342` never printed once, so **the
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

2. ✅ **The environment is now asserted, and the interpreter is pinned.**
   `scripts/check_environment.py` names the interpreter it is speaking for, prints every relevant
   package version, **constructs** each requested model rather than merely importing its package,
   and additionally *fits* the two that import cleanly and fail on contact. It is wired into the
   job template (`slurm_scripts_qm9_rerun/generate_scripts.py`) so a task dies in seconds rather
   than after five folds, and §1b of the runbook has the copy-paste block that runs it under both
   cluster interpreters and diffs them. The two dead jobs were `--wrap` submissions with no output
   path, so they inherited whatever interpreter was active and left no log saying which; the
   runbook now states that jobs are submitted by script, never by `--wrap`.

**🔴 One finding for you, and it is not mine to decide.** The probe also checks whether each
package's own declared requirements are satisfied, because pip never re-checks that after the
fact. On this laptop three of them are not:

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

### 2.9 The Methods figure does not show the experiment

`paper.tex:359` captions it as the QM9 label distribution. The code that draws it
(`generate_paper_figures_v2.py:2541-2562`) uses a synthetic three-component Gaussian mixture and
reimplements two of the noise types differently from the pipeline — threshold as a median split,
value-proportional as additive where the pipeline is multiplicative. Counting the Python injector,
"threshold" therefore has three different definitions in three places.

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

### 3.1a One check in it never fires

The placebo check on the training side is permanently blank. It recomputes the noise pattern from
the model's *predicted* label instead of the true one, so that if uncertainty correlates with that
just as strongly, you know the model is tracking its own prediction rather than the noise.

The guard is `if sigma in extras['oof_mean']:` at `alternative_data_noise_robustness.py:941`. The
line that puts the current noise level into `extras['oof_mean']` is `:954`, thirteen lines further
down. So the guard is evaluated before the thing it tests for exists, and it is False on every
pass. Every training row writes a blank into that column. The same dead block appears again in the
neural runner at `:1031-1035`.

**Fix: move the block below the out-of-fold block, in both runners.** Two lines each.

Worth knowing why nothing caught it: the regression test for these fixes checks several of them by
searching the pipeline source for a matching string (`smoke_nine_fixes.py:79`, `src = open(PIPE).read()`).
A string match passes whether or not the matched line ever runs.

### 3.1b Two silent no-ops to assert against at launch

Both are in the same writer, and both fail quietly rather than loudly.

- Per-molecule uncertainty is written only if the zero-noise level is present in the run
  (`:1422`, `if save_this and uncertainties.get(0.0) is not None`). A run whose level grid omits
  zero writes **no uncertainty rows at all**, without complaint.
- Only the Gaussian condition is written unless `--unc-strategies all` is passed; the default is
  `legacy` (`:1671`). The generated job scripts do pass it, and they also pass `--oof-folds 5` and
  `--oof-outer-folds 1` — so the three-fold saving on the cross-fitting is already baked in.

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
| The injected noise recorded per molecule | no — reconstructed by regression, and now identically zero | yes | The exact draw **does** exist in Rust (`main.rs:200`, `:275`, `:442`) and is simply never written. This is a serialisation change, not a redesign |
| Per-molecule noise scale recorded | no — computed then discarded (`main.rs:309-317`) | yes | Same serialisation change |
| Held-out noise scale computed against the *training* cut-points | no | yes | Follows once QM9 records a scale at all |
| The level-invariant noise pattern (the confound control) | no — appears nowhere | yes | Without it the zero-noise subtraction cannot be formed on QM9 |
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
3. 🔴 **On QM9 the fix is now correct but starved.** `save_uncertainty_values`
   (`scripts/utils.py:212-221`) does not record the injected noise — it *reconstructs* it by
   regressing the noisy label on the clean one and keeping the residuals. After the held-out fix the
   test labels are an exact affine function of the clean labels, so those residuals are identically
   zero and there is nothing left to correlate against. **The fix is fine. Its input died.** Recording
   the draw at source (§5.2) is what revives it.

**So the honest status is:** the author's fix is correct, implemented, and load-bearing. What is
missing is the *data* to feed it on QM9, and a 1:1 review of the KIRBy machinery that feeds it on
the experimental datasets (§13 chat F).

---

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
| **SNS** | counts computed (`sub_counts=True`) then **thrown away** — `np.array(sns_fp, dtype=np.uint8)` then `packbits` makes every nonzero a 1 (`:366-371`) | raw counts, then standardised | ❌ binary vs standardised counts |
| **MHG-GNN / mol2vec** | per-molecule min-max to 8 bits, never standardised (§2.8c) | full precision, no quantisation | ❌ see §2.8c |

**The fingerprint one is the most serious thing found in this whole audit.** `paper.tex:203`
describes ECFP4 as *"circular substructures with radius r=2"*. On QM9 that is false — it is a
path-based fingerprint, a different substructure family with different bit density and different
similarity behaviour. Around 254 job invocations pass `-r ecfp4`.

It is a wrong-function bug rather than a naming slip: the same crate file ships
`morgan_fingerprint_mol` and it is never imported. Note that binding is **radius 3**, so it is
ECFP6 — switching to it would still not give ECFP4, and a new binding is needed.

**A second, quieter data-loss bug in the same area.** Because the substructure counts are cast to
`uint8` *before* being packed, a count that is an exact multiple of 256 wraps to zero and the
substructure records as **absent**. Rare, silent, wrong.

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

**Still outstanding on that side:** the feature-sampling setting, pending the measurement below.

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
| **Forest feature sampling** | `sqrt`, every feature, or 30%? | Measured at production scale (§3.4.4b). Representation-dependent: `sqrt` is fine on descriptors and bad on fingerprints | **30% on both.** The only setting that wins or ties everywhere. **Changes QM9 too**, so it invalidates every forest result — but everything is being re-run |
| **Early stopping** | Keep the last epoch, or roll back to the best? | QM9 counts twenty epochs of no improvement and then returns *that* epoch's weights. The experimental side snapshots and restores the best. Those twenty extra epochs are spent memorising injected corruption, and more of it at higher noise — so **QM9's neural degradation curves are steeper for a procedural reason, pointing the same way as the finding** | **Roll back, both sides.** It is what almost everyone means by early stopping. Caveat: it means selecting on validation labels, which under decision 2 would be noisy — but that is correct, since nobody gets clean labels when deciding when to stop |
| **Uncertainty calibration** | Report calibrated or raw? | A single multiplier fitted after training so predicted uncertainties match observed errors. QM9 does it, the experimental side does not, and the figure script silently prefers the calibrated column where it exists. Because it is one positive multiplier it **cannot change the order** of molecules — so both uncertainty-tracking questions are unaffected either way. It moves coverage and calibration-error numbers only, which are exactly what it is fitted to fix | **Raw as primary**, calibrated as a clearly-labelled secondary if wanted. Reporting coverage after calibrating is close to circular. Either way the analysis must state which column it read — it does not today. Free: aligning down needs no re-run |
| **Embedding standardisation** | Standardise the learned embeddings per feature, or leave them raw? | Separate from the storage fix in §2.8c, which is not optional. Without it, a kernel with one shared lengthscale across a thousand dimensions is dominated by whichever dimensions are widest | ✅ **SETTLED 2026-08-26 — standardise.** Approved as part of chat C's plan and implemented: `CONTINUOUS_REPS` in `process_and_train.py` now covers PDV and all three learned embeddings, fitted on the training split. It changes every embedding number, which is why it is recorded here rather than left implicit |

**Decision 4 — sign off the noise design.**
`NOISE_DESIGN.md` §7 still has Laplace open. My view: include it. It is one extra condition on
QM9 only, and it is the only distribution actually *fitted* to real bioactivity error, so it buys
a citation for a claim the paper wants to make. The other open item there — whether to keep a
deliberately unrealistic label-keyed condition as a positive control — I now think is answered by
censoring (§3.2), so it can be closed.

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

---

## 6. The re-run design

### 6.1 Noise types and levels

From `NOISE_DESIGN.md` §2 and §6.4, with the levels set by the range-finding run rather than
chosen in advance.

**Five zero-mean types, all delivering the same amount of noise:** Gaussian; Student-t at three
tail weights; Laplace (QM9 only, pending decision 4); Grouped by scaffold; Outlier with random
selection. **Plus censoring on its own axis**, because it is not zero-mean and cannot be
dose-matched.

| Dataset | Axis | Levels |
|---|---|---|
| QM9 | fraction of the label spread — there is no assay error to anchor to | **`NOISE_DESIGN.md` §6.4** |
| LogD, Caco-2, hERG | log units, each anchored to that endpoint's published assay error | **`NOISE_DESIGN.md` §6.4** |
| Censoring, all datasets | fraction of labels clipped | **`NOISE_DESIGN.md` §6.4** |

**The numbers are deliberately not repeated here.** They were, and the two documents disagreed
about them for a fortnight. §6.4 of the design owns every level grid; this section owns what the
grids are *for*. Seven levels on QM9 and six on each experimental dataset is what the cost
arithmetic in §13.1 assumes.

Each experimental grid brackets one unit of real assay error for that endpoint and runs to about
twice it. Report the fraction-of-spread alongside, because one unit of real error is 0.13 of the
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

Scope, updated for the new noise types: 3 datasets × 7 models that emit a per-molecule
uncertainty × 4 representations × 6 noise conditions × 6 levels × 5 folds. That is fewer levels
than the built scripts assume (six instead of eleven), which is roughly a 45% saving before any
other lever.

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
| **Q3** | What does model choice actually buy you at a realistic amount of error? | Accuracy at the anchored noise level; best-minus-worst across models at each level; retention area **printed beside its clean baseline, never alone** | Per (model, representation, noise type, level) | The anchored level chosen per dataset (§6.1); the QM9 reporting level still 🔴 TODO |
| **Q4** | Can a model's uncertainty tell you which labels are bad? | Spearman correlation between predicted uncertainty and the size of the injected noise, **within each noise level**, **minus the same correlation at zero noise**, scored **out-of-fold**. Reported with a permutation null | Per (dataset, model, representation, noise type, level) — never pooled | The noise **recorded**, not reconstructed (§5.2); out-of-fold scoring on scaffold groups; a zero-noise run of the same type to subtract |
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

# the pipeline, over real mmap files — 14 gates
cd rust && cargo test --release
```

Gate 2 is chat B's (`scripts/crosscheck_injectors.py`). Gates 6 and 9 need a training run and are
chat H's. Gates 8, 10 and 11 are chat D's. Each of chat A's gates was checked by removing the fix
and confirming the gate fails.

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
2. **The two injectors agree.** Same labels, same seed, same target — Rust and Python within the
   tolerance in `NOISE_DESIGN.md` §5.1b (half a percent, except the heaviest-tailed Student-t
   where sampling variability is 2.2% and is expected).
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
8. ✅ **A short record no longer desyncs the file.** Implemented both sides (chat D):
   `rust/tests/writer_guards.rs` feeds the binary a molecule RDKit cannot parse and asserts the
   file is still the exact sum of its records' expected lengths;
   `scripts/test_record_alignment.py` hand-builds a short record and asserts the reader raises
   rather than returning silently wrong features. `cargo test --test writer_guards` and
   `python scripts/test_record_alignment.py`.
9. **Every new column is populated** in a smallest-possible end-to-end run.
9b. ✅ **The interpreter can build what the job asks for.** `python scripts/check_environment.py
    --models <what this job runs>`, wired into the job template; runbook §1b runs it under both
    cluster interpreters and diffs them (§2.8d).
10. ✅ **Two tasks running at once do not corrupt each other.** Implemented (chat D):
    `python scripts/test_config_isolation.py` launches two binaries concurrently in one directory
    with different representations and asserts each keeps its own data, plus an instant static
    half that fails if a fixed `config.json` reappears anywhere in the tree. `--end-to-end` runs
    two real pipeline tasks side by side (§2.8a).
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
| 2 | **One decision, not a document.** `NOISE_DESIGN.md` §7 is down to a single open item — whether Laplace is queued as a condition. The dose-matching rule was approved 2026-08-21; the positive-control question and the level grids were closed 2026-08-26. Context for the Laplace call is §4 Decision 4 and §13.5 | you |
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

Where the features are binary the two kernels are indistinguishable. Where they are the two learned
embeddings the radial basis collapses — and that is the rescaling defect in §2.8c, verified in the
source at `process_and_train.py:971-975` and `:828-831`, not a property of the kernel.

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

### 13.1 🔴 TODO — the run design: replicates, and what runs at full grid

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
| **0 — screen** | every model × every representation, Gaussian and censoring, full level grid | 1 | Choose what stage 2 goes deep on. **Reused as replicate 0 of stage 1, not thrown away** |
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

1. 🔴 **How many replicates in the end?** The arithmetic says six is the floor if Q2 is to be
   answerable and MIN_CELL_ITERS is to have headroom. Ten costs 67% more than six and buys
   precision, not new answers. Author's note: *"I believe I was doing 10 and holding off replicates
   for uncertainty."*
2. 🔴 **Is one replicate for the uncertainty runs still right?** The uncertainty statistics are
   correlations over thousands of molecules, so their precision comes from the molecule count, not
   the replicate count — one replicate is defensible **provided** a permutation null is reported so
   the reader has a reference distribution. Without repeats there is no run-to-run error bar at all.
3. 🔴 **Which noise types run at full grid in stage 1?** Q1 asks for a decomposition *per noise
   type*, so every type that needs its own decomposition must run at full grid. The three
   structurally distinct ones are Gaussian (even), Grouped (structure-keyed) and Censoring
   (one-directional). The heavy-tailed and sparse-contamination types are what stage 0 and stage 2
   would establish behave like Gaussian.
4. 🔴 **Which models and representations go deep in stage 2?** Cannot be chosen before stage 0 runs.
5. 🔴 **The QM9 reporting level** (§6.1). Every table that reports accuracy at one level needs it.

**One caution to hold on to.** A staged design is only honest if the reduced set in stage 1 is
justified by what stage 0 and stage 2 show, and the paper says so. "We ran everything under Gaussian
and a subset under the rest" is defensible; presenting it as a full factorial is not.

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
| **C** | Embedding storage fix, and the Gaussian-process re-test | — | ✅ **yes** |
| **D** | Infrastructure: settings race, writer guards, environment | — | ✅ **DONE 2026-08-26** |
| **E** | Cross-pipeline parity | — | ✅ **yes** — but check what another session already did |
| **F** | Uncertainty machinery: audit, fix the clear bugs, report the rest | — | ✅ **yes** — it has real work in it, and produces the material for the 1:1 |
| **G** | Local test: which noise settings earn their place | A | ✅ **yes** — it tests the settings, not the implementation |
| **H** | Job scripts, preflight, gates, launch | A ✅ D ✅ + B C E G + §13.1 | ❌ blocked |
| **I** | The uncertainty decomposition build | F | ❌ blocked on F's findings |
| **J** | One figure script, and the five analyses | 1:1 on details, then the new columns | ❌ blocked |
| **K** | Sync the two documents, fix the bibliography | — | ✅ **yes** — smallest, entirely self-contained |

**Eight can start immediately and run unattended: A, B, C, D, E, F, G, K.** They touch
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

⚠️ **One thing it found is not chat A's to settle.** `affected_molecule_fraction` means 1.0 in the
pipeline and 0.0 in the reference for the conditions with no selection rule. The pipeline's reading
is the one failure mode 6's guard needs — under the reference's convention every uniform condition
reads as zero and trips a guard meant to catch a degenerate condition — but the column is also
compared against the Python injector by chat B's gate, so a one-sided change would break that.
**Whoever merges A, B and the Python side settles it in one edit across all three.**

**Ten more gates** came out of doing the work: the ν ≤ 2 refusal, a mismatched scaffold file
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

**Still open, and not chat A's:** whether the validation split gets its own independently drawn
noise (§5.1 item 5, §13.5 — the author's), the ECFP4 truncation and index-drift guards (§5.1 item
7, chat D), whether Laplace is queued (`NOISE_DESIGN.md` §7 — built either way), and the
`affected_molecule_fraction` convention above.

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

**🔴 TODO in this chat:** whether Laplace is in (§13.3); how many Student-t and Outlier settings
(answered by chat G).

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

#### Chat C — Embedding storage, and the Gaussian-process re-test ✅ DONE 2026-08-26

**What landed** is in §2.8c. Beyond the three storage changes: Avalon was added to both pipelines,
Avalon and ChemBERTa were wired into the experimental runner, the guard is
`scripts/test_embedding_storage.py`, and the measurement is
`scripts/retest_embedding_kernels.py`. **The noise scheme was not touched** — no change to
`NOISE_DESIGN.md` was needed or made.

**Confirming it at full size on the cluster.** The local measurement is at a few thousand molecules;
the harvest is at ten thousand, so only the paired difference transfers. These reproduce the harvest
cell for cell, post-fix. Rebuild the binary first — the record widened, so an old binary reads every
field after the embedding at the wrong offset.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models     # confirm the live checkout first (§2.8b)
git pull
cd rust && cargo build --release && cd ../scripts

for rep in mhggnn mol2vec chemberta avalon continuous_pdv; do
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

- **The `morgan` representation is wired in Rust and not in Python.** `rust/src/main.rs` carries a
  `morgan_buf`, reads it and writes it, but `process_and_train.py` neither builds nor writes it. Ask
  for it today and every record after the first is read at the wrong offset. It is half of the
  repair §5.6 describes; the other half of that section, collapsing the two descriptor-vector names
  into one, changes the same record layout. **Both should be done in one pass over that layout, and
  neither is assigned to a chat.**
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

**🔴 TODO:** which additional embedding to add, if any. Candidates and reasoning are in §13.4.

> **Prompt.** Fix the learned-embedding storage defect in `scripts/process_and_train.py` and re-test
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
3. **The environment (§2.8d).** `scripts/check_environment.py`, wired into the job template and
   into runbook §1b for both cluster interpreters. The local `torch_geometric` import is fixed, so
   the QM9 pipeline runs on the laptop and every other chat can verify locally.

**Two things found that were not in the brief, and both were worse than what was**

- **An unparseable SMILES killed the process rather than taking the error branch.** RDKit's
  binding returns a null pointer as `Ok`, so the fingerprint call dereferenced null: SIGSEGV, no
  message, no partial output. The old `Err(_) => continue` branch was unreachable for the ordinary
  bad-SMILES case. Verified at exit 139 and fixed (§2.7 item 2).
- **`morgan` was written by the Rust writer and has never been read by the Python reader.** Any run
  including it was misaligned by 256 bytes per record. The reader now refuses an unknown
  representation by name, and `morgan` has since been deleted from the writer — it was a leftover
  of the `avalon` work (§2.7 item 5).

**One decision left with you** (§2.8d): `quantile-forest`, `ngboost` and `torchcp` all declare a
newer scikit-learn than the 1.3.2 installed, and the quantile forest is outright broken by it —
it constructs and then fails inside `fit()`. Upgrade scikit-learn before launch, or downgrade the
quantile forest. Run runbook §1b on the cluster first; this may be a laptop-only problem.

**Scope note.** `slurm_scripts_qm9_rerun/*.sh` were not hand-edited — they are generated, and they
still carry the pre-redesign CLI flags, so chat H regenerates them. The environment guard went into
`generate_scripts.py`, where it will land in every regenerated script.

---

#### Chat E — Cross-pipeline parity

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

---

#### Chat F — Uncertainty machinery, reviewed with the author

🔴 **TODO. This is a conversation, not a task**, at the author's explicit request: *"I am not
confident that uncertainty machinery is built and reviewed — I've had repeated issues with this and
we need to go over it 1:1 before the plan is finalized."*

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

#### Chat H — Job scripts, preflight, gates, launch

**Blocked** on A, B, C, D, E, G and the run design in §13.1.

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
> verification gates into a preflight, clear the caches, and launch. The design is `RERUN_PLAN.md`
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

🔴 **Blocked on chat F.** The author's assessment: *"seems like this needs a massive look. We really
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
| §7 of the design had three open items; two were answerable | Positive-control question closed (censoring *is* the label-keyed condition, §3.2); grids closed (the range-finding run set them). **Laplace is the only open item left** |
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
**Laplace** was added as a sixth and is still open. So: four core, plus censoring confirmed, plus
Laplace open. `NOISE_DESIGN.md` §2 lists all six in one table without that structure, which is why
it reads as six.

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
| Embedding values as the model produces them | ❌ each molecule rescaled to fill 0–255 using its own smallest and largest value, then stored as bytes | ✅ returned as ordinary decimal numbers, no rescaling, no quantisation (`KIRBy/src/kirby/representations/molecular.py` — mol2vec `:1565`, MHG-GNN `:2074`, ChemBERTa `:2201`) |
| Features put on a common scale before the model | ❌ only PDV (`process_and_train.py:1800-1809`) | ✅ every representation, fitted on the training fold and applied to the rest (`alternative_data_noise_robustness.py:882-884` for the tree and kernel models, `:967-970` for the neural ones) |

**Three consequences.**

1. **The experimental results for the learned embeddings are trustworthy. The QM9 ones are not.**
   The paper's claim that learned embeddings are weak comes from QM9, which is the side with the
   defect.
2. **The fix is a port, not a design.** Copy what the experimental pipeline already does.
3. **The Gaussian process comparison on the two embeddings has to be re-measured, not argued.** The
   0.87-against-−0.02 gap is *attributed* to this defect. That attribution is an inference. Chat C
   measures it — `scripts/retest_embedding_kernels.py`, which scores the same molecules, the same
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

#### ✅ SETTLED 2026-08-26 — the representation set

*"Drop SMILES from the list, add Avalon and ChemBERTa."*

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
and one-hot SMILES.

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
