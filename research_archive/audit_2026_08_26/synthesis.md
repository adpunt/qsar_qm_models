# Audit conclusion — what to fix before the re-run

Every claim below was read in the source during the audit and survived an adversarial second pass. Two standing cautions on the citations: several files were being edited while the audit ran, so **line numbers drift — cite `rust/src/main.rs` by function name (`fn prepare_ecfp4`), not by line**; and several docstrings in this codebase state the opposite of what the code does, so none of the findings rests on one.

The single sentence that matters: **the worst defects are still identity errors, not parameter errors.** Four representation names and three column names mean different things in the two pipelines, and nothing in the shared settings file or the parity audit can see any of them, because both sides emit arrays of identical shape and dtype.

---

## A. Fix before quoting any number — four names, two meanings each

**1. `ecfp4` on QM9 is a path fingerprint, not a circular one.**
`prepare_ecfp4` calls `rdk_fingerprint_mol`, which binds RDKit's Daylight-style path enumeration (min path 1, max path 7, 2048 bits, two bits per hash); the experimental side computes a genuine radius-2 Morgan fingerprint. Every QM9 row labelled `ecfp4` is a different feature function from every experimental row labelled `ECFP4`, and `paper.tex:203` ("circular substructures with radius r=2") is false for the QM9 half; measured on 1,500 QM9 molecules the two set 50.1 vs 13.9 bits per molecule, and their pairwise similarities agree only weakly.
Where: `rust/src/main.rs:22` (import) and `fn prepare_ecfp4`; `KIRBy/src/kirby/representations/molecular.py:295,308`; joined under one label at `scripts/generate_paper_figures_v2.py:1077`.
Caution for the fix: the crate's own `morgan_fingerprint_mol` is hard-coded to radius 3, so it is **not** a drop-in — a radius-2 call has to be added.
Knock-on: the exclusion of the separate `morgan` representation as "redundant with ecfp4, rho = 0.995" (`generate_paper_figures_v2.py:142`) was measured against the path fingerprint and must be re-derived; so must the ECFP4-vs-PDV correlation at `paper.tex:413` and the SNS exclusion argument at `paper.tex:264`, which asserts both encode overlapping circular substructures.

**2. `sns` is binary on QM9 and a count matrix on the experimental side.**
QM9 genuinely computes Sort-and-Slice counts and then destroys them: `np.array(sns_fp, dtype=np.uint8)` followed by `np.packbits` maps every nonzero count to a single bit, so `sub_counts=True` is a no-op. Trees split differently on counts than on presence bits, and the kernel geometry the SVM and GP see is not comparable at all — so no SNS ranking transfers between the two studies.
Where: `scripts/process_and_train.py:488-490` (destroys), `:1015` (computes), `:1204` (reads back); `KIRBy/src/kirby/representations/molecular.py:1243`.
Two riders: the shared standardisation gate then diverges on the same name — binary QM9 SNS is left unscaled, count-valued experimental SNS is z-scored (`models/model_defaults.py:316-332`, called at `process_and_train.py:2069` and `KIRBy:1139,:1254`) — and QM9 fits the substructure vocabulary on training molecules only while the experimental side fits it on the whole set (`KIRBy:1414`), which is a leakage question worth answering separately.

**3. `pdv` on QM9 is 200 descriptors thresholded at zero.**
`pdv_binary = (pdv > 0)` turns 200 real-valued RDKit descriptors into 200 bits, of which roughly 74 are constant across QM9 (molecular weight, Labute surface area, Chi0, Kappa1, Bertz all collapse to a constant 1), while `continuous_pdv` keeps the same numbers as floats. The paper's `pdv` vs `continuous_pdv` comparison is therefore a ~126-bit sign vector against a 200-dimensional descriptor set, not two encodings of the same information — and the experimental side's PDV is z-scored real values, so **`continuous_pdv` is the only PDV that is comparable across the two studies.**
Where: `scripts/process_and_train.py:496-498` (binarise), `:944-948` (descriptors), `:502-505` (continuous); `KIRBy/src/kirby/representations/molecular.py:941-977`.

**4. "ChemBERTa" is two different pretrained networks.**
The experimental side loads DeepChem's 77M multi-task-regression model (3 layers, 384-wide); QM9 loads seyonec's ZINC masked-language model (6 layers, 768-wide), with a different tokenizer vocabulary and a different pretraining objective. Any cross-pipeline ChemBERTa comparison, and any analysis treating ChemBERTa as one representation spanning both, compares two representations.
Where: `KIRBy/src/kirby/representations/molecular.py:2230-2231` (its docstring at `:2214` wrongly says 768); `scripts/process_and_train.py:869`, buffer at `rust/src/main.rs` (`chemberta_buf`, 3072 bytes = 768 floats).
Rider: `process_and_train.py:910-914` silently zero-pads any embedding to 768, so pointing that path at a narrower model would pad rather than fail.

---

## B. The uncertainty numbers cannot carry the claims made from them

**5. The GP and every variational model write only the epistemic part of their uncertainty.**
The GP reads the latent posterior (`model(x)`, not `likelihood(model(x))`), so observation noise is excluded, and the variational path uses the spread over 100 sampled forward passes; both compute the full predictive spread on the next line and then never reference it. Coverage at one and two standard deviations, the uncertainty-vs-error correlation, and the mean-uncertainty column of table 4 are all computed from an interval that is not the model's own predictive interval, so every "under-confident / over-confident" statement for those models is about an incomplete quantity.
Where: `models/models.py:1846,1864,1888` (GP), `:2252-2262`, `:2856-2868`; total defined at `scripts/utils.py:173`. The same latent-only GP read exists on the experimental side at `KIRBy:838`, so that half is shared.
Do not repeat the earlier claim that this explains the low variational coverage — the calibration temperature can absorb a uniform under-scaling, so treat it as a hypothesis to test, not the cause. The epistemic and aleatoric parts are both written to disk as separate columns, so the total is reconstructible in analysis.

**6. The figure script reads the calibrated column; the shared settings say to read the raw one, and four different quantities are written into that one column.**
All four selection sites try `y_pred_std_calibrated` first and the writer always emits it, so the raw column is unreachable — while `UNCERTAINTY_DEFAULTS['primary_column'] = 'raw'` records the measured reason (NGBoost raw coverage spans 0.546→0.978 across noise levels; calibrated spans 0.637→0.686, i.e. nominal by construction). Worse, that single column holds temperature-scaled values for NGBoost and the GP, unscaled values for the quantile forest, the epistemic part alone for graph models, and the full spread for the graph GP, so the mean-uncertainty column compares four different quantities under one heading.
Where: `scripts/generate_paper_figures_v2.py:757-761, 3076-3079, 3369-3373, 3511-3515`; `models/model_defaults.py:364`; `models/models.py:1410, 1543, 1860, 3345, 3546`. The figure script never imports the settings file.

**7. QM9's `injected_noise` column is float dust, and the analysis correlates uncertainty against it.**
Test labels are never noised (correctly), so the writer records exactly 0.0 for every QM9 uncertainty row; the figure script then overwrites that column with the residuals of a regression of the noisy label on the clean one, which for an affine transform is rounding error — and prints "Fixed injected_noise via linear regression", which reads as a repair. Every "uncertainty tracks the injected noise" number for QM9 is a correlation against rounding error, and because the residuals are not exactly constant it returns a finite, meaningless value rather than a missing one.
Where: `scripts/utils.py:322`; `scripts/generate_paper_figures_v2.py:980-1020` (called unconditionally at `:3997`), consumed at `:1047, :3427, :3558`.
Extra hazard: that recomputation groups without the data-file identifier (`:995`), so a group spanning two files with different standardisation constants produces a systematic, entirely spurious signal. The columns needed to do this properly — `split`, `noise_scale`, `noise_pattern`, `noise_pattern_pred` — are already written and the figure script reads none of them.

**8. The experimental side's `noise_pattern` column describes a selection that was never injected.**
For the two conditions where the selection is random (the 1/5/10 percent outlier conditions and the widened-group condition), the pattern is built from an injector seeded at level 0.0 while the injection at each level uses a differently seeded one; measured overlap is chance (7 of ~105 molecules for the 5% outlier condition; 2 of ~12 group identifiers for the widened-group condition). Any "does the model learn where the data is unreliable" result scored against that column for those two conditions is a null by construction, and the neighbouring `noise_scale` column, which is correct, disagrees with it in the same row.
Where: `KIRBy/tests/alternative_data_noise_robustness.py:1164, :1184, :1208-1209` (and `:1269, :1284, :1319-1320`); selection drawn at `NoiseInject/noiseInject/core.py:313-326`. QM9 already solves this — its noise stream is seeded from the run seed and split only, never from the level (`rust/src/main.rs:703-712`) — so copy that.

**9. The out-of-fold control column was empty in every experimental uncertainty file produced before today.**
A guard tested a dictionary key that was only written at the end of the same loop pass, so it was never true and `noise_pattern_pred` was blank on every out-of-fold training row. **This is already fixed in your uncommitted working tree** (`KIRBy:1226-1236, :1356-1366`, helper at `:1034`) — commit it and ship it; the action item is that every existing uncertainty file must be regenerated, because the question that column exists to answer cannot be defended from them.

**10. The validation uncertainty loader mixes out-of-fold training rows with held-out rows, then deletes most of both.**
It accepts any file with the four expected columns and never looks at `split`, and its de-duplication key omits fold, condition and split — so what survives is largely out-of-fold training rows from one arbitrary fold of one arbitrary condition, and the stable molecule identifier the experimental writer provides for exactly this purpose is never read. The files currently in the repo are the older format, so nothing published is wrong yet; this lands the moment the re-run output is copied back.
Where: `scripts/generate_paper_figures_v2.py:1217-1234`; writer at `KIRBy:1946-1969` (test rows) and `:2011-2027` (training rows), molecule identifier at `:1956, :2017`.

---

## C. Fix these together — they all change what a noise level means

These six are one problem. Fixing any one alone leaves a number that looks comparable and is not.

**11. No QM9 run exists under the redesigned scheme.** All 22 job scripts pass the retired `--sigma` and `--noise-strategy`, which the parser refuses by name, exiting before anything is built; the six condition names they loop over are the retired set, of which only `outlier` survives. Any table today putting a QM9 number beside a LogD/Caco-2/hERG number compares an old-scheme result against a new-scheme one.
Where: `slurm_scripts_qm9_rerun/qm9_rf.sh:49, :83-84` and all 22 scripts; refused at `scripts/process_and_train.py:397-406`; the generator is equally stale (`generate_scripts.py:90, :207-208`), so regenerating from it fixes nothing.

**12. QM9 result rows carry no condition column at all.** The condition is recovered by matching the output filename against eight retired names, and `outlier` is a valid name in *both* schemes — so the three noise shapes run under outlier targeting would all be labelled `outlier`, be de-duplicated against each other, and silently reduce to whichever file was written last. The experimental side writes the condition, the shape and the targeting on every row; QM9 should too.
Where: `scripts/utils.py:23-26, :65-72`; parsed at `generate_paper_figures_v2.py:558-562, :579-584`; de-duplicated at `:643`; crashes at `:4037` in the mixed case.

**13. The experimental neural models standardise their targets using the noisy spread; QM9 was explicitly fixed not to.** The comment in the Rust writer names the noisy-spread version as "the old behaviour" that "moved the target scale with the noise level". The forward and inverse scalings cancel, so this is not the large mechanical inflation of uncertainties it first appeared to be — but the two pipelines are solving differently-scaled problems at every nonzero level, so fix it for agreement and measure the change rather than asserting it.
Where: `KIRBy/tests/alternative_data_noise_robustness.py:868-873`, fed from `:1310-1312`; `rust/src/main.rs:1896-1903`.

**14. `sigma` names three different physical quantities and the robustness metric divides each by its own range.** On QM9 it is a fraction of the clean label spread; on the experimental sets it is the label's own log units, with a different span per dataset (LogD 0–1.0, Caco-2 0–0.7, hERG 0–1.1); for the censoring condition it is the fraction of labels clipped, 0–50%. The normalised area under the retention curve is therefore "mean retention across whatever axis this configuration happened to have", and three places put those side by side and average them — a model can rank differently purely because one dataset's grid runs further than another's.
Where: `KIRBy:170-174, :180, :1789-1794`; `scripts/process_and_train.py:290-293`; metric at `generate_paper_figures_v2.py:1791-1800`; mixed at `:1712, :1719, :1729-1730`. Within one dataset every model shares a grid, so within-panel rankings are safe; the damage is to the cross-dataset ranking and the transferability comparison, where the design is unbalanced.
Worth stating in the paper either way: in units of published assay error those three axes run to 6.7, 2.0 and 2.0 — they are not matched.

**15. The figure script looks for a column the experimental writer no longer produces, and still lists the six retired condition names.** It groups on `strategy`; the experimental side writes `noise_type`, and no rename exists — so no validation figure or table can be produced at all. Renaming the column alone is worse than leaving it, because the *values* also changed: the hardcoded order list still holds the six retired names, so every dataset panel would quietly come out empty instead of crashing.
Where: `generate_paper_figures_v2.py:1364, :1483-1486, :1539, :85`; writer at `KIRBy:1855, :2163`; the old names now raise on the experimental side (`KIRBy:1581-1583`).

**16. The condition names diverge for anything off the catalogue.** The eleven catalogued conditions agree exactly between the two injectors, including the censoring grid, but a shape-and-targeting pair outside that list is named one way by the Python injector and another by the Rust one, and the command-line keyword (`grouped_wide`) differs from the emitted name (`grouped_wider`). Stay on the catalogued set, or make both name it identically.
Where: `NoiseInject/noiseInject/core.py:248-259`; `rust/src/main.rs` (`condition_name`, `:297-325`).

---

## D. The regeneration is blocked — loud failures, nothing wrongly published

**17. The results glob also picks up the noise-manifest file the same run writes beside it.** Its rows carry no model, representation, level or R², they survive every filter, and the figure run dies with a type error when it sorts the model names. Add a skip for it beside the existing one, or require the results columns before appending a frame.
Where: `generate_paper_figures_v2.py:564, :568-569`; sidecar written at `process_and_train.py:1780`. (The per-epoch half of this is not live — that output is off by default.)

**18. The validation de-duplication collapses five cross-validation folds and seven conditions into one row.** The key is effectively dataset, model, representation and level, so 34 of every 35 rows are deleted and the survivor is arbitrary. It is masked today only because the run dies first, at the missing column above — fix that one without this one and you get confidently wrong numbers instead of a crash.
Where: `generate_paper_figures_v2.py:1151-1152`; the experimental writer stamps fold and condition on every row (`KIRBy:1853-1857`).

**19. Uncertainty filenames strip hyphens, so the four Bayesian and variational models never join to their own accuracy rows.** They arrive as `BNNFULL`, `VBLLFULL`, `MLPBNNFULL`, `MLPVBLLFULL` in one table and as `dnn_bnn_full`, `dnn_vbll`, `mlp_bnn_full`, `mlp_vbll` in the other, so any side-by-side reading silently drops all four — which is exactly the comparison figures 4 and 5 are waiting on. The same patch was already applied for representation names and not for model names.
Where: `KIRBy:1522`; name map at `generate_paper_figures_v2.py:1070-1074`; the filename overwrites the in-file model column at `:1219`. Two independent fixes: add the stripped keys to the map, and prefer the in-file column when one exists.

**20. An experimental job run on a subset of conditions overwrites the whole results file.** The merge-with-existing guard tests only the model and representation filters; the condition and level filters are in neither the guard nor the removal mask, and the write is unconditional — so splitting a run by condition, which the flag's own help text describes as how the uncertainty jobs are spread across the queue, silently deletes the other conditions. With a model filter it is subtler: that model's rows are dropped for every condition and only the one just run is added back.
Where: `KIRBy:2116-2130` (and `:2177-2189`, `:2420-2432`); the output directory is per-dataset, not per-condition, so the collision is real under the documented workflow.

---

## E. Code paths that produce nothing — fix before turning them on

None of these has produced a number in the paper. All of them will bite the moment the corresponding models are run.

- **Graph models get the wrong molecules entirely.** The shuffle inside the split function rebinds a local copy (the library returns a new object), so the caller keeps the unshuffled dataset while the indices and the stored labels are in shuffled order — every graph is paired with an unrelated molecule's label, and the scaffold split is void for these models. `scripts/process_and_train.py:726-727, :2169, :2178, :1646-1648`.
- **Attaching noisy labels to graph objects is a no-op**, so every graph fit raises on the first batch — and because the graph models run first, including them alongside other representations discards the *entire* repetition, not just the graph cells, with one line in the job log and nothing in the results file. `scripts/process_and_train.py:1637-1643`; consumers at `models/models.py:3248, :3313, :3329, :3430, :3486, :3510`. The conformal graph path is not affected by this one.
- **The graph GP unpacks three return values into two** and dies before it can write anything; even fixed, its per-molecule uncertainty is a constant (the posterior variance is never computed — only the mean), and the constant it uses is the observation noise relabelled as the epistemic part. `models/models.py:3521-3522, :3500, :3518`.
- **The graph network's Bayesian path calls a two-argument function with one argument**, so no Bayesian graph uncertainty has ever been produced; if any row in a results file claims one, trace where it came from. `models/models.py:3279`.
- **Conformal graph models train on permuted labels**, re-randomised every epoch, because the data list is rebuilt from a shuffled loader and labels are then assigned by position. Coverage will still land near nominal — that is what split conformal guarantees — so coverage is not evidence anything worked; the interval width and the accuracy are numbers for a model that learned nothing, at every level including the clean baseline. `models/models.py:3966-3968, :3979-3981`; loader at `process_and_train.py:1650`.
- **The last-layer Bayesian flag matches no branch** — the job scripts pass `last`, the code tests `last_layer`, and there is no list of allowed values — so those models train as ordinary deterministic networks in evaluation mode and write a column of exact zeros as their uncertainty. One-character fix, plus adding an allowed-value list so the next typo fails loudly. `slurm_scripts_qm9_rerun/generate_scripts.py:80-81`; tested at `models/models.py:2194, :2550, :2786`; flag defined without constraints at `process_and_train.py:339-352`.
- **The neural-tangent graph model is unreachable and would fit clean labels if it were reached** — it takes the noisy labels as arguments and never mentions them again, reading the untouched dataset attribute instead. Its robustness curve would have been flat by construction. `models/models.py:7147, :7229, :7280`.
- **The non-QM9 loading path misaligns Sort-and-Slice features with labels** for training rows (queue filled in split order, drained in ascending order). Nothing has ever run it — every job script requests QM9 — but fix it before any run on the other datasets. `scripts/process_and_train.py:625-628, :644, :671-672`.

---

## F. Already handled — do not redo this work

`models/model_defaults.py` and `scripts/audit_pipeline_parity.py --strict` already own the following, and the audit found no reason to reopen any of it:

- **Every tree and kernel model's parameters on both sides.** Both pipelines read one spec; the audit loads it *through both pipelines* and fails if the hashes differ, then builds every model with the installed libraries and diffs the effective parameters against a stored baseline. The tree-count and learning-rate divergences that motivated it are closed and cannot silently return.
- **Library-default drift** — an upgrade that moves a default you do not pin shows up in that diff rather than as a changed result later.
- **Environment facts that are not parameters** — the NGBoost scoring rule and natural-gradient default, whether the Bayesian optimisation fitter accepts a plain GP, that no job script passes the tuned-parameters flag, the Rust binary, and every package version.
- **The feature-standardisation gate.** `should_standardise` decides from the data rather than the name, both pipelines call it, and QM9 fits on the training split alone. The one live consequence is listed above under SNS, and it is a consequence of the *content* divergence, not of the gate.
- **The known-differences list is already written down** with file and line: the ECFP4, PDV, SNS and stored-embedding identity errors; the seven model families that merge validation into training (about a 29% difference in effective training-set size); ten repetitions on QM9 against one on the experimental side, which is your decision of 2026-08-26 and which the paper has to state, since it leaves the experimental analysis with no run-to-run error term; label standardisation; and calibration. Those are open items with an owner, not new findings.

**But the clean bill is narrower than the version hash suggests.** These read as covered and are not: the Bayesian settings block is imported by both pipelines and referenced by neither — every prior and initialisation constant is a literal in both files (they match today); the uncertainty settings block is never read by the figure script; most neural settings keys (optimiser, weight decay, epochs, dropout, activation, batch size, restore-best-weights, validation loss reduction) are restated as literals, and batch size is 32 in one place and 64 for the graph loaders; the GP's noise and output scale are hard-coded at the experimental call site rather than read; and the GP kernel comes from a command-line default, not the spec. Changing any of those in the spec would move one pipeline and not the other, while both results files carry the same version hash asserting parity. Either wire them up or delete them from the spec so they stop reading as a guarantee.

---

## G. What no automated check can catch — a human, every time a featuriser changes

The parity audit compares parameters. Every finding in section A is invisible to it, because both pipelines emit arrays of the same shape and dtype under the same name. These are the questions that need a person to answer by reading the code, each time any featuriser, storage format or pretrained model changes:

1. **Which library function does this name actually bind to?** Path enumeration versus circular environments, radius 2 versus radius 3. Following the binding through to the C++ wrapper is what caught the ECFP4 error; nothing shorter would have.
2. **Does the storage format preserve what the featuriser computed?** Bit-packing turns counts into presence flags, and an unsigned byte cast wraps. Counts were computed, configured on purpose, and thrown away one line later.
3. **Is a continuous quantity thresholded anywhere on its way to disk?** The PDV binarisation is a single comparison against zero, in the writer, far from the featuriser.
4. **Which checkpoint does this model name load, and how wide is it?** One docstring claims 768 for a 384-wide model, and one path silently pads any embedding to 768 rather than failing.
5. **Are the feature rows still lined up with the labels?** Three separate mechanisms broke this — a queue filled in one order and drained in another, a shuffled loader rebuilt into a list and then labelled by position, and a library call that returns a new object instead of mutating the one you passed. All three are silent; two of them look correct at a glance.
6. **What happens to a molecule that cannot be featurised?** The Rust writer records a failure and refuses to finish; the experimental builders append a row of zeros with no log line, and that row gets a real label and is trained on. Opposite policies for the same event.
7. **Does this column name mean the same physical quantity in both writers?** `sigma`, `injected_noise`, `pdv` and `noise_pattern` each currently mean two different things.
8. **Never take a docstring as evidence here.** Several are provably wrong about their own code — the 768-dimension claim, the "IQR-based" comment on an arithmetic that is not the interquartile range (correct as written; do not "fix" it), the module header quoting dataset sizes from a file the script never opens, and a repair message printed over a computation that repairs nothing.

A cheap standing guard for most of items 1–4: assert the featuriser's identity from its *output* at run time, on a handful of fixed reference molecules with known answers, and store those signatures beside the results. A parameter diff will never see any of this; a fixed set of reference fingerprints would have caught the ECFP4, SNS and PDV errors on the first run.