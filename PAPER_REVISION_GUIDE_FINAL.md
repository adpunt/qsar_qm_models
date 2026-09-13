# NoiseInject Paper: Revision Guide (final)

> Whole-paper revision guide. Same shape as `PAPER_REVISION_GUIDE.md`, which it supersedes.
>
> **`paper.tex` is never edited.** The copy in this repository is a read-only reference — the live
> manuscript is the Overleaf project, and the local file is copied down from it, never up. Every
> change to the paper is written here, as replacement text, and moved across by hand.
>
> **Only what can be written now is written.** That is the Methods, in full, and the Limitations
> paragraph. Everything else is a heading and one line saying what it waits for. Those units need
> the re-run, and drafting them before it would mean inventing findings and then rewriting them.
>
> **Part One is sourced from the code, and has now been through two passes.** The second pass
> (12 September 2026) re-read every claim against the pipelines that implement it and **corrected 74
> of 302**; 31 more could not be established from code at all. Nothing comes from `RERUN_PLAN.md`,
> from a code comment, or from memory. Where a number could not be established it is marked `TODO`
> rather than guessed. The corrections that changed what a sentence says are listed under each
> subsection as **Corrected against the code in this pass**, so you can see what moved.
>
> **What is left is nine code defects, not writing choices.** They are in `HANDOFF.md`, each verified
> at a line number, and each names where it lands in the Methods. Where the text needed a call
> between two defensible forms I made it and said which comparable paper it came from, so any of them
> reverses in one edit. Four numbers are `TODO` because they wait on a run or a read.
>
> Line numbers are from `paper.tex` as of 4 September 2026 and were each confirmed by searching for
> the quoted text. They will drift as you edit, so anchor on the quotation.
>
> **One boundary inside the Methods.** The metrics and statistics computed *after* a run —
> auc_norm, the ANOVA, Kendall's *W*, coverage, the uncertainty statistics — are being handled
> elsewhere. §M6 holds a place for them and lists only the fixes that are certain either way.

---

## Where this guide stands

| Unit | State |
|---|---|
| Methods, M0–M7 | **Drafted**, second pass applied. Nine code defects in `HANDOFF.md` block four sentences |
| Limitations | **Drafted** |
| Results | **Outlined** — structure and figure set fixed, prose waiting on the re-run |
| Everything else | Heading only — waiting on the re-run |

---

## What broke, and why the Methods cannot be patched

The Methods in `paper.tex` describes a study that no longer exists in the code. Not in emphasis — in
substance. Nine things changed, and each one forces text:

1. **The six noise strategies are gone.** Gaussian, Outlier, Quantile, Threshold, Value-proportional
   and Heteroscedastic were, on measurement, one strategy at six doses. They were replaced on
   2026-08-27 by seven conditions, each an explicit (shape, targeting) pair, and by dose matching,
   which makes the noise level the amount actually delivered rather than a knob each condition
   interprets its own way. Table 1 (`tab:regression_noise`, lines 318–350) describes four conditions
   that exist in neither injector.
2. **The noise axis changed.** Eleven levels of 0 to 1.0 became seven levels of 0, 0.2, 0.3, 0.5,
   0.75, 1.0, 1.5, read as a fraction of the clean training label spread. Censoring runs a separate
   axis of clipped fractions.
3. **Validation labels now carry noise.** Line 313 says validation and test are both clean. Test
   still is, on both pipelines and by hard wiring rather than by a flag. Validation is corrupted by
   default, from an independent draw at the same amount.
4. **The representation set changed.** mol2vec is deleted and one-hot SMILES is refused by name;
   Avalon and ChemBERTa are in and appear nowhere in the paper. Sort & Slice now carries
   substructure counts, not bits, so the paper's "two binary fingerprints" framing is wrong about
   which two.
5. **The model roster is 19 configurations, not 11.** The heteroscedastic Gaussian process, the two
   heteroscedastic VBLL networks and the two mean-variance networks are in the run and absent from
   the paper. Four Bayesian transformations are applied to each base network, not three, and the
   last four configurations are the ones line 218 says were not decomposed.
6. **The uncertainty decomposition is real machinery now.** One shared definition, five routines, a
   declaration on every row of whether each half varies per molecule, and a check that refuses a row
   disagreeing with it. Line 218's list of what was not decomposed is now wrong in both directions.
7. **hERG is 1,415 compounds, not 1,482** (lines 197 and 556).
8. **Several load-bearing implementation choices are new and undescribed**: the split is no longer
   DeepChem's; target standardisation uses the clean training statistics; the Gaussian process
   lengthscale is initialised from the data and a collapsed fit is flagged and keeps its accuracy
   score while losing its per-molecule uncertainty; Gaussian processes are capped at 5,000 training
   molecules on QM9; the full BNNs now carry a KL term that they did not have when the paper's
   numbers were produced; and four Bayesian network configurations run at tuned settings rather than
   at the shared default.
9. **QM9 and the three assay datasets are two implementations, not one method**, and they differ in
   thirteen ways that change how a result reads. This is the largest finding of the second pass and
   it has its own subsection, M0.

---

# PART ONE — METHODS

Six subsections become seven. The added one carries material currently squeezed into the last
paragraph of Models, which has grown too large to sit there.

| Current | Becomes | Change |
|---|---|---|
| Dataset (192–200) | **M1 Datasets** | Rewrite. New pool count, replicates redefined, split re-described, standardisation stated, assay counts corrected, fold structure added |
| Molecular Representations (201–206) | **M2 Molecular Representations** | Rewrite. Six representations, two new to the text, two removed |
| Models (207–219) | **M3 Models** | Rewrite, and the last paragraph moves out |
| Performance Metrics (220–311) | **M6 Performance metrics** | **Hold.** Owned elsewhere |
| Noise Strategies (312–362) | **M4 Label noise** | Full rebuild, new table |
| — | **M5 Uncertainty quantification** | New subsection |
| NoiseInject Framework (364–373) | **M7 NoiseInject framework** | Three sentence fixes |

The current Methods is about 3,200 words of prose. The drafts below come to about 4,000 — a quarter
longer, not double, and the cuts made on the literature's advice paid for roughly half the growth.
Ten comparable benchmarks run 1,740 to 6,000 words of Methods, four of them in this journal, so
there is room either way; what matters is where the words sit.

**Naming.** The three measured datasets are called **assay datasets** throughout. The code calls
them "validation" everywhere, which collides with the early-stopping validation split inside each
fold; nothing is being renamed in the code, but the paper should not use that word for them.

---

## M0. The thing that broke every draft before this one: two pipelines, not one method

This is the largest finding of the re-check and it shapes every subsection below. QM9 and the three
assay datasets are **two separate implementations of the same design**, and they differ in ways a
reader would never guess and that change how a result reads:

| | QM9 | Assay datasets |
|---|---|---|
| Split | one 80/10/10 scaffold split per replicate | 5-fold scaffold cross-validation, one deterministic partition |
| Repetition | 10 replicates, each a new 10,000-molecule draw and a new split | one fit per cell |
| Fitted share | 80% of the sample | about 65% of the dataset |
| Noise dosed against | that replicate's clean **training** label spread | the **whole** clean label column's spread |
| Noise drawn | per split | once over the whole column, then indexed by molecule |
| Grouped conditions | affected groups chosen on the training split, so held-out molecules are unaffected | affected groups chosen over the whole column, so held-out molecules **can** be affected |
| Conditions run | 3 at full breadth, 3 more on named pairs, censoring on named pairs | 3 at full breadth; the other four were not submitted |
| Level zero | run only for Gaussian and censoring; copied to the rest | run for every condition |
| Validation split | every model has one, and it is noised | only the neural models have one |
| Temperature calibration | fitted on validation, recorded | none |
| GP training cap | binds (5,000 of ~8,000) | never binds |
| NGBoost early stopping | on the held-out validation split | on a fifth it carves from its own training rows |
| Noise injector | Rust | Python (`noiseInject`) |

Every one of those rows is code-verified, and the last three are being fixed rather than described
(`HANDOFF.md` items 1, 2 and the confirmation at the end).

**The drafts below scope every sentence** — "on QM9 …; on the three assay datasets …" — rather than
writing about "the study" and leaving a reader to guess. It costs perhaps eighty words across the
whole Methods and it is the only form that is true everywhere. The table above is the supplementary
version if you would rather state the differences once and write unscoped prose; that is a smaller
Methods and a reader who has to hold a table in their head.

---

## M1. Datasets: full rewrite (replaces lines 192–200)

**Why wholesale:** five of its factual claims are wrong (the subset is not fixed across replicates,
the splitter is not DeepChem's, six strategies and eleven levels are gone, hERG's count is wrong,
and the standardisation sentence is silent on the two things that were fixed), and the assay
datasets need their fold structure stated because their error bars mean something different from
QM9's.

> The majority of experiments were conducted on the QM9 molecular property dataset
> \citep{Ramakrishnan2014}, which contains approximately 130,000 small organic molecules with
> pre-computed quantum-mechanical properties obtained from density functional theory (DFT)
> calculations at the B3LYP/6-31G(2df,p) level of theory \citep{Ramakrishnan2014}. Molecules flagged
> as uncharacterised in the original release, and a further set that could not be processed by
> \texttt{RDKit}, were excluded, leaving a pool of 129,428 molecules. Each experiment was replicated
> 10 times; each replicate draws its own random subset of $N = 10{,}000$ molecules from that pool
> and generates its own split, so replicates differ in which molecules they contain as well as in
> how those molecules are divided. Preliminary experiments were done with $N = 5{,}000$,
> $10{,}000$, $20{,}000$, and $30{,}000$, resulting in similar performance at all sizes except
> $N = 5{,}000$ which experienced minor degradation.
>
> We selected the HOMO--LUMO energy gap as the primary prediction target in QM9, as it captures
> electronic characteristics relevant to molecular reactivity and stability \citep{Islam2019,
> Hllermeier2021, Fediai2023}. Labels are used in electronvolts; across a replicate the clean
> training labels have a mean near 6.86\,eV and a standard deviation near 1.28\,eV, and that
> training standard deviation is the scale against which injected noise is dosed. We also assumed
> that thanks to the consistent calculation method, and notwithstanding approximations in the level
> of theory used, the QM data were ``free'' of noise.
>
> For each replicate, a scaffold-based train/validation/test split (80/10/10) was generated on
> Bemis--Murcko scaffolds computed with \texttt{RDKit} \citep{rdkit}. Acyclic molecules have no
> Murcko scaffold; rather than pooling them into a single empty-scaffold group, each distinct
> acyclic structure forms a group of its own. Groups are placed in random order, and each is
> assigned to the first of training, validation and test it still fits in under an 80\% and a 90\%
> cumulative cap, with the remainder going to test. The distinction matters because pooling the
> acyclic molecules places all of them in the training set and removes them from evaluation
> entirely; in a typical replicate they are about a tenth of the sample. Scaffold splits were used
> to minimize structural overlap between sets. While predictive performance typically drops when
> switching from random splitting to a more challenging scenario like scaffold splitting, the use of
> scaffold splitting discourages the QSAR model from overfitting \citep{Heid2023}.
>
> Prior to modeling, molecules were parsed and sanitized with \texttt{RDKit} and rewritten as
> canonical SMILES without stereochemistry \citep{rdkit}; anything \texttt{RDKit} could not parse was
> dropped. Every representation is therefore computed from a stereochemistry-free structure. Target
> values were standardized to zero mean and unit variance using the mean and standard deviation of
> the \emph{clean training} labels alone, so that the target scale does not move with the amount of
> noise injected; noise is added to the raw label before this step, and errors are converted back
> into the label's own units before being reported. Data loading and pre-processing were performed
> from a fixed base random seed, from which each replicate derives its own. The molecule subset and
> the split depend on the replicate index alone, so a given replicate holds the same molecules,
> divided the same way, under every noise condition, every noise level and every representation.
>
> To assess whether noise robustness results generalize beyond the HOMO--LUMO gap, we evaluated the
> same models and representations on three datasets with experimentally measured endpoints. From the
> OpenADMET initiative \citep{openadmet} we used LogD (lipophilicity, $N = 5{,}039$) and Caco-2
> efflux permeability ($N = 2{,}161$), the latter modelled as $\log_{10}$ of the efflux ratio. We
> also used hERG binding data from ChEMBL \citep{Zdrazil2023}, extracted following a protocol
> inspired by \citet{landrum2024}: binding assays only, unqualified pChEMBL values deduplicated to a
> median per compound, and compounds whose inter-assay standard deviation exceeded 1.0 log unit
> removed, giving $N = 1{,}415$ compounds. For all three datasets, structures were standardized by
> retaining the largest covalently bonded fragment and canonicalizing with \texttt{RDKit}, and
> duplicate structures were collapsed to their median label. The clean label standard deviations are
> 1.19, 0.44 and 0.91 log units respectively, and each is the scale against which noise is dosed on
> that dataset.
>
> Each assay dataset was evaluated under five-fold scaffold cross-validation using
> \texttt{GroupKFold} on Murcko scaffolds; within each fold a further fifth of the training block's
> scaffold groups was held out as an early-stopping set, so approximately 65\% of a dataset is
> fitted in any one fold. Labels are standardized within a fold on that fold's clean training labels,
> as on QM9. The five folds are a single deterministic partition, and each configuration is fitted
> once rather than repeated under new seeds; the noise is likewise drawn once over the whole label
> column and indexed by molecule, so a molecule carries the same corruption in whichever fold it
> falls. The assay results therefore carry variation across folds, which mixes sampling with
> scaffold difficulty, but no run-to-run error term.

**What that last sentence buys you, and why it is not optional.** With one fit per cell there is no
estimable residual term in an assay-side variance decomposition, and no assay model-versus-model
comparison carries a run-to-run error bar. A reader who assumes the fold spread is a replicate
spread will over-read every assay comparison in the paper. State it once here and the Results can
lean on it. The second half — that even the noise draw is fixed across folds — makes the claim
stronger than the earlier draft admitted.

**Corrected against the code in this pass.**

- The acyclic share is **about a tenth of a replicate**, not "close to half". 1,024 of 10,000 in
  replicate 0, reproduced from the pipeline's own sampling code. The 42.5% in the splitter's
  docstring is the rate in the first 2,000 rows of QM9 *in file order*, and QM9 is ordered by
  molecule size. The earlier draft carried the docstring's number into prose.
- The label spread is **1.28 eV, not 1.27**, and it varies 1.27–1.30 across draws. 1.27 was one
  replicate. The dose anchor is specifically the clean **training** split's population standard
  deviation, which is a different 80% subset from the replicate the earlier draft attributed it to.
- Group assignment is **first-fit under cumulative caps**, not "until each reaches its quota". Test
  has no cap and takes the remainder; a group too large for training does not block a later smaller
  one.
- Acyclic groups are keyed on the **stereochemistry-free canonical SMILES**, not on the row index,
  so two copies of one acyclic molecule stay in the same group and the same split.
- Sanitisation **drops** what it cannot parse; nothing repairs a valence state. "To remove invalid
  valence states and standardize atom and bond typing" describes work the code does not do.
- The assay early-stopping carve is **20% of the training block's scaffold groups**, not of its rows;
  the realised row share runs 16.5–22.5%. The fitted share is **65%** (measured 0.620–0.668), not
  "approximately two thirds", which was close but was arrived at from the wrong arithmetic.

**Blocked on a fix, not on a choice.**

- **The hERG sentence above cannot be written yet.** The binding-assay filter, the median collapse
  and the inter-assay standard-deviation filter are real code that **does not run**: the loader
  returns a cached two-column CSV before reaching any of them. The release-36 stamp and the
  release-37 re-check live only in a provenance file that says it was reconstructed on 2026-09-04.
  `HANDOFF.md` item 8 settles it by fetching into a separate file and comparing — **not** by
  re-fetching onto the working path, which would change N and invalidate every assay result. Until
  that comparison comes back, the fallback wording is *"a cached extract of 1,415 compounds prepared
  by the protocol of \citet{landrum2024}"*, with the release numbers dropped.
- **"Could not be processed by \texttt{RDKit}" rests on a comment.** No script regenerates
  `data/valid_qm9_indices.pth`. `HANDOFF.md` item 7. If it comes back unreproducible, the clause
  goes and only the count stays.

**Decided here, so you do not have to.**

- **The HOMO--LUMO paragraph is cut from 110 words to two clauses.** Heid et al. (JCIM 2023) select
  the same target with "as well as the HOMO-LUMO gap as a size-intensive property"; Kolmar & Grulke
  justify all eight of their endpoints in one sentence. The original is at `paper.tex:194` if you
  want it back — it is the only correct prose I cut from M1.
- **The preliminary sample-size sentence stays.** Nothing in the repository sets up that sweep, so it
  is yours rather than the code's, but one-sentence pilot justifications are house style: Scalia et
  al. (2020) and Kolmar & Grulke both carry the identical move.
- **The dose-anchor difference is stated by saying each dataset's own spread is its scale**, which is
  true on both sides without a sentence about the difference. QM9 doses against the clean *training*
  labels; the assay runner doses against the whole clean label column, deliberately, so a molecule's
  corruption does not depend on which fold it lands in.
- **The ChEMBL field list goes in once item 8 reports** — target, standard type, pChEMBL not null,
  relation, data-validity comment, about thirty words. Landrum & Riniker name theirs verbatim and you
  cite them for the protocol.


---

## M2. Molecular Representations: full rewrite (replaces lines 201–206)

**Why wholesale:** it describes six representations, two of which are no longer in the study, and
omits two that are (Avalon, ChemBERTa). It also calls Sort & Slice binary, which it no longer is,
and attributes feature extraction to Rust, which does none.

> In this study we evaluated six molecular representations spanning circular fingerprints, a
> substructure-count fingerprint, physicochemical descriptors, and pretrained embeddings. Two are
> binary vectors of 2048 bits. The ECFP4 fingerprint uses circular substructures with radius $r=2$,
> hashing them into a standard $d=2048$-bit vector using \texttt{RDKit} \citep{rdkit}. Each bit
> encodes the presence of one or more substructures capturing the local chemical environment around
> each atom. One downside of hashed fingerprints is bit collisions, in which chemically distinct
> substructures map to the same index, resulting effectively in representation-level noise. The
> Avalon fingerprint \citep{avalon} hashes a different, fixed enumeration of structural features,
> and is included so that no conclusion about fingerprints rests on a single hashing scheme.
>
> To address bit collisions directly, the \emph{Sort \& Slice} (SNS) fingerprint \citep{sns} is a
> collision-free alternative based on the same Morgan circular substructures with radius $r=2$.
> Substructures are sorted by prevalence in the training set, and the top $L$ ($L=1024$) are sliced.
> We retain the substructure \emph{counts} rather than reducing them to presence bits, so SNS is a
> small-integer count vector rather than a binary one. The vocabulary is fitted on training
> molecules only, and is refitted for every replicate on QM9 and within every fold on the assay
> datasets, so no held-out structure enters the feature basis. A molecule whose count vector is
> identically zero under the fitted vocabulary carries no information in this representation, and is
> removed from every representation rather than from SNS alone, so that all six are compared on the
> same molecules; on QM9 this removes none, one or two molecules from a replicate's 10,000. Part of
> the QM9 grid completed before this rule was applied, and runs from either side are reported
> together: the measured difference in clean $R^2$ between matched runs is smaller than the spread
> already present between replicates of the same model, representation and noise condition
> (Additional file~[N]).
>
> We also used 200-dimensional physicochemical descriptor vectors (PDVs) computed from RDKit
> molecular descriptors (\texttt{MolecularDescriptorCalculator}) \citep{Cherkasov2014}.
>
> Two pretrained learned representations were included. ChemBERTa \citep{Chithrananda2020}
> embeddings were taken from the \texttt{ChemBERTa-77M-MTR} checkpoint as the mean over non-padding
> token embeddings, giving a 384-dimensional vector. The tokenizer distributed with this checkpoint
> falls back to single characters, so the encoding does not distinguish two-letter halogens, formal
> charges, stereocentres or azole tautomers, and a proportion of molecules consequently receive
> identical embeddings (TODO\% on QM9, TODO\% on hERG). Results for ChemBERTa here should be read as
> results for this checkpoint rather than for transformer embeddings in general. As a representative
> GNN-based learned representation, we included graph embeddings generated by Molecular Hypergraph
> Grammar GNNs (MHG-GNN) \citep{kishimoto2023}, a GIN-based autoencoder pre-trained on 1.34 million
> PubChem molecules using $\beta$-VAE loss, producing 1024-dimensional embeddings through iterative
> message passing.
>
> No representation is rescaled per molecule. The continuous representations (PDV, ChemBERTa and
> MHG-GNN) are z-score normalized per feature, with the mean and standard deviation computed on the
> training set and constant-variance features set to unit scale, while binary fingerprints and
> substructure counts are passed to the model unscaled. Stereochemistry is absent from every
> representation: the substructure fingerprints are generated without chirality, and on QM9 the
> SMILES they are computed from are canonicalized without it. Representations were precomputed in
> Python using \texttt{RDKit} and the two pretrained encoders; on QM9 a Rust component performs
> label processing, noise injection and serialization, and the three assay datasets use a Python
> implementation of the same noise scheme.

**Corrected against the code in this pass.**

- **The pool count was wrong here and right in M1.** The earlier draft said each replicate draws
  10,000 "from 132,480". The code's pool is 129,428. 132,480 is the non-blank line count of an
  untracked SMILES dump on disk and appears nowhere in the pipeline.
- **The three-molecule story does not survive.** Methane's and water's single Morgan substructure
  each occur in one molecule, but **ammonia's occurs in 74**, because QM9 holds multi-component
  entries carrying an NH$_3$ fragment. More importantly the code is not a list of three: it rebuilds
  the vocabulary every replicate and drops whatever comes back all-zero. On three simulated
  replicates the drop count was 0, 0 and 1. The Methods should state the rule, not the anecdote —
  which is what the draft above now does, and it is shorter for it.
- **"The earlier runs trained on 10,000 molecules"** was wrong twice over: 10,000 is the sample, and
  the training block is about 8,000.
- **Only three of the six representations are 32-bit floats.** ECFP4 and Avalon are stored as packed
  bits, SNS as unsigned 16-bit counts. The sentence claiming float32 for all six also contradicted
  the same paragraph's "binary vectors of 2048 bits". Cut, on the literature's advice as well —
  no comparable paper reports its float width.
- **mol2vec is not refused by name**, because it has no code path at all. The names refused outright
  are `smiles`, `randomized_smiles` and `continuous_pdv`. None of this needs to be in the paper;
  it corrects the *reason* the subsection is being rewritten, not the text.
- **Stereo-blindness is not a ChemBERTa-only caveat.** The substructure fingerprints are generated
  with `chirality=False` and QM9's SMILES are canonicalized without stereochemistry, so no
  representation in the study sees a stereocentre. That is now one clause in the standardisation
  paragraph, and it is worth having because QM9's labels are conformer-specific.
- **The Sort & Slice vocabulary is refitted per replicate on QM9 too**, not only per fold on the
  assay side, so slot $k$ does not name the same substructure across replicates.

**Blocked on a fix.**

- **The two ChemBERTa percentages are `TODO`.** `scripts/crosscheck_chemberta.py` gate 4 counts them
  and has never been run — `HANDOFF.md` item 6. The sentence works with the mechanism and no numbers
  if the count does not arrive, but a percentage placeholder is the one thing a reviewer will
  certainly catch.

**Decided here.**

- **The Sort & Slice passage is two sentences, not 180 words**, and it still states the thing you
  decided on 2026-09-07 to state rather than re-run. No comparable paper gives an exclusion more than
  a sentence: *Count your bits* (J Cheminform 2026) writes "We removed 30 entries from the 718,097
  compounds in the dataset that RDKit could not convert to fingerprints" and stops. The matched-pair
  range, the median of 0.062 and the 314 of 351 go to an Additional file. The long form is in this
  file's git history if you want it back.
- **The restriction to three representations is stated in M5**, where the out-of-fold pass is
  described, not here — M2 is about what the six representations are, and six is right for every
  robustness grid.
- **MHG-GNN keeps its corpus and its dimension and loses the polymers-and-photoresistors sentence**,
  which is prior-work framing and belongs in the Introduction if anywhere.

**Mechanical.** `\citep{avalon}` is in neither bib file — Gedeck, Rohde & Bartels, *J. Chem. Inf.
Model.* 46(5):1924–1936, 2006. `\citep{Chithrananda2020}` is present in `refs.bib` (line 1427).


---

## M3. Models: rewrite (replaces lines 207–219; the last paragraph moves to M5)

**Why wholesale:** it names eleven families and the run has nineteen configurations; its statements
about the SVM and GP kernels are contradicted by the code; the VBLL description is of a variant
deliberately excluded from the run; and the closing decomposition paragraph is now its own
subsection.

**Keep the opening paragraph as it stands** (lines 207–208, the Kolmar framing). It is still exactly
right and it sets up the whole paper. Change only its final sentence, and continue:

> All hyperparameters are listed in Additional file~1. Nineteen model configurations were evaluated.
> In the main grids each runs on all six representations, except the Tanimoto-kernel Gaussian
> process, which is defined on binary vectors and runs on ECFP4 alone, giving 109
> model--representation pairs; the reduced runs described in the following section use named subsets
> of those pairs. Validation is used for early stopping and, on QM9, for a temperature fit, and is
> never merged into training.
>
> Random Forests (RFs) are one of the most common choices for QSAR modeling thanks to their
> robustness and interpretability \citep{Svetnik2003}. During bootstrapping, each tree trains on a
> separate subset of data, reducing the influence of any one individual label. This mechanism is
> particularly useful when working with noisy labels \citep{Breiman2001}. Quantile Regression
> Forests (QRFs) extend RF predictions to full distributions by keeping the distribution of training
> labels in each leaf and computing quantiles across all trees for a given prediction
> \citep{Meinshausen2006}; we report the median as the prediction and half the 16--84\% interval as
> its uncertainty, and fit 300 trees rather than 100 because the quantile estimate is this model's
> deliverable. It inherently accounts for heteroscedasticity, such that the quantiles will reflect
> the spread of a target variable's domain. Both forests use a minimum leaf size of five rather than
> one, which is what allows a within-leaf spread to be computed for every prediction (see
> Uncertainty quantification). We also used eXtreme Gradient Boosting (XGBoost) \citep{Mustapha2016,
> Tian2022}, Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm}, and Natural Gradient
> Boosting (NGBoost) \citep{Duan2020}. NGBoost extends deterministic gradient boosting by treating
> the parameters of a chosen parametric distribution as regression targets and learns them via
> boosting with a natural gradient update rule \citep{Duan2020}; we fit a Gaussian distribution by
> maximum likelihood over at most 500 boosting iterations, with the number used chosen by early
> stopping after fifty rounds without improvement and predictions taken at the best iteration. On
> QM9 that early stopping uses the held-out validation split; on the assay datasets, where the
> non-neural models have no validation split, NGBoost holds out a scaffold-grouped fifth of its own
> training rows for the purpose.
>
> Support Vector Machines (SVMs) are a well-established baseline in QSAR modeling, mapping inputs
> into a high-dimensional feature space where a maximum-margin hyperplane separates predictions
> \citep{Vapnik1995, Svetnik2003}. A radial basis function (RBF) kernel with $C = 1.0$ and
> $\gamma = \texttt{scale}$ was used on every representation, so that the SVM carries no
> kernel-by-representation confound.
>
> One ML method that has had particular success in molecular property prediction are Gaussian
> Processes (GPs) \citep{Obrezanova2007, gauche}, a non-parametric class of models that use a kernel
> to produce a Gaussian predictive distribution over every data point \citep{Obrezanova2007}. We
> fitted exact Gaussian processes with a constant mean and a scaled kernel using \texttt{gpytorch};
> the Tanimoto kernel is taken from the \texttt{Gauche} framework \citep{gauche, Ralaivola2005,
> moss2020}. Rather than assigning a kernel per representation, we evaluated the two kernels as
> separate models: an RBF kernel, which runs on all six representations, and the Tanimoto kernel,
> which is defined on binary vectors and therefore runs on ECFP4 alone, so that the kernel
> comparison is like-for-like where both are defined. The RBF lengthscale is initialised at the
> median pairwise distance between training molecules rather than at the library default, because on
> these representations pairwise distances are far from unity and a lengthscale left there gives the
> kernel nothing to learn from, so the process returns its prior and predicts a single value for
> every molecule. A fit whose predictions vary by less than 5\% of the standard deviation of the
> labels it was fitted on is recorded as collapsed: its accuracy is reported and its per-molecule
> model uncertainty is withheld (TODO: how many fits, on which models and representations). Because
> exact inference costs $O(N^3)$ in the number of training points \citep{Rasmussen2005}, Gaussian
> processes on QM9 were fitted on a random subsample of at most 5,000 of the roughly 8,000 training
> molecules, where every other model sees the full training split; on the three assay datasets the
> training blocks are smaller than that cap and no subsampling occurs. We additionally evaluated a
> heteroscedastic variant in which the single observation-noise parameter is replaced by a small
> feed-forward network predicting the noise variance from the molecule's features, trained in the
> same optimisation against the squared residuals of the fitted process.
>
> While standard feed-forward neural networks (NNs) are deterministic, they can be transformed into
> probabilistic models. We tested two architectures: (i) NN-$\alpha$ with two hidden layers of sizes
> [128, 64] and dropout after each hidden layer and (ii) NN-$\beta$ with two hidden layers of size
> 128 and a single dropout layer before the output. Both use ReLU activations and dropout $p=0.2$,
> and were implemented in \texttt{PyTorch} \citep{pytorchGeometric}. Each was trained using the Adam
> optimizer at a learning rate of $10^{-3}$ with a batch size of 32, for up to 100 epochs, with
> early stopping on the mean validation loss after ten epochs without improvement and the best
> weights restored.
>
> We used Bayesian transformations to convert NNs into Bayesian neural networks (BNNs). BNNs
> introduce priors on the weights and compute a posterior distribution given the data, providing
> uncertainty quantification. This can be done by approximations such as Monte Carlo dropout,
> variational inference, or ensembles \citep{gal2016}. Four transformations were applied to each of
> the two architectures. In the full-BNN, every linear layer is replaced by a Bayesian layer with a
> Gaussian prior $\mathcal{N}(0, 0.1^2)$ on all weights, and the network is fitted on the evidence
> lower bound with the Kullback--Leibler term scaled by $1/N$. The Variational Bayesian Last Layers
> (VBLL) transformation \citep{Harrison2024} replaces every linear layer with a mean-field
> variational layer $q(\mathbf{W}) = \mathcal{N}(\boldsymbol{\mu}_W,
> \text{diag}(\boldsymbol{\sigma}_W^2))$ against a standard normal prior, and the output layer
> additionally carries a learned scalar observation-noise variance; it is trained by maximizing the
> evidence lower bound, which can be thought of as minimizing the reconstruction loss plus KL
> divergence $D_{\text{KL}}(q(\mathbf{W}) \| p(\mathbf{W}))$, scaled by $1/N$ \citep{Harrison2024}.
> A heteroscedastic VBLL variant replaces that scalar with a noise variance predicted from the
> input. Finally, a mean-variance estimation (MVE) variant takes the full-BNN and widens its output
> head to a mean and a log-variance, fitted with a Gaussian negative log-likelihood. The
> heteroscedastic VBLL and the MVE variants are the two network families in which both uncertainty
> components vary per molecule. All BNN variants estimated predictive distributions with 100 Monte
> Carlo forward passes at inference.
>
> Hyperparameters are held fixed across representations and across noise levels, so that the noise
> axis is the only thing that moves. Fifteen of the nineteen configurations run at a single shared
> default. The four Bayesian and variational networks run at a tuned setting instead, chosen by a
> random search on clean labels: candidates were ranked by their mean change from the default across
> representations, and the top-ranked one adopted subject to three conditions --- that it was not an
> extreme width or depth, that it cost no more than twice the default's slowest fit, and that it was
> no worse than the default when refitted with training labels noised to half the clean training
> label spread. One setting is adopted per model and used for all six representations, so that no
> part of a model-by-representation difference is tuning; on the assay datasets the choice is made
> separately for each dataset. Searching separately at each noise level was not attempted, as it
> would make ``the tuned setting'' a different setting at every level and confound the noise axis
> with model capacity.

**Corrected against the code in this pass.** These are the ones that change what the paragraph
says, not how it reads.

- **"No model merges its validation split into its training data, so every configuration is fitted
  on the same 80% of the sample."** The first half is true. The second does not follow from it and
  is false three ways: the Gaussian processes fit 5,000 of about 8,000 on QM9; on the assay datasets
  every model fits about 65% of the dataset; and NGBoost there fits four fifths of that again. The
  positive half survives in the draft above, which is also what every comparable paper does — none
  of them describes what its code does not do.
- **"Validation is reserved for early stopping and calibration"** is a QM9 sentence. On the assay
  datasets only the neural models have a validation split at all; the trees, the SVM and the Gaussian
  processes never see those rows, and nothing there fits a temperature.
- **A collapsed Gaussian-process fit is not excluded.** Its $R^2$ and RMSE are computed and written
  with a `gp_collapsed` flag, and the figure script drops nothing. What is withheld is the
  per-molecule model-uncertainty column. The earlier draft said "excluded rather than scored", which
  would have been a claim about which rows exist in your results.
- **The lengthscale probe is at most 500 training molecules**, not all of them — and QM9's
  heteroscedastic Gaussian process does not do the initialisation at all, because it runs its own
  joint optimisation rather than the shared fitting path. The assay-side heteroscedastic process
  does. That asymmetry is not in the draft above; see D15.
- **Only the Tanimoto kernel comes from Gauche.** Both pipelines build a plain `gpytorch` exact
  process, and the RBF kernel is `gpytorch`'s own. "We implemented GP models using the Gauche
  framework" overstates it.
- **Four transformations, not three.** The heteroscedastic VBLL variant is a fourth, and both
  generators submit it.
- **MVE is not the only network family with both components per molecule.** The heteroscedastic VBLL
  variants carry both too, by the support table that gates every written row.
- **Hyperparameters are not held at a single shared default.** Four Bayesian and variational network
  families run tuned whenever the two tuned files are present when a task starts, which is the
  configuration on disk. The adoption rule is the ranking described above, not "beat the default on
  the held-out QM9 test split and survive the same comparison on the three assay datasets" — that is
  a different route, in different code, and it did not produce the files the cluster reads. The
  assay-side confirmation script cannot reach these four models at all.
- **The tuned count.** 24 QM9 pairings carry a tuned setting (four models by six representations),
  though only four distinct settings exist, one per model. On the assay side there are seven
  model-and-dataset entries: Caco-2 has BNN-$\alpha$, VBLL-$\alpha$ and BNN-$\beta$; hERG has
  VBLL-$\alpha$ and BNN-$\beta$; LogD has VBLL-$\alpha$ and VBLL-$\beta$. This is repository state on
  12 September 2026, not a value fixed in code, so re-read it before it goes in.

**Cut, on the literature's advice.**

- **NGBoost's complexity passage** (the $O(Np^3)$ per iteration and the scalability prediction).
  All three passes said cut, two of them "clear". Dutschmann et al. (J Cheminform 2023) describe all
  five of their modelling techniques in 166 words total; Kolmar & Grulke give four algorithms 1,200
  words and no complexity analysis. If NGBoost's cost shaped the study — and it did, since three of
  five out-of-fold folds are scored on QM9 — say it where that restriction is described.
- **The GP sparse-approximation and tree-efficiency sentences.** Same objection, with one exception
  kept: the $O(N^3)$ clause survives, attached directly to the 5,000-molecule cap it motivates,
  because there it explains a choice you made rather than one you did not.

**Blocked on fixes, and this subsection cannot be finished until they land.** All five are in
`HANDOFF.md`.

- **The architecture paragraph above describes architectures that eight of the nineteen
  configurations do not run at.** The shipped tuned files put BNN-$\alpha$ at [64, 32] with **tanh**,
  VBLL-$\alpha$ at [64, 64] with tanh, BNN-$\beta$ at **one** hidden layer of 64 with dropout 0.379
  and learning rate 4.29e-3, and VBLL-$\beta$ at one hidden layer of 64 with dropout 0.357. The
  paragraph is right about the two *base* networks and wrong about what the Bayesian variants
  inherit. Once items 3, 4 and 5 settle what the tuning regime actually is, the fix here is one
  clause — the base architectures in the text, the adopted settings in Additional file 1.
- **Items 4 and 5 are why that is not just a wording problem.** NN-$\alpha$'s learning rate and
  dropout are hard-coded and NN-$\beta$'s are read from the tuned file, so the two are tuned over
  different parameter sets; and the MVE and heteroscedastic-VBLL variants run at the shared default
  beside tuned siblings, which makes a BNN-versus-MVE comparison partly a tuned-versus-untuned one.
- **Item 3**: the setting was ranked over five representations and shipped to six.
- **Item 1**: QM9's heteroscedastic Gaussian process skips the median-distance lengthscale
  initialisation that the paragraph presents as the reason the processes produce numbers at all. The
  assay-side one does it. Once fixed, the draft's sentence becomes true of both.
- **The count of collapsed Gaussian-process fits is a `TODO` in the draft.** Every literature pass
  agreed a rule that changes what is reported needs its count; the number of molecules each fit saw
  is already written onto every results row, so it is a read rather than a run.
- **Additional file 1 contradicts this draft and nothing generates it** — `HANDOFF.md` item 9. The
  draft's first sentence points at it.

**Decided here.**

- **`\citep{gauche}` moves to sit beside the Tanimoto kernel**, which is the only thing that comes
  from that framework; both pipelines build a plain `gpytorch` exact process and the RBF kernel is
  `gpytorch`'s own.


---

## M4. Noise Strategies → Label noise: full rebuild (replaces lines 312–362, including Table 1)

**Why wholesale:** four of the six strategies in Table 1 exist in neither injector; the two that
survive by name have different definitions; $\sigma$ no longer exists as a concept; the level grid
changed; and the sentence that validation is clean is now false.

> Our objective is to evaluate the robustness of QSAR models against label noise. To do so, we
> ``inject'' artificial noise into the labels during training. Test labels are never modified, so
> that degradation is measured against a fixed reference; this differs from \citet{Kolmar2021}, who
> corrupt the evaluation labels as well. Validation labels carry their own noise, drawn
> independently at the same amount, because a model that early-stops against clean labels is
> deciding when to stop using information no practitioner has.
>
> Experimental noise is often modeled as homoscedastic Gaussian noise added evenly across all
> labels. However, not all experimental noise is random. It may be heteroscedastic, where the
> variance depends on the molecule itself or on experimental factors; systematic, where biases from
> factors like assay conditions are introduced; or censored, where a measurement outside an assay's
> range is recorded at its limit. We therefore define a noise \emph{condition} as a pair: a
> \emph{shape}, the distribution a single error is drawn from, and a \emph{targeting}, which decides
> which molecules are affected and by how much. Table~\ref{tab:noise_conditions} lists the seven
> conditions evaluated. Separating the two makes the comparison interpretable: two conditions that
> share a targeting and differ only in shape isolate the effect of the error distribution, and two
> that share a shape and differ in targeting isolate the effect of where the error lands.
>
> The amount of noise is dose-matched across conditions. A condition is defined by a per-molecule
> scale $s_i$ and a shape $\mathcal{D}$ of unit variance, and the error applied to molecule $i$ is
> \begin{equation}
> \varepsilon_i = a\,s_i z_i, \qquad z_i \sim \mathcal{D}(0,1), \qquad
> a = \tau \Big/ \sqrt{\tfrac{1}{n}\textstyle\sum_j s_j^2}, \qquad \tau = \ell\,\mathrm{sd}(y),
> \end{equation}
> so that the expected root mean square perturbation equals the requested amount $\tau$ whatever the
> shape and whatever the scale map. The noise level $\ell$ is a fraction of the standard deviation of
> the clean labels, which makes one level the same relative corruption on QM9 and on each assay
> dataset. Every dose-matched condition therefore delivers the same expected amount of corruption at
> the same level, and a difference in outcome between two conditions is a difference of pattern
> rather than of magnitude. We swept levels $\ell \in \{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$. The
> amount actually delivered is recorded for every run alongside the amount requested.
>
> Censoring sits outside this scheme, because recording a label at an assay limit is not a
> perturbation of a chosen size. Its level is instead the fraction of labels clipped, swept over
> $\{0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50\}$. The limit is a quantile of the clean labels, computed
> once and applied unchanged to the training and validation labels, as a fixed property of an assay
> would be; test labels are left uncensored with the rest.
>
> Which molecules a condition targets is drawn from a seed that does not depend on the noise level,
> so the affected set is the same at every level and the clean-label run is a negative control on the
> same molecules rather than on a different draw.
>
> The grouped conditions interact with the scaffold split in one way worth stating. Group membership
> is the Bemis--Murcko scaffold. On QM9 the affected groups are chosen within the training split and
> whole groups are held out, so a held-out molecule shares no scaffold with any affected training
> group and its recorded noise pattern is flat; questions about which individual molecules were
> corrupted are therefore answered on the training molecules by cross-fitting rather than on the test
> set. On the assay datasets the noise is drawn once over the whole label column, so a held-out
> molecule can belong to an affected group.
>
> The conditions were not all run at the same breadth. Gaussian and the two grouped conditions were
> run across the full grid of models and representations, on QM9 and on the three assay datasets. The
> Student-$t$, Laplace and outlier conditions probe the shape of the error and the size of the
> contaminated fraction rather than the model ranking, and were run on QM9 alone, on a reduced set of
> model--representation pairings on which Gaussian and the two grouped conditions were re-run so that
> the six are compared on the same pairings. Censoring was run on a named subset of five pairings, to
> measure the size of its effect rather than to compare models or representations under it.

**New Table 1.** This replaces `tab:regression_noise` entirely. Per-point scale formulas do not
belong in it any more — the whole point of dose matching is that the delivered magnitude is equal
across rows and is set by the level, not by the row.

| Condition | Shape | Targeting | Simulated real-world source |
|---|---|---|---|
| Gaussian | Gaussian | every molecule, same amount | Random measurement error |
| Student-$t$ ($\nu = 5$) | Student-$t$, $\nu = 5$ | every molecule, same amount | Measurement error with heavier tails |
| Laplace | Laplace | every molecule, same amount | Error distributions fitted to bioactivity data |
| Grouped-wider | Gaussian | whole scaffold groups, added until they cover about 20\% of molecules, receive a $3\times$ wider error | Part of the chemical space measured less reliably |
| Grouped-shifted | Gaussian | every molecule receives one offset shared by its scaffold group plus an independent draw of its own, with $\rho = 0.62$ of the variance in the offset | Between-laboratory bias \citep{Bentz2013} |
| Outlier ($p = 0.10$) | Gaussian | a random 10\% of molecules receive a $3\times$ wider error | Transcription errors, sample mix-ups |
| Censoring | — | labels beyond an upper assay limit are recorded as the limit; the level is the fraction clipped | Assay dynamic range |

Suggested caption: *Noise conditions. Each condition is a shape (the distribution one error is drawn
from) paired with a targeting (which molecules are affected, and by how much). Every condition
except censoring is dose-matched, so at a given noise level each is solved to deliver the same
expected root mean square perturbation and they differ in how that perturbation is distributed
across molecules rather than in its size. $\rho$ is the share of total variance carried by the
group-level offset.*

**Corrected against the code in this pass.**

- **Grouped-shifted is not a group-level offset alone.** Every molecule gets a group offset *plus*
  an independent draw of its own; $\rho = 0.62$ of the variance sits in the offset and 38% does not.
  The earlier draft's table row and its M5 sentence both read as though the whole error were the
  offset, which is what made grouped-shifted look like a flat condition. Table row rewritten.
- **Dose matching equalises the *expected* root mean square, not the realised one.** What a draw
  actually delivers varies, is recorded on the row, and is checked against a band computed from the
  draw's own kurtosis and effective sample size — and that check warns rather than stopping the run.
  "Delivers the same amount" is now "the same expected amount", in the prose and in the caption.
- **The censoring limit reaches validation as well as training, and never reaches test.** The earlier
  draft's "applied unchanged to every split" read literally as clipping the test labels, which
  contradicted the paragraph above it. Both halves now say which splits.
- **The level-zero negative control does not hold on QM9 for five of the seven conditions.** Only
  Gaussian and censoring run level zero there; the other five have the Gaussian clean row copied in
  afterwards, and get **no clean uncertainty row at all**. The accuracy row is bit-identical, so
  AUC_norm is unaffected — but any per-molecule analysis that subtracts a level-zero baseline can
  only do it under Gaussian on QM9. On the assay datasets every condition runs its own level zero.
  **This is a Results constraint as much as a Methods sentence; see D22.**
- **The grouped conditions behave oppositely on the two pipelines.** On QM9 held-out molecules are
  unaffected by construction. On the assay datasets the affected families are chosen over the whole
  column, so held-out molecules can sit in one. The earlier draft stated the QM9 behaviour flatly.
- **The reduced-breadth conditions ran on QM9 only.** The nineteen submitted assay arrays all pass
  `--conditions gaussian grouped_wider grouped_shifted`. Student-$t$, Laplace, outlier and censoring
  are not in any assay script on disk. The earlier draft's "the same noise conditions and the same
  noise levels as QM9" in M1 was wrong about the conditions and right about the levels.
- **The deep run re-runs Gaussian and both grouped conditions on the reduced pairs**, which is what
  makes the six comparable there. The earlier draft presented full-grid and reduced as a partition.
- **The full grid is 109 model--representation pairs**, not 19 × 6 = 114, because the Tanimoto
  process runs on ECFP4 alone.
- **Censoring is five pairs in the robustness runs and full breadth in the uncertainty runs.** The
  selection file says so in its own scope field, and the uncertainty generator does not read it.
- **Grouped-wider's 20% is a closest approach**, not an exact fraction: whole groups are added until
  the running molecule fraction is nearest the target, and the realised fraction is written to every
  row.

**Cut, on the literature's advice.**

- **The classification-flipping paragraph.** Both passes that looked at it said cut, both "clear":
  comparable Methods describe what was run. It also duplicates the M7 sentence, and listing six named
  strategies nobody ran invites a reviewer to ask why not. The M7 sentence carries it.
- **Two sentences of seed justification** collapse into the one above. Kolmar & Grulke's entire
  treatment of replication is "repeated five times at each noise level to give 75 total datasets".

**I got one of these wrong, and it is worth saying plainly.** I reported the missing level-zero runs
on QM9 as a defect. They are not. `slurm_scripts_qm9_rerun/generate_scripts.py:468-503` carries your
decision of 2026-08-28 and its reasoning: of the three conditions with a shape, `grouped_wider` is
keyed to the scaffold group that the out-of-fold pass splits on and `outlier_p10` picks its victims
at random, so both are structural nulls whatever is measured, and censoring is the only condition
whose own clean level buys anything. The accuracy baseline is copied because at level zero the fit is
bit-identical whichever condition labels it. Nothing to fix, and nothing for the Methods to say
beyond what M5 already says about undefined questions.

**Blocked on a fix.** `fig_methods_noise_strategies` needs regenerating for the seven conditions.
`scripts/generate_paper_figures_v2.py:3365-3389` already builds its panels from
`noise_conditions.json` intersected with the installed injector, so this is a re-run — but the v1
script writes a file of the **same name** from the retired six at
`scripts/generate_paper_figures.py:2969`, so check which one produced the copy in the paper.

**Decided here.**

- **The dose-matching solve is an equation and the prose around it is two sentences.** Kolmar &
  Grulke give their far simpler construction as two numbered equations. The equation does the work
  the three coined phrases were doing, and "per-molecule scale map" — which no reader outside the
  code would parse — is gone. Check my notation against how you write the rest of the paper.
- **"The amount actually delivered is recorded" stays.** One pass called it run provenance; the other
  called it the only sentence in M4 that lets a reader tell a dosing failure from a model result,
  and named Heid et al. (JCIM 2023) as the paper whose argument needed exactly that and did not have
  it. One clause.
- **The level anchor difference stays implicit**, as in M1: "a fraction of the standard deviation of
  the clean labels" is true of both pipelines as written.
- **Outlier selection is random, not by $z$-score.** The paper's $|z| > 2$ rule is gone; the premise
  that measurement error tracks the measured value was tested and did not hold. If any sentence about
  outlier noise survives, this is the thing it must say differently.

**Mechanical.** $\rho = 0.62$ is attributed to Bentz et al. 2013 Table 7 in both injectors' comments
and in `noise_conditions.json`; nothing in the code reads that paper, so the attribution is yours to
stand behind. `Bentz2013` is in `citations.bib` (line 2107) and **not** in `refs.bib`.


---

## M5. New: Uncertainty quantification (new subsection, absorbing lines 217–219)

**Why it needs its own subsection:** the current text is one paragraph at the end of Models saying
which models were *not* decomposed. That list is wrong in both directions — the forests and NGBoost
are decomposed, and the variance-head networks carry both halves per molecule — and the machinery
that replaced it needs more room than a paragraph in Models can give it.

> Several of the models produce a predictive distribution rather than a point estimate. We separate
> that distribution's variance into an aleatoric component, which reflects noise in the labels, and
> an epistemic component, which reflects the model's uncertainty about itself. Epistemic uncertainty
> is model-driven and can be reduced by collecting either more or higher quality training data;
> aleatoric is data-driven and reflects the intrinsic noise in the labels.
>
> How the two components are obtained depends on the family. For the stochastic networks we use the
> sampling decomposition of \citet{kendall2017}: over $T$ Monte Carlo forward passes returning a mean
> $\hat\mu_t(x)$ and, where the network predicts one, a variance $\hat\sigma_t^2(x)$,
> \begin{equation}
> \sigma^2_{\mathrm{epi}}(x) = \mathrm{Var}_t\big[\hat\mu_t(x)\big], \qquad
> \sigma^2_{\mathrm{ale}}(x) = \mathbb{E}_t\big[\hat\sigma^2_t(x)\big].
> \end{equation}
> For the two forests we apply the law of total variance across the $B$ fitted trees, using the trees
> already fitted and requiring no retraining: with $\bar y_b(x)$ the mean and $s^2_b(x)$ the variance
> of the labels tree $b$ was grown on that fall in $x$'s leaf,
> \begin{equation}
> \sigma^2_{\mathrm{ale}}(x) = \mathbb{E}_b\big[s^2_b(x)\big], \qquad
> \sigma^2_{\mathrm{epi}}(x) = \mathrm{Var}_b\big[\bar y_b(x)\big].
> \end{equation}
> The aleatoric term here varies per molecule only because the minimum leaf size is five; at a leaf
> of one it is identically zero and the whole predictive variance falls into the epistemic term. For
> the Gaussian process the epistemic term is the latent posterior variance and the aleatoric term is
> the likelihood noise \citep{Rasmussen2005}; in the heteroscedastic variant the latter is predicted
> per molecule.
>
> Not every model supports both components, and the distinction matters when reading any
> per-molecule result. NGBoost is a single fit, so it has an aleatoric term per molecule and no
> epistemic term at all --- absent, rather than zero. The plain Bayesian networks predict a mean and
> nothing else, and so have an epistemic term and no aleatoric one. The Gaussian process with a
> homoscedastic likelihood, and the VBLL networks, each learn a single observation-noise value shared
> by every molecule: a homoscedastic aleatoric uncertainty, which is the correct aleatoric term for
> those models but cannot rank molecules, so its correlation with any per-molecule quantity is
> undefined rather than zero. Every reported uncertainty is accompanied by a declaration of whether
> each component varies per molecule, is one value per fit, or does not exist, and every result is
> read against that declaration (Additional file~[N]).
>
> Questions about individual molecules cannot be asked of the training set using a model that has
> already fitted those labels. For the model--representation pairings where per-molecule uncertainty
> is analysed --- six models on three representations on QM9, and seven on the same three on the
> assay datasets --- training molecules were therefore scored out of fold: the training block was
> divided into five folds using \texttt{GroupKFold} on the same Murcko scaffolds as the outer split,
> and each molecule was scored by a model fitted without it. Noise is injected once, before this
> division, so a molecule carries the same corruption in whichever fold it falls. On QM9, three of
> NGBoost's five folds were scored for reasons of cost, so roughly two fifths of its training
> molecules carry no out-of-fold score; on the assay datasets all five are scored.
>
> Under the conditions that give every molecule the same noise scale --- the three uniform-targeting
> conditions and grouped-shifted --- the question of which individual molecules are unreliable is
> undefined rather than answered negatively, and we report it as such. The correlation between a
> model's stated uncertainty and the amount injected into a molecule's label is computed and reported
> under those conditions regardless, as a check on the out-of-fold procedure rather than as a result:
> a substantial correlation there would indicate that a molecule had been scored by a model that had
> seen its own corrupted label.
>
> Under the grouped conditions the noise a molecule receives is partly a property of its scaffold
> group, and the out-of-fold division holds out whole scaffold groups. A per-molecule result under
> those conditions therefore distinguishes affected groups from unaffected ones; it does not
> distinguish molecules within a group, and should not be read as doing so.
>
> Uncertainties are reported without post-hoc calibration. A single temperature multiplier is fitted
> on the held-out validation split on QM9 and recorded alongside each prediction, but is not applied
> to the reported values and cannot change the ordering of molecules.

**Corrected against the code in this pass.**

- **"Two networks are the study's cleanest decomposition case" is not supportable, and the only
  written ranking says the opposite.** `uncertainty_pairs.json` records the Gaussian process
  separating 11.2×/18.7×/11.9× — "an order of magnitude above everything else" — against network
  medians of 1.6×, 1.6× and 1.7×. The framing has been removed from the draft; do not put it back
  without a measurement.
- **Four conditions spread the noise evenly, not two.** Gaussian, Student-$t$, Laplace and
  grouped-shifted. The generator's own `FLAT_BY_DESIGN` list names all four.
- **The forest aleatoric term is computed over each tree's in-bag rows**, reproduced by index, not
  over all training labels. And leaf size affects only the aleatoric half — the per-tree leaf means
  still differ at a leaf of one.
- **"Matching the outer split" is not true of QM9**, whose outer split is a single 80/10/10 scaffold
  division, not a `GroupKFold`. The draft now says "on the same Murcko scaffolds as the outer split",
  which is true of both. The inner division also falls back to a deterministic random split when
  fewer scaffold groups exist than folds, or when a molecule is missing from the group map.
- **NGBoost's three-of-five folds is a QM9 restriction.** The assay pipeline scores all five.
- **The out-of-fold pass is a named subset, not "the pairings where uncertainty is analysed" in
  general** — six models on ECFP4, PDV and ChemBERTa on QM9, seven on the assay side. Two of the
  three models whose aleatoric term varies per molecule on a kernel or variational route, the
  heteroscedastic Gaussian process and the two heteroscedastic VBLL networks, are **not** on the QM9
  list, so they get no out-of-fold rows there at all even though they are in the deep run. On the
  assay side the heteroscedastic Gaussian process was added on 2026-09-07 and is on the list, so a
  kernel model can be asked the per-molecule question there and cannot on QM9.
- **The temperature is applied**, to the total predicted spread, and written to its own column beside
  the uncalibrated one; the two decomposed components are never scaled by it. Only QM9 fits one. The
  earlier draft implied it was fitted and left unused everywhere.
- **The guard does not behave the same on the two pipelines.** The assay runner stops the task. On
  QM9 a `DecompositionError` falls into a general handler, is logged, the loop continues to the next
  pairing, and the task exits non-zero at the end. Both refuse to write the disagreeing row, which is
  the part the paper would have claimed — but "a check refuses to record a value" is now cut from the
  draft anyway on the literature's advice.
- **`noise_size_constant` does not mark the flat conditions.** It is computed from the uniqueness of
  the realised $|\varepsilon_i|$, so under Gaussian, Laplace and Student-$t$ the individual draws
  differ and the flag is false; it is effectively true only at level zero. The M5a-i provenance note
  named it as though it marked the condition. `rho_raw` is the column a Methods sentence should name;
  the zero-level-subtracted version is degenerate and emitted as NaN.
- **`q4_plain_correlation` is at `scripts/uncertainty_stats.py:1034`**, not 1040, which is inside its
  docstring.

**Folded in, and what happened to M5a.** The three M5a additions are now sentences inside M5 rather
than a numbered appendix to it: the null-correlation check and the grouped-conditions caveat are in
the draft above, both kept at the literature's recommendation — Hirschfeld et al. (2020) and
Dutschmann et al. (2023) both carry a Methods sentence telling a reader how to read a correlation
that will look disappointing. M5a-iii, the model-by-condition coverage, is the one both passes said
to compress; see D30.

**Cut, on the literature's advice.**

- **"The separation is computed by a single shared routine used by both pipelines, in variance space
  throughout, with one conversion to a standard deviation at the point of reporting."** No comparable
  Methods describes its own code path. Once the equations are in variance terms, that the code is in
  variance terms is not a claim about the study.
- **"and a check refuses to record a value that disagrees with that declaration."** Two passes called
  it "clear" to cut: none of the ten audited papers describes its own test suite, and in a Methods it
  reads as defensiveness rather than method. The declaration itself stays — it is an artefact a
  reader can see.
- **The circularity argument for not reporting calibrated coverage.** It is an argument, and under
  this journal's combined format arguments go in Results and discussion. It is also only true if the
  multiplier is fitted on the molecules the coverage is measured on, and yours is not — it is fitted
  on validation, so a reviewer who spots that will ask for the calibrated numbers anyway.

**Blocked on a fix.** The heteroscedastic Gaussian process and both heteroscedastic VBLL networks
are in the deep run and absent from `uncertainty_pairs.json`, so three of the five configurations
whose aleatoric term varies per molecule get no out-of-fold rows on QM9 — while the assay side added
GP-Hetero on 2026-09-07 and can ask them. That is `HANDOFF.md` item 2, and it is the one thing in
this subsection that changes what the paper can claim rather than how it is phrased. The draft above
says "six models on three representations on QM9, and seven on the same three on the assay datasets",
which is today's state; it becomes one number if the two lists are reconciled.

**Decided here.**

- **The two decomposition equations go in.** This was the only place in the whole Methods where all
  three literature passes said the draft was *under*-specified. Yang & Li (J Cheminform 2023) give it
  as their equation 5, Busk et al. (2022) with the two terms under-braced, Scalia et al. (2020) as
  their equation 4, Gruich et al. (2023) as two short equations — always the algebra *beside* the
  Kendall & Gal citation, never instead of it. The forest version has no published precedent in the
  comparison set, which is exactly why it is the one that most needs to be checkable.
- **The model-by-condition coverage is one sentence, not a numbered sub-subsection.** Add it beside
  the reduced-breadth sentence in M4: under the three shape conditions on QM9 the decomposition rests
  on two of the four models, named in an Additional file.
- **How many models emit no uncertainty is worth a clause.** XGBoost, LightGBM, the SVM and the two
  deterministic networks are declared `none/none`, so "several of the models produce a predictive
  distribution" is covering nearly half the roster. A reader counting nineteen configurations against
  an uncertainty table will want the number.
- **The full support table goes to an Additional file.** Nineteen QM9 declarations plus the assay
  roster's; both are verified and ready.

**Passed to the metrics pass, not decided here.** Busk et al. (2022) and Scalia et al. (2020) both
turn the per-molecule-versus-constant distinction into a reported number rather than a declaration —
the coefficient of variation of the predicted spread across test molecules, where zero means constant
and therefore uninformative. One column per model and representation, no re-run.


---

## M6. Performance metrics: hold (lines 220–311)

**Owned elsewhere.** Do not rewrite this subsection from this guide. Only the run-time half is fixed
by the pipeline, and it is small:

> Root mean squared error (RMSE), mean absolute error (MAE), the coefficient of determination
> ($R^2$) and Pearson's $r$ are computed on the held-out test split for every fitted model. RMSE and
> MAE are converted back into the label's own units before reporting.

Three fixes here are certain regardless of how the rest is settled, because they are about things
the code no longer computes at all:

- **ECE is gone.** Expected Calibration Error is computed nowhere in the analysis path. Its
  definition (lines 234–238), its row in the metrics table (line 300) and its columns in the results
  tables (lines 503, 508) cannot be regenerated. Since the current paper defines, tabulates and
  reports it, removing it silently will read as a result that went missing — Scalia et al. (2020)
  drop the same metric and say why in a one-sentence footnote, which is the cheapest form.
- **Mean prediction-interval width is not computed** by anything in the study, only by the
  standalone package. Lines 368 and 620 both list it.
- **"Eleven noise levels"** (lines 240, 245) and $\sigma \in \{0, 0.1, \ldots, 1.0\}$ are wrong
  wherever they appear, including inside the metric definitions.

**Two things to pass to whoever owns this subsection**, both from the literature pass rather than
from the code:

- **Coverage at 1$\sigma$ and 2$\sigma$ is thinner than any of the seven uncertainty benchmarks
  read.** Hirschfeld et al. (2020) use miscalibration area; Scalia et al. (2020) use the area under
  the calibration error curve plus the maximum; Yang & Li (2023), in this journal, use two calibration
  errors; Busk et al. (2022) use negative log-likelihood plus calibration curves. Miscalibration area
  is computed from the same curve the two coverage points already sit on, so it costs a line in the
  figure script and no re-run, and it is the number a reviewer in this area looks for first.
- **The coefficient of variation of the predicted spread** — see D32.

---

## M7. NoiseInject framework: three sentence fixes (lines 364–373)

Not a rebuild. Three claims to correct, all of the same kind — the package does more than the study
uses, and the text currently reads as though everything listed was benchmarked here.

- Line 368 lists ECE and mean prediction-interval width among the metrics computed. Both are package
  features; neither appears in this study's results. Either say so, or drop them from the sentence.
- Line 352 says the classification strategies "mirror the regression set". They still exist, but the
  six regression strategies they mirrored no longer do, so the sentence needs rewriting rather than
  deleting. Suggested: *"NoiseInject also implements six label-flipping strategies for
  classification tasks, which are not benchmarked here."* This sentence now carries the classification
  material cut from M4.
- Line 371's reference wrappers include split-conformal prediction. Conformal prediction is refused
  by name in this study's pipeline and appears in no result. Keep it as a stated package feature, or
  drop it; do not leave it ambiguous.

**One more, found in this pass.** The assay runner still takes `--sigmas`, still calls its grid
`sigma_levels`, and still writes a column literally named `sigma` into every assay results row, with
a `level_units` column beside it saying whether that number is a fraction of the label spread or a
fraction of labels clipped. QM9 refuses the flag by name. Nothing in the paper needs to say this, but
anyone reading an assay results file will meet `sigma` and should not conclude the old parameter
survived.

---

## Methods: sentences the code contradicts

Every line number confirmed by searching for the quoted text in the current `paper.tex`.

| Line | What it says | What the code does | Fix |
|---|---|---|---|
| **193** | split "implemented by DeepChem" | A purpose-written scaffold splitter that gives each distinct acyclic molecule its own group; DeepChem's was replaced because it put every acyclic molecule in training | M1. Drop the `\citep{deepchem}` here |
| **193** | replicates split "from this fixed subset" | Each replicate draws its own 10,000 molecules | M1 |
| **193** | "Experiments that involved tracking uncertainty values were only run once" | On QM9 the out-of-fold pass runs inside the grid tasks and is replicated ten times; it is the assay side that has one fit per cell | Delete. See Part Two |
| **197** | "Tanimoto kernel was used for SVM" for binary representations | SVM is RBF on every representation, both pipelines. Lines 212 and 214 already say so | Delete the clause; the paper contradicts itself. Also retires Additional file 12 |
| **197** | hERG "N = 1,482" | 1,415 | M1. Also line 556 |
| **197** | "all six noise injection strategies and eleven levels" | Seven conditions, seven levels on a different axis — and the assay datasets ran three of the seven | M4 |
| **199** | targets "mean-centered and normalized" | True, but silent on the two things that were fixed: the statistics are the clean *training* ones, and noise is added before standardisation | M1 |
| **203** | one-hot SMILES and mol2vec are study representations | One-hot and randomized SMILES are refused by name; mol2vec has no code path at all | M2 |
| **203** | Sort & Slice gives "a binary vector" | It carries substructure counts | M2 |
| **205** | "Rust was used to perform … feature extraction" | Rust computes no representation. It does label processing, noise injection and serialization, on QM9 only | M2 |
| **214** | Tanimoto for fingerprints, RBF for PDV | An RBF process on all six representations and a Tanimoto process on ECFP4 alone, as two separately reported models | M3. Also line 418's "incompatible with PDV" |
| **216** | VBLL is a posterior "over the last-layer weights" | Every layer is variational; the last-layer-only variant is excluded from the run | M3 |
| **216** | three Bayesian transformations | Four: full-BNN, VBLL, heteroscedastic VBLL, MVE | M3 |
| **218** | forests, NGBoost and BNN variants were not decomposed | The forests and NGBoost are. The plain Bayesian networks really do carry an epistemic term alone, which is the one part of this sentence that survives | M5 |
| **222, 240** | $\sigma$ is "the noise scaling factor shared by all strategies", swept 0 to 1.0 in elevens | No such concept; the level is the delivered amount, seven values to 1.5 | M4, M6 |
| **234–238, 300** | ECE defined and reported | Computed nowhere | M6 |
| **313** | "Validation and test data remain free of noise" | Test yes. Validation no on QM9; on the assay datasets only the neural models have a validation split, and it is noised | M4 |
| **318–350** | Table 1, six strategies with per-point scales in $\sigma$ | Four of the six exist nowhere; the surviving two are defined differently | M4, new table |
| **354** | outlier noise hits samples with $z > 2$; threshold acts on $|y| > 1$ "on normalized data" | Outlier selection is random; noise is applied to the raw label before standardisation, so no rule can be phrased in standardised units | M4 |
| **368, 620** | interval width and ECE among computed metrics | Package features only | M7 |

**Smaller ones**, unchanged from the previous guide and still open: line 274 "observations … are
independence" → "are independent"; lines 260 and 432 disagree on `R² ≤ 0.6` versus `R² < 0.6`, and
both are superseded by whatever the metrics pass settles.

---

## What is still stopping Part One being finished

Two kinds of thing, and only one of them is writing.

**Nine are code or submission defects, and they are in `HANDOFF.md` as of 12 September 2026.** None
is a choice; each is something the code or the queue is doing wrong, verified at a line number.
Ordered there by what a wrong result costs:

| | What | Where it lands in the Methods |
|---|---|---|
| 1 | QM9's heteroscedastic Gaussian process never initialises its RBF lengthscale from the data; KIRBy does it at both its sites | M3's lengthscale sentence is true of one pipeline until this is fixed |
| 2 | `uncertainty_pairs.json` names six models, the assay uncertainty generator names seven | M5's out-of-fold coverage sentence |
| 3 | The tuned setting was ranked over five representations and shipped to six | M3's tuning paragraph |
| 4 | NN-$\alpha$'s learning rate and dropout are hard-coded; NN-$\beta$'s are tuned | M3's architecture and tuning paragraphs |
| 5 | The MVE and heteroscedastic-VBLL variants train at defaults beside tuned siblings | M3's tuning paragraph |
| 6 | ChemBERTa's collision count has never been produced | M2's two `TODO` percentages |
| 7 | `data/valid_qm9_indices.pth` has no generator | M1's exclusion clause |
| 8 | Nothing proves which filters produced the cached hERG file — **verify, do not re-fetch** | M1's hERG sentence |
| 9 | Additional file 1 is hand-typed and contradicts `model_defaults.py` | M3's first sentence points at it |

**Four numbers are waiting on a run or a read**, and are marked `TODO` in the drafts rather than
guessed: the two ChemBERTa percentages, the count of collapsed Gaussian-process fits, and the tuned
pairing count if the regime changes.

**Three are mechanical**: `avalon` is in neither bib file, `Bentz2013` is in `citations.bib` only,
and `paper.tex` points `\bibliography` at a file that does not exist.

Everything else in Part One is written. Where I cut correct prose or chose between two defensible
forms, the subsection says so and names the comparison paper behind it, so you can reverse any of
them in one edit: the HOMO--LUMO paragraph, the Sort & Slice passage, MHG-GNN's track record, the
NGBoost and Gaussian-process complexity passages, the classification-flipping paragraph, the
internal-guard sentence, and the circularity argument about calibrated coverage.

---

## One thing to confirm, because it changes what runs

All nineteen assay scripts pass `--conditions gaussian grouped_wider grouped_shifted`. The
Student-$t$, Laplace and outlier conditions are opt-in on that generator and were not asked for.
`noise_conditions.json` gives them no `applies_to` scope, so nothing declares they should run there.
The assay datasets therefore carry three of the seven conditions and QM9 carries all seven, which is
what M4 now says. **If that is not what you intended it is a submission, not a rewrite.**

---

## What the second pass changed, and how it was checked

Eleven agents re-read the code in September 2026 against the previous draft of this Part One: seven
reading one subsection each against the pipelines that implement it, three reading comparable
published Methods sections for how much detail is normal, and the whole set instructed to default to
"corrected" or "unverifiable" rather than to confirm. **302 claims were checked. 74 were corrected
and 31 could not be established from code at all.** Nothing below came from `RERUN_PLAN.md`, from a
code comment, or from memory.

The single largest finding is **M0**: the previous draft repeatedly stated as a property of the study
something that is a property of QM9 alone, or of the assay pipeline alone. Twenty-four of the
seventy-four corrections are of that shape: the fix is the same in every case, to say which pipeline
the sentence is about.

The literature calibration read ten papers in full, four of them in this journal: Kolmar & Grulke
(J Cheminform 2021), the closest published analogue; Yang & Li (J Cheminform 2023) and Dutschmann et
al. (J Cheminform 2023) for uncertainty; Jiang et al. (J Cheminform 2021), *Count your bits*
(J Cheminform 2026) and Dablander et al. (J Cheminform 2023) for multi-representation benchmarks;
Heid et al. (JCIM 2023), Scalia et al. (JCIM 2020), Hirschfeld et al. (JCIM 2020) and Busk et al.
(2022) for the uncertainty half; Landrum & Riniker (JCIM 2024) for ChEMBL provenance. Their Methods
run 1,740 to 6,000 words. **Four things were cut on their evidence** — the NGBoost and Gaussian
process complexity passages, the classification-flipping paragraph, the internal-guard sentence and
the code-path sentence — **and three added**: the two decomposition equations, the dose-matching
equation, and the clause distinguishing this study's clean test labels from Kolmar & Grulke's noised
ones.

---

# PART TWO — LIMITATIONS

New paragraph, to sit before the closing paragraph of the Conclusion. Written now because every item
in it is a property of the design rather than of the results.

> This study has several limitations. The QM9 labels are computed rather than measured, so the noise
> we inject is imposed on a target that is otherwise clean; on the three assay datasets the injected
> noise sits on top of real measurement error that we cannot separate from it, and the noise level
> there should be read as an amount added rather than an amount present. Each configuration on the
> assay datasets is fitted once rather than repeated under new seeds, and the noise there is drawn
> once over the whole label column rather than per fold, so those results carry variation across the
> five cross-validation folds --- which mixes sampling with scaffold difficulty --- but no
> run-to-run error term of any kind, and no comparison between two models on those datasets should
> be read as significant on the strength of the fold spread alone. Two questions are undefined by
> construction rather than answered negatively: under the conditions that give every molecule the
> same noise scale, and for the models whose observation-noise term is a single learned value,
> asking which individual molecules a model finds unreliable has no answer to give. On QM9 the
> Gaussian processes are fitted on a subsample of at most 5,000 of the roughly 8,000 training
> molecules where every other model sees the full training split, so their comparison with the other
> families is not held at equal training size there. The ChemBERTa results are results for one
> checkpoint whose tokenizer operates character by character, and should not be read as a statement
> about pretrained transformer embeddings in general; no representation in the study encodes
> stereochemistry. Finally, hyperparameters are held fixed across the noise axis, which keeps that
> axis clean but means we do not measure whether a model could be retuned to resist noise better
> than it does at its default.

## L2. The cells that were run twice (append to the Limitations paragraph)

Some tasks ran a second time and appended to the same results file, so a few cells hold more than one
row. Nothing can be re-run, so the copies are resolved by a stated rule rather than by file order,
and the rule is `figlib_load._resolve_duplicates`: where the standardisation differs between copies
the two runs were fitted on different training draws and are not repeat measurements of one thing, so
the copy matching the majority standardisation for that model, representation and condition is kept;
where the standardisation matches, the copies differ only by nondeterminism in the fit and the median
across them is taken.

Two sentences to append:

> A small number of configurations were fitted more than once and both fits were retained in the
> results files. Where two fits of one configuration were standardised differently they were trained
> on different draws of the training split and the fit matching the majority split for that model,
> representation and noise condition is used; where they were standardised identically they differ
> only by nondeterminism in fitting and the median across them is used, with every disagreeing cell
> listed in Additional file [N].

**The counts, and where they come from.** `results/decisions/d0_duplicate_disagreements_qm9.csv`,
written by the run of 9 September 2026. **Refresh all six numbers from the final run's copy of that
file before the paragraph goes into the paper** — every task that lands changes them.

One cell is one model, one representation, one noise condition, one noise level, one replicate.

| | 9 Sep 2026 run |
|---|---|
| QM9 cells whose copies disagree | 621 of about 26,100 (2.4%) |
| rows involved | 1,622 |
| cells where the copies were trained on different draws | 168 |
| cells where the copies differ only by nondeterminism in the fit | 453 |
| median range in R2 within a disagreeing cell | 0.011 |
| cells whose range in R2 exceeds 0.05 | 43 |

The 43 wide ones sit at the top of the noise ladder (39 of the 43 at levels 0.75 and above), where R2 is
small and a swing of 0.2 is a large share of a small number. AUC_norm integrates the whole ladder, so
a cell like that moves the robustness score by much less than it moves the single R2 it came from --
**unmeasured**, and worth one sentence of measurement before the Limitations text claims it.

**Four corrections the second pass made to this paragraph.** The noise draw on the assay side is
fixed across folds as well as the fit, which makes the no-error-term claim stronger than the first
draft said. Four conditions give every molecule the same noise scale, not two — Gaussian, Student-*t*,
Laplace and grouped-shifted — and "scale" is the right word, because under grouped-shifted every
molecule still gets an independent draw of its own on top of its group's offset. The 5,000-molecule
Gaussian-process cap binds on QM9 only; on the three assay datasets the training blocks are smaller
than the cap. And the stereochemistry clause is new: the substructure fingerprints are generated
without chirality and QM9's SMILES are canonicalised without it, so stereo-blindness is a property of
the whole study rather than a ChemBERTa caveat.

**One thing this paragraph deliberately does not say.** It does not call the uncertainty results
single-seed. On QM9 the out-of-fold uncertainty pass runs inside the grid tasks and is therefore
replicated ten times like everything else; it is the assay side that has one fit per cell. The
submitted paper's line 193 ("Experiments that involved tracking uncertainty values were only run
once") is wrong for QM9 and should be deleted rather than moved here.

---

# PART THREE — WAITING ON THE RE-RUN

Headings only. Nothing here can be written until there are results, and drafting it now would mean
inventing findings and then rewriting them.

- **Abstract** — waiting on every Results unit.
- **Scientific contribution** — to be redone from scratch once the uncertainty result exists.
- **Introduction close** — research questions and preview; waiting on the spine.
- **Results** — outlined below. The prose waits on the re-run; the structure does not.
- **Two-mechanism synthesis** — new passage; waiting on both Results halves.
- **Conclusion** — waiting on everything above it.
- **Additional files** — waiting on the re-run.

---

## The Results section — proposed structure

**Outline only.** The figure and table set is specified in `RERUN_PLAN.md` §14, which owns the
options, the visual descriptions and the open choices; nothing here restates them. Six figures and
seven tables, against the eight and six `paper.tex` carries today.

**Two things fix the order.** QM9 leads, because it is the clean data (§0.3). And the three research
aims stated at the close of the Introduction — representation against architecture; probabilistic
models and their uncertainty; generalisation across noise mechanisms and properties — are the units
the Results has to deliver, in that order of argument if not of section.

⚠️ **Section order is the author's call and §4 decision 5 is still open** — whether the assay
datasets move to the front rather than sitting as a validation unit at the end. The order below
assumes they do not.

**One Methods figure sits outside this section**: `fig_methods_noise_strategies`, rebuilt for the
settled conditions with a dose-matching panel added (`RERUN_PLAN.md` §14.5 F1). It is handled in M4.

### R1. What label noise costs you

The overview unit. Establishes the shape of the problem before anything is decomposed.

- **Figure — F4.** Two panels: R² against noise level as curves, and a model-by-noise-type grid of
  AUC_norm with the clean R² beside it as a separated left-hand column. QM9, one representation.
- **Table — T4.** AUC_norm by model and noise type, with a clean-R² column and **no mean column**.

*Carries the standing rule that AUC_norm is never printed without its baseline. Also absorbs "does
the kind of noise matter", which was a separate figure in the last plan and is now F4's second
curve panel.*

**TBD:** whether F4's top half is one panel or two — built as two and reviewed on the rendered
figure (`RERUN_PLAN.md` §14.5 F4).

### R2. Model, representation, or their pairing

The paper's first research aim, and the unit the ANOVA exists for.

- **Figure — F2.** Variance decomposition by noise type, for predictive performance and for
  AUC_norm, **with the spread across the ten replicates shown**. No version of this figure has ever
  carried one.
- **Table — T3.** The same η² values, plus the replicate spread. QM9 in the main text; the assay
  datasets as an additional file, each stating it has no residual term because it has no repeats.
- **Figure — F3.** Model against representation, as grids, so the interaction term is shown as
  actual pairings rather than only as a share of variance.

**TBD:** how many noise types F3 shows, and which. Results-dependent; the selection rule is fixed
(`RERUN_PLAN.md` §14.5 F3 and §14.6 row 14).

⚠️ **The existing text's claims in this unit are all up for reversal**, including "representation
explains less than 10% of variance" and "PDV stood out as having particularly strong robustness".
Both predate every 2026-08 fix.

### R3. Do probabilistic models resist noise better?

The robustness half of the second research aim. Was a figure; **is now a table** (author,
2026-09-04) because it is five paired comparisons and a figure of five numbers is not worth a float.

- **Table — T5.** One row per pair (NN-α against BNN-α, RF against QRF, and so on), **one column per
  representation**, each cell the change in AUC_norm with a significance mark. No averaging — the
  current table's single number per row is a mean across representations.

### R4. Does it hold on assay data?

The third research aim, and the only data in the study that has never been contaminated.

- **Figure — F8.** Three panels, one per assay dataset: models against noise types, AUC_norm, one
  shared colour scale, clean-R² column at the left. One representation, named in the title.
- **Table — T7 (new).** Rank transfer: each model's AUC_norm rank on QM9 beside its rank on logD,
  Caco-2 and hERG. **One table per representation.** Replaces the cross-dataset figure, which
  averaged over representation and noise type together.

⚠️ **No error bars here, and the Methods must say so.** One fit per cell, seed pinned; the five folds
are a partition, not repeats (§3.2b).

### R5. Does noisy training make a model less sure?

- **Figure — F6.** The aleatoric/epistemic decomposition, one small chart per model, two lines each.
  **Settled as a headline figure** (author, 2026-09-04).
- **Table — T6, upper half.** Mean predicted uncertainty against noise level as a slope, coverage at
  1σ and 2σ, and the support flags saying whether each component varies per molecule or is one
  number per fit.

*This unit is the population-level statement and must be labelled as one — the paper has repeatedly
fused it with the per-molecule question that follows.*

### R6. Can uncertainty tell you which labels are bad?

The sharp question, and the one the submitted paper got wrong by pooling noise levels.

- **Figure — F7.** Three options; which one runs depends on what the uncertainty runs say
  (`RERUN_PLAN.md` §14.5 F7 and §14.6 rows 1–3).
- **Table — T6, lower half.** The Q4 statistic with its permutation band, and the error-ranking
  correlation against the clean label, within level.

**TBD, and it is the largest one in the Results.** A clean null is a result here and gets a figure
rather than a sentence.

### R7. Two-mechanism synthesis

No figure. The passage that separates *resisting* noise from *noticing* it, which the submitted paper
runs together. Waits on R1–R6.

---

## The contingent figures

`RERUN_PLAN.md` §14.6 is a live list of figures that exist only if the results say so — fourteen
rows, each naming the trigger, the statistic that fires it, and where the figure goes. **Read it
before drafting any Results unit**, because several of its rows change which unit a finding belongs
in. The ones most likely to fire: censoring actually flagging clipped labels; one representation
turning out to be an outlier; and the noise types failing to separate the models, which is a headline
finding either way it comes out.

---

# Suggested order

Everything below can be done now.

1. **Label noise (M4).** The largest change, it removes a table, and four other subsections refer to
   it. Nothing else can be finished while the paper still describes six strategies and eleven levels.
2. **Datasets (M1).** It carries the corrected counts that also propagate into the Results and the
   figure captions, so doing it early stops those being written twice.
3. **Uncertainty quantification (M5) and Models (M3) together.** They were one subsection and the
   split has to be made in one pass, or material will fall between them.
4. **Representations (M2)**, once the ChemBERTa numbers exist.
5. **NoiseInject framework (M7), the abbreviations fix, and the Limitations paragraph.** All three
   are short.
6. **Performance metrics (M6)** whenever the metrics pass hands you its definitions.
