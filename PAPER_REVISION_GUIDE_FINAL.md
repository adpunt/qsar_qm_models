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
> **Part One is sourced from the code.** Every factual claim in it was read out of the code in
> September 2026 and then independently re-checked against the same lines by a second pass —
> nothing comes from `RERUN_PLAN.md`, from a code comment, or from memory. Where a number could not
> be established from code it is marked `TODO` rather than guessed.
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
| Methods, M1–M7 | **Drafted** |
| Limitations | **Drafted** |
| Results | **Outlined** — structure and figure set fixed, prose waiting on the re-run |
| Everything else | Heading only — waiting on the re-run |

---

## What broke, and why the Methods cannot be patched

The Methods in `paper.tex` describes a study that no longer exists in the code. Not in emphasis — in
substance. Eight things changed, and each one forces text:

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
   the paper. The last two are the study's flagship decomposition case, which line 218 says was not
   attempted.
6. **The uncertainty decomposition is real machinery now.** One shared definition, five routines, a
   declaration on every row of whether each half varies per molecule, and a check that refuses a row
   disagreeing with it. Line 218's list of what was not decomposed is now wrong in both directions.
7. **hERG is 1,415 compounds, not 1,482** (lines 197 and 556).
8. **Several load-bearing implementation choices are new and undescribed**: the split is no longer
   DeepChem's; target standardisation uses the clean training statistics; the Gaussian process
   lengthscale is initialised from the data and a collapsed fit is recorded rather than scored;
   Gaussian processes are capped at 5,000 training molecules; the full BNNs now carry a KL term that
   they did not have when the paper's numbers were produced.

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

The current Methods is 3,226 words of prose. The drafts below come to roughly 4,000. The growth is
in Label noise and the new uncertainty subsection; everything else is close to its current length or
shorter.

**Naming.** The three measured datasets are called **assay datasets** throughout. The code calls
them "validation" everywhere, which collides with the early-stopping validation split inside each
fold; nothing is being renamed in the code, but the paper should not use that word for them.

---

## M1. Datasets: full rewrite (replaces lines 192–200)

**Why wholesale:** five of its factual claims are now wrong (the subset is not fixed across
replicates, the splitter is not DeepChem's, six strategies and eleven levels are gone, hERG's count
is wrong, and the standardisation sentence is silent on the two things that were fixed), and the
assay datasets need the fold structure stated because their error bars mean something different from
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
> We selected HOMO--LUMO energy gap as the primary prediction target in QM9, as it captures
> electronic characteristics relevant to molecular reactivity and stability \citep{Islam2019,
> Hllermeier2021}. The HOMO--LUMO gap is a well-established target for quantum chemistry benchmarks,
> and is defined as the difference between the highest occupied and lowest unoccupied molecular
> orbital energies, directly relating to a molecule's electronic excitability, charge transfer
> capability, and chemical stability \citep{Fediai2023}. Labels are used in electronvolts; across a
> typical replicate the gap has a mean of 6.86\,eV and a standard deviation of 1.27\,eV, which is
> the scale against which injected noise is dosed. We also assumed that thanks to the consistent
> calculation method, and notwithstanding approximations in the level of theory used, the QM data
> were ``free'' of noise.
>
> For each replicate, a scaffold-based train/validation/test split (80/10/10) was generated on
> Bemis--Murcko scaffolds computed with \texttt{RDKit} \citep{rdkit}. Acyclic molecules have no
> Murcko scaffold, and each is treated as its own group rather than being pooled into a single
> empty-scaffold group; groups are then assigned to the three parts in random order until each
> reaches its quota. The distinction matters on QM9, where acyclic molecules are close to half the
> sample: pooling them places nearly all of them in the training set and removes them from
> evaluation entirely. Scaffold splits were used to minimize structural overlap between sets. While
> predictive performance typically drops when switching from random splitting to a more challenging
> scenario like scaffold splitting, the use of scaffold splitting discourages the QSAR model from
> overfitting \citep{Heid2023}.
>
> Prior to modeling, all molecular structures were sanitized and canonicalized using \texttt{RDKit}
> to remove invalid valence states and standardize atom and bond typing \citep{rdkit}. Target values
> were standardized to zero mean and unit variance using the mean and standard deviation of the
> \emph{clean training} labels alone, so that the target scale does not move with the amount of
> noise injected; noise is added to the raw label before this step. Errors are converted back into
> the label's own units before being reported. Data loading and pre-processing were performed from a
> fixed base random seed, from which each replicate derives its own.
>
> To assess whether noise robustness results generalize beyond the HOMO--LUMO gap, we evaluated the
> same models and representations on three datasets with experimentally measured endpoints. From the
> OpenADMET initiative \citep{openadmet} we used LogD (lipophilicity, $N = 5{,}039$) and Caco-2
> efflux permeability ($N = 2{,}161$), the latter modelled as $\log_{10}$ of the efflux ratio. We
> also selected hERG $K_i$ data from ChEMBL \citep{Zdrazil2023}, following a protocol inspired by
> \citet{landrum2024}: filter for binding assays and deduplicate by median pChEMBL value. We then
> removed compounds with inter-assay standard deviation $> 1.0$ log unit \citep{Zdrazil2023}. This
> results in $N = 1{,}415$ compounds, extracted from ChEMBL release 36 and re-checked against
> release 37 without change. For all three datasets, structures were standardized by retaining the
> largest covalently bonded fragment and canonicalizing with \texttt{RDKit}, and duplicate
> structures were collapsed to their median label. The clean label standard deviations are 1.19,
> 0.44 and 0.91 log units respectively.
>
> The three assay datasets were evaluated under the same noise conditions and the same noise levels
> as QM9. Five-fold scaffold cross-validation was conducted using \texttt{GroupKFold} on Murcko
> scaffolds; within each fold a further 20\% of the training block was carved off, again by scaffold
> group, to provide an early-stopping set, so approximately two thirds of each dataset is fitted in
> any one fold. Unlike QM9, each configuration here is fitted once rather than repeated under new
> seeds. The assay results therefore carry variation across the five folds, which mixes sampling
> with scaffold difficulty, but no run-to-run error term.

**What that last sentence buys you, and why it is not optional.** With one fit per cell there is no
estimable residual term in an assay-side variance decomposition, and no assay model-versus-model
comparison carries a run-to-run error bar. A reader who assumes the fold spread is a replicate
spread will over-read every assay comparison in the paper. State it once here and the Results can
lean on it.

---

## M2. Molecular Representations: full rewrite (replaces lines 201–206)

**Why wholesale:** it describes six representations, two of which (one-hot SMILES, mol2vec) are
refused by name in the code, and omits two that are in the run (Avalon, ChemBERTa). It also calls
Sort & Slice binary, which it no longer is, and attributes feature extraction to Rust, which does
none.

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
> molecules only; on the assay datasets it is refitted within each cross-validation fold, so that no
> held-out structure enters the feature basis.
>
> We also used 200-dimensional physicochemical descriptor vectors (PDVs) computed from RDKit
> molecular descriptors (\texttt{MolecularDescriptorCalculator}), encompassing molecular weight,
> LogP, topological polar surface area, topological connectivity indices, partial charges, VSA bins,
> EState indices, hydrogen bond donor/acceptor counts, and functional group counts, among others
> \citep{Cherkasov2014}.
>
> Two pretrained learned representations were included. ChemBERTa \citep{Chithrananda2020}
> embeddings were taken from the \texttt{ChemBERTa-77M-MTR} checkpoint as the mean over non-padding
> token embeddings, giving a 384-dimensional vector. The tokenizer distributed with this checkpoint
> falls back to single characters, so the encoding does not distinguish two-letter halogens, formal
> charges, stereocentres or azole tautomers, and a small proportion of molecules consequently
> receive identical embeddings (TODO\% on QM9, TODO\% on hERG). Results for ChemBERTa here should be
> read as results for this checkpoint rather than for transformer embeddings in general. As a
> representative GNN-based learned representation, we included graph embeddings generated by
> Molecular Hypergraph Grammar GNNs (MHG-GNN) \citep{kishimoto2023}. MHG-GNNs are a GIN-based
> autoencoder pre-trained on 1.34 million PubChem molecules using $\beta$-VAE loss, producing
> 1024-dimensional embeddings through iterative message passing. MHG-GNN was originally developed
> for material science, and has demonstrated strong performance on property prediction tasks for
> polymers, photoresistors, and chromophores \citep{kishimoto2023}.
>
> All representations are stored as 32-bit floating point values with no per-molecule rescaling.
> Whether a representation is standardized is decided from the feature matrix rather than from its
> name: the continuous representations (PDV, ChemBERTa and MHG-GNN) are z-score normalized per
> feature, with the mean and standard deviation computed on the training set and constant-variance
> features set to unit scale, while binary fingerprints and substructure counts are passed to the
> model unscaled. Representations were precomputed in Python using \texttt{RDKit} and the two
> pretrained encoders; a Rust component performs label processing, noise injection and serialization.

**Two notes.**

- The ChemBERTa caveat is not optional. The tokenizer collapses the checkpoint's chemical vocabulary
  to single characters, so two chemically distinct molecules can receive one vector. The counting
  code exists (`scripts/crosscheck_chemberta.py`, gate 4) but **has not been run**, so the two
  percentages are `TODO` rather than quoted. Run it before this paragraph goes in.
- `\citep{avalon}` does not exist in `refs.bib`. Add it — Gedeck, Rohde & Bartels, *J. Chem. Inf.
  Model.* 46(5):1924–1936, 2006. `\citep{Chithrananda2020}` **does** exist in `refs.bib` (line 1427).

---

## M3. Models: rewrite (replaces lines 207–219; the last paragraph moves to M5)

**Why wholesale:** it names eleven families and the run has nineteen configurations; two of its
statements about the SVM and the GP kernels are contradicted by the code; the VBLL description is of
a variant that is deliberately excluded from the run; and the closing decomposition paragraph is now
its own subsection.

**Keep the opening paragraph as it stands** (lines 207–208, the Kolmar framing). It is still exactly
right and it sets up the whole paper. Change only its final sentence, and continue:

> All hyperparameters for the models below are included in Additional file~1. Nineteen model
> configurations were evaluated, each on all six representations except where noted. No model merges
> its validation split into its training data, so every configuration is fitted on the same 80\% of
> the sample and validation is reserved for early stopping and calibration.
>
> Random Forests (RFs) are one of the most common choices for QSAR modeling thanks to their
> robustness and interpretability \citep{Svetnik2003}. During bootstrapping, each tree trains on a
> separate subset of data, reducing the influence of any one individual label. This mechanism is
> particularly useful when working with noisy labels \citep{Breiman2001}. Quantile Regression
> Forests (QRFs) extend RF predictions to full distributions by keeping the distribution of training
> labels in each leaf and computing quantiles across all trees for a given prediction
> \citep{Meinshausen2006}. It inherently accounts for heteroscedasticity, such that the quantiles
> will reflect the spread of a target variable's domain. Both forests use a minimum leaf size of
> five rather than one, which is what allows a within-leaf spread to be computed for every
> prediction (see Uncertainty quantification). We also used eXtreme Gradient Boosting (XGBoost)
> \citep{Mustapha2016, Tian2022}, Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm},
> and Natural Gradient Boosting (NGBoost) \citep{Duan2020}. NGBoost extends deterministic gradient
> boosting by treating the parameters of a chosen parametric distribution as regression targets and
> learns them via boosting with a natural gradient update rule \citep{Duan2020}; we fit a Gaussian
> distribution by maximum likelihood, over at most 500 boosting iterations, with the number used
> selected by early stopping on the validation split. Computationally, NGBoost scales like standard
> boosting in $N$ but with larger constants that depend on the number of distributional parameters
> $p$: each iteration fits $p$ learner series and inverts $N$ matrices of size $p \times p$, adding
> $O(Np^3)$ cost per iteration \citep{Duan2020}. Even with a two-parameter Gaussian distribution,
> this additional cost is non-trivial and may limit scalability to larger datasets.
>
> Support Vector Machines (SVMs) are a well-established baseline in QSAR modeling, mapping inputs
> into a high-dimensional feature space where a maximum-margin hyperplane separates predictions
> \citep{Vapnik1995, Svetnik2003}. A radial basis function (RBF) with $C = 1.0$ and
> $\gamma = \texttt{scale}$ was used as the kernel, on every representation, so that the SVM carries
> no kernel-by-representation confound.
>
> One ML method that has had particular success in molecular property prediction are Gaussian
> Processes (GPs) \citep{Obrezanova2007, gauche}, a non-parametric class of models that use a kernel
> to produce a Gaussian predictive distribution over every data point \citep{Obrezanova2007}. One of
> the major limitations of using GPs in practice is computational scaling; inference has $O(N^3)$
> time complexity and $O(N^2)$ memory, where $N$ is the number of training points
> \citep{Rasmussen2005}. This can be mitigated with sparse GP approximations, but these introduce
> approximation error and can be more difficult to tune \citep{quinonero2005}. Tree-based ensembles
> are typically more efficient for larger data sizes \citep{lakshminarayanan2017}. We implemented GP
> models using the \texttt{Gauche} framework \citep{gauche}, which provides kernels optimized for
> molecular input \citep{gauche}. Rather than assigning a kernel per representation, we evaluated
> the two kernels as separate models: an RBF kernel, which runs on all six representations, and the
> Tanimoto kernel \citep{Ralaivola2005, moss2020, gauche}, which is defined on binary vectors and
> therefore runs on ECFP4 alone, so that the kernel comparison is like-for-like where both are
> defined. The RBF lengthscale is initialised at the median pairwise distance between training
> molecules rather than at the library default. This matters: on these representations pairwise
> distances run in the tens to hundreds, and a lengthscale left near unity gives the kernel nothing
> to learn from, so the process returns its prior and predicts a single value for every molecule. A
> fit whose predictions vary by less than 5\% of the standard deviation of the labels it was fitted
> on is recorded as collapsed and excluded rather than scored. Gaussian processes were fitted on a
> random subsample of at most 5,000 training molecules; every other model sees the full training
> split. We additionally evaluated a heteroscedastic variant in which the single observation-noise
> parameter is replaced by a small feed-forward network predicting the noise variance from the
> molecule's features, trained jointly with the process.
>
> While standard feed-forward neural networks (NNs) are deterministic, they can be transformed into
> probabilistic models. We tested two architectures: (i) NN-$\alpha$ with two hidden layers of sizes
> [128, 64] and dropout after each hidden layer and (ii) NN-$\beta$ with two hidden layers of size
> 128 and a single dropout layer before the output. Both use ReLU activations and dropout $p=0.2$,
> and were implemented in \texttt{PyTorch} \citep{pytorchGeometric}. Each was trained using the Adam
> optimizer at a learning rate of $10^{-3}$ with a batch size of 32, for up to 100 epochs, with
> early stopping on the validation loss after ten epochs without improvement and the best weights
> restored.
>
> We used Bayesian transformations to convert NNs into Bayesian neural networks (BNNs). BNNs
> introduce priors on the weights and compute a posterior distribution given the data, providing
> uncertainty quantification. This can be done by approximations such as Monte Carlo dropout,
> variational inference, or ensembles \citep{gal2016}. Three transformations were applied to each of
> the two architectures. In the full-BNN, every linear layer is replaced by a Bayesian layer with a
> Gaussian prior $\mathcal{N}(0, 0.1^2)$ on all weights, and the network is fitted on the evidence
> lower bound with the Kullback--Leibler term scaled by $1/N$. The Variational Bayesian Last Layers
> (VBLL) transformation \citep{Harrison2024} replaces every linear layer with a mean-field
> variational layer $q(\mathbf{W}) = \mathcal{N}(\boldsymbol{\mu}_W,
> \text{diag}(\boldsymbol{\sigma}_W^2))$ against a standard normal prior, and the output layer
> additionally carries a learned scalar observation-noise variance; it is trained by maximizing the
> evidence lower bound, which can be thought of as minimizing the reconstruction loss plus KL
> divergence $D_{\text{KL}}(q(\mathbf{W}) \| p(\mathbf{W}))$, scaled by $1/N$
> \citep{Harrison2024}. A heteroscedastic VBLL variant replaces that scalar with a noise variance
> predicted from the input, so that observation noise varies per molecule. Finally, a mean-variance
> estimation (MVE) variant takes the full-BNN and widens its output head to a mean and a
> log-variance, fitted with a Gaussian negative log-likelihood; this is the only network family in
> which both uncertainty components vary per molecule. All BNN variants estimated predictive
> distributions with 100 Monte Carlo forward passes at inference.
>
> Hyperparameters were held at a single shared default across every model, representation and noise
> level, so that the noise axis is the only thing that moves. A random search of 12 settings per
> model--representation pairing was run on clean labels; a tuned setting was adopted only where it
> beat the shared default on the held-out QM9 test split and survived the same comparison on the
> three assay datasets. TODO: state how many pairings were adopted, once the sweep is final.
> Searching separately at each noise level was not attempted, as it would make ``the tuned setting''
> a different setting at every level and confound the noise axis with model capacity.

**Three notes.**

- **The SVM/Tanimoto claim at line 197 must go.** The SVM is RBF on every representation, on both
  pipelines, with no branch on the representation anywhere. The paper contradicts itself here: lines
  212 and 214 already say RBF. Line 197 is the wrong one.
- **The VBLL description at line 216 describes a model that is deliberately not run.** The paper
  says VBLL "maintains a mean-field variational posterior over the last-layer weights". A
  last-layer-only variant exists in the code and is excluded from the roster; what runs makes
  *every* layer variational and keeps the learned observation noise on the output layer alone. Both
  halves of that sentence need changing.
- **NN-$\beta$'s dropout.** The paper's "[128, 128] with dropout applied before the output layer" is
  right, and is worth keeping in that form — the two architectures differ in where dropout sits as
  well as in width, and that is the point of having both.

---

## M4. Noise Strategies → Label noise: full rebuild (replaces lines 312–362, including Table 1)

**Why wholesale:** four of the six strategies in Table 1 exist in neither injector; the two that
survive by name have different definitions; $\sigma$ no longer exists as a flag or as a concept; the
level grid changed; and the sentence that validation is clean is now false. There is nothing here to
patch.

> Our objective is to evaluate the robustness of QSAR models against label noise. To do so, we
> ``inject'' artificial noise into the labels during training. Test labels are never modified.
> Validation labels carry their own noise, drawn independently at the same amount, because a model
> that early-stops against clean labels is deciding when to stop using information no practitioner
> has.
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
> The amount of noise is dose-matched across conditions. For each condition we compute the root mean
> square perturbation its per-molecule scale map and its shape would deliver at unit scale, and then
> solve for the internal scale that makes the delivered root mean square perturbation equal the
> requested amount. Every dose-matched condition therefore delivers the same amount of corruption at
> the same noise level, and a difference in outcome between two conditions is a difference of
> pattern alone rather than of magnitude. The noise level is expressed as a fraction of the standard
> deviation of the clean training labels, so that one level means the same relative corruption on
> QM9 and on each assay dataset. We swept levels $\{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$. The amount
> actually delivered is recorded for every run alongside the amount requested.
>
> Censoring sits outside this scheme, because recording a label at an assay limit is not a
> perturbation of a chosen size. Its level is instead the fraction of labels clipped, swept over
> $\{0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50\}$. The limit is a quantile of the clean training labels
> and is applied unchanged to every split, as a fixed property of an assay would be.
>
> Which molecules a condition targets is drawn from a seed that does not depend on the noise level,
> so the affected set is the same at every level, including zero. The clean-label run is therefore a
> negative control on the same molecules rather than on a different draw, which is what makes a
> level-zero baseline subtractable from the per-molecule results in the following section.
>
> The grouped conditions interact with the scaffold split in one way worth stating. Group membership
> is the Bemis--Murcko scaffold, and the split holds out whole scaffold groups, so a held-out
> molecule shares no scaffold with any affected training group. For those conditions the recorded
> noise pattern on held-out molecules is flat, and questions about which individual molecules were
> corrupted are answered on the training molecules by cross-fitting rather than on the test set.
>
> Gaussian and the two grouped conditions were run across the full grid of models and
> representations. The Student-$t$, Laplace and outlier conditions were run on a reduced set of
> model--representation pairings, since they probe the shape of the error and the size of the
> contaminated fraction rather than the model ranking. Censoring was run on a named subset of five
> pairings to measure the size of its effect, not to compare models or representations under it.
>
> Although classification tasks were not explicitly tested in this study, NoiseInject also
> implements six label-flipping strategies for classification: uniform, class-imbalance,
> binary-asymmetric, instance, class-dependent, and confusion-directed flipping. These use a flip
> probability $p$ in place of the noise level.

**New Table 1.** This replaces `tab:regression_noise` entirely. Per-point scale formulas do not
belong in it any more — the whole point of dose matching is that the delivered magnitude is equal
across rows and is set by the level, not by the row.

| Condition | Shape | Targeting | Simulated real-world source |
|---|---|---|---|
| Gaussian | Gaussian | every molecule, same amount | Random measurement error |
| Student-$t$ ($\nu = 5$) | Student-$t$, $\nu = 5$ | every molecule, same amount | Measurement error with heavier tails |
| Laplace | Laplace | every molecule, same amount | Error distributions fitted to bioactivity data |
| Grouped-wider | Gaussian | whole scaffold groups covering 20\% of molecules receive a $3\times$ wider error | Part of the chemical space measured less reliably |
| Grouped-shifted | Gaussian | every scaffold group receives a constant offset plus a within-group term, with $\rho = 0.62$ of the variance carried by the offset | Between-laboratory bias \citep{Bentz2013} |
| Outlier ($p = 0.10$) | Gaussian | a random 10\% of molecules receive a $3\times$ wider error | Transcription errors, sample mix-ups |
| Censoring | — | labels beyond an upper assay limit are recorded as the limit; the level is the fraction clipped | Assay dynamic range |

Suggested caption: *Noise conditions. Each condition is a shape (the distribution one error is drawn
from) paired with a targeting (which molecules are affected, and by how much). Every condition
except censoring is dose-matched, so at a given noise level each delivers the same root mean square
perturbation and differs only in how that perturbation is distributed across molecules. $\rho$ is
the share of total variance carried by the group-level offset.*

**Four notes.**

- **Outlier selection is random, not by $z$-score.** The paper's $|z| > 2$ rule is gone; the premise
  that measurement error tracks the measured value was tested and did not hold. If you keep any
  sentence about outlier noise, this is the one thing it must say differently.
- **$\rho = 0.62$ comes from Bentz et al. 2013, Table 7.** `Bentz2013` is in `citations.bib`
  (line 2107) but **not** in `refs.bib`. Port it.
- **The censoring sentence is deliberate wording**, carried from the settled condition file: it
  keeps a reader from reading a five-pair result as a model comparison.
- **`fig_methods_noise_strategies` is the one Methods figure and must be regenerated** for the seven
  conditions. It should make dose matching visible, since equal delivered magnitude across
  conditions is the point of the redesign.

---

## M5. New: Uncertainty quantification (new subsection, absorbing lines 217–219)

**Why it needs its own subsection:** the current text is one paragraph at the end of Models saying
which models were *not* decomposed. That list is now wrong in both directions — the forests and
NGBoost are decomposed, and two networks are the study's cleanest decomposition case — and the
machinery that replaced it needs more room than a paragraph in Models can give it.

> Several of the models produce a predictive distribution rather than a point estimate. We separate
> that distribution's variance into an aleatoric component, which reflects noise in the labels, and
> an epistemic component, which reflects the model's uncertainty about itself. Epistemic uncertainty
> is model-driven and can be reduced by collecting either more or higher quality training data;
> aleatoric is data-driven and reflects the intrinsic noise in the labels. The separation is
> computed by a single shared routine used by both pipelines, in variance space throughout, with one
> conversion to a standard deviation at the point of reporting.
>
> How the two components are obtained depends on the family. For the stochastic networks we use the
> sampling decomposition of \citet{kendall2017}: the epistemic term is the variance across Monte
> Carlo forward passes of the predicted mean, and the aleatoric term is the mean across those passes
> of the variance the network itself predicted for that molecule. For the two forests we apply the
> law of total variance across the fitted trees, so that the aleatoric term is the mean within-leaf
> variance of the training labels in each molecule's leaf and the epistemic term is the variance of
> the per-tree leaf means; this uses the trees already fitted and requires no retraining, and both
> terms vary per molecule only because the minimum leaf size is five. For the Gaussian process the
> epistemic term is the latent posterior variance and the aleatoric term is the likelihood noise
> \citep{Rasmussen2005}; in the heteroscedastic variant the latter is predicted per molecule.
>
> Not every model supports both components, and the distinction matters when reading any per-molecule
> result. NGBoost is a single fit, so it has an aleatoric term per molecule and no epistemic term at
> all --- absent, rather than zero. The plain Bayesian networks predict a mean and nothing else, and
> so have an epistemic term and no aleatoric one. The Gaussian process with a homoscedastic
> likelihood, and the VBLL networks, each learn a single observation-noise value shared by every
> molecule; that is the correct aleatoric term for those models, but it cannot rank molecules, and
> its correlation with any per-molecule quantity is undefined rather than zero. Every reported
> uncertainty is accompanied by a declaration of whether each component varies per molecule, is one
> value per fit, or does not exist, and a check refuses to record a value that disagrees with that
> declaration.
>
> Questions about individual molecules cannot be asked of the training set using a model that has
> already fitted those labels. For the model--representation pairings where per-molecule uncertainty
> is analysed, training molecules were therefore scored out of fold: the training block was divided
> into five folds using \texttt{GroupKFold} on Murcko scaffolds, matching the outer split, and each
> molecule was scored by a model fitted without it. Noise is injected once, before this division, so
> a molecule carries the same corruption in whichever fold it falls. For NGBoost, three of the five
> folds were scored for reasons of cost, and molecules in the remaining two carry no out-of-fold
> score.
>
> Two of the conditions give every molecule the same amount of noise by construction --- the
> uniform-targeting conditions, and grouped-shifted, whose offset applies to a whole group at once.
> Under those conditions the question of which individual molecules are unreliable is undefined
> rather than answered negatively, and we report it as such.
>
> Uncertainties are reported without post-hoc calibration. A single temperature multiplier is also
> fitted and recorded; because it is one positive number it cannot change the ordering of molecules,
> and reporting coverage after fitting a multiplier that corrects coverage would be circular.

**Note.** Lines 217–219 of the current Methods are fully replaced by this. In particular, "We did
not decompose the uncertainty from other models such as BNN variants … and NGBoost and QRF, in which
there is no straightforward mechanism to separate the aleatoric and epistemic components" is now
wrong about all three: both forests are decomposed by the law of total variance, NGBoost carries a
per-molecule aleatoric term with an explicitly absent epistemic one, and the two mean-variance
networks carry both halves per molecule.

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
  tables (lines 503, 508) cannot be regenerated.
- **Mean prediction-interval width is not computed** by anything in the study, only by the
  standalone package. Lines 368 and 620 both list it.
- **"Eleven noise levels"** (lines 240, 245) and $\sigma \in \{0, 0.1, \ldots, 1.0\}$ are wrong
  wherever they appear, including inside the metric definitions.

---

## M7. NoiseInject framework: three sentence fixes (lines 364–373)

Not a rebuild. Three claims to correct, all of the same kind — the package does more than the study
uses, and the text currently reads as though everything listed was benchmarked here.

- Line 368 lists ECE and mean prediction-interval width among the metrics computed. Both are package
  features; neither appears in this study's results. Either say so, or drop them from the sentence.
- Line 352 says the classification strategies "mirror the regression set". They still exist, but the
  six regression strategies they mirrored no longer do, so the sentence needs rewriting rather than
  deleting. Suggested: *"NoiseInject also implements six label-flipping strategies for
  classification tasks, which are not benchmarked here."*
- Line 371's reference wrappers include split-conformal prediction. Conformal prediction is refused
  by name in this study's pipeline and appears in no result. Keep it as a stated package feature, or
  drop it; do not leave it ambiguous.

---

## Methods: sentences the code contradicts

Every line number confirmed by searching for the quoted text in the current `paper.tex`.

| Line | What it says | What the code does | Fix |
|---|---|---|---|
| **193** | split "implemented by DeepChem" | A purpose-written scaffold splitter that gives each acyclic molecule its own group; DeepChem's was replaced because it put nearly all acyclic molecules in training | M1. Drop the `\citep{deepchem}` here |
| **193** | replicates split "from this fixed subset" | Each replicate draws its own 10,000 molecules | M1 |
| **197** | "Tanimoto kernel was used for SVM" for binary representations | SVM is RBF on every representation, both pipelines. Lines 212 and 214 already say so | Delete the clause; the paper contradicts itself |
| **197** | hERG "N = 1,482" | 1,415 | M1. Also line 556 |
| **197** | "all six noise injection strategies and eleven levels" | Seven conditions, seven levels on a different axis | M4 |
| **199** | targets "mean-centered and normalized" | True, but silent on the two things that were fixed: the statistics are the clean *training* ones, and noise is added before standardisation | M1 |
| **203** | one-hot SMILES and mol2vec are study representations | Both refused by name; mol2vec is deleted from the pipeline | M2 |
| **203** | Sort & Slice gives "a binary vector" | It carries substructure counts | M2 |
| **205** | "Rust was used to perform … feature extraction" | Rust computes no representation; it does label processing, noise injection and serialization | M2 |
| **214** | Tanimoto for fingerprints, RBF for PDV | An RBF GP on all six representations and a Tanimoto GP on ECFP4 alone, as two separately reported models | M3. Also line 418's "incompatible with PDV" |
| **216** | VBLL is a posterior "over the last-layer weights" | Every layer is variational; the last-layer-only variant is excluded from the run | M3 |
| **218** | forests, NGBoost and BNN variants were not decomposed | All three are, and the two mean-variance networks are the cleanest case | M5 |
| **222, 240** | $\sigma$ is "the noise scaling factor shared by all strategies", swept 0 to 1.0 in elevens | No such flag or concept; the level is the delivered amount, seven values to 1.5 | M4, M6 |
| **234–238, 300** | ECE defined and reported | Computed nowhere | M6 |
| **313** | "Validation and test data remain free of noise" | Test yes, validation no | M4 |
| **318–350** | Table 1, six strategies with per-point scales in $\sigma$ | Four of the six exist nowhere; the surviving two are defined differently | M4, new table |
| **354** | outlier noise hits samples with $z > 2$; threshold acts on $|y| > 1$ "on normalized data" | Outlier selection is random; noise is applied to the raw label before standardisation, so no rule can be phrased in standardised units | M4 |
| **368, 620** | interval width and ECE among computed metrics | Package features only | M7 |

**Smaller ones**, unchanged from the previous guide and still open: line 274 "observations … are
independence" → "are independent"; lines 260 and 432 disagree on `R² ≤ 0.6` versus `R² < 0.6`, and
both are superseded by whatever the metrics pass settles.

---

## Methods: flag before you write

Eight things I could not settle from the code. Each is phrased as what would settle it.

1. **The 1,403 excluded QM9 molecules.** The pool is 129,428 because a pre-computed index file
   removes 1,403 of PyG's 130,831, and no script in any of the three repositories writes that file.
   Its only description is a comment saying they could not be processed by RDKit. The draft says
   "could not be processed by \texttt{RDKit}" on that authority alone. **Settle by** regenerating the
   list — parse every SMILES and compare the surviving set — or by dropping the clause and saying
   only the count.
2. **The two ChemBERTa collision percentages.** The counting code exists and has not been run.
   **Settle by** running `scripts/crosscheck_chemberta.py` and reading gate 4's output.
3. **Stereochemistry differs between the two pipelines.** QM9 canonicalises without stereochemistry;
   the assay side keeps it. The four static representations are unaffected, but MHG-GNN is not — the
   runner's own measurement puts the difference at several feature standard deviations, and about a
   quarter of the LogD molecules carry stereochemistry. The code records this as an open decision,
   not a defect. **Settle by** choosing one and changing one side, or by stating the difference in
   Methods.
4. **The dose anchor differs between the two pipelines.** QM9 doses against the clean *training*
   label spread; the assay runner doses against the standard deviation of the whole clean label
   column, deliberately, so that a molecule's corruption does not depend on which fold it lands in.
   The draft in M1 quotes the assay spreads without saying which anchor produced them. **Settle by**
   deciding whether one sentence in Methods covers it or whether the two should be aligned.
5. **The tuning sweep.** Four Bayesian networks are marked for tuned settings in the files on disk;
   every other model runs at the shared default. The draft leaves a `TODO` for the count. **Settle
   by** confirming the sweep is final.
6. **Two missing citations.** `avalon` is in neither bib file; `Bentz2013` is in `citations.bib`
   only. `Chithrananda2020` is present in `refs.bib`. Also worth deciding which bib file is live —
   `paper.tex` currently points at neither, it says `\bibliography{sn-bibliography}`.
7. **Which models the paper says decompose.** The M5 draft describes the machinery for every family.
   Whether the Results report all of them, or only the ones that pass the level-response check, is a
   Results decision that changes one sentence here.
8. **Abbreviations (lines 579–615).** **ECE** and **NDS** are both retired and must come out;
   **MVE** needs adding. Mechanical, and doable at the same time as the Methods.

---

# PART TWO — LIMITATIONS

New paragraph, to sit before the closing paragraph of the Conclusion. Written now because every item
in it is a property of the design rather than of the results.

> This study has several limitations. The QM9 labels are computed rather than measured, so the noise
> we inject is imposed on a target that is otherwise clean; on the three assay datasets the injected
> noise sits on top of real measurement error that we cannot separate from it, and the noise level
> there should be read as an amount added rather than an amount present. Each configuration on the
> assay datasets is fitted once rather than repeated under new seeds, so those results carry
> variation across the five cross-validation folds --- which mixes sampling with scaffold difficulty
> --- but no run-to-run error term, and no comparison between two models on those datasets should be
> read as significant on the strength of the fold spread alone. Two questions are undefined by
> construction rather than answered negatively: under the conditions that give every molecule the
> same amount of noise, and for the models whose observation-noise term is a single learned value,
> asking which individual molecules a model finds unreliable has no answer to give. The Gaussian
> processes are fitted on a subsample of at most 5,000 training molecules where every other model
> sees the full training split, so their comparison with the other families is not held at equal
> training size. The ChemBERTa results are results for one checkpoint whose tokenizer operates
> character by character, and should not be read as a statement about pretrained transformer
> embeddings in general. Finally, hyperparameters are held fixed across the noise axis, which keeps
> that axis clean but means we do not measure whether a model could be retuned to resist noise
> better than it does at its default.

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
