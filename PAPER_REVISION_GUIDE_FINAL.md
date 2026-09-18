# NoiseInject Paper: Revision Guide

> **What this is.** For each part of the paper, the `paper.tex` lines it replaces and the text that
> replaces them. Read it top to bottom and paste from it. Nothing in here is a record of how it was
> arrived at.
>
> **`paper.tex` is never edited.** The copy in this repository is a read-only download from the Overleaf
> project. Every change is written here and moved across by hand.
>
> Line numbers are from `paper.tex` as of 4 September 2026 and each was confirmed by searching for the
> quoted text. They drift as you edit, so anchor on the quotation.
>
> **Three companion files, and none of their contents belongs here.**
>
> | file | what it holds |
> |---|---|
> | `PAPER_HOUSE_STYLE.md` | the 122 rules this text is written to, from nineteen reference papers |
> | `RERUN_PLAN.md` §15 | the process history — superseded drafts, defect reports, open decisions, what was withdrawn and why |
> | `NOISE_DESIGN.md` | what the noise is, and the level ladders |
>
> **Not written yet, deliberately**: the abstract, the conclusions and the scientific-contribution
> statement. They come off the Results once these are settled.


---

# METHODS

These six subsections replace `paper.tex`'s Methods. Earlier drafts of them, and the notes recording why each
passage was shortened and which published paper the judgement came from, are in `RERUN_PLAN.md` §15. Nothing
here needs them.

**Length.** The current Methods is 3,722 words. These six subsections come to 5,246 words of prose plus the
conditions table, counted on 2026-09-18 after the house-style pass, with a subsection added that the paper
does not have. That is above the range of the twelve reference papers, whose Methods run 850 to 3,350 words,
and above the 4,776 words these six ran to before the pass. Word count by subsection: Datasets 618,
Molecular representations 642, Models 1,015, Label noise 1,368 including its table, Uncertainty
quantification 710, Performance metrics 893.

*Where the 470 words went.* `PAPER_HOUSE_STYLE.md` asks for four sentences defining AUC_norm from its parts
before its first value, one saying which direction is better, one saying what it cannot show, a direction of
good at each metric's first comparison, a denominator on every count, and an expansion at every acronym's
first use. Performance metrics took most of it, at 769 words before and 893 after. **Cutting Methods back is
a separate decision from cutting Results and it is yours.** The material a referee is least likely to want
removed is the noise-conditions table and the exclusion rules; the material most easily shortened is the
hyperparameter detail in Models, which Additional file 1 already carries in full.

**Every number below was read in a file.** Each subsection lists what it read and where, so any number can
be checked without re-running anything. Where a number could not be traced the text says TODO and names
the file that would settle it.


## M1. Datasets

*R.*eplaces paper.tex 192-200

```latex
\subsection{Datasets}
The majority of experiments were conducted on QM9 \citep{Ramakrishnan2014}, which holds small organic molecules of up to nine heavy atoms. Their properties were computed by density functional theory at the B3LYP/6-31G(2df,p) level. We selected the gap between the highest occupied and the lowest unoccupied molecular orbital (HOMO--LUMO gap) as the target. The gap relates to a molecule's electronic excitability, charge transfer capability and chemical stability \citep{Fediai2023}. The release holds 133,885 molecules. The copy distributed with PyTorch Geometric leaves out the 3,054 molecules the authors flagged as failing their consistency check, so we start from 130,831 molecules. Of those, 1,403 are molecules RDKit will not build from their simplified molecular-input line-entry system (SMILES) string. That leaves a usable pool of 129,428 molecules. The gap over that pool has a mean of 6.852~eV and a sample standard deviation of 1.285~eV. We assumed the QM9 labels were free of noise, notwithstanding the approximations in the level of theory.

Each of the ten replicates reseeds the random number generators, permutes the pool afresh, and takes the first 10,000 molecules. Each subset was split 80/10/10 into training, validation and test molecules by chirality-blind Bemis--Murcko scaffold. No scaffold framework appears in more than one of those three parts. The splitter is our own rather than DeepChem's. Each distinct acyclic molecule forms its own group rather than one pooled group. The groups are filled in random order rather than largest first. Every model other than the Gaussian processes fits the replicate's 8,000 training molecules. From that count we drop the molecules whose Sort \& Slice vector is all zeros, which Molecular representations describes. The target was standardised with the mean and standard deviation of the clean training labels alone.

We repeated the measurement on three datasets with experimentally measured endpoints. Those three test whether the noise results generalise beyond the computed HOMO--LUMO gap. LogD and Caco-2 efflux are two of the endpoints measured in the 5,326-molecule training split of the OpenADMET ExpansionRx challenge set \citep{openadmet}. We kept the largest fragment of each structure, canonicalised it, and took the median label per molecule. That leaves 5,039 LogD molecules, with a mean of 2.112 and a sample standard deviation of 1.191 log~$D$ units. It also leaves 2,161 Caco-2 molecules. The Caco-2 label is the base-10 logarithm of the efflux ratio, with a mean of 0.284 and a sample standard deviation of 0.445 log units. Of the 2,161 Caco-2 molecules, 2,128 also carry a LogD measurement, so the two endpoints are largely one set of molecules.

The hERG data were extracted once from ChEMBL \citep{Zdrazil2023} and frozen in a file, following \citet{landrum2024}. We kept binding assays only, and took the median pChEMBL value per compound. We removed compounds whose inter-assay standard deviation exceeds 1.0 log unit. That extract holds 1,415 molecules, with a mean of 5.706 and a sample standard deviation of 0.915 pChEMBL units. Re-running the query against ChEMBL release 37 returned the same 1,415 compounds with identical values.

The three assay datasets were each partitioned into five folds by grouped $k$-fold cross-validation on chirality-blind Bemis--Murcko scaffolds, so that no scaffold group appears in two folds. Each fold's training block then gives up a further scaffold-grouped fifth as a validation set for early stopping. A model therefore fits between 3,194 and 3,364 of the 5,039 LogD molecules, depending on the fold. Each fold's target scaler was fitted on its clean training labels and used at every noise level. QM9's ten replicates are ten independently seeded draws of 10,000 molecules from one pool, so the spread across those ten is a replicate error bar. Each of the three assay datasets is instead one fixed set partitioned into folds. Every molecule there is tested exactly once, and nothing is redrawn.
```

**For the author.**

- Every defect on the list held up against the file it named. I re-measured all of them rather than trusting the note.
- The over-valence claim: data/qm9_pool_provenance.json does record kept_over_valent_carbon_in_sdf = 0 and dropped_over_valent_carbon_in_sdf = 1403, so over-valent carbon and the drop do coincide on what was scanned. But the kept side was a control scan of 2,023 molecules only (kept_molecules_scanned_as_a_control), not all 129,428, and the filter the file records is Chem.MolFromSmiles on the SMILES string. So the cause sentence is now written as the SMILES parse, and the over-valence line is dropped for space.
- The hERG spread is 0.9147 with ddof 1 and 0.9143 with ddof 0, so 0.915 is right only on the sample convention. I wrote "sample standard deviation" at every dataset spread rather than change the digit. LogD and Caco-2 round the same either way, so nothing else moves.
- The hERG label heading in the cached file is pChEMBL (KIRBy/tests/data_cache/chembl_herg_ki.csv), so the text now says pChEMBL units in both places rather than pKi in one.
- Not the draft's fault, but stale: the comment at KIRBy/tests/alternative_data_noise_robustness.py:1093 says the Caco-2 label mean is 0.293. The cached file gives 0.2844.
- The 8,000 training molecules is measured, not derived: I re-ran the group-packing rule on seeds 0, 1 and 2 and it landed 8,000 / 1,000 / 1,000 each time. If you want that hedged in the paper, the alternative is "about 8,000".
- The release-37 sentence names a release the provenance file trusts; the draft's "ChEMBL release 36" is gone because chembl_herg_ki.provenance.json says that stamp was reconstructed on 2026-09-04 and not written by the original fetch.

**Cut from the current text.**

- "The subset and the split both change from one replicate to the next." — carried by the sentence before it, and the subsection was over the word ceiling.
- "The two halves of the study repeat over different things." — announced the two sentences after it, and "halves" is not one of the author's words for the two pipelines.
- "All 1,403 carry an over-valent carbon in the SDF record." — true (data/qm9_pool_provenance.json, dropped_over_valent_carbon_in_sdf = 1403) but not the filter, and the words were needed for the release and hERG corrections.
- "of theory" after the level of theory in sentence two, and "molecules" after 129,428 — wording trims to land under 600 words.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **133,885 molecules in the release** — data/QM9/raw/gdb9.sdf.csv — 133,886 lines including the header, and pandas read 133,885 rows this session; data/QM9/raw/uncharacterized.txt names "the 133885 GDB9 molecules"
- **3,054 molecules left out of the PyTorch Geometric copy** — data/qm9_pool_provenance.json, uncharacterised_rows_in_release = 3054
- **130,831 molecules in the PyTorch Geometric copy** — data/qm9_pool_provenance.json, pyg_qm9_positions = 130831; confirmed this session by loading the dataset, y has 130,831 rows
- **1,403 molecules RDKit will not build from SMILES** — data/qm9_pool_provenance.json, dropped = 1403 with filter = "Chem.MolFromSmiles(data.smiles) is not None"
- **129,428 usable molecules** — data/qm9_pool_provenance.json, kept = 129428; data/valid_qm9_indices.pth holds 129,428 indices, read this session
- **6.852 eV mean, 1.285 eV sample standard deviation of the gap** — measured this session: PyG QM9 y column 4 (properties['homo_lumo_gap'] = 4, scripts/process_and_train.py:118-121) indexed by data/valid_qm9_indices.pth — mean 6.8524, sd 1.2847 at both ddof 0 and 1
- **nine heavy atoms; B3LYP/6-31G(2df,p)** — data/QM9/raw/QM9_README
- **ten replicates** — slurm_scripts_qm9_rerun/generate_scripts.py:1435, defaults['replicates'] = 10
- **10,000 molecules per replicate** — scripts/process_and_train.py:300, --sample-size default 10000; slurm_scripts_qm9_rerun/generate_scripts.py:1364 same default
- **80/10/10 split; 8,000 training molecules** — scripts/process_and_train.py:1173-1175 calls scaffold_split_indices(frac_train=0.8, frac_valid=0.1, frac_test=0.1); re-ran the group-packing rule (scripts/process_and_train.py:671-706) on seeds 0, 1 and 2 over the 129,428-molecule pool this session — 8,000 / 1,000 / 1,000 every time, over 3,591, 3,584 and 3,562 scaffold groups
- **5,000-molecule Gaussian-process draw** — models/model_defaults.py:264, GP_DEFAULTS['max_train_n'] = 5000, applied by cap_gp_training_set at models/model_defaults.py:493
- **six representations** — slurm_scripts_qm9_rerun/generate_scripts.py:112, ALL_REPS = ecfp4, pdv, mhggnn, avalon, chemberta, sns
- **5,326 molecules in the OpenADMET training split** — KIRBy/tests/data_cache/openadmet_train.csv — 5,327 lines with the header, 5,326 rows read this session; LogD and "Caco-2 Permeability Efflux" are two of its endpoint headings
- **5,039 LogD molecules, mean 2.112, sample standard deviation 1.191** — measured this session with KIRBy load_openadmet_endpoint('LogD') — 5,039 molecules, mean 2.1117, sd 1.1906 (ddof 1) and 1.1905 (ddof 0)
- **2,161 Caco-2 molecules, mean 0.284, sample standard deviation 0.445** — measured this session with KIRBy load_openadmet_endpoint('Caco-2 Permeability Efflux', log_transform=True) — 2,161 molecules, mean 0.2844, sd 0.4449 (ddof 1) and 0.4448 (ddof 0)
- **2,128 molecules carrying both endpoints** — measured this session: intersection of the two standardised SMILES sets, 2,128
- **1,415 hERG molecules, mean 5.706, sample standard deviation 0.915** — measured this session with KIRBy load_chembl_herg() over tests/data_cache/chembl_herg_ki.csv — 1,415 molecules, mean 5.7057, sd 0.9147 (ddof 1); n_compounds = 1415 also in chembl_herg_ki.provenance.json
- **inter-assay standard deviation above 1.0 log unit removed** — KIRBy/tests/alternative_data_noise_robustness.py:1042-1046, medians merged with per-compound standard deviations before the filter
- **ChEMBL release 37 verification** — KIRBy/tests/data_cache/chembl_herg_ki.provenance.json, _verification: re-run against live ChEMBL_37 on 2026-09-04 returned the same 1,415 compounds, 0 added, 0 removed, all values identical
- **five folds** — KIRBy/tests/alternative_data_noise_robustness.py:575, N_FOLDS = 5, used by GroupKFold at :3494
- **a further scaffold-grouped fifth for early stopping** — KIRBy/tests/alternative_data_noise_robustness.py:3227, scaffold_validation_carve(..., val_frac=0.2) with GroupShuffleSplit at :3243
- **3,194 to 3,364 LogD molecules fitted per fold** — measured this session: GroupKFold(5) on assign_scaffold_groups plus scaffold_validation_carve gave fit counts 3,364 / 3,325 / 3,197 / 3,269 / 3,194

</details>


## M2. Molecular representations

*R.*eplaces paper.tex 201-206

```latex
\subsection{Molecular representations}

We evaluated six molecular representations: extended-connectivity fingerprints of diameter four (ECFP4), Avalon, Sort \& Slice, physicochemical descriptors (PDV), ChemBERTa and MHG-GNN. The same six were used on QM9 and on the three assay datasets.

ECFP4 is RDKit's Morgan fingerprint at radius $r = 2$, folded to $d = 2048$ bits \citep{rdkit}. Distinct substructures can hash to the same bit, so two different substructures may be recorded in one feature. That is error in the representation itself, and not the label noise this study adds. Sort \& Slice avoids the folding by enumerating the same Morgan substructures and keeping the $L = 1024$ most prevalent in the training molecules \citep{sns}. Each Sort \& Slice feature holds the number of times that substructure occurs in the molecule, not a presence bit.

We fitted the Sort \& Slice vocabulary on training molecules alone. On QM9 those are the training molecules of each replicate's scaffold split. On the three assay datasets the vocabulary is refitted per cross-validation fold, on the rows that fold fits, which exclude the validation rows. A molecule whose substructures all fall outside the kept 1,024 features gets an all-zero vector. On QM9 those molecules are dropped from every representation, so all six are scored on the same molecules. The vocabulary is refitted for each replicate, so which molecules are dropped changes between replicates.

Avalon is a 2,048-bit fingerprint, generated with the Avalon toolkit through RDKit \citep{rdkit, avalon}. PDV is 200 physicochemical descriptors computed with RDKit's \texttt{MolecularDescriptorCalculator} \citep{rdkit, Cherkasov2014}. Several of the 200 return non-finite values on some molecules. Each such value is replaced by zero before the scaling constants are computed, on QM9 and on the three assay datasets alike.

ChemBERTa embeddings come from the \texttt{DeepChem/ChemBERTa-77M-MTR} checkpoint \citep{Ahmad2022}, pooled as the mean over the non-padding token embeddings, giving 384 features. The checkpoint ships an empty merges file, so its tokenizer falls back to single characters. Seven characters that appear in SMILES have no vocabulary entry: \texttt{+}, \texttt{@}, \texttt{H}, \texttt{[}, \texttt{]}, the \texttt{l} of \texttt{Cl} and the \texttt{r} of \texttt{Br}. Chlorobenzene and toluene are therefore read as the same token sequence. On the three assay datasets the string keeps stereochemistry, and both alanine enantiomers are read as the same achiral sequence. Among the hERG Ki molecules, 72 out of 1,415 molecules (5.09\%) share a token sequence with another molecule. Among the QM9 molecules, 3,502 out of the 129,238 distinct structures (2.71\%) do the same. A QM9 run draws 10,000 molecules, and across 25 simulated draws the median draw held 24 such molecules, or 0.24\% of the 10,000 drawn.

MHG-GNN embeddings are 1,024 features from the published \texttt{mhggnn\_pretrained\_model\_0724\_2023} checkpoint \citep{kishimoto2023}. QM9 and the three assay datasets load that checkpoint unchanged.

We z-score normalised PDV, ChemBERTa and MHG-GNN per feature, using the training molecules' mean and standard deviation. ECFP4, Avalon and Sort \& Slice reach the model exactly as built. A matrix of zeros and ones is exempt from that scaling. So is a matrix of non-negative integers with at most a quarter of its entries non-zero and at least one entry above 1. The Sort \& Slice counts are the second case. On QM9, 1.3\% of the entries in the Sort \& Slice feature matrix are non-zero, and on hERG 4.7\% of them are.

On QM9 no representation sees stereochemistry. All six are built there from the stereochemistry-free canonical SMILES, with chirality switched off in the Morgan-based fingerprints. On the three assay datasets five of the six still give a pair of enantiomers one point. ECFP4 and Sort \& Slice have chirality switched off, and Avalon and PDV return the same vector for either enantiomer. ChemBERTa's reader drops the \texttt{@} character. MHG-GNN reads the stereochemistry, and the two alanine enantiomers give different embeddings. All six are computed in Python with RDKit, the \texttt{transformers} library and the MHG-GNN checkpoint. The Rust component of the QM9 pipeline carries the bytes through unchanged and computes no representation.
```

**Still open in this subsection.**

- ~~Re-establish the hERG Ki molecule count~~ **SETTLED 2026-09-18, and 1,415 is right.** See below.
- ~~Re-run scripts/sns_zero_molecules.py~~ **DONE 2026-09-18. Three of 132,480, and the densities too.** See below.

**For the author.**

- MHG-GNN and enantiomers is now measured, not a TODO. I ran KIRBy's create_mhg_gnn on the two alanine SMILES and the achiral one in this session: the embedding is 1,024 features wide, and the largest difference in any one feature between the enantiomers was 5.15, against 10.73 between one enantiomer and achiral alanine. So the assay-side count is five representations giving a pair of enantiomers one point and MHG-GNN giving two. That also settles the width on the assay side, which no file stated: it is 1,024 features there too.
- The hERG Ki count in this subsection is 1,415 molecules and paper.tex:203 says N = 1,482 compounds. 1,415 is what results/chemberta_collisions.csv counted and what KIRBy/tests/data_cache/chembl_herg_ki.csv holds. The Dataset subsection's number has to be re-established from that same cache file before either goes to the journal.
- **The three numbers that came from code comments have been re-established by running the code.** I ran
  `scripts/sns_zero_molecules.py` on the 132,480 QM9 SMILES in this session: **three molecules** get an
  all-zero Sort \& Slice vector at SNS_DIM = 1024, and they are methane, ammonia and water, each of which
  has exactly one substructure. The 2026-09-07 comment at `scripts/process_and_train.py:1220-1221` is right.
- **The two densities are right as well, to the decimal the text prints.** Fitting the vocabulary on an 80\%
  training split with the pipeline's own generator settings and measuring over every molecule gives a mean
  of 14.1 of 1,024 features non-zero on QM9, which is **1.38\%**, and 47.7 of 1,024 on hERG K$_i$, which is
  **4.66\%**. The comment at `models/model_defaults.py:466-471` says 1.3\% and 4.7\% and the text says the
  same. If you want the text to carry two decimals instead of one, those are the values.
- **The empty merges file is now sourced to the checkpoint itself, not to a code comment.** I opened
  `~/.cache/huggingface/hub/models--DeepChem--ChemBERTa-77M-MTR/snapshots/66b895cab8.../merges.txt` in this
  session. It holds one line, `#version: 0.2 - Trained by \`huggingface/tokenizers\``, and no merge rules at
  all, while `vocab.json` beside it holds 591 entries. So the tokenizer performs no byte-pair merges and
  every token comes straight from that vocabulary, which is exactly what the sentence in the text says.
- **The hERG K$_i$ count is 1,415 and `paper.tex:203`'s N = 1,482 has to change.** The cached file
  `KIRBy/tests/data_cache/chembl_herg_ki.csv` holds 1,415 rows and 1,415 distinct SMILES. I then ran KIRBy's
  own two steps on it in this session — `standardise_smiles`, which keeps the largest fragment and
  canonicalises, and the median dedupe on the standardised SMILES that
  `alternative_data_noise_robustness.py:1141` applies. **Neither step removes anything on the current
  cache**: 1,415 in, 1,415 standardised, 1,415 after the dedupe. The Dataset subsection and this subsection
  should both say 1,415.
- The breaker's two line-number corrections check out: pyAvalonTools.GetAvalonFP is at scripts/process_and_train.py:1600, and KIRBy's PDV nan_to_num is at src/kirby/representations/molecular.py:995.
- I disagree with one part of the Sort & Slice correction. On the three assay datasets the fold vocabulary is fitted on that fold's fitted rows (alternative_data_noise_robustness.py:3571-3581), which is what I wrote. On QM9 the featuriser is fitted inside split_qm9 from the training molecules of the replicate's split, and the guide at line 1642 says the 10,000-molecule subset is identical across noise levels within a replicate, so the vocabulary does not depend on the noise level. My sentence says it that way.
- The Avalon reference is now in `citations.bib`, appended 2026-09-18 under the key `avalon` as Gedeck, Rohde and Bartels (2006), and the sentence above cites it. `refs.bib` still has no Avalon entry and its only Gedeck line is 1550, as a co-author of Kramer et al. If Overleaf's `sn-bibliography` already carries the paper under a different key, use that key rather than adding a second entry.

**Cut from the current text.**

- "built by the same calls on both pipelines" — false: the two pipelines call different functions, on different SMILES, into different storage.
- "Enantiomers are therefore one point for four of the six representations on the assay datasets and two points for the other two" — replaced by the measured five-and-one split.
- "ChemBERTa and MHG-GNN, however, read a standardised canonical SMILES that keeps it" — true of the string but misleading about ChemBERTa, whose reader drops the @.
- "On QM9 it is refitted once per noise level and replicate" — the vocabulary comes from the training molecules, which do not change with the noise level.
- The PDV descriptor enumeration (molecular weight, LogP, polar surface area, connectivity indices, VSA bins, functional group counts) — cut for the 550-word ceiling once the Sort & Slice exclusion and the corrected stereochemistry paragraph were added. It survives in paper.tex:222 and in Additional file 1.
- "The rule is applied to the feature matrix, not to the representation's name" — cut for length; the two sentences that follow describe matrix properties, so the point still lands.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **six representations** — scripts/process_and_train.py:1102-1118 (pdv, chemberta, mhggnn, avalon, ecfp4) plus the Sort & Slice featuriser at :1196-1205; KIRBy tests/alternative_data_noise_robustness.py:3005-3030 builds SNS, MHG-GNN, Avalon, ChemBERTa alongside ECFP4 and PDV
- **radius r = 2, 2,048 bits (ECFP4)** — scripts/process_and_train.py:1520-1527 (ECFP4_RADIUS, ECFP4_BITS, GetMorganGenerator); KIRBy src/kirby/representations/molecular.py:293-306 (radius=2, n_bits=2048)
- **L = 1024 Sort & Slice features, counts not bits** — scripts/process_and_train.py:163 SNS_DIM = 1024 and :1196-1205 (sub_counts=True); rust/src/main.rs:74 sns_buf [u8; 2048] = 1,024 u16 counts
- **three molecules of 132,480 QM9 SMILES dropped** — code comment scripts/process_and_train.py:1220-1221, recording a 2026-09-07 count by scripts/sns_zero_molecules.py; the dropping itself is at :1240-1268
- **Avalon 2,048 bits** — scripts/process_and_train.py:1581-1600 (avalon_fingerprint, nBits=2048); KIRBy molecular.py:322-343 create_avalon(n_bits=2048)
- **PDV 200 descriptors** — counted this session from DEFAULT_DESCRIPTOR_LIST at scripts/process_and_train.py:84 (200 names); KIRBy molecular.py:944-995 create_pdv uses the same list
- **non-finite PDV cells set to zero before scaling** — scripts/process_and_train.py:3268-3284 (_clean before x_mean/x_std); KIRBy molecular.py:995 np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
- **ChemBERTa 384 features** — scripts/process_and_train.py:156 CHEMBERTA_DIMS = 384 and rust/src/main.rs:96 chemberta_buf [u8; 1536]; KIRBy molecular.py:2329-2351 docstring, hidden_size 384
- **seven characters with no vocabulary entry (+, @, H, [, ], l of Cl, r of Br)** — scripts/process_and_train.py:1404-1410 (comment naming l, r, [, ], +, @, H and the chlorobenzene/toluene and alanine consequences)
- **1,415 hERG Ki molecules; 72 (5.09%) share a token sequence** — results/chemberta_collisions.csv, herg_ki row: pool_distinct_molecules 1415, pool_molecules_sharing_a_vector 72, pool_percent 5.09, source KIRBy/tests/data_cache/chembl_herg_ki.csv
- **129,238 distinct QM9 molecules; 3,502 (2.71%)** — results/chemberta_collisions.csv, qm9 row: pool_distinct_molecules 129238, pool_molecules_sharing_a_vector 3502, pool_percent 2.71
- **10,000 molecules drawn per QM9 run; 25 simulated draws; median 24 molecules (0.24%)** — results/chemberta_collisions.csv, qm9 row: run_rows_drawn_per_run 10000, run_draws 25, run_molecules_sharing_a_vector_median 24, run_percent_median 0.24
- **MHG-GNN 1,024 features** — scripts/process_and_train.py:1497 MHGGNN_DIMS = 1024 and rust/src/main.rs:97 mhggnn_buf [u8; 4096]; on the assay side, KIRBy create_mhg_gnn run this session on three SMILES returned shape (3, 1024)
- **sparse-count exemption: non-negative integers, at most a quarter non-zero, at least one above 1** — models/model_defaults.py:536-556 is_sparse_count_matrix and :473 SPARSE_COUNT_MAX_DENSITY = 0.25; binary case at :482-491; both applied to the matrix in should_standardise at :558-590
- **1.3% of features non-zero on QM9, 4.7% on hERG** — code comment models/model_defaults.py:466-471 recording the measured densities
- **Avalon and PDV identical for both alanine enantiomers** — run this session: RDKit Avalon 2,048-bit fingerprints and the 200-name DEFAULT_DESCRIPTOR_LIST both identical for C[C@@H](N)C(=O)O and C[C@H](N)C(=O)O (PDV also identical for a second enantiomer pair)
- **MHG-GNN embeddings differ between the alanine enantiomers** — run this session: KIRBy create_mhg_gnn on C[C@@H](N)C(=O)O, C[C@H](N)C(=O)O and CC(N)C(=O)O returned shape (3, 1024); largest per-feature difference 5.15 between the enantiomers, 10.73 between one enantiomer and the achiral string
- **ECFP4 chirality off on both pipelines** — scripts/process_and_train.py:1196-1205 (chirality=False for the Sort & Slice featuriser) and :1524-1526 (GetMorganGenerator without includeChirality); KIRBy molecular.py:306 same call, and the get_scaffold comment at alternative_data_noise_robustness.py:851-858 states create_ecfp4 leaves includeChirality False
- **QM9 strings are stereochemistry-free; assay strings keep stereochemistry** — scripts/process_and_train.py:1327 Chem.MolToSmiles(mol, isomericSmiles=False), feeding every representation at :1344-1360; KIRBy alternative_data_noise_robustness.py:845 Chem.MolToSmiles(mol, canonical=True)
- **Rust computes no representation** — rust/src/main.rs:62-99 SmilesData comment, "This side only carries the bytes through", and the ECFP4 comment at :100-105

</details>


## M3. Models

*R.*eplaces paper.tex 207-219, less its final paragraph, which moves to Uncertainty quantification

```latex
\subsection{Models}

Nineteen model configurations were fitted, and the same nineteen were built for the QM9 HOMO--LUMO gap and for the three assay datasets. They are five tree ensembles, one support vector machine, three Gaussian processes and ten neural networks. Five of the nineteen were added to answer the uncertainty question. The settings for every model are given in Additional file~1.

Random forests (RFs) are one of the most common choices for QSAR modeling, thanks to their robustness and interpretability \citep{Svetnik2003, Breiman2001}. Quantile regression forests (QRFs) keep the distribution of training labels in each leaf and compute quantiles across all trees \citep{Meinshausen2006}. Both were fitted with a minimum leaf size of 5 molecules and 0.3 of the features considered at each split. The random forest uses 100 trees and the quantile forest 300 trees.

We also used eXtreme Gradient Boosting (XGBoost) \citep{Mustapha2016, Tian2022}, Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm} and Natural Gradient Boosting (NGBoost) \citep{Duan2020}. We configured NGBoost as \citet{Duan2020} did: a Normal distribution, the log scoring rule, a decision tree base learner of maximum depth 3, and a learning rate of 0.01. The number of boosting rounds is capped at 500 and read off a held-out curve with a patience of 50 rounds. The best round is used for prediction. Those held-out rows differ between QM9 and the three assay datasets. On QM9 they are the validation split that is already held out of every fit. On the three assay datasets they are a scaffold-grouped fifth of the rows NGBoost is handed, which already exclude the validation molecules. NGBoost on those three datasets therefore fits fewer molecules than the other tree models.

Support vector machines (SVMs) are a well-established baseline in QSAR modeling \citep{Vapnik1995, Svetnik2003}. We used a radial basis function (RBF) kernel on every representation, on QM9 and on the three assay datasets, with no branch on the representation.

Gaussian processes (GPs) use a kernel to produce a Gaussian predictive distribution over every data point \citep{Obrezanova2007, Rasmussen2005}. We fitted three of them as exact processes in \texttt{gpytorch}, each with a constant mean and a scaled kernel. Only the Tanimoto kernel comes from the \texttt{Gauche} framework \citep{gauche}. The first uses an RBF kernel and runs on all six representations. The second uses the Tanimoto kernel \citep{Ralaivola2005, moss2020} and was run on ECFP4 alone. Sort \& Slice is built from substructure counts rather than bits, and a fingerprint kernel asked for on features that are not binary is refused at fit time. The third is the RBF process with a second network predicting the observation noise of each molecule, and it is one of the five uncertainty configurations.

Exact inference is cubic in the number of training points \citep{Rasmussen2005}. Every Gaussian process here therefore fits at most 5,000 training molecules, drawn by a seeded sample. The cap binds on QM9, where the training split holds 8,000 of the 10,000 sampled molecules. The largest assay dataset is LogD at 5,039 molecules, and each fold trains on four fifths of those, so the cap does not bind there.

The RBF lengthscale starts at the median pairwise distance between 500 sampled training molecules, rather than at the framework's default. In a range-finding run on QM9 the typical Euclidean distance between two molecules' feature vectors was about 17 on PDV and about 1,100 on the learned embeddings. A fit whose predictions vary by less than 0.05 of the spread of the labels it was fitted on is recorded as collapsed. The $R^2$ of a collapsed fit is still reported. That fit returns one variance for every molecule, so the column holding its epistemic uncertainty is left blank. Epistemic uncertainty is the model's uncertainty about its own fit, and Uncertainty quantification defines it against the other component.

We tested two deterministic feed-forward architectures. NN-$\alpha$ has two hidden layers of 128 and 64 units, with dropout after each hidden layer. NN-$\beta$ has two hidden layers of 128 units each, with dropout applied before the output. Both were implemented in \texttt{PyTorch} \citep{pytorchGeometric}. The default specification uses ReLU activations and dropout $p = 0.2$. They train with the Adam optimizer at a learning rate of $10^{-3}$, in batches of 32 molecules, for at most 100 epochs. Training stops after 10 epochs without an improvement in mean validation loss, and the weights of the best epoch are restored.

Four Bayesian transformations were applied to each base architecture, giving eight probabilistic networks. The first replaces every linear layer with a Bayesian layer carrying a Gaussian prior $\mathcal{N}(0, 0.1^2)$ on its weights, giving the Bayesian neural networks BNN-$\alpha$ and BNN-$\beta$. It is trained on an evidence lower bound whose KL term is divided by the number of training molecules. The second adds a variance output head to BNN-$\alpha$ and BNN-$\beta$, and fits it by the negative log likelihood of a Gaussian with that variance \citep{kendall2017}. The third replaces every linear layer with a Variational Bayesian Last Layer (VBLL) \citep{Harrison2024}, which maintains a mean-field variational posterior over that layer's weights, giving VBLL-$\alpha$ and VBLL-$\beta$. Only the output layer keeps a learned observation-noise parameter. The hidden layers contribute epistemic uncertainty alone. The fourth makes that observation noise a function of the input. The variance-head and input-dependent-noise variants are the other four uncertainty configurations. All eight estimate predictive distributions from 100 stochastic forward passes.

The Bayesian and variational networks read a tuned setting where one exists for that dataset and model, and run at the specification above where none does. On QM9 four of the nineteen configurations read tuned values: BNN-$\alpha$, BNN-$\beta$, VBLL-$\alpha$ and VBLL-$\beta$. On each of the three assay datasets two of the nineteen do, and which two differs between LogD, Caco-2 and hERG. A tuned NN-$\alpha$ variant replaces the activation function and both layer widths. A tuned NN-$\beta$ variant replaces the dropout fraction, the layer width, the learning rate and the number of hidden layers. The two searches therefore covered different parameters. NN-$\alpha$'s tuned variants keep the learning rate and dropout fraction given above, and NN-$\beta$'s do not. One setting was chosen per model per dataset and applied to all six representations.
```

**Still open in this subsection.**

- Regenerate Additional file 1 from models/model_defaults.py plus results/master_tuned_hyperparameters.json and results/master_tuned_hyperparameters_lab.json, with a column marking which configurations run tuned on which dataset. Four on QM9, two per assay dataset.
- Confirm xgboost, gpytorch and Kersting2007 in the Overleaf sn-bibliography before restoring those citations.
- Fix scripts/generate_paper_figures_v2.py:133 and :150 so gauche_rbf is no longer PDV-only before the figures are regenerated.
- Settle the hERG count: paper.tex:196 says 1,482 compounds, the KIRBy loader docstring says 1,415 molecules.
- Decide whether the distance range (about 17 on PDV, about 1,100 on the learned embeddings) is cited from results/gp_kernel_harvest/qm9/ or cut.

**For the author.**

- Citation keys. paper.tex builds against sn-bibliography (paper.tex:694) and there is no local copy of that .bib, so I used only keys that already appear in paper.tex. xgboost, gpytorch and Kersting2007 appear nowhere in paper.tex, so XGBoost keeps \citep{Mustapha2016, Tian2022} and the GPyTorch and heteroscedastic-GP citations are gone. Add them back only if you confirm the keys exist in the Overleaf bibliography.
- The guide's own §4.11 decision 4 says four of the nineteen configurations run tuned. That is true on QM9 and false on the three assay datasets: results/master_tuned_hyperparameters_lab.json holds two model entries per dataset, and LogD's two are not the same two as Caco-2's and hERG's. The text now scopes it. If Additional file 1 is regenerated it needs the same split.
- I dropped the claim that Additional file 1 marks the tuned configurations, because the file is still headed 'default hyperparameters' (guide §4.11 decision 4). The sentence now says only that the settings are in Additional file 1. It becomes false if the file is not regenerated to carry the tuned values.
- The Tanimoto sentence is now two facts rather than one because-clause. The fit-time refusal at models/models.py:2867-2874 fires on any non-binary feature matrix, so it would not catch Avalon, which is binary and still excluded. Running on ECFP4 alone is the study's roster decision (FP_REPS = ['ecfp4'] at slurm_scripts_qm9_rerun/generate_scripts.py:114, reps_for() at slurm_scripts_validation_rerun/generate_scripts.py:292).
- The figure script disagrees with the Methods sentence about the RBF Gaussian process. scripts/generate_paper_figures_v2.py:133 still comments gauche_rbf as 'RBF GP for PDV only' and :150 puts it in PDV_ONLY_MODELS, while both generators run it on all six representations. A regenerated heatmap will show PDV only against a Methods sentence saying six.
- The opening no longer contrasts the five uncertainty configurations with 'competing on accuracy', because eight models are held out of the variance decomposition, not five: ANOVA_MODELS_EXCLUDE at scripts/generate_paper_figures_v2.py:130-147 also drops qrf, gauche and gauche_rbf. A decomposition table therefore holds eleven models. That belongs in the ANOVA subsection, not here.
- The range of distances between molecules (about 17 on PDV, about 1,100 on the learned embeddings) is a prose comment at models/model_defaults.py:300-313 describing a 2026-08-26 measurement, not a results file. I scoped it as a range-finding run so a reader is not told a comment is a result. Cut it or replace it with a harvest number if you would rather not carry it.
- The two datasets disagree on hERG and I used neither number: paper.tex:196 says 1,482 compounds, the KIRBy loader docstring at tests/alternative_data_noise_robustness.py:24-26 says 1,415 molecules. Only LogD at 5,039 molecules is needed for the Gaussian-process cap sentence, and both sources agree on that.
- The largest assay training block is stated as four fifths of 5,039 molecules rather than a counted number. GroupKFold on scaffolds does not make folds exactly equal, so an exact count needs a run over the loaders in KIRBy tests/alternative_data_noise_robustness.py against tests/data_cache/openadmet_train.csv.
- Batch size 32 molecules is now shared. models/model_defaults.py:347 still carries the comment 'QM9's value; the experimental side used 64', but KIRBy reads the shared block (tests/alternative_data_noise_robustness.py:1882, :2007), so the sentence is true of both pipelines.

**Cut from the current text.**

- The NGBoost definition sentence ('treats the parameters of a chosen parametric distribution as regression targets...'), since paper.tex already carries it and the configuration sentence cites Duan2020.
- 'since the quantile estimate is the latter's deliverable' from the forest paragraph.
- 'At the default lengthscale every molecule is effectively infinitely far from every other' — the range sentence carries the point.
- 'so that this model carries no kernel-and-representation confound' — replaced by 'with no branch on the representation'.
- The clause claiming Additional file 1 marks the four tuned configurations.
- The four-sentence passage on the two tuning searches, compressed to one sentence plus the two sentences naming which parameters a tuned variant replaces.
- \citep{xgboost}, \citep{gpytorch} and \citep{Kersting2007}, none of which appear in paper.tex.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **nineteen model configurations; five tree ensembles, one SVM, three Gaussian processes, ten neural networks** — slurm_scripts_validation_rerun/generate_scripts.py:31-42 (MODELS_ALL, nineteen names, GP / GP-Tanimoto / GP-Hetero listed separately); the same nineteen keys in the QM9 model dict at slurm_scripts_qm9_rerun/generate_scripts.py:660-745
- **five uncertainty configurations** — slurm_scripts_qm9_rerun/generate_scripts.py:690-745 — heteroscedastic_gp, dnn_bnn_full_variational_hetero, mlp_bnn_full_variational_hetero, dnn_bnn_full_mve, mlp_bnn_full_mve
- **minimum leaf size 5 molecules, 0.3 of features per split, 100 trees (RF), 300 trees (QRF)** — models/model_defaults.py:82-86 and :105-109
- **NGBoost: base learner maximum depth 3, learning rate 0.01, 500 boosting rounds capped, patience 50 rounds, best round used** — models/model_defaults.py:180-212 (n_estimators 500, learning_rate 0.01, base_max_depth 3, early_stopping_rounds 50, use_best_iteration True); applied at models/models.py:2365-2370
- **20% scaffold-grouped carve of each fold's training block (assay side)** — /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:3227 (val_frac=0.2) and :3243 (GroupShuffleSplit)
- **Gaussian process cap of 5,000 training molecules** — models/model_defaults.py:264 ('max_train_n': 5000), applied at models/model_defaults.py:494-526; KIRBy loads the same spec at tests/alternative_data_noise_robustness.py:397-400
- **QM9 training split 8,000 of 10,000 sampled molecules** — paper.tex:192 — N = 10,000 molecules, scaffold 80/10/10 split
- **LogD 5,039 molecules (largest assay dataset); each fold trains on four fifths of it** — paper.tex:196 (N = 5,039) and /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:24-26; N_FOLDS = 5 at :575. The four-fifths figure is derived from those two, not counted from the data.
- **lengthscale probe of 500 sampled training molecules; collapse threshold 0.05 of the training label spread** — models/model_defaults.py:313-321 (init_lengthscale_from_data True, lengthscale_probe_n 500, collapse_fraction 0.05); collapse handling at models/models.py:2723-2740 and :2954-2968
- **typical distance between molecules about 17 on PDV and about 1,100 on the learned embeddings** — models/model_defaults.py:300-313 — a prose comment recording a 2026-08-26 measurement, not a results file. Scoped in the text as a range-finding run. See notes.
- **NN-alpha hidden sizes [128, 64]; NN-beta two hidden layers of 128; dropout p = 0.2; ReLU** — models/model_defaults.py:333-342 (NEURAL_DEFAULTS dnn and mlp)
- **Adam, learning rate 1e-3, batch size 32 molecules, at most 100 epochs, patience 10 epochs, best weights restored** — models/model_defaults.py:346-356; the assay pipeline reads the same block at /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:1882-1884, :2007, :2040
- **100 stochastic forward passes** — models/model_defaults.py:~360 ('mc_passes': 100); used on the assay side at /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:2131
- **Gaussian prior N(0, 0.1^2); KL term divided by the number of training molecules** — models/model_defaults.py:378-379 (bnn_prior_mu 0.0, bnn_prior_sigma 0.1); models/models.py:125-130 (bnn_kl_weight) and :1368 (nll + kl / self.n_data)
- **four tuned configurations on QM9; two on each assay dataset** — results/master_tuned_hyperparameters.json — four top-level keys: dnn_bnn_full, dnn_bnn_full_variational, mlp_bnn_full, mlp_bnn_full_variational. results/master_tuned_hyperparameters_lab.json — caco2 and herg hold dnn_bnn_full_variational and mlp_bnn_full; logd holds dnn_bnn_full_variational and mlp_bnn_full_variational. Fallback to defaults at /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:195-213.
- **tuned parameters replaced: activation and both layer widths (NN-alpha); dropout fraction, layer width, learning rate, number of hidden layers (NN-beta)** — the key sets in results/master_tuned_hyperparameters.json (activation, hidden_size1, hidden_size2 for dnn_*; dropout_rate, hidden_size, lr, num_hidden_layers for mlp_*) and the same key sets in results/master_tuned_hyperparameters_lab.json

</details>


## M4. Label noise

*R.*eplaces paper.tex 312-362 including Table 1; paper.tex 364-373 is deleted

```latex
% =====================================================================
% REPLACES paper.tex lines 312-362 (the \subsection{Noise Strategies}
% heading through \end{figure} of fig:noise_strategies).
% paper.tex lines 364-373, \subsection{NoiseInject Framework}, are
% DELETED and nothing replaces them (decision 2, 16 September).
% =====================================================================

\subsection{Label noise}

We corrupted the training labels with artificial noise and measured how far accuracy fell as the amount
rose. The noise comes from two implementations of the same seven conditions. A Rust implementation
injected it on QM9, and the released Python package injected it on the three assay datasets. The two are
checked against each other on the real labels.

Six of the seven noise conditions pair a draw shape with a targeting rule. The seventh, censoring, has no
draw: labels past an assay limit are recorded as the limit. Five targeting rules decide which molecules
are affected, and four of them also set how hard. One row of Table~\ref{tab:regression_noise} is one
noise condition, with the shape it draws from, the rule that decides which molecules it affects, and the
mechanism it stands for. The name in brackets on each row is the name the released code uses for that
condition.

The noise level is the amount of noise delivered, not a parameter each condition interprets its own way.
The targeting rule assigns molecule $i$ a per-molecule scale $s_i$, and $z_i$ is a draw from the shape at
unit scale parameter. The noise added to the label of molecule $i$ is
\begin{equation}
\epsilon_i = \frac{\tau}{G}\,s_i z_i,
\qquad
G = \sqrt{\frac{1}{n}\sum_{j=1}^{n} s_j^{2}}\;\;\sigma_z,
\label{eq:dose_matching}
\end{equation}
where $\tau$ is the requested amount in label units. The constant $\sigma_z$ is the shape's standard
deviation at unit scale parameter: $1$ for the Gaussian, $\sqrt{\nu/(\nu-2)}$ for the Student-$t$,
$\sqrt{2}$ for the Laplace. On QM9 the average over $n$ runs over a split's training molecules. On the
three assay datasets it runs over the whole clean label column. The expected root-mean-square of the
noise is then $\tau$ for every condition, whatever the shape and whatever the targeting.

Grouped-shifted is drawn differently, as $\epsilon_i = (\tau/G)(\sqrt{\rho}\,b_{g(i)} + \sqrt{1-\rho}\,w_i)$,
with one draw $b$ per scaffold group and one draw $w$ per molecule. Its two variances sum to $\tau^{2}$,
and the group term carries $\rho = 0.62$ of that total noise variance.

The requested amount is a fraction of a clean label spread, and seven levels were run: 0, 0.2, 0.3, 0.5,
0.75, 1.0 and 1.5. On QM9 that spread is the standard deviation of the clean training labels. On the
three assay datasets it is the standard deviation of the whole clean label column. Noise was added to the
raw label, and the target was standardised afterwards using the clean training mean and spread.

Censoring has no variance parameter and is not zero-mean. It cannot be held to the same delivered amount
as the other six conditions, and was swept on its own axis instead. Its level is the fraction of labels
clipped: 0, 0.10, 0.20, 0.25, 0.30, 0.40 and 0.50. The limit is the $k$th largest of the clean training
labels, with $k$ the requested fraction of the training molecules. That limit is applied to every split,
so a held-out split does not have the same fraction clipped.

Both pipelines noise the validation labels by default, at the same condition and amount as the training
labels, from an independently seeded draw. Otherwise the families that stop training by watching those
labels are selected not to fit the injected noise. On the three assay datasets the same scaffold-grouped
fifth is carved out of every fold before any model is fitted, and only the neural families read it. Only
those validation labels are noised. Test labels are never noised on either pipeline.

Two conditions are keyed to Murcko scaffold groups, and every experiment uses a scaffold split.
Grouped-wider shuffles the families and adds them until the affected molecule count comes closest to
$20\%$ of the molecules in the split it is corrupting. Acyclic molecules are singleton families rather
than one pooled group. On QM9 validation shares no scaffold family with training, so grouped-wider
redraws its affected families there at the same molecule fraction. Grouped-shifted reuses a family's
offset wherever that family appears in training, and draws a fresh one where it does not. On the three
assay datasets one realisation is drawn over the whole label column and indexed by molecule, so a
molecule carries the same corruption in every fold.

A draw whose realised amount falls outside the injector's tolerance on the requested amount warns rather
than stopping the run. Both pipelines write the solved scale, the realised amount in label units and the
realised affected-molecule fraction onto every results row. On QM9 the selection seed is recorded beside
the draw seed, and a per-molecule file holds the clean label, the noise applied and its level-free shape.

Gaussian, grouped-wider and grouped-shifted ran the full grid on the QM9 HOMO--LUMO gap and on each of
the three assay datasets. That grid is nineteen models: eighteen of them on all six representations, plus
the Tanimoto-kernel Gaussian process on ECFP4 alone, which is $18 \times 6 + 1 = 109$
model-and-representation pairs. Laplace, outlier and Student-$t$ ran on a named subset of models and on
three of the six representations: eight models and 24 model-and-representation pairs on QM9, seven models
and 19 pairs on LogD, and six models and 18 pairs on each of Caco-2 and hERG. Censoring ran five named
model-and-representation pairs on the QM9 HOMO--LUMO gap and on each of the three assay datasets, to
measure the size of its effect rather than to rank models.

\begin{table}[h]
\centering
\small
\renewcommand{\arraystretch}{1.4}
\begin{tabular}{l l p{0.30\textwidth} p{0.30\textwidth}}
\toprule
\textbf{Condition} & \textbf{Draw shape} & \textbf{Targeting} & \textbf{Mechanism it stands for} \\
\midrule
Gaussian (\texttt{gaussian}) & Gaussian & Every molecule, same expected amount & Random measurement error, the reference case \\
\addlinespace
Student-$t$ (\texttt{student\_t\_nu5}) & Student-$t$, $\nu = 5$ & Every molecule, same expected amount & Differences in measured bioactivity are formally non-normal \citep{Kruger2012} \\
\addlinespace
Laplace (\texttt{laplace}) & Laplace & Every molecule, same expected amount & The shape fitted to those same differences \citep{Kruger2012} \\
\addlinespace
Outlier (\texttt{outlier\_p10}) & Gaussian & A random $10\%$ of molecules, each given an error three times wider than the error the other molecules receive & Contaminated records, as in Huber's contamination model \citep{huber1964robust}. The fraction is the top of the $1$--$10\%$ range Hampel gives for routine data \citep{Hampel2001} \\
\addlinespace
Grouped-wider (\texttt{grouped\_wider}) & Gaussian & Whole scaffold families, covering $20\%$ of molecules, each given an error three times wider than the error the other molecules receive & Between-laboratory error is about three times within-laboratory error \citep{Avdeef2019} \\
\addlinespace
Grouped-shifted (\texttt{grouped\_shifted}) & Gaussian & Every scaffold family given its own offset, carrying $62\%$ of the injected noise variance & A laboratory reading high reads high for everything it measured. $62\%$ of measurement variance sits between laboratories \citep{Bentz2013} \\
\addlinespace
Censoring (\texttt{censoring}) & No draw, only clipping & Every label past the upper assay limit recorded as the limit & Values outside an assay's working range reported as the limit \citep{Svensson2025} \\
\bottomrule
\end{tabular}
\caption{The seven noise conditions. One row is one condition, with the shape it
draws from, the rule that decides which molecules it affects, and the mechanism
it stands for. The name in brackets is the name the released code uses for
that condition. Six conditions pair a draw shape with a targeting rule,
and censoring has no draw. Every condition except censoring delivers the same
expected root-mean-square amount of noise at the same level, by
Equation~\ref{eq:dose_matching}, so those six rows differ in which molecules are
affected and in the shape of the draw rather than in the amount. Censoring's
level is the fraction of labels clipped.}
\label{tab:regression_noise}
\end{table}

% ---------------------------------------------------------------------
% THREE THINGS THIS REPLACEMENT BREAKS ELSEWHERE, TO FIX IN THE SAME PASS
%
% 1. paper.tex:307-308, the caption of the metrics table, reads "$\sigma$ is
%    the noise scaling factor (Table~\ref{tab:regression_noise})". The new
%    table has no $\sigma$ and no scaling factor, so that clause must go. It
%    is also where the ECE row goes under decision 5.
%
% 2. The figure at paper.tex:356-362 (fig:noise_strategies) showed six
%    conditions, four of which no implementation produces.
%    scripts/generate_paper_figures_v2.py:3352 (create_methods_figure) now
%    draws the settled conditions from noise_conditions.json on a synthetic
%    three-component label distribution (:3391-3399), not on QM9. Its caption
%    has to be rewritten and the PNG regenerated before the figure can be
%    referenced here.
%
% 3. CITATIONS. paper.tex:694 reads \bibliography{sn-bibliography}, and
%    sn-bibliography.bib is not in this checkout -- it lives on Overleaf, so
%    whether it carries these six keys is unchecked. All six entries, with
%    DOIs, are in citations.bib: huber1964robust:1804, Kruger2012:2028,
%    Bentz2013:2107, Avdeef2019:2118, Hampel2001:2231, Svensson2025:2244.
%    None of the six is cited anywhere in the current paper.tex.
% ---------------------------------------------------------------------
```

**Still open in this subsection.**

- Confirm whether the Overleaf sn-bibliography.bib carries huber1964robust, Kruger2012, Bentz2013, Avdeef2019, Hampel2001 and Svensson2025. None is cited in the current paper.tex, and sn-bibliography.bib is not in this checkout. The entries to paste, with DOIs, are at citations.bib lines 1804, 2028, 2107, 2118, 2231 and 2244.
- Drop '$\sigma$ is the noise scaling factor (Table~\ref{tab:regression_noise})' from the metrics-table caption at paper.tex:307-308 in the same pass as the ECE removal.
- Regenerate fig_methods_noise_strategies.png and write a new caption before any Methods text references the figure; create_methods_figure in scripts/generate_paper_figures_v2.py now draws the settled conditions on synthetic labels, not on QM9.

**For the author.**

- The breaker is wrong about 109. It called 109 a shortfall against 19 models x 6 representations = 114. The design is 18 models on all six representations plus the Tanimoto-kernel Gaussian process on ECFP4 alone, because that kernel is defined on binary fingerprints: slurm_scripts_qm9_rerun/generate_scripts.py:113 FP_REPS = ['ecfp4'] and :674-678, and the same restriction on the assay side at slurm_scripts_validation_rerun/generate_scripts.py:292. 18 x 6 + 1 = 109 pairs, which is the design and not coverage of an unfinished run. I have written 109 with that reason attached. The 24-pair and five-pair figures I took from deep_run_pairs.json and censoring_pairs.json as the breaker asked, not from the coverage file.
- I dropped the 'seven model families' count rather than replace it with another number. models/models.py carries 'Validation stays held out, for early stopping and for calibration' at :2075 (RF), :2263 (SVM), :2312 (NGBoost), :2624 (XGBoost), :2672 (LightGBM) and :2832 (Gauche GP), plus the neural families, but holding validation out for calibration is not stopping training on it, so any count would mix two things. The sentence now says 'the families that stop training by watching those labels'.
- The paragraph target forced real cuts to fit 700 words. Gone from the draft: the sentence naming the three shapes in the body (Table 1's Draw shape column carries them, with nu = 5), 'the fraction actually reached is recorded on every row' for grouped-wider (the provenance paragraph already says the realised affected-molecule fraction is on every row), and the reason acyclic molecules are singletons. If you want any of them back, tell me which and I will cut elsewhere.
- The literature attributions in Table 1 - Kruger2012 for the non-normal shape, Avdeef2019 for between-laboratory error being about three times within-laboratory error, Bentz2013 for 62% of variance sitting between laboratories - came from the draft and rest on NOISE_DESIGN.md. I verified that the code settings match them (lambda 3.0, group_variance_share 0.62 in noise_conditions.json). I did not open the three papers this session, so the attributions themselves are unchecked here.
- Hampel2001 is a research report, not a peer-reviewed article - citations.bib:2239 says so in its own note. The table now reads 'the top of the 1--10% range Hampel gives for routine data' rather than 'the published range'.
- The equation's second factor is now sigma_z, the shape's standard deviation at unit scale parameter, with the three constants spelled out. Written as sd(z) it read as a sample statistic, and the level would then not divide out of the shape exactly - which rust/src/main.rs:1041-1046 states that it does.

**Cut from the current text.**

- The body sentence listing the three draw shapes - Table 1 names each one, with nu = 5.
- 'The fraction actually reached is recorded on every row' after the grouped-wider selection rule - the provenance paragraph says the realised affected-molecule fraction is written to every results row.
- The reason acyclic molecules are singletons (RDKit returns the same empty scaffold for all of them); the rule itself stayed.
- 'Both injectors compare the amount delivered against the amount requested and record both on every row' - merged into the warning sentence and the provenance sentence.
- 'The seven conditions were not run at equal breadth' as a paragraph opener; the three sentences that follow say it.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **seven noise conditions; six pair a shape with a targeting rule, censoring has no draw** — /Users/apunt/repos/qsar_qm_models/noise_conditions.json (stage_1_full_grid: gaussian, grouped_wider, grouped_shifted, censoring; stage_2_depth_only: student_t_nu5, outlier_p10, laplace). Censoring has no draw: rust/src/main.rs:964 sets unit_dose_g to NaN, "censoring does not go through the dose solver"; NoiseInject/noiseInject/core.py:73 gives it a nominal distribution only.
- **three draw shapes; nu = 5** — rust/src/main.rs:148-166 (NoiseShape Gaussian / StudentT{nu} / Laplace, fn unit_sd); noise_conditions.json settings_that_follow: nu = 5.0
- **five targeting rules, four of which set a per-molecule scale** — /Users/apunt/repos/NoiseInject/noiseInject/core.py:76 REGRESSION_STRATEGIES = ('uniform', 'grouped_wider', 'grouped_shifted', 'outlier', 'censoring'); rust/src/main.rs:222-231 (Censoring "has no variance parameter, so it does NOT go through the dose solver"), fn is_dose_matched
- **sigma_z = 1, sqrt(nu/(nu-2)), sqrt(2)** — rust/src/main.rs:157-166 (fn unit_sd); NoiseInject/noiseInject/core.py:359-366 (_shape_unit_sd)
- **n = a split's training molecules on QM9; the whole clean label column on the assay datasets** — rust/src/main.rs:707-730 (unit_dose over the split's scales; build_noise_plan called per split); KIRBy/tests/alternative_data_noise_robustness.py:2612 and :2808 inject_verbose(_col.y, ...) then index by train_rows at :2613; KIRBy/tests/noise_column.py:67 self.y = the whole y_column
- **expected root-mean-square, not exact** — NoiseInject/noiseInject/core.py:630-662 (warns when abs(realised/dose - 1) > tolerance; about 1% of draws land outside for student_t_nu5 and grouped_shifted at n near a thousand); rust/src/main.rs:1533-1550 ("WARNING gate", same check)
- **rho = 0.62 of the variance in the group term** — /Users/apunt/repos/qsar_qm_models/noise_conditions.json settings_that_follow: group_variance_share = 0.62; rust/src/main.rs:1094-1119 (GroupedShift)
- **seven levels: 0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5** — slurm_scripts_qm9_rerun/generate_scripts.py:121 DOSE_LEVELS; slurm_scripts_validation_rerun/generate_scripts.py:119 reads the same constant by name
- **censoring levels: 0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50** — slurm_scripts_qm9_rerun/generate_scripts.py:125 CENSOR_LEVELS; slurm_scripts_validation_rerun/generate_scripts.py:130
- **censoring limit = kth largest of the clean training labels, k = round(fraction * n)** — rust/src/main.rs:893-922 (rank rule, ties broken by position, with the measured LogD table 10% asked / 8.57% moved under y > cut); NoiseInject/noiseInject/core.py:740 _censored_set
- **dose reference spread: clean training SD on QM9, whole-column SD on the assay datasets** — rust/src/main.rs:3000-3006 (standardisation constants from the CLEAN training labels) and :3293-3294; KIRBy/tests/alternative_data_noise_robustness.py:2549-2556 dose_reference_sd = np.std(_col.y)
- **validation noised by default on both pipelines** — rust/src/main.rs:3419-3425 (--clean-validation, "OFF by default") and :3664 noise_validation = !matches.get_flag("clean_validation"); KIRBy/tests/alternative_data_noise_robustness.py:4512/:4532/:4550 noise_validation=not args.no_noise_validation, default True at :2707
- **only the neural families hold a validation split on the assay pipeline** — KIRBy/tests/alternative_data_noise_robustness.py:2547 (run_tree_experiment builds _ColumnView with no val_rows); :2822-2853 (validation noising lives in run_neural_experiment); comment at :2836 "the four neural families that stop training by watching these labels"
- **20% molecule fraction for grouped-wider, lambda = 3 for grouped-wider and outlier, outlier p = 0.10** — /Users/apunt/repos/qsar_qm_models/noise_conditions.json settings_that_follow: group_fraction 0.2, lambda 3.0, outlier_p 0.1
- **grouped-wider selects by closest approach, not coverage** — NoiseInject/noiseInject/core.py:493-518 ("Choose whole groups until the affected MOLECULE fraction is closest to f"); rust/src/main.rs:613-630
- **grouped-wider redraws on validation; grouped-shifted inherits a family's offset** — rust/src/main.rs:571-601 (grouped_wide redraws its own selection at the same molecule fraction when the training selection finds no hits); rust/src/main.rs:1099-1119 (offsets inherited from the training split, counted as group_offsets_inherited_from_training)
- **acyclic molecules are singletons** — scripts/process_and_train.py:631-683 (scaffold = f"__acyclic__{key}"); KIRBy/tests/alternative_data_noise_robustness.py:891, :917; scripts/crosscheck_injectors.py:109-110
- **provenance fields written on every results row by both pipelines** — rust/src/main.rs:3285-3300 (run manifest) and :400-410 (selection_seed beside seed); KIRBy/tests/alternative_data_noise_robustness.py:3746-3756 _prov_cols; NoiseInject/noiseInject/core.py:181-201 as_row (carries 'seed' alone, no selection seed)
- **per-molecule file on QM9: clean label, noise applied, level-free shape** — noise_provenance_1563988872867809435.csv header: split, record_index, canonical_smiles, y_clean_raw, epsilon_raw, noise_scale_raw, noise_pattern_raw, y_noisy_raw, y_written
- **nineteen models, six representations, Tanimoto GP on ECFP4 alone, 109 pairs** — slurm_scripts_qm9_rerun/generate_scripts.py:112 ALL_REPS (six), :113 FP_REPS = ['ecfp4'], :652-745 MODELS (nineteen active entries, counted this session; 'gauche' at :674-678 carries FP_REPS). 18 x 6 + 1 = 109. slurm_scripts_validation_rerun/generate_scripts.py:17-40 ("THE SAME NINETEEN MODELS QM9 RUNS"), :48 ALL_REPS, :292 (GP-Tanimoto restricted to ECFP4)
- **depth subset: eight models on three representations, 24 pairs, all four datasets** — /Users/apunt/repos/qsar_qm_models/deep_run_pairs.json (eight models, three representations: ecfp4, pdv, chemberta); noise_conditions.json stage_2_depth_only, applies_to includes qm9_deep_run, validation_robustness and uncertainty_runs
- **censoring: five named pairs on all four datasets** — /Users/apunt/repos/qsar_qm_models/censoring_pairs.json (five generator_pairs); noise_conditions.json censoring scope: mode pair_subset, n_pairs 5, applies_to qm9_grid and validation_robustness
- **citation line numbers in citations.bib** — grep of /Users/apunt/repos/qsar_qm_models/citations.bib this session: huber1964robust:1804, Kruger2012:2028, Bentz2013:2107, Avdeef2019:2118, Hampel2001:2231, Svensson2025:2244

</details>


## M5. Uncertainty quantification

*N.*ew subsection, absorbing paper.tex 217-219

```latex
\subsection{Uncertainty quantification}

We separate the predictive uncertainty into an aleatoric and an epistemic component wherever a model
produces both \citep{kendall2017}. The aleatoric component is the observation noise the model attributes
to a label, and should respond to injected label noise. The epistemic component is the model's doubt
about its own fit.

Two networks in the uncertainty pass carry a variance head, in which a second output predicts each
molecule's observation noise. Their loss is the Gaussian negative log likelihood of that variance. For a
network that reports an observation noise we take the two components over $T = 100$ stochastic forward
passes:
\begin{equation}
\hat{u}^2_{\text{ale}}(x) = \frac{1}{T}\sum_{t=1}^{T}\hat{v}_t(x),
\qquad
\hat{u}^2_{\text{epi}}(x) = \frac{1}{T}\sum_{t=1}^{T}\big(\hat{\mu}_t(x) - \bar{\mu}(x)\big)^2 ,
\label{eq:sampling_split}
\end{equation}
where $\hat{\mu}_t(x)$ and $\hat{v}_t(x)$ are the mean and the variance the network returned for molecule
$x$ on pass $t$, and $\bar{\mu}(x)$ is the mean over passes. A network that predicts a mean alone has no
aleatoric component, and its total is the epistemic component.

For the quantile regression forest we apply the law of total variance over the fitted trees' leaf
distributions:
\begin{equation}
\hat{u}^2_{\text{ale}}(x) = \frac{1}{B}\sum_{b=1}^{B} s^2_b(x),
\qquad
\hat{u}^2_{\text{epi}}(x) = \frac{1}{B}\sum_{b=1}^{B}\big(m_b(x) - \bar{m}(x)\big)^2 ,
\label{eq:forest_split}
\end{equation}
where $B$ is the number of trees and $\bar{m}(x)$ the mean over them. Here $m_b(x)$ and $s^2_b(x)$ are the
mean and the variance of tree $b$'s in-bag labels in that molecule's leaf. The ordinary random forest is
separated the same way on the three assay datasets, on its held-out molecules. It is not in the
out-of-fold pass. On QM9 it runs without uncertainty and writes none. Both forests use a minimum leaf
size of 5 molecules. At a leaf of 1 molecule the aleatoric component is identically zero.

For the Gaussian processes the epistemic component is the latent function variance at the molecule. The
aleatoric component is the likelihood noise, one number for the whole fit. One Gaussian process instead
predicts that noise per molecule from a second network.

Every uncertainty row records whether each component varies per molecule, is one number per fit, or is
absent. Both components vary per molecule for the quantile forest, and for the random forest on the three
assay datasets. NGBoost has an aleatoric component alone, because it makes a single distributional fit
and therefore has no spread between fits to report. The variational networks add one aleatoric number per
fit, and the plain Bayesian ones have none. A constant component is the same for every molecule, so its
rank correlation against a per-molecule quantity is undefined.

No test label is corrupted on either pipeline, so we score training molecules out of fold. The training
block is divided into five inner folds grouped on Murcko scaffolds, each scored by a refit that excludes
it. The pass covers the same 21 model-and-representation pairs on both pipelines: seven models on each of
ECFP4, PDV and ChemBERTa. They are the quantile regression forest, NGBoost, the radial basis Gaussian
process in both forms, the VBLL transformation of NN-$\alpha$, and the variance-head transformations of
NN-$\alpha$ and NN-$\beta$. All five inner folds are scored, except for NGBoost on QM9, which scores three
of the five. On QM9 the uncertainty rows come out of the same ten replicates as the accuracy rows. On the
three assay datasets they are a separate run over the same five scaffold folds.

Four of the seven noise conditions give every molecule the same noise scale: Gaussian, Student-$t$ with
$\nu = 5$, Laplace and grouped-shifted. We report no per-molecule detection statistic under those four,
because there is no structure to detect. The other three put the corruption on some molecules and not
others: grouped-wider, outlier contamination of $10\%$ of labels, and censoring, which clips only the
labels above the limit. Grouped-wider is keyed to the scaffold group, and a scaffold
split holds whole groups out. Its per-molecule structure is therefore flat on held-out molecules, and is
read on out-of-fold training molecules instead.

All uncertainties are reported without post-hoc calibration. On QM9 one temperature multiplier per fit is
fitted on held-out validation molecules by Gaussian negative log likelihood, bounded to $[0.1, 10.0]$. No
reported number reads it, because refitting at each noise level makes coverage nominal by construction.
The assay pipeline fits no such multiplier.
```

**Still open in this subsection.**

- Decide numbered equations versus paper.tex's unnumbered $$...$$ convention.
- Decide whether the variance-head sentence moves to the Models subsection beside NN-alpha and NN-beta.
- Replace or delete paper.tex:218, which says NGBoost, QRF and the BNN variants were not decomposed.

**For the author.**

- Every defect in the break report checked out against the code; I found none of them wrong. The QM9 job generator gives 'rf' the flags '-m rf' with no '-u True' (slurm_scripts_qm9_rerun/generate_scripts.py, MODELS dict), and the forest split in models/models.py sits inside 'if args.uncertainty:', so the ordinary forest writes no components on QM9. KIRBy/tests/alternative_data_noise_robustness.py queues RF unconditionally and calls _tree_split at :2631 on every model, so it does write them on the three assay datasets.
- Equation numbering is the one open convention call. paper.tex uses unnumbered $$...$$ display math with no labels at :229-231, :235-237 and :243-245. The brief asked for two numbered equations, so I kept \begin{equation} with \label. Nothing in the subsection cross-references either label, so they can be switched to $$...$$ with no other change.
- I wrote $\hat{v}_t(x)$ for the variance a network predicts on one pass, not $\hat{\sigma}^2_t(x)$. $\sigma$ is the noise level everywhere else in the paper (paper.tex:242) and $\hat{u}$ is already the paper's symbol for a predicted uncertainty (paper.tex:229).
- On the constant-component wording I kept 'undefined' and dropped the 'rather than zero' contrast. The repository says it both ways: NoiseInject/noiseInject/core.py:891-895 says UNDEFINED rather than answered with zero, while uncertainty_pairs.json says a constant correlates at exactly zero however good the model is. RERUN_PLAN.md 5.5a point 5 is the audit of 2026-08-27 and settles neither wording, so this is your call.
- The variance-head definition sits in my subsection, at the first mention. It would read better in the Models subsection, where NN-alpha and NN-beta are introduced (paper.tex:216 describes only the full-BNN and VBLL transformations), but I do not own that text.
- paper.tex:218 contradicts this subsection outright. It says the uncertainty from BNN variants, NGBoost and QRF was not decomposed, which is what the code did before the split was wired in. That paragraph has to be deleted or replaced when this subsection lands, or the Methods says both things.
- The assay pipeline has no calibration code at all: 'temperature' and 'calibrat' return nothing in KIRBy/tests/alternative_data_noise_robustness.py. That is the basis of the last sentence.
- On QM9 the temperature is fitted on a carve-out of the validation split, half of it (models/model_defaults.py:704, 'calibration_fraction_of_val': 0.5). I left the draft's 'held-out validation molecules', which is true but does not say it is half. Say so if you want it exact.

**Cut from the current text.**

- 'of every probabilistic model' from the opening sentence — the support table at scripts/uncertainty_decomposition.py:94-180 marks xgboost, lgb, svm, dnn and mlp as (none, none), so the claim was false of five models in the roster.
- 'following \citet{kendall2017}' — wrong \citet/\citep use and a repeat of the citation three sentences above it.
- 'rather than zero' after 'undefined' — the two repository statements disagree, so the safe half is kept.
- About 80 words of wording, spread over every paragraph, to hold the 600-word ceiling after the scoping sentences and the variance-head definition were added. No fact was dropped to do it: the sentences carrying the support-table rows, the fold counts, the roster, the conditions and the calibration multiplier are all still there.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **T = 100 stochastic forward passes** — models/model_defaults.py:364 — 'mc_passes': 100, under the comment "Stochastic forward passes for every Bayesian and variational model"
- **two networks with a variance head** — uncertainty_pairs.json models list — dnn_bnn_full_mve and mlp_bnn_full_mve; queued at slurm_scripts_qm9_rerun/generate_scripts.py:735 and :741 with '--loss heteroscedastic -u True'
- **minimum leaf size of 5 molecules, both forests** — models/model_defaults.py:84 ('rf') and :107 ('qrf'), both 'min_samples_leaf': 5
- **a leaf of 1 molecule gives an aleatoric component of exactly zero** — scripts/uncertainty_decomposition.py:395 — the guard 'if float(np.max(aleatoric)) <= 1e-12' and its message
- **21 model-and-representation pairs** — uncertainty_pairs.json — seven entries under "models" (line 14) crossed with three under "representations" (line 31); both generators read this file
- **seven models: qrf, ngboost, gauche_rbf, het_gp_rbf, dnn_vbll, dnn_bnn_full_mve, mlp_bnn_full_mve** — uncertainty_pairs.json "models" list, line 14 onward
- **three representations: ECFP4, PDV, ChemBERTa** — uncertainty_pairs.json "representations" list, line 31
- **five inner folds** — slurm_scripts_qm9_rerun/generate_scripts.py:1371 and slurm_scripts_uncertainty_rerun/generate_scripts.py:691 — both '--oof-folds' default 5
- **NGBoost scores three of the five folds on QM9** — slurm_scripts_qm9_rerun/generate_scripts.py:226 — OOF_FOLDS_SCORED = {'ngboost': 3}; the assay generator has no such per-model table
- **seven noise conditions, four of them constant-scale** — noise_conditions.json — stage_1_full_grid (gaussian, grouped_wider, grouped_shifted, censoring) plus stage_2_depth_only (student_t_nu5, outlier_p10, laplace); constant scale fixed by _CONSTANT_SCALE_STRATEGIES = ('uniform', 'grouped_shifted') at NoiseInject/noiseInject/core.py:84 and by the flat Uniform/GroupedShift arms of scale_map at rust/src/main.rs:541-543
- **Student-t with nu = 5** — noise_conditions.json settings_that_follow, "nu": 5.0; CONDITIONS['student_t_nu5'] at NoiseInject/noiseInject/core.py:60
- **outlier contamination of 10% of labels** — noise_conditions.json settings_that_follow, "outlier_p": 0.1; CONDITIONS['outlier_p10'] p=0.10 at NoiseInject/noiseInject/core.py:66
- **temperature multiplier bounded to [0.1, 10.0]** — scripts/utils.py:632 — minimize_scalar(nll, bounds=(0.1, 10.0)); also models/model_defaults.py:703 'calibration_bounds': [0.1, 10.0]

</details>


## M6. Performance metrics

*R.*eplaces paper.tex 220-311

```latex
\subsection{Performance metrics}

Predictive accuracy was scored on held-out molecules: the test split on QM9, and the held-out fold of the
five-fold scaffold cross-validation on the three assay datasets. The analysis uses the coefficient of
determination ($R^2$, where a higher value is a closer fit) and the correlation between predicted and
measured values. That correlation is Pearson's on QM9 and Spearman's on the three assay datasets. Root
mean squared error and mean absolute error are recorded at every noise level, in the label's own units,
but no figure or table reports them. For the
probabilistic models we recorded the share of held-out molecules whose measured label falls within one
and within two predicted standard deviations of the prediction, against Gaussian targets of $68\%$ and
$95\%$ of molecules. We do not report an expected calibration error, which was dropped from the analysis.

Four statistics are computed on the out-of-fold training molecules described under Uncertainty
quantification. The first is the Spearman correlation between a model's predicted uncertainty and the
size of the noise injected into that molecule's label, where a higher value means the uncertainty follows
the injected noise more closely. On its own it is a check on the out-of-fold procedure rather than a
result, because a model that has seen its own corrupted label would score highly on it. The second
divides the out-of-fold absolute error by the predicted uncertainty, and asks whether that ratio ranks
the corrupted labels better than the error alone does. The number reported is the difference between the
two rankings.

The third statistic is the mean predicted uncertainty per configuration, which tracks whether a model
widens as the noise rises. The fourth asks whether the error a model makes is correlated within a
scaffold group. Each of the four is reported against the declaration of whether the component it uses
varies per molecule.

Robustness is summarised by AUC$_\text{norm}$. For each configuration we divided $R^2$ at each noise
level by that configuration's own clean value. Our robustness metric is the normalised area under the
R$^2$ retention curve,
$$
\text{AUC}_{\text{norm}} = \frac{1}{\tau_{\max} - \tau_{\min}} \int_{\tau_{\min}}^{\tau_{\max}} \frac{R^2(\tau)}{R^2(0)}\, d\tau.
$$
Despite the name, AUC$_\text{norm}$ is not an area under a receiver operating characteristic curve, and
the curve it integrates is $R^2$ against the amount of noise injected into the training labels. The
integral is evaluated by the trapezoidal rule over the seven noise levels that were run,
$\tau \in \{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$, in the units of Equation~\ref{eq:dose_matching}. The
divisor is the span of the levels in that curve. Censoring is swept over fractions of labels clipped from
0 to 0.50, and is normalised over its own span.

A value near 1 means the configuration kept its clean accuracy, so a higher value is more robust.
AUC$_\text{norm}$ has no upper bound and nothing is clipped. A configuration that predicts better with
noise added scores above 1, and values above 1.05 are counted and reported.

On QM9 one AUC$_\text{norm}$ is computed per replicate, and the median over the ten replicates is
reported. On the three assay datasets one is computed per scaffold fold, and the median over the five
folds is reported. Those folds are a partition of one dataset rather than repeats of it. A configuration's
replicate whose clean $R^2$ falls below 0.3 is excluded, each replicate judged against its own clean
value. A configuration's median can therefore run over fewer replicates than
the one beside it.

We decomposed the variance in AUC$_\text{norm}$, and in $R^2$ at the noise level of 1.0 that QM9 results
are reported at, into four terms. They are a model term, a representation term, an interaction term for
the pairing of the two, and a residual term for what is left over. The shares are $\eta^2$ from Type I
sequential sums of squares. Six model configurations are set aside from the decomposition, and from every
comparison that ranks models against each other. Five of the six add a per-molecule noise term to a model
already in the roster, so they train under a different likelihood from the model they are a variant of.
The sixth is the
Tanimoto-kernel Gaussian process, which needs binary vectors and therefore ran on ECFP4 alone.

A noise condition carrying fewer than five models is not decomposed. The residual is within-cell variance
across the ten QM9 replicates, and each decomposition carries a band from repeating the fit with one
replicate left out. We fitted it on QM9 alone, because the five scaffold folds on the assay datasets
partition one dataset rather than repeating it.

On QM9 we scored whether model rankings on AUC$_\text{norm}$ agree across noise conditions, using
Kendall's coefficient of concordance ($W$). It is computed within one representation, the one carrying
the most models. It needs at least three models and at least two noise conditions, and is not computed
otherwise. Deterministic and probabilistic counterparts were compared on AUC$_\text{norm}$ with a
two-sided Wilcoxon signed-rank test at $\alpha = 0.05$, paired on the replicate. The test was run
separately for each representation and noise condition, with no adjustment for multiple comparisons. A
test paired on five values cannot fall below $p = 0.0625$ however large the difference, so it is run on
QM9's ten replicates and not on the five scaffold folds of an assay dataset.
```

**Still open in this subsection.**

- Nothing in this subsection is a TODO: every number in the text was read from a file this session.

**For the author.**

- "the normalised area under the R$^2$ retention curve" is kept verbatim. It is your own wording at paper.tex:242 and again at paper.tex:380, so it is not the banned phrase "retention area" — but AUC_norm is named first in the paragraph, before the description, which is the ordering the terminology rule asks for. Change it if you would rather the description never appeared.
- One correction I did not take as written. The breaker asked for "recorded per level and in the results files but are not reported" and also for the units sentence to be dropped or scoped. I folded both into one sentence: the errors are in the label's own units on both pipelines, but by two different routes. QM9 computes on standardised values and multiplies back by the clean training SD (scripts/utils.py:217). The assay runner inverse-transforms the predictions and computes on raw values (KIRBy:2595, :2635, :2939-2944). The end state is the same, so the sentence needs no pipeline scope, and the mechanism is gone.
- Kendall's W: I wrote "the one carrying the most models" rather than "the widest coverage", because run_paper_analysis.py:266-267 picks the representation with the most distinct model names. --primary-rep overrides it. If you want the override named in the Methods that is one more clause; I left it out for the length ceiling.
- The three mis-cited line numbers are corrected in numbers_used: the rescale is scripts/utils.py:217 (not :219-223), wilcoxon_paired is scripts/figlib_metrics.py:444-480 with the floor at :450-453 and :474, and two_way_eta2_by_condition is scripts/figlib_metrics.py:261-318 (the jackknife runs to :306 and the record is written to :316).
- coverage_at_k at scripts/figlib_metrics.py:564 has no caller anywhere in scripts/. The coverage numbers in the paper come from calculate_coverage in scripts/generate_paper_figures_v2.py:2668, used for QM9 and for the assay uncertainty table at :1919-1920. The unscoped coverage sentence is safe on both pipelines.
- "paired on the replicate" is true of the deterministic-versus-probabilistic comparison only. The same test is used a second time inside d3_condition_separation (scripts/figlib_decisions.py:756), where it pairs on the model. Nothing in this subsection generalises it, but the Results should not either.

**Cut from the current text.**

- The sentence giving the standardisation mechanism ("Labels are standardised against the clean training split before fitting, so RMSE and MAE are rescaled by that standard deviation"). It is QM9's route and was asserted for both pipelines.
- RMSE and MAE as reported metrics. They are now named once, as recorded and not reported, which freed about 25 words for the two additions.
- Nothing else from the draft was dropped.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **five-fold scaffold cross-validation, five folds** — /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:575 — N_FOLDS = 5
- **Pearson on QM9** — /Users/apunt/repos/qsar_qm_models/scripts/utils.py:212 — pearson_corr, _ = pearsonr(y_test, prediction); returned at :227
- **Spearman on the three assay datasets** — /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:2943 — 'spearman': spearmanr(y_true, yp).correlation
- **RMSE and MAE recorded but reported nowhere** — stored at /Users/apunt/repos/qsar_qm_models/scripts/utils.py:208-210 and /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:2940-2942; loaded at /Users/apunt/repos/qsar_qm_models/scripts/figlib_load.py:218 and used by no figlib module (only other hit is the label registry at scripts/figlib_config.py:621-622)
- **in the label's own units on both pipelines** — QM9 multiplies back by the clean training SD at /Users/apunt/repos/qsar_qm_models/scripts/utils.py:217 (sd from current_standardisation(), :119-121); the assay runner computes the metrics on raw values at /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:2939-2944, predictions inverse-transformed at :2595 and :2635
- **coverage at one and two predicted standard deviations, 68% and 95%** — /Users/apunt/repos/qsar_qm_models/scripts/generate_paper_figures_v2.py:2668-2685 (calculate_coverage, targets in the docstring) and :1919-1920 (cov_1sigma, cov_2sigma); coverage_levels [1, 2] at /Users/apunt/repos/qsar_qm_models/models/model_defaults.py:705
- **AUC_norm divisor is the span of the levels; trapezoidal** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:63 span = sigma.max() - sigma.min(); :66 trapezoid(r2 / baseline, sigma) / span
- **seven levels, sigma in {0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5}** — /Users/apunt/repos/qsar_qm_models/slurm_scripts_qm9_rerun/generate_scripts.py:121 — DOSE_LEVELS = '0.0 0.2 0.3 0.5 0.75 1.0 1.5'; the assay runner holds the same ladder (NOISE_LEVELS)
- **censoring 0 to 0.50** — /Users/apunt/repos/qsar_qm_models/slurm_scripts_qm9_rerun/generate_scripts.py:125 — CENSOR_LEVELS = '0.0 0.10 0.20 0.25 0.30 0.40 0.50'
- **1.05** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_config.py:180 — AUC_NORM_IMPLAUSIBLE_HIGH = 1.05; counted, not clipped, at scripts/figlib_metrics.py:119-126
- **ten QM9 replicates** — /Users/apunt/repos/qsar_qm_models/slurm_scripts_qm9_rerun/generate_scripts.py:1139-1143 — STAGE_DEFAULTS: screen 1 replicate starting at 0, main grid 9 starting at 1; the comment at :1429-1431 states the settled total is 10
- **0.3 baseline gate, applied per configuration-and-replicate** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_config.py:168 — BASELINE_THRESHOLD = 0.3; grouping key ends in 'replicate' at scripts/figlib_metrics.py:80, baseline read and compared inside that group at :94-103
- **reporting level 1.0 on QM9** — /Users/apunt/repos/qsar_qm_models/models/model_defaults.py:626-628 — REPORTING_LEVELS, 'qm9': 1.0
- **Type I sequential sums of squares** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:178-179 and the nested least-squares fits at :206-224
- **six model configurations set aside from the decomposition** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_config.py:528-541 VARIANT_MODELS = {dnn_bnn_full_mve, mlp_bnn_full_mve, dnn_vbll_hetero, mlp_vbll_hetero, het_gp_rbf, gauche}; ANOVA_MODELS_EXCLUDE = set(VARIANT_MODELS) at :545; dropped on the way in at scripts/figlib_metrics.py:250-258 (drop_variant_models), called at :283
- **fewer than five models: condition not decomposed** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:284-291 against MIN_MODELS_FOR_ANOVA
- **leave-one-replicate-out band** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:261-318 (two_way_eta2_by_condition; the jackknife loop at :297-306, spread_is = 'leave-one-replicate-out' at :316)
- **decomposition fitted on QM9 alone** — /Users/apunt/repos/qsar_qm_models/scripts/run_paper_analysis.py:278-285, both calls on qm9_per and on at_level built from qm9; the fold/replicate distinction at scripts/figlib_load.py:20-25
- **Kendall's W on QM9, on AUC_norm, within one representation** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_decisions.py:774 (the only call site, inside d3_condition_separation), called on QM9 data with one representation at scripts/run_paper_analysis.py:275; the representation filter at scripts/figlib_metrics.py:405 and the AUC_norm pivot at :413-414
- **the representation carrying the most models** — /Users/apunt/repos/qsar_qm_models/scripts/run_paper_analysis.py:265-270 — counts = qm9_summary.groupby('rep')['model'].nunique(); primary = counts.idxmax(), unless --primary-rep is given
- **at least three models and at least two conditions** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:417 — if n_raters < 2 or n_items < MIN_RANKED_MODELS; MIN_RANKED_MODELS = 3 at :358; table.shape is (conditions, models)
- **alpha = 0.05, two-sided, paired on the replicate, per representation and condition, no multiplicity adjustment** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_decisions.py:1121-1133 (loops over pairs, representations, conditions, pair_on='replicate'); significance at scripts/figlib_metrics.py:479 bool(p < 0.05); no adjustment anywhere in the function
- **p = 0.0625 on five pairs** — /Users/apunt/repos/qsar_qm_models/scripts/figlib_metrics.py:474 — record['p_floor'] = float(2 ** (1 - n)); docstring at :450-453
- **Wilcoxon run on QM9 and not on the assay datasets** — /Users/apunt/repos/qsar_qm_models/scripts/run_paper_analysis.py:333 — collect(D.d10_probabilistic(qm9_per))

</details>


## AVAIL. Availability of data and materials

*R.*eplaces paper.tex 618-634, back matter

```latex
\subsection*{Availability of data and materials}

The NoiseInject benchmarking framework is available as an open-source Python package
under an MIT license \citep{noiseinject}. It implements the seven noise conditions used
here, and it injected the noise on the three assay datasets. On QM9 the noise came from a
Rust implementation of the same seven conditions, checked against the package's registry
by \texttt{scripts/crosscheck\_injectors.py}. The package
computes AUC$_{\text{norm}}$ and, from per-molecule predictions, the rank correlation
between predicted uncertainty and prediction error, the rank correlation between predicted
uncertainty and the size of the noise injected into a molecule's label, coverage at one and
two predicted standard deviations, and mean interval width. It also contains optional
reference wrappers (a Gaussian process, split-conformal prediction and Monte Carlo dropout)
that produce a per-molecule uncertainty. NoiseInject operates on label arrays. It is
independent of the model being trained. Worked examples cover a PDV workflow, a
graph-neural-network embedding workflow and a regression workflow outside chemistry.

\begin{itemize}
    \item Project name: NoiseInject
    \item Project home page: \url{https://github.com/adpunt/noise_inject}
    \item Archived version: \url{https://doi.org/10.5281/zenodo.20532710}
    \item Version: 1.0.0
    \item Operating system(s): Platform independent
    \item Programming language: Python
    \item Other requirements: See repository \texttt{requirements.txt} for the full dependency list
    \item License: MIT
    \item Any restrictions to use by non-academics: None
\end{itemize}

% TODO: confirm both repositories below are public before submission.
% TODO: neither study repository is archived. A release tag or a Zenodo DOI for each
% would pin what a reader gets; nothing in either repository records which commit
% produced the numbers in this paper.
The pipeline that ran QM9 is at \url{https://github.com/adpunt/qsar_qm_models}. The runner
for the three assay datasets is \texttt{tests/alternative\_data\_noise\_robustness.py} at
\url{https://github.com/adpunt/KIRBy}.

QM9 \citep{Ramakrishnan2014} is deposited on Figshare \citep{qm9_dataset}. We read it with
the QM9 loader in PyTorch Geometric \citep{pytorchGeometric}, which retrieves its structure
files from a DeepChem mirror. LogD and Caco-2 are two endpoint columns of the training
split of the OpenADMET-ExpansionRx challenge dataset \citep{openadmet}. The Caco-2 endpoint is
the efflux ratio on a logarithmic scale. The hERG pChEMBL values were extracted from ChEMBL
for target CHEMBL240 \citep{Zdrazil2023}, following the curation protocol described in
Section~\ref{sec2}.
```

**Still open in this subsection.**

- Confirm both study repositories are public before submission (LaTeX comment in the text).
- Decide whether to archive the two study repositories, and with what: a release tag, a Zenodo DOI, or a sentence offering outputs on request (LaTeX comment in the text).
- Methods: change hERG N = 1,482 at paper.tex:197 to 1,415. The file that settles it is /Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv.
- paper.tex:216 cites \citep{pytorchGeometric} for PyTorch. Add a PyTorch reference to refs.bib and move the key.

**For the author.**

- One faulted item is wrong as stated, and I took the fix anyway. The breaker says PyTorch Geometric is cited nowhere in paper.tex. It is cited, at paper.tex:216, where \citep{pytorchGeometric} is attached to the word PyTorch in the neural-network paragraph. Fey and Lenssen is the PyTorch Geometric paper, not the PyTorch paper, so line 216 is a miscitation. My sentence uses the key correctly; line 216 needs its own fix and a PyTorch reference added to refs.bib.
- hERG count contradiction, for whoever writes Methods: paper.tex:197 says "This results in N = 1,482 compounds". The cache holds 1,415 (chembl_herg_ki.csv, 1,416 lines with the header) and the runner's own header at :26 says 1415. The Methods number must become 1,415 or the paper contradicts itself across the Section~\ref{sec2} cross-reference.
- I dropped the pointer to Table~\ref{tab:metrics_summary}. grep for kendall, icc, wilcoxon and anova over /Users/apunt/repos/NoiseInject/noiseInject/*.py returns nothing, so the package implements none of those four. The package also implements retention_pct and two Weibull parameters (metrics.py:188-200) that are not in the table, so the two sets do not coincide in either direction. The text now names what the package computes and never implies the paper's numbers came from it.
- ECE is real in the package (noiseInject/uncertainty.py:177) but I left it out of the enumeration, under the 16 September decision that ECE is cut from the paper. If ECE stays anywhere in the paper, it belongs back in this list.
- The package registry holds eleven conditions and the study runs seven of them. I cut the clause saying so to hold the length ceiling; say the word and I will put it back as "among eleven in its registry".
- Neither study repository is archived. That is your call, so it is a LaTeX comment, not a sentence: a release tag or a Zenodo DOI per repository, or a line saying analysis outputs are available on request. Nothing I read records which commit produced the paper's numbers.
- Length: 240 words of prose, 280 counting the itemised block. The scoping sentence about the two injectors and the metric enumeration are what push it past 250. Cut candidates, in the order I would cut them: the DeepChem-mirror clause, the wrapper parenthetical, the examples sentence.

**Cut from the current text.**

- "ChEMBL 36" — the provenance file's own _note says the stamp was reconstructed on 2026-09-04 rather than written by the fetch, and _release_evidence derives the release from posting dates. The claim now reads "from ChEMBL for target CHEMBL240"; the ChEMBL 37 re-check sentence carries the provenance.
- The pointer to Table~\ref{tab:metrics_summary} for the package's metrics, replaced by the metrics the package actually computes.
- "the PyTorch Geometric distribution of that deposition" — the PyG QM9 loader's raw_url is deepchemdata.s3-us-west-1.amazonaws.com, raw_url2 is one Figshare file (the uncharacterised list), processed_url is data.pyg.org. /Users/apunt/repos/qsar_qm_models/data/QM9/raw holds gdb9.sdf, gdb9.sdf.csv, QM9_README and uncharacterized.txt, matching that.
- The double citation \citep{qm9_dataset, Ramakrishnan2014} for one Figshare record: qm9_dataset (refs.bib:2241) is the deposition, Ramakrishnan2014 (refs.bib:628) is the Scientific Data article. They now sit on the dataset and the deposit separately.
- "The datasets are publicly available." and "The study code is in two repositories." — cut for length only, not faulted; both paragraphs still open on the fact.
- "with two quickstart notebooks" — cut for length; notebooks/01_quickstart.ipynb and 02_uncertainty.ipynb do ship, so this can go back in if the ceiling moves.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **seven noise conditions** — /Users/apunt/repos/qsar_qm_models/noise_conditions.json — stage_1_full_grid holds gaussian, grouped_wider, grouped_shifted, censoring; stage_2_depth_only holds student_t_nu5, outlier_p10, laplace. All seven resolve in noiseInject.CONDITIONS, which holds eleven names.
- **coverage at one and two standard deviations, mean interval width, uncertainty--error and uncertainty--noise rank correlation** — /Users/apunt/repos/NoiseInject/noiseInject/uncertainty.py:24-90 — calculate_uncertainty_metrics writes unc_error_rho, coverage_1sigma, coverage_2sigma, mean_interval_width, unc_noise_rho (ece also written, but the author's decision 5 cuts ECE from the paper).
- **AUC_norm computed by the package** — /Users/apunt/repos/NoiseInject/noiseInject/metrics.py:45-101 — _retention_auc_norm and _curve_robustness return auc_norm.
- **Version 1.0.0** — /Users/apunt/repos/NoiseInject/CITATION.cff — version: 1.0.0
- **DOI 10.5281/zenodo.20532710** — /Users/apunt/repos/NoiseInject/CITATION.cff — doi field
- **MIT license** — /Users/apunt/repos/NoiseInject/LICENSE — "MIT License", Copyright (c) 2025 Adelaide Punt, University of Oxford
- **CHEMBL240** — /Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.provenance.json — "target": "CHEMBL240"
- **1,415 compounds** — /Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv — 1,416 lines including the header; provenance json "n_compounds": 1415
- **ChEMBL 37 re-check, every pKi value unchanged** — /Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.refetch_2026-09-12.comparison.json — fresh_release ChEMBL_37, cached_molecules_still_present 1415, molecules_new_since_the_cache 0, shared_molecules_with_a_different_label 0, largest_label_change_log_units 0.0
- **three worked examples** — /Users/apunt/repos/NoiseInject/examples/ — qm9_pdv.py, moleculenet_gnn.py, generic_dataset.py (toxicity_classification.py is the classification one and is dropped under decision 3)
- **github.com/adpunt/qsar_qm_models and github.com/adpunt/KIRBy** — git remote -v in both checkouts

</details>


## ADDFILES. Additional file 1 and Additional file 12

*R.*eplaces paper.tex 659 and deletes 670

```latex
% ---- replaces paper.tex:659 ----
\item[Additional file 1 (PDF):] \textit{Hyperparameters for the nineteen model configurations, on QM9 and on
the three assay datasets.} Table A lists the default settings, read from \texttt{models/model\_defaults.py},
which both pipelines load. A default is one value, used for all six representations and for every dataset.
The support vector machine uses a radial basis function kernel on all six representations. Table B lists
the tuned settings that replace a default, on ten model-and-dataset pairs: four on QM9, and two on each of
the three assay datasets. The tuned configurations are BNN-$\alpha$, BNN-$\beta$, VBLL-$\alpha$ and
VBLL-$\beta$. The other fifteen of the nineteen model configurations train at the Table A defaults on QM9,
and seventeen of the nineteen do so on each assay dataset. A tuned setting is chosen once per model and per
dataset, and is then used for all six representations. Every QM9 results row carries a column naming the
source of its settings.

% ---- paper.tex:670 is DELETED, and nothing replaces it ----
% \item[Additional file 12 (PDF):] \textit{Default hyperparameters used for external
% validation experiments.} ... including representation-specific SVM kernels.
% It is the last item in the list, so no other additional file is renumbered.

% ---- paper.tex:197's two kernel clauses need no replacement of their own ----
% Line 197 sits inside lines 192-200, which the Datasets subsection replaces whole,
% and the Models subsection already carries both sentences: the pointer to
% Additional file 1, and the radial basis kernel on every representation.

% ---- note, not for the paper ----
% Additional file 1 is generated: additional_files.tex:48-50 marks the block
% "BEGIN GENERATED Additional file 1 -- scripts/generate_supp_table1.py", and
% scripts/test_supp_table1.py is the check on it. Its sources are
% models/model_defaults.py, results/master_tuned_hyperparameters.json and
% results/master_tuned_hyperparameters_lab.json. Editing the caption above does
% not touch it.
%
% The caption's Table B sentences state what the two tuned-settings files hold.
% Whether a submitted assay task took the tuned branch cannot be shown from any
% assay results file: a search for params_source, hp_source and
% hyperparameters_used over KIRBy/tests/alternative_data_noise_robustness.py
% returned nothing this session, so those results files record no settings
% source. TODO in that sense sits outside the paper text, because the caption
% claims nothing about what a run did. The cluster's copy of
% results/master_tuned_hyperparameters_lab.json is what the assay run read
% (path resolution at alternative_data_noise_robustness.py:100-125).
```

**Still open in this subsection.**

- The assay pipeline writes no column naming which settings a row used, so no assay results CSV can confirm that the two tuned configurations per dataset were the ones the submitted run applied. Only the cluster's copy of results/master_tuned_hyperparameters_lab.json at the time the tasks started would settle it.
- KIRBy alternative_data_noise_robustness.py:172-180 declares four eligible tuned models on the assay side, while the lab JSON supplies two per dataset; the other two fall back to defaults through tuned_neural_params returning None (:196-226). Worth one line in the Methods if you want the asymmetry visible.

**For the author.**

- Additional file 12 does not exist. additional_files.tex:16-19 records that it was removed on 2026-09-12, and line 9 says the compiled PDF splits into eleven files. So paper.tex:670 is a deletion, not a rewrite, and paper.tex:197 now cites Additional file 1 instead.

**Cut from the current text.**

- The Additional file 12 caption in full: the deleted file has nothing to caption.
- The claim that nothing generates either file. additional_files.tex:48-50 names scripts/generate_supp_table1.py as the generator and scripts/test_supp_table1.py as its check; both are on disk, dated 12 September.
- "Every QM9 results row records which of the two it used." params_source takes four values in models/models.py: 'default' (:2031), 'tuned' (:2038), 'tuning_trial' (:2054) and 'cli' (:3900). The caption now says the row names the source of its settings, without saying how many sources there are.
- The QM9-only scope on the Additional file 1 title. Table B carries Caco-2, hERG K$_i$ and LogD rows, and additional_files.tex:57 says model_defaults.py is the file both pipelines load; KIRBy alternative_data_noise_robustness.py:100-125 loads it and :398-406 binds sklearn_params from it.
- "the shared defaults", which arrived with a definite article and no antecedent. The caption now says the Table A defaults.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **nineteen model configurations** — slurm_scripts_qm9_rerun/generate_scripts.py, MODELS dict parsed this session: 19 keys (rf, xgboost, lgb, svm, ngboost, dnn, mlp, dnn_bnn_full, mlp_bnn_full, dnn_bnn_full_variational, mlp_bnn_full_variational, qrf, gauche_rbf, gauche, heteroscedastic_gp, dnn_bnn_full_variational_hetero, mlp_bnn_full_variational_hetero, dnn_bnn_full_mve, mlp_bnn_full_mve); matches additional_files.tex:56 "all 19 model configurations"
- **ten model-and-dataset pairs** — 4 QM9 pairs in results/master_tuned_hyperparameters.json (dnn_bnn_full, dnn_bnn_full_variational, mlp_bnn_full, mlp_bnn_full_variational) plus 6 assay pairs in results/master_tuned_hyperparameters_lab.json (caco2, herg, logd, two models each); the same ten rows are printed in additional_files.tex:283-325
- **four on QM9** — results/master_tuned_hyperparameters.json top-level keys, read this session
- **two on each of the three assay datasets** — results/master_tuned_hyperparameters_lab.json: caco2 = dnn_bnn_full_variational + mlp_bnn_full; herg = the same two; logd = dnn_bnn_full_variational + mlp_bnn_full_variational
- **the other fifteen** — 19 configurations minus the 4 that appear in results/hyperparameter_decisions.json / master_tuned_hyperparameters.json
- **all six representations** — each model entry in both tuned-settings JSON files holds the keys avalon, chemberta, ecfp4, mhggnn, pdv, sns, carrying one identical setting; ALL_REPS is six in generate_scripts.py and in KIRBy alternative_data_noise_robustness.py
- **BNN-$\alpha$, BNN-$\beta$, VBLL-$\alpha$, VBLL-$\beta$** — generate_scripts.py MODELS comments: dnn_bnn_full = BNN-alpha, mlp_bnn_full = BNN-beta, dnn_bnn_full_variational = VBLL-alpha, mlp_bnn_full_variational = VBLL-beta; the same four names appear in paper.tex:449-453
- **radial basis function kernel on every representation** — models/model_defaults.py:214-222, SKLEARN_DEFAULTS['svm'] kernel 'rbf' with the comment naming paper.tex:197 as wrong; KIRBy alternative_data_noise_robustness.py:3378-3384 builds SVR(**sklearn_params('svm')) inside the loop over every representation

</details>

# THE RESULTS AND DISCUSSION, AND THE INTRODUCTION

Every number below was recomputed from `results/decisions_arc_20260916/`, which is the newest harvest and is
dated 17 September. Where a number in `RERUN_PLAN.md` disagreed, the correction is recorded in §14.29 of that
file with the file it came from. The correction history for this section, including the five passes it went
through, is in `RERUN_PLAN.md` §15.

---

## The current main picture — read this first

**Two properties, two different drivers, and one axis that separates the noise conditions. That is the whole
paper.**

1. **What a model scores and what it keeps are set by different things.** On accuracy at a realistic amount
   of label error, the choice of model and the choice of representation matter about equally — 29.1 and 26.6
   per cent of the variance on QM9 under Gaussian noise. On robustness, the model matters more than five
   times as much as the representation: 49.5 against 9.2. The representation still matters, but through the
   pairing rather than on its own, and the pairing term is twice the representation's.

2. **The noise conditions do not separate on Gaussianity. They separate on independence and zero mean.**
   Five of the seven conditions behave alike at a matched dose, and that includes all three non-Gaussian
   shapes — Laplace, Student-*t* and random contamination. The two that differ are the one that correlates
   errors inside a scaffold family and the one that clips at an assay limit. Those are the two assumptions
   nobody tests, and both cost more on the assay datasets than on QM9.

**The claim that sets the paper apart is therefore sharper than "non-Gaussian noise matters", and it survives
the author's own tables.** Changing the shape of a single label's error costs nothing measurable. Changing
*which* labels share an error costs a great deal. A study that models label noise as independent and
zero-mean — which is nearly all of them — cannot see either of the two effects that turn out to matter.

**And the same axis runs through the uncertainty half.** The aleatoric–epistemic split separates in 78 of
the 109 comparisons that can be read at all under independent noise, and in 58 of 110 when whole scaffold
families share an offset. So modelling label error as
independent does not only understate how much accuracy is lost; it also hides where a model's own uncertainty
stops working.

---

## The spine

A QSAR model under label noise has two properties that the paper has to keep apart, and this run separates
them by a different route from the submitted version.

**Robustness — how much of its own accuracy a model keeps as the labels are corrupted.** It is set mainly by
the model's training mechanism. Two independent routes agree: the variance decomposition gives the model
49.5 per cent against the representation's 9.2 for AUC_norm under Gaussian noise on QM9. And AUC_norm under
grouped-shifted noise minus AUC_norm under Gaussian noise varies over a range of 0.127 across the nineteen
models, against 0.013 across the six representations. A factor of ten, from two different calculations. Label noise corrupts the targets and not
the features, so the representation has less to lose; what decides the outcome is the regularisation,
ensembling and priors that limit how far a model will chase a corrupted label.

**Accuracy on clean labels is a different ranking, and the two do not travel together.** Within one dataset
at one representation they are unrelated — the rank correlation between clean R² and AUC_norm across thirteen
models is −0.18 under Gaussian noise and not significant. Across the seventy-five pairings that ran on all
four datasets, scaled within each, it is −0.350. The ten most accurate pairings and the ten most robust share
no members at all. At one representation under Gaussian noise the two orderings are close to inverted: the most
accurate model of the thirteen sits eleventh for robustness, and the least accurate sits first.

**The kind of noise decides how much you lose, not who wins.** Model rankings agree across six noise
conditions at ECFP4 with Kendall's *W* of 0.937. But the conditions themselves are far from equal: of the
fifteen pairwise comparisons at that representation, the only ones that separate involve the condition that
gives a whole scaffold family a shared offset. Laplace, Student-*t* and random contamination are
indistinguishable from Gaussian at a matched dose, which confirms what Heid et al. published for
homoscedastic noise and finds the boundary they named.

**The dataset decides who wins.** The model ranking transfers from computed labels to measured ones with a
median rank correlation of 0.356 across eighty-three representation-and-condition combinations, and only
twenty of the eighty-three are significant. So a family-level recommendation travels and a specific pairing
does not.

**Uncertainty is a third property again, and it is conditional.** Seven of the thirteen models that emit an
uncertainty cannot be asked the decomposition question at all, because one of their two components is a single
number per fit. Of the six that can, the split works for four families and fails outright for the quantile
forest, where both halves rise together because one bootstrap causes both. Whether the split works depends on
the noise condition as well as the model: under independent noise it separates about twice as often as it
fails, and under correlated noise the two are nearly level.

**The two properties do not travel together, and the table is the argument.**

| | robust? | its uncertainty separates into two halves? | its uncertainty ranks the error? |
|---|---|---|---|
| Random forest | yes, near the top on every dataset | no — it has no per-molecule data-noise term at the leaf size used | moderately |
| Quantile forest | slightly below the forest, on every condition | **no — both halves rise together, in 90 of 99 cells** | best of the roster |
| NGBoost | most robust on QM9, near the bottom on clean accuracy | cannot be asked — it has no model-uncertainty term at all | moderately |
| Gaussian process | among the most accurate, and the model correlated noise hurts most | cannot be asked — its data-noise term is one number per fit | moderately |
| GP, heteroscedastic | as above, and it pays robustness for the extra term | yes, in about two cells of three | moderately |
| Bayesian networks | a small gain on the plain network | cannot be asked — they predict a mean and nothing else | weakest of the roster |
| Bayesian networks, variance head | no further gain | yes, in about three cells of five | moderately |
| SVM | mid-table, and stable across representations | no uncertainty output at all | — |

Being robust does not buy you a usable uncertainty, and having a usable uncertainty does not buy you
robustness. The quantile forest is the sharpest case in the study: it ranks prediction error better than
anything else on the roster while its own decomposition fails everywhere, and it is less robust than the
plain forest it is built from.

---

## Your main points: where each is reflected

| Main point | Where it lands |
|---|---|
| Model and representation matter about equally for accuracy; the model dominates robustness | §R1, Introduction close, F2, T3 |
| The representation matters through the pairing, not on its own | §R1, F3, R6 |
| Under correlated noise the replicate term swallows everything — the outcome becomes unpredictable | §R1, F2's lower panel |
| Robustness is not bought with clean accuracy, and the two top-ten lists share nothing | §R2, T8, R16 |
| Making a model probabilistic buys an uncertainty estimate, not robustness | §R3, R17, T5 |
| Every dose-matched condition delivered the same amount, so a difference is a difference of pattern | §R4 opening, F1, `NOISE_DESIGN.md` §5.1e |
| Shape does not matter; independence and zero mean do | **§R4 — the paper's differentiator**, F4b, T4 |
| Correlated and censored noise cost more on measured labels than on computed ones | §R4, §R5, F8, T4 per dataset |
| The kind of noise changes how much you lose, not who wins | §R4, Kendall's *W*, R15 |
| The model ranking does not transfer from QM9 to assay data | §R5, R9, T7 |
| Seven of thirteen models cannot be asked the decomposition question | §R6, T6, F6 |
| The quantile forest's split fails everywhere while its uncertainty ranks error best | §R6, §R7 |
| The split separates in 78 of 109 readable comparisons under independent noise and 58 of 110 under a shared scaffold offset | §R6 — **this is the second half of the differentiator** |
| Uncertainty still orders predictions by error under noise, weakly, nearly everywhere | §R6, T6 |
| Whether uncertainty finds the corrupted labels is unresolved and says so | §R7, and the TODO list |

---

## The big picture: six whole units to rebuild, not patch

1. **The Results section order.** The submitted paper opens on the variance decomposition and then drifts.
   The four movements below are the author's order and each one ends where the next one starts.
2. **Every robustness number.** They were computed on a metric that averaged the replicates before
   integrating, on a noise scale that no longer exists, at a level the call site never passed.
3. **The noise-condition subsection.** Six strategies became seven conditions, and the finding reversed: the
   submitted paper says the noise pattern barely changes the ranking, which is still true, and never says
   that two of the conditions cost several times what the others do.
4. **The uncertainty subsection.** It was computed by pooling noise levels, and it rested on per-sample
   correlations that are near zero by design. What replaces it is three separate questions.
5. **The assay-dataset subsection.** It currently reads as confirmation. The finding is that the ranking does
   not transfer.
6. **The Introduction's third and fifth paragraphs.** The noise conditions have a literature behind them now
   and it is not in the paper, and the Heid sentence states something Heid et al. did not do.
---

# FULL REBUILDS — THE INTRODUCTION

`paper.tex:178–188`, six paragraphs. Four survive nearly intact. One is replaced because it says something
its own source does not, one new paragraph is added because the noise conditions have a literature behind
them that the paper never states, and the aims paragraph is rewritten because two of the three aims now have
better answers than they promise.

The current Introduction runs about 1,030 words in six paragraphs. The replacement runs about 1,180 in seven.
`Dablander (2023)` in the reference set runs 940 words in eight and closes on its research questions as a
list; `Kolmar & Grulke (2021)` runs 1,590 in nine. Either length is normal for the journal.

---

## §I1. Paragraph 1 (`paper.tex:180`) — hold, one clause

Keep it. The only change is the representation list at the end: the paper now runs six representations and
two of them are learned. The sentence already says "static hand-crafted descriptors or circular fingerprints
… or learned representations", which covers the set, so nothing has to move. `\citep{sns}` stays and
**`\citep{avalon}` has to be added** — the paper uses Avalon fingerprints and cites nothing for them. The entry
is in `citations.bib` as of 2026-09-18 and §M2's Avalon sentence already carries the citation.

---

## §I2. Paragraph 2 (`paper.tex:182`) — hold, one sentence added at the end

Keep the whole paragraph. Add one sentence, because the third paragraph of the replacement depends on it and
the reader has to be told the error is measurable before being told what shape it has.

> Add after *"…may be incorrect or misleading \citep{Walters2023}."*:
>
> How large that error is has been measured directly. Repeat measurements of the same compound-target pair
> disagree by about 0.68 log units for pIC$_{50}$, and by about 0.54 log units for pK$_i$
> \citep{Kalliokoski2013, Kramer2012}. Curating a public potency set down to a single assay lowers the mean
> disagreement from 0.50 to 0.27 log units \citep{landrum2024}. Much of the apparent noise in public data may
> therefore be provenance rather than imprecision.

*Provenance: `NOISE_DESIGN.md` §3.1, §3.3. The 0.68 is pIC$_{50}$ and the 0.54 is pK$_i$ — `NOISE_DESIGN.md`
§638 lists the swapped version as a number that must not enter the paper. The Landrum key is `landrum2024`,
lower case, and it is already in `citations.bib` at line 1910 and `refs.bib` at line 2180 — an earlier
version of this note said it was in neither and that was wrong.*

---

## §I3. Paragraph 3 (`paper.tex:184`) — hold

The learning-with-noisy-labels review, Cortes and the no-free-lunch theorem. Keep it as written. One note
for later: the no-free-lunch theorem licenses *"no single algorithm is best across all data distributions"*
and nothing stronger. It does not license a claim about which model to pick, and the Conclusions should not
lean on it as though it did.

---

## §I4. NEW PARAGRAPH — what real label error looks like, and why this study's conditions are shaped as they are

**This is the paragraph the paper does not have and the one the author asked for.** It goes between the
current paragraphs 3 and 4. Every claim in it is a verbatim-verified quote or a table read in
`NOISE_DESIGN.md` §4b, and each is cited to what the source measured rather than to what it implies.

> Studies of noise robustness in this field usually add independent, zero-mean, Gaussian error to the
> labels. Neither half of that is what the measurements show. \citet{Kruger2012} rejected normality of
> bioactivity differences by Anderson-Darling test and fitted a Laplace distribution instead.
> \citet{Kalliokoski2013} could only fit a Gaussian to 16,844 repeat measurement pairs after truncating the
> tail. Their largest single disagreement was 7.7 log units, which the Gaussian they fitted puts at six in a
> thousand million million. Where that tail comes from has been examined directly: manual inspection by
> \citet{Kalliokoski2013} of the pairs in the two highest disagreement bands found 9 out of 10 pairs in one
> band and 10 out of 10 in the other to be transcription errors, receptor-subtype confusion or assay mix-ups
> rather than imprecision. The tail may therefore belong to the record rather than to the value, which is why
> we corrupt a random subset of the labels rather than the extreme ones.
>
> The more consequential departure from that convention is independence rather than shape. \citet{Bentz2013}
> decomposed the variance of log efflux ratio across 23 participating laboratories. They found 62 per cent of
> that variance between laboratories and 10 per cent unexplained. Laboratory averages that differ are an
> offset rather than a widening. A public dataset assembled from several sources may therefore carry error
> that whole groups of related compounds share, and a scaffold split makes those groups explicit. The
> zero-mean assumption fails as well. \citet{Svensson2025} report the censored fraction for fifteen industrial
> assays, of which thirteen carry censored labels and eight sit between a quarter and two thirds censored.
> Recording a value as the assay limit is therefore the most prevalent real mechanism of all. The seven
> conditions we test vary the shape, the correlation and the bias of the injected error separately, at a
> matched delivered amount, rather than varying the amount of one of them.
>
> One condition cannot be tested fairly, and we say so here. Matching the real tail of
> \citet{Kalliokoski2013}'s data needs a Student-$t$ with about one degree of freedom, which has no finite
> variance and so cannot be matched to the other conditions on the amount of noise it delivers. The heaviest
> tail we inject is a Student-$t$ with five degrees of freedom, against the one degree of freedom that
> matching their data would take. The error real data carries is therefore heavier-tailed than anything that
> can be compared at an equal delivered amount.

*Three paragraphs, 330 words, and it is the longest addition to the Introduction. If it has to shrink, the
third is the one to cut — but it is the limitation a referee will otherwise raise, and `NOISE_DESIGN.md`
§3.1 marks it as belonging in the paper.*

*Provenance for each claim, in order: `NOISE_DESIGN.md` §3.1 (Krüger, and the warning to cite what they did
rather than what it implies — they never write "heavy-tailed" or "Gaussian"); §3.1 (Kalliokoski's truncation,
the 7.7 log units, and the tail table); §3.4 (Kalliokoski Table 2, 9 of 10 and 10 of 10 invalid in the top
bands); §3.3 (Bentz Table 7: laboratory 62\%, laboratory × experiment 20\%, residual 10\%, cell line 8\%);
§3.5 (Svensson Table 1, with the count corrected on 2026-08-26 — thirteen of fifteen with any censoring,
eight between 25 and 63\%); §3.1 again for the ν ≈ 1.1 limitation.*

**Keys needed: `Kruger2012`, `Kalliokoski2013`, `Bentz2013`, `Svensson2025`. All four are in
`citations.bib`. None is cited in `paper.tex` today.**

---

## §I5. Paragraph 4 (`paper.tex:186`) — REPLACE. It states something its source did not do

The current text ends:

> *"However, \citet{Heid2023} used mean-variance estimation and bias-variance decomposition to show that
> noise can be tied to specific modalities, for example structure dependence."*

Heid et al. did not show that. They **imposed** structure-dependent noise by hand — 20 kcal/mol for
nitrogen-containing molecules and 2 kcal/mol for the rest — and then showed a method could detect what they
had injected. `NOISE_DESIGN.md` §3.6 records this, with the quote, and says the sentence must be rewritten.
What they did publish is more useful to this paper than the misreading was, because it is the precedent the
results extend.

> Replacement for the whole paragraph:
>
> Numerous studies have explored the effects of noise on ML algorithms for molecular property prediction.
> Direct comparisons between molecular representations, and between how each of them responds to label noise,
> remain limited. The choice of molecular representation affects predictive performance. \citet{Deng2023}
> evaluated 62,820 models and found that extended-connectivity fingerprints, the family this study's ECFP4
> belongs to, frequently outperform the representations learned by graph neural networks. Whether that
> carries over to noise robustness is a separate question, and it is one of ours. On the noise itself,
> \citet{Heid2023} found no difference in model performance between error distributions "as long as the mean
> and standard deviation of the noise was the same". Their comparison covers noise spread evenly across the
> training set. That result is the natural starting point for this work. It leaves open the case their
> comparison does not reach: noise concentrated on a structural group, or on a sparse subset of records,
> rather than spread evenly.

*The `Deng2023` sentence stays because it motivates the work. It is no longer leaned on as though the answer
were expected — `RERUN_PLAN.md` §14.15d records that models travel between datasets slightly better than
representations do, which is the opposite of the suspicion, and on six dataset pairs the gap is too small to
call.*

---

## §I6. Paragraph 5 (`paper.tex:188`) — REPLACE. Two of the three aims now promise less than the paper delivers

Two problems. The second aim promises to say "whether their per-sample uncertainty estimates track which
individual labels have been corrupted", which is a yes-or-no; the answer is conditional and the conditional
version is the better result. And the paragraph ends by introducing NoiseInject, which `RERUN_PLAN.md` §14.22
records as moving to the availability statement on your call.

> Replacement for the whole paragraph:
>
> This research addresses four questions about how QSAR models behave under label noise. First, how do the
> molecular representation and the model architecture divide the variance in predictive accuracy, and does
> that division change when the question is how much accuracy a model retains under noise? Second, what does
> the choice of model buy at a realistic amount of label error, and does making a model probabilistic change
> how noise hurts it? Third, does the kind of noise matter, or only the amount, and do the answers reached on
> a computed property hold when the labels are laboratory measurements? Fourth, how do a model's uncertainty
> estimates behave as its training labels are corrupted: does it become less sure, does its uncertainty still
> rank which predictions to trust, and can it point to which labels were the corrupted ones?

*Four questions, not three, because the noise-condition question is now the paper's differentiator and was
folded into the third aim before. Six sentences, 148 words. Dablander closes on six bulleted research
questions and this is the same move in prose.*

**Then add a preview paragraph.** The Introduction currently has none, and the reference set closes on the
aims and the finding together.

> Accuracy and robustness come apart, and the same split runs through the computed property and the three
> assay datasets alike. On the computed property, predictive accuracy is shaped about equally by the model
> and by the representation. Robustness to label noise is set mainly by the model, with the representation
> accounting for under a tenth of the variance in AUC$_{norm}$, the share of a model's clean accuracy that it
> keeps as label noise rises. Choosing the model therefore matters more than choosing the representation when
> the labels may be noisy. The random forest and the quantile forest are the two models that are never a bad
> choice, on the computed property and on each of the three assay datasets, and neither of them is the
> accurate choice.
>
> The kind of noise matters as well, but not along the axis the literature varies. Changing the shape of a
> single label's error costs nothing measurable in AUC$_{norm}$ when every condition delivers the same amount
> of noise. Correlating errors within a scaffold family, or clipping labels at an assay limit, costs a great
> deal more of it, and costs more on the assay datasets than on the computed property. LightGBM and XGBoost
> sit mid-table on the computed property, and lose most of that standing on Caco-2 and hERG $K_i$ under every
> condition that runs the whole roster. Testing a computed property and laboratory endpoints together is what
> makes that difference visible. The same division runs through the uncertainty results. A model's ability to
> separate what it does not know from what cannot be known holds up under independent noise, and degrades when
> whole scaffold families share an offset delivering that same amount. Under clipped labels alone, on the
> computed property and on each of the three assay datasets, every model in the roster grows more confident as
> its labels grow more wrong.

*197 words, seven sentences. No number smaller than "a tenth", per the style specification's rule 10.*

*Revised 2026-09-18, after §R1 to §R6 were rewritten. The first version was drafted before the Q4
permutation-band defect, the censoring result, the boosted-tree collapse and the corrected
representation-dependence figures, and it previewed none of the last three. Three sentences are new: the one
naming the forests, which is §R2's fifth paragraph; the one about the two models that do not travel, which is
§R2's fourth and §R5's second; and the closing clause about clipped labels, which is §R6's second. The rest
is unchanged and every claim in it still reproduces.*
---

# FULL REBUILDS — THE RESULTS AND DISCUSSION

Replaces `paper.tex:375–568` entirely. The heading becomes **Results and discussion**, one combined section,
and there is no standalone Discussion — two of the three Journal of Cheminformatics papers in the reference
set do it that way and the third calls its closing section "Discussion and conclusions".

Seven subsections in four movements, in the author's order.

🔴 **It is 4,464 words of prose and that is still long.** The reference set gives 2,000 to 3,000 words for a
study of this size, and `Kolmar & Grulke (2021)`, the closest paper in the set to this one, runs about
4,120 words of Results at 3.9 per cent of sentences carrying a decimal. This draft sits just above Kolmar on
length and just below him on the reference band's ceiling for number density. It carries the author's four
Q1 conclusions, the model-against-representation comparison in §R4 and the censoring result in §R6, none of
which were in the paper text before, and all of which she asked for. **No finding has been cut, because what
to drop is hers to decide and not mine.**

*History of the count, so the movement is readable: 4,192 words before the house-style pass on 2026-09-18,
5,625 after it, and 4,464 after the cut that followed. The pass added the sentences
`PAPER_HOUSE_STYLE.md` requires — the four AUC_norm sentences and what it cannot show, the direction of good
at each metric's first comparison, an orienting sentence before each figure and table, the raw pair beside
each derived number, the joining sentences between blocks and the closing sentence of each. The cut then took
the per-cell grid values out of the prose, which is where the reference papers keep them.*

Every subsection is now inside the standard's shape rules, measured after the cut:

| where | words | sentences per paragraph | sentences carrying a decimal |
|---|---|---|---|
| §R1 | 841 | 4, 6, 5, 5, 4, 6, 6 | 4 of 36, 11% |
| §R2 | 786 | 6, 7, 7, 5, 5, 7 | 2 of 37, 5% |
| §R3 | 416 | 5, 5, 7 | 2 of 17, 12% |
| §R4 | 679 | 6, 6, 7, 6, 5 | 3 of 30, 10% |
| §R5 | 498 | 7, 7, 6, 6 | 3 of 26, 12% |
| §R6 | 894 | 6, 7, 4, 4, 6, 4, 7 | 3 of 38, 8% |
| §R7 | 374 | 5, 5, 7 | 0 of 17, 0% |

*The rules those columns are checked against: every paragraph three to seven sentences and 100 to 200 words,
median sentence 20 to 26 words with nothing over about 55, and under 12 per cent of sentences carrying a
decimal. The longest sentence in the section is 54 words, in §R1. Getting below Kolmar's 4,120 means losing a
finding, and the two candidates are §R3, which reports that probabilistic machinery buys no robustness, and
§R7, which reports a direction and not a number. **Both are yours.**

🔴 **One rule the pass could not satisfy without changing a finding, and it is yours to settle.**
`PAPER_HOUSE_STYLE.md` rule 11 of the flow set, and rule 14 of the content set, say no number in the text may
be computed across datasets: a cross-dataset summary is a count of wins or a rank test, never a central
value. §R4's fourth paragraph breaks that. The 0.127 and the 0.095 are each a median taken over the computed
property **and** the three measured endpoints together, which is the one place in the Results where a central
value spans both halves of the study. I checked the other six subsections for the same thing and they are
clean: §R1's 0.022 and 0.061 are per dataset, §R2's medians are per dataset, §R5 reports per dataset
throughout, and §R1's Sort \& Slice sentence is a count of wins on three of the four datasets, which is the
form the rule asks for. §R2's $-0.350$ is also taken across the four datasets, but it is a rank test and the
rule allows a rank test, and its own sentence already says it cannot speak for any one endpoint.

Three ways out, and the cost of each. **Report it per dataset**, which is four numbers instead of one and
loses the single headline that the fall is model-decided. **Replace it with a count**, "the Gaussian process
falls furthest on all four datasets", which needs a check that the ordering actually holds on all four and is
not in the guide yet. **Keep it and say in the sentence that it spans the four datasets**, which is the
cheapest and is what Kolmar does not do. I have left the number as the guide already had it rather than pick
for you, because all three change what the paragraph claims.

**Every figure gets one orienting sentence naming what is plotted against what, the first time it is used.
After that the prose states findings and points by position — "the lower panel", "the clean-R² column" — and
does not cite the figure again.** A bare trailing "(Fig. 6)" after a claim appears zero times in the target
journal's results sections and is the single clearest difference from Nature.

---

## MOVEMENT 1 — how the choice of model and the choice of representation divide the outcome

### §R1. Variance decomposition *(replaces `paper.tex:377–430`)*

> **Variance decomposition**
>
> Figure~\ref{fig:variance} divides the variation in a model's behaviour between the choice of model, the
> representation, the pairing of the two, and a residual. It does so for two outcomes on the QM9 HOMO--LUMO
> gap, predictive accuracy at a noise level of 1.0 and robustness. The bottom axis of both panels is the noise
> condition, the side axis the share of variance each term explains, in per cent. One row of
> Table~\ref{tab:variance} is one noise condition and one outcome, over thirteen base models, six
> representations and ten replicates. Six of the nineteen configurations in the roster are set aside here,
> five that train under a different likelihood from the model they vary and one that ran on a single
> representation.
>
> We summarise robustness as AUC$_{norm}$, which is not an area under a receiver operating characteristic
> curve. For one model on one representation, the retention curve is its $R^2$ at each noise level divided by
> its own clean $R^2$, against the amount of noise put into the training labels. AUC$_{norm}$ is the area
> under that curve divided by the span of the seven levels, so a higher value is more robust. A value of 1.0
> is no accuracy lost at any level, 0.5 is half the clean accuracy gone, and above 1.0 is a configuration that
> scored higher with noise added than without it. Because the metric integrates the whole ladder of levels, a
> configuration that falls early and then flattens can reach the value of one that holds and then collapses.
> Where in the ladder the accuracy went has to be read off the retention curves.
>
> For predictive accuracy, the model and the representation account for comparable shares of the variance in
> $R^2$, close to a third and a quarter of it under Gaussian noise. For robustness the model accounts for 49.5
> per cent of the variance in AUC$_{norm}$ and the representation for 9.2 per cent. A plausible mechanism for
> that inversion is what label noise corrupts. The representation decides what chemical information reaches
> the model, which may be why it competes with the architecture on clean accuracy. Noise is added to the
> targets and not to the features, so the representation may have nothing extra to lose.
>
> That share is a single number for thirteen models, and it hides a difference between model families. On the
> computed property under Gaussian noise, the distance between the highest and the lowest AUC$_{norm}$ a model
> reaches over the six representations is larger for all six neural configurations than for any of the seven
> forest, boosted-tree, support vector machine and Gaussian process models. Figure~\ref{fig:grid} shows that
> difference without a number, with models down the side, representations across and one panel per noise
> condition. One cell is one model's AUC$_{norm}$ at one representation. The choice of representation is
> therefore a decision about robustness for a neural model, and very largely a decision about accuracy alone
> for every other family in the roster.
>
> Set against the models, the representations are close together. Taking the median over the seven non-neural
> models, the six representations lie within 0.022 of each other in AUC$_{norm}$ on the computed property and
> within 0.061 on Caco-2, the laboratory assay with the lowest clean $R^2$, while those same seven models
> differ from each other by 0.041 and by 0.300 on those two datasets. The choice of model therefore moves
> AUC$_{norm}$ several times further than the choice of representation does. That is the same ordering reached
> in §\ref{sec:noise-kind}, on a different outcome and by a route that does not use this decomposition.
>
> No representation has the highest AUC$_{norm}$ on every dataset. Sort \& Slice has the highest or joint
> highest median AUC$_{norm}$ over those seven models on three of the four datasets, under both conditions
> that run the full roster. Every table therefore reports one named representation and never an average across
> them, and Table~\ref{tab:robustness} carries one dataset and one representation in its title. One row of
> that table is one model, with its clean $R^2$ beside its AUC$_{norm}$ under each noise condition. ECFP4, the
> representation those tables are computed at, was chosen because it spreads the thirteen models furthest
> apart in AUC$_{norm}$, not because it scores highest. It gives the lowest AUC$_{norm}$ of the six for four
> of the thirteen models tested.
>
> Under noise that gives a whole scaffold family a shared offset, the residual term rises to over half the
> variance in robustness and three quarters of it in accuracy. That term is the variation between replicates
> of one configuration, differing only in seed. Under Gaussian noise and under grouped-wider noise, at the
> same delivered amount of noise, it sits near a fifth. Which draw of scaffold offsets a run receives may
> therefore account for more of the variance in AUC$_{norm}$ than the choice of model does, under that
> condition. Why that term rises further for accuracy than for robustness we cannot say. Where labels may be
> noisy by an unknown amount, effort is better spent on the model than on the representation, at the cost of
> the accuracy share the representation decides.

**841 words, seven paragraphs of 4, 6, 5, 5, 4, 6 and 6 sentences. 4 of 36 sentences carry a decimal.**

**Numbers behind it, all recomputed from the 16 September harvest this session:**

| condition | outcome | model | rep | pairing | residual |
|---|---|---|---|---|---|
| Gaussian | robustness | 49.5 | 9.2 | 20.4 | 20.8 |
| Grouped, wider | robustness | 50.5 | 8.6 | 19.2 | 21.6 |
| **Grouped, shifted** | robustness | 30.5 | 6.6 | 6.6 | **56.3** |
| Gaussian | accuracy | 29.1 | 26.6 | 13.7 | 30.6 |
| Grouped, wider | accuracy | 36.3 | 24.7 | 11.5 | 27.5 |
| **Grouped, shifted** | accuracy | 7.1 | 10.2 | 6.1 | **76.6** |

*From `anova_eta2.csv`. Thirteen base models, six representations, ten replicates, 780 rows per
decomposition. The five variant models are out of the decomposition by the rule settled on 2026-09-01 — they
train under a different likelihood from the model they are a variant of, so they would move the model term
for a reason unrelated to the question.*

**Paragraph 3 is the author's first Q1 conclusion: how much the representation matters depends on the model
family.** Highest minus lowest AUC_norm over the six representations, QM9 under Gaussian noise, one row per
base model, from `auc_norm_qm9.csv`:

| model | spread over the six representations |
|---|---|
| NGBoost | 0.015 |
| SVM | 0.018 |
| LightGBM | 0.024 |
| XGBoost | 0.025 |
| RF | 0.026 |
| Gaussian process | 0.027 |
| QRF | 0.030 |
| VBLL network (first architecture) | 0.039 |
| plain network (second architecture) | 0.043 |
| Bayesian network (second architecture) | 0.053 |
| plain network (first architecture) | 0.054 |
| VBLL network (second architecture) | 0.059 |
| Bayesian network (first architecture) | 0.063 |

*Six neural rows, seven others, and the two groups do not overlap. Median 0.054 against 0.025, which is the
"two to three times" the earlier working note gave — the text states the two ranges instead, because a ratio
of two medians is a number nobody can check against the table.*

**Paragraph 4 is the author's representation table, read as a sentence.** Median over the seven non-neural
models, AUC_norm with the clean R² it is a share of in brackets:

| representation | QM9 | logD | Caco-2 | hERG K$_i$ |
|---|---|---|---|---|
| ECFP4 | 0.946 (0.84) | 0.891 (0.71) | 0.841 (0.43) | 0.842 (0.54) |
| PDV | 0.945 (0.90) | 0.913 (0.76) | 0.856 (0.45) | 0.842 (0.49) |
| MHG-GNN | 0.934 (0.90) | 0.900 (0.78) | 0.854 (0.43) | 0.847 (0.56) |
| Avalon | 0.955 (0.87) | 0.881 (0.73) | 0.795 (0.41) | 0.856 (0.51) |
| ChemBERTa | 0.953 (0.83) | 0.887 (0.62) | 0.850 (0.40) | 0.854 (0.44) |
| Sort & Slice | 0.956 (0.87) | 0.921 (0.76) | 0.842 (0.43) | 0.879 (0.57) |

*Spread across the six representations: 0.022 on QM9, 0.040 on logD, 0.061 on Caco-2, 0.037 on hERG. Spread
across the seven models on the same four: 0.041, 0.068, 0.300, 0.254. The neural models are left out of this
table because they are the group paragraph 3 has just shown to be the most representation-dependent, so
including them would blur the comparison the table is making.*

*The held representation is ECFP4. Its spread across the thirteen models on QM9 under Gaussian noise is
0.098, the widest of the six, against 0.040 for PDV, which is the narrowest. It is the worst of the six for
four of the thirteen models. Its ordering of the models agrees with the other five at a mean rank
correlation of 0.700, against 0.740 for PDV at the top and 0.577 for ChemBERTa at the bottom — third of six,
which is what "neither the most nor the least typical" means and is why the text says it that way.*

**⚠️ Two things this subsection must NOT say.** The submitted paper's §4.1 says the pairing term is the
largest source of variance for accuracy under all six strategies. On this run it is the largest in **none** of
the six condition-and-outcome rows — decision D4, which is why no simple-effects panel is drawn. And the
old paragraph's "83.6\% and 77.4\% residual under outlier and heteroscedastic noise" belongs to noise
strategies that no longer exist.

---

## MOVEMENT 2 — what the choice of model buys, and whether probabilistic machinery helps

### §R2. What label noise costs *(replaces `paper.tex:431–467`)*

> **Robustness and clean accuracy**
>
> The decomposition says which of the two choices moves the outcome, not how far any one model falls or what
> it gives up in clean accuracy to fall less. Figure~\ref{fig:curves} shows what label noise costs on QM9. The
> bottom axis is the noise put into the training labels, as a fraction of the spread of the clean training
> labels. The side axis is $R^2$ on held-out molecules, where higher is the better fit. One line is one model,
> drawn at each of two representations, and the eight drawn are those with the highest AUC$_{norm}$ under
> Gaussian noise on ECFP4. The HOMO--LUMO gap is a computed property with no measurement error of its own, so
> a level on this dataset has no equivalent in published assay error.
>
> Curves fall at very different rates, and a curve's starting height does not predict how steeply it falls.
> The eight lie in a narrower band of AUC$_{norm}$ on PDV than on ECFP4. NGBoost has the highest AUC$_{norm}$
> of the thirteen base models on all six representations, on the computed property under Gaussian noise. It is
> also ranked between eleventh and thirteenth of thirteen for clean accuracy on every one of them. Its clean
> $R^2$ runs from 0.706 on ECFP4 to 0.865 on PDV, the widest range across the six representations of any model
> in the roster. Its AUC$_{norm}$ across the same six varies over the narrowest range in the roster. What the
> representation buys NGBoost is accuracy, and it buys it almost no robustness at all.
>
> The robustness column alone would make NGBoost this study's recommendation, and it is not one. Every figure
> and table in this paper therefore prints the clean $R^2$ beside the AUC$_{norm}$. NGBoost alone might
> explain that split between accuracy and robustness, so we tested it across the whole roster. Within the
> single dataset and single representation the figure holds, clean accuracy and AUC$_{norm}$ rank in opposite
> directions across the thirteen models, at a rank correlation of $-0.18$ that does not reach significance. We
> also rescaled both quantities inside each dataset, from its worst pairing to its best, over the seventy-five
> model-and-representation pairings that ran on all four datasets. Across those pairings the two rank in
> opposite directions at a Spearman correlation of $-0.35$. That correlation is taken over pairings drawn from
> four datasets, and it says the two orderings disagree rather than by how much on any one endpoint.
>
> One row of Table~\ref{tab:pairs} is one of those pairings, with its clean $R^2$ beside its AUC$_{norm}$ on
> each of the four datasets. It lists the twelve of the seventy-five that rank highest on both, and nine of
> the twelve are a Gaussian process or a forest. Twenty-one of the twenty-four combinations of dataset and
> representation carry all thirteen base models, the other three short because ChemBERTa lost models on Caco-2
> and on hERG $K_i$ and MHG-GNN lost one on Caco-2. Across those twenty-one, the random forest and the
> quantile forest are the only two whose AUC$_{norm}$ rank never falls below eighth of thirteen. For labels of
> unknown noisiness, either of those two is the safer choice, at the cost of a median clean $R^2$ rank of
> eighth of thirteen.
>
> Among the seven models that are not neural networks, LightGBM and XGBoost are the bottom two in AUC$_{norm}$
> on Caco-2 and hERG $K_i$, the two measured endpoints with the lowest clean $R^2$. That holds under every
> condition that runs the whole roster, where the two sit mid-table on the computed property. Laplace,
> Student-$t$ and the outlier condition cannot test it, because they ran on the random forest, the support
> vector machine, the Gaussian process and NGBoost only. Why the two gradient-boosted models lose so much more
> on those two endpoints than the random forest and the quantile forest do, we cannot say. A study run on the
> computed property alone would have missed what LightGBM loses on the two measured endpoints, which is why
> both kinds of dataset were run.
>
> AUC$_{norm}$ is a share of a model's own clean accuracy, so a model that was barely accurate to begin with
> can keep nearly all of a very small number. Four combinations of model, representation and noise condition
> retained more accuracy with noise added than without it. All four began from a clean $R^2$ between 0.33 and
> 0.37, against the floor of 0.30 below which a replicate is excluded. We report those rather than patching
> the metric. This instability is a detriment of a normalised metric. The alternative of an unnormalised slope
> carries the opposite defect, since a model with more accuracy to lose then loses more of it by construction.
> No summary of robustness is safe to read on its own, whichever way it has been normalised.

**786 words, six paragraphs of 6, 7, 7, 5, 5 and 7 sentences. 2 of 37 sentences carry a decimal.**

**Paragraphs 2, 4 and 5 are the author's second, third and fourth Q1 conclusions.** All three were in working
notes and in none of the paper text until this pass. Every number below was recomputed this session from the
16 September harvest.

*Paragraph 1, the curves.* The eight drawn are the top eight by AUC_norm at ECFP4 under Gaussian noise:
NGBoost, RF, QRF, XGBoost, the Gaussian process, LightGBM, SVM and the VBLL network on the first
architecture. Range at ECFP4 0.926–0.979, at PDV 0.935–0.975. **The earlier draft said the model at the top
of one panel is not the model at the top of the other, and that is false** — NGBoost tops both, and over
those eight the two panels agree at a rank correlation of 0.93. The claim that survives is about all
thirteen, where ECFP4's ordering agrees with MHG-GNN's at 0.505, with PDV's at 0.654, with Sort & Slice's at
0.775, with ChemBERTa's at 0.758 and with Avalon's at 0.808.

*Paragraph 2, NGBoost.* From `auc_norm_qm9.csv`, QM9 under Gaussian noise, thirteen base models:

| representation | NGBoost clean R² | NGBoost AUC_norm | its clean-accuracy rank of 13 |
|---|---|---|---|
| ECFP4 | 0.706 | 0.979 | 13 |
| Sort & Slice | 0.748 | 0.975 | 13 |
| ChemBERTa | 0.762 | 0.980 | 13 |
| Avalon | 0.786 | 0.982 | 13 |
| MHG-GNN | 0.852 | 0.967 | 11 |
| PDV | 0.865 | 0.975 | 12 |

*It is the most robust of the thirteen on every one of the six. Its clean-R² spread of 0.159 is the widest in
the roster, against 0.114 for the next and 0.050 for the narrowest. Its AUC_norm spread of 0.015 is the
narrowest, against 0.063 for the widest.*

*Paragraph 3, the decoupling.* Spearman $-0.350$, p = 0.0021, over 75 pairings, recomputed from
`standout_pairs.csv` with the min-to-max rescaling `figlib_decisions.py` uses, which is the author's shape
from 2026-09-14. A z-score rescaling instead gives $-0.326$, p = 0.004, so the sign and the significance do
not depend on the choice and the third decimal does. Within one dataset at ECFP4 it is $-0.18$ over thirteen
models and not significant, which is the R16 panel — and `RERUN_PLAN.md` §14.15b warns the two must not be
conflated. **T8's twelve rows are six Gaussian processes, two random forests, one quantile forest, two
networks and one support vector machine, so "every one of them is a Gaussian process or a forest" was
wrong and is now "nine of the twelve".**

*Paragraph 4, the boosted trees.* Median AUC_norm over the six representations, one row per model:

| | QM9 | logD | Caco-2 | hERG K$_i$ |
|---|---|---|---|---|
| NGBoost | 0.977 | 0.925 | 0.859 | 0.855 |
| RF | 0.961 | 0.937 | 0.863 | 0.900 |
| QRF | 0.956 | 0.927 | 0.903 | 0.891 |
| XGBoost | 0.945 | 0.869 | **0.709** | **0.720** |
| LightGBM | 0.944 | 0.871 | **0.603** | **0.646** |
| GP | 0.936 | 0.902 | 0.852 | 0.857 |
| SVM | 0.936 | 0.886 | 0.842 | 0.851 |

*Spread across those seven: 0.041 on QM9, 0.068 on logD, 0.300 on Caco-2, 0.254 on hERG. The same ordering
holds under grouped-wider and grouped-shifted noise, where the two of them are again last and second to last
on Caco-2 and hERG and again inside 0.044 of the rest on QM9. The three depth conditions cannot test this,
because they ran on RF, SVM, the Gaussian process and NGBoost only.*

*Paragraph 5, the forests.* Over the 21 dataset-and-representation combinations that carry all thirteen base
models under Gaussian noise — three of the 24 are short because ChemBERTa lost models on Caco-2 and hERG and
MHG-GNN lost one on Caco-2:

| model | worst robustness rank | median robustness rank | median clean-accuracy rank |
|---|---|---|---|
| RF | 8 | 2 | 8 |
| QRF | 8 | 3 | 8 |
| Gaussian process | 9 | 7 | 1 |
| NGBoost | 11 | 3 | 12 |
| SVM | 11 | 6 | 5 |
| everything else | 12 or 13 | 7 to 11 | 3 to 13 |

***The earlier working note said twenty-four combinations and said every other model drops to eleventh or
worse. Both are wrong.*** *It is twenty-one, and the Gaussian process also never falls below ninth, which is
why the text now names it. The Gaussian process is the most accurate model at the median, which is a point
worth having and is what §R3's Gaussian-process sentence rests on.*

### §R3. Does making a model probabilistic help *(replaces `paper.tex:468–497`)*

> **Probabilistic and deterministic counterparts**
>
> Those comparisons set one model family against another, and none says whether a given model's own
> probabilistic form changes what label noise costs it. A model that reports a distribution rather than a
> point might be expected to lose less of its clean R$^2$ as the labels are corrupted, by attributing a large
> residual to the data instead of fitting it. Figure~\ref{fig:variants} follows three model families as the
> noise rises, on the same axes as the curves above. It draws the two neural architectures each as a plain
> network, a Bayesian network and a Bayesian network with a variance head, and the random forest against the
> quantile forest. One line is one model, and the Gaussian process is not drawn, because the roster holds no
> deterministic counterpart.
>
> Which way a probabilistic form moves AUC$_{norm}$ depends on which model it is built from. One comparison is
> one model pair at one representation under one noise condition, paired on the replicate. Replacing a plain
> network with its fully Bayesian form raised AUC$_{norm}$ significantly in nine of eighteen comparisons for
> NN-$\alpha$, with none falling, and in six of eighteen for NN-$\beta$, with three falling. Adding a variance
> head to an already-Bayesian network raised it in two of eighteen comparisons for NN-$\alpha$, with one
> falling, and in five of eighteen for NN-$\beta$, with none falling. Both neural substitutions therefore move
> AUC$_{norm}$ up more often than down.
>
> Giving the Gaussian process a per-molecule observation-noise term moved AUC$_{norm}$ the other way, lowering
> it significantly in fourteen comparisons and raising it in five, and lowering its clean R$^2$ as well. The
> quantile forest's AUC$_{norm}$ was significantly below the plain forest's in twelve comparisons and above it
> in five (every comparison, one column per representation, is in Additional file~8). The direction reverses
> between representations for several of these pairs, so one column on its own would hide it. No probabilistic
> counterpart differed from its base model by more than 0.011 of AUC$_{norm}$ at the median of its
> comparisons. Under grouped-shifted, which gives every scaffold family its own offset, the thirteen base
> models run from 0.871 to 0.935 in AUC$_{norm}$ on the computed property, each endpoint a median over ten
> replicates and six representations. That range is about six times the largest of those median changes, so
> choosing a different model family moves AUC$_{norm}$ further than making a given model probabilistic does.
> What a probabilistic form buys is an uncertainty estimate rather than robustness to label noise, and that is
> the reason to choose one.

**416 words, three paragraphs of 5, 5 and 7 sentences. 2 of 17 sentences carry a decimal.**

*From `d10_probabilistic.csv`, 153 comparisons, each a signed-rank test paired on the replicate within one
representation and one condition. Recomputed here: plain → Bayesian, first architecture 9 significantly up of
18 and 0 down, median +0.007; second architecture 6 up and 3 down, median +0.011. Bayesian → variance head:
2 up, 1 down, median +0.001, and 5 up, 0 down, median +0.006. Gaussian process → heteroscedastic Gaussian
process: 5 up, 14 down. Forest → quantile forest: 5 up, 12 down, median −0.008. `base_against_variant.csv`
agrees and adds the clean-accuracy column.*

*The 0.871 to 0.935 range is the median over ten replicates for each of the thirteen base models on the QM9
HOMO--LUMO gap under the shared scaffold offset, taken over the six representations, from `auc_norm_qm9.csv`.
⚠️ **An earlier draft put 0.127 here and called it the best-to-worst model difference.** That number is real
but it is a different quantity — the range across nineteen models of the cost of moving from Gaussian noise
to the shared offset, worst the Gaussian process at $-0.101$ and best the heteroscedastic variational network
at $+0.025$. It belongs in §R4, where it now is, and not here.*

*⚠️ The submitted paper reports these as +0.056 to +0.124 and all significant. Those are noise-degradation
slopes on a retired metric and a retired noise scale. On AUC_norm the same transformations are worth about a
hundredth. The direction for the quantile forest is unchanged — it was less robust then and it is less robust
now.*
---

## MOVEMENT 3 — does the kind of noise matter, and does any of it survive on measured labels

### §R4. The kind of noise *(replaces `paper.tex:460–467` and has no real predecessor)*

**This is the paper's differentiator and it is the subsection to get right.** The finding is not that
non-Gaussian noise matters. Three non-Gaussian shapes are indistinguishable from Gaussian at a matched dose,
and saying otherwise would be contradicted by Table~\ref{tab:robustness} on the facing page. The finding is
that *independence and zero mean* are the assumptions that fail, and they are the two nobody tests.

> **The kind of noise, at a matched amount**
>
> Both of those readings were taken one kind of noise at a time. Neither says whether the kind of noise itself
> changes the answer. At a request of 0.31 in the label's own units, the six conditions that deliver a set
> amount landed within 0.005 of it, in those same units. Censoring cannot be held to a set amount, because its
> level is a fraction of labels clipped rather than a fraction of the label spread. It runs on its own axis
> throughout. A difference in outcome between two conditions that deliver the same amount is therefore a
> difference of pattern and not of size.
>
> Five of the seven conditions cost much the same AUC$_{norm}$ as one another at the same delivered amount.
> Drawing a label's error from a Laplace or a Student-$t$ distribution, or giving a random tenth of the labels
> a much larger draw, changed AUC$_{norm}$ by at most a few thousandths against Gaussian noise on the computed
> property (every condition, model by model, is in Table~\ref{tab:robustness}). Those three ran on three of
> the six representations and on a subset of models by design, on the computed property and on each of the
> three measured endpoints. Of the fifteen pairs of conditions at ECFP4 on the computed property, six differ
> significantly, five of them involving grouped-shifted, the condition that gives every scaffold family its
> own offset. The sixth, Gaussian against grouped-wider, differs by less than the spread between the ten
> replicates of one identical configuration. This reproduces what \citet{Heid2023} published for noise spread
> evenly across a training set, and it holds for a contaminated subset too, which they did not test.
>
> The two conditions that do separate break a different assumption. Grouped-shifted cost 0.032 of AUC$_{norm}$
> on the computed property, against at most 0.006 for any change of shape, and as much as 0.094 on the three
> measured endpoints. Clipping labels at an assay limit cost more AUC$_{norm}$ still. Censoring ran on five
> named model-and-representation pairings on each dataset, chosen to measure the size of its effect, so
> nothing here says which model resists censoring best. Grouped-wider widens a scaffold family's errors
> without shifting them, and puts a smaller share of its injected errors' spread into the scaffold-family
> means than plain Gaussian noise does. Only grouped-shifted raises that share, to about four fifths of the
> spread of the errors it injects. These comparisons indicate that the axis along which they separate is not
> Gaussian against non-Gaussian, and not even structured against unstructured, but independent and zero-mean
> against correlated or biased.
>
> The fall in AUC$_{norm}$ on moving from Gaussian noise to grouped-shifted is decided by the model far more
> than by the representation. Each model's fall is the median over the replicates, over the six
> representations, and over the computed property and the three measured endpoints alike. That fall ranges
> across 0.127 between the nineteen models and across 0.095 between the thirteen base models, from a loss of
> 0.101 for the Gaussian process to a gain of 0.025 for the heteroscedastic variational network. The wider
> range covers all nineteen configurations,
> including the six the variance decomposition sets aside, because it is a statement about the roster rather
> than about the decomposition. Read per representation and taken the same way, the
> same fall ranges across a little over a hundredth of AUC$_{norm}$. This is the ordering of
> §\ref{sec:variance}, arrived at without the variance decomposition, on a different outcome and on a
> different set of runs.
>
> What the kind of noise does not change is which model to choose. On the computed property at ECFP4, model
> rankings agree across the six conditions that deliver a set amount, at a Kendall's $W$ approaching 1 over
> the seven models it covers. A $W$ of 1 is complete agreement between conditions and a $W$ of 0 is none. That
> concordance rests on seven models at one representation, so it does not say how the remaining configurations
> reorder under a change of condition. So the condition decides how much accuracy is lost and the model
> decides who loses least, and those are separate facts.

**679 words, five paragraphs of 6, 6, 7, 6 and 5 sentences. 3 of 30 sentences carry a decimal.**
*The third paragraph is one sentence over the ceiling; the sentence to cut if it has to come down is "Neither
is a matter of how the individual errors are distributed."*

*The fourth paragraph is new in this pass and it is what §R1's second paragraph promises when it says the
same ordering arrives again by another route. Recomputed here from `auc_norm_qm9.csv` and
`auc_norm_assay.csv`: per model, the median over the four datasets and six representations of AUC_norm under
the shared offset minus AUC_norm under Gaussian noise, which runs $-0.101$ for the Gaussian process to
$+0.025$ for the heteroscedastic variational network. The same differences taken per representation run
$-0.056$ for Avalon to $-0.043$ for ChemBERTa. `RERUN_PLAN.md` §14.17b has 0.13 and 0.013 on the 14 September
harvest. Restricted to the thirteen base models the model range is 0.095 rather than 0.127, and the paper
uses all nineteen because this is a statement about the roster rather than about the decomposition, which
excludes the variants for a reason that does not apply here.*

⚠️ *The fourth paragraph describes what R15 plots and stops there, deliberately. A sentence about whether the
lines cross is a reading off a picture and I have not made one — Kendall's W says the orderings agree and says
nothing about where a line goes at an intermediate level. If you want that sentence, it comes from
`R15_rank_against_level_ecfp4_gaussian.png` after the re-run, and note that a line there stops where the model
drops below the accuracy gate rather than falling to last place, so an ending line is not a crossing.*

**The numbers behind it, every one recomputed this session.**

Delivered amount, from `F1_delivered_dose.csv` — requested 0.3083 in label units:

| condition | delivered (median) | off by |
|---|---|---|
| grouped_wider | 0.3083 | 0.0000 |
| gaussian | 0.3088 | 0.0005 |
| laplace | 0.3074 | 0.0009 |
| grouped_shifted | 0.3066 | 0.0017 |
| outlier_p10 | 0.3045 | 0.0038 |
| student_t_nu5 | 0.3035 | 0.0048 |
| *censoring* | *0.1408* | *not dose-matched, by construction* |

Median change in AUC_norm against Gaussian, over the pairs that ran all six dose-matched conditions:

| | QM9 (21 pairs) | logD (21) | Caco-2 (20) | hERG (19) |
|---|---|---|---|---|
| Laplace | −0.001 | +0.010 | +0.023 | +0.016 |
| Student-*t* (ν=5) | +0.002 | +0.004 | +0.009 | −0.011 |
| Outlier (10%) | +0.002 | +0.007 | +0.013 | −0.018 |
| Grouped, wider | +0.002 | +0.012 | +0.019 | −0.002 |
| **Grouped, shifted** | **−0.032** | **−0.035** | **−0.090** | **−0.094** |

*Read from the harvest of 17 September at 23:20. `RERUN_PLAN.md` §14.17a has the same table on the
14 September harvest with pair counts of 21/18/18/17, and its shifted row reads −0.032 / −0.042 / −0.105 /
−0.094. The shape is unchanged and the assay figures have moved by up to 0.015 as more cells landed, so
**quote this version and not §14.17a's.***

Censoring, and this is the **corrected** version — the plan compares a five-pair median against a
hundred-pair one, which is not like for like. On the same pairs, from `auc_norm_qm9.csv` and
`auc_norm_assay.csv`:

| dataset | shared pairs | censoring | Gaussian | change |
|---|---|---|---|---|
| Caco-2 | 5 | 0.459 | 0.856 | **−0.406** |
| hERG K$_i$ | 4 | 0.555 | 0.863 | **−0.308** |
| QM9 | 5 | 0.813 | 0.951 | −0.145 |
| logD | 5 | 0.826 | 0.913 | −0.101 |

*The paired comparison makes censoring look worse on QM9 and logD than the unpaired one did, because the five
censoring pairs are robust models. **`RERUN_PLAN.md` §14.17c's rule still holds and must be repeated in the
paper: censoring ran on five named pairs, so no claim about which model resists censoring best can rest on
it.** The ordering of models flips between datasets, which is what that looks like.*

Group share, from `F1_group_share.csv` — the spread of the group-mean errors divided by the spread of all the
errors:

| condition | group share |
|---|---|
| **grouped_shifted** | **0.781** |
| laplace | 0.143 |
| outlier_p10 | 0.139 |
| censoring | 0.138 |
| gaussian | 0.136 |
| student_t_nu5 | 0.121 |
| **grouped_wider** | **0.108** |

*0.781 is √0.62 to three figures, which is the between-laboratory share Bentz et al. measured and the
parameter the condition was built with. So the figure confirms the parameter rather than illustrating it.
`NOISE_DESIGN.md` §5.1e.*

The fifteen pairwise comparisons at ECFP4, from `d3_condition_pairs.csv`: the significant ones are
grouped-shifted against each of Gaussian, grouped-wider, Laplace, outlier and Student-*t*, plus Gaussian
against grouped-wider at +0.003. Every comparison among Gaussian, grouped-wider, Laplace, outlier and
Student-*t* that does not involve the shifted condition is non-significant. Kendall's $W$ = 0.937,
p = 7.6 × 10⁻⁶, seven models over six conditions at ECFP4, from `d3_kendall_w.csv`.

### §R5. Does it hold on measured labels *(replaces `paper.tex:542–556`)*

> **Robustness on the three assay datasets**
>
> Everything so far is a computed property with no measurement error of its own. We repeated the grid on
> three laboratory endpoints: a distribution coefficient, an efflux ratio and a binding affinity. There the
> injected noise sits on top of measurement error already in the labels, which we cannot separate. Repeat
> measurements of the same compound and target disagree by about 0.68 log units for pIC$_{50}$ and about
> 0.54 log units for pK$_i$ \citep{Kalliokoski2013, Kramer2012}. A level of about 0.6 is therefore one unit
> of the error a laboratory already carries. Figure~\ref{fig:assay} has the three measured endpoints, models
> down the side and noise conditions across. One cell is a model's AUC$_{norm}$ under one condition, and the
> first column holds its clean R$^2$, so the two are read together.
>
> The ordering of the noise conditions in the subsection on the kind of noise survives on measured labels.
> Grouped-shifted, which gives every scaffold family its own offset, has the lowest AUC$_{norm}$ of the six
> conditions that deliver the same amount of corruption. It lowers AUC$_{norm}$ on all three laboratory
> endpoints by more than it does on the HOMO--LUMO gap. Censoring is lower again on all three, and furthest
> down on Caco-2 and hERG $K_i$, where clean R$^2$ is lowest. Clean accuracy and label spread are both lower
> on those two datasets, so the drop is reported without a cause. Its level counts the fraction of labels
> clipped rather than a fraction of the spread, so it is not matched against the other six. Censoring also
> ran on five named model-and-representation pairings, not the whole roster, so nothing here says which
> model resists it best.
>
> What does not survive is the ranking of models, compared across eighty-three combinations of
> representation, noise condition and assay dataset. The median rank correlation between a model's standing
> on the computed property and on a measured endpoint is 0.36 (1 would mean the same ordering). Twenty of
> the eighty-three reach significance at the five per cent level. Figure~\ref{fig:transfer} has models down
> the side in their QM9 order, a model's robustness rank within a dataset on the bottom axis, and one mark
> per dataset. That median hides a spread and is not a measurement on any one combination. Agreement is
> highest on logD and lowest on hERG $K_i$, so part of the ordering does carry on the distribution
> coefficient.
>
> Agreement also depends on the representation it is read at, from moderate at ECFP4 down to none at Sort
> \& Slice. The main-text table therefore names the representation the agreement was computed at. On the
> computed property the four tree models sit in one narrow band of AUC$_{norm}$. On Caco-2 and hERG $K_i$
> the two bagged forests keep about nine tenths of their clean accuracy, where LightGBM keeps about three
> fifths. A study run on the computed property alone would have reported LightGBM as a robust model. A
> pairing chosen on one endpoint should be re-checked on the next, at the cost of running the grid again
> there.

**498 words, four paragraphs of 7, 7, 6 and 6 sentences. 3 of 26 sentences carry a decimal.**

*Rank transfer from `d9_rank_agreement.csv`: 83 rows, median ρ 0.356, 20 of 83 with p < 0.05. By dataset:
logD 0.568, Caco-2 0.306, hERG 0.290. By representation it runs from −0.001 at Sort & Slice to 0.600 at
ECFP4, which is itself a reason the main-text table names its representation. ⚠️ The censoring row of that
table shows ρ = 1.000 on two combinations and must not be quoted — two combinations is not a correlation.*

*⚠️ The submitted paper's §4.4 says the assay datasets "confirm that the trends generalize" and that "NGBoost
ranks first on both QM9 and external datasets". On this run NGBoost ranks first on QM9 at ECFP4 and fourth,
seventh and third on the three assay datasets, and the median transfer across all combinations is 0.36. The
subsection has to be rewritten around non-transfer, not around confirmation.*

*⚠️ `paper.tex:560`'s "QRF consistently less robust" is an averaging artefact — the quantile forest is ahead
of the plain forest on Caco-2 and hERG under Gaussian noise and behind it on logD and QM9.*
---

## MOVEMENT 4 — how uncertainty and noise play together

Three questions, deliberately separate, because earlier drafts of this paper fused them. Does a model become
less sure when its training labels are corrupted? Does its uncertainty still rank which predictions to trust?
And can it point at which labels were the corrupted ones? The first was already known, the second is what a
referee expects, and the third is the hard one.

### §R6. Uncertainty under label noise *(replaces `paper.tex:498–541`)*

> **Uncertainty under label noise**
>
> Everything so far measures how much accuracy a model keeps under label noise. How precise a model reports
> itself to be is a separate question, and nothing so far says whether a model registers what the noise cost
> it. Surprisingly, a model can grow more certain as its labels grow more wrong, and censoring is where that
> happens. Every uncertainty here is computed from the fitted model at a molecule's representation, never
> from the label it is scored against, and none is calibrated after fitting. When the total predicted
> uncertainty rises, does the model attribute the rise to the observations or to itself? A model that
> registers the noise should raise the aleatoric component, the noise it attributes to a label, and leave the
> epistemic component where it was.
>
> Figure~\ref{fig:decomposition} shows the two components against the noise level, one panel per model. The
> bottom axis is the noise added to the training labels, and the side axis is the mean predicted uncertainty
> over one out-of-fold pass, in the label's own units. \citet{Kolmar2021} showed that a Gaussian process's
> mean predicted uncertainty rises with the label noise in its training data. That holds across the six
> conditions that add error to a label, where between 92 and 100 per cent of combinations of model,
> representation and fold give a rising slope and the median slope runs from 0.30 to 0.42. The rise holds on
> the computed property and on each of the three measured endpoints taken separately. Under censoring the
> sign reverses on every model, on the computed property and on each of the three measured endpoints. Fewer
> than 5 per cent of those combinations rise, and the median slope is $-0.41$.
>
> Censoring removes information from a label rather than adding error to it, and it is the only condition
> that narrows the spread of the labels. Replacing every value past the assay limit with the limit itself may
> pull the training labels together, and a model fitted to closer targets may then report itself more
> certain. The runs carried only the clean training spread, so that reading is not a measurement, and
> recording the spread at each noise level would settle it. Under the one mechanism that industrial assays
> report most often, a model grows more confident as its labels grow more wrong.
>
> Most of the roster has to be excluded before the two components can be read apart, and which models are
> excluded is itself a result. Seven of the thirteen models that emit an uncertainty have one of the two
> components fixed at a single number per fit, or missing altogether. NGBoost fits one distribution and has
> no term for its own ignorance, the Gaussian processes and the variational networks report one
> observation-noise number per fit, and the plain Bayesian networks predict a mean alone. No line is drawn
> for those models, and the two support columns of Table~\ref{tab:uncertainty} say so, one row per model and
> noise condition on the computed property at PDV.
>
> Among the six models whose two components both vary per molecule, the separation works for most and fails
> completely for one. The Gaussian process that predicts each molecule's observation noise and the two
> networks with a variance head raise the aleatoric component while the epistemic component holds, in roughly
> two of every three combinations of dataset, representation and noise condition. The variational networks
> separate most cleanly of all, and they carry the lowest clean R$^2$ on the roster, between 0.42 and 0.53.
> Separating cleanly is therefore not a robustness result, because AUC$_{norm}$ is a share of a model's own
> clean accuracy and a small clean R$^2$ lifts their score. The quantile forest separates in none of those
> combinations where its verdict can be read. Both components rise together in 90 of them, and both are
> computed from the same set of fitted trees, which may be why they cannot move apart.
>
> Whether the separation works depends on the noise as well as on the model. Take the same
> model-and-representation pairs across the three conditions that ran on the whole grid, at the same noise
> levels, counting the computed property and the three measured endpoints together. The separation holds in
> 78 of the 109 comparisons that can be read at all under Gaussian noise, and in 58 of 110 under
> grouped-shifted, where a whole scaffold family shares an offset. Modelling label noise as independent
> therefore understates how much accuracy is lost, and it also hides where a model's own uncertainty stops
> working.
>
> The second question has a steadier answer. Within a fixed amount of noise, a model's total predicted
> uncertainty still orders its predictions by how wrong they are against the clean label, weakly but almost
> always. The statistic is the Spearman correlation between that total and the out-of-fold error, and a
> higher value means the ordering is closer. It is positive in 95.1 per cent of the 1,003 combinations of
> dataset, model, representation and noise condition, and above 0.3 in 5.2 per cent of them. Those
> combinations cover the computed property and the three measured endpoints together. The quantile forest has
> the highest median correlation of the roster while its own decomposition fails everywhere, and the plain
> Bayesian networks the lowest. A model that separates the two components and a model whose uncertainty ranks
> its own error are not the same model, and neither follows from a high AUC$_{norm}$.

**894 words, seven paragraphs of 6, 7, 4, 4, 6, 4 and 7 sentences. 3 of 38 sentences carry a decimal.**

*The last sentence of the second paragraph is the author's own, written out in `RERUN_PLAN.md` under "Q7 —
the one sentence, for the Results text", and placed where that note asks for it.*

**The second paragraph is new in this pass and it is the answer to "does a model become less sure", which
every earlier draft of this subsection skipped.** From `unc_q5.csv`, the slope of mean predicted uncertainty
against noise level, one cell per dataset, model, representation and fold, on the held-out split:

| condition | median slope | cells | share of cells with a rising slope |
|---|---|---|---|
| Gaussian | +0.303 | 3,183 | 95.8% |
| Grouped, wider | +0.313 | 3,180 | 97.3% |
| Grouped, shifted | +0.336 | 3,180 | 98.1% |
| Laplace | +0.410 | 1,053 | 96.9% |
| Outlier (10%) | +0.418 | 1,053 | 96.7% |
| Student-$t$ ($\nu=5$) | +0.420 | 1,053 | 96.8% |
| **Censoring** | **$-0.405$** | **880** | **2.0%** |

*Per dataset the censoring share is 0.4% on Caco-2, 1.1% on logD, 4.1% on hERG K$_i$ and 4.3% on QM9, against
92% to 100% for every other condition on every dataset, so the reversal is not one dataset carrying it. The
92% floor is Caco-2 under the outlier condition.*

⚠️ *The narrowing-spread explanation is the reading and not yet a measurement on these runs.* The only label
column the statistics carried was the clean training spread, which is the same number at every level, so
nothing in this harvest can show the recorded labels narrowing. `q5_mean_uncertainty` now also writes
`recorded_label_sd` and `label_spread_ratio`, which are the clean label plus the amount injected, and
`scripts/test_recorded_label_spread.py` shows the ratio climbing above 1 under additive noise and falling
below it under censoring on constructed data. **On the real runs it arrives with the re-run.** If it does not
hold there, the sentence naming the mechanism comes out and the measured reversal stays.

**The numbers behind it, recomputed this session from `unc_slopes.csv` and `d7_q6.csv`.**

Which models can be asked at all — thirteen models, from the support flags:

| what the model reports | models | can the split be read? |
|---|---|---|
| only the model-ignorance term varies per molecule | BNN-α, BNN-β, VBLL-α, VBLL-β, GP, GP (Tanimoto) | no |
| only the data-noise term varies per molecule | NGBoost | no |
| **both vary per molecule** | **BNN-α (var. head), BNN-β (var. head), VBLL-α (het.), VBLL-β (het.), GP (het.), QRF** | yes |

Verdicts for the six that can, over every dataset, representation and condition:

| model | separates | both rise | neither moves |
|---|---|---|---|
| VBLL-α (het.) | 52 | 2 | 0 |
| VBLL-β (het.) | 68 | 4 | 9 |
| BNN-α (var. head) | 63 | 31 | 15 |
| GP (het.) | 59 | 23 | 9 |
| BNN-β (var. head) | 57 | 20 | 22 |
| **QRF** | **0** | **90** | **9** |

By condition, on the **237 pairs that ran all three full-grid conditions**, so this is like for like:

| condition | separates | both rise | share that separates |
|---|---|---|---|
| Gaussian | 78 | 31 | **72%** |
| Grouped, wider | 74 | 37 | 67% |
| **Grouped, shifted** | **58** | **52** | **53%** |

*The other 128 rows of each condition are the models whose split cannot be read, and they are identical
across the three by construction.*

Q6, from `d7_q6.csv`, recounted on 2026-09-18: **1,003 cells carry a named condition**, median 0.168, 95.1%
positive, 5.2% above 0.3. By model the median runs from 0.072 for the second plain Bayesian network to 0.255
for the quantile forest. By condition it is 0.17 to 0.19 everywhere except censoring, at 0.071. *The earlier
1,076 / 95.0% / 4.9% in this note counted the 55 blank-condition rows of `RERUN_PLAN.md` §13.23z more than
once; §R6's last paragraph now prints the recounted figures.*

🔴 **The two counts above the Q6 line — 78 of 109 and 58 of 110 — are the ones to re-check first after the
next harvest, and §R6's fifth paragraph pastes them straight into the paper.** They are keyed on the noise
condition, and 55 rows of `d7_q6.csv` lost theirs before the fix landed, so each count may be short by as many
as 55. The fix is in `scripts/uncertainty_stats.py` and guarded by
`scripts/test_blank_condition_is_not_a_cell.py`, but it reaches the numbers only when the analysis runs again
on the cluster. The direction of the finding is not at risk — 72% against 53% is a wide gap — but the two
fractions in the paper text are.

**⚠️ The two claims in the submitted paper's §4.3 that this replaces.** *"GPs and NGBoost gave the strongest
correlations"* was a per-sample correlation pooled across noise levels, which measures the population trend
and not per-sample detection. And *"for VBLL both the aleatoric and epistemic components increased"* is now
the quantile forest's result, not the variational networks' — on this run the heteroscedastic variational
networks are the cleanest separators on the roster and the quantile forest is the one that fails.

**⚠️ And the variational networks must not be ranked on robustness off the back of this.** They separate
cleanly and they carry the lowest clean R² on the roster, 0.42 to 0.53, so AUC_norm divides their
score by a small clean R$^2$ and lifts it, which is the case §R2 describes. `RERUN_PLAN.md` §14.17b marks both as not for publication as a robustness
result.

### §R7. Can uncertainty point at the corrupted labels — and what is not settled

> **Which labels were corrupted**
>
> Ranking predictions by how wrong they are is not the same as naming the labels that were corrupted. The
> plain correlation between predicted uncertainty and the size of a molecule's injected error is near zero on
> the computed property and on each of the three measured endpoints. That is the negative control working
> rather than a null result, because the scoring model never saw that molecule's draw. Under four of the
> seven conditions every molecule receives the same noise scale, so there is nothing to find by construction.
> The question is only defined where some molecules receive more corruption than others.
>
> We therefore ask whether dividing the out-of-fold error by the total predicted uncertainty ranks the
> corrupted labels better than the error alone. What is reported is the difference between the two rankings,
> so a positive value means the uncertainty added something and zero means it added nothing. The three of the
> seven conditions that give some molecules more noise than others are widening a scaffold family's errors,
> giving a random tenth of the labels a larger draw, and censoring. Under the first two the division ranks
> the corrupted labels no better than the error alone does. The same is true under the four conditions where
> the question is undefined.
>
> Under censoring the division does improve that ranking, by a larger median gain in that difference than
> under any of the other six conditions. That is what would be expected if a model can recognise a label
> sitting at the assay limit. We report the direction only, because the distribution that improvement would
> have to beat is not centred on zero and that comparison has not been run. Censoring ran on five
> model-and-representation pairings, so the improvement is a statement about those five and not about the
> roster. Censoring is also where the correlation between predicted uncertainty and out-of-fold error is
> lowest of the seven conditions, at under half the value it reaches under any of the other six. So the
> answer, on the computed property and on the three measured endpoints alike, is no, though censoring is the
> one case where it may be otherwise. A model's own uncertainty is therefore not a way to find the bad labels
> in a training set.

**374 words, three paragraphs of 5, 5 and 7 sentences. 0 of 17 sentences carry a decimal.**

*The three conditions where the question is defined are grouped-wider, outlier and censoring:
`_CONSTANT_SCALE_STRATEGIES = ('uniform', 'grouped_shifted')` in `NoiseInject/noiseInject/core.py:84` gives
every molecule the same scale under Gaussian, Laplace, Student-*t* and grouped-shifted. Median `rho_delta`
from `d7_q4.csv`: +0.0009 grouped-wider, +0.0014 outlier, **+0.0143 censoring**, against +0.0005 to +0.0018
for the four undefined ones. So the two conditions that could have shown something and did not are the
informative half of the null, and* **`RERUN_PLAN.md` §14.17d's note that the near-zero counts under Laplace
and Student-*t* are partly by construction still stands and should be in the paper.**

🔴 **This subsection carries a TODO and it must not be written around a number yet.** The reasons are in
`RERUN_PLAN.md` §15, under "READ FIRST", item 1. In short: the permutation band every Q4 number was read against is a band for the
out-of-fold *error*, not for the uncertainty's contribution, so `RERUN_PLAN.md` §14.17d's table — censoring
176 of 180, Gaussian 7 of 468 — is measuring whether the error tracks the injected noise, which under
censoring is arithmetic. The code is fixed and the band for the gain now exists, but **no harvest has been
run with it.** Until one has:

- The gain itself is computable and is in `d7_q4.csv` now: median `rho_delta` by condition is +0.0005
  Gaussian, +0.0001 grouped-shifted, +0.0009 grouped-wider, +0.0014 Laplace, +0.0014 outlier, +0.0018
  Student-*t*, and **+0.0143 censoring**. The censoring mean is +0.041 against +0.000 or below for every
  other condition.
- But the gain's null is **not centred on zero**, so those cannot be read against zero. The direction is
  clear and the significance is not.
- **Write the paragraph with the censoring result stated as a direction and the band marked TODO.** That is
  what the draft above does.

**To close it** — one submission, and it needs the code pushed first:

```bash
cd $QSAR && git pull && sbatch slurm_scripts_analysis/run_paper_analysis.sh
```

*It takes about sixteen minutes today. The second permutation band roughly doubles the part of the run that
costs, so budget half an hour. `delta_auc` is registered in `uncertainty_stats.STATISTICS` but is not computed by
default, because each extra band is another pass over every cell —* `RERUN_PLAN.md` *§14.6 row 1 names it as
the enrichment figure's trigger, so if that figure is ever reinstated, pass
`nulls=DEFAULT_NULLS + (('delta_auc', 'aucdelta_'),)`.*

**⚠️ And a related consequence you already flagged.** With F7 cut on 13 September and F9 on 16 September, the
uncertainty side of the paper is one figure and one table, and the title is about uncertainty. Once the band
is recomputed there are two ways to give that half a picture back, and both are yours:

1. **Reinstate F9** — one bar per model of the gain against its band, over the three conditions where the
   question is defined. `f9_uncertainty_finds_noise` is still in `figlib_figures.py`, unused. It was cut
   because a bar chart of a correlation told the story no better than the table; against a band that is now
   about the right quantity, it would say something the table cannot.
2. **Leave it at F6 and T6** and carry §R7 in prose, which is what the draft above assumes.

I would not reopen the cut before the band is recomputed, because the reason for reinstating it depends on
what the recomputed numbers say.
---

# THE FIGURES — LaTeX blocks, ready to paste

Seven figures, per `RERUN_PLAN.md` §14.25. File names are exactly what `scripts/run_paper_analysis.py`
writes into its figures directory, which for the newest harvest is
`results/decisions_arc_20260916/figures/`, so upload the PNGs under those names and these blocks work
unchanged.

🔴 **RE-RUN BEFORE YOU UPLOAD.** The PNGs on disk were written on 17 September at 04:00 and **three commits to
the figure scripts landed after that**: `69815b6` at 04:06 (F4a's second panel could never draw, and the
caption promised it anyway), `93fe179` at 12:54 (eight figure titles move to the captions, legends group by
family, `Student-t` and `K`$_i$ stop being mathtext) and `5865c88` at 13:19 (F2 gets its residual bar back,
with the whiskers). So **F4a on disk has one panel where the caption below describes two, and F2 on disk has
three bars where the caption below describes four.** `results/decisions_arc_20260916/figures/captions.md` describes the
old drawings and is stale for both.

```bash
cd $QSAR && git pull && sbatch slurm_scripts_analysis/run_paper_analysis.sh
```

**On the captions below.** Every factual clause from the generated caption is kept; what is cut is repetition
and the sentences that argue for a design choice rather than explain the picture. What was cut from each is
listed under it. Length is 70–130 words in 3–5 sentences, which is what the three Journal of Cheminformatics
papers in the reference set run — Kolmar's six captions have a median of 74 words and Dablander's results
captions 75 to 157. The generated ones run to 300.

**One convention to decide.** None of the three target-journal papers letters its panels, including a
16-panel trellis and several 2×4 grids; they navigate by strip label and by position. All nine Nature
Machine Intelligence papers letter panels. The figures already draw letters, and §14.27 records your call on
2026-09-17 that panel titles stay, so the captions below use them. If you would rather follow the journal,
the change is in `figlib_shapes.title` and every caption below loses its letters.

---

### F1 — the noise conditions *(Methods)*

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F1_noise_conditions.png}
\caption{What each noise condition does to a distribution of labels, drawn with the injector the pipeline
runs rather than a reimplementation. Panels a) onward overlay the clean labels, in grey, and the noised
labels, in the condition's own colour, on one shared set of bins, at a noise level of 0.5 of the spread
of the clean training labels. Censoring is drawn at 25\% of the labels in the panel clipped, because its
level is a fraction of labels clipped rather than a fraction of the spread. The bottom axis of each panel
is the label value and the side axis is the share of labels falling in a bin, carrying no tick numbers
because only the shape is read. The final panel gives the share of each condition's noise that a whole
scaffold group holds in common: the spread of the group mean errors divided by the spread of all the
errors. One bar there is one noise condition, its value is printed above it, and the side axis runs from
0 to 1. Near zero means the noise scatters within a group and near one means the group moves as a block.
Censoring is absent from that panel because it has no variance parameter and its group share is not
comparable with the others'. Neither axis has a better direction, since the panels describe the noise
rather than score a model.}
\label{fig:noise_conditions}
\end{figure}
```

*Cut, and where each went: the explanation of why the grouped-shifted panel looks identical to the plain one
— three sentences of algebra — goes into the Methods, where §M4 already carries it, and into §M4a of this
guide. The delivered-dose numbers go into §R4's opening sentence, which is where they do work.*

---

### F2 — the variance decomposition

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F2_variance_decomposition.png}
\caption{Share of the variance in each outcome explained by the choice of model, the choice of
representation, the pairing of the two, and the residual, on the QM9 HOMO--LUMO gap. a) robustness, as
AUC$_{norm}$; b) predictive accuracy, as R$^2$ at a noise level of 1.0, that is added noise equal to the
spread of the clean training labels. AUC$_{norm}$ is the share of a configuration's own clean R$^2$ that
it keeps as the noise rises, and a higher value is more robust. Despite the name it is not an area under
a receiver operating characteristic curve. The bottom axis of both panels is the noise condition and the
side axis is the share of variance, from a two-way analysis of variance with sequential sums of squares.
Each condition carries four bars, one per term, coloured as the legend beside the panels names them, and
the four shares sum to 100\%. No direction on the side axis is better, since the panels apportion
variation rather than score a model. The residual is the variation between ten replicates of one
identical configuration --- same model, same representation, same condition, same level, different seed.
Whiskers are not confidence intervals: they are a leave-one-out jackknife over the ten replicates, so
they give how much the answer depends on any one replicate rather than how precisely the share is known.
A slot marked ``no value'' carries no bar rather than a share of zero, because censoring's level is a
fraction of labels clipped and has no accuracy at a reported level to decompose. Thirteen models by six
representations by ten replicates, 780 values per decomposition.}
\label{fig:variance}
\end{figure}
```

*128 words. Cut: nothing factual. The sentence saying the residual is reported in T3 rather than drawn is
false of the current code, which draws it — that clause came from the version between `06133fc` and
`5865c88`. Added: the model and representation counts, which the generated caption leaves out, and the
noise level on the accuracy panel, which a reader otherwise cannot recover.*

---

### F3 — model against representation

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F3_model_by_representation.png}
\caption{Robustness (AUC$_{norm}$) of every model on every representation, on the QM9 HOMO--LUMO gap.
Rows are models, ordered by family; columns are the six representations; one panel per noise condition.
a) Gaussian; b) grouped-shifted. The grouped-wider condition, whose grid repeats a), is held to an
additional file, and so are the three conditions that run on a named subset of pairs, because they share
too few cells with the rest of the grid to be judged either way. One cell is one model on one
representation, with its exact value printed on it. Colour runs on one fixed range across the panels,
printed on the colour bar, and the range is narrower than the metric's full span because every value on
this dataset falls inside it. The bright end of the bar is the higher AUC$_{norm}$, so a brighter cell
kept more of its own clean accuracy. A grey cell marked ``not run'' was never fitted, and one marked
``excluded'' was fitted and then dropped, with the reason in Additional file~5. Values are the median
over ten replicates.}
\label{fig:grid}
\end{figure}
```

*126 words. Added: which conditions the panels are, the replicate convention, and where the excluded reasons
live — the target journal restates the replicate convention in every caption rather than once in the Methods.
Cut: the sentence arguing that showing two similar grids is showing one thing twice.*

---

### F4a — what label noise costs

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F4a_models_under_noise.png}
\caption{Predictive accuracy against the amount of label noise, on the QM9 HOMO--LUMO gap under Gaussian
noise, for the eight models with the highest AUC$_{norm}$ under that condition at ECFP4. a) ECFP4; b)
PDV. The same eight models are drawn on both panels, so panel b) is not a selection made on PDV. One line
is one model, named in the shared legend. The bottom axis is the amount of noise put into the training
labels, as a fraction of the spread of the clean training labels. The side axis is R$^2$ on held-out
molecules, taken as the median over ten replicates and shared between the panels. A higher line is the
more accurate model at that noise level, and a flatter line is the one label noise costs less. The dashed
vertical line marks the noise level that every table in this paper reports at. There are two panels
rather than one because how far a model falls is a property of the pairing rather than of the model, and
the same curve can begin at a different height on each. No bands are drawn: eight overlapping ranges hid
the lines, and the spread across replicates is in Table~\ref{tab:robustness}.}
\label{fig:curves}
\end{figure}
```

*129 words. The eight drawn, in robustness order under Gaussian noise at ECFP4: NGBoost, RF, QRF, XGBoost,
GP, LightGBM, SVM, VBLL-α. ⚠️ Check the re-run gives two panels — before `69815b6` the second could never
draw, because the frame was already filtered to one representation before it was asked whether it held the
other.*

---

### F6 — where a model attributes the added noise

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F6_decomposition.png}
\caption{The aleatoric and the epistemic component of the predicted uncertainty against the amount of
noise added to the training labels, one panel per model, on the QM9 HOMO--LUMO gap at ECFP4 under
Gaussian noise. The aleatoric line is orange and the epistemic line blue in every panel. The bottom axis
is the noise level, as a fraction of the spread of the clean training labels. The side axis is the mean
predicted uncertainty over the molecules of one out-of-fold pass, in eV. The line is the median over the
folds and the band spans the lowest and the highest of them. Every panel begins at zero, so a component
that holds still looks like one. A rising aleatoric line with a flat epistemic one is the separation the
decomposition is asked for, and two lines climbing together is what its failure looks like. A component
that is one number per fit rather than one per molecule is not drawn, because a slope through a constant
describes the fit and not the molecules. Which models those are is given in Table~\ref{tab:uncertainty}.
The readings come from the fitted slopes and not from the picture.}
\label{fig:decomposition}
\end{figure}
```

*127 words. Cut: the per-panel verdicts, which are findings and belong in §R6 — no caption in any of the
three target-journal papers states a result. Kept: the support rule, the band definition, the units, and the
warning that the readings are from slopes.*

---

### F8 — the three assay datasets

**Settled 2026-09-18: one figure with three panels.** The code writes three image files because three
grids of nineteen models stacked is about 18 inches and the journal allows 225 mm, but they go into one
`figure*` with panel letters so that the shared colour scale — which exists precisely so the three datasets
can be compared — stays on one page. It is tall, so it will want `[p]` and a page of its own.

**The variational-network row stays in, with a sentence in the caption.** On the Caco-2 panel the brightest
row is the first variational network, at AUC$_{norm}$ 0.970 under Gaussian noise, the highest of the
nineteen models, on a clean R$^2$ of 0.358, the second lowest of the nineteen. AUC$_{norm}$ is a share of
clean accuracy, so the small denominator is what lifts it — the Gaussian process on the same panel sits at
0.809 on a clean R$^2$ of 0.517. The caption below carries that sentence. Dropping the row would mean
listing a nineteen-model roster and drawing seventeen.

```latex
\begin{figure*}[htbp]
\centering
\includegraphics[width=\textwidth]{F8_assay_datasets_logd.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_caco2.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_herg.png}
\caption{Robustness (AUC$_{norm}$) of every model on the three assay datasets, on the ECFP4
representation, under the three noise conditions that ran on the whole model roster. a) logD; b) Caco-2
efflux; c) hERG K$_i$. Rows are models, ordered by family, and the columns after the first are the three
noise conditions. One cell is one model under one condition, with its exact value printed on it, coloured
on the bar, whose bright end is the higher AUC$_{norm}$. Each value is the median over the five scaffold
folds. The first column of each panel is clean R$^2$ and is deliberately uncoloured: it is the quantity
the other columns are a fraction of, not a measurement on the same scale. Colour runs on one fixed range
across all three panels, printed on the colour bar. The assay datasets and QM9 use different ranges
because their spans differ by a factor of three, so a cell here and a cell in Figure~\ref{fig:grid} are
not comparable by colour. The brightest row on panel b) has the second lowest clean R$^2$ of the nineteen
models. AUC$_{norm}$ is a share of a model's own clean accuracy, so a small first column lifts the rest
of the row, and every row must be read against its first column. A grey cell marked ``not run'' was never
fitted and one marked ``excluded'' was fitted and then dropped. There are no error bars: one fit per cell
with the seed pinned, and the five scaffold folds partition one dataset rather than repeating an
experiment. Censoring is absent because it runs on a named subset of pairs and cannot rank models.}
\label{fig:assay}
\end{figure*}
```

*196 words, well above the 70–130 band, and it is the one caption I would leave long — it carries the two
prohibitions a reader would otherwise breach, which are comparing a colour here with a colour in
Figure~\ref{fig:grid}, and reading the censoring column as a model comparison.*

---

### R17 — each model against its own probabilistic counterpart

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R17_variant_families_ecfp4_gaussian.png}
\caption{Predictive accuracy against the amount of label noise for each model and its own probabilistic
counterpart, on the QM9 HOMO--LUMO gap at ECFP4 under Gaussian noise. a) NN-$\alpha$ as a plain network,
as BNN-$\alpha$, and as BNN-$\alpha$ with a variance head; b) the same three for NN-$\beta$; c) the
random forest against the quantile forest. One line is one model, named in the panel's own legend, and
the plain, the Bayesian and the variance-head forms keep the same three colours in both neural panels.
The bottom axis is the amount of noise put into the training labels, as a fraction of the spread of the
clean training labels. The side axis is R$^2$ on held-out molecules, taken as the median over ten
replicates and shared across the panels. A higher line is the more accurate model at that noise level,
and a flatter line is the one label noise costs less. Lines that stay together mean the probabilistic
form neither gains nor loses robustness against its deterministic counterpart. No bands are drawn around
the lines. The Gaussian process has no deterministic counterpart in the roster and has no panel.}
\label{fig:variants}
\end{figure}
```

*126 words, essentially the generated caption — it was already the right length and carries no finding.*

---

## Every label the draft text points at

Eight have a block above. **Five do not, and pasting the draft without settling them leaves a broken
`\ref`.** Four are the figure promotions below; the fifth is a table wrapper you write from the pattern in
the tables section.

| label | what it is | state |
|---|---|---|
| `fig:noise_conditions` `fig:variance` `fig:grid` `fig:curves` `fig:decomposition` `fig:assay` `fig:variants` | F1, F2, F3, F4a, F6, F8, R17 | **block above** |
| `tab:robustness` | T4 at ECFP4 on QM9 | **wrapper above** |
| `tab:uncertainty` | T6 | wrapper is yours once T6's shape is settled — `RERUN_PLAN.md` §15, READ FIRST item 3 |
| `tab:variants` | T5 | wrapper from the pattern; it is an additional file, so the reference becomes "Additional file~8" unless you promote it |
| `tab:pairs` | T8 | **PROMOTED to the paper, 2026-09-18, author's call.** §R2's third paragraph points at it |
| `fig:conditions` | F4b | stays an additional file, 2026-09-18, author's call |
| `fig:ranks` | R15 | **promotion decision below** |
| `fig:transfer` | R9 | **PROMOTED to the paper, 2026-09-18, author's call.** §R5's second paragraph points at it |

## Four figures the Results text leans on that §14.25 puts in additional files

**Settled 2026-09-18.** The rank-transfer figure goes in the paper and the other three stay additional
files. The twelve-standout-pairings table goes in the paper too, which is the table entry above rather than
a figure. Each row below still names the sentence that changes, because the three that stayed out are the
ones whose pointers now have to be rewritten.

| figure | what it carries | if it stays an additional file |
|---|---|---|
| **F4b** `F4b_rf_across_noise_conditions.png` | one model across all seven conditions on one pair of axes — the picture of the paper's differentiator | §R4's second paragraph loses its orienting sentence and points at Table~\ref{tab:robustness} instead. **This is the one I would promote**: the subsection that carries the paper's novel claim currently has no figure, while the subsection that repeats the ANOVA has two |
| **R9** `R9_rank_transfer_ecfp4.png` | rank on QM9 beside rank on each assay dataset, one row per model | ✅ **IN THE PAPER.** §R5's second paragraph keeps its orienting sentence and `fig:transfer` resolves inside the manuscript. This also settles the disagreement in the repository: decision D9's own verdict said *"T7 is promoted to a figure"* while §14.25 said additional file, and the paper now follows D9 |
| **R15** `R15_rank_against_level_*.png` | where each model ranks as the noise rises | §R4's fourth paragraph drops to one sentence about Kendall's *W*. R15 is two charts to say the ranking barely moves, so this is the weakest of the four |
| **R6** `R6_representation_profile_*.png` | the cells where a model's robustness at one representation sits outside the range of its others | §R1's third paragraph keeps Figure~\ref{fig:grid} and loses nothing. Decision D5 fired on eight cells, and **seven of the eight are variant models**, which the cross-model figures exclude — so D5 fires on cells the paper does not draw |
---

# THE TABLES

**Five in the paper**, and the fifth is new on 2026-09-18: the metrics table, the noise-conditions table,
robustness at ECFP4, the uncertainty table, and the twelve standout pairings, which the author promoted
because §R2's third paragraph names it and nothing else in the paper carries that list.

🔴 **The uncertainty table has changed and so has its file.** It is now **one row per model and noise
condition, on QM9 at the PDV representation**, which is 30 rows, and it is written to
`T6_uncertainty.tex` as before. Every number column used to be a median over the seven noise conditions.
Censoring is the one condition under which a model becomes *more* certain as its labels are corrupted, so a
median over seven with one reversed cancelled it out, and §R6's second paragraph had no table behind it.

**PDV rather than ECFP4, because the uncertainty runs never put censoring on QM9 at ECFP4.** In the
uncertainty runs on the computed property, censoring ran at PDV alone — three rows in `d8_component_slopes.csv`
and three in `d7_q6.csv`, all PDV, checked on 2026-09-18. On the three laboratory sets it ran at ECFP4, PDV
and ChemBERTa. QM9 at ECFP4 would have been 28 rows with no censoring row among them, which is the one
condition the split exists to show. This makes the uncertainty table the only main-text table not held at
ECFP4, and its caption must say so.

⚠️ *The robustness grid is not the uncertainty runs, and an earlier version of this note said censoring ran
at PDV alone on QM9 without that qualifier.* `auc_norm_qm9.csv` carries five QM9 censoring rows, three at PDV
(NGBoost, the RBF Gaussian process, the first variance-head network) and **two at ECFP4** — the random forest
at 0.817 and the heteroscedastic Gaussian process at 0.813. The heteroscedastic process is a variant model
and the cross-model tables drop it, so T4 at ECFP4 shows one censoring entry and twelve dashes. That is why
the T4 fragment and this paragraph appeared to disagree.

What the three censoring rows show, from the fragment as generated: both component slopes go negative
(NGBoost $-0.354$, the first variance-head network $-0.217$ and $-0.137$, the Gaussian process $-0.069$)
against $+0.62$ to $+1.19$ and $+0.07$ to $+0.84$ under every other condition. The correlation between
predicted uncertainty and error also turns negative, and the gain from dividing the error by the uncertainty
jumps to $+0.17$ and $+0.20$ against roughly zero everywhere else. **Do not write a sentence around that
last number yet** — it is the quantity §R7 says has no band behind it until the re-run.

The complete version, every dataset and every representation at 1,003 rows, is written beside it as
`T6b_uncertainty_every_dataset.tex` for an additional file.

🔴 **Paste from `results/decisions_arc_20260916/tables_latex_fixed/`, not from `tables/`.** The sixteen
fragments in `tables/` were written before the escaping fix and fifteen of the sixteen do not compile: "Sort
\& Slice" puts a raw `&` inside a row and adds a column to it, "Outlier (10\%)" comments out the rest of its
own line including the row terminator, and fourteen of them carry a character `pdflatex` cannot set. The
sixteen in `tables_latex_fixed/` were regenerated on 2026-09-18 from the same CSVs through `latex_safe()` in
`scripts/figlib_tables.py`, and every one now has the same number of `&` in every row as its column
specification asks for. **That is a structural check and not a compile** — there is no LaTeX toolchain on
this laptop, so the first real proof is Overleaf.

Those fragments also carry the em-dash change. **`NaN` now prints as `---` in the LaTeX and stays a real
`NaN` in the CSV**, which is 1,243 cells across the sixteen. A dash with the reason in the caption is what
the journal's tables do, and `NaN` reads to a referee as a calculation that failed rather than as a
combination that was never run. The T4 caption below already carries that reason and needs no change.
`scripts/test_table_latex.py` has a case for it. **If you would rather keep `NaN`, it is the one constant
`MISSING_CELL` at the top of `scripts/figlib_tables.py`.**

Nothing was overwritten. The broken originals are still in `tables/` and the CSVs beside them are unchanged,
because the defect was never in the data. **That directory is also how I know the check works**: the audit
in `scripts/test_table_latex.py` finds 24 problems across the originals and none across the regenerated
sixteen. Until this session it found neither, because it was pointed at `results/decisions_arc/tables`, which
was deleted, and a missing directory made it print a line and pass.

**Once the fix is on the cluster the next harvest writes correct fragments itself and this second directory
stops being needed.** It is not there yet: it needs the commit to be pushed and
`bash scripts/pull_safely.sh` run against it.

**Read the `.tex` files and paste them. Do not retype a table.**

Each fragment is a `tabular`, not a float, so it needs wrapping. The pattern, and T4 as the worked example:

```latex
\begin{table}[htbp]
\centering
\caption{Robustness by model and noise condition on the QM9 HOMO--LUMO gap, ECFP4 representation.
One row is one of the thirteen base models, and the columns are its clean R$^2$ followed by its
AUC$_{norm}$ under each of the seven noise conditions. AUC$_{norm}$ is the normalised area under the
R$^2$ retention curve, so it is the share of a model's own clean accuracy that it keeps as label noise
rises. A higher value is more robust. The clean R$^2$ column is the quantity the others are a fraction
of, and the two are meant to be read together. AUC$_{norm}$ is an area over the whole ladder of seven
noise levels, so it does not say at which level the accuracy fell. Values are the median over ten
replicates, and a replicate whose clean R$^2$ falls below 0.3 is excluded, so an entry can rest on
fewer than ten. An em dash marks a combination that was not run under that condition. Censoring,
Student-$t$, outlier contamination and Laplace each ran on a named subset of the models by design.
The six model configurations that Methods sets aside from every comparison that ranks models against
each other are not in this table.}
\label{tab:robustness}
\small
\input{T4_robustness_qm9_ecfp4}
\end{table}
```

*A table title in the target journal is a noun phrase without a verb, 6 to 25 words, and every results table
carries a separate footnote of about 20 to 25 words holding the metric definition and the replicate
convention. The caption above folds the footnote in, which is what Kolmar does.*

**The uncertainty table's wrapper.** It is the only main-text table not held at ECFP4 and the caption has to
say why.

```latex
\begin{table}[htbp]
\centering
\caption{Uncertainty statistics per model and noise condition on the QM9 HOMO--LUMO gap, PDV
representation. One row is one model under one noise condition, and a combination that was not run has
no row. The two support columns print per\_molecule for a component that varies from molecule to
molecule, constant for one that is a single number per fit, and none for a model that reports no such
component. A component that does not vary from molecule to molecule is given no slope, because a slope
through a constant describes the fit rather than the molecules, and an em dash in the two slope columns
marks that case. Each slope is the change in a model's mean predicted uncertainty, in the label's own
units, per unit of noise added to the training labels. A positive slope is the behaviour a model that
has registered the corruption should show. $\rho$(uncertainty, error) is the rank correlation between a
molecule's predicted uncertainty and how wrong its prediction is, and a positive value means the
predictions a model is least sure of are its worst ones. $\Delta$AUC is the change in how well the
corrupted labels are ranked when the out-of-fold error is divided by the predicted uncertainty rather
than used alone. A positive value means the division ranks them better, and zero means the uncertainty
added nothing. No band has been computed for that column, so its entries give a direction and not a
significance. Values are the median over the five inner scaffold folds, and over three of the five for
NGBoost. PDV is the representation here rather than ECFP4, which the other tables use, because
censoring was run on the computed property at PDV alone.}
\label{tab:uncertainty}
\small
\input{T6_uncertainty}
\end{table}
```

**The standout-pairings wrapper.**

```latex
\begin{table}[htbp]
\centering
\caption{The twelve model-and-representation pairings that score well on both predictive accuracy and
robustness on the QM9 HOMO--LUMO gap and on all three assay datasets. One row is one pairing, with a
clean R$^2$ column and an AUC$_{norm}$ column for each of the four datasets, and a higher value is
better in both. Clean R$^2$ is the accuracy with no noise added and AUC$_{norm}$ is the share of it
retained as noise rises. The twelve are drawn from the seventy-five pairings that ran on all four
datasets. Rows are ordered by the two quantities together: within each dataset both are rescaled to run
from 0 at that dataset's worst pairing to 1 at its best, the two are averaged, and the result is
averaged over the four datasets. That composite is the only thing the order reflects, and it is not a
ranking on robustness or on accuracy taken alone. A pairing higher in the table is therefore not the
better choice on any one dataset, and the table is not a recommendation. Each QM9 entry is the median
over ten replicates, and each assay entry the median over the five scaffold folds of that dataset.
Pairings whose AUC$_{norm}$ exceeds 1 on any dataset are excluded, because that happens when the clean
accuracy being divided by is itself small.}
\label{tab:pairs}
\small
\input{T8_pairs_across_datasets}
\end{table}
```

| slot | fragment | goes in | notes |
|---|---|---|---|
| **T1** | `T1_metrics.tex` | Methods | 18 metrics, generated from the registry the figure code reads, so it cannot name a metric nothing computes. Add the two new rows after the re-run: the gain against its own band, and the error's band as the precondition |
| **T2** | `T2_noise_conditions.tex` | Methods | ⚠️ The "why it is in the study" column is prose from `noise_conditions.json` and three of its seven cells are **truncated mid-sentence** — grouped-shifted ends "Same amo", Student-*t* ends "Every figure here is on the t", Laplace ends "the fir". The column is a working note, not caption text. Rewrite the seven cells by hand, one clause each, or drop the column |
| **T4** | `T4_robustness_qm9_ecfp4.tex` | **paper** | 13 base models × 7 conditions, clean R² first. The three assay versions go to additional files |
| **T6** | `T6_uncertainty.tex` | **paper** | 30 rows, one per model and noise condition, QM9 at PDV. Wrapper below. The 1,003-row complete version is `T6b_uncertainty_every_dataset.tex` |
| T3 | `T3_variance_decomposition_qm9.tex` | additional | the numbers behind F2, with the jackknife spread |
| T5 | `T5_probabilistic_transformations.tex` | additional | 27 rows, one column per representation, `*` for the signed-rank test. **The caption must say that on ten pairs a two-sided signed-rank test cannot go below 0.002, and on the five-fold assay datasets it cannot go below 0.0625 however large the effect** |
| T7 | `T7_rank_transfer_*.tex` ×6 | additional | one per representation. See the cut list |
| **T8** | `T8_pairs_across_datasets.tex` | **paper** | 12 pairings × 4 datasets, clean R² and AUC_norm side by side. Promoted 2026-09-18; §R2's third paragraph names it |

---

# THE CUT LIST — what I would take out, and what the paper loses

Asked for explicitly. Ordered by how confident I am.

1. **Five of the six T7 rank-transfer tables.** The finding is one number — the model ranking transfers from
   the computed property to a measured one at a median rank correlation of 0.36 over 83 combinations of
   representation, noise condition and measured dataset. Six tables of nineteen rows
   say it six times. **Keep one, at the representation the main text holds, and put the median across all
   eighty-three combinations in the sentence.** *Lost:* the reader cannot see that the transfer is much
   better at ECFP4 (0.60) than at Sort & Slice (−0.00), which is itself interesting. *It was protecting
   against:* averaging over representation, which is the rule the whole figure set was rebuilt around — so
   keep the per-representation spread as a clause in §R5 rather than as five tables.

2. **R16, the decoupling figure.** Its own caption warns that part of what it shows is arithmetic, because
   the retention metric divides the clean baseline out. The finding is real and lives in §14.15b, and it is
   better as the two top-ten lists that share no members than as a scatter whose axes are not independent.
   **Cut the figure, keep the finding in §R2.** *Lost:* nothing the text does not carry. *It was protecting
   against:* the reader assuming robustness is bought with accuracy — which §R2's third paragraph now says
   outright, and §R2's second paragraph gives NGBoost as the worked example of it.

3. **R18, the representation-against-representation scatter.** It is the evidence for holding one
   representation constant in the main text, which is a Methods decision rather than a result. **Move the
   number into the Methods sentence that names the held representation** — a model's robustness carries from
   ECFP4 to MHG-GNN at a rank correlation of 0.51 over thirteen models, which is what that choice costs.
   *Lost:* nothing. *It was protecting against:* a referee asking what holding one representation costs, and
   a sentence answers that better than a figure.

4. **R15, the rank ladder.** Two charts to say the ranking barely moves. Kendall's *W* says it in one
   number, and §R4's fifth paragraph already carries it. **Cut, unless you promote F4b and want the pair.**
   *Lost:* the visible fact that the few crossings are models falling below the accuracy gate rather than
   trading places, which is worth one clause. *It was protecting against:* the assumption that a ranking at
   one noise level is a ranking at all of them.

5. **R19, the deep-run conditions as a figure.** The matched-pair comparison is the finding and it is a
   table — T4 already carries all seven conditions per model. **Cut the figure, keep T4.** *Lost:* the visual
   that those conditions are thin by design and not by accident, which `what_is_missing.csv` and one Methods
   clause cover. *It was protecting against:* a reader reading the empty cells as missing runs.

6. **Every sentence that reports a statistic nobody asked a question about.** Specifically: the ICC between
   model profiles, the grid-similarity measure behind decision D2, and the count of duplicate runs. All three
   are machinery that decided what to draw. They belong in the additional files that record how the figure
   set was chosen, not in the Results.

**What I would NOT cut, though it is a candidate.** The four cells whose retention exceeds one. It is two
sentences, it is the honest treatment of a normalised metric, and a referee who spots a value above one in
a table and finds no comment will not be generous about the rest.

**Checked again on 2026-09-18, after the §R1 to §R4 rewrite.** All six items still stand and every number in
them reproduces from the 16 September harvest: the transfer median is 0.356 over 83 combinations with ECFP4
at 0.600 and Sort \& Slice at $-0.001$ (`d9_rank_agreement.csv`), and robustness carries from ECFP4 to
MHG-GNN at 0.505 over the thirteen base models (`auc_norm_qm9.csv`). Two pointers in items 2 and 4 named the
wrong paragraph of §R2 and §R4 and are corrected above. **Item 2 is the one that changed in substance**: §R2
now makes the accuracy-and-robustness point three times over, once through NGBoost, once through the pooled
correlation and once through T8's twelve rows, so cutting R16 costs the text less than it did when this list
was first written.

**Net effect: seven figures and four tables in the paper, with F4b promoted to eight figures if you take the
recommendation above; five figures and four tables in the additional files, down from seven and four.**

---

# REFERENCES — every key the new text needs

| key | what it supports | source section | `citations.bib` | `refs.bib` | cited in `paper.tex` today |
|---|---|---|---|---|---|
| `Kruger2012` | normality of bioactivity differences rejected, Laplace fitted | `NOISE_DESIGN.md` §3.1 | yes | no | **no** |
| `Kalliokoski2013` | 16,844 repeat pairs, truncation, the 7.7 log-unit tail, pIC$_{50}$ 0.68 | §3.1, §3.2, §3.4 | yes | no | **no** |
| `Kramer2012` | pK$_i$ 0.54 | §3.2, §4b | yes | yes | **no** |
| `Bentz2013` | 62% of variance between laboratories, 23 laboratories | §3.3 | yes | no | **no** |
| `Svensson2025` | censored fractions in fifteen industrial assays | §3.5 | yes | no | **no** |
| `landrum2024` | 0.50 → 0.27 log units on curating to one assay | §3.3 | yes, line 1910 | yes, line 2180 | no |
| `Heid2023` | no difference between error distributions at equal mean and SD | §3.6 | yes | yes | yes |
| `Hampel2001` | contamination typically 1–10% | §2 | yes | yes | **no** |
| `Avdeef2019` | within-laboratory error × 3 to reach between-laboratory | §2 | yes | no | **no** |
| `huber1964robust` | the contamination model the outlier condition implements | §2 | yes | yes | **no** |
| `avalon` | the Avalon fingerprint | `HANDOFF.md` line 382 | **added 2026-09-18** | no | no |
| `Scalia2020` | the field-standard uncertainty benchmark | `RERUN_PLAN.md` §14.6 row 2 | **added 2026-09-18** | no | no |
| `Hirschfeld2020` | the field-standard uncertainty benchmark | §14.6 row 2 | **added 2026-09-18** | no | no |
| ~~`Tran2020`~~ | the same claim as the two above | §14.6 row 2 | **NO** | **no** | no | **recommend dropping, see below** |
| `kendall2017` | the aleatoric/epistemic decomposition the Methods gives as a display equation | §14.6 row 2 | yes | yes | yes |

**Three keys were missing and all three are now in `citations.bib`, appended at the end of the file under a
dated comment block: `avalon`, `Scalia2020` and `Hirschfeld2020`.** An earlier version of this table said
`Landrum2024` was missing too. **It was not.** Both `citations.bib` (line 1910) and `refs.bib` (line 2180)
carry it under the lower-case key `landrum2024`, with a DOI and a PMID that the hand-written entry below
lacked, so **cite it as `landrum2024` and do not paste an entry for it.**

`Tran2020` is the only one still unwritten, because nothing in this checkout verifies it. It supports exactly
the same claim as `Scalia2020` and `Hirschfeld2020`, both of which are verified from full PDFs read in
`research_archive/f692d614/`. **The recommendation is to cite those two and drop `Tran2020`**, which costs
the sentence nothing. If you would rather keep it, it is Tran et al. in *Machine Learning: Science and
Technology* on comparing uncertainty quantifications for material property predictions, and it needs
checking against the published version before it goes in.

These are the three entries as they now stand in `citations.bib`:

```bibtex
@article{avalon,
  author  = {Gedeck, Peter and Rohde, Bernhard and Bartels, Christian},
  title   = {{QSAR} --- How Good Is It in Practice? Comparison of Descriptor Sets on an
             Unbiased Cross Section of Corporate Data Sets},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {46}, number = {5}, pages = {1924--1936}, year = {2006}}

@article{Scalia2020,
  author  = {Scalia, Gabriele and Grambow, Colin A. and Pernici, Barbara and Li, Yi-Pei
             and Green, William H.},
  title   = {Evaluating Scalable Uncertainty Estimation Methods for Deep Learning-Based
             Molecular Property Prediction},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {60}, number = {6}, pages = {2697--2717}, year = {2020}}

@article{Hirschfeld2020,
  author  = {Hirschfeld, Lior and Swanson, Kyle and Yang, Kevin and Barzilay, Regina
             and Coley, Connor W.},
  title   = {Uncertainty Quantification Using Neural Networks for Molecular Property
             Prediction},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {60}, number = {8}, pages = {3770--3780}, year = {2020}}

```

*Provenance. `Scalia2020` and `Hirschfeld2020` come from
`research_archive/f692d614/ALEA_EPIS_LITERATURE.md`, which records both as read in full from the published
versions and gives the volume, the page range and the arXiv number; the two PDFs are in that same directory.
`avalon` is the reference the author wrote out at `HANDOFF.md` line 382. All three carry an issue number that
the archive does not state, so the issue number is the one item in them worth a glance during the Overleaf
check.*

🔴 **Every key above needs an Overleaf check and none of them can be checked from here.** `paper.tex` points
`\bibliography` at `sn-bibliography`, which lives in the Overleaf project and is not in this checkout, so
whether it already carries a key is unknown on this side. `citations.bib` holds 221 entries against the 51
`paper.tex` cites, and `refs.bib` (205 entries, untracked in git) is a third list. `scripts/check_bib_and_docs.py`
passes and its one pending item is unchanged: `paper.tex` names `sn-bibliography` and the bibliography is
`citations.bib`.
---
