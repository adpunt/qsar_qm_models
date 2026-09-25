# NoiseInject Paper: Revision Guide

> 🔴 **Two sections of this guide are superseded, 2026-09-21. Do not paste from them.**
>
> | Superseded here | Read instead |
> |---|---|
> | M6 Performance metrics, §I4, §I6, "The current main picture" | `PAPER_METHODS_INTRO_REWRITE.md` |
> | *(M5 Uncertainty quantification is live here again, rewritten 2026-09-23)* | — |
> | §R1 through §R7, and the figure blocks they carry | `PAPER_RESULTS_REWRITE.md` |
>
> Both were written after reading the figures as images and re-reading the primary sources, and both
> carry the citation corrections in `RERUN_PLAN.md` §17.1. What stays live here is M1 to M4, the
> availability and additional-files sections, the tables, the cut list and the references.

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
While we are investigating the concept of noise in experimental data, for the purposes of testing under artificial noise we chose to primarily conduct experiments on a data set with clean labels. Data were extracted from the QM9 molecular property data set distributed by PyTorch Geometric \citep{Ramakrishnan2014}, which contains 130,831 small organic molecules with pre-computed quantum-mechanical properties obtained from density functional theory (DFT) calculations at the B3LYP/6-31G(2df,p) level of theory. We selected the gap between the highest occupied and the lowest unoccupied molecular orbital (HOMO--LUMO gap) as the prediction target in QM9, as it captures a molecule's electronic excitability, charge transfer capability, and chemical stability \citep{Fediai2023}. We assumed that QM9 labels were free of noise, notwithstanding approximations in the level of theory. Each experiment was replicated ten times with a distinct random seed and a randomly selected subset of $N = 10{,}000$ molecules, split 80/10/10 into 8,000 training, 1,000 validation and 1,000 test molecules. All models were fitted on those 8,000 training molecules, except the Gaussian processes, which were fitted on a random subset of 5,000 of them; the validation and test molecules were unchanged.

We repeated these experiments on three molecular property data sets with experimentally determined endpoints. From the OpenADMET initiative \citep{openadmet}, we used LogD and Caco-2 efflux permeability, with $N = 5{,}039$ and $N = 2{,}161$ respectively. We also selected hERG $K_i$ data from ChEMBL \citep{Zdrazil2023}, following a protocol inspired by \citet{landrum2024}: filter for binding assays, deduplicate by median pChEMBL value, and remove compounds with an inter-assay standard deviation greater than 1.0 log unit, resulting in $N = 1{,}415$ compounds. Each of the three was partitioned into five folds by grouped $k$-fold cross-validation, and each fold's training block gave up a further grouped fifth as a validation set for early stopping. A model therefore fits between 3,194 and 3,364 of the 5,039 LogD molecules, depending on the fold, so the Gaussian-process subsample never applies to these data sets. Unlike QM9's ten replicates, which are ten independent draws from one pool, each assay data set is a single fixed set in which every molecule is tested exactly once and nothing is redrawn.

All splits and cross-validation folds were grouped by the chirality-blind Bemis--Murcko scaffold, implemented such that each distinct acyclic molecule forms its own group, and groups are filled in random order. Prior to modeling, all molecular structures were sanitized and canonicalized using RDKit to remove invalid valence states and standardize atom and bond typing \citep{rdkit}. Prediction targets were standardized to zero mean and unit variance. Experiments that track uncertainty were run once rather than in replicates.
```

**For the author.**

- Every defect on the list held up against the file it named. I re-measured all of them rather than trusting the note.
- 🔴 Rewritten 2026-09-20 to the author's own four-paragraph structure, and her prose is kept. Two things I had cut are back and were the reason for the rewrite. The Gaussian processes fit 5,000 of the 8,000 QM9 training molecules, not 8,000, and the version before this one asserted 8,000 for every model. The three QM9 molecules with an all-zero Sort \& Slice vector are dropped from all six representations, and that had disappeared entirely.
- The 8,000 / 1,000 / 1,000 counts are checked on seeds 0, 1 and 2 only, not on all ten replicates. The splitter packs whole scaffold groups until it reaches a cutoff, and the three Sort \& Slice molecules come out afterwards without the split being redone, so the counts are not guaranteed by the design. If you want them safe, write "split 80/10/10 by scaffold" with no counts, or ask me to run all ten seeds.
- The means and standard deviations of the four label distributions are gone with the rewrite. Nothing in the Results reads them except §R5, whose "a level of about 0.6 is therefore one unit of the error a laboratory already carries" rests on the hERG spread of 0.915 log units. That sentence is now unverifiable from the paper alone. Either put the spread back in one clause, or add it to a datasets table.
- The over-valence claim: data/qm9_pool_provenance.json does record kept_over_valent_carbon_in_sdf = 0 and dropped_over_valent_carbon_in_sdf = 1403, so over-valent carbon and the drop do coincide on what was scanned. But the kept side was a control scan of 2,023 molecules only (kept_molecules_scanned_as_a_control), not all 129,428, and the filter the file records is Chem.MolFromSmiles on the SMILES string. So the cause sentence is now written as the SMILES parse, and the over-valence line is dropped for space.
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
- **Gaussian processes fit a seeded draw of 5,000 of the 8,000** — models/model_defaults.py:264 sets max_train_n to 5000 and models/model_defaults.py:494-526 draws the subset from a seed; both Gaussian-process fitters call it with the run's own seed (models/models.py:2849 and :8453), so the two draw the same molecules; QM9 rows in results/decisions/d0_duplicate_disagreements_qm9.csv carry n_train=5000 and nothing else

</details>


## M2. Molecular representations

*R.*eplaces paper.tex 201-206

```latex
\subsection{Molecular representations}

In this study, we evaluated six molecular representations: extended-connectivity fingerprints of diameter four (ECFP4), Avalon, Sort \& Slice (SNS), physicochemical descriptors (PDV), ChemBERTa and Molecular Hypergraph Grammar GNNs (MHG-GNNs). The same six were used on QM9 and on the three assay datasets. ECFP4 is a Morgan fingerprint with radius $r = 2$ with $d = 2048$ bits \citep{rogers2010, rdkit}. SNS takes the same substructures as ECFP4, but it ranks them by prevalence in the training set and selects the top $L$ ($L=1024$) \citep{sns}. Avalon is a 2,048-bit fingerprint, generated with the Avalon toolkit through RDKit \citep{avalon, rdkit}. PDV is a vector of 200 physicochemical descriptors computed with RDKit's \texttt{MolecularDescriptorCalculator} \citep{rdkit, Cherkasov2014}; the 200 are listed in Additional file~1, Table~C, and the list is fixed in our code rather than taken from RDKit's default descriptor set, which grows between releases. ChemBERTa embeddings are the mean over the non-padding token embeddings of the \texttt{DeepChem/ChemBERTa-77M-MTR} checkpoint, producing 384 features \citep{Ahmad2022}. MHG-GNNs are a GIN-based autoencoder pre-trained on 1.34 million PubChem molecules using $\beta$-VAE loss, producing 1024-dimensional embeddings through iterative message passing and found in the published \texttt{mhggnn\_pretrained\_model\_0724\_2023} checkpoint \citep{kishimoto2023}. We z-score normalised PDV, ChemBERTa and MHG-GNN per feature, using the training molecules' mean and standard deviation; ECFP4, Avalon and Sort \& Slice reach the model as built.

We fitted the Sort \& Slice vocabulary on the training molecules alone, refitting it for each QM9 replicate and for each cross-validation fold. Several of the 200 PDV descriptors return non-finite values on some molecules, and each such value is replaced by zero before the scaling constants are computed.

The ChemBERTa checkpoint ships an empty merges file, so its tokenizer falls back to single characters and has no entry for seven characters that occur in SMILES, leaving it unable to distinguish halogens from one another, charged from neutral forms, enantiomers, or azole tautomers. Among the hERG $K_i$ molecules 5.09\% share a token sequence with another molecule, and among the QM9 structures 2.71\% do. We retained the representation and report this as a limitation on its results.

On QM9 no representation sees stereochemistry: all six are built from the stereochemistry-free canonical SMILES, with chirality switched off in the Morgan-based fingerprints. On the three assay datasets only MHG-GNN distinguishes a pair of enantiomers, and the other five give them one point.
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

- 🔴 **Cut 2026-09-20, 602 words down to 383.** Five things went, all of them mechanism a reader does not need:
  the rule that decides which feature matrices skip standardisation (a binary matrix, or a sparse non-negative integer matrix with at least one entry above 1) and the two Sort \& Slice densities, 1.3% of entries non-zero on QM9 and 4.7% on hERG;
  the clause saying the per-fold vocabulary is fitted on the rows that fold fits and excludes the validation rows;
  the all-zero Sort \& Slice vector and the molecules dropped for it, which M1's preprocessing paragraph now carries in full, so this was saying it twice;
  the per-representation enumeration of which five miss a pair of enantiomers, replaced by naming the one that does not;
  and the two sentences saying the representations are computed in Python and that the Rust component computes none, which paper.tex:218 already says.
- The seven characters ChemBERTa cannot read are no longer listed individually, and neither are the chlorobenzene/toluene and alanine examples, the 3,502-of-129,238 and 72-of-1,415 raw counts, or the 25-draw simulation. What is left is the cause, the four things it cannot separate, and one collision rate per dataset.

- "built by the same calls on both pipelines" — false: the two pipelines call different functions, on different SMILES, into different storage.
- "Enantiomers are therefore one point for four of the six representations on the assay datasets and two points for the other two" — replaced by the measured five-and-one split.
- "ChemBERTa and MHG-GNN, however, read a standardised canonical SMILES that keeps it" — true of the string but misleading about ChemBERTa, whose reader drops the @.
- "On QM9 it is refitted once per noise level and replicate" — the vocabulary comes from the training molecules, which do not change with the noise level.
- The PDV descriptor enumeration (molecular weight, LogP, polar surface area, connectivity indices, VSA bins, functional group counts) — cut for the 550-word ceiling once the Sort & Slice exclusion and the corrected stereochemistry paragraph were added. 🔴 The earlier note here said it survived in paper.tex:222 and in Additional file 1, and both were wrong: paper.tex:222 opens the Models subsection, and Additional file 1 held hyperparameters only. **All 200 names are now Additional file 1, Table C**, generated by `scripts/generate_supp_table1.py` from `DEFAULT_DESCRIPTOR_LIST` and guarded by two checks in `scripts/test_supp_table1.py`. The Methods sentence should point at it, and should say the list is pinned in the code rather than taken from RDKit's own descriptor set, which grows between releases.
- "The rule is applied to the feature matrix, not to the representation's name" — cut for length; the two sentences that follow describe matrix properties, so the point still lands.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **six representations** — scripts/process_and_train.py:1102-1118 (pdv, chemberta, mhggnn, avalon, ecfp4) plus the Sort & Slice featuriser at :1196-1205; KIRBy tests/alternative_data_noise_robustness.py:3005-3030 builds SNS, MHG-GNN, Avalon, ChemBERTa alongside ECFP4 and PDV
- **radius r = 2, 2,048 bits (ECFP4)** — scripts/process_and_train.py:1520-1527 (ECFP4_RADIUS, ECFP4_BITS, GetMorganGenerator); KIRBy src/kirby/representations/molecular.py:293-306 (radius=2, n_bits=2048)
- **L = 1024 Sort & Slice features, counts not bits** — scripts/process_and_train.py:163 SNS_DIM = 1024 and :1196-1205 (sub_counts=True); rust/src/main.rs:74 sns_buf [u8; 2048] = 1,024 u16 counts
- **three molecules of 132,480 QM9 SMILES dropped** — code comment scripts/process_and_train.py:1220-1221, recording a 2026-09-07 count by scripts/sns_zero_molecules.py; the dropping itself is at :1240-1268
- **Avalon 2,048 bits** — scripts/process_and_train.py:1581-1600 (avalon_fingerprint, nBits=2048); KIRBy molecular.py:322-343 create_avalon(n_bits=2048)
- **PDV 200 descriptors** — counted this session from DEFAULT_DESCRIPTOR_LIST at scripts/process_and_train.py:84 (200 names, no duplicates); KIRBy src/kirby/representations/molecular.py:314 carries the same 200 names in the same order, diffed character for character on 2026-09-20, so both pipelines build the same vector. The names are published as Additional file 1, Table C. RDKit version 2022.09.5, from data/qm9_pool_provenance.json
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

We fitted tree ensembles, a support vector machine, Gaussian processes and neural networks, and built the same set for the QM9 HOMO--LUMO gap and for the three assay datasets. Several are variants of one another, differing only in the probabilistic machinery they add, and Additional file~1 gives every configuration with its settings.

Random forests (RFs) are one of the most common choices for QSAR modeling, thanks to their robustness and interpretability \citep{Svetnik2003, Breiman2001}. Quantile regression forests (QRFs) keep the distribution of training labels in each leaf and compute quantiles across all trees \citep{Meinshausen2006}. The quantile forest uses 300 trees, since the quantile estimate it is fitted for is noisy at 100. We therefore fitted a second ordinary forest at 300 trees, identical to it in every other setting, so that the two differ by the quantile machinery alone.

We also used eXtreme Gradient Boosting (XGBoost) \citep{Mustapha2016, Tian2022}, Light Gradient-Boosting Machine (LightGBM) \citep{ke2017lightgbm} and Natural Gradient Boosting (NGBoost) \citep{Duan2020}. We configured NGBoost as \citet{Duan2020} did, with a Normal distribution over the label and the log scoring rule, and read the number of boosting rounds off a held-out curve. Those held-out rows differ between the two pipelines: on QM9 they are the validation split that is already held out of every fit, and on the three assay datasets they are a scaffold-grouped fifth of the rows NGBoost is handed, which already exclude the validation molecules. NGBoost on those three datasets therefore fits fewer molecules than the other tree models.

Support vector machines (SVMs) are a well-established baseline in QSAR modeling \citep{Vapnik1995, Svetnik2003}. We used a radial basis function (RBF) kernel on every representation, on QM9 and on the three assay datasets, with no branch on the representation.

Gaussian processes (GPs) use a kernel to produce a Gaussian predictive distribution over every data point \citep{Obrezanova2007, Rasmussen2005}. We fitted three of them as exact processes in \texttt{gpytorch}, each with a constant mean and a scaled kernel. The first uses an RBF kernel and runs on all six representations. The second uses the Tanimoto kernel \citep{Ralaivola2005, moss2020}, which is the only kernel here taken from the Gauche framework \citep{gauche}, and was run on ECFP4 alone. Sort \& Slice is built from substructure counts rather than bits, and a fingerprint kernel asked for on features that are not binary is refused at fit time. The third is the RBF process with a second network predicting the observation noise of each molecule, and it is one of the five uncertainty configurations.

Exact inference is cubic in the number of training points \citep{Rasmussen2005}, so every Gaussian process here fits at most 5,000 training molecules, drawn by a seeded sample. The cap binds on QM9, where the training split holds 8,000 of the 10,000 sampled molecules. It does not bind on any assay dataset, where the largest training block is 3,364 of the 5,039 LogD molecules.

The RBF lengthscale starts at the median pairwise distance between 500 sampled training molecules, rather than at the framework's default. In a range-finding run on QM9 the typical Euclidean distance between two molecules' feature vectors was about 17 on PDV and about 1,100 on the learned embeddings. A fit whose predictions vary by less than 0.05 of the spread of the labels it was fitted on is recorded as collapsed. The $R^2$ of a collapsed fit is still reported, but that fit returns one variance for every molecule, so the column holding its epistemic uncertainty is left blank. Epistemic uncertainty is the model's uncertainty about its own fit, and Uncertainty quantification defines it against the other component.

Two deterministic feed-forward architectures were used, both fully connected and both implemented in PyTorch \citep{pytorchGeometric}. NN-$\alpha$ has two hidden layers that narrow, with dropout after each, while NN-$\beta$ repeats one width over a variable number of layers, with dropout before the output. Both train with the Adam optimizer, stopping after 10 epochs without an improvement in mean validation loss and restoring the best epoch's weights, and Additional file~1 gives their widths, dropout fractions, learning rates and batch size.

Four Bayesian transformations were applied to each base architecture, giving eight probabilistic networks. The first replaces every linear layer with a Bayesian layer carrying a Gaussian prior $\mathcal{N}(0, 0.1^2)$ on its weights, giving the Bayesian neural networks BNN-$\alpha$ and BNN-$\beta$. It is trained on an evidence lower bound whose KL term is divided by the number of training molecules. The second adds a variance output head to BNN-$\alpha$ and BNN-$\beta$, and fits it by the negative log likelihood of a Gaussian with that variance \citep{kendall2017}. The third replaces every linear layer with a Variational Bayesian Last Layer (VBLL) \citep{Harrison2024}, which maintains a mean-field variational posterior over that layer's weights, giving VBLL-$\alpha$ and VBLL-$\beta$. Only its output layer keeps a learned observation-noise parameter, so the hidden layers of a VBLL network contribute epistemic uncertainty alone. The fourth makes that observation noise a function of the input, and the variance-head and input-dependent-noise variants together are the other four uncertainty configurations. All eight probabilistic networks estimate their predictive distributions from 100 stochastic forward passes.

Each network and its Bayesian counterpart share one setting, so that NN-$\alpha$ against BNN-$\alpha$ and NN-$\beta$ against BNN-$\beta$ differ by the Bayesian layers alone. We chose that setting by ranking candidates on their weaker member rather than their stronger. The variational and variance-head networks were not matched this way, so a comparison between two probabilistic networks still differs in settings as well as in machinery. One setting was chosen per model and dataset, and applied to all six representations.
```

**Still open in this subsection.**

- Regenerate Additional file 1 from models/model_defaults.py plus results/master_tuned_hyperparameters.json and results/master_tuned_hyperparameters_lab.json, with a column marking which configurations run tuned on which dataset. Four on QM9, two per assay dataset.
- Confirm xgboost, gpytorch and Kersting2007 in the Overleaf sn-bibliography before restoring those citations.
- Fix scripts/generate_paper_figures_v2.py:133 and :150 so gauche_rbf is no longer PDV-only before the figures are regenerated.
- Settle the hERG count: paper.tex:196 says 1,482 compounds, the KIRBy loader docstring says 1,415 molecules.
- Decide whether the distance range (about 17 on PDV, about 1,100 on the learned embeddings) is cited from results/gp_kernel_harvest/qm9/ or cut.

**For the author.**

- Citation keys. paper.tex builds against sn-bibliography (paper.tex:694) and there is no local copy of that .bib, so I used only keys that already appear in paper.tex. xgboost, gpytorch and Kersting2007 appear nowhere in paper.tex, so XGBoost keeps \citep{Mustapha2016, Tian2022} and the GPyTorch and heteroscedastic-GP citations are gone. Add them back only if you confirm the keys exist in the Overleaf bibliography.
- The guide's own §4.11 decision 4 says four configurations run tuned. That is true on QM9 and false on the three assay datasets: results/master_tuned_hyperparameters_lab.json holds two model entries per dataset, and LogD's two are not the same two as Caco-2's and hERG's. The text now scopes it. If Additional file 1 is regenerated it needs the same split.
- I dropped the claim that Additional file 1 marks the tuned configurations, because the file is still headed 'default hyperparameters' (guide §4.11 decision 4). The sentence now says only that the settings are in Additional file 1. It becomes false if the file is not regenerated to carry the tuned values.
- The Tanimoto sentence is now two facts rather than one because-clause. The fit-time refusal at models/models.py:2867-2874 fires on any non-binary feature matrix, so it would not catch Avalon, which is binary and still excluded. Running on ECFP4 alone is the study's roster decision (FP_REPS = ['ecfp4'] at slurm_scripts_qm9_rerun/generate_scripts.py:114, reps_for() at slurm_scripts_validation_rerun/generate_scripts.py:292).
- The figure script disagrees with the Methods sentence about the RBF Gaussian process. scripts/generate_paper_figures_v2.py:133 still comments gauche_rbf as 'RBF GP for PDV only' and :150 puts it in PDV_ONLY_MODELS, while both generators run it on all six representations. A regenerated heatmap will show PDV only against a Methods sentence saying six.
- The opening no longer contrasts the five uncertainty configurations with 'competing on accuracy', because eight models are held out of the variance decomposition, not five: ANOVA_MODELS_EXCLUDE at scripts/generate_paper_figures_v2.py:130-147 also drops qrf, gauche and gauche_rbf. A decomposition table therefore holds eleven models. That belongs in the ANOVA subsection, not here.
- The range of distances between molecules (about 17 on PDV, about 1,100 on the learned embeddings) is a prose comment at models/model_defaults.py:300-313 describing a 2026-08-26 measurement, not a results file. I scoped it as a range-finding run so a reader is not told a comment is a result. Cut it or replace it with a harvest number if you would rather not carry it.
- The two datasets disagree on hERG and I used neither number: paper.tex:196 says 1,482 compounds, the KIRBy loader docstring at tests/alternative_data_noise_robustness.py:24-26 says 1,415 molecules. Only LogD at 5,039 molecules is needed for the Gaussian-process cap sentence, and both sources agree on that.
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

- **the roster: six tree ensembles, one SVM, three Gaussian processes, ten neural networks, after rf300 was added 2026-09-21** — slurm_scripts_validation_rerun/generate_scripts.py:31-42 (MODELS_ALL, with GP / GP-Tanimoto / GP-Hetero listed separately); the same keys in the QM9 model dict at slurm_scripts_qm9_rerun/generate_scripts.py:660-745
- **five uncertainty configurations** — slurm_scripts_qm9_rerun/generate_scripts.py:690-745 — heteroscedastic_gp, dnn_bnn_full_variational_hetero, mlp_bnn_full_variational_hetero, dnn_bnn_full_mve, mlp_bnn_full_mve
- **minimum leaf size 5 molecules, 0.3 of features per split, 100 trees (RF), 300 trees (QRF)** — models/model_defaults.py:82-86 and :105-109
- **NGBoost: base learner maximum depth 3, learning rate 0.01, 500 boosting rounds capped, patience 50 rounds, best round used** — models/model_defaults.py:180-212 (n_estimators 500, learning_rate 0.01, base_max_depth 3, early_stopping_rounds 50, use_best_iteration True); applied at models/models.py:2365-2370
- **20% scaffold-grouped carve of each fold's training block (assay side)** — /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:3227 (val_frac=0.2) and :3243 (GroupShuffleSplit)
- **Gaussian process cap of 5,000 training molecules** — models/model_defaults.py:264 ('max_train_n': 5000), applied at models/model_defaults.py:494-526, called for the RBF process at models/models.py:2849 and for the heteroscedastic one at :8453 with the same seed, and written onto every results row as gp_fit_method|n_train= at models/models.py:2895; KIRBy loads the same spec at tests/alternative_data_noise_robustness.py:397-400
- **QM9 training split 8,000 of 10,000 sampled molecules** — paper.tex:192 — N = 10,000 molecules, scaffold 80/10/10 split
- **LogD 5,039 molecules (largest assay dataset); largest fold trains on 3,364 of them** — paper.tex:196 (N = 5,039) and /Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py:24-26; N_FOLDS = 5 at :575. The 3,364 is the counted maximum of the five fold fit counts recorded at M1.
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
% heading through the \end{figure} of the old noise figure, which was
% labelled fig:noise_strategies; the replacement figure is F1, labelled
% fig:noise_conditions, and its block is in this guide's FIGURES section).
% paper.tex lines 364-373, \subsection{NoiseInject Framework}, are
% DELETED and nothing replaces them (decision 2, 16 September).
% =====================================================================

\subsection{Label noise}

We corrupted the training labels with artificial noise under seven conditions, and measured how far
accuracy fell as the amount of noise rose. Three of the conditions vary the shape of the error,
drawing it from a Gaussian, from a Student-$t$ with $\nu = 5$, or from a Laplace distribution. Two
concentrate the error on particular molecules, either on a random tenth of them (outlier) or on
whole Bemis--Murcko scaffold families (grouped-wider). A sixth gives every scaffold family its own
offset rather than a wider error (grouped-shifted), and the seventh records every label past an
assay limit as that limit (censoring). Test labels were never corrupted, and validation labels
carried noise of the same condition and amount as the training labels, drawn independently. Without
that, the models that use validation loss to decide when to stop training would be selected to
ignore the corruption.

All conditions but censoring add noise of the same standard deviation at a given level, so the six
differ in the distribution of the error and in which molecules receive it, not in its magnitude. The
noise added to the label of molecule $i$ is
\begin{equation}
\epsilon_i = \frac{\tau}{G}\,s_i z_i,
\qquad
G = \sqrt{\frac{1}{n}\sum_{j=1}^{n} s_j^{2}}\;\;\sigma_z,
\label{eq:dose_matching}
\end{equation}
Here $\tau$ is the target standard deviation of the noise, in the units of the label, and $z_i$ is
drawn from the condition's distribution at unit scale. The multiplier $s_i$ is $1$ for most
molecules and $\lambda = 3$ for those the condition affects, which are a random $10\%$ of molecules
under outlier and whole scaffold families covering $20\%$ of the split under grouped-wider. The
constant $\sigma_z$ is the standard deviation of $z$, equal to $1$ for the Gaussian,
$\sqrt{\nu/(\nu-2)}$ for the Student-$t$ and $\sqrt{2}$ for the Laplace. The factor $G$ divides both
of these out, so that $\epsilon$ has standard deviation $\tau$ in expectation whatever the
distribution and whichever molecules are affected.

Grouped-shifted widens no molecule's error, and instead gives every scaffold family $g$ an offset of
its own,
\begin{equation}
\epsilon_i = \frac{\tau}{G}\left(\sqrt{\rho}\,b_{g(i)} + \sqrt{1-\rho}\,w_i\right),
\label{eq:grouped_shifted}
\end{equation}
with one draw $b$ per family and one draw $w$ per molecule. The two variances sum to $\tau^{2}$, and
the family offset carries a share $\rho$ of that total.

We took each parameter from published measurement error where one exists. Within-laboratory error has
to be multiplied by about three to reach between-laboratory error, which fixes $\lambda = 3$
\citep{Avdeef2019}, and a contaminated fraction of $0.1$ is the upper end of the one to ten percent
Hampel reports for scientific routine data \citep{Hampel2001}. Laboratory identity accounts for
$62\%$ of the variance in a Caco-2 round robin across eleven laboratories, which fixes
$\rho = 0.62$ \citep{Bentz2013}. We took the Student-$t$ and Laplace shapes from the distributions
fitted to repeated public bioactivity measurements \citep{Kruger2012}, and contaminating a fraction
of records with a wider error is Huber's contamination model \citep{huber1964robust}. No published
figure exists for the fraction of scaffold families a difference between laboratories would cover,
so we chose the $20\%$ above rather than sourcing it.

A noise level is a fraction of the spread of the clean labels rather than a number of log units, and
we ran seven levels, at $0$, $0.2$, $0.3$, $0.5$, $0.75$, $1.0$ and $1.5$. That spread is the
standard deviation of the clean training labels on QM9, and of the whole clean label column on the
three assay datasets. We added the noise to the raw label and standardised the target afterwards,
using the clean training mean and spread. Censoring has no amount to match and we swept it on its own
axis instead, recording every label above the $k$th largest clean training label as that limit and
taking the level to be the fraction clipped, at $0$, $0.1$, $0.2$, $0.25$, $0.3$, $0.4$ and $0.5$
\citep{Svensson2025}.

Gaussian, grouped-wider and grouped-shifted noise ran on every model and representation, and the
remaining four conditions ran on named subsets of those pairs, since those four were included to
size an effect rather than to rank models against one another. The seven conditions are implemented
twice, in Rust for QM9 and in the released Python package for the three assay datasets, and we check
the two implementations against each other on the real labels.
```

**Still open in this subsection.**

- Confirm whether the Overleaf sn-bibliography.bib carries huber1964robust, Kruger2012, Bentz2013, Avdeef2019, Hampel2001 and Svensson2025. None is cited in the current paper.tex, and sn-bibliography.bib is not in this checkout. The entries to paste, with DOIs, are at citations.bib lines 1804, 2028, 2107, 2118, 2231 and 2244.
- Drop '$\sigma$ is the noise scaling factor (Table~\ref{tab:regression_noise})' from the metrics-table caption at paper.tex:307-308 in the same pass as the ECE removal. There is no noise table any more, so every other \ref{tab:regression_noise} in paper.tex has to go the same way.
- 🔴 **Nothing in the Methods points at the noise figure, and the figure exists.** `F1_noise_conditions.png` was drawn on 17 September by `scripts/run_paper_analysis.py` into `results/decisions_arc_20260916/figures/`, and its LaTeX block and caption are in this guide's FIGURES section under `\label{fig:noise_conditions}`. No sentence in M4 carries a `\ref{fig:noise_conditions}`, and `paper.tex` has no reference to any noise figure at all, so as the text stands the figure would be uploaded and never cited. **A sentence in M4 has to reference it.** The superseded drawing is `fig_methods_noise_strategies.png` in `results/paper_figures_v2/`, dated 8 July; it is not what F1 shows and the label `fig:noise_strategies` is dead.

**For the author.**

- The table is gone and nothing replaces it. Everything it carried is now in the prose: the seven condition names in the first paragraph, the three shapes in the $\sigma_z$ sentence, who gets singled out in the $s_i$ sentence, and the six citations in the parameters paragraph. `tab:regression_noise` is now an undefined label anywhere in paper.tex that still points at it.
- The section is 717 words and two equations, against 1,020 words plus a seven-row table before. It is written in the voice the rest of the Methods uses: nine instances of "we" where the previous version had none, and the conditions called by the prose names the Results already use (grouped-wider, grouped-shifted, outlier, censoring) rather than by their code strings in every sentence. The seven code strings are given once, in the last sentence, so that a reader joining the deposited results rows still has them. Cut outright, with no replacement: the grouped-wider group-selection algorithm, the acyclic-singleton rule, the dose-tolerance warning, the provenance fields written onto every results row, the per-molecule provenance file, and the sentence saying the grouped conditions are keyed to scaffold groups under a scaffold split. Say which of those you want back.
- The last of those is the one real loss. Under a scaffold split whole families are held out, so the grouped conditions have no structure on held-out molecules, and whether a model becomes less certain where the data is unreliable is undefined there. `RERUN_PLAN.md` §3.1d calls that a Methods sentence that has to be written. It belongs in M5 beside the uncertainty question it bounds, not here.
- $\lambda = 3$ is quoted once and used by two conditions. Avdeef 2019 is the source for the between-laboratory ratio, which is what `grouped_wider` stands for. For `outlier_p10` the same value of three is Tukey's conventional contaminated-normal setting, a different justification for the same number, and it is not in the text.
- Hampel2001 is a research report, not a peer-reviewed article, per the note at citations.bib:2239.
- The parameter sources are NOISE_DESIGN.md §3 at :169-175 and §2a at :216-219. I checked that the code matches them: lambda 3.0, group_variance_share 0.62, outlier_p 0.1, group_fraction 0.2 in noise_conditions.json. I have not opened Avdeef 2019, Bentz 2013, Hampel 2001 or Kruger 2012 this session, so the attributions themselves are unchecked here.
- Two errors in the text this replaces, both now gone. It said five targeting rules of which four set a per-molecule scale; it is two, `grouped_wider` and `outlier`, against `_CONSTANT_SCALE_STRATEGIES = ('uniform', 'grouped_shifted')` at NoiseInject/noiseInject/core.py:84 and the `vec![1.0; n]` branches at rust/src/main.rs:541-543 and :673. It also used "targeting rule" as a term of art; that is a name this project coined and no reader would know it.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **seven noise conditions; six draw a random number per molecule, censoring has none** — /Users/apunt/repos/qsar_qm_models/noise_conditions.json (stage_1_full_grid: gaussian, grouped_wider, grouped_shifted, censoring; stage_2_depth_only: student_t_nu5, outlier_p10, laplace). Censoring has no draw: rust/src/main.rs:964 sets unit_dose_g to NaN, "censoring does not go through the dose solver"; NoiseInject/noiseInject/core.py:73 gives it a nominal distribution only.
- **three draw shapes; nu = 5** — rust/src/main.rs:148-166 (NoiseShape Gaussian / StudentT{nu} / Laplace, fn unit_sd); noise_conditions.json settings_that_follow: nu = 5.0
- **two conditions vary the multiplier across molecules; the other five give every molecule 1** — /Users/apunt/repos/NoiseInject/noiseInject/core.py:76 REGRESSION_STRATEGIES = ('uniform', 'grouped_wider', 'grouped_shifted', 'outlier', 'censoring') and :84 _CONSTANT_SCALE_STRATEGIES = ('uniform', 'grouped_shifted'); rust/src/main.rs fn scale_map at :532, with vec![1.0; n] at :541, :543 and :673, lambda at :574 and :667
- **lambda = 3 from Avdeef 2019, p = 0.10 from Hampel 2001, rho = 0.62 from Bentz 2013 Table 7, 20% of families chosen with no published source** — NOISE_DESIGN.md:169-175 (the sourced parameter list, which also gives Llinas & Avdeef 2019 and Kalliokoski 2013 as two further lines for lambda, and Tukey for the contaminated normal) and :216-219 (Bentz Table 7: laboratory 62%, laboratory x experiment 20%, residual 10%, cell line 8%). The four papers were not opened this session
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
- **the roster, six representations, Tanimoto GP on ECFP4 alone; 109 pairs before rf300 was added** — slurm_scripts_qm9_rerun/generate_scripts.py:112 ALL_REPS (six), :113 FP_REPS = ['ecfp4'], :652-745 MODELS ('gauche' at :674-678 carries FP_REPS), giving 18 x 6 + 1 = 109 pairs before rf300. slurm_scripts_validation_rerun/generate_scripts.py:48 ALL_REPS, :292 (GP-Tanimoto restricted to ECFP4). 🔴 That file's own comments still count the roster in words at :17, :20, :28, :314, :647, :1221, :1224, :1232 and :1236, and they are stale now that rf300 exists
- **depth subset: eight models on three representations, 24 pairs, all four datasets** — /Users/apunt/repos/qsar_qm_models/deep_run_pairs.json (eight models, three representations: ecfp4, pdv, chemberta); noise_conditions.json stage_2_depth_only, applies_to includes qm9_deep_run, validation_robustness and uncertainty_runs
- **censoring: five named pairs on all four datasets** — /Users/apunt/repos/qsar_qm_models/censoring_pairs.json (five generator_pairs); noise_conditions.json censoring scope: mode pair_subset, n_pairs 5, applies_to qm9_grid and validation_robustness
- **citation line numbers in citations.bib** — grep of /Users/apunt/repos/qsar_qm_models/citations.bib this session: huber1964robust:1804, Kruger2012:2028, Bentz2013:2107, Avdeef2019:2118, Hampel2001:2231, Svensson2025:2244

</details>


## M5. Uncertainty quantification

*Rewritten 2026-09-23. Replaces the uncertainty subsection at `paper.tex:262–289` and the decomposition
paragraph in Models at `paper.tex:225` ("We decomposed the uncertainty values…"). Delete that Models
paragraph outright: everything it says is here, and it says the BNNs and the QRF were not decomposed,
which the code no longer does.*

```latex
\subsection{Uncertainty quantification}

Probabilistic models have different mechanisms of estimating uncertainty. Some report a single total
variance, while others report one that can be decomposed into an aleatoric and an epistemic component
\citep{kendall2017}. The aleatoric component describes underlying noise the model associates with the
data, while the epistemic component represents the uncertainty within the model's fit. We tested
whether the aleatoric component rises with injected label noise while the epistemic component does not.

For the BNN and VBLL transformations, we took the two components over $T = 100$ stochastic forward
passes \citep{kendall2017}:
\begin{equation}
\hat{u}^2_{\text{ale}}(x) = \frac{1}{T}\sum_{t=1}^{T}\hat{v}_t(x),
\qquad
\hat{u}^2_{\text{epi}}(x) = \frac{1}{T}\sum_{t=1}^{T}\big(\hat{\mu}_t(x) - \bar{\mu}(x)\big)^2 ,
\label{eq:sampling_split}
\end{equation}
where $\hat{\mu}_t(x)$ and $\hat{v}_t(x)$ are the mean and the variance returned for molecule $x$ on
pass $t$, and $\bar{\mu}(x)$ is the mean over passes. Here $\hat{v}_t(x)$ is the predicted observation
noise variance. It varies per molecule in the variance-head and heteroscedastic variants, and is one
number for the whole fit in the VBLL transformation. A full-BNN without a variance head returns no
variance, so its uncertainty is epistemic alone.

The other models follow the same split. For the GPs, the epistemic component is the posterior variance
of the latent function and the aleatoric component is the learned observation noise
\citep{Rasmussen2005}. That noise is one number for the whole fit, except in the heteroscedastic
variant. For the QRF, each tree places a molecule in a leaf
shared with several training molecules. The aleatoric component is the variance of the labels within
each leaf, averaged over the trees. The epistemic component is the variance of the leaf means across
the trees. NGBoost predicts one variance for each molecule from a single fit, which is aleatoric
alone \citep{Duan2020}. In every model the total uncertainty is the sum of the two components.

No held-out label carries injected noise, so every comparison against the injected noise was made on
training molecules. Each training set was divided into five folds grouped on Murcko scaffolds, and each
molecule was scored by a refit that excluded its own fold. For NGBoost on QM9, three of the five folds
were scored to limit its cost. No uncertainty was calibrated after fitting.
```

**What changed from your draft.**

- The QRF equation is gone. It was the same law of total variance as the network equation, written a
  second time. The QRF now gets one sentence, the way Hirschfeld et al. 2020 treat their forest.
- The GPs and NGBoost are described. Your draft had no line for the GP, which is the model the
  Results lean on hardest.
- "In theory … stays stable" became "We tested whether". No source says the epistemic component holds
  still under label noise.
- The last paragraph, about which conditions corrupt specific molecules, is dropped here. The first
  paragraph of §R7 in `PAPER_RESULTS_REWRITE.md` already says it, in the place where the per-molecule
  question is actually asked.

**Checked against the code this session.**

- $T = 100$: `models/model_defaults.py:383`, `'mc_passes': 100`.
- The network split, including no aleatoric term when a network returns no variance:
  `decompose_sampling` in `scripts/uncertainty_decomposition.py`. The VBLL variance is the layer's single
  `noise_var` unless the heteroscedastic form records one per molecule: `models/models.py:1440–1459`.
- The GP split, latent variance plus one likelihood noise: `decompose_gp`. The heteroscedastic GP's
  noise network trained jointly with the GP (the two losses are summed before one backward pass):
  `models/models.py` around line 8615.
- The QRF split over in-bag leaf labels: `decompose_forest`. NGBoost has no epistemic term:
  `decompose_single_distribution`.
- Five scaffold-grouped folds: `oof_predict` in `models/models.py:1598` uses `GroupKFold` on Murcko
  groups. The default is 5 in both job generators. NGBoost scores 3 on QM9:
  `slurm_scripts_qm9_rerun/generate_scripts.py:226`. **Not checked:** that the assay pipeline in KIRBy
  also groups its inner folds on scaffolds.

**Still open.**

- This subsection assumes the Models subsection introduces the six variant models: the two full-BNNs
  with a variance output head, the two heteroscedastic VBLL transformations, the heteroscedastic GP and
  the Tanimoto GP (`scripts/figlib_config.py:528–541`). The current Models text covers only the Tanimoto
  GP. The three sentences that close that gap were given in the chat of 2026-09-23 and are not yet in
  M3. The heteroscedastic VBLL is this project's extension of Harrison et al.'s layer, so it should not
  carry `Harrison2024` alone.


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
the corrupted labels better than the error alone does, with the difference between the two rankings
being what is reported.

The third statistic is the mean predicted uncertainty per configuration, which tracks whether a model
widens as the noise rises. The fourth asks whether the error a model makes is correlated within a
scaffold group. Each of the four is reported against the declaration of whether the component it uses
varies per molecule.

Robustness is summarised by AUC$_\text{norm}$, for which $R^2$ at each noise level is divided by that
configuration's own clean value. The metric is the normalised area under the resulting $R^2$ retention
curve,
$$
\text{AUC}_{\text{norm}} = \frac{1}{\tau_{\max} - \tau_{\min}} \int_{\tau_{\min}}^{\tau_{\max}} \frac{R^2(\tau)}{R^2(0)}\, d\tau.
$$
Despite the name, AUC$_\text{norm}$ is not an area under a receiver operating characteristic curve, and
the curve it integrates is $R^2$ against the amount of noise injected into the training labels. The
integral is evaluated by the trapezoidal rule over the seven noise levels that were run,
$\tau \in \{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$, in the units of Equation~\ref{eq:dose_matching}, and the
divisor is the span of the levels in that curve. Censoring is swept over fractions of labels clipped from
0 to 0.50, and is normalised over its own span instead.

A value near 1 means the configuration kept its clean accuracy, so a higher value is more robust.
AUC$_\text{norm}$ has no upper bound and nothing is clipped, so a configuration that predicts better with
noise added scores above 1, and values above 1.05 are counted and reported.

On QM9 one AUC$_\text{norm}$ is computed per replicate, and the median over the ten replicates is
reported. On the three assay datasets one is computed per scaffold fold, and the median over the five
folds is reported. Those folds are a partition of one dataset rather than repeats of it. A configuration's
replicate whose clean $R^2$ falls below 0.3 is excluded, each replicate judged against its own clean
value. A configuration's median can therefore run over fewer replicates than
the one beside it.

We decomposed the variance in AUC$_\text{norm}$, and in $R^2$ at the noise level of 1.0 that QM9 results
are reported at, into four terms: a model term, a representation term, an interaction term for the pairing of the two,
and a residual term for what is left over. The shares reported are $\eta^2$ values from Type I sequential
sums of squares. Six model configurations are set aside from the decomposition, and from every
comparison that ranks models against each other. Five of the six add a per-molecule noise term to a model
already in the roster, so they train under a different likelihood from the model they are a variant of.
The sixth is the
Tanimoto-kernel Gaussian process, which needs binary vectors and therefore ran on ECFP4 alone.

A noise condition carrying fewer than five models is not decomposed at all. The residual term is the
within-cell variance across the ten QM9 replicates, and each decomposition carries a band from repeating
the fit with one replicate left out. We fitted it on QM9 alone, because the five scaffold folds on the assay datasets
partition one dataset rather than repeating it.

On QM9 we scored whether model rankings on AUC$_\text{norm}$ agree across noise conditions, using
Kendall's coefficient of concordance ($W$), computed within one representation, the one carrying the
most models. It needs at least three models and at least two noise conditions, and is not computed
where either is missing. Deterministic and probabilistic counterparts were compared on AUC$_\text{norm}$ with a
two-sided Wilcoxon signed-rank test at $\alpha = 0.05$, paired on the replicate. The test was run
separately for each representation and noise condition, with no adjustment for multiple comparisons. A
test paired on five values cannot fall below $p = 0.0625$ however large the difference, so it is run on
QM9's ten replicates and not on the five scaffold folds of an assay dataset.
```


Alright here's what I have, I stopped at a certain point, my ANOVA section is old (it's really concerning you didn't use the phrase ANOVA so I stopped paying attention to what you were saying). This was like a 2/10 subsection. I preferred my papers a million times over. You turned what should have been a mathematical reproducible methods section into a creative writing piece. It was really bad and convoluted. I need you to look at what I have, fix the mismatches with additional files and fill in missing info about ANOVA. I kept a lot of my old stuff, so you need to make sure its accurate

```

Predictive accuracy was scored on held-out molecules: the test split on QM9, and the held-out fold of the
five-fold scaffold cross-validation on the three assay datasets. We used the coefficient of determination ($R^2$) and the correlation between predicted and measured values, Pearson's on QM9 and Spearman's on the three assay datasets. 


To evaluate the effect of label noise, we examined performance retention under increasing artificial noise with $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$. For each configuration we recorded $R^2(\sigma)$ and normalised it by the clean-label value to obtain the
retention ratio $R^2(\sigma)/R^2(0)$. Our robustness metric is the normalised area under this retention curve,
$$
\text{AUC}_{\text{norm}} = \frac{1}{\sigma_{\max}} \int_0^{\sigma_{\max}} \frac{R^2(\sigma)}{R^2(0)}\, d\sigma,
$$
evaluated using the trapezoidal rule over eleven noise levels, $\tau \in \{0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5\}$, in the units of Equation~\ref{eq:dose_matching}. Censoring is swept over fractions of labels clipped from
0 to 0.50, and is normalised over its own span instead. A value near 1 means that the predictive accuracy has been retained with increased noise, meaning a higher value is more robust. 

$\text{AUC}_{\text{norm}}$ falls on $[0, 1]$ such that a value near 1 indicates strong predictive performance retention, while lower values imply a faster degradation. Since each curve is normalized by its own clean-label performance, $\text{AUC}_{\text{norm}}$ is not confounded by baseline accuracy. Configurations with baseline R$^2 < 0.3$ were excluded from the robustness analysis (Additional file~5), since performance retention ratios become unstable when the clean-label denominator approaches zero. On QM9 one AUC$_\text{norm}$ is computed per replicate, and the median over the ten replicates is
reported. On the three assay datasets one is computed per scaffold fold, and the median over the five
folds is reported.

To evaluate uncertainty, we used Spearman's rank correlation coefficient ($\rho$) to quantify the relationship between predicted uncertainty ($u_i$) and absolute error ($|y_i - \hat{y}_i|$), as well as between predicted uncertainty and injected noise magnitude. The uncertainty–noise correlation was computed within each fixed noise level $\sigma$. If, on average, predicted uncertainty rises with label noise, it is a good indication that the uncertainty estimates are tracking injected label noise. 

We conducted a separate two-way analysis of variance (ANOVA) decomposition for each noise strategy to identify the relative contributions of molecular representation and model architecture choice on both predictive performance and noise robustness. We sought to answer the question of how much of the variation in performance and robustness is explained by each factor. For a given metric $y$ (either $R^2$ at fixed noise or AUC$_\text{norm}$"):
$$
y_{ijr} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijr},
$$
where $\alpha_i$ represents the effect of model architecture $i$, $\beta_j$ represents the effect of molecular representation $j$, $(\alpha\beta)_{ij}$ is the interaction term, and $\epsilon_{ijr}$ is the residual for replicate $r$.

The proportion of variance explained by each factor was calculated as the effect size $\eta^2$:
$$
\eta^2_{\text{factor}} = \frac{SS_{\text{factor}}}{SS_{\text{total}}},
$$
using Type I (sequential) sum of squares, where $SS$ denotes the sum of squares. Both factors (model architecture and molecular representation) are treated as fixed effects, and the 10 experimental replicates per scenario provide the error term.

ANOVA relies on the assumption that observations both within and across groups are independence. Across the whole set of models and representations in this study, some pairs produced similar predictions that are not independent. This causes the residual degrees of freedom to increase, while the residual sum of squares barely moves. To handle this issue, we computed pairwise Spearman rank correlations between all models and between all representations across noise strategies. Models whose profiles correlated at $\rho > 0.99$ with another model were excluded, including quantile regression forests (QRF), which were very similar to RF, as well as different NN architectures. The GP model was also excluded because different kernels are used for different molecular representations. This was not a problem with SVM as the RBF was used across all representations. 

Across representations, SNS correlated at $\rho > 0.90$ with the ECFP4 fingerprint and was excluded to avoid inflating representation degrees of freedom, as both encode overlapping circular substructure information. We also computed intraclass correlation coefficients ICC(1,1) for all model pairs. ICC(1,1) measures the proportion of total variance attributable to between-subject differences, with values near 1.0 indicating that two models rank configurations almost identically. The full redundancy and ICC tables are reported in Additional files~2--4.
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
that produce a per-molecule uncertainty. NoiseInject operates on label arrays alone and is
therefore independent of the model being trained. Worked examples cover a PDV workflow, a
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

**These are Additional files throughout, which is the journal's own word.** Journal of Cheminformatics is a BMC title and its submission system labels them `Additional file N`. The guide called them Supplementary Materials between 2026-09-20 and 2026-09-21; that was reverted on the author's instruction and nothing in the repository says Supplementary any more. The filename `additional_files.tex`, the two generated block markers inside it and the `\item[...]` label in the availability list all carry the journal's word, and renaming any of those breaks the splice in `scripts/generate_supp_table1.py`.

*R.*eplaces paper.tex 659 and deletes 670

```latex
% ---- replaces paper.tex:659 ----
\item[Additional file 1 (PDF):] \textit{Hyperparameters for every model configuration, on QM9 and on
the three assay datasets, and the descriptors that make up the PDV representation.} Table A lists the default settings, read from \texttt{models/model\_defaults.py},
which both pipelines load. A default is one value, used for all six representations and for every dataset.
The support vector machine uses a radial basis function kernel on all six representations. Table B lists
the tuned settings that replace a default, on ten model-and-dataset pairs: four on QM9, and two on each of
the three assay datasets. The tuned configurations are BNN-$\alpha$, BNN-$\beta$, VBLL-$\alpha$ and
VBLL-$\beta$. Every other configuration trains at the Table A defaults on QM9, and all but two do so on
each assay dataset. A tuned setting is chosen once per model and per
dataset, and is then used for all six representations. Every QM9 results row carries a column naming the
source of its settings. Table C lists the 200 descriptor names that make up the PDV vector, in the order
they are computed. That list is fixed in our code rather than taken from RDKit's current default descriptor
set, which grows between releases, so the same 200 features are built on every dataset and in every run.

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

- 🔴 Table C is new, 2026-09-20: the 200 PDV descriptor names. The Methods said only "200 physicochemical descriptors computed with RDKit's MolecularDescriptorCalculator", which names a calculator rather than a descriptor set, and RDKit's own list grows between releases, so nobody could have rebuilt PDV from the paper. `scripts/generate_supp_table1.py` now parses `DEFAULT_DESCRIPTOR_LIST` out of `scripts/process_and_train.py` and emits the names in four columns, with the RDKit version (2022.09.5) in the caption. Two checks in `scripts/test_supp_table1.py` fail if a name is added to the pipeline and not to the table, or if the version disappears. The Methods sentence in M2 still has to be updated to point at Table C and to say the list is pinned.
- Additional file 12 does not exist. additional_files.tex:16-19 records that it was removed on 2026-09-12, and line 9 says the compiled PDF splits into eleven files. So paper.tex:670 is a deletion, not a rewrite, and paper.tex:197 now cites Additional file 1 instead.

**Cut from the current text.**

- The Additional file 12 caption in full: the deleted file has nothing to caption.
- The claim that nothing generates either file. additional_files.tex:48-50 names scripts/generate_supp_table1.py as the generator and scripts/test_supp_table1.py as its check; both are on disk, dated 12 September.
- "Every QM9 results row records which of the two it used." params_source takes four values in models/models.py: 'default' (:2031), 'tuned' (:2038), 'tuning_trial' (:2054) and 'cli' (:3900). The caption now says the row names the source of its settings, without saying how many sources there are.
- The QM9-only scope on the Additional file 1 title. Table B carries Caco-2, hERG K$_i$ and LogD rows, and additional_files.tex:57 says model_defaults.py is the file both pipelines load; KIRBy alternative_data_noise_robustness.py:100-125 loads it and :398-406 binds sklearn_params from it.
- "the shared defaults", which arrived with a definite article and no antecedent. The caption now says the Table A defaults.

<details><summary>Every number in this subsection, and the file it was read in</summary>

- **the roster as parsed from slurm_scripts_qm9_rerun/generate_scripts.py, MODELS dict, before rf300** — 19 keys (rf, xgboost, lgb, svm, ngboost, dnn, mlp, dnn_bnn_full, mlp_bnn_full, dnn_bnn_full_variational, mlp_bnn_full_variational, qrf, gauche_rbf, gauche, heteroscedastic_gp, dnn_bnn_full_variational_hetero, mlp_bnn_full_variational_hetero, dnn_bnn_full_mve, mlp_bnn_full_mve); matches additional_files.tex:56 "all 19 model configurations"
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

2. **The noise conditions do not separate on Gaussianity. They separate on random error against systematic error.**
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
grouped-shifted noise minus AUC_norm under Gaussian noise varies over a range of 0.127 across the
models in the roster, against 0.013 across the six representations. A factor of ten, from two different calculations. Label noise corrupts the targets and not
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
| Shape does not matter; random error against systematic error does | **§R4 — the paper's differentiator**, F4b, T4 |
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

## §I2. Paragraph 2 (`paper.tex:182`) — hold, two sentences added at the end

Keep the whole paragraph. Add one sentence, because the third paragraph of the replacement depends on it and
the reader has to be told the error is measurable before being told what shape it has.

> Add after *"…may be incorrect or misleading \citep{Walters2023}."*:
>
> Most chemical data sets carry too few repeat measurements of the same compound to characterise the
> distribution a label is drawn from, so a single value stands in for the population mean
> \citep{Kolmar2021}. How large that error is has been measured directly. Repeat measurements of the same compound-target pair
> disagree by about 0.68 log units for pIC$_{50}$, and by about 0.54 log units for pK$_i$
> \citep{Kalliokoski2013, Kramer2012}. Curating a public potency set down to a single assay lowers the mean
> disagreement from 0.50 to 0.27 log units \citep{landrum2024}. Much of the apparent noise in public data may
> therefore be provenance rather than imprecision.

*Provenance: `NOISE_DESIGN.md` §3.1, §3.3. The 0.68 is pIC$_{50}$ and the 0.54 is pK$_i$ — `NOISE_DESIGN.md`
§638 lists the swapped version as a number that must not enter the paper. The Landrum key is `landrum2024`,
lower case, and it is already in `citations.bib` at line 1910 and `refs.bib` at line 2180 — an earlier
version of this note said it was in neither and that was wrong.*

- 🔴 **Moved out of the Models subsection, 2026-09-20.** `paper.tex`'s Models subsection opened with four ideas about label quality. Two of them were already in the Introduction and would have been said twice: that labels are assumed to be true values, which `paper.tex:182` opens with, and that noisy test sets give optimistic assessments, which the same paragraph already says. The other two were nowhere in the Introduction and are now here. §I2 gains the replication point, which is why the measured disagreements below it exist, cited once to `Kolmar2021` rather than twice as the draft did. §I6 gains the point-value-versus-distribution sentence, which is the only thing in the Introduction that says why anyone would expect a probabilistic model to behave differently — its second question asks exactly that. It is written so it does not predict the answer, because §R3 finds the probabilistic machinery mostly does not help.

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
> Most models treat a label as a single value rather than as a draw from a distribution, and the
> probabilistic models in this study are the ones that do not. This research addresses four questions
> about how QSAR models behave under label noise. First, how do the
> molecular representation and the model architecture divide the variance in predictive accuracy, and does
> that division change when the question is how much accuracy a model retains under noise? Second, what does
> the choice of model buy at a realistic amount of label error, and does making a model probabilistic change
> how noise hurts it? Third, does the kind of noise matter, or only the amount, and do the answers reached on
> a computed property hold when the labels are laboratory measurements? Fourth, how do a model's uncertainty
> estimates behave as its training labels are corrupted: does it become less sure, does it attribute the
> added noise to the labels rather than to itself, does its uncertainty still rank which predictions to
> trust, and can it point to which labels were the corrupted ones?

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
> makes that difference visible. The uncertainty decomposition carries the same lesson. Only the Bayesian
> networks with a variance head place added label noise in the aleatoric component on every representation
> and noise condition. For the other models, whether the decomposition works depends on the representation,
> and a shared offset across a scaffold family disturbs it most. Under clipped labels, most models grow more
> confident as their labels lose information.

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

🔴 **TODO — the matched-settings re-run moves numbers in this Results section, and nothing below has
been rewritten for it.** Five models change or appear on all four datasets: `rf300` is new, and
`dnn`, `dnn_bnn_full`, `mlp` and `mlp_bnn_full` move onto shared settings. That is 162 QM9 tasks and
90 laboratory tasks. Anything that ranks models against one another has to be recomputed once those
land — the AUC$_{norm}$-by-model table, the clean-R$^2$ table, the variance decomposition with model
as a factor in §R1, the model heatmaps, and T6. §R3 is the section the re-run exists for and carries
its own note. Untouched: the deep run, censoring and the uncertainty runs, so §R6 and §R7 stand.
None of the five models is in `deep_run_pairs.json`, `censoring_pairs.json` names `rf` at ECFP4
rather than `rf300`, and `uncertainty_pairs.json` carries none of them. Not yet checked: whether
§R4's condition-level medians and §R5's assay numbers are computed over a model set that includes
any of the five.

---

## MOVEMENT 1 — how the choice of model and the choice of representation divide the outcome

### §R1. Variance decomposition *(replaces `paper.tex:387–440`)*

**Scrapped and restarted 2026-09-23.** What was here was written against noise strategies that no
longer exist. This version starts from what is in `paper.tex` today.

**Carried out in `paper.tex` on 2026-09-23**, on the author's instruction to break the
never-edit rule for this one restructure. See "Where everything now sits in `paper.tex`" further
down for the line numbers and what moved where. The list below is what was agreed and done:

- This subsection carries **one figure and one table**: F2, and T3 with a clean-label row at the top.
- **F2b and T3b move down** to "Does it hold on measured labels" (`paper.tex:526`). Comparing the
  four datasets this early is premature; the clean-label comparison for QM9 alone lives in T3's
  first row.
- **F4c comes out of the paper.** It is `fig:robustness` at line 484 and it is a heatmap of printed
  numbers — a table drawn. **T4 replaces it**, carrying all seven noise conditions instead of three,
  and fixes `Table~\ref{tab:robustness}`, which line 497 cites and which exists nowhere.
- **Line 421 moves out** with `fig:grid`, into "Robustness and clean accuracy". It reads F3, not the
  decomposition.
- **Line 439 is deleted.** It repeats the first paragraph.

Net across the Results: the paper goes from ten figures and one table to nine and four.

**The text as it stands, and what happens to it:**

| Where | What it says now | What to do |
|---|---|---|
| 389 | The long opening paragraph | Keep the argument and most of the sentences. Typos: "tength", "beacuse", and "heightened with label." does not finish |
| 391 | Grouped-shifted varies, higher residuals | Keep, expand with the numbers |
| 393–394 | `TODO: wait for new figure` and the clean-label PLACEHOLDER | Becomes a real paragraph once T3's clean row exists; the four-dataset comparison goes to §R5 |
| 397–405 | `fig:variance` | Keep |
| 407–418 | `fig:variance_clean` | **Move to §R5** |
| 421 | "All of the neural networks span a wider range…" | **Move to "Robustness and clean accuracy"**, with `fig:grid` |
| 439 | "The choice of model architecture is instead the largest source…" | **Delete** |

---

**The takeaways this subsection has to land**

- Representation and model matter about equally for accuracy. For robustness, model dominates and
  representation nearly vanishes — its share drops by about two-thirds. Under Gaussian, model takes
  about half the variance in AUC$_{norm}$ and representation under a tenth; for $R^2$ under noise
  the two are close to level, around 29 and 27 per cent. The same two choices swap importance
  depending on what is being asked.
- That drop makes sense: the noise goes into the labels, not the features. The representation still
  decides how much is learnable, so it keeps its share of accuracy. It does not decide how much
  survives corruption, so it loses its share of robustness.
- The pairing of model and representation counts for more in robustness than in accuracy — about
  20 per cent against 14. Picking a representation does matter, but only for particular models, and
  that is a real effect rather than leftover scatter.
- Under grouped-shifted the decomposition largely stops working. The residual takes 56 per cent of
  the robustness variance and 77 per cent of the accuracy variance; model drops to 30 and 7. That
  condition does not just lower scores, it makes them unstable.
- The leftover share is seed-to-seed variation. Even under Gaussian it is about a third of the
  accuracy variance. That caps how much any single comparison of two models or two representations
  can claim.
- Only three conditions can be decomposed at all. The other four did not run on the full cross of
  models and representations, so there is nothing to decompose. One sentence, or the reader wonders.
- 🔴 **TODO — the clean-label comparison for QM9**, from T3's first row, once the figures have been
  regenerated. Which term leads before any noise is added is the comparison the whole subsection
  turns on, and no number for it exists yet.
- The three assay datasets are §R5's business now, not this subsection's.

---

**Each noise condition, for accuracy and for robustness** *(the comparison itself belongs at
`paper.tex:502`, not here — see the second sample block below)*

- **Gaussian** — the reference everything else is read against. Model dominates robustness; model
  and representation split accuracy.
- **Grouped, wider** — indistinguishable from Gaussian on everything: the four variance shares match
  to within a point or two, AUC$_{norm}$ does not move, the curves lie on top of each other. The same
  amount of noise, arranged so whole scaffolds share it, and nothing notices.
- **Grouped, shifted** — the same total noise again, but here it costs. It cost every base model on
  every representation between 0.005 and 0.082 of AUC$_{norm}$, and the decomposition collapses into
  residual. The difference from grouped-wider is that whole scaffolds move as a block rather than the
  noise spreading out inside them. That is the one structural arrangement that bites.
- **Laplace** — heavier tails than Gaussian, no measurable effect.
- **Student-*t* (ν=5)** — heavier tails still, no measurable effect.
- **Outlier (10%)** — all the noise dumped on a tenth of the molecules. Not worse than Gaussian. On
  RF at ECFP4 it is the *least* damaging of the lot at the highest level. 🔴 **TODO — check on F4d
  whether that holds on the other five representations.**
- **Censoring** — the largest effect anywhere in the study, 0.77 to 0.82 of AUC$_{norm}$ against 0.94
  to 0.98 for those same pairings under Gaussian. Its level means a fraction of labels clipped rather
  than a fraction of the spread, so it cannot sit on the same axis as the rest. R19 is where it
  appears at all, which is why R19 goes to the main text.

---

**The curves, and what the area under them hides**

- Every condition stays together until about half the label spread, then separates. That shape is
  what matters and AUC$_{norm}$ flattens it into one number.
- Grouped-shifted peels off first, around level 0.5, and keeps falling.
- Outlier ends highest at level 1.5 on RF at ECFP4 — above Gaussian, Laplace, Student-*t* and
  grouped-wider. Gaussian ends lowest of the five. Small, but the opposite of what a condition that
  concentrates all its damage ought to do.
- Two conditions can share an AUC$_{norm}$ and lose it in different places. The area does not
  distinguish a model that holds flat and then falls off a cliff from one that declines steadily.
- 🔴 **This was unreportable until 2026-09-23.** The per-level $R^2$ was written out for Gaussian
  only, averaged across representations, as a decision aid. The individual curves existed nowhere on
  disk — the only drawing of them was F4b, one model on one representation. Two additions fix it and
  neither needs a new experiment: `r2_by_level.csv` and **F4d**. Both come out of the next figures run.

---

**Two defects found while reading the figures, both fixed 2026-09-23**

- **R19 was dropping three of its seven models.** It ran through the variant-model filter, which
  exists for cross-model comparisons and was taking out GP (het.), BNN-Full-MVE and VBLL-Full-Hetero
  — three of the models the deep run was chosen to include. The caption then reported the surviving
  four as everything that ran.
- **F4c dropped the deep conditions and censoring without saying so.** Its caption now says where
  they went — though F4c is leaving the paper anyway.

**One defect still open.** QM9 ran VBLL-Full-Hetero in the deep run; logD, Caco-2 and hERG ran the
MLP version instead. Both are listed in `deep_run_pairs.json`, added a few days apart, each needing
its own resubmit — it looks like one resubmit reached each pipeline and neither reached both.
Nothing in the paper depends on it yet, but any cross-dataset statement about the deep run would.

---

**SAMPLE TEXT, as LaTeX.** Your paragraphs kept where they work. The first sentence is yours
unchanged and so is most of the first paragraph; what is added is the numbers those sentences
describe in words, the table reference, and the clause saying why only three conditions are
decomposed. Every number traces to `results/decisions_arc/`: T3 for the variance shares,
`auc_norm_qm9.csv` for the rest.

```latex
\subsection{Variance decomposition}

Considering the two main components of a QSAR model, the molecular representation and the model
architecture, we sought to understand how each contributes to both predictive performance and noise
robustness. ANOVA decomposition divides the variance between model architecture, molecular
representation, the pairing of the two, and a residual. We ran this decomposition across three
noise conditions: Gaussian, grouped-wider and grouped-shifted (Figure~\ref{fig:variance},
Table~\ref{tab:variance}). These conditions were chosen as they are the methods by which injected
label noise is related to the feature space, and they are the conditions every model was given on
every representation; the rest were run on a named subset and leave no full grid to decompose. For
predictive accuracy measured in $R^2$, model architecture and molecular representation account for
comparable shares of the variance, 29.1 and 26.6 per cent under Gaussian noise. However, for noise
robustness, quantified by the normalised area under the R$^2$ retention curve (\aucnorm; higher
values indicate greater robustness), model architecture accounts for roughly half of the variance
while molecular representation for under a tenth, 49.5 and 9.2 per cent. This pattern is observed
across noise conditions. This makes sense, as label noise does not impact the features and
increasing it would impact the model's abilities. However, the interaction term between model
architecture and molecular representation is slightly more influential with noise robustness than
with predictive accuracy, 20.4 per cent against 13.7, though it is on par with the residual. This is
because some models are heavily influenced by different sets of features, while others are not, and
this difference is heightened with label noise.

% PLACEHOLDER -- the clean-label comparison, from the first row of
% Table~\ref{tab:variance} once the figures have been regenerated. Before any noise is added,
% [which term leads] accounts for [x] per cent of the variance in accuracy against [y] for
% [the other]. This is the row the argument above turns on: it separates "molecular representation
% governs accuracy" from "molecular representation governs accuracy only once the labels are
% wrong". Whether the assay datasets agree is taken up in Section~\ref{sec:measured}.

While we observe similar patterns between noise conditions, the predictive performance of
grouped-shifted noise varies significantly, resulting in higher overall residuals: 56.3 per cent of
the variance in \aucnorm and 76.6 per cent of the variance in $R^2$, with the model architecture
term falling to 30.5 and 7.1. This indicates that the offset applied to each scaffold may account
for more of the variance in both R$^2$ and \aucnorm. The residual is the variation between
replicates of one identical configuration, differing only in seed, so what this says is that under
a scaffold-wide offset a configuration stops giving a repeatable answer. Which scaffolds are held
out interacts with which scaffolds were shifted, and that draw moves the outcome further than the
choice of model architecture does.

That residual also sets a floor on what the rest of this work can claim. Even under Gaussian noise
it accounts for 20.8 per cent of the variance in \aucnorm and 30.6 per cent of the variance in
$R^2$. A difference between two model architectures, or between two molecular representations, that
is smaller than the spread between seeds of a single configuration is not a difference we report.
```

---

**SAMPLE TEXT for the condition comparison — this goes at `paper.tex:502`.**

That subsection already carries F4b as `fig:conditions`, and it gains R19 as `fig:deep` and T4 as
`tab:robustness`. Your sentence there currently reads:

> On the HOMO--LUMO gap at ECFP4 it moved the median \aucnorm of nineteen models from 0.926 under
> Gaussian noise to 0.905, reaching significance. Group-wider, which widens the error within a
> scaffold family without moving its mean, only shifted the \aucnorm to 0.927.

That is a median over the roster and a count of it. The replacement says more and claims less,
because it holds for each pairing separately:

```latex
However, the shape is not the only aspect of noise we modified. Grouped-shifted noise gives every
scaffold family its own offset added to every label. On the HOMO--LUMO gap it cost every model
architecture robustness, on every molecular representation, between 0.005 and 0.082 of \aucnorm
depending on the pairing (Table~\ref{tab:robustness}). Grouped-wider, which widens the error within
a scaffold family without moving its mean, did not: it moved \aucnorm by between $-0.025$ and
$+0.023$, up for some pairings and down for others. The two conditions deliver the same amount of
noise and differ only in whether a scaffold family takes a common offset or has its own errors
widened, so what costs a model is neither the amount of noise nor its position in the feature
space, but whether a family of related molecules moves together.

The shape of an individual error mattered less still. NGBoost, RF, SVM, GP, GP-Hetero,
BNN-Full-MVE and VBLL-Full-Hetero were each given Laplace, Student-$t$ and outlier noise on ECFP4,
PDV and ChemBERTa (Figure~\ref{fig:deep}). Under each of those shapes every model's \aucnorm sat
within its own replicate-to-replicate spread of its Gaussian value, with GP-Hetero on PDV the
single exception. This coincides with \cite{Heid2023}, who found no difference between Gaussian,
uniform, hyperbolic and bimodal errors drawn at a matched mean and standard deviation. Thus we see
that the shape of the error injected does not change how well a model tolerates it.

Censoring was the exception to all of it. On the pairings it was run on, \aucnorm fell to between
0.77 and 0.82, against 0.94 to 0.98 for those same pairings under Gaussian noise. Censoring is the
one condition that removes information rather than corrupting it, and its level is a fraction of
labels clipped rather than a fraction of the label spread, so it does not sit on the same axis as
the rest and cannot be read as a harsher dose of the same thing.

% PLACEHOLDER -- the curves, once Figure~\ref{fig:conditions_by_rep} and r2_by_level.csv are in
% hand. \aucnorm is an area, and two conditions can share one while losing it in different places.
% On RF at ECFP4 every condition tracks together to about half the label spread and only then
% separates, grouped-shifted peels away first, and outlier noise ends the highest of the rest at
% the largest level -- above Gaussian -- which is the opposite of what a condition that
% concentrates all its damage on a tenth of the molecules ought to do. Whether that holds on the
% other five molecular representations is what the new figure is for.
```

---

**SAMPLE TEXT for §R5, where F2b and T3b now live.** Goes after `fig:assay` at `paper.tex:545`.

```latex
The same decomposition on clean labels puts the three assay datasets beside the HOMO--LUMO gap
(Figure~\ref{fig:variance_clean}, Table~\ref{tab:variance_clean}). Here there is no noise-condition
axis: the clean fit is made once per replicate and every noise condition starts from it.

% PLACEHOLDER -- fill from anova_eta2_clean.csv once the figures have been regenerated. Say
% whether the three assay datasets divide clean accuracy the way the HOMO--LUMO gap does, and if
% they do not, which term moves. The assay bars carry no band: their five scaffold folds partition
% one dataset rather than repeating an experiment, so dropping one in turn does not measure what
% dropping one of the ten QM9 replicates measures.
```
## MOVEMENT 2 — what the choice of model buys, and whether probabilistic machinery helps

### §R2. Robustness and clean accuracy *(rewrites `paper.tex:416–448`)*

**Rewritten 2026-09-25 from the figures themselves.** QM9 only — anything about the assay datasets
belongs in "Robustness on the three assay datasets". Every number below was read off a figure in
`results/decisions_arc/figures/` or checked against `auc_norm_qm9.csv`, and each bullet says which.

---

#### What the figures actually show

**F4a `fig:curves`** — accuracy against noise level, the eight most robust models, ECFP4 and PDV.

- Every line is together until about 0.5 of the label spread and then fans out. That is where your
  existing sentence comes from and the figure supports it plainly.
- **NGBoost's line is flat and sits below everything else on ECFP4** — it starts at 0.705 against
  0.83 to 0.845 for the rest and ends at 0.669 against 0.69 to 0.74. It never crosses anyone. Being
  the most robust model on ECFP4 never makes it the most accurate one at any level tested.
- **On PDV it does cross.** It starts lowest at 0.866 against about 0.90 for the rest, and by 1.5 it
  is the highest line on the panel. The crossover sits between levels 1.0 and 1.15.
- That contrast is the paragraph. Whether a robust-but-weak model ever overtakes a strong-but-
  fragile one is a property of the pairing, not of the model — and the same two panels that were
  drawn to show the models also show that.
- **LightGBM starts highest on ECFP4 at 0.845 and ends lowest of the eight at 0.693.** Your
  existing claim, visible on the panel.
- RF ends highest on ECFP4 at 0.741, the GP second at 0.733.

**R16 `R16_decoupling_ecfp4.png`** — clean accuracy against robustness, one point per model, one
panel per noise condition. **This figure is in no version of the paper and it is the subsection's
argument drawn.**

- The points are not scattered, they are in three groups. Trees and kernels sit top right, good at
  both. The plain and Bayesian neural networks sit bottom right — **the best clean accuracy on the
  panel and the worst robustness.** NN-β has the highest clean $R^2$ of the thirteen at 0.851 and an
  AUC$_{norm}$ of 0.892, near the bottom. NGBoost sits alone top left.
- That is why the Spearman correlation is $-0.18$ and misses significance. It is not "no
  relationship" — it is two groups with opposite characters plus one outlier, and a rank correlation
  over that returns nearly nothing. **Reporting the $-0.18$ without saying what the picture looks
  like throws away the finding.**
- Panel b) is the same arrangement pushed down. Under grouped-shifted the correlation strengthens to
  $-0.50$ at $p = 0.082$, still short of significance with thirteen models.

**F3 `fig:grid`** — robustness by model and representation. **It sits in this subsection now and no
sentence refers to it.** The paragraph that read it was deleted at some point and it was right:

- **The six neural models are the six widest rows.** Their spread across the six representations
  runs 0.039 to 0.063 of AUC$_{norm}$; every tree and kernel model is between 0.015 and 0.030.
- Between ECFP4 and PDV specifically, the four plain and Bayesian networks give up 0.043, 0.045,
  0.054 and 0.063. Every tree and kernel model moves by 0.011 or less, and three of them move the
  other way.
- **The two variational networks do not follow the family**, moving 0.009 and 0.019 between those
  two representations. So "neural networks are representation-sensitive" is not a clean family
  statement and should not be written as one.

---

#### The takeaways this subsection has to land

- Robustness and clean accuracy are not the same ranking, and the subsection's job is to say how
  they differ rather than that they are uncorrelated.
- **The shape of the disagreement is the finding.** Neural networks buy clean accuracy and pay for
  it under noise; trees and kernels give up a little accuracy and keep it. NGBoost is a third thing
  again, flat and weak.
- **NGBoost is first on AUC$_{norm}$ on all six representations under Gaussian, with RF second on
  all six.** Checked in `auc_norm_qm9.csv`, not read off a picture.
- **NGBoost is last of the thirteen base models on clean $R^2$ on four of the six representations**,
  12th on PDV and 11th on MHG-GNN. Your text says "one of the worst", which is softer than what the
  data supports.
- **NGBoost's lead is Gaussian-specific.** Under grouped-shifted, RF takes first place on ECFP4,
  Avalon and ChemBERTa; NGBoost keeps it on PDV, MHG-GNN and Sort & Slice. A sentence built on
  NGBoost being the most robust model needs "under Gaussian noise" in it.
- **The metric divides the clean baseline out, so part of the decoupling is arithmetic.** R16's own
  caption says it: a model with a weak baseline scores well by having less to lose, and NGBoost is
  exactly that model. This caveat has to be in the paragraph, not left for a reader to find. It is
  also the reason F4a matters — the crossover on PDV is in raw $R^2$ and owes nothing to the ratio.
- Whether the choice of representation moves a model's robustness is a question about families, and
  it splits: trees and kernels barely move, plain and Bayesian networks move a lot, variational
  networks do not. That is F3's paragraph and it needs writing again.

---

#### Claims in the current text, checked

| Claim | Verdict |
|---|---|
| All models lose accuracy as labels are corrupted | **True on QM9** — no AUC$_{norm}$ exceeds 1. Do not generalise; the exceptions are on the assay datasets and belong in that subsection |
| Loss stays small until about half the label spread | **True**, visible on both F4a panels |
| NGBoost has the highest AUC$_{norm}$ across all representations | **True**, first on all six — add "under Gaussian noise" |
| NGBoost clean $R^2$ 0.706 on ECFP4 to 0.865 on PDV | **True** |
| One of the worst on clean data | **True and understated** — last on four of six |
| No other model has such a wide range across representations | **True**, 0.159 against 0.114 for the next. The margin is thinner than the sentence implies |
| LightGBM good early, drops sharply after 1.0, ends below most | **True** on ECFP4 — ends lowest of the eight drawn |
| Spearman $-0.18$, not significant | **True** — Gaussian, ECFP4, thirteen models, $p = 0.57$ |
| Rescaled within each dataset, Spearman $-0.35$ | 🔴 **No source.** Nothing in the results is a $-0.35$ correlation between clean $R^2$ and AUC$_{norm}$. The nearest number is $+0.356$, the median rank agreement between QM9 and the assay datasets over 83 combinations — a different question, positive, and belonging to the assay subsection. **Do not use it** |
| "predictive accuracy is not just a poor indicator of noise robustness," | Sentence ends on a comma |

---

#### Figures and tables this subsection should carry

| Slot | State | Call |
|---|---|---|
| F4a `fig:curves` | in the paper | keep — the crossover on PDV is the best evidence in the subsection |
| F3 `fig:grid` | in the paper, unreferenced | keep, and write its paragraph |
| **R16** `R16_decoupling_ecfp4.png` | generated, never in the paper | **this is the argument drawn.** My call would be to promote it; without it the $-0.18$ is a number with no picture |
| T4 `tab:robustness` | in the condition subsection | cite it from here too — it is where clean $R^2$ sits beside AUC$_{norm}$ for every model |
| R18 `R18_ecfp4_against_pdv_gaussian.png` | generated, never in the paper | optional. Robustness on one representation against another, paired on the model, Spearman 0.51. Says what holding one representation costs |

🔴 **TODO — the per-level numbers.** Everything above about curve shape is read off F4a. The table
behind it, `r2_by_level.csv`, does not exist until the next figures run. Nothing in this subsection
should quote a level-by-level number from a picture.

---
#### SAMPLE TEXT

One block, one argument: clean accuracy does not predict robustness (R16), robustness only pays off
if you started close enough (F4a), and how much the representation matters depends on the family
(F3). Your sentences kept wherever they work. Numbers read off F4a are marked in the note below the
block; everything else is from `auc_norm_qm9.csv`. R16 is cited as `fig:decoupling` and is not yet in
the paper.

```latex
\subsection{Robustness and clean accuracy}

The variance decomposition tells us what aspect of QSAR modeling moves the needle on accuracy and
robustness, however it does not tell us about the characteristics of individual models. All models
lose accuracy as the labels get increasingly corrupted with noise (Figure~\ref{fig:curves},
Table~\ref{tab:robustness}). The loss remains small until the artificial noise reaches about half
the spread of the clean training labels, and then it begins to drop. The rate it drops and the
quantity of noise retained is not necessarily correlated to the model's predictive accuracy on clean
labels. Comparing clean $R^2$ against \aucnorm across all models for Gaussian noise with ECFP4, the
two rank in opposite directions at a Spearman correlation of $-0.18$, which does not reach
significance.

\begin{table}[htbp]
\centering
\caption{Robustness (\aucnorm) of each model under each noise condition on the QM9 HOMO--LUMO gap at
ECFP4, with clean $R^2$ in the first column as the quantity the rest are a fraction of. A dash is a
pairing that condition was not run on: Laplace, Student-$t$ and outlier noise were given to a named
subset of model architectures, and censoring to a named subset of pairings. Values are medians over
replicates.}
\label{tab:robustness}
\input{T4_robustness_qm9_ecfp4}
\end{table}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F4a_models_under_noise.png}
\caption{Predictive accuracy against label noise on the QM9 HOMO--LUMO gap under Gaussian noise, for
the eight models with the highest \aucnorm at ECFP4. a) ECFP4; b) PDV, drawn for the same
eight models, so panel b) is not a separate selection. Bottom axis: the noise level, as a fraction
of the spread of the clean training labels. Side axis: $R^2$ on held-out molecules, as the median
over ten replicates. The dashed vertical line marks the noise level every table in this paper
reports at.}
\label{fig:curves}
\end{figure}

That correlation is close to zero because the models do not sit on a line, but in three groups
(Figure~\ref{fig:decoupling}). The forests, the boosted trees and the kernel models occupy the top
right of the panel: strong on clean labels and strong under noise. The plain and Bayesian neural
networks occupy the bottom right, holding the highest clean accuracy on the panel and the lowest
robustness with it. NN-$\beta$ is the clearest case, with the highest clean $R^2$ of the thirteen
base models at 0.851 and an \aucnorm of 0.892, near the bottom of the panel. NGBoost sits alone in
the top left. A rank correlation over two groups of opposite character and one outlier returns
almost nothing, so the number understates a real pattern: neural networks buy clean accuracy and
give it back under noise, while trees and kernels give up a little accuracy and keep it.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R16_decoupling_ecfp4.png}
\caption{Predictive accuracy on clean labels against robustness, on the QM9 HOMO--LUMO gap at ECFP4,
under a) Gaussian and b) grouped-shifted noise. One point is one of the thirteen base models; marker
shape and colour name the model. Bottom axis: $R^2$ with no noise added to the training labels. Side
axis: \aucnorm, on one scale across both panels. The Spearman correlation between the two axes is
$-0.18$ ($p = 0.57$) under Gaussian noise and $-0.50$ ($p = 0.082$) under grouped-shifted noise, over
thirteen models in each case. \aucnorm divides each model's accuracy under noise by its own clean
accuracy, so a model with a low clean $R^2$ has less to give up and scores higher for it; the side
axis should be read together with the bottom one rather than alone.}
\label{fig:decoupling}
\end{figure}

NGBoost is the standout case. Under Gaussian noise it holds the highest \aucnorm on every one of the
six molecular representations, with the random forest second on all six. However, it is one of the
worst performing models with respect to $R^2$ on clean data, ranging from 0.706 on ECFP4 to 0.865 on
PDV, and it is the least accurate of the thirteen base models on four of the six representations. No
other model displayed such a wide range of predictive performance across representations. Because
\aucnorm is the area under a curve divided by its own clean starting point, a model that begins
poorly has less to lose and scores well for it, so NGBoost's position is partly a property of the
metric. The curves show what it means in absolute terms. On ECFP4, NGBoost begins below every other
model drawn and remains below them at every noise level tested, so its robustness never makes it the
more accurate choice. On PDV it begins lowest and ends highest, overtaking the rest between noise
levels of 1.0 and 1.5. Robustness therefore only pays when a model starts close enough to those it
is being compared against, and whether it does is a property of the pairing rather than of the model.
The ranking is also specific to the noise condition: under grouped-shifted noise the random forest
takes first place on ECFP4, Avalon and ChemBERTa, and NGBoost keeps it only on PDV, MHG-GNN and Sort
\& Slice. LightGBM tells a different story again, performing well under smaller quantities of label
noise, yet dropping sharply beyond a noise level of 1.0 to end below every other model drawn on
ECFP4.

Which molecular representation a model is given matters to its robustness, but only for some
families (Figure~\ref{fig:grid}). The six neural models span the six widest ranges of \aucnorm across
the six representations, between 0.039 and 0.063, while every tree and kernel model stays within
0.030. Between ECFP4 and PDV in particular, the four plain and Bayesian networks give up between
0.043 and 0.063 of \aucnorm, where no tree or kernel model moves by more than 0.011 and three of them
move in the opposite direction. This does not hold for the family as a whole: the two variational
networks move by 0.009 and 0.019 between the same two representations, no more than the trees. So
while molecular representation is not the driving factor in whether a model is noise robust, the
choice of which representation to use matters a great deal for particular architectures, and the
neural networks that are most sensitive to it are the same ones that gave up the most robustness in
the first place.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F3_model_by_representation.png}
\caption{Robustness (\aucnorm) of the thirteen base models on the QM9 HOMO--LUMO gap under a)
Gaussian and b) grouped-shifted noise. Rows are models, ordered by family; columns are the six
molecular representations. Colour is fixed across both panels, its bright end the higher \aucnorm.
Grouped-wider repeats panel a) and is in an additional file. Values are medians over ten replicates,
excluded configurations in Additional file~5.}
\label{fig:grid}
\end{figure}
```

**Author notes.**

- `fig:decoupling` is R16, `R16_decoupling_ecfp4.png`. It is generated and has never been in the
  paper. The second paragraph does not work without it.
- **`tab:robustness` is T4 and now belongs to this subsection** (the author's call, 2026-09-25).
  It is the only place clean $R^2$ sits beside \aucnorm for every model, which is this
  subsection's whole argument. In `paper.tex` the block currently sits in "Artificial noise
  conditions" and has to be cut from there and pasted here; that subsection keeps citing it,
  which is fine, a table belongs in one place. **`paper.tex` has not been touched** — the move
  is one block and it is yours to make or to tell me to make.
- Read off Figure~\ref{fig:curves} rather than from a table: that NGBoost stays below every other
  line on ECFP4 at every level, and that it ends highest on PDV between 1.0 and 1.5. Confirm both
  against `r2_by_level.csv` when the next figures run produces it, and replace "between noise levels
  of 1.0 and 1.5" with the level it actually crosses at.
- Dropped from the old text: the Spearman of $-0.35$ after rescaling within each data set. No such
  number exists in the results. The nearest is $+0.356$, the median rank agreement between QM9 and
  the assay datasets over 83 combinations, which is a different question and belongs in the assay
  subsection.
- Dropped: "That correlation is taken over pairings drawn from four datasets" and the twelve-pairings
  sentence that cites `tab:pairs`. Both are about all four datasets, so they belong in the assay
  subsection. `tab:pairs` is T8, `T8_pairs_across_datasets.tex`, generated and defined nowhere in
  `paper.tex`.
- The unfinished sentence "predictive accuracy is not just a poor indicator of noise robustness,"
  is replaced by the second paragraph's closing sentence, which says what it was reaching for.
- "All models lose accuracy" is left as it stands because it is true on QM9. Four cells exceed an
  \aucnorm of 1 on the assay datasets, every one with a clean $R^2$ near 0.35; that exception belongs
  in the assay subsection.
---

### §R3. Does making a model probabilistic help *(replaces `paper.tex:468–497`)*

🔴 **TODO — every number in this section is superseded and nothing below has been rewritten.** The
re-run forces `dnn` and `dnn_bnn_full` onto one setting, and `mlp` and `mlp_bnn_full` onto another,
and adds `rf300`, a 300-tree ordinary forest whose settings are identical to the quantile forest's.
Three of the eight paired comparisons here become clean once it lands: `dnn` against
`dnn_bnn_full`, `mlp` against `mlp_bnn_full`, and `rf300` against `qrf`. The 5-up / 12-down count,
the per-pair numbers, the Wilcoxon results and Figure~\ref{fig:variants} all have to be recomputed
from the new rows, and the figure needs `rf300` drawn in place of `rf`. What stays true without a
re-run: the comparisons that set one probabilistic model against another, such as `dnn_bnn_full`
against `dnn_vbll`, still differ in settings as well as in machinery, because only the direct pairs
were forced onto a shared setting. The text below is the pre-re-run version, kept so the structure
and the sentences that do not depend on the numbers can be reused.


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
but it is a different quantity — the range across the roster of the cost of moving from Gaussian noise
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
that the error is *systematic* rather than random, which is the case nobody tests. Methods already introduces the term
(`paper.tex:236`); the Results should use it rather than paraphrase it.

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
> Gaussian against non-Gaussian, but random error against systematic error.
>
> The fall in AUC$_{norm}$ on moving from Gaussian noise to grouped-shifted is decided by the model far more
> than by the representation. Each model's fall is the median over the replicates, over the six
> representations, and over the computed property and the three measured endpoints alike. That fall ranges
> across 0.127 between the configurations in the roster and across 0.095 between the base models, from a loss of
> 0.101 for the Gaussian process to a gain of 0.025 for the heteroscedastic variational network. The wider
> range covers every configuration,
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
uses the whole roster because this is a statement about the roster rather than about the decomposition, which
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

### §R5. Robustness on the three assay datasets *(replaces `paper.tex:506–593`; rewritten 2026-09-25)*

**How this block is built.** It starts from your current `paper.tex`, not from the old §R5 text in this
guide or in `PAPER_RESULTS_REWRITE.md`. Your opening paragraph and Figure~\ref{fig:assay} are word for
word from `paper.tex:506–527`. So are your clean-accuracy paragraph with its figure and table
(`paper.tex:532–556`), which still waits on the regenerated figure, and the R9 figure block
(`paper.tex:561–571`). New paragraphs sit between `% ---- NEW` and `% ---- end NEW`, and they follow
your comments: the curves, the noise conditions, the models one by one, and a quick reading of the
rank transfer. Your text they replace is kept at the end.

**Two new outputs, not yet on disk.** The figure script now draws F8c (the curves on all four datasets,
one file per representation) and writes T10 (held-out R² at level 1.0). Both need the analysis
re-run. Every sentence that depends on them is a PLACEHOLDER comment.

**Model names.** "The thirteen base models" and every rank below exclude the six variants
(variance-head BNNs, heteroscedastic VBLLs, the heteroscedastic GP and the Tanimoto GP), as
Figure~\ref{fig:assay} does. The rank-transfer paragraph uses Figure~\ref{fig:transfer}'s ranks, which
are out of nineteen.

```latex
\subsection{Robustness on the three assay datasets}

So far, we have only discussed results coming on HOMO--LUMO gap from QM9, a computed property whose labels carry no measurement error. We repeated these experiments on three experimentally-obtained data sets, 
LogD, Caco-2 \citep{openadmet}, and hERG Ki \citep{Zdrazil2023}, replicating our process of adding artificial noise. Unlike QM9, the experimental endpoints carry their own measurement noise in both training and test labels; the injected artificial noise therefore adds on top of an unknown noise floor. Repeat measurements of the same compound disagree by about 0.54 log units for pK$_i$ \citep{Kalliokoski2013, Kramer2012}. The hERG $K_i$ labels have a spread of 0.915 log units, so a noise level of 0.6 of that spread is about one unit of that estimated laboratory error. Each model's \aucnorm under each noise condition on the three endpoints is given with its clean R$^2$ beside it (Figure~\ref{fig:assay}). 

\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth]{F8_assay_datasets_logd.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_caco2.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_herg.png}
\caption{Robustness (\aucnorm) on the three assay datasets, on the ECFP4 representation,
under the three noise conditions that ran on every model: a) logD; b) Caco-2; c) hERG K$_i$. Rows
are the thirteen base models, ordered by family, with the six variant models in Additional file~8.
The first column of each panel is clean R$^2$ and is left uncoloured; the brightest row on panel b)
has the lowest clean R$^2$ on that panel. Each value is the median over the scaffold folds that met
the clean-accuracy gate, the lowest clean R$^2$ a fold may have and still be scored (Additional
file~5). Colour runs on one fixed range across the three panels. That range differs in span
from the range in Figure~\ref{fig:grid} by a factor of three, so colour is not comparable between
the two figures. Censoring is absent because it runs on a named subset of pairs and cannot rank
models.}
\label{fig:assay}
\end{figure*}

On clean labels, model architecture and molecular representation divide the variance in predictive accuracy differently from how they divide it once noise is present (Figure~\ref{fig:variance_clean}, Table~\ref{tab:variance_clean}). PLACEHOLDER: state which term leads on QM9 and whether the three assay datasets agree, once the figure has been regenerated.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F2b_clean_decomposition.png}
\caption{Share of the variance in predictive accuracy on clean labels explained by model architecture,
molecular representation, their pairing, and the residual, on all four datasets. Accuracy is $R^2$ on
held-out molecules with no noise added to the training labels. The bottom axis is the dataset, the side
axis the share of variance; the four shares within a dataset sum to 100\%. There is no noise-condition
axis: the clean fit is made once per replicate and every noise condition starts from it. Whiskers are a
leave-one-replicate-out jackknife, not confidence intervals. Thirteen base models by six representations,
ten replicates on QM9 and five on each assay dataset.}
\label{fig:variance_clean}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Share of the variance in predictive accuracy on clean labels explained by model
architecture, molecular representation, their pairing and the residual, one row per dataset. The
HOMO--LUMO gap row is the same decomposition as the first row of Table~\ref{tab:variance}. The assay
datasets carry no band, as their five scaffold folds partition one dataset rather than repeating an
experiment.}
\label{tab:variance_clean}
\input{T3b_variance_decomposition_clean}
\end{table}

% ---- NEW: curves and accuracy at one level (placeholders wait on the re-run) ----
Figure~\ref{fig:assay_curves} gives held-out accuracy against the noise level on all four datasets,
one row per dataset and one column per noise condition. Table~\ref{tab:accuracy_at_level} gives the
same accuracy at one noise level, one spread of the clean training labels.
% PLACEHOLDER (F8c, T10): on QM9 the loss stays small until the noise reaches about half the label
% spread, then steepens. Say whether that shape holds on logD, Caco-2 and hERG K_i, dataset by dataset.
% PLACEHOLDER (F8c): on QM9 LightGBM holds up at low noise and falls sharply after level 1.0. Say
% whether it falls early or late on Caco-2 and hERG K_i, where its AUC_norm is lowest.
% PLACEHOLDER (F8c, column d): your note at paper.tex:452 says outlier noise tracks Gaussian up to
% about 0.4 on QM9 and then falls away. Say whether the assay datasets show the same break.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F8c_curves_every_dataset_ecfp4.png}
\caption{Held-out accuracy against label noise on every dataset, at ECFP4. Rows are datasets and
columns noise conditions: a--d) QM9; e--h) logD; i--l) Caco-2; m--p) hERG $K_i$. One line per base
model, the median over ten replicates on QM9 and over the scaffold folds on the assay datasets. The
bottom axis is the noise level, a fraction of the clean training label spread. Each row has its own
side axis. Outlier noise ran on a named subset of models, so its column has fewer lines. The same
figure at PDV is in Additional file~X.}
\label{fig:assay_curves}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Held-out $R^2$ with no added noise and at a noise level of 1.0, on ECFP4. One row per base
model; for each dataset the clean value, then one column per noise condition. Medians over ten
replicates on QM9 and over the scaffold folds on the assay datasets. [TODO: final caption once the
table is in.]}
\label{tab:accuracy_at_level}
% [T10_accuracy_at_level_ecfp4.tex]
\end{table}
% ---- end NEW ----

% ---- NEW: the noise conditions on measured labels ----
Grouped-shifted noise is the hardest condition on every dataset. For RF, QRF, XGBoost, NGBoost, SVM
and the GP, it lowers \aucnorm below its Gaussian value on all four datasets and all six
representations (Figure~\ref{fig:assay}). LightGBM is the one tree model with exceptions, on Caco-2
at Avalon and MHG-GNN, where it is 0.02 higher. The cost grows as the dataset gets harder. For those
seven models, grouped-shifted costs 0.02 to 0.06 of \aucnorm on QM9 and 0.02 to 0.11 on logD. On
Caco-2 and hERG $K_i$ it costs up to 0.22 and 0.21. The two VBLL networks are the clear exception:
on logD they keep more \aucnorm under grouped-shifted than under Gaussian noise, on every
representation.

The other noise conditions separate the models in a way that holds on every dataset. Grouped-wider,
outlier, Laplace and Student-$t$ noise put the same total amount of error on fewer labels than
Gaussian noise does. Against Gaussian noise, SVM never loses \aucnorm under any of the four, on any dataset or
representation that ran them. Its gain is 0.01 to 0.02 on QM9 and reaches 0.10 on the assay
datasets. NGBoost moves the other way. On Caco-2 and hERG $K_i$ it loses \aucnorm under all four, by
up to 0.15, with one exception at Avalon on Caco-2. On QM9 and logD it moves by 0.04 or less. RF and
the GP never lose more than 0.03 under any of the four. Under grouped-wider noise, RF gains up to
0.09 on Caco-2.
% Grouped-wider and the boosted trees, checked because the author's first reading was that it hurts
% XGBoost, LightGBM and NGBoost: it hurts NGBoost (above). XGBoost loses 0.09 to 0.14 on hERG at
% ECFP4, MHG-GNN and Sort & Slice, and moves 0.08 or less elsewhere. LightGBM does not lose: it
% gains 0.03 to 0.10 on Caco-2 and hERG at most representations, none of it beyond the fold spread.
% TODO (reading, not tested): SVM's loss grows linearly with a residual, so a few large errors pull
% it less than many moderate ones; NGBoost fits a Normal likelihood per molecule. Whether that is
% the mechanism is not tested. Decide whether it goes in the text.
% ---- end NEW ----

% ---- NEW: which models, on measured labels ----
The models that hold up on the computed property do not all hold up on measured labels. RF is the
most consistent. Among the thirteen base models it ranks between first and eighth for \aucnorm under
Gaussian noise on every dataset and representation. It is first on logD and second on QM9 at every
representation. Its clean $R^2$ sits in the middle of the thirteen or below. The GP is the most
accurate model on clean labels on all three assay datasets, first or second at every
representation, and ranks third to eighth for \aucnorm. On the assay datasets SVM never leads either
list and never falls to the bottom of either. It ranks third to tenth for clean $R^2$ and fourth to
ninth for \aucnorm.

NGBoost, the most robust model on QM9 at every representation, does not stay there. It ranks first
to third on logD, second to eleventh on Caco-2 and sixth to ninth on hERG $K_i$. Its clean $R^2$
stays among the lowest, except on Caco-2 at Avalon and ChemBERTa. At PDV, where it stood out on QM9 with an \aucnorm of 0.98,
it keeps 0.94 on logD, 0.84 on Caco-2 and 0.88 on hERG $K_i$. RF at PDV keeps 0.94, 0.88 and 0.92 on
the same three. LightGBM is the least consistent of the tree models. On Caco-2 and hERG $K_i$ it
ranks tenth to thirteenth for \aucnorm at every representation, while being among the most accurate
on clean labels at most of them. VBLL-$\alpha$ is the reverse case. It ranks first or second for
\aucnorm on hERG $K_i$ at every representation and last on logD at every representation. Its clean
$R^2$ is among the lowest everywhere except Caco-2 at Avalon. An \aucnorm close to 1 on a weak baseline is partly the ratio
itself, since a model with little to lose loses little.
% TODO: plain NNs and full-BNNs vary by dataset and representation with no single pattern. Say so
% in one sentence, or leave them to the figure.

Clean accuracy does not predict robustness on the assay datasets either. The rank correlation
between clean $R^2$ and \aucnorm across the thirteen base models is never positive and significant.
On Caco-2 it is negative at all six representations, from $-0.32$ to $-0.66$, and significant at
ECFP4 and Sort \& Slice. On logD it lies between $-0.23$ and $0.16$.
% ---- end NEW ----

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R9_rank_transfer_ecfp4.png}
\caption{Where each model ranks for robustness on the QM9 HOMO--LUMO gap and on each of the three
assay datasets, at ECFP4 under Gaussian noise. Rows are models, ordered by their rank on the
HOMO--LUMO gap with the most robust at the top. The bottom axis is the rank by
\aucnorm within a dataset, where 1 is the most robust of the nineteen models ranked.
Colour and shape together name the dataset. The thirteen base models are drawn and the six variant
models are not, so some ranks on the axis belong to a model that has no row.}
\label{fig:transfer}
\end{figure}

% ---- NEW: the rank transfer, read quickly ----
Figure~\ref{fig:transfer} puts the four robustness rankings side by side at ECFP4, and it shows the
pattern above model by model. The forests keep their place: RF and QRF are in the top five of
nineteen on all four datasets. NGBoost slides from first on QM9 to seventh on hERG $K_i$. The
boosted trees fall furthest. XGBoost goes from fourth on QM9 to fourteenth and fifteenth on Caco-2
and hERG $K_i$, and LightGBM from sixth to last on both. VBLL-$\alpha$ goes the other way, from
tenth to first on both. A robustness ranking made on the computed property carries for the forests,
but not for the boosted trees or the networks.

For a user who suspects label noise, RF is the safe default among the models tested. It never
ranks below eighth of the thirteen base models for robustness under Gaussian noise, on any dataset
or representation, at a middling clean accuracy. SVM is the alternative when a minority of badly
wrong labels is expected, since it never loses under the conditions that concentrate the error.
% ---- end NEW ----

```

**Where every number comes from.** All from `results/decisions_arc_20260916/`, computed this session
and checked against the fold ranges (`auc_norm_lo`, `auc_norm_hi`):
- ranks among the thirteen base models, the grouped-shifted costs, the concentrated-noise gains and
  losses, and the NGBoost and RF values at PDV: `auc_norm_assay.csv` and `auc_norm_qm9.csv`;
- the clean-against-robust correlations: the same two files, `baseline_r2` against `auc_norm`, one
  dataset and one representation at a time, Gaussian noise;
- the rank-transfer ranks: `d9_rank_transfer.csv` at ECFP4, Gaussian, as drawn in R9.

No number is averaged across models, representations or datasets. The fold spread on the assay
datasets is large: the median spread of \aucnorm across folds is 0.09 over every cell in both files. Many
single differences of a few hundredths are inside it, which is why the text leans on patterns that
repeat across representations and datasets.

**TODOs, not paper text:**
- 🔴 **The re-run.** F8c and T10 do not exist until the analysis runs again (same command as §R6). After
  it, fill the three PLACEHOLDER comments in the curves paragraph and re-check every number above.
- **What to look at in F8c when it lands:** whether the QM9 shape holds on each assay dataset, where
  LightGBM loses its accuracy, and whether outlier noise breaks away from Gaussian at about 0.4 on
  the assay datasets as your note says it does on QM9. The PDV version answers the NGBoost question
  in curves rather than in \aucnorm.
- **Why SVM gains and NGBoost loses under concentrated noise** is a reading, not a test. It sits in a
  comment in the noise-conditions paragraph. Yours to decide whether it goes in.
- **Grouped-wider and the boosted trees.** Your reading was that it hurts XGBoost, LightGBM and
  NGBoost, and leaves RF and QRF alone. The data agree for NGBoost, and for RF and QRF, which gain
  slightly. XGBoost loses only on hERG $ at three representations. LightGBM does not lose at all.
- **The clean-accuracy paragraph** (`paper.tex:532`) still waits on the regenerated F2b.
- **Additional file number** for the PDV version of F8c.
- **Plain NNs and full-BNNs** have no single pattern across datasets. One sentence, or leave them to the
  figure.

**Your text this replaces** (`paper.tex:559`, `577–581` and `584–593`), kept word for word. The
paragraphs at `584–593` read as conclusions for the whole paper, so they may belong in the
Conclusion rather than here.

> The different noise conditions display the same relative performance across these additional datasets. Grouped-shifted resulted in the lowest overall \aucnorm
>
> That correlation is taken over pairings drawn from four datasets;
> it says the orderings disagree rather than by how much on any one endpoint. Nine of the twelve
> pairings that rank highest on both are a Gaussian process or a forest, one pairing to a row of
> Table~\ref{tab:pairs}. The two orderings disagree wherever they were tested, and not only at
> NGBoost.
>
> Per-dataset NDS heatmaps across all model architectures and noise strategies are shown in Figure~\ref{fig:validation_overview}. The ANOVA on these additional datasets confirms that model architecture dominates robustness variance on all three datasets, as seen in Additional file~10, and that the trends with respect to noise robustness generalize. NGBoost ranks first on both QM9 and external datasets, SVM also transfer quite well. Some models' predictive capabilities are limited on these external datasets, likely due to a combination of smaller training sets, narrower chemical coverage, and the unknown experimental noise in the labels. XGBoost and BNN variants suffer the most here, (Figure~\ref{fig:validation_combined}a).
> 
> QRF was consistently less robust than RF on every external data set (Additional file~11). As seen with QM9, choice of molecular representation has a minimal effect on noise robustness across all three external datasets, reinforcing the finding that model architecture, not representation, is the main driver of noise robustness. 
> 
> These results lead to two main conclusions. The first is that noise robustness, defined as the rate at which predictive performance degrades with increasing label noise, is primarily determined by the model's training mechanism. The margin maximization of SVM, the weight priors of BNNs, and the ensembling in tree methods all appear to keep those models from fitting to noise. The choice of representation has a stronger influence on predictive performance than noise robustness, accounting for less than \~10\% of the variance in robustness. The interaction term between representation and model has a stronger impact; some models like SVM and full BNNs remain robust across all representations, while RF and NNs are robust only with certain ones. While the pattern of noise has a strong impact on predictive performance, it does not influence noise robustness rankings between models or representations much.
> 
> The second conclusion concerns whether a model's uncertainty estimates track per-sample label noise. Some models struggle, particularly those like BNNs whose uncertainty comes from a posterior over weights or QRF which does produce a full quantile, but is unable to successfully track noise. 
> Models that fit a separate, per-sample scale or observation-noise parameter during training are able to track that label noise more effectively. NGBoost identifies noisy labels through a per-sample predicted scale that absorbs large residuals. Similarly, the GP absorbs these larger residuals with a per-sample posterior variance. Among representations, uncertainty tracking collapses with the learned embeddings MHG-GNN and mol2vec, while fingerprints and descriptors produce a viable signal. Neither strong predictive performance nor noise robustness implies noise tracking; again it is up to the model's architecture and how it handles loss. 
> 
> Although the choice of representation carries less weight than that of models, we observed an interesting phenomenon. PDVs produce the strongest predictive performance on clean data, but are the least noise-resistant. Embeddings, particularly mol2vec, are the most robust to noise yet perform the worst noise tracking. At the end of the day, the choice of both model and representation comes down to your problem, including the amount of noise in your data, your objectives, and your compute limits. 

---

## MOVEMENT 4 — how uncertainty and noise play together

Three questions, deliberately separate, because earlier drafts of this paper fused them. Does a model become
less sure when its training labels are corrupted? Does its uncertainty still rank which predictions to trust?
And can it point at which labels were the corrupted ones? The first was already known, the second is what a
referee expects, and the third is the hard one.

### §R6. Uncertainty under label noise *(replaces `paper.tex:613–658`; rewritten 2026-09-25)*

**How this block is built (updated later on 2026-09-25).** You said your population-level and per-sample
text will be replaced, so the block below is new text around the decomposition paragraphs. Your three
paragraphs from `paper.tex:615`, `:649` and `:658` are kept word for word under "Your text this
replaces", after the TODO list, so nothing is lost. The order is: the question, the population-level
rise (new table), the decomposition (unchanged from this morning), then per-sample tracking.
The decomposition paragraphs connect to four earlier parts of the paper:
- the Methods expectation that the aleatoric component rises while the epistemic one holds (`paper.tex:281`);
- the variance decomposition, where representation carries 9.2\% of the variance in \aucnorm;
- the noise-conditions subsection, where shape did not matter and grouped-shifted did;
- the counterparts subsection, where the variance head cost the neural networks almost no \aucnorm.

**Model names.** α is the DNN architecture and β the MLP, as in the figures: BNN-α (var. head) is
`dnn_bnn_full_mve` and VBLL-β (het.) is `mlp_vbll_hetero`.

```latex
\subsection{Uncertainty under label noise}

Until now we have asked how much accuracy a model keeps when its training labels are corrupted. We now
ask what the model reports about its own confidence. \citet{Kolmar2021} found that the mean uncertainty
predicted by GPs rises with the amount of label noise in the training data. We test that for every
probabilistic model in the study, and then ask two further questions. The first is why the uncertainty
rises: whether the model attributes the added noise to the labels or to itself. The second is whether a
model's uncertainty on one molecule tracks the noise on that molecule's label.

On QM9, mean predicted uncertainty rises with label noise for every model, representation and noise
condition, in every fold (Table~\ref{tab:uncertainty_rise}). Here "rises" means the uncertainty at the
highest noise level is above its value with no noise. On the three assay data sets it rises in every fold
for NGBoost, both GPs, QRF and VBLL-$\alpha$ (het.). Every exception is a neural network. The full-BNNs
without a variance head fail to rise in at least one fold on several configurations. BNN-$\alpha$ fails
on all three assay data sets and all six representations. BNN-$\beta$ fails on hERG $K_i$ and Caco-2 with
Avalon, ChemBERTa, MHG-GNN and Sort \& Slice. With a variance head, both full-BNNs fail on MHG-GNN for
hERG $K_i$ and Caco-2. BNN-$\beta$ with a variance head also fails on
ChemBERTa for hERG $K_i$ and Caco-2. VBLL-$\beta$ (het.) fails on logD with Avalon, ChemBERTa and
MHG-GNN, and on Caco-2 with Avalon. So Kolmar's finding carries from the GP to every probabilistic model
on the computed property. On measured labels it breaks for some networks, most often on learned
embeddings.
% Single-fold exceptions not named in the text: BNN-beta (var. head) on hERG, PDV, outlier; VBLL-alpha
% on hERG, PDV, Student-t; VBLL-beta on hERG, PDV, grouped-shifted. All on held-out molecules.
% PLACEHOLDER -- needs the next harvest. The table is built by the figure script as
% tables/T9_uncertainty_rise_pdv.tex (main text) and T9_uncertainty_rise_<rep>.tex for the other five
% (additional file). Paste the fragment here once the analysis has been re-run.
\begin{table}[htbp]
\centering
\caption{Mean predicted uncertainty with no added noise and at the highest noise level, PDV, Gaussian
noise, held-out molecules. [TODO: final caption once the table is in.]}
\label{tab:uncertainty_rise}
% [T9_uncertainty_rise_pdv.tex]
\end{table}
% TODO (next harvest): re-check every exception named in this paragraph against
% total_uncertainty_rise.csv. The names above are from the 16 September harvest.

% ---- NEW: uncertainty decomposition ----
A rise in total uncertainty tells us that a model is less sure, but not why. We expected the aleatoric
component to rise with injected label noise while the epistemic component stayed relatively
consistent. That expectation has a practical side. A rising aleatoric component tells a user that the
labels are noisy and repeat measurements are needed. A rising epistemic component instead points to
missing training data near that molecule. The expectation can only be tested where both components vary
per sample: the full-BNNs with a variance head, the heteroscedastic GP and QRF
(Table~\ref{tab:uncertainty}).
% TODO: the two heteroscedastic VBLLs also vary per sample in both components, but they were not
% scored out of fold, so they are not in the comparison.

The full-BNNs with a variance head behave as expected (Figure~\ref{fig:decomposition}). As the noise
level rises, almost all of the added uncertainty goes to the aleatoric component. The epistemic component
grows by at most 0.43 of the aleatoric growth, on all four data sets, all six representations and every
noise condition. The variance head cost these networks almost no \aucnorm; what it buys is uncertainty
that attributes label noise to the labels. QRF does not behave as expected. Both of its components rise
with the noise, and on several representations the epistemic component rises faster than the aleatoric.

This matters most where the epistemic component chooses the next experiment. Active learning selects the
molecules with the highest epistemic uncertainty for measurement. With QRF, noisy labels inflate that
component, so the loop would keep selecting molecules whose labels are unreliable. We did not run such a
loop, so this follows from the measurements rather than being one.
% TODO: cite an active-learning study that selects on epistemic uncertainty.

Molecular representation carries under a tenth of the variance in \aucnorm, but it decides whether some
decompositions work. For QRF, the epistemic growth is 0.63 to 1.09 times the aleatoric growth on the
three fingerprints (ECFP4, Avalon and SNS). It is 1.01 to 1.45 times on PDV, and 1.28 to 2.05 times on
ChemBERTa and MHG-GNN. The heteroscedastic GP keeps its epistemic growth below about a quarter of its
aleatoric growth on PDV, ChemBERTa and MHG-GNN under every noise condition. On ECFP4 under grouped-shifted
noise, both of its components grow by about the same amount. As with the neural networks' robustness, the
representation matters through its pairing with the model. A decomposition checked on one representation
cannot be assumed to hold on another.
% TODO: Avalon, SNS and MHG-GNN come from held-out molecules on the three assay data sets only, under
% Gaussian, grouped-wider and grouped-shifted. PDV, ECFP4 and ChemBERTa have out-of-fold molecules under
% all six conditions. The heteroscedastic GP ran on QM9 at ECFP4 alone.

The noise conditions repeat the pattern seen for robustness. The shape of the error made no measurable
difference to \aucnorm, and the three heavy-tailed conditions leave the full-BNNs' and the
heteroscedastic GP's decomposition where Gaussian noise left it. For QRF they lower the epistemic share
slightly, by a median of 0.11 to 0.14. Grouped-shifted, the condition that cost the most \aucnorm,
is also the one that moves added noise into the epistemic component. It raises the epistemic share of
the heteroscedastic GP and QRF in every pairing of data set and representation, most for the GP on
fingerprints. One reading is that a scaffold family sharing one offset looks to these models like
structure in the data rather than noise on its labels.

The aleatoric component shows that label noise increased, but not by how much
(Figure~\ref{fig:aleatoric_injected}). For each unit of added noise it rises by 0.36 to 0.99 units,
depending on model, data set and representation. The full-BNNs with a variance head and NGBoost already
report 0.31 to 1.26 label units on clean labels. That starting point, not their rate of rise, brings
them close to the injected amount at high noise. The heteroscedastic GP starts near zero and ends within
a tenth of the injected amount on PDV, but not on ECFP4. The aleatoric component can compare noise
between data sets or noise levels, but it does not estimate an assay's measurement error.

Censoring ran on five pairings in the robustness experiments, but on every uncertainty model on the
three assay data sets. Clipping labels narrows their spread: with half the labels clipped, the spread
falls to between 0.27 and 0.61 of its clean value, depending on the data set. NGBoost and QRF shrink
their total uncertainty in step with that spread on all four data sets. The GPs do so on hERG $K_i$ and
Caco-2, and the full-BNNs with a variance head shrink far less. On hERG $K_i$ and Caco-2, the
heteroscedastic GP's aleatoric component holds steady up to a quarter of labels clipped, while its
epistemic component falls by about half. None of these models treats a clipped label as a noisy one, so
they grow more confident as their labels lose information.
% TODO: on QM9, censoring ran at PDV alone on three models (the variance-head full-BNN, the RBF GP and
% NGBoost). The spread is computed on whole data sets, not per training fold.
% ---- end NEW ----

% ---- NEW: per-sample tracking ----
Finally, we ask whether a model's uncertainty on one molecule tracks the noise on that molecule's label.
We measure this as the Spearman correlation between predicted uncertainty and the size of the injected
noise, per fold and per noise level. Each training molecule is scored by a model fitted without it. The
question is therefore whether a model can flag noisy labels it never saw. Under six of the seven noise
conditions, nothing about a molecule says how much noise it received. Gaussian, Laplace, Student-$t$ and
grouped-shifted noise give every molecule the same noise scale. Outlier noise widens a random tenth of the
labels. Grouped-wider noise widens whole scaffold groups, and each scaffold group falls in a single fold.
Under these six conditions the correlation stays between $-0.16$ and $0.16$ for every model,
representation and data set, in every fold and at every noise level. Folds that fall outside the range
expected by chance do so on both sides, not in one direction. The one lean is on QM9 under
grouped-wider noise, which is slightly negative, down to $-0.11$, for every model and representation
except NGBoost on ChemBERTa.

Censoring is the one condition where the noise follows the label, because only the highest labels are
clipped. It is also the only condition with large correlations, and they are mostly negative. A negative
correlation means the model is least uncertain on the labels that were clipped the most. On QM9 with half
the labels clipped, on PDV, the correlation across folds is $-0.84$ to $-0.80$ for NGBoost. It is $-0.84$
to $-0.40$ for BNN-$\alpha$ with a variance head, and $-0.47$ to $-0.41$ for the GP. On logD with half the
labels clipped, every fold is at $-0.29$ or lower for every model except the two GPs, on every
representation. For most models the correlation becomes more negative as more labels are clipped. One
likely reason is that the clipped labels share one value, so the models see little spread around them.
That fits the fall in total uncertainty with label spread described above, but it has not been tested
molecule by molecule.
% PLACEHOLDER -- needs the next harvest (unc_censoring_control.csv). The clipped molecules are also the
% ones with the highest true labels. Keep ONE of these two sentences once the result is in:
% (a) "The same fits with nothing clipped already show this correlation, so it comes from the high
%     labels, not from the clipping."
% (b) "The same fits with nothing clipped show no such correlation, so the clipping causes it."
% If it is (a), the "one likely reason" sentence above and the "negative result" paragraph below
% both need rewriting. Check `rule_check` first: if it is not near 1, the control is not valid.

Three exceptions qualify the censoring result. On Caco-2 with a tenth of the labels clipped, the correlation is positive
or near zero for every model, up to $0.41$ for VBLL-$\alpha$ on PDV. For NGBoost, QRF and the
heteroscedastic GP it is negative in every fold once a third of the labels are clipped. The GP stays
between $-0.07$ and $0.16$ on all three assay data sets, at every level and in every fold. On QM9 it
does not. On hERG $K_i$, the full-BNNs with a variance head reach $-0.52$ on ECFP4 but stay between
$-0.10$ and $0.14$ on PDV and ChemBERTa. Representation therefore decides, for these networks, whether
their uncertainty is inverted under censoring.

This is a negative result, and a worse one than no tracking. Under censoring, a user who trusts low
uncertainty would trust the corrupted labels first. None of the seven models scored out of fold flags
noisy labels on molecules it did not train on. Where the noise is tied to the label value, most of them
point the wrong way.
% The seven: NGBoost, GP, heteroscedastic GP, QRF, BNN-alpha and BNN-beta with a variance head, VBLL-alpha,
% on PDV, ECFP4 and ChemBERTa.
% TODO: uncertainty on the molecules a model was trained on was never recorded, so tracking in that
% setting is untested. See the TODO list.
% ---- end NEW ----
```

**Where every new number comes from.** All are in `results/decisions_arc_20260916/`:
- the epistemic growth divided by the aleatoric growth, from `decomposition_ratio.csv`;
- the censoring spread and the matching uncertainty ratios, from `censoring_spread.csv`;
- the slopes and clean-label starting values, from `unc_q5.csv` as drawn in F6b;
- which configurations rise in every fold, from `total_uncertainty_rise.csv`. One row is one data set,
  model, representation, noise condition and molecule set; the ratio is the mean uncertainty at the top
  level divided by its value with no noise, lowest and highest fold;
- every per-sample correlation, from `per_sample_by_level.csv`. One row is one data set, model,
  representation, noise condition and level; it gives the median, lowest and highest fold, and how many
  folds fall above and below chance. It is built from `d7_q4.csv`, column `rho_plain_NOT_THE_ANSWER`,
  which is out-of-fold training molecules only.

`scripts/uncertainty_followups.py` writes all four of those files. "Cost almost no \aucnorm" refers to your
own sentence at `paper.tex:603` and adds no new number.

**TODOs, not paper text:**
- **F6 has to change.** The F6 on disk shows QM9 at ECFP4 under Gaussian noise, in three panels. The
  representation paragraph needs a figure that shows the representation effect. One option: rows are the
  variance-head full-BNN, the heteroscedastic GP and QRF; columns are ECFP4, PDV and ChemBERTa; one assay
  data set. Not built.
- **The heteroscedastic VBLLs.** They have no out-of-fold rows.
- **More representation evidence.** See the TODO comment in the representation paragraph.
- **The active-learning paragraph** has no citation yet.
- **Contradiction in Models.** `paper.tex:228` still says the BNN variants and QRF are not decomposed,
  which contradicts `paper.tex:283–299` and these paragraphs.
- 🔴 **Everything below waits on one re-run of the analysis.** Every number in the new paragraphs
  is from the 16 September harvest. The figure script now writes the table and the clipping test, but
  neither exists until it is re-run:
  ```bash
  cd $QSAR && git pull && sbatch slurm_scripts_analysis/run_paper_analysis.sh
  ```
  Then re-check each number and exception in §R6 against the new `total_uncertainty_rise.csv`,
  `per_sample_by_level.csv` and `unc_censoring_control.csv`.
- **Table `tab:uncertainty_rise` (T9).** Built by the figure script, one table per representation:
  `tables/T9_uncertainty_rise_<rep>.csv` and `.tex`. One row is one model. For each data set it gives
  the mean predicted SD with no noise and at level 1.5 (median over folds, eV or log units), and the
  top value divided by the no-noise value in its lowest and highest fold. † marks a model that fails
  to rise in some fold. Held-out molecules, Gaussian noise. Built locally from the 16 September
  numbers, it has 12 or 13 rows per representation. The alternative layout, one table per data set
  with the noise levels as columns, is not built.
- **A value to look at in T9.** BNN-β (var. head) on MHG-GNN, hERG $K_i$: a mean predicted SD of
  17.9 log units with no noise and 18.3 at level 1.5. BNN-α (var. head) on the same pairing: 6.6 with
  no noise. The other models on hERG $K_i$ at MHG-GNN sit between 0.26 and 2.28 with no noise. Not investigated.
- **QM9 is missing rows in T9.** On QM9 at PDV, the heteroscedastic GP, both plain full-BNNs, VBLL-β
  and both heteroscedastic VBLLs have no Gaussian held-out rows. This is the QM9 no-condition
  question below.
- **QM9 held-out rows with no noise condition.** Many QM9 held-out rows in `unc_q5.csv` carry no
  condition, so only six or seven models appear under Gaussian there. What those rows are is not
  established.
- **Clipping or high labels.** Under censoring, the clipped molecules are the ones with the highest true
  labels. The test: take each censoring configuration's no-clipping fits (level 0, which exist). Mark the
  molecules that would be clipped at each fraction, by the same rule, with their would-be clipped
  amount. Compute the same correlation. If it is already negative with nothing clipped, the high labels
  cause it; if it is near zero, the clipping does. No new fits. **Now in the figure script**
  (`censoring_control` in `scripts/figlib_uncertainty.py`), written as `unc_censoring_control.csv` on
  the next run. One row is one data set, model, representation, fold and clipped fraction:
  `rho_clipped` is the reported correlation, `rho_unclipped` the control, and `rule_check` whether
  the clipping rule reproduces the injector (it should be near 1).
- **Chance range.** The per-sample paragraph's "expected by chance" is 1.96/√(n−1) per fold: ±0.016 on
  QM9 up to ±0.065 on hERG $K_i$. It is my approximation, and it assumes molecules are independent, which
  grouped noise breaks. The permutation band in `d7_q4.csv` is for the error, not the uncertainty.
- **Coverage of the per-sample result.** No out-of-fold rows exist for the full-BNNs without a variance
  head, VBLL-β, either heteroscedastic VBLL, or for Avalon, Sort & Slice and MHG-GNN. On QM9 the
  heteroscedastic GP has one fold, on ECFP4 under Gaussian. QM9 censoring ran at PDV on three models.
- **Molecules the model trained on.** Uncertainty on those was never recorded, so whether a model's
  uncertainty tracks the noise on labels it fitted is untested.
- **§R7 now conflicts with this.** §R7 reads the censoring gain from dividing error by uncertainty as a
  model recognising a clipped label. The per-sample paragraph shows the division works because the
  uncertainty is *lowest* on clipped labels. §R7 also says censoring ran on five pairings; for the
  uncertainty runs it ran on every out-of-fold model on the three assay data sets.

**Your text this replaces**, kept word for word from `paper.tex:615`, `:649` and `:658`. The table
`tab:top_unc_noise` (`paper.tex:619–647`) and Figure `fig:uncertainty_combined` (`paper.tex:651–656`)
go with it. Your mechanism sentences in the second paragraph (how NGBoost, the GP, BNNs and VBLL handle
large residuals) are not results and can come back if you want them. "A single global observation noise
term" holds for the RBF GP, not the heteroscedastic GP.

> Until now we've discussed how much accuracy a model is able to retain whilst its training labels are corrupted with noise. Now, we consider how confident the model reports its own predictions are in the presence of noise, and whether or not those estimates match the increased label noise. \citet{Kolmar2021} found that the mean predicted uncertainty derived from the GPs increases with the amount of label noise in the training data. We go beyond the population level and instead ask if a model's per-sample uncertainty tracks label noise. One key sign that a model is able to handle noise is its ability to track it, represented by the per-sample Spearman correlation between predicted uncertainty and noise magnitude (Table~\ref{tab:top_unc_noise}).
>
> Samples containing label noise tend to produce large residuals in training, and every model has its own way of handling them. NGBoost contains predicted scales that increase to absorb residual \citep{Duan2020}. The GP contains has a single global observation noise term. It does not absorb individual residuals, so its per-sample uncertainty derives primarily from the posterior variance rather than label noise  \citep{Rasmussen2005, Obrezanova2007}. BNNs treat these large residuals differently; they broaden their weight posteriors, increasing predicted uncertainty without explicitly modeling observation noise \citep{gal2016, kendall2017}. VBLL adds a learned noise variance to the BNN loss \citep{Harrison2024}, but this value is a global scalar, independent of the input. As seen in Table~\ref{tab:top_unc_noise} and Additional file~9, both BNNs and VBLLs achieve moderate correlations which improve when paired with fingerprint or descriptor-based representations. Among models that learn separate distributional parameters during training, GP and NGBoost produced the strongest correlations between uncertainty and label noise, though QRF, which separates parameters using quantile distributions, produced both poor label predictions and uncertainty-noise correlations. At the population level, mean predicted uncertainty increases with artificial noise. Although we expect to see aleatoric uncertainty increase with injected noise, for VBLL both the aleatoric and epistemic components increased (Figure~\ref{fig:uncertainty_combined}b).
>
> The choice of representation has a much stronger impact on per-sample uncertainty tracking than on noise robustness. Fingerprints and physicochemical descriptors provided a clear signal with which models such as GP and NGBoost could separate noise-induced residuals from structural variation. However, learned embeddings such as MHG-GNN and mol2vec did not produce viable uncertainty-noise correlations (Table~\ref{tab:top_unc_noise}). Although these learned embeddings can be effective representations for many tasks, this research suggests that they provide challenges for per-sample noise tracking. However, there is room for further investigation, as only two embeddings were tested.

---


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

*Rewritten 2026-09-23 against the PNG and `f1_noise_conditions` in `scripts/figlib_figures.py`. Goes
after the paragraph in Artificial noise injection that ends "…reassigned to that limit." Cite it at the
end of that paragraph's first sentence: "…using seven noise conditions (Figure~\ref{fig:noise_conditions})."*

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F1_noise_conditions.png}
\caption{The seven noise conditions applied to one set of labels. The labels are 2,000 synthetic
values rather than a study dataset, split at random into 40 groups that stand in for scaffold
families. a)--g) Clean labels in grey and noised labels in colour, at a noise level of 0.5 of the
clean label spread. Censoring is drawn with 25\% of labels clipped, since its level is the fraction
clipped. The bottom axis is the label value and the side axis is the share of labels in each bin.
h) For each condition except censoring, the standard deviation of the mean error within a group,
divided by the standard deviation of all errors. A value near 1 means a group's labels move
together. With about 50 labels per group, noise that ignores the groups gives about 0.14.}
\label{fig:noise_conditions}
\end{figure}
```

*Three corrections to the earlier caption. The figure is drawn on synthetic labels and random groups,
which it did not say. The bottom axis does carry tick numbers. And the Gaussian bar at 0.14 is what
chance gives for groups of about 50, which a reader would otherwise read as group structure.*

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

**TODO (2026-09-25):** the decomposition paragraphs in §R6 now rest on the representation effect, which this
single-representation figure cannot show. See the TODO list under §R6.

---

### F6b — the aleatoric component against the noise added *(new 2026-09-25)*

File: `results/decisions_arc_20260916/figures/F6b_aleatoric_against_injected.png`, drawn by
`f6b_aleatoric_against_injected` in `scripts/figlib_figures.py`.

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F6b_aleatoric_against_injected.png}
\caption{The aleatoric component against the amount of noise added to the training labels, under
Gaussian noise. The top row is PDV and the bottom row ECFP4. The columns are the QM9 HOMO--LUMO gap,
logD, hERG $K_i$ and Caco-2. The bottom axis is the standard deviation of the added noise, in eV for QM9
and log units for the assay datasets. The side axis is the mean predicted aleatoric standard deviation
over the molecules of one out-of-fold pass, in the same units. Each line is one model, drawn as the median
over folds, and the band spans the folds. The dashed line marks where the reported noise equals the added
noise. The heteroscedastic Gaussian process was not run on QM9 at PDV. The two heteroscedastic variational
networks were not scored out of fold and are not drawn.}
\label{fig:aleatoric_injected}
\end{figure}
```

---

### F8 — the three assay datasets

**Settled 2026-09-18: one figure with three panels.** The code writes three image files because three
grids of the full roster stacked is about 18 inches and the journal allows 225 mm, but they go into one
`figure*` with panel letters so that the shared colour scale — which exists precisely so the three datasets
can be compared — stays on one page. It is tall, so it will want `[p]` and a page of its own.

**The variational-network row stays in, with a sentence in the caption.** On the Caco-2 panel the brightest
row is the first variational network, at AUC$_{norm}$ 0.970 under Gaussian noise, the highest on the
panel, on a clean R$^2$ of 0.358, the second lowest on it. AUC$_{norm}$ is a share of
clean accuracy, so the small denominator is what lifts it — the Gaussian process on the same panel sits at
0.809 on a clean R$^2$ of 0.517. The caption below carries that sentence. Dropping the row would mean
listing the whole roster and drawing seventeen of it.

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
not comparable by colour. The brightest row on panel b) has the second lowest clean R$^2$ on that
panel. AUC$_{norm}$ is a share of a model's own clean accuracy, so a small first column lifts the rest
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

🔴 **TODO — regenerate after the matched-settings re-run.** Anything keyed by model changes:
the model heatmaps, the AUC$_{norm}$-by-model and clean-R$^2$ tables, the variance-decomposition
figure, T6, and Figure~\ref{fig:variants} in §R3, which needs `rf300` drawn in place of `rf`. The
the roster gained rf300, so any caption or column count that states a number of models moves with them.


Eight have a block above. **Five do not, and pasting the draft without settling them leaves a broken
`\ref`.** Four are the figure promotions below; the fifth is a table wrapper you write from the pattern in
the tables section.

| label | what it is | state |
|---|---|---|
| `fig:noise_conditions` `fig:variance` `fig:grid` `fig:curves` `fig:decomposition` `fig:assay` `fig:variants` | F1, F2, F3, F4a, F6, F8, R17 | **block above** |
| `tab:robustness` | T4 at ECFP4 on QM9 | **wrapper above** |
| `tab:uncertainty` | T6 | wrapper is yours once T6's shape is settled — `RERUN_PLAN.md` §15, READ FIRST item 3 |
| `tab:variants` | T5 | wrapper from the pattern; it is a additional file, so the reference becomes "Additional file~8" unless you promote it |
| `tab:pairs` | T8 | **PROMOTED to the paper, 2026-09-18, author's call.** §R2's third paragraph points at it |
| `fig:conditions` | F4b | stays a additional file, 2026-09-18, author's call |
| `fig:ranks` | R15 | **promotion decision below** |
| `fig:transfer` | R9 | **PROMOTED to the paper, 2026-09-18, author's call.** §R5's second paragraph points at it |

## Four figures the Results text leans on that §14.25 puts in the additional files

**Settled 2026-09-18.** The rank-transfer figure goes in the paper and the other three stay additional
files. The twelve-standout-pairings table goes in the paper too, which is the table entry above rather than
a figure. Each row below still names the sentence that changes, because the three that stayed out are the
ones whose pointers now have to be rewritten.

| figure | what it carries | if it stays a additional file |
|---|---|---|
| **F4b** `F4b_rf_across_noise_conditions.png` | one model across all seven conditions on one pair of axes — the picture of the paper's differentiator | §R4's second paragraph loses its orienting sentence and points at Table~\ref{tab:robustness} instead. **This is the one I would promote**: the subsection that carries the paper's novel claim currently has no figure, while the subsection that repeats the ANOVA has two |
| **R9** `R9_rank_transfer_ecfp4.png` | rank on QM9 beside rank on each assay dataset, one row per model | ✅ **IN THE PAPER.** §R5's second paragraph keeps its orienting sentence and `fig:transfer` resolves inside the manuscript. This also settles the disagreement in the repository: decision D9's own verdict said *"T7 is promoted to a figure"* while §14.25 said additional file, and the paper now follows D9 |
| **R15** `R15_rank_against_level_*.png` | where each model ranks as the noise rises | §R4's fourth paragraph drops to one sentence about Kendall's *W*. R15 is two charts to say the ranking barely moves, so this is the weakest of the four |
| **R6** `R6_representation_profile_*.png` | the cells where a model's robustness at one representation sits outside the range of its others | §R1's third paragraph keeps Figure~\ref{fig:grid} and loses nothing. Decision D5 fired on eight cells, and **seven of the eight are variant models**, which the cross-model figures exclude — so D5 fires on cells the paper does not draw |
---

---

## WHERE EVERYTHING NOW SITS IN `paper.tex` — 2026-09-23

**`paper.tex` was edited directly on the author's instruction, 2026-09-23.** The standing rule is
that it is never edited; this was a one-off to carry out the moves agreed in chat, and it was
copy-and-paste plus the new figure and table blocks. No prose was rewritten. The file as it stood
before is `paper.tex.before_restructure_2026-09-23`.

Nothing was deleted. The two blocks that came out are commented in place with a line saying why.

### The Results subsections, and what is in each

| Line | Subsection | Figures | Tables |
|---|---|---|---|
| 387 | Variance decomposition | F2 `fig:variance` | T3 `tab:variance` |
| 426 | Robustness and clean accuracy | F4a `fig:curves`, F3 `fig:grid`, R16 `fig:decoupling` | T4 `tab:robustness` |
| 487 | Artificial noise conditions | F4b `fig:conditions`, R19 `fig:deep` | — (cites `tab:robustness`) |
| 548 | Robustness on the three assay datasets | F8 `fig:assay`, F2b `fig:variance_clean`, R9 `fig:transfer` | T3b `tab:variance_clean` |

### What moved, and where it went

| What | From | To |
|---|---|---|
| F3, the model-by-representation grid | Variance decomposition | Robustness and clean accuracy, after `fig:curves` |
| F2b, the clean-label decomposition | Variance decomposition | The assay subsection, after `fig:assay` |
| The clean-label paragraph, "On clean labels, model architecture and…" | Variance decomposition | The assay subsection, immediately above F2b |

### What was added

| What | Where | Needs |
|---|---|---|
| T3 `tab:variance` | Variance decomposition, between the second paragraph and F2 | `T3_variance_decomposition_qm9.tex` on the graphics path |
| T4 `tab:robustness` | **Robustness and clean accuracy** as of 2026-09-25; the block is still sitting in Artificial noise conditions in `paper.tex` and has to be cut and pasted up | `T4_robustness_qm9_ecfp4.tex` |
| R19 `fig:deep` | Artificial noise conditions, after `fig:conditions` | `R19_deep_conditions_qm9.png` |
| T3b `tab:variance_clean` | The assay subsection, after F2b | `T3b_variance_decomposition_clean.tex` |

All four come out of the next figures run. Until then the `\input` lines will fail to compile, so
either run the figures first or comment the four `\input` lines while drafting.

### What was commented out, not deleted

- **F4c** `fig:robustness`, which was in "Robustness and clean accuracy". It is nineteen rows by
  four columns of printed numbers, which is a table drawn, and T4 carries the same content with all
  seven noise conditions instead of three. Uncommenting it is one line if you disagree.
- **The paragraph beginning "The choice of model architecture is instead the largest source of
  variance"**, which repeated the subsection's opening paragraph. Its four TODO comments are
  commented with it and still readable.

### What this fixed on the way

- `Table~\ref{tab:robustness}` was cited at the old line 493 and defined nowhere. T4 defines it.
- `Table~\ref{tab:variance}` was cited in the opening paragraph and defined nowhere. T3 defines it.
- `fig:deep` and `tab:variance_clean` are cited where they are introduced, so no float is orphaned.

### Still broken, and none of it mine

- `fig:validation_combined`, `fig:validation_overview`, `fig:variants` and `tab:pairs` are cited and
  defined nowhere. All four pre-date this edit.
- `fig:conditions` (F4b) and `fig:transfer` (R9) are defined and never cited in the body. Both
  pre-date this edit.


---

## THE FIGURES AND TABLES ADDED ON 2026-09-23, LaTeX blocks

Settled with the author on 2026-09-23. **F4c leaves the paper, T4 replaces it, F2b and T3b move to
the assay subsection, and the decomposition subsection keeps one figure and one table.** Line
numbers are against `paper.tex` as of 2026-09-23.

### T3 — the decomposition, with clean labels as its first row *(main text, Variance decomposition)*

Currently generated and cited nowhere. The clean-label row is new: it rides at the top of the same
table so the comparison the subsection turns on can be made without looking at a second table.
Goes after the first paragraph, before `fig:variance`.

```latex
\begin{table}[htbp]
\centering
\caption{Share of the variance in each outcome explained by model architecture, molecular
representation, their pairing and the residual, on the QM9 HOMO--LUMO gap. The first row is
predictive accuracy before any noise is added, which has no noise condition: the clean fit is made
once per replicate and every noise condition starts from it. The $\pm$ is half the
leave-one-replicate-out range, the decomposition repeated with each replicate dropped in turn. It
is not a per-replicate spread, as decomposing a single replicate leaves one observation per cell
and the residual is then arithmetically zero.}
\label{tab:variance}
\input{T3_variance_decomposition_qm9}
\end{table}
```

### T4 — every model under every noise condition *(main text, the condition comparison)*

**This replaces F4c**, which is `fig:robustness` at line 484: a heatmap of nineteen rows by four
columns of printed numbers, which is a table drawn. T4 carries all seven conditions instead of
three, and inserting it resolves `Table~\ref{tab:robustness}`, cited at line 497 and defined
nowhere. Delete lines 478–485 and put this where the condition comparison discusses them.

```latex
\begin{table}[htbp]
\centering
\caption{Robustness (\aucnorm) of each model under each noise condition on the QM9 HOMO--LUMO gap
at ECFP4, with clean $R^2$ in the first column as the quantity the rest are a fraction of. A dash
is a pairing that condition was not run on: Laplace, Student-$t$ and outlier noise were given to a
named subset of model architectures, and censoring to a named subset of pairings. Values are
medians over replicates.}
\label{tab:robustness}
\input{T4_robustness_qm9_ecfp4}
\end{table}
```

### R19 — the conditions run on a named subset *(promoted to main text 2026-09-23)*

This is where censoring appears at all. Goes in the condition comparison, after line 511, the
`\end{figure}` that closes `fig:conditions`.

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R19_deep_conditions_qm9.png}
\caption{Robustness (\aucnorm) under the noise conditions given to a named subset of model
architectures and molecular representations, on the QM9 HOMO--LUMO gap. a) ECFP4, b) PDV,
c) ChemBERTa. Rows are the models these conditions were run on, chosen before the run to cover one
from each family, and Gaussian is the left column as the reference the rest are read against.
Values are medians over replicates and colour is on one fixed range across all panels. Censoring is
here rather than on the curves because its level is a fraction of labels clipped rather than a
fraction of the label spread; a grey square marked ``not run'' is a pairing it was never given.}
\label{fig:deep}
\end{figure}
```

### F2b and T3b — moved to "Does it hold on measured labels"

Cut `fig:variance_clean` from lines 407–418 and put it after `fig:assay` at line 545, with the
table beside it. Comparing the four datasets belongs where the assay datasets are introduced, not
before the reader has met them.

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F2b_clean_decomposition.png}
\caption{Share of the variance in predictive accuracy on clean labels explained by model
architecture, molecular representation, their pairing, and the residual, on all four datasets.
Accuracy is $R^2$ on held-out molecules with no noise added to the training labels. The bottom axis
is the dataset, the side axis the share of variance; the four shares within a dataset sum to 100\%.
There is no noise-condition axis: the clean fit is made once per replicate and every noise
condition starts from it. Whiskers on the HOMO--LUMO gap are a leave-one-replicate-out jackknife,
not confidence intervals; the assay datasets have none, as their five scaffold folds partition one
dataset rather than repeating an experiment.}
\label{fig:variance_clean}
\end{figure}

\begin{table}[htbp]
\centering
\caption{Share of the variance in predictive accuracy on clean labels explained by model
architecture, molecular representation, their pairing and the residual, one row per dataset. The
HOMO--LUMO gap row is the same decomposition as the first row of Table~\ref{tab:variance}. The
assay datasets carry no band, as their five scaffold folds partition one dataset rather than
repeating an experiment.}
\label{tab:variance_clean}
\input{T3b_variance_decomposition_clean}
\end{table}
```

### F3b and F4d — additional files, or nothing

**F3b** `F3b_every_condition_qm9_<condition>.png`, one image per condition: the robustness grid for
every noise condition the whole roster ran, grouped-wider included. `fig:grid`'s caption already
promises it. Three `\includegraphics` in one figure environment, with a
`% Additional file N — Title` comment above the section so `scripts/stage_additional_files.py`
lists it.

**F4d** `F4d_conditions_by_representation_<model>.png`: one model's curves under every noise
condition, one panel per molecular representation. A check figure first. If outlier noise being the
least damaging at high level holds beyond RF at ECFP4, it earns a place beside
Figure~\ref{fig:conditions}; if it is an RF quirk, it is an additional file or nothing.

### What the Results then hold

| Subsection | Figures | Tables |
|---|---|---|
| Variance decomposition | F2 `fig:variance` | T3 `tab:variance` |
| Robustness and clean accuracy | F3 `fig:grid` (moved in with line 421), F4a `fig:curves` | — |
| The condition comparison | F4b `fig:conditions`, R19 `fig:deep` | T4 `tab:robustness` |
| Does it hold on measured labels | F8 `fig:assay`, F2b `fig:variance_clean`, R9 `fig:transfer` | T3b `tab:variance_clean` |

F4c is gone. The paper goes from ten figures and one table to nine and four. For scale: Venkatraman
2021 runs 12 pages with 3 figures and 6 tables; Kolmar and Grulke 2021, the closest comparator,
19 pages with 6 figures and 7 tables; Dablander 2023, 16 pages with 9 figures and 1 table.

# THE TABLES

**Seven in the paper as of 2026-09-23.** The five below, plus the two variance-decomposition
tables added in the section above: the metrics table, the noise-conditions table, robustness at
ECFP4, the uncertainty table, the twelve standout pairings, T3 and T3b. Three of them — T3, T3b
and the robustness table — are generated already and cited nowhere, which is the whole of the
"where are the tables" problem.

The five that were here before, the fifth new on 2026-09-18: the metrics table, the
noise-conditions table, robustness at ECFP4, the uncertainty table, and the twelve standout
pairings, which the author promoted because §R2's third paragraph names it and nothing else in the
paper carries that list.

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
`T6b_uncertainty_every_dataset.tex` for a additional file.

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
| **T4** | `T4_robustness_qm9_ecfp4.tex` | **paper** | 13 base models × 7 conditions, clean R² first. The three assay versions go to the additional files |
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
   representation, noise condition and measured dataset. Six tables, one row per model,
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

## 🔴 One entry the Molecular representations paragraph needs, 2026-09-20

`scripts/check_bib_and_docs.py` validates against **`citations.bib`**, not `refs.bib`, and it fails today on
`rogers2010`. Checked key by key: `avalon` and `sns` are both in `citations.bib` already, and `rogers2010` is
in `refs.bib:869` only. So one entry has to be pasted into `citations.bib`.

```bibtex
@article{rogers2010,
  author  = {Rogers, David and Hahn, Mathew},
  title   = {Extended-Connectivity Fingerprints},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {50},
  number  = {5},
  pages   = {742--754},
  year    = {2010},
  doi     = {10.1021/ci100050t},
  note    = {PMID: 20426451},
}
```

Copied field for field from `refs.bib:869`. Until it is pasted, `check_bib_and_docs.py` fails with
`undefined: rogers2010`, and that failure is real rather than the pending `sn-bibliography` one.
`.bib` files. Verify it before submitting.


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
