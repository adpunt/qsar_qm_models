# How published papers define the noise-level parameter in label-noise regression studies

Compiled 2026-08-21. Every claim below is traceable to a source in "Sources actually read".
Nothing here is my own invented convention.

---

## A. Distinct conventions found in the literature

### (i) Absolute noise SD in label units
**Heid, McGill, Vermeire, Green, "Characterizing Uncertainty in Machine Learning for Chemistry",
J. Chem. Inf. Model. 2023, 63(13), 4012-4029. DOI 10.1021/acs.jcim.3c00373.
Open access via PMC10336963.**

Verbatim: *"Adding Gaussian noise with standard deviation, i.e. magnitude, of 1 kcal/mol to the
training data"*. The parameter is the SD itself, in the physical units of the label (kcal/mol).
Heteroscedastic case, verbatim: *"we apply Gaussian noise of standard deviation 20 kcal/mol for
nitrogen-containing molecules and Gaussian noise of standard deviation 2 kcal/mol for
non-nitrogen-containing molecules."*

Also **Li & Fourches, "Inductive transfer learning for molecular activity
prediction: Next-Gen QSAR Models with MolPMoFiT", J. Cheminform. 2020, 12:27,
DOI 10.1186/s13321-020-00430-x (PMC7178569)**: *"a Gaussian noise (mean set at 0 and standard
deviation sigma_noise) is added to the labels of augmented SMILES which could be considered as a
simulation of experimental errors."* sigma_noise is treated as a tunable hyperparameter; values
tested were {0, 0.1, 0.3, 0.5} (Lipophilicity) and {0, 0.3, 0.5, 1} (FreeSolv). The paper does not
state whether these are absolute or relative; read in context they are label units.

### (ii) Noise SD as a fraction/multiple of the label's own scale (RMS or SD)
**La Cava, Orzechowski, Burlacu, de Franca, Virgolin, Jin, Kommenda, Moore, "Contemporary Symbolic
Regression Methods and their Relative Performance" (SRBench), NeurIPS 2021 Datasets & Benchmarks
track, arXiv:2107.14351.**
The paper text names the parameter "target noise" with levels {0.0, 0.001, 0.01, 0.1} (Table 2 /
Fig. 3: *"Color/shape indicates level of noise added to the target variable"*), but does not define
it in prose. The authors' own code does:

```python
# srbench/experiment/evaluate_model.py, lines 122-126
if target_noise > 0:
    print('adding',target_noise,'noise to target')
    y_train_scaled += np.random.normal(0,
                target_noise*np.sqrt(np.mean(np.square(y_train_scaled))),
                size=len(y_train_scaled))
```
i.e. sigma_noise = target_noise x RMS(y_train). For a centred target, RMS(y) = SD(y), so the
parameter is a *multiple of the label SD*. The CLI flag is `-target_noise`; there is no Greek symbol.

### (iii) Noise scaled to the RANGE of the endpoint
**Kolmar & Grulke, "The effect of noise on the predictive limit of QSAR models", J. Cheminform.
2021, 13:92. DOI 10.1186/s13321-021-00571-7. Open access (PMC8613965).** See section B.

**Cortes-Ciriano, Bender, Malliavin, "Comparing the Influence of Simulated Experimental Errors on 12
Machine Learning Algorithms in Bioactivity Modeling Using 12 Diverse Data Sets", J. Chem. Inf.
Model. 2015, 55(7), 1413-1425. DOI 10.1021/acs.jcim.5b00101. PAYWALLED (ACS, 403).** Abstract
verbatim (via Europe PMC, PMID 26038978): *"The noise was simulated by sampling from Gaussian
distributions with increasingly larger variances, which ranged from zero to the range of pIC50
values comprised in a given data set."* So the top of their noise grid is anchored to the endpoint
range, dataset by dataset. Kolmar & Grulke describe this study as *"a full factorial study of random
experimental error on 12 different datasets, 12 algorithms, and 10 levels of simulated random
experimental error"*.

### (iv) Signal-to-noise ratio (or its inverse)
**Nakamura & Fukagata, "Robust training approach of neural networks for fluid flow state
estimations", arXiv:2112.02751 (Int. J. Heat Fluid Flow 2022).** Verbatim: *"SNR =
sigma^2_data / sigma^2_noise, where sigma^2_data and sigma^2_noise are the variances of input data
and noise, respectively."* Levels tested: 1/SNR = {0.01, 0.05, 0.10}. NOTE: this is *input* noise,
not label noise, and it is fluid dynamics, not chemistry. I did not find an SNR-parameterised
*label*-noise study in QSAR.

### (v) Noise variance as a proportion of total variance / R^2 ceiling / irreducible error
This is used as a *framing of the consequence*, not usually as the injection parameter.
- Kolmar & Grulke frame the whole study as a "predictive limit" set by noise (title, and the
  asymptote analysis).
- **Kalliokoski, Kramer, Vulpetti, Gedeck, "Comparability of Mixed IC50 Data - A Statistical
  Analysis", PLoS ONE 2013, 8(4): e61007. DOI 10.1371/journal.pone.0061007 (PMC3628986).**
  Verbatim numbers: *"sigma_pIC50 = 0.68, MUE_pIC50 = 0.55 and MedUE_pIC50 = 0.43"*; *"A standard
  deviation of 0.68 corresponds to a factor of 4.8, meaning that 68.2% of all IC50 measurements
  agree within a factor of 4.8"*; and, after filtering pairs with dpIC50 >= 2.5, *"the correlation
  coefficient becomes R2 = 0.53"*.
- **Brown, Muchmore, Hajduk, "Healthy skepticism: assessing realistic model performance", Drug
  Discov. Today 2009, 14(7-8), 420-427. DOI 10.1016/j.drudis.2009.01.012. PAYWALLED (Elsevier).**
  Only the PubMed abstract was read; it is the standard citation for assay error capping achievable
  model R^2. Do not quote an equation from it without reading it.

### (vi) Matching to published experimental error magnitudes
Kolmar & Grulke do this explicitly, as a *post-hoc interpretation* of an otherwise range-scaled
grid. Verbatim: *"Kramer, Kalliokoski and colleagues found from an examination of the ChemBL
database that heterogeneous pIC50 data has an average standard deviation of 0.68 log units."* and
*"For the BACE dataset, which uses a pIC50 endpoint, 1.1 log units of noise were added, or 1.6 times
the average standard deviation reported in ChemBL."*

---

## B. The dominant convention in QSAR/cheminformatics

**Kolmar & Grulke 2021 (the direct predecessor).** Verbatim from "Random error generation":

> "Random error was added to datasets by sampling from a Gaussian distribution of zero mean and
> increasing standard deviation sigma_noise. Noise was added only to the target variables and not to
> the descriptors."

Equations as rendered in the PMC HTML:
- Eq. 7: `Y_noise[n,i] = Y + N(0, sigma_noise[n])`
- Eq. 8: `sigma_noise[n] = (Y_max - Y_min) * multiplier * n`

> "This sigma_noise was determined from the product of the range of endpoint values in the dataset,
> the noise level n, and a multiplier. This multiplier was set to 0.01 after experimentation with a
> range of values and observing the effect on RMSE."

> "Each dataset was used to generate 15 noise levels with 5 replicates at each noise level. Because
> n starts at 0, the 0th noise level has no added noise."  (n in 0..14, i in 1..5)

So: **noise level n is a dimensionless integer index; the actual SD is 1% of the endpoint range per
index step.** They used ONE noise type (Gaussian), so the question of normalising across noise types
does not arise in their paper. They did NOT normalise across noise mechanisms. They DID translate
the grid into absolute label units and into multiples of published experimental error when
discussing results (see A(vi)).

**Cortes-Ciriano et al. 2015** also anchor to the endpoint range (abstract quote above); Methods are
paywalled so I cannot report their exact grid or symbol.

**Summary for B:** the two closest QSAR precedents both scale to the *endpoint range*, per dataset.
Heid et al. (chemistry, but QM energies) use *absolute label units*. There is no single dominant
convention across the field — QSAR leans range-relative, computational chemistry leans absolute.
That inconsistency is real and should be stated as such rather than papered over.

---

## C. Named convention for comparing noise MECHANISMS at matched magnitude

**Yes, and Heid et al. 2023 is the citation.** Verbatim:

> "We also tested uniform, hyperbolic, and bimodal noise distributions, where the respective
> parameters were chosen so that each distribution had a standard deviation of 1 kcal/mol and was
> centered around 0 kcal/mol."

What they matched: **the standard deviation** (1 kcal/mol) and **the mean** (0). Not variance
explicitly, not range, not entropy. They give it no special name — no "matched-variance" or
"equal-variance" term of art appears. They describe the rationale as: *"Though noise distributions
found in real data may be non-Gaussian, if homoscedastic, they should still follow the same trends
of approaching an asymptote due to noise."*

I found no *named*, citable term ("matched-variance design", "equal-variance noise") in the papers
read. The practice exists (Heid et al.); the terminology does not. Do not coin one.

---

## D. Would sigma still be called sigma?

Actual notation observed:
| Paper | Symbol | Name used | What it means |
|---|---|---|---|
| Kolmar & Grulke 2021 | `sigma_noise[n]`, index `n` | "noise level n" | n is the dimensionless index; sigma_noise is the derived SD in label units |
| Heid et al. 2023 | standard deviation, in kcal/mol | "standard deviation, i.e. magnitude" | absolute SD |
| MolPMoFiT 2020 | `sigma_noise` | "standard deviation sigma_noise" | absolute SD, tuned as a hyperparameter |
| SRBench 2021 | `target_noise` | "target noise" / "level of noise added to the target variable" | multiplier on RMS(y) |
| Nakamura & Fukagata 2022 | `SNR` | "signal-to-noise ratio" | variance ratio |

Pattern: **sigma (or sigma_noise) is reserved for a quantity in label units.** When the parameter is
a dimensionless multiplier, the papers give it a different name (`n`, `target_noise`, SNR) and
reserve sigma for the resulting SD. No paper read here calls a multiple-of-label-SD "sigma".

---

## E. Concrete options, ranked by literature support

### Option 1 (best supported in QSAR): Endpoint-range scaling, Kolmar-style
**Definition:** sigma_noise[n] = (Y_max - Y_min) x multiplier x n. Kolmar & Grulke 2021, Eq. 8;
Cortes-Ciriano et al. 2015 anchor their upper limit to the same range.
**Transformation:** keep an 11-point integer index n; for each strategy, set the strategy's internal
scale so the realised RMS injected noise equals (Y_max - Y_min) x c x n in label units. That means
dividing the current sigma by the per-strategy factor k = {Gaussian 1.000, Threshold 2.000,
Value-prop 1.701, Hetero 0.669, Quantile 0.899, Outlier 0.502}.
**Advantages:** matches the direct predecessor exactly; a reviewer of a J. Cheminform. noise paper
will recognise it; per-dataset scaling makes the eV endpoint and the log-unit endpoints commensurate.
**Disadvantages:** range is outlier-sensitive (one bad QM9 gap stretches the whole grid);
Kolmar's 0.01 multiplier is admittedly arbitrary (*"set to 0.01 after experimentation"*); requires
re-running all 6 strategies x 11 levels x 4 datasets; and it destroys the current interpretation of
sigma=0.6 as one unit of assay error unless you re-derive it.

### Option 2: Absolute SD in label units, matched across mechanisms, Heid-style
**Definition:** every mechanism's parameters are chosen so the injected noise has the same standard
deviation, centred on zero. Heid et al. 2023: *"the respective parameters were chosen so that each
distribution had a standard deviation of 1 kcal/mol and was centered around 0 kcal/mol."*
**Transformation:** the grid becomes achieved noise SD in eV / log units; per strategy, divide the
current sigma by k above. Your Gaussian strategy is already exactly this.
**Advantages:** it is precisely the fix for the false "consistent across strategies" claim, and it
has a direct chemistry citation for matching *mechanisms* at equal SD; it keeps a physically
meaningful axis (log units), so the sigma=0.6-vs-0.54-hERG comparison survives.
**Disadvantages:** Heid matched only homoscedastic distributions. Your Heteroscedastic,
Value-proportional and Outlier strategies are not homoscedastic, and matching their *marginal* SD is
an extension of Heid's practice, not something they state; you would have to say so. Also, a single
absolute SD is not comparable across QM9 (SD 1.35 eV) and Caco-2 (SD 0.44), so you would need a
per-dataset grid or accept different effective difficulty per dataset.

### Option 3: Multiplier on the label's own RMS/SD, SRBench-style
**Definition:** sigma_noise = c x RMS(y_train), from the SRBench reference implementation
(`evaluate_model.py` L122-126); levels used there {0, 0.001, 0.01, 0.1}.
**Transformation:** replace sigma with c, and set each strategy's internal scale so realised RMS
noise = c x RMS(y) - again dividing by k.
**Advantages:** dimensionless and directly comparable across your four datasets; robust to outliers
in a way range-scaling is not; a well-known benchmark implements it.
**Disadvantages:** the definition lives in code, not in the paper's prose - you must cite the repo
alongside the paper. It is not a cheminformatics precedent. And it makes the achieved noise depend
on how heterogeneous your dataset happens to be, which severs the link to published assay error
(see the user's own recorded position that label-SD matching is the wrong normaliser - that is a
judgement call the literature does not settle).

### Option 4 (lowest change, and it IS done in the literature): keep sigma, report achieved noise SD
**Definition/precedent:** Kolmar & Grulke keep a dimensionless noise index n throughout and report
the achieved magnitude in label units and against experimental error when interpreting:
*"For the BACE dataset, which uses a pIC50 endpoint, 1.1 log units of noise were added, or 1.6 times
the average standard deviation reported in ChemBL."*
**Transformation:** none to the code. Delete the false claim that difficulty scaling is consistent
across strategies. Add a Methods table giving, per strategy, the RMS injected noise per unit sigma
(1.000 / 2.000 / 1.701 / 0.669 / 0.899 / 0.502) and the achieved noise SD at each of the 11 levels
per dataset; carry the achieved SD on figure axes or in a secondary axis label.
**Advantages:** no re-runs; nothing already computed is invalidated; it is honest and it is what the
predecessor paper does when it needs to interpret magnitude; it preserves the sigma=0.6-as-one-unit-
of-assay-error reading for the Gaussian strategy.
**Disadvantages:** cross-strategy comparisons at fixed sigma remain confounded by magnitude, so any
statement of the form "strategy X is harder than strategy Y" is not licensed - you can only compare
strategies at matched achieved SD, which your grid does not exactly provide (it would require
interpolation). A reviewer may say this is documenting the problem rather than fixing it.

**Ranking by literature support:** Option 1 (two direct QSAR precedents) > Option 2 (one chemistry
precedent, and the only one that explicitly matches mechanisms) > Option 4 (practised by the
predecessor, but as reporting rather than as design) > Option 3 (strong implementation, no
cheminformatics precedent, definition only in code).

**Statement of inconsistency, required by the brief:** the literature does not agree. Kolmar 2021
and Cortes-Ciriano 2015 scale to endpoint range; Heid 2023 uses absolute label units; SRBench uses a
multiple of target RMS; fluid-dynamics work uses SNR. No paper read here states a best practice for
making six *different mechanisms* comparable, beyond Heid's equal-SD matching of four homoscedastic
distributions.

---

## Sources actually read (URLs fetched)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC8613965/ - Kolmar & Grulke 2021, J. Cheminform. 13:92, DOI 10.1186/s13321-021-00571-7 (fetched 4x for Methods, Introduction, Discussion)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10336963/ - Heid, McGill, Vermeire, Green 2023, JCIM 63:4012, DOI 10.1021/acs.jcim.3c00373 (fetched 2x)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3628986/ - Kalliokoski, Kramer, Vulpetti, Gedeck 2013, PLoS ONE 8:e61007, DOI 10.1371/journal.pone.0061007
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7178569/ - Li & Fourches (MolPMoFiT) 2020, J. Cheminform. 12:27, DOI 10.1186/s13321-020-00430-x
- https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:26038978... - Cortes-Ciriano, Bender, Malliavin 2015 ABSTRACT ONLY, JCIM 55:1413, DOI 10.1021/acs.jcim.5b00101
- https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:28691113... - Zhu et al. 2017 ACS Omega 2:2805 ABSTRACT ONLY, DOI 10.1021/acsomega.7b00274
- https://arxiv.org/pdf/2107.14351 (downloaded, pdftotext) - La Cava et al., SRBench, arXiv:2107.14351
- https://raw.githubusercontent.com/cavalab/srbench/master/experiment/evaluate_model.py and .../analyze.py - SRBench reference implementation of `target_noise`
- https://ar5iv.labs.arxiv.org/html/2112.02751 - Nakamura & Fukagata, arXiv:2112.02751
- https://arxiv.org/html/2502.17771 - Kim et al., "Sample Selection via Contrastive Fragmentation for Noisy Label Regression" (Gaussian noise SD randomised up to 30%/50% of the label RANGE; levels 30% and 50%)

## Paywalled / could not retrieve
- **Cortes-Ciriano, Bender, Malliavin 2015, JCIM 55(7):1413-1425** - pubs.acs.org returned HTTP 403; no preprint on HAL (Anubis block), none on arXiv/ChemRxiv. Abstract only. **Their exact noise grid and symbol are unknown to me.**
- **Sheridan, Karnachi, Tudor, Xu, Liaw, Shah, Cheng, Joshi, Glick, Alvarez, "Experimental Error, Kurtosis, Activity Cliffs, and Methodology: What Limits the Predictivity of QSAR Models?", JCIM 2020, 60(4):1969-1982, DOI 10.1021/acs.jcim.9b01067** - ACS paywalled, no open version found. Not read; do not cite a noise parameterisation from it.
- **Kramer, Kalliokoski, Gedeck, Vulpetti, "The Experimental Uncertainty of Heterogeneous Public Ki Data", J. Med. Chem. 2012, 55(11):5165-5173, DOI 10.1021/jm300131x** - ACS paywalled, not retrieved. The 0.68 log-unit figure quoted above is from the companion Kalliokoski PLoS ONE paper, which IS open, and from Kolmar's citation of both.
- **Brown, Muchmore, Hajduk 2009, Drug Discov. Today 14:420-427** - Elsevier paywalled; PubMed abstract only.
- **Zhu, Tropsha et al. 2017 ACS Omega** - full text not fetched (pubs.acs.org 403); abstract confirms they simulated errors by "randomizing the activities of part of the compounds" (a corruption-FRACTION convention, not an SD convention) but I did not read their equations.
