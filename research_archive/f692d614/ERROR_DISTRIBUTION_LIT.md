# Is a Gaussian-only noise study missing a real experimental-error case?
Literature check, 2026-08-21. Every quote below was read in the source named.

---

## E. BOTTOM LINE (first, because it is the answer)

**Dropping the distribution axis is defensible and directly supported by a published control experiment.**
Heid et al. tested Gaussian, uniform, hyperbolic and bimodal noise at matched SD and the curves
overlapped. Two other noise-injection studies in this literature (Cortes-Ciriano 2015,
Kolmar & Grulke 2021) used Gaussian noise only.

Suggested Methods sentence (one):

> All noise was drawn from Gaussian distributions; Heid et al. found no difference in model
> performance between Gaussian, uniform, hyperbolic and bimodal error at matched standard
> deviation, so we vary *where* and *how much* error is placed rather than its distributional shape.

**The real gap a reviewer can raise is not shape — it is that all six strategies are zero-mean and
random.** Two documented, non-random error modes are not represented:

1. **Censoring at assay limits** (right-censored ">" values). Documented at 42% of hERG,
   58-63% of CYP, 8% of permeability measurements in an industrial ADME-T set
   (Svensson et al. 2024, Table 1). Censoring is a one-sided, deterministic distortion at one
   end of the range, not a symmetric perturbation. The Outlier and Quantile strategies put
   *more spread* at the extremes; censoring instead *compresses and biases* them. Not covered.
2. **Systematic offsets between labs / assays / data sources.** Documented: pKi values run
   0.355 log units above pIC50 values for the same pairs (Kalliokoski 2013); cytotoxicity
   sigma_E drops from 0.98 (metabolic assays) to 0.69 (identical setup) across labs
   (Cortes-Ciriano 2016); ChEMBL curation records 1000-fold unit-transcription errors
   (Papadatos 2015). A constant additive offset applied to a subset is zero-variance in the
   mean sense the six strategies use, so none of them produce it. Not covered.

Everything else the empirical literature documents — a roughly Gaussian core in log units,
heavy tails from annotation errors, a spread of ~0.5-1.0 log units — is covered by the
Gaussian and Outlier strategies.

**Honest caveat on B:** the strongest evidence I found runs *against* the value-proportional and
heteroscedastic strategies for potency data. Kalliokoski et al. state flatly that
"The DeltapIC50's depend neither on the average measured pIC50 nor on any of the ligand
properties examined." Those two strategies are best justified as censoring/dynamic-range proxies,
not as "assay error grows with potency."

---

## A. Is experimental error Gaussian in log units?

**Kalliokoski, Kramer, Vulpetti, Gedeck (2013), PLoS ONE 8(4):e61007, doi:10.1371/journal.pone.0061007** (read, full text, PMC3628986)
- "By fitting a Gaussian distribution to the central part of the distribution we were able to
  compare the variability of the pIC50 data to the variability of the pKi data."
- "Standard deviations for the fitted Gaussian distributions are sigma_pIC50 = 0.87 and
  sigma_pKi = 0.69" (upper threshold 2.0).
- "sigma_pIC50 = 0.68, MUE_pIC50 = 0.54 and MedUE_pIC50 = 0.43" (threshold 2.5).
- "Roughly 70% of all DeltapIC50's are smaller than one log unit."
- Tails: "We found that very high differences in pIC50 (DeltapIC50>2.5) were in most cases due to
  annotation errors." / "Unit errors are the most common error." / "The two outer diagonal lines
  indicate the 2.5 log unit threshold, outside which the probability for finding faulty pairs of
  measurements is very high."

Reading: Gaussian **core** in log units, plus a heavy tail that is *not* assay noise but curation
error. That is exactly a Gaussian + Outlier mixture.

**Cortes-Ciriano & Bender (2016), ChemMedChem 11(1):57-71, doi:10.1002/cmdc.201500424** (abstract read in full)
- "mean unsigned error (MUE) value of 0.61-0.76 ... and a standard deviation of 0.76-1.00 pIC50 units"
- "sigma_E = 0.47-0.48 pKi units and sigma_E = 0.57-0.61 pIC50 units, respectively" for ligand-protein data
- "annotation errors are responsible for the high discordance observed for some pairs of measurements"
- No explicit statement about distribution shape.

**Landrum & Riniker (2024), JCIM 64(5):1560-1567, doi:10.1021/acs.jcim.4c00049** (read, PMC10934815)
- Minimal curation: "almost 65% of the points differ by more than 0.3 log units, 27% differ by
  more than one log unit"; maximal curation: "48% ... by more than 0.3 log units, 13% by more than one".
- Does **not** characterise the shape; quantifies total disagreement only.
- Identifies specific bad assays rather than treating error as random: "These assays share a
  corresponding author and include a significant number of overlapping compounds, with results that
  are sometimes inconsistent." 239 assays manually removed.

---

## B. Heteroscedastic / value-dependent error?

**Against, for potency data:** Kalliokoski 2013 — "The DeltapIC50's depend neither on the average
measured pIC50 nor on any of the ligand properties examined" and "the variability does not depend on
any specific ligand properties such as logP, MW, PSA etc."

**For, via censoring:** Svensson, Friesacher, Winiwarter, Mervin, Arany, Engkvist (2024),
"Enhancing Uncertainty Quantification in Drug Discovery with Censored Regression Labels",
arXiv:2409.04313 (read, HTML). Table 1 censoring fractions: CYP3A4 61% (all right-censored),
CYP2C9 63% / 58%, hERG 42%, permeability 8%, solubility 5% (6% left), CLint 8%, LogD 0%.
"if no response is observed within this range, the experiment may only indicate that the response
lies above or below the tested concentrations."

Note for the paper: LogD is reported as 0% censored, hERG 42%. So the censoring concern is
endpoint-specific and bites hERG hardest.

---

## C. Systematic / non-random error

**Kalliokoski 2013**: "On average, the measured pKi values are 0.355 log units larger than the
measured pIC50 values, corresponding to a factor of 2.3." — a documented constant offset between
data types.

**Cortes-Ciriano & Bender 2016**: sigma_E = 0.98 for metabolic assays vs 0.69 "when using the 1388
pIC50 pairs measured using exactly the same experimental setup" — protocol identity halves the
variance.

**Papadatos, Gaulton, Hersey, Overington (2015), J Comput Aided Mol Des 29:885-896,
doi:10.1007/s10822-015-9860-5** (read, PMC4607714), reporting on ChEMBL curation and summarising
Tiikkainen et al.:
- "the most frequent source of discrepancies was the structure of the ligand, followed by the target
  assignment, the activity value and finally the activity type"
- putative activity issues: "Unrealistically high or low activity values; Multiple values for the same
  ligand-protein pair derived from a single publication; Multiple citations of a specific activity
  value (exact or rounded) for the same ligand-protein pair across several publications leading to
  redundancy; Unit transcription and conversion errors"
- "1000-fold activity value difference being recorded in the database"
- "even when assay conditions appear to be the same, significant variability is observed between
  measurements taken in different labs"

None of the six strategies produce a non-zero-mean perturbation, a duplicate-record conflict, or a
wrong-structure record. This is the honest gap.

---

## D. Does distribution shape matter?

**Heid, McGill, Vermeire, Green (2023), "Characterizing Uncertainty in Machine Learning for
Chemistry", JCIM 63(13):4012-4029, doi:10.1021/acs.jcim.3c00373** (read, PMC10336963) — verified:
- "Both the training and test sets contained noise. We did not observe any difference in overall
  model performance between different error distributions, as long as the mean and standard
  deviation of the noise was the same, respectively."
- Figure 2 caption: "Performance for models trained with different noise distributions applied to
  the data set. Both the training and test sets contain noise. Left: The applied noise distributions,
  each with standard deviation of 1 kcal/mol, shown at left. Right: Root mean squared error for the
  different noise distributions as a function of data set size. The four noisy data sets yield very
  similar performance (points overlap in the figure)."
- Their own framing of the axis the user kept: "Noise in the target data can be of random, uniform
  nature (homoscedastic), afflicting all data points with the same error probability distribution,
  or systematic (heteroscedastic), where different domains of data are affected by different error
  probability distribution." (NB: Heid's use of "heteroscedastic" = domain-dependent, which is
  exactly the Threshold/Quantile/Outlier axis.)

**No counter-example found.** I searched for a cheminformatics noise-injection study where
distribution shape at matched SD changed the conclusion and did not find one.

Precedent for Gaussian-only:
- **Cortes-Ciriano, Bender, Malliavin (2015), JCIM 55(7):1413-1425, doi:10.1021/acs.jcim.5b00101**
  — "noise simulated by sampling from Gaussian distributions with increasingly larger variances"
  (from search-result metadata; full text not retrieved).
- **Kolmar & Grulke (2021), J Cheminform 13, doi:10.1186/s13321-021-00571-7** (abstract read in full
  via Europe PMC, PMC8613965) — "Up to 15 levels of simulated Gaussian distributed random error was
  added to the datasets"; conclusions explicitly scoped "at least under the conditions of Gaussian
  distributed random error".

---

## Sources actually read

| Source | What I read |
|---|---|
| Kalliokoski, Kramer, Vulpetti, Gedeck, PLoS ONE 8(4):e61007, 2013, doi:10.1371/journal.pone.0061007 | Full text (PLOS + PMC3628986); verbatim quotes above |
| Heid, McGill, Vermeire, Green, JCIM 63(13):4012, 2023, doi:10.1021/acs.jcim.3c00373 | Full text (PMC10336963); Fig 2 caption + noise paragraph verbatim |
| Landrum & Riniker, JCIM 64(5):1560-1567, 2024, doi:10.1021/acs.jcim.4c00049 | Full text (PMC10934815) |
| Cortes-Ciriano & Bender, ChemMedChem 11(1):57-71, 2016, doi:10.1002/cmdc.201500424 | Full abstract (Wiley landing page); no full text |
| Svensson et al., arXiv:2409.04313, 2024 | Full HTML incl. Table 1 censoring fractions |
| Papadatos, Gaulton, Hersey, Overington, JCAMD 29:885-896, 2015, doi:10.1007/s10822-015-9860-5 | Full text (PMC4607714) — used as the readable source for Tiikkainen's findings |
| Kolmar & Grulke, J Cheminform 13, 2021, doi:10.1186/s13321-021-00571-7 | Full abstract (Europe PMC REST); no full text |

## Could not retrieve

- **Kramer, Kalliokoski, Gedeck, Vulpetti, "The Experimental Uncertainty of Heterogeneous Public Ki
  Data", J Med Chem 55(11):5165-5173, 2012, doi:10.1021/jm300131x.** Paywalled at ACS; abstract
  elided by publisher in the Semantic Scholar API; the Novartis OAK open-access archive
  (oak.novartis.com/6818) has been discontinued. I confirmed title/authors/venue/year/DOI only.
  **I have NOT read any statement by Kramer et al. about distribution shape and make no claim about
  one.** The sibling paper (Kalliokoski 2013, same group) is the substitute used above.
- **Tiikkainen, Bellis, Light, Franke, "Estimating Error Rates in Bioactivity Databases",
  JCIM 53(10):2499-2505, 2013, doi:10.1021/ci400099q.** Paywalled at ACS (403), abstract elided in
  Semantic Scholar. All Tiikkainen content above is quoted from Papadatos et al. 2015 reporting on it,
  and is labelled as such. **No numeric error rate from Tiikkainen was retrieved.**
- **Cortes-Ciriano, Bender, Malliavin, JCIM 55(7), 2015, doi:10.1021/acs.jcim.5b00101.** Paywalled;
  the "Gaussian distributions with increasingly larger variances" phrasing comes from search-result
  metadata, not from the paper text.
- **Sheridan, Karnachi, Tudor, Xu, Liaw, Shah, Cheng, Joshi, Glick, Alvarez, "Experimental Error,
  Kurtosis, Activity Cliffs, and Methodology", JCIM 60(4), 2020, doi:10.1021/acs.jcim.9b01067.**
  Paywalled; I retrieved only a machine-generated summary (activity-cliff density predicts
  modelability better than experimental-error metrics) and **quote nothing from it**. Its "kurtosis"
  angle is the one place a shape argument could exist — worth a library copy before citing.
