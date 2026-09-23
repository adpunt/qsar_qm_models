# Methods and Introduction — replacement text

Written 2026-09-21. **This document supersedes sections M5, M6, §I4, §I6 and "The current main picture"
of `PAPER_REVISION_GUIDE_FINAL.md`.** Nothing has to be read in the guide to use what is here.

Everything in a ```latex fence pastes into the manuscript. Everything under an **Author notes**
heading is for you and does not go into the paper.

`paper.tex` was not edited and is never edited here; it is an inert download from Overleaf.

The text is written to `PAPER_HOUSE_STYLE.md`, which is 122 rules taken from nineteen reference
papers. Every block below was drafted against that file and then audited against it a second time by
a reader who had not written it.

## What is in here

| Section | Replaces | Why it was rewritten |
|---|---|---|
| The harvest, question by question | "The current main picture" in the guide | you did not recognise the questions it claimed you had asked, so it was rebuilt from the aims paragraph and `RERUN_PLAN.md` |
| §I4 | new paragraph in the Introduction | the argument held and the prose did not; three of its source claims were also wrong |
| §I6 | `paper.tex:188`, the aims | two of the three aims promised less than the paper delivers |
| M5 | the Uncertainty quantification subsection | you called everything after the equations garbage and asked for it far shorter |
| M6 | the Performance metrics subsection | you were not happy with the last pass, which had grown it to 893 words |

---

# The harvest, question by question

*Replaces "The current main picture — read this first" in the guide.*

## What the harvest answers, question by question

These are the four questions the replacement aims paragraph sets, in its order. Every answer was
recomputed this session from `results/decisions_arc_20260916/`, harvested on 17 September 2026. Of
the 1,565 combinations of dataset, model, representation and noise condition in the grid, 1,551 are
complete (`d0_coverage.csv`). No answer below covers the whole grid.

**One. How do the representation and the model divide the variance in predictive accuracy, and does
that division change when the question is robustness?** The division changes between the two
questions. AUC$_{norm}$ is the normalised area under the R² retention curve, and a higher value
means more accuracy kept as the noise level rises. On QM9 under the `gaussian` condition, the model
explains 29.1 per cent of the variance in R² and the representation explains 26.6 per cent. On
AUC$_{norm}$ under that same condition, the model explains 49.5 per cent of the variance and the
representation explains 9.2 per cent. The interaction between model and representation explains a
further 20.4 per cent of the variance in AUC$_{norm}$. Those four shares come from a two-way
analysis of variance on QM9 over 13 models, 6 representations and 10 replicates (`anova_eta2.csv`).

**Two. What does the choice of model buy at a realistic amount of label error, and does making a
model probabilistic change how noise hurts it?** The second half of that question is answered and
the first half is not. Each model was compared against its own probabilistic version 153 times,
paired on the replicate, once per representation and noise condition. The probabilistic version has
the higher AUC$_{norm}$ in 46 of those 153 comparisons and the lower AUC$_{norm}$ in 39. Both counts
are at p below 0.05 on a two-sided Wilcoxon signed-rank test (`d10_probabilistic.csv`). The
remaining 68 of the 153 comparisons separate the two versions no further than the replicates do. The
first half was Q3 in `RERUN_PLAN.md` §7.0, and you withdrew it on 2026-09-18.

**Three. Does the kind of noise matter or only the amount, and do the QM9 answers hold on measured
labels?** The kind of noise matters only where the errors are correlated within a scaffold family.
Fifteen pairs of noise conditions were compared on QM9 at ECFP4, and 6 of those 15 pairs differ at p
below 0.05 (`d3_condition_pairs.csv`). The `grouped_shifted` condition is one side of 5 of those 6
differing pairs. At the same noise level, `laplace`, `student_t_nu5` and `outlier_p10` are each
indistinguishable from `gaussian`, at p = 0.69, p = 0.94 and p = 0.94. The model ranking measured on
QM9 does not carry over to the three assay datasets. For each of 83 combinations of representation,
noise condition and assay dataset, QM9's model ranking was compared against that dataset's own
ranking. Only 20 of those 83 comparisons agree at p below 0.05 (`d9_rank_agreement.csv`). Read a
model ordering as holding for the dataset it was measured on and no further.

**Four. How do a model's uncertainty estimates behave as its training labels are corrupted?**
Predicted uncertainty rises with the noise level in 922 of the 1,003 combinations of dataset, model,
representation and noise condition that report a slope for both uncertainty components
(`unc_slopes.csv`). Predicted uncertainty also still ranks which predictions to trust. Its Spearman
correlation with the absolute error against the clean label is positive in 1,004 of 1,058
combinations of dataset, model, representation and noise condition (`d7_q6.csv`). Whether predicted
uncertainty points at the corrupted labels has no answer in this harvest. That question is settled
against a permutation band, and the band in this harvest was computed on the out-of-fold error
rather than on the predicted uncertainty. The corrected band has not been re-run (`RERUN_PLAN.md`
§14.29, defect 1).

Clipping the labels runs the other way from adding noise to them. Sixty-six combinations of dataset,
model and representation ran under the `censoring` condition, which clips labels instead of
perturbing them. In none of those 66 does predicted uncertainty rise with the fraction of labels
clipped (`unc_q5.csv`). No model in this harvest warns you that its training labels have been
clipped. Do not read a low predicted uncertainty as evidence that the labels behind it are intact.

---

## Clean accuracy against AUC$_{norm}$, and which models vary both uncertainty components per molecule

What a model scores on clean labels and what it keeps as those labels are corrupted are separate
properties. Seventy-five model-and-representation pairings ran on all four datasets under the
`gaussian` condition, with each pairing's two scores rescaled inside its own dataset. Across those
75 pairings the rank correlation between clean R² and AUC$_{norm}$ is −0.350, at p = 0.0021
(`standout_pairs.csv`). A pairing that scores well on clean labels therefore tends to keep less of
that score as the noise level rises.

The same separation shows in the uncertainty estimates. Thirteen models emit a predicted
uncertainty, and they are not the same 13 models that entered the analysis of variance above. Seven
of those 13 report one of the two uncertainty components as a single number per fit
(`d7_support.csv`). For those 7 models the per-molecule component and the per-fit component cannot
be compared against each other. Do not print an aleatoric and an epistemic value from them side by
side.

One row below is one of the 6 models whose aleatoric and epistemic components both vary per
molecule. The first count is the combinations of dataset, representation and noise condition in
which the aleatoric component rises with the noise level while the epistemic one holds. The second
count is the combinations in which both components rise. The remainder of each row's total is the
count in which neither component moves clearly (`unc_slopes.csv`).

| model | aleatoric rises, epistemic holds | both components rise | combinations of dataset, representation and noise condition |
|---|---|---|---|
| GP, heteroscedastic (`het_gp_rbf`) | 59 | 23 | 91 |
| DNN, variational, heteroscedastic (`dnn_vbll_hetero`) | 52 | 2 | 54 |
| MLP, variational, heteroscedastic (`mlp_vbll_hetero`) | 52 | 4 | 63 |
| DNN, Bayesian, variance head (`dnn_bnn_full_mve`) | 63 | 31 | 109 |
| MLP, Bayesian, variance head (`mlp_bnn_full_mve`) | 57 | 20 | 99 |
| Quantile forest (`qrf`) | 0 | 90 | 99 |

The quantile forest never separates its two components. Both of its components rise together in
90 of its 99 combinations of dataset, representation and noise condition, and the aleatoric
component rises alone in none of them. Report the quantile forest's uncertainty as one number, and
do not split it into a data-noise part and a model-uncertainty part.

---

# §I4 — what real label error looks like

Rewritten from scratch 2026-09-21, then run against `PAPER_HOUSE_STYLE.md` the same day.
Two paragraphs, 327 words, 15 sentences. First paragraph 7 sentences and 143 words, second
8 sentences and 184 words. Median sentence 22 words, longest 29, shortest 16. No \texttt in
the LaTeX.

Goes between `paper.tex:184` (the Learning with Noisy Labels paragraph) and the
representation-comparison paragraph. It replaces the two-sentence stub at `paper.tex`
marked `% TODO: finish later`, which already carries the opening move in the author's
own words.

---

## The LaTeX

```latex
Studies of noise robustness typically add independent, zero-mean, Gaussian error to the
labels, and real experimental error is none of those three things. \citet{Kruger2012}
rejected normality of bioactivity differences by an Anderson-Darling test and fitted a
Laplace distribution instead. \citet{Kalliokoski2013} fitted a Gaussian to 16,844 pairs of
repeat pIC50 measurements, and only to the inner part of that distribution, out to 2.5 log
units. The largest difference within a single pair was 7.7 log units, far outside what the
Gaussian they fitted allows. They sorted the pairs into bands by the size of that
difference, and inspected ten pairs from each band. In the two widest bands, 9 and 10 of
those ten pairs were annotation errors rather than imprecision. A heavy tail in a public
bioactivity dataset may therefore record what went into the database rather than how precise
the assay was.

Shape is only one of the three assumptions, and independence and zero mean fail as well.
\citet{Bentz2013} had 23 laboratories measure P-glycoprotein transport, and ascribed 62\%
of the variance in log efflux ratio to differences between laboratories. A dataset assembled
from several sources can therefore carry a shared offset across whole families of related
compounds, rather than independent error. Zero mean fails because a value too low to measure
is recorded at the assay's detection limit rather than as a measured number.
\citet{Svensson2025} count censored labels in fifteen industrial assays, of which thirteen
carry some censoring and eight have between a quarter and two thirds of their labels
left-censored. The seven noise conditions tested here vary shape, correlation and bias
separately, at a level set as a fraction of the spread of the clean training labels. A
Student-$t$ heavy enough to have no finite variance cannot be set to a fixed fraction of
that spread. The heaviest tail injected here therefore stops at five degrees of freedom,
which is a limit of how the level is set and not a reading of the literature.
```

Citation keys used: `Kruger2012`, `Kalliokoski2013`, `Bentz2013`, `Svensson2025`. All four
are defined in `citations.bib` and none is cited in `paper.tex` today. None of them is in
`paper_inline_bbl.bbl`, so that build will not resolve them until it is regenerated.

---

## What changed from the previous draft, and why

Three paragraphs became two, and 330 words became 284. A later house-style pass took it to
327 words, all of the growth being denominators, a band definition and one sentence saying
what five degrees of freedom does not claim. Those changes are listed at the end.

**Dropped: "which the Gaussian they fitted puts at six in a thousand million million."** It
is arithmetic done on Kalliokoski's numbers by an earlier session, not a figure they print.
Replaced by "far outside what the Gaussian they fitted allows", which makes the same point
without a number nobody can trace.

**Dropped: "could only fit a Gaussian ... after truncating the tail."** Kalliokoski never
attempt an untruncated fit and never report one failing. They give a different reason for
the cut — removing invalid pairs. The replacement says what they did: they fitted to the
inner part of the distribution, out to 2.5 log units.

**Dropped: "transcription errors, receptor-subtype confusion or assay mix-ups."** That is a
paraphrase of the paper's list that leaves out three of the error types it names. Their own
summary word is "annotation errors", and the paragraph now uses it.

**Dropped: the whole third paragraph and the claim that matching the real tail needs a
Student-$t$ with about one degree of freedom.** Nothing on disk supports one degree of
freedom, and the source paper contains no Student-$t$ at all. See the open question below.

**Dropped: "The more consequential departure from that convention is independence rather
than shape."** It ranked two departures against each other with nothing measured on either
side. The replacement opener says only that shape is one of three assumptions and the other
two fail as well, which is what the paragraph then shows.

**Dropped: "The tail may therefore belong to the record rather than to the value."**
Replaced by "may therefore record what went into the database rather than how precise the
assay was."

---

## Every source claim, and where I read it

Files read this session in `/Users/apunt/repos/qsar_qm_models/research_archive/28450b4e/`.

| Claim in the paragraph | Source | Where I read it |
|---|---|---|
| Rejected normality of bioactivity differences by Anderson-Darling test | Krüger & Overington 2012 | `kruger.txt:107` — "is non-normal as established by Anderson-Darling test [24] (p<2e-16)" |
| Fitted a Laplace distribution instead | Krüger & Overington 2012 | `kruger.txt:253` — "Both distributions can be approximately described by a Laplace distribution" |
| 16,844 pairs of repeat pIC50 measurements | Kalliokoski et al. 2013 | Figure 2 caption for the count of pairs. Note the conflict below. The quantity is pIC50: `kalliokoski.txt:329` writes the same distribution's extreme as a ΔpIC50, and the paper's own subject is published IC50 data. |
| Gaussian fitted to the inner part only, out to 2.5 log units | Kalliokoski et al. 2013 | `kalliokoski.txt:431` — Table 3 title, "fitted to the inner part of the distribution"; `:420` — "between 0.05 (lower threshold) and a variable upper threshold (1.5, 2.0 and 2.5)" |
| Largest difference within a single pair, 7.7 log units | Kalliokoski et al. 2013 | `kalliokoski.txt:329` — "The largest ΔpIC 50 is 7.7 log units." |
| Pairs sorted into bands by size of difference, ten inspected per band; 9 and 10 in the two widest | Kalliokoski et al. 2013 | Table 2, column "# invalid pairs out of 10": band 4.7–7.8 gives 9, band 3.2 gives 10. The order in the paragraph follows that order, widest band first. |
| Those were annotation errors | Kalliokoski et al. 2013 | `kalliokoski.txt:404` — "very high differences in pIC 50 (ΔpIC 50 >2.5) were in most cases due to annotation errors"; Figure 4 caption, "The extreme disagreements are all due to clear errors." |
| 23 laboratories, P-glycoprotein transport | Bentz et al. 2013 | Abstract — "23 participating pharmaceutical and contract research laboratories and one academic institution" |
| 62\% of the variance in log efflux ratio between laboratories | Bentz et al. 2013 | `bentz.txt:601` — "with 62% of the variance ascribed to laboratory-to-laboratory variability"; Table 7, Log ER w/o Inhibitor column |
| Fifteen industrial assays, thirteen with some censoring, eight with between a quarter and two thirds of their labels left-censored | Svensson et al. 2025 | Table 1, `svensson.txt:282-380`. Left-censored: 32, 43, 0, 12, 35, 0, 25, 61, 63, 58, 8, 5, 42, 0, 8. Right-censored: 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 6, 0, 8, 6. Any censoring: 13 of 15. Between 25\% and 63\% left-censored: 32, 43, 35, 25, 61, 63, 58, 42 — 8 of 15. Counted this session. |
| Heaviest injected tail is a Student-$t$ with five degrees of freedom | this study | `noise_conditions.json:63` names `student_t_nu5` and `:139` sets `"nu": 5.0` |

---

## Three things I did not write, and would not without your word

**Kalliokoski's 16,844 does not reconcile inside their own paper.** The text says 10,895
IC50 values on 3,480 protein-ligand systems "yielding 20,356 pairs of independent
measurements"; Figure 2's caption says 16,844 pairs, and Figure 1's says 9,465 values. The
paper does not explain the difference. I used 16,844 because it is the count attached to the
distribution the 7.7 log units comes from, but it is the smaller of two published numbers.

**Bentz's 62\% is not a Caco-2 number.** Table 7's log efflux ratio column pools three cell
lines — 7 MDCKII-MDR1, 4 LLC-PK1-MDR1 and 11 Caco-2 laboratory rows in Table 6. The
paragraph says "23 laboratories measure P-glycoprotein transport", which is the paper's own
headline count and does not claim a cell line. `PAPER_REVISION_GUIDE_FINAL.md:341-343`
currently says "a Caco-2 round robin across eleven laboratories" for the same 62\%, and that
is wrong; it also contradicts line 1068 of the same file, which says 23. That is a separate
fix, in the Methods, not this paragraph.

**The Krüger sentence at `PAPER_REVISION_GUIDE_FINAL.md:343-344` is wrong and is not fixed
here.** It says "We took the Student-$t$ and Laplace shapes from the distributions fitted to
repeated public bioactivity measurements \citep{Kruger2012}". Krüger fitted no Student-$t$ —
the word does not appear in their paper — and the Laplace was fitted to human-to-rat ortholog
and human paralog differences, which is a difference between two proteins, not a repeat
measurement. That sentence is in the Methods and needs its own rewrite.

---

## The house-style pass, 2026-09-21

Six passes over the LaTeX, following the "How to run this on a draft" section of
`PAPER_HOUSE_STYLE.md`. No number changed value. Nine changes.

**"disagreement" was the name of a measured quantity.** The quantity Kalliokoski measure is
the difference between two pIC50 measurements of the same pair. The paragraph now says
difference, which decomposes into a statistic and the data it was taken on.

**"the two widest bands of disagreement" arrived pre-known.** Bands were never introduced.
One sentence now says they sorted the pairs into bands by the size of the difference and
inspected ten pairs from each band, before any band is read off.

**"the censored fraction" was a coined name.** Replaced by counting censored labels, which
is what Svensson do.

**Two counts had no denominator.** "thirteen carry censored labels and eight are between a
quarter and two thirds censored" now reads "fifteen industrial assays, of which thirteen
carry some censoring and eight have between a quarter and two thirds of their labels
left-censored". The fifteen is named before both numbers, and the eight is a count of
assays while the quarter and the two thirds are fractions of labels, which the old sentence
did not distinguish.

**"a matched amount of noise" and "that amount" named a quantity the project already has a
name for.** The noise level is a fraction of the spread of the clean training labels, and
the paragraph now says so at the point the Student-$t$ argument needs it.

**"a public potency set" was invented.** It is a public bioactivity dataset.

**"a Gaussian of that width" left the unit unnamed.** Replaced by the Gaussian they fitted.

**"Shape is the smaller departure, and independence is the larger one" ranked two things
with nothing measured on either side.** Replaced by a join sentence pointing back at the
three assumptions in the first sentence and forward at the two that the paragraph shows
failing.

**The paragraph ended on a number.** It now ends on one sentence saying that five degrees of
freedom is a limit of how the noise level is set, not a reading of the literature, which is
the same over-reading the previous draft invited with one degree of freedom.

Measured after the pass: two paragraphs, 327 words, 15 sentences, median 22 words, longest
29, shortest 16, no sentence over 30 or under 12, and no \texttt.

---

# §I6 — the close of the Introduction (replaces `paper.tex:188`)

## Replacement text, paragraph 1: the questions

```latex
Most QSAR models treat a label as a single value rather than as a draw from a distribution. The
probabilistic models in this study are the ones that do not. This research addresses four
questions about the behaviour of QSAR models under label noise. First, we ask how variance in
predictive accuracy divides between molecular representation and model architecture, and whether
it divides the same way for noise robustness. Second, we compare probabilistic models with their
deterministic counterparts, and ask whether being probabilistic makes a model more robust to
label noise. Third, we ask whether the kind of noise matters or only how much of it is added, and
whether the patterns we find carry across molecular properties. Fourth, we follow a model's
per-sample uncertainty estimates as its training labels are corrupted. We ask whether those
estimates still rank which predictions to trust. We also ask whether they point to the labels
that were corrupted, over and above the rise in uncertainty that any noise produces. The same
models and representations are run on a computed quantum-mechanical property and on three
experimentally measured endpoints. Together the four questions ask what, if anything, makes a
QSAR model robust to label noise.
```

## Replacement text, paragraph 2: the preview

```latex
Accuracy and noise robustness do not have the same causes. Under Gaussian noise on the computed
property, the model and the representation shape predictive accuracy about equally. Of the
variance in \mbox{AUC$_{norm}$} under that same noise, the model accounts for about half and the
representation for under a tenth. \mbox{AUC$_{norm}$} is the share of its clean accuracy a model
keeps as label noise rises, not an area under a receiver operating characteristic curve. The kind
of noise matters as well, though not in the way earlier noise studies have varied it. Changing
the shape of an individual label's error does not measurably change \mbox{AUC$_{norm}$}, once
every condition delivers the same amount of noise. Giving a whole scaffold family a shared offset
lowers \mbox{AUC$_{norm}$} against each of the five other noise conditions. Across 153
comparisons of a probabilistic model with its deterministic counterpart, 46 significantly favour
the probabilistic model and 39 the deterministic one. Making a model probabilistic does not by
itself make it more robust. The model ranking on the computed property and the ranking on the
three measured endpoints disagree in 56 of 83 comparisons. The uncertainty estimates depend on
the kind of noise as well. Under independent noise, a model's uncertainty separates what the
model does not know from what cannot be known in a minority of settings. It separates them in
fewer settings still under a shared offset. Under labels clipped at an assay limit it does not
separate them at all. On every model and representation where clipping ran, the model grows more
confident as its labels grow more wrong.
```

---

# Author notes

## What each of the three original aims promised, and what it now promises

**Original aim one** promised "the contributions of molecular representation and model architecture
on both overall predictive performance and, more specifically, noise robustness". **Unchanged in
substance.** It is now worded as a division of variance between the two, because that is what the
analysis of variance measures and what §R1 reports.

**Original aim two** promised two things welded into one sentence: probabilistic against
deterministic robustness, and "whether their per-sample uncertainty estimates track which
individual labels have been corrupted". **Split into two questions, and the second now promises
less than a yes or no.** The paired comparison gives 46 pairings where the probabilistic version is
significantly more robust against 39 where it is significantly less, out of 153
model-pair-by-representation-by-condition comparisons (`d10_probabilistic.csv`, summarised at D10).
That is not a yes or no, so question two now asks what making a model probabilistic buys, and
question four asks separately what happens to the uncertainty estimates. Your clause about
controlling for the population-level rise in average uncertainty is kept, in plainer words.

**Original aim three** promised "the generalizability of noise-robustness patterns across different
noise-injection mechanisms and molecular properties". **This one now promises more, and is split.**
It carried the noise-kind question as a sub-clause of a generalisability question. The noise-kind
result is the paper's differentiator: only the shared-offset condition separates from the others,
by about 0.03 in \mbox{AUC$_{norm}$} against every one of the other five, while the five shape
conditions sit within 0.004 of each other (`d3_condition_pairs.csv`). So it becomes question three
in its own right, and the transfer across properties stays attached to it.

**NoiseInject is gone from this paragraph.** `RERUN_PLAN.md` §14.22 records it moving to the
availability statement on your call. Nothing here replaces it, so if that decision has changed the
sentence needs to come back.

## Counts

Paragraph one: 200 words, 11 sentences, median 18 words, longest 27, shortest 11.
Paragraph two: 264 words, 15 sentences, median 18 words, longest 24, shortest 10.
No sentence under five words in either, and \texttt appears nowhere in the replacement text.

## Where each preview claim comes from, read this run

- Model 49.5 per cent against representation 9.2 per cent of the variance in \mbox{AUC$_{norm}$},
  and model 29.1 against representation 26.6 for R$^2$, both under Gaussian noise, 13 models and 6
  representations, 780 rows: `results/decisions_arc_20260916/anova_eta2.csv`.
- The shape conditions costing nothing and the shared offset costing about 0.03 in
  \mbox{AUC$_{norm}$}: `d3_condition_pairs.csv`, all 15 condition pairs.
- Probabilistic against deterministic, 46 significant wins against 39 significant losses of 153:
  `d10_probabilistic.csv` and D10.
- Ranking transfer, median agreement 0.356 across 83 combinations, disagreeing in 56 of them:
  D9 and `d9_rank_transfer.csv`.
- Uncertainty separating the two components: 78 of 238 rows under Gaussian, 58 of 237 under the
  shared offset, 0 of 66 under clipped labels. Under clipped labels all 47 rows with a
  per-molecule data-noise slope and all 56 with a per-molecule model-uncertainty slope are
  negative. `unc_slopes.csv`.

## Three things to decide

**One. Whether the preview paragraph goes in at all.** Of the reference papers I read this run,
Kolmar and Grulke close their Introduction on a hypothesis and Dablander closes on six research
questions as a bulleted list. Neither previews a result. Hirschfeld does, in a closing paragraph
that opens "No method consistently satisfied the objective of producing a strong ranking of
errors" and ends on what the study could not identify. Hirschfeld is an arXiv preprint, not
Journal of Cheminformatics. So the preview has a precedent in the reference set, but not in the
target journal, and `PAPER_HOUSE_STYLE.md` rule 20 says framing names the question rather than the
finding.

**Two. Prose or a list.** Your original paragraph runs First, Second, Third in prose and that is
what I matched. Dablander runs six research questions as bullets and is in the target journal, so
a bulleted list is available if you prefer the questions to stand apart on the page. Cost of the
list: the two sentences that are not questions, the one about point values and distributions and
the one naming the four datasets, have to move out of the list and around it.

**Three. The variance split is quoted under Gaussian noise only.** Under the shared-offset
condition the model accounts for 30.5 per cent rather than 49.5, and the residual rises to 56.3.
The preview says "under Gaussian noise" for that reason. If you would rather the preview state the
result without a condition attached, it needs a different sentence, because the split is not the
same under all three conditions that have the full grid.

## Not checked

`anova_eta2.csv` has no dataset column. I read it as the computed property because the decisions
run's QM9 directory is the only source with 13 models by 6 representations by 10 replicates, but I
did not trace the rows back to the run.

The guide's current §I6 preview also claims that the random forest and the quantile forest are
never a bad choice and that LightGBM and XGBoost lose their standing on Caco-2 and hERG. I could
not confirm either from `DECISIONS.md` or the CSV files beside it this run, so neither is in the
replacement text.

---

## M5. Uncertainty quantification

> 🔴 **Superseded 2026-09-23.** The live M5 is in `PAPER_REVISION_GUIDE_FINAL.md`. Do not paste from this one.

*N.*ew subsection, absorbing paper.tex 217-219

```latex
\subsection{Uncertainty quantification}

Probabilistic models have different mechanisms of estimating uncertainty. Some of these produce an
uncertainty estimate which collapses all elements of uncertainty into a single confidence score,
others produce estimates which can be decomposed into an aleatoric and an epistemic component
\citep{kendall2017}. The aleatoric component describes underlying noise the model associates with
the data, while the epistemic component represents the uncertainty within the model's fit. In
theory, in response to injected label noise the aleatoric component should rise while the epistemic
component stays stable.

For a network that estimates label noise, we took the two components over $T = 100$ stochastic
forward passes:
\begin{equation}
\hat{u}^2_{\text{ale}}(x) = \frac{1}{T}\sum_{t=1}^{T}\hat{v}_t(x),
\qquad
\hat{u}^2_{\text{epi}}(x) = \frac{1}{T}\sum_{t=1}^{T}\big(\hat{\mu}_t(x) - \bar{\mu}(x)\big)^2 ,
\label{eq:sampling_split}
\end{equation}
where $\hat{\mu}_t(x)$ and $\hat{v}_t(x)$ are the mean and the variance returned for molecule $x$ on
pass $t$, and $\bar{\mu}(x)$ is the mean over passes.

For the QRF specifically we applied the law of total variance across the trees,
\begin{equation}
\hat{u}^2_{\text{ale}}(x) = \frac{1}{B}\sum_{b=1}^{B} s^2_b(x),
\qquad
\hat{u}^2_{\text{epi}}(x) = \frac{1}{B}\sum_{b=1}^{B}\big(m_b(x) - \bar{m}(x)\big)^2 ,
\label{eq:forest_split}
\end{equation}
where $B$ is the number of trees, and $m_b(x)$ and $s^2_b(x)$ are the mean and the variance of tree
$b$'s in-bag labels in that molecule's leaf.

Seven models were scored for uncertainty, each on ECFP4, PDV and ChemBERTa, giving 21
model-and-representation pairs on QM9 and on the three assay datasets. Four of them are the QRF, NGBoost, the GP
with an RBF kernel and the same GP with a small network predicting an observation noise for each
molecule. The other three are network transformations: the VBLL form of NN-$\alpha$, and the
full-BNN forms of NN-$\alpha$ and NN-$\beta$, each with a variance output head. Three of the seven return one of the two components as a
single number for the whole fit, or not at all. NGBoost makes one distributional fit and has no
epistemic component. The GP with one likelihood noise term and the VBLL transformation each learn one
aleatoric value that is copied onto every molecule. Each component is reported with a statement of
whether it varies per molecule. A component that does not vary is never rank-correlated against a
per-molecule quantity.

Because no held-out label is corrupted, we scored training molecules out of fold. Each training
block was divided into five inner folds grouped on Murcko scaffolds, and each molecule was scored by
a refit that excluded its own fold. All five inner folds are scored, except for NGBoost on QM9,
which scores three.

Of the noise conditions in this study, only three corrupt specific subsets of molecules, namely
grouped-wider, outlier and censoring. As the others give every label the same noise scale, there is
no structure to detect. Grouped-wider is keyed to the scaffold family, and a scaffold split holds
whole families out. Its structure is therefore flat on held-out molecules, and it is read on the
out-of-fold training molecules instead. Every uncertainty is reported as the model produced it, with
no post-hoc calibration.
```

**Still open in this subsection.**

- Decide numbered equations versus paper.tex's unnumbered $$...$$ convention.
- Decide whether the variance output head is defined in the Models subsection instead, beside NN-alpha and NN-beta.
- Replace or delete paper.tex:218, which says NGBoost, QRF and the BNN variants were not decomposed.

---

# AUTHOR NOTES — not part of the LaTeX

## Q1. The models that can be asked the decomposition question

Seven models are on the uncertainty runs. The list is `uncertainty_pairs.json`, which is the only
place that membership is written down, and each one's two components are declared in the table at
`scripts/uncertainty_decomposition.py:94-180`. I read both this session. Both generators read the
first file and the writer refuses a row that disagrees with the second.

| model, in the paper's words | code name | aleatoric | epistemic |
|---|---|---|---|
| QRF | `qrf` | yes, per molecule | yes, per molecule |
| NGBoost | `ngboost` | yes, per molecule | none |
| GP, RBF kernel | `gauche_rbf` | yes, one number per fit | yes, per molecule |
| GP with a per-molecule observation noise | `het_gp_rbf` | yes, per molecule | yes, per molecule |
| VBLL transformation of NN-alpha | `dnn_vbll` | yes, one number per fit | yes, per molecule |
| full-BNN on NN-alpha with a variance head | `dnn_bnn_full_mve` | yes, per molecule | yes, per molecule |
| full-BNN on NN-beta with a variance head | `mlp_bnn_full_mve` | yes, per molecule | yes, per molecule |

Six of the seven models have two components and can be asked the question at all. NGBoost is the
one that cannot, because one distributional fit has nothing to disagree with itself about.

Of those six, four have both components varying per molecule: the QRF, the GP with a per-molecule
observation noise, and the two variance-head networks. Those four are the only ones that can be
asked whether the aleatoric component points at the individual molecules whose labels were
corrupted. The GP with an RBF kernel and the VBLL transformation can only be asked the population
question, because their aleatoric value is the same number on every row.

One decision is yours here. `uncertainty_pairs.json` names four models whose split is reported as a
split: the GP with an RBF kernel, the VBLL transformation and the two variance-head networks. The
QRF and the GP with a per-molecule observation noise are on the uncertainty runs and are not on that
reported list. The file's own reason for the QRF is that both of its halves track the injected
corruption at +0.84 and +0.81 Spearman, which it calls one signal reported twice. For the GP with a
per-molecule observation noise the file says outright that whether its split is reported beside the
ordinary GP is a call you have not been asked. Adding either costs no run, because both models are
already fitted and already write both components.

## Q2. Expected calibration error

**The history, and it is short.** You removed it on 2026-08-19. `RERUN_PLAN.md:82` records the
instruction in your words: *"add in the revision guide the complete removal of ECE and remove it in
the figure generation scripts. Full removal not commenting out."* That is the only statement of the
reason I can find. I grepped every session log in
`~/.claude/projects/-Users-apunt-repos-qsar-qm-models/` for ECE this session, and the only hits are
later sessions quoting that same table row. The logs on disk start on 24 August, so the conversation
that decided it is not among them, and I cannot tell you what argument was made.

**What is on disk now.** Nothing on the live analysis path computes it. Three implementations
survive: `scripts/generate_figures.py:375-388`, which is the retired v1 figure script;
`scripts/uncertainty_analysis.py:130-160`; and `NoiseInject/noiseInject/uncertainty.py:177-201`. All
three do the same thing, which I read line by line: sort the molecules into ten equal-count bins by
predicted uncertainty, and in each bin take the gap between the mean predicted standard deviation
and the mean absolute error, weighted by how many molecules are in the bin.

**What it would cost to compute now.** One re-run of the figure job and no cluster experiment.
`scripts/generate_paper_figures_v2.py` already computes coverage at one and two standard deviations
from exactly the arrays an ECE needs, at lines 1919-1920, with the function at line 2668. A
calibration number would be one more function called in that loop and one more column in
`table4_uncertainty_metrics.csv`. It has to be computed on the raw uncertainty column rather than
the calibrated one, which is what the figure script already selects at lines 838-860.

**The binning choice, and why it is contested.** Regression has no classes, so the bins have to be
cut on something the model produced, and all three implementations cut them on the model's own
predicted uncertainty. That has two consequences. No two models and no two noise levels share bin
edges, so the values are not on a common footing. And the comparison inside a bin is a predicted
standard deviation against a mean absolute error, which for a perfectly calibrated Gaussian differ
by a factor of $\sqrt{2/\pi} = 0.798$ — so a perfect model scores about a fifth of its own mean
predicted standard deviation rather than zero, and a model with wider intervals scores worse for
that reason alone. Comparing predicted variance against mean squared error removes that factor.

**Your options, with what each costs.**

1. Leave it out. Cost is zero runs. The paper loses the formula at `paper.tex:234-238` and the ECE
   column at `paper.tex:503-526`, which cannot be regenerated anyway. `RERUN_PLAN.md:25312` records
   that none of the nineteen reference papers reports an expected calibration error in running text.
2. Compute the miscalibration area instead, from the calibration curve the two coverage points
   already sit on. Cost is one function in the figure script, one column, one re-run of
   `sbatch run_figures_v2.sh`, and no cluster experiment. `RERUN_PLAN.md` M6 already flags it as the
   number a reviewer in this area asks for first, citing Hirschfeld 2020, Scalia 2020 and Yang and
   Li 2023.
3. Bring ECE back in the decile form already on disk. Same cost as option 2, plus reversing the
   2026-08-19 instruction, and the number carries the two problems above unless you also change the
   comparison to variance against squared error.

You are right that seeing it costs little. Option 2 gets you the same look at calibration for the
same price, and the resulting number is comparable across models, which the decile ECE is not.

## Q3. The four statistics, rewritten

They are computed in `scripts/uncertainty_stats.py`: `q4_plain_correlation` at line 1059,
`q4_error_ratio` at line 1103, `q5_mean_uncertainty` at line 1543 and `q7_group_correlated_error` at
line 1741. Every one calls `assert_single_cell` first, so each value comes from one dataset, model,
representation, noise condition and noise level.

**Where it belongs: Performance metrics, where it already is.** That subsection defines $R^2$,
coverage and AUC$_{norm}$, so keeping these four beside them means every reported number is defined
in one place. Uncertainty quantification then holds only the split and the out-of-fold design.

Replacement text:

```latex
Four statistics are computed on the out-of-fold training molecules, each within one combination of
dataset, model, representation, noise condition and noise level. For molecule $i$, let $\epsilon_i$
be the noise injected into its label, $u_i$ the predicted uncertainty, $y_i$ the clean label and
$\hat{y}_i$ the prediction.

The first is $\rho(u, |\epsilon|)$, the Spearman correlation between predicted uncertainty and the
size of the injected noise. It is a check on the out-of-fold procedure rather than a result, since a
model that had seen its own corrupted label would score highly on it.

The second asks whether the uncertainty adds anything to the error. Write $e_i = |y_i + \epsilon_i -
\hat{y}_i|$ for the error against the corrupted label and $r_i = e_i / u_i$ for that error divided by
the predicted uncertainty. We report $\rho(r, |\epsilon|) - \rho(e, |\epsilon|)$, which is zero when
dividing by the uncertainty ranks the corrupted labels no better than the error alone does.

The third is the mean of $u$ over the molecules of one configuration, in the label's own units,
reported against the noise level. The fourth groups the signed error $\hat{y}_i - y_i$ by Murcko
scaffold and reports $\mathrm{Var}(\bar{e}_g) / \mathrm{Var}(e)$, where $\bar{e}_g$ is the mean
signed error of scaffold family $g$. A value near zero means the error scatters within a family,
and a value near one means a whole family is wrong as a block.

A statistic may read one half of the split rather than the total uncertainty. Each such statistic is
reported beside the statement of whether that half varies per molecule.
```

Two things the old wording did not say and this one does. The second statistic's error is taken
against the corrupted label, which is why it tracks the noise at all. The fourth statistic uses the
signed error against the clean label, and a signed error is what makes a shared family offset
visible.

## Q4. The calibration paragraph: a design choice, not a gap

Nothing is missing and nothing has to run. The old paragraph read as a confession because it
described the multiplier before saying why it is not read.

On QM9 a temperature multiplier is fitted per fit, bounded to $[0.1, 10.0]$
(`scripts/utils.py:620-634`), on half of the validation split
(`models/model_defaults.py:703-704`). It is refitted at every noise level. That makes calibrated
coverage nominal at each level by construction, and it flattens the very thing the paper asks
about. `models/model_defaults.py:698-712` records the measurement across twelve combinations of
model and noise level: NGBoost's coverage at one standard deviation runs 0.546, 0.876 and 0.978 on
the raw uncertainty, a span of 0.432 across three noise levels, against 0.637, 0.658 and 0.686
calibrated, a span of 0.049. For the QRF the raw span is 0.157 and the calibrated span is 0.006. I
read those numbers in that file this session and did not re-run them.

The figure script therefore reads the raw column everywhere, which it selects by name at
`scripts/generate_paper_figures_v2.py:838-860`. The assay runner fits no multiplier at all: grep for
`temperature` and `calibrat` in `KIRBy/tests/alternative_data_noise_robustness.py` returns zero
matches this session.

So the one sentence in the draft is the whole story, and it now sits at the end of the conditions
paragraph. If you want a calibration number that is not nominal by construction, that is option 2
under Q2, which is a re-run of the figure job and no cluster time.

## Numbers in this rewrite, and where each was read

- **seven models, three representations, 21 pairs** — `uncertainty_pairs.json`, seven entries under
  `models` and three under `representations`; `model_representations` says every model runs on all
  three.
- **which component each model emits, and whether it varies per molecule** —
  `scripts/uncertainty_decomposition.py:94-180`.
- **the four reported splits** — `uncertainty_pairs.json`, `decomposition.models`.
- **five inner folds, and three of five for NGBoost on QM9** —
  `slurm_scripts_qm9_rerun/generate_scripts.py:1386` (`--oof-folds` default 5) and `:226`
  (`OOF_FOLDS_SCORED = {'ngboost': 3}`).
- **T = 100 stochastic forward passes** — `models/model_defaults.py:383`, `'mc_passes': 100`. Note
  that the earlier draft cited line 364 for this; it is at 383 today.
- **temperature bounds and the calibration carve-out** — `scripts/utils.py:620-634`,
  `models/model_defaults.py:703-704`.
- **the coverage spans, raw against calibrated** — `models/model_defaults.py:698-712`, read as a
  recorded measurement, not re-run.
- **the three surviving ECE implementations** — `scripts/generate_figures.py:375-388`,
  `scripts/uncertainty_analysis.py:130-160`, `NoiseInject/noiseInject/uncertainty.py:177-201`.
- **$\sqrt{2/\pi} = 0.798$** — arithmetic, not a file.

## Word count

The LaTeX subsection is 431 words, from about 715.

---

# M6. Performance metrics — replacement text

*Replaces `paper.tex` 293-326.* 840 words of prose, against 880 in the version you rejected and 675
in the text now in `paper.tex`. Median sentence 17 words across 48 sentences, longest sentence 27
words, shortest 6 words. Nothing that was defined in either version is dropped.

```latex
\subsection{Performance metrics}

Predictive accuracy was scored on held-out molecules: the test split on QM9, and the held-out fold
of the five-fold scaffold cross-validation on the three assay datasets. We used the coefficient of
determination ($R^2$, higher is better) and the correlation between predicted and measured values,
Pearson's on QM9 and Spearman's on the assay datasets.

For each dataset, model, representation and noise condition we recorded $R^2(\tau)$ at each of the
seven noise levels and normalised it by the clean-label value, giving $R^2(\tau)/R^2(0)$. Our
robustness metric is the normalised area under this retention curve,
\begin{equation}
\text{AUC}_{\text{norm}}
  = \frac{1}{\tau_{\max} - \tau_{\min}}
    \int_{\tau_{\min}}^{\tau_{\max}} \frac{R^2(\tau)}{R^2(0)}\, d\tau ,
\label{eq:auc_norm}
\end{equation}
evaluated by the trapezoidal rule. For censoring the seven levels are instead fractions of labels
clipped, from $0$ to $0.50$, and AUC$_\text{norm}$ is normalised over that span. Despite the name,
AUC$_\text{norm}$ is not an area under a receiver operating characteristic curve. An
AUC$_\text{norm}$ near 1 means accuracy was almost fully retained as the noise rose, and higher is
better. An AUC$_\text{norm}$ near 0 means accuracy was gone by the highest noise level.
AUC$_\text{norm}$ is unbounded above and nothing is clipped, so a model predicting better with noise
added scores above 1. Replicates whose clean-label $R^2$ fell below $0.3$ were excluded (Additional
file~5), since $R^2(\tau)/R^2(0)$ becomes unstable as its denominator approaches zero. On QM9 one
AUC$_\text{norm}$ is computed per replicate, and the median over the ten replicates is reported. On
the assay datasets one is computed per scaffold fold, and the median over the five folds is
reported.

Seven statistics describe the predicted uncertainty. The first six are computed on the out-of-fold
training molecules, each within one dataset, model, representation, noise condition, noise level and
fold. The first is Spearman's $\rho$ between predicted uncertainty $u_i$ and the size of the noise
injected into that molecule's label. The second is Spearman's $\rho$ between $u_i$ and the absolute
error against the clean label, $|y_i - \hat{y}_i|$. The third divides that error by $u_i$ and checks
whether the ratio ranks the corrupted labels better than the error alone. We report the change in
$\rho$. We also report the change in the probability that a molecule from the tenth with the largest
injected noise outranks one outside that tenth. The fourth is the spread of that change over 200
permutations of the injected noise. The fifth is the mean predicted uncertainty in each of the
groups above, read against noise level. That mean is one number per group, not one per molecule. The
sixth is the share of out-of-fold error variance that lies between Murcko scaffold groups rather
than within them. The seventh is the fraction of held-out molecules falling within one and within
two predicted standard deviations, against Gaussian targets of $68\%$ and $95\%$.

We conducted a separate two-way analysis of variance (ANOVA) for each noise condition. The metric
$y$ entering it is either $R^2$ at a fixed noise level or AUC$_\text{norm}$, and the model fitted
is
\begin{equation}
y_{ijr} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijr} .
\end{equation}
Here $\alpha_i$ is the effect of model architecture $i$ and $\beta_j$ the effect of molecular
representation $j$. The term $(\alpha\beta)_{ij}$ is the interaction between the two, and
$\epsilon_{ijr}$ is the residual for replicate $r$. The proportion of variance explained by each
factor is the effect size $\eta^2$,
\begin{equation}
\eta^2_{\text{factor}} = \frac{SS_{\text{factor}}}{SS_{\text{total}}} ,
\end{equation}
where $SS$ denotes the sum of squares. The sums of squares are Type I (sequential), entered in the
order model, representation, interaction. Both factors are treated as fixed effects, and the ten
replicates in each model-by-representation combination provide the error term. We fitted the
decomposition on QM9 alone, because the five scaffold folds on the assay datasets partition one
dataset rather than repeating it. A noise condition with fewer than five models is not decomposed.
Each decomposition also carries a range for $\eta^2$, obtained by refitting it with one replicate
left out. Six models are held out of the decomposition and of every cross-model comparison. Five of
them add a per-molecule noise term to a base model that is compared in its own right, and so train
under a different likelihood. The sixth is the Tanimoto-kernel Gaussian process, which needs binary
vectors and ran on ECFP4 alone.

ANOVA relies on the assumption that observations within and across groups are independent. Some
model pairs produced similar predictions, which are not independent. Treating them as independent
raises the residual degrees of freedom while the residual sum of squares barely moves. Redundancy
was therefore checked with pairwise Spearman rank correlations, between model AUC$_\text{norm}$
profiles and between representation profiles. Intraclass correlation coefficients ICC(1,1) were also
computed for all model pairs, and both sets of values are tabulated in Additional files~2--4.
ICC(1,1) measures the share of total variance that lies between profiles rather than within them. A
value near 1.0 means two models produced almost the same AUC$_\text{norm}$ profile.

Agreement between model rankings across noise conditions was scored on QM9 with Kendall's
coefficient of concordance ($W$), within one representation. Where a model has both a deterministic
and a probabilistic form, the two forms were compared on AUC$_\text{norm}$. That comparison used a
two-sided Wilcoxon signed-rank test at $\alpha = 0.05$, paired on the replicate. It was run
separately for each representation and noise condition.
```

---

## For the author

### Six things in your text that the code contradicts

Each was read in a file this session, with the file and line given.

**1. The noise levels.** Your text says `$\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$` and "eleven
noise levels", then lists seven. The level grid is the seven levels 0, 0.2, 0.3, 0.5, 0.75, 1.0 and 1.5
(`slurm_scripts_qm9_rerun/generate_scripts.py:121`, `DOSE_LEVELS`). Artificial noise injection
already prints that list at `paper.tex:238`, so I cut the second copy and say "the seven noise
levels". That removes the eleven-against-seven clash in one move.

**2. The symbol.** Artificial noise injection uses $\tau$ for the noise level (`paper.tex:243`).
Performance metrics used $\sigma$ in the prose and in the integral, and $\tau$ in the list. It is
$\tau$ throughout now.

**3. "AUC$_{norm}$ falls on $[0,1]$" is false, and your own previous paragraph says so.**
`figlib_metrics.py:119-126` counts the replicate values above `AUC_NORM_IMPLAUSIBLE_HIGH = 1.05`
(`figlib_config.py:180`), prints how many, and does not clip them. I kept the unboundedness and
deleted the interval. Losing "faster degradation" in the process is why the "near 0" sentence is
there.

**4. The divisor of the integral.** `retention_auc_norm` divides by `sigma.max() - sigma.min()`
(`figlib_metrics.py:63-66`), not by $\tau_{\max}$. The two agree only because the lowest level is 0,
and they stop agreeing for censoring, which you already give its own span.

**5. The $0.3$ cut-off is applied per replicate, not per model and representation.** The grouping
key ends in `replicate` (`figlib_metrics.py:80`) and the baseline is read and compared inside that
group (`:94-103`). So one model-and-representation median can run over fewer than ten replicates.
The sentence now says "Replicates whose clean-label $R^2$ fell below 0.3".

**6. Additional file 5 does not say what your sentence says it says.** Its caption reads "baseline
R$^2 \leq 0.6$" and its cells count "noise strategies (out of 6)" (`additional_files.tex:658-660`).
The Methods say 0.3, and there are seven noise conditions. Its columns are Mol2Vec, MHGGNN, SMILES
and Randomized SMILES, and three of those four are not in the study. It is output from the retired
metric and has to be regenerated before that pointer is true.

### The ANOVA exclusions are the one real decision, and it is yours

Your text says three things were excluded on redundancy: quantile regression forests at
$\rho > 0.99$ with the random forest, the Gaussian process because different kernels are used on
different representations, and Sort & Slice at $\rho > 0.90$ with ECFP4. **None of the three is
excluded by the code that writes the current numbers.**

`figlib_config.py:558-574` holds one exclusion set, `VARIANT_MODELS`, and it has six entries:
`dnn_bnn_full_mve`, `mlp_bnn_full_mve`, `dnn_vbll_hetero`, `mlp_vbll_hetero`, `het_gp_rbf` and
`gauche`. The quantile forest is in. The radial-basis Gaussian process is in. No representation is
dropped at all, and the comment at `:549-552` says why: "a representation this study is measuring is
not dropped on a correlation. That one is the author's to reinstate if it was meant."

The older script still does it your way. `generate_paper_figures_v2.py:130-158` excludes `qrf`,
`gauche`, `gauche_rbf`, then `dnn_bnn_full_mve`, `mlp_bnn_full_mve`, `dnn_vbll_hetero`,
`mlp_vbll_hetero` and `het_gp_rbf`, and then everything in `GLOBAL_MODELS_EXCLUDE`. From the
representation factor it excludes `sns`, `morgan` and both SMILES representations, at
`ANOVA_REPS_EXCLUDE`.

So two scripts disagree and your paragraph is true of one of them. Three ways out:

- **Reinstate the exclusions.** Add `qrf`, `gp_rbf` and `sns` to `ANOVA_MODELS_EXCLUDE`, add a
  representation exclusion set to `figlib_config.py`, and re-run the analysis pass. Your paragraph
  then stands, once the thresholds below are re-read off AUC$_{norm}$. Costs one figure re-run and
  nothing from the cluster.
- **Drop them,** which is what the code does now. The paragraph becomes the redundancy check alone: the
  correlations and ICC(1,1) are computed and tabulated, nothing is removed on them, and the degrees
  of freedom are not adjusted. That is the version written above.
- **Reinstate the model exclusions and not the representation one.** Sort & Slice is one of the six
  representations being measured, and removing it from the factor whose size is the paper's headline
  is the hardest of the three to defend.

I wrote the second because it is what the pipeline does. The sentence for the first is one line and
I will write it on your word.

### The three redundancy numbers all come from the retired metric

Whichever way that goes, these need re-reading before any of them goes in a sentence.

- Additional file 2 is titled "Pairwise Spearman rank correlations between model **noise degradation
  slope (NDS)** profiles" (`additional_files.tex:468`), and Additional file 3 and Additional file 4
  say the same. NDS is what AUC$_{norm}$ replaced. The Methods describe the redundancy check as being on
  AUC$_{norm}$ profiles.
- Your $\rho > 0.90$ for Sort & Slice against ECFP4 is not in that table. Additional file 3 prints
  ECFP4 against SNS at $\rho = 0.992$, and on those profiles every representation pair sits at 0.89
  or above, PDV against ECFP4 included at 0.990 (`additional_files.tex:565-596`). A 0.90 threshold
  read off that table removes almost every representation, not one.
- Additional file 4 puts QRF against RF at ICC(1,1) = 0.810, not near 1.0
  (`additional_files.tex:618`). Additional file 2 has the same pair at $\rho = 0.995$, which does
  clear your $>0.99$. The $\rho$ claim survives regeneration in form; the ICC claim does not.

The paragraph above therefore says what the two statistics measure and points at Additional files 2
to 4, with no threshold in it, so it is true either way. The thresholds go back in once the tables
are regenerated on AUC$_{norm}$.

### `tab:regression_noise` is already fixed

The guide at line 368 records it as an undefined label pointed at from the metrics caption at
`paper.tex:307-308`. Grepping `paper.tex` this session, the string `regression_noise` appears
nowhere in the file, and the only four table labels defined are `tab:anova_decomposition`,
`tab:auc_ranking`, `tab:wilcoxon_bnn` and `tab:top_unc_noise`. You removed it when you rewrote the
subsection. Nothing replaces it and nothing needs to, because the condition names and the level
grid are prose in Artificial noise injection. That open thread can be closed.

### Two smaller repairs

- "observations both within and across groups are independence" is now "are independent".
- "for each noise **strategy**" is now "for each noise **condition**". The code groups on
  `condition` (`figlib_metrics.py:285`), and "strategy" is a word this study retired.

### Getting from 840 words to 716

I did not cut below the point where a definition disappears, so it is 840 rather than the 700 you
may want. Six passages take it to 716, and each costs something specific:

| passage | words | what goes with it |
|---|---|---|
| "Despite the name, AUC$_\text{norm}$ is not an area under a receiver operating characteristic curve." | 14 | the only warning that the name collides with ROC |
| "An AUC$_\text{norm}$ near 0 means accuracy was gone by the highest noise level." | 13 | the low end of the metric's range |
| "We report the change in $\rho$. We also report the change in the probability that a molecule from the tenth with the largest injected noise outranks one outside that tenth." | 30 | what the third uncertainty statistic reports |
| "That mean is one number per group, not one per molecule." | 11 | the warning against reading the fifth statistic per molecule |
| "Some model pairs produced similar predictions, which are not independent. Treating them as independent raises the residual degrees of freedom while the residual sum of squares barely moves." | 28 | the reason the redundancy check exists |
| "ICC(1,1) measures the share of total variance that lies between profiles rather than within them. A value near 1.0 means two models produced almost the same AUC$_\text{norm}$ profile." | 28 | ICC(1,1) is then named and never defined |

### What I could not check

- Whether the seven uncertainty statistics above are the seven you mean. The live driver
  (`figlib_uncertainty.py:565-570`) computes the support flags, the two question-4 correlations with
  their permutation band, the mean-uncertainty slopes, the clean-label error ranking and the
  scaffold-group error share. Coverage comes from a different function, `calculate_coverage` in
  `generate_paper_figures_v2.py:2668`. I counted coverage as the seventh and left the support flags
  out, because they are declarations rather than statistics and M5 already carries them. Counting
  the flags instead moves coverage up to the accuracy paragraph and the count still reads seven.
- `confound_controlled_effect` is in `uncertainty_stats.py`, but nothing in `figlib_uncertainty.py`
  calls it, so it is in none of the figures the current driver writes. It is not described above.
  If it is meant to be reported it needs a call site before it needs a sentence.
- Whether root mean squared error and mean absolute error should be named. They are recorded at
  every noise level (`scripts/utils.py:208-210`) and no figure module reads them. Your version drops
  them, the rejected version named them, and I followed yours. Restoring them is one sentence.
