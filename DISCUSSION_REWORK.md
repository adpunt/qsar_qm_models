# NoiseInject — the new takeaways

Working document. Nothing applied to `paper.tex` or any script.
Last updated 2026-08-20.

Twelve candidate takeaways survived adversarial checking; two were refuted. They collapse into the
**five** below. Everything else — the audit of what's currently wrong, the figure list, the re-run
spec — is in the appendices, and you shouldn't need to read them to have this conversation.

---

## The five

### 1. You have six noise strategies but roughly three mechanisms

Most of what looks like "different kinds of noise" is the same Gaussian noise at different strengths.
The strategies differ in how *much* they perturb a label at a given σ, by about a factor of four, and
that difference alone predicts almost everything about how much damage they do.

On QM9, the amount injected per unit σ is roughly: outlier 0.49, heteroscedastic 0.67, quantile 0.90,
Gaussian 1.00, value-proportional 1.69, threshold 2.00. The correlation between how much a strategy
injects and how much performance it destroys is essentially perfect — Spearman −0.94 to −1.00,
depending on the model, and it holds for each of the eleven models separately.

Worse for the "six mechanisms" framing: three of them aren't really mechanisms on your data.
**Threshold noise is exactly plain Gaussian noise at double strength on QM9** — the rule only fires on
molecules with a label above 1.0, and every QM9 gap in eV is above 1.0, so every molecule gets the same
multiplier. The same is true on hERG. Value-proportional and heteroscedastic vary their per-molecule
noise by under 10%, so they're near-homoscedastic too. **Only outlier and quantile genuinely corrupt a
subset of molecules**, and when you rescale σ to equalise the injected amount, the other curves collapse
onto the Gaussian one while quantile's does not.

**Why this matters most:** it explains your headline ANOVA result. Outlier and heteroscedastic have
enormous unexplained residuals (83.6% and 77.4%) not because model architecture stops mattering under
those noise types, but because those two inject the *least* noise and therefore separate the models
least. The between-model spread varies four-fold with dose while the run-to-run noise barely moves. The
residual is a sensitivity floor, not a finding about noise.

⚠ **One correction I owe you.** I told you earlier that QM9 labels are standardised before noise is
added. That's wrong — I've now read `rust/src/main.rs`, and noise goes onto the raw label first,
standardisation happens after. Your loader is PyTorch Geometric's QM9, whose gap is in eV. That's what
makes every label exceed the threshold cut-point, and it's why threshold is the harshest strategy rather
than the mildest.

---

### 2. Your robustness metric picks the wrong model

`auc_norm` measures the *shape* of a degradation curve, not how much a model actually predicts. Because
each model is scored against its own clean-data performance, a model with a weak baseline scores well by
having less to lose.

The consequence is concrete. On QM9, LightGBM is the most accurate model at every noise level tested,
and NGBoost is 11th of 13 on clean data and 5th at the highest noise — yet `auc_norm` crowns NGBoost in
five of the six strategies and puts LightGBM third. On the experimental datasets the two metrics name
the same best model in only a small minority of cells, and **LightGBM wins on delivered accuracy 22
times out of 72 while never once winning on retention**.

The clearest single case: a Gaussian process on hERG scores `auc_norm` 0.97 with a clean-data R² of
−0.16. It retains almost all of nothing.

**What to do:** select and rank models on R² at σ = 0.6. Keep `auc_norm` as a secondary
curve-shape descriptor, and never print it without clean-data R² beside it. If you want one robustness
number, use the drop — clean R² minus R² at σ = 0.6 — which is just as independent of baseline, is in R²
units you can subtract by eye, and has no denominator to explode.

**Caveat from the verifiers:** on Caco-2 the ordering partly reverses, so this is "usually and
substantially" rather than "always".

---

### 3. Making models Bayesian doesn't buy robustness — it buys a better-looking retention score

Four of the five probabilistic transformations you tested lose delivered accuracy, on every noise
strategy, while the retention metric records some of them as improvements. On the experimental data,
converting a network to a Bayesian one lowers both clean accuracy and accuracy under noise in essentially
every paired comparison. The one genuine win worth keeping is NN-β to full BNN.

Your own output already breaks the paper's version of this claim: the VBLL-α comparison is **not
significant** (p = 0.25), so "both transformations improved both networks" is three of four, not four of
four. Every number in that table also differs from the current output.

---

### 4. Two completely different failures are being counted as "poor noise robustness"

Neural networks and their Bayesian variants **fail before any noise is added at all** — 7% to 18% of
runs collapse at σ = 0, entirely on the descriptor and pretrained-graph inputs, never on the binary
fingerprints. Their failure rate is essentially flat as noise rises. That's a numerical conditioning
problem, not noise sensitivity.

Boosted trees do the opposite: perfectly stable at zero noise, then degrading steadily to 29% failure at
the highest noise level, sliding smoothly into negative R². That *is* a noise result, and it's the
cleanest architectural finding in the data.

There's also a free replication experiment nobody noticed: at σ = 0 all six strategies train on
identical data, so any spread across them is pure run-to-run variation. Trees, kernels and the GP give
**exactly zero** spread. A Bayesian network refit on identical data moves its own R² by about 0.18 —
roughly three times the entire between-model spread your QM9 ranking is built on. **Without repeat
training seeds, no neural comparison in this paper is interpretable.**

---

### 5. Representation decides accuracy; model decides degradation

Which representation you pick largely determines how accurate the model is, and for neural networks
whether it trains at all. Which *model* you pick determines how gracefully it degrades. Holding one
representation constant is safe for every robustness statement — the model ranking under different
representations agrees at 0.77 to 0.94, with no negative cell in 108 comparisons — but it is **not** safe
for clean-data accuracy claims, where 12 of those 108 comparisons disagree, one at −0.71.

**Caveat:** the "model matters more than representation" ratio is weakest on LogD, your highest-signal
dataset, where the two nearly rival each other.

---

## The sixth, weaker one, and where uncertainty landed

**Does knowing your noise type change which model to pick?** Mostly no — the gain from switching to the
per-strategy optimum is a median of about 0.01 R², and it beats run-to-run noise in 1 of 18 cases. But
one verifier refuted this, because the agreement is much weaker on your low-signal datasets. Treat it as
promising, not established. If it holds it's a genuinely useful practical result.

**Uncertainty.** One clean finding survives: **per-sample uncertainty identifies which labels were
corrupted only when the corruption is concentrated in a few molecules.** Under outlier and quantile
noise there's real signal; under Gaussian, threshold, value-proportional and heteroscedastic there is
none in any model tested. What uncertainty *does* do reliably is rank a model's own errors, weakly but
almost universally — and QRF is best at that, which the paper currently dismisses.

Three things to cut, from the literature search:
- **Drop the aleatoric/epistemic split.** It isn't identifiable, only your GP and VBLL models have the
  ingredients for it, and — importantly — both parts rising with added noise is the mathematically
  *correct* behaviour, not the anomaly the paper presents it as.
- **Drop the conformal arm.** The 16 files in `results/calibration_grid/` contain no intervals, no
  coverage and no widths. There is nothing in them to analyse.
- **Add one metric only: miscalibration area.** It's the regression version of ECE, it's what Chemprop
  and UNIQUE use, and coverage at 1σ and 2σ is literally two points on the curve it integrates. Also
  rename "mean predicted uncertainty" to sharpness and print it beside coverage.

⚠ **Before any uncertainty number goes in the paper**, two things need settling on the server. The
uncertainty files don't record how much noise each molecule got — the code reconstructs it by fitting a
line and calling the leftovers "noise", which gives non-zero answers even at zero noise, and which
systematically under-recovers for the strategies whose noise depends on the label value. And it's
unclear whether these are training or test molecules; if they're test molecules with clean labels, then
there are no corrupted labels to find and the question can't be asked on this data at all.

---

## What I need from you

1. **Which of the five do you want to build the paper around?** I'd argue 1, 2 and 4 are the strongest
   and most novel. 3 and 5 are corrections to what's already there.
2. **Does the QM9-versus-experimental disagreement become a headline or a limitation?** Model rankings
   don't transfer between them, which currently reads as a failure of your third research question but
   could instead be the finding: large clean computed data and small noisy assay data behave differently,
   and model choice matters far more on the latter.
3. **Do we add repeat training seeds for the neural models?** Without them, point 4 stands but the
   Bayesian comparisons can't be interpreted.

---

# Appendix A — what's wrong with the current Results

Short version. Every item verified against the figure script's own output.

| # | Problem |
|---|---|
| A1 | The ANOVA table's robustness half doesn't match a single cell of the current output. Heteroscedastic: 37.0 → 14.0 for model, 41.0 → 77.4 for residual. The prose above the table already quotes the correct numbers, so the paragraph contradicts its own table. |
| A2 | Every value in the Wilcoxon table is wrong and one significance verdict flips. The test also silently pools four representations and six strategies while sitting in a section about one representation. |
| A3 | The retired metric NDS appears 19 times, including in the Conclusion, where it's defined as *the* robustness metric. ECE is defined and tabulated but has been deleted from the code, so those cells become unfillable. |
| A4 | PDV is described as both the most and the least noise-robust representation, in the same section. |
| A5 | The exclusion threshold is stated as R² < 0.3 in four places and 0.6 in two. The live gate is 0.3. |
| A6 | Additional file 8 describes an artefact that was never computed. Additional file 10's residual of 0.0 is a saturation artefact from averaging folds before integrating. |
| A7 | "XGBoost and BNN variants suffer the most" cites a figure whose data contains no BNN rows. "NGBoost ranks first on external datasets" — it's third. |
| A8 | The uncertainty figure's second panel has already been deleted from the script, so regenerating removes the only evidence for that claim. |
| A9 | The Methods claim that σ is a consistent difficulty scale across strategies is false (see takeaway 1). The value-proportional formula in the paper uses 0.1; the code uses 0.05. |
| A10 | Methods say uncertainty experiments were run once. Most were run ten times. |
| A11 | Additional file 12 claims representation-specific SVM kernels; the code uses RBF throughout, as do three other places in the Methods. The same error is in the Methods text. |

# Appendix B — figures and tables

Currently 8 figures + 6 tables. Comparable J Cheminform papers carry fewer: Kolmar & Grulke (same
journal, same topic, bigger grid) use 6 figures of which only 4 carry results. **Target 6 figures + 4
tables.**

Redundancies to resolve: the ANOVA figure and table show identical numbers; the overview figure's second
panel and the ranking table show identical data; there are three heatmaps carrying the same message; and
the combined validation figure duplicates the overview figure with strategies averaged away.

**Delete:** the combined validation figure — it's the only float that averages across noise strategies,
and it shows an incomplete model roster.

**Rebuild:** the ranking table, so each cell carries clean R², R² at σ = 0.6, and `auc_norm` together —
the convention Kolmar & Grulke use of printing a ratio's components beside it. The uncertainty table,
with within-σ correlations replacing pooled ones.

**New, and the one I'd fight for:** six panels, one per strategy, clean R² on the horizontal axis and R²
at σ = 0.6 on the vertical, with the diagonal drawn. Distance below the diagonal is the drop. It makes
takeaway 2 self-evident without any argument.

# Appendix C — what has to be re-run

**QM9:** one change to the figure script unblocks most of this — `calculate_robustness` currently emits
only `auc_norm` and baseline, so **R² at σ = 0.6 does not exist anywhere for QM9**. Add it, plus per-σ
curves, plus a small no-training-required table of how much each strategy actually perturbs labels.

**Validation:** a clean re-run covering all 13 models on all 4 representations, the GP on more than one
representation, repeat seeds for the neural models, per-fold results kept rather than averaged before
integrating, and per-sample uncertainty saved for all strategies with the clean and noisy label as
separate columns so nothing has to be reconstructed.

# Appendix D — positioning

Kolmar & Grulke 2021 is in your target journal, same experiment, one noise strategy, no representation
axis — and their robustness metric has the same ratio problem as `auc_norm`. They handle it well and you
should copy them: print the components beside the ratio, refuse to compute it where the denominator is
degenerate, and state the instability in the same paragraph as the numbers. Cortes-Ciriano 2015 concluded
that algorithm ranking *does* change with noise level, which collides with the weaker sixth takeaway
above — that has to be argued against them by name rather than asserted.
