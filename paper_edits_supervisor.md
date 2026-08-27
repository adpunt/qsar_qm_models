# Supervisor edits — proposed responses

These are drafted to keep the existing voice and avoid new jargon. Bracketed
notes are mine, not for the paper.

---

## 1. GP kernel situation

The supervisor's confusion ("I thought we were using a RBF kernel for GP/PDV?")
comes from the fact that the current paragraph buries the answer in the middle
of a long block. The paper *does* say it: Tanimoto for binary fingerprints, RBF
for PDV. But because the paragraph opens with "we used the Tanimoto kernel,"
that's what sticks.

Replacement for the GP paragraph (last paragraph before NN-α / NN-β
discussion). Lead with the two-kernel setup, then explain why:

> We implemented GP models using the \texttt{Gauche} framework
> \citep{gauche}, which provides kernels optimised for molecular
> representations. Because no single kernel is appropriate for both binary
> fingerprints and continuous descriptors, we used two GP variants throughout
> this study. For the binary fingerprint representations (ECFP4, SNS), we used
> the Tanimoto kernel \citep{Ralaivola2005, moss2020, gauche}, which measures
> similarity between binary feature vectors. For continuous-valued
> representations (PDV, mol2vec, MHG-GNN), the Tanimoto kernel is
> ill-defined, so we used a radial basis function (RBF) kernel implemented via
> \texttt{GPyTorch} \citep{gpytorch}; we refer to this variant as GP~(RBF) in
> PDV-specific analyses. To avoid confounding kernel choice with representation
> in the cross-representation ANOVA, the GP is excluded from that comparison
> and reported separately in the PDV case study (Section~\ref{...}).

### 1b. SVM kernel — paper is wrong, fix while you're here

Code-checked against `models/models.py:1427-1469` and
`KIRBy/tests/alternative_data_noise_robustness.py:630-635`.

- **QM9 SVM** (`train_svm_model`): always `kernel='rbf'`, `C=1.0`,
  `gamma='scale'`. No representation switch, no Tanimoto branch. The
  Models-section line in the paper is correct *for QM9*.
- **Validation-dataset SVM** (KIRBy): hyperparameters were tuned per
  representation, giving:
  - `ECFP4` → `kernel='rbf'`
  - `SNS` → `kernel='poly'` (degree 2)
  - `PDV` → `kernel='poly'` (degree 2)
  - default → `rbf`

  Tanimoto is **not used for SVM anywhere in the codebase.**

The current Dataset-section sentence — "For binary vector-based
representations, a Tanimoto kernel was used for SVM, while an RBF kernel was
used for continuous-valued vectors" — is therefore incorrect on both
counts: binary SNS uses poly (not Tanimoto), continuous PDV uses poly (not
RBF), and only ECFP4 uses RBF. Replace that sentence with:

> For the validation-dataset SVM, kernel and other hyperparameters were
> selected per representation by Bayesian optimisation
> (Supplementary Table~S1); the chosen kernels were RBF for ECFP4 and
> polynomial (degree 2) for SNS and PDV. For QM9, the SVM used a fixed RBF
> kernel ($C = 1.0$, $\gamma = \texttt{scale}$) as described in the
> Models subsection.

Also remove the duplicate / partially-redundant "A radial basis function
(RBF) with $C = 1.0$ and $\gamma = \texttt{scale}$ was used as the kernel"
line from the Models paragraph if you keep the QM9 detail above, or leave
that line and trim the Dataset version — pick one home for it.

---

## 2. Summary table of metrics

New table to add at the end of the Performance Metrics subsection (or
immediately after the introductory sentence). Caption can be tightened.

```latex
\begin{table}[htbp]
\centering
\small
\begin{tabular}{p{2.5cm} p{4.5cm} p{6cm}}
\toprule
\textbf{Metric} & \textbf{Definition} & \textbf{Used to assess} \\
\midrule
RMSE
  & $\sqrt{\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2}$
  & Predictive accuracy (lower is better) \\
$R^2$
  & $1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$
  & Variance explained by the model; comparable across datasets and noise levels \\
NDS
  & $dR^2 / d\sigma$ (slope of $R^2$ vs $\sigma$)
  & Robustness to label noise; values near zero indicate stability, more negative values indicate sensitivity \\
Wilcoxon signed-rank
  & Paired non-parametric test, $\alpha=0.05$ (two-sided)
  & Pairwise comparison of probabilistic vs deterministic counterparts (e.g.\ NN-$\alpha$ vs BNN-$\alpha$) \\
Kendall's $W$
  & Coefficient of concordance, $[0, 1]$
  & Stability of model rankings across noise strategies; $W > 0.7$ indicates strong agreement \\
ANOVA $\eta^2$
  & $SS_{\text{factor}} / SS_{\text{total}}$ (Type~I)
  & Variance attribution to model architecture vs molecular representation \\
ICC(1,1)
  & Intraclass correlation, $[0, 1]$
  & Profile-level redundancy between models prior to ANOVA \\
Spearman's $\rho$
  & Rank correlation, $[-1, 1]$
  & Calibration of predicted uncertainty against (i) absolute error and (ii) injected noise magnitude \\
Coverage
  & $\frac{1}{N}\sum_i \mathbb{1}[|y_i - \hat{y}_i| \le k\hat{u}_i]$
  & Fraction of test points whose true value lies within $\hat{y}\pm k\hat{u}$; targets 68\% at $k=1$, 95\% at $k=2$ \\
ECE
  & $\sum_b \frac{|B_b|}{N}\,|\bar{u}_b - \bar{e}_b|$
  & Calibration of predicted uncertainty: deviation between binned predicted uncertainty and mean error (lower is better) \\
\bottomrule
\end{tabular}
\caption{Summary of metrics used in this study, their definitions, and their
intended interpretation. $\hat{y}_i$ and $\hat{u}_i$ denote the predicted
value and predicted uncertainty for sample $i$; $\sigma$ is the noise scaling
factor (Table~\ref{tab:regression_noise}).}
\label{tab:metrics_summary}
\end{table}
```

---

## 3. Classification noise — brief mention

The supervisor wants this in the Noise Strategies subsection (it currently
opens with "The following strategies were implemented for regression…" with
nothing said about classification). Suggested addition immediately *before*
the regression table, replacing line 285's lead-in:

> The following strategies were implemented for regression
> (Table~\ref{tab:regression_noise}, Figure~\ref{fig:noise_strategies}). For
> classification tasks, label noise is injected by stochastic class flipping
> rather than additive perturbation: each label is flipped with a strategy-
> dependent probability $p$, where $p$ replaces $\sigma$ as the noise scaling
> factor. NoiseInject implements six classification strategies that mirror
> the regression set in spirit (uniform, class-imbalance, binary-asymmetric,
> instance, class-dependent, and confusion-directed flipping); these are
> documented in the NoiseInject framework and used in the supplementary
> classification benchmarks rather than in the QM9 regression experiments
> reported here.

[From `noiseInject/core.py`: the six classification strategies are uniform,
class_imbalance, binary_asymmetric, instance_noise, class_dependent,
confusion_directed. The interface uses `flip_probability` instead of `sigma`.]

---

## 4. "Clean and noisy" wording (the comment "That's between the clean and noisy data — should we say it explicitly?")

This refers to RMSE annotation in the noise-strategies figure caption. Make
the caption explicit:

> Effect of each noise injection strategy on the HOMO–LUMO gap label
> distribution at $\sigma = 0.5$. Blue outline: clean label distribution;
> coloured fill: distribution after noise injection. **The RMSE annotated in
> each panel is computed between the clean and the noisy labels for that
> strategy** (i.e.\ the magnitude of perturbation, not a model error).

---

## 5. Hetero last in the table

The supervisor's point: in the explanatory paragraph below the table, the
strategies are described in the order Gaussian → outlier → quantile →
threshold → value-proportional → heteroscedastic, with Hetero last. The
table currently has Hetero in row 4. Reorder the table rows to match:

```latex
\begin{tabular}{lll}
\toprule
\textbf{Noise Type} & \textbf{Noise Scaling} & \textbf{Simulated Real-World Source} \\
\midrule
Gaussian   & $\sigma$ (fixed)              & Random measurement error \\
Outlier    & $3\sigma$ / $0.1\sigma$       & Transcription errors, batch effects \\
Quantile   & $2\sigma$ / $0.1\sigma$       & Systematic error in hard-to-predict domains \\
Threshold  & $2\sigma$ / $0.1\sigma$       & Regime-dependent assay errors \\
Valprop    & $\sigma(1 + 0.1|y|)$          & Percentage-based measurement uncertainty \\
Hetero     & $\sigma\sqrt{0.1 + 0.05|y|}$  & Heteroscedastic measurement precision \\
\bottomrule
\end{tabular}
```

---

## 6. Unifying the noise-strategy order across text, table, and figures

Current orderings:
- **Table**: Gaussian, Outlier, Quantile, Hetero, Threshold, Valprop
- **Text paragraph**: Gaussian, Outlier, Quantile, Threshold, Valprop, Hetero
- **Figure code** (`generate_paper_figures.py:2582` etc.):
  legacy, valprop, quantile, threshold, outlier, hetero
- **Figure code (one place at 3983)**: legacy, outlier, threshold, quantile, hetero, valprop

Recommendation — adopt **Gaussian, Outlier, Quantile, Threshold, Valprop,
Hetero** (matches the text paragraph and the supervisor's "Hetero last"
preference). To apply this:

- Reorder table rows as in §5 above.
- In `generate_paper_figures.py`, change every occurrence of
  `['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']`
  (lines 83, 572, 1467, 2582, 2681, 2742) and the variant at line 3983 to
  `['legacy', 'outlier', 'quantile', 'threshold', 'valprop', 'hetero']`.
  Define one constant `STRATEGY_ORDER` near the top and import it everywhere
  so this never drifts again.
- Re-run figure generation on the server.

[If you really don't want to change the figures, the cheapest option is to
reorder *only* the table (which the supervisor explicitly asked for) and the
text paragraph to match the figure order. But that flips Hetero away from
last, which is the opposite of what he wants. So I think you do need to
re-run.]

---

## 7. Why the clean (blue) outline differs across panels — bug + fix

This is a real plotting artefact, not different data. In
`generate_paper_figures.py:2543–2599`:

```python
np.random.seed(42)               # set once
y_clean = ...                    # same for every panel — good
for i, strategy in enumerate(strategies):
    y_noisy = apply_noise(y_clean, sigma, strategy)
    all_vals = np.concatenate([y_clean, y_noisy])
    bins = np.linspace(all_vals.min(), all_vals.max(), 51)   # ← per-panel bins
    ax.hist(y_clean, bins=bins, ...)                         # ← clean re-binned
```

Because `bins` are recomputed from `y_clean ∪ y_noisy`, and `y_noisy` has a
wider range under heavy-tailed strategies (outlier, hetero), the bin edges
shift between panels. Re-binning the same `y_clean` array against shifted
edges produces visibly different "clean" histograms in each panel, even
though the underlying data is identical. Reviewers would flag this.

**Fix:** compute one fixed bin grid before the loop, large enough to contain
all noisy distributions, and use it for both histograms in every panel.
Concretely, replace the per-panel `bins = np.linspace(...)` with something
like:

```python
# Pre-compute a single bin grid that accommodates all strategies.
# Run each strategy once on a copy of the RNG to get the worst-case range,
# then use that range for every panel's histogram.
all_y = [y_clean]
rng_state = np.random.get_state()
for strategy in strategies:
    np.random.set_state(rng_state)            # reproducible per strategy
    all_y.append(apply_noise(y_clean, sigma, strategy))
np.random.set_state(rng_state)                # restore for the real plotting loop
lo = min(arr.min() for arr in all_y)
hi = max(arr.max() for arr in all_y)
bins = np.linspace(lo, hi, 51)
```

Then inside the loop drop the `all_vals` / per-panel `bins` lines and reuse
`bins`. The clean outline will then be pixel-identical across all six panels,
and the coloured noisy outlines will still line up with the same reference.

[I can do this edit and regenerate the figure when you're ready — say the
word.]

---

## 8. BNN paragraph reorganisation (currently lines covering "This can be
done by approximations…" through "…early stopping on validation loss")

Current structure:
1. General BNN intro and how priors/posteriors work.
2. Generic taxonomy of approximations (MC dropout, VI, ensembles).
3. The two specific approaches we used (full-BNN, VBLL).
4. NN-α / NN-β architecture details (these are then *repeated* in the
   following paragraph — that's a bug independently of the supervisor's note).

Supervisor wants: lead with what we did, then give context. Drafted
replacement (also de-duplicates the architecture description):

> We used two standard feed-forward neural network architectures throughout
> this study, NN-$\alpha$ and NN-$\beta$, each available as a deterministic
> baseline and as two Bayesian variants. The NN-$\alpha$ architecture has two
> hidden layers of size [128, 64] with dropout ($p=0.2$) after each hidden
> layer; NN-$\beta$ uses two hidden layers of size [128, 128] with dropout
> applied before the output layer. Both architectures use ReLU activations,
> are implemented in \texttt{PyTorch} \citep{pytorchGeometric}, and are
> trained with the Adam optimiser using early stopping on validation loss.
>
> The Bayesian variants convert these networks into BNNs, which place priors
> on the weights and learn an approximate posterior distribution over them
> rather than a single point estimate, providing a principled source of
> predictive uncertainty. Exact posterior inference for deep networks is
> intractable, so practical BNNs rely on one of several approximations,
> including Monte Carlo dropout, variational inference, and ensembling
> \citep{gal2016}. We implemented two such approximations:
>
> \begin{itemize}
>   \item \textbf{Full BNN.} All linear layers are replaced with Bayesian
>   layers, with Gaussian priors $\mathcal{N}(0, 0.1^2)$ on the weights and
>   posteriors learned by variational inference. This is the most expressive
>   variant but also the most computationally costly.
>   \item \textbf{Variational Bayesian Last Layer (VBLL).} Only the final
>   layer is made Bayesian. We maintain a mean-field variational posterior
>   $q(\mathbf{W}) = \mathcal{N}(\boldsymbol{\mu}_W, \text{diag}(\boldsymbol{\sigma}_W^2))$
>   on the last-layer weights, trained by maximising the evidence lower
>   bound — equivalently, minimising the reconstruction loss plus
>   $D_{\text{KL}}(q(\mathbf{W}) \| p(\mathbf{W}))$ from a standard normal
>   prior, scaled by $1/N$ \citep{Harrison2024}. The VBLL also learns a
>   scalar observation-noise variance, which provides a direct estimate of
>   aleatoric uncertainty.
> \end{itemize}
>
> All BNN variants estimate predictive distributions with 100 Monte Carlo
> forward passes at inference.

Side note: this lets you delete the duplicate architecture paragraph that
follows the BNN block in the current draft.

---

## 9. Mini-explanation of epistemic vs aleatoric

In the uncertainty-decomposition paragraph, replace the bare phrase with a
short gloss in parentheses on first use. Suggested wording (matching the
supervisor's hint):

> We decomposed the uncertainty values derived from probabilistic methods
> into **epistemic (knowledge-driven, reducible by collecting more or
> higher-quality training data) and aleatoric (data-driven, reflecting
> intrinsic noise in the labels and therefore irreducible)** components when
> applicable. For GP and VBLL models, we derive epistemic uncertainty from
> the posterior variance or weight sampling, and aleatoric uncertainty from
> a learned observation-noise term \citep{Rasmussen2005, Harrison2024}. We
> did not decompose uncertainty from other models such as BNN variants, in
> which the variance across Monte Carlo weight samples primarily reflects
> epistemic uncertainty \citep{kendall2017, gal2016}, or NGBoost and QRF, in
> which there is no straightforward mechanism to separate the two
> components from learned distributional parameters or quantiles
> \citep{Duan2020, Meinshausen2006}.

---

## Summary of file changes if you accept all of the above

- **paper.tex**:
  - Rewrite GP paragraph (§1).
  - Insert metrics-summary table (§2).
  - Add classification-noise sentence to Noise Strategies (§3).
  - Update noise-strategies figure caption (§4).
  - Reorder rows in `tab:regression_noise` (§5).
  - Rewrite BNN paragraph and delete duplicate architecture paragraph (§8).
  - Add epistemic/aleatoric gloss (§9).
- **scripts/generate_paper_figures.py**:
  - Define a single `STRATEGY_ORDER` constant and replace 7 hard-coded
    lists (§6).
  - Fix the per-panel bins bug in the noise-strategies figure (§7).
- Re-run figures on the server and pull the updated PNGs.
