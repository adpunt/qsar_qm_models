# Should the six noise strategies be cut? — decision memo

Date: 2026-08-21. Evidence: `scratchpad/val_rerun.parquet` (48,510 rows; 3 experimental
datasets x 6 strategies x 11 sigma x 13 models x 4 reps x 5 folds) plus the verified
QM9 facts supplied in the brief. All numbers below were computed in this session.

---

## 1. RECOMMENDATION

**Cut from six to four in the main text. Reframe as two mechanisms, each with two
implementations, spanning a range of doses.**

| Keep (main text) | Mechanism | RMS dose at sigma=1 (QM9) |
|---|---|---|
| Gaussian | noise spread evenly over all molecules | 1.000 |
| ValueProportional | noise spread evenly, scaled by y | 1.701 |
| Outlier | noise concentrated on a small subset | 0.502 |
| Quantile | noise concentrated on a small subset | 0.899 |

| Move to SI | Why |
|---|---|
| Threshold | On QM9 the cut-point is inert — every molecule clears it — so Threshold *is* Gaussian at 2 sigma. It is a dose relabelling, not a mechanism. |
| Heteroscedastic | Correlates with ValueProportional at **exactly rho = 1.00** across the 11 QM9 model rankings. Same mechanism, same answer. |

Nothing is deleted or hidden. All six runs stay in the SI with one short table
demonstrating the redundancy, plus a sentence in Methods: "we implemented six
mechanisms; two proved redundant by construction and are reported in Additional file 1."

**Plot against effective RMS dose, not sigma.** This is the single biggest
presentational win and it costs no new compute. It makes the dose axis an actual
manipulated variable and it lets you answer Heid et al. head-on.

### Is "two mechanisms at a range of doses" stronger than "six mechanisms"?

Stronger — decisively. Six strategies invites the reviewer question "why these six?",
which the author cannot answer and does not want to answer ("my goal is not to prove
these strategies unique"). Two mechanisms crossed with a dose axis is a *design*.
It states a hypothesis (where the noise lands matters; the exact distribution shape
does not) and tests it.

But go to two mechanisms, **not** to two strategies. RQ3 asks whether robustness
patterns generalise *across noise-injection mechanisms*. With one implementation per
mechanism, "mechanism" is perfectly confounded with "the particular function I wrote".
Two per mechanism is the minimum that makes RQ3 answerable at all. Four is therefore
the floor, not a compromise.

---

## 2. EVIDENCE

### 2.1 Most of the strategy effect is dose, not mechanism

Median R2 drop from sigma=0 to sigma=1, pooled over models/reps/folds, ranked against
the QM9 RMS dose ordering:

| dataset | Spearman(damage, dose) |
|---|---|
| ChEMBL-hERG-Ki | 0.886 |
| OpenADMET-LogD | 0.943 |
| OpenADMET-Caco2_Efflux | 0.086 |

Median R2 drop, sigma 0 -> 1:

| dataset | hetero | outlier | quantile | legacy(Gauss) | valprop | threshold |
|---|---|---|---|---|---|---|
| hERG-Ki | 0.068 | 0.162 | 0.213 | 0.186 | 0.362 | 0.475 |
| LogD | 0.038 | 0.068 | 0.110 | 0.127 | 0.166 | 0.293 |
| Caco2 | 0.092 | 0.420 | 0.490 | 0.456 | 0.456 | 0.326 |

On hERG and LogD the ordering is essentially the dose ordering. Caco2 is the exception
and it is informative: there Threshold does *less* damage than Gaussian (0.326 vs
0.456), which is inconsistent with a dose of 2.0 — the cut-point evidently bites on
Caco2, so Threshold is corrupting only a subset. That means Threshold is a *different
mechanism on different datasets*. A strategy whose mechanism you cannot state uniformly
cannot support a generalisation claim. It belongs in the SI.

### 2.2 Model rankings agree across strategies

Spearman between per-model mean-R2 rankings, computed per dataset:

- LogD: 0.78–0.99 across all 15 pairs.
- hERG: 0.93–0.98 *within* {hetero, gaussian, outlier, quantile}; 0.97 between
  {threshold, valprop}; 0.57–0.82 between the blocks — and that split tracks dose
  (threshold/valprop are the two heaviest doses; at sigma=1 threshold has already
  driven median R2 to -0.04, so its "ranking" is partly ranking of collapse).
- Caco2 is noisy (0.24–1.00) and is the weakest dataset throughout.
- QM9 (from the brief): mean pairwise rho 0.895, Kendall's W 0.912.

So the answer to "which model is most robust" barely changes with strategy. Six
answers to a question with one answer is six times the figure real estate for no
extra information.

### 2.3 The one real mechanism effect: concentration

Damage per unit of RMS dose, sigma=1:

- hERG: Outlier 0.162 / 0.502 = **0.323** vs Gaussian 0.186 / 1.000 = **0.186**.
  Outlier does ~1.7x more damage per unit of injected variance.
- LogD: Outlier 0.135 vs Gaussian 0.127 per unit dose (small but same sign).

And per-sample uncertainty shows signal **only** under Outlier and Quantile.

That is the paper's actual mechanism finding, and it is a *concentration* finding, not
a *distribution-shape* finding. Keeping Outlier and Quantile in the main text is
non-negotiable. Keeping a third and fourth variant of "spread evenly" is not.

### 2.4 It agrees with the literature

Heid et al. (JCIM 2023) found Gaussian, uniform, hyperbolic and bimodal noise at matched
SD gave overlapping learning curves. Distribution shape at matched dose does not matter.
This work reproduces that (the four spread-evenly strategies are interchangeable once
dose is accounted for) and then extends it: *where* the noise lands does matter. That is
a citable, defensible contribution. Six undifferentiated strategies obscures it; two
mechanisms x dose states it.

---

## 3. DRAFT PARAGRAPH (author's voice, for Methods or start of Results)

> We injected label noise in six ways, but they are not six independent tests. Two are
> redundant by construction: on QM9 the threshold strategy's cut-point never triggers, so
> it is simply Gaussian noise at twice the standard deviation, and the heteroscedastic
> and value-proportional strategies produce rankings that agree perfectly (Spearman 1.00
> across the eleven models). We therefore report four in the main text and the remaining
> two in Additional file 1. The four fall into two groups: noise spread over every
> molecule (Gaussian, value-proportional) and noise concentrated on a small number of
> molecules (outlier, quantile). Within a group the choice makes almost no difference to
> which model or representation comes out ahead once the amount of noise is matched,
> which agrees with earlier work showing that the shape of the noise distribution matters
> less than its size. Between the groups it does matter: concentrated noise damages
> accuracy more per unit of injected variance, and it is the only kind of noise that the
> models' own uncertainty estimates detect. Because the strategies do not deliver the
> same amount of noise at a given sigma, we plot results against the effective noise
> standard deviation rather than sigma, so that the amount of noise and the way it is
> distributed can be read separately.

---

## 4. WHAT IS LOST — HONESTLY

1. **The "six mechanisms" breadth claim disappears.** Any sentence of the form "patterns
   held across six distinct noise-injection mechanisms" must go. Six sounded like more
   evidence than it was, and the author will feel the loss even though the information
   content is unchanged.
2. **Heteroscedastic noise has the best chemistry story and it is the one being cut.**
   Assay error genuinely scales with the magnitude of the measured value; a reviewer
   from an experimental background may specifically want to see it. Mitigation: keep
   ValueProportional (same family, simpler to state) in the main text and say in one
   sentence that the heteroscedastic variant gave identical rankings.
3. **Caco2's disagreement gets less airtime.** Caco2 is where the strategies genuinely
   diverge (rho with dose 0.086; rank correlations 0.24–1.00). With four strategies there
   is less room to argue that this divergence is noise rather than mechanism. It is
   probably noise — Caco2 is the weakest, most variable dataset — but the case is thinner.
4. **Reviewer objections to expect:**
   - "You ran six and report four. Which four did you choose, and when?" Answer this
     pre-emptively and completely: state the redundancy criteria, show all six in SI,
     never let it look like selective reporting. This is the only real risk in the whole
     recommendation and it is fully mitigated by transparency.
   - "Your redundancy argument for Threshold is QM9-specific." Correct, and the Caco2
     numbers show it. Say so explicitly rather than letting a reviewer find it.
   - "Two implementations per mechanism is still a small sample of mechanisms." True, and
     unanswerable without new experiments. It is at least two more than one.

---

## 5. EFFECT ON RQ3 (generalisation across noise mechanisms) — HELPS

RQ3 is the question most obviously at risk, and the cut **strengthens** it.

As written, RQ3 is currently answered by "we tried six things and got similar answers".
That is weak because the six were not chosen to differ in any stated way, so similarity
is uninformative — you cannot tell whether the patterns are robust or whether the six
things were secretly the same thing. Given that Hetero and ValProp correlate at 1.00 and
Threshold is Gaussian in disguise, they *were* secretly the same thing. Reported as six,
RQ3's evidence is partly circular.

Reframed, RQ3 becomes a real test with a stated axis of variation: does the answer
change when the noise moves from spread to concentrated, and when the dose changes?
The answer is "mostly no for model and representation rankings (rho 0.78–0.99), yes for
damage per unit dose and yes for uncertainty detection". That is a specific, falsifiable,
interesting answer. It is a better RQ3 than the one currently supported.

Caveat to state plainly: generalisation is demonstrated across *two* mechanisms and a
dose range, not across the space of all noise processes. Do not overclaim.

---

## 6. COUNTER-ARGUMENT — THE STRONGEST CASE AGAINST CUTTING

**(a) The experiments are already run and paid for.** No compute is saved. Cutting only
removes information from the reader. Journal of Cheminformatics has no hard page limit
and tolerates long Results.

**(b) Redundancy is a finding, not a reason to hide.** "Hetero and ValProp agree at
rho = 1.00" is itself worth reporting prominently — it tells the field that two commonly
proposed noise models are interchangeable. Demoting them to SI buries a useful negative
result. A reviewer could reasonably say: you discovered the redundancy *by running all
six*, so all six earned their place in the main text.

**(c) The redundancy argument is dataset-dependent and therefore shaky.** Threshold is
inert on QM9, but the Caco2 damage ordering (0.326 vs Gaussian's 0.456) says it is *not*
inert there. So "Threshold = Gaussian at 2 sigma" is true on the main dataset and false
on one validation dataset. Building a cut on a claim that fails on one third of the
validation data is uncomfortable.

**(d) Cutting on the basis of results looks like post-hoc selection.** The strongest
version of this objection: you decided which strategies to feature *after* seeing that
they agreed. Even fully disclosed, this invites suspicion. Reporting all six is
selection-proof.

**(e) The author's own stated position cuts the other way.** "My goal is not to prove
these strategies unique." If uniqueness is not the claim, then redundancy is not a
problem, and the tidy-up buys nothing the author cares about.

**(f) Four is an unprincipled middle.** If dose is what matters, be honest and go to
Gaussian only, varying dose. If mechanism matters, keep all six. Four is a compromise
that satisfies neither logic cleanly.

### Why I still hold the recommendation

On (a) and (b): the cost of the sixth strategy is not compute, it is the reader's
attention and the figure's legibility. Six panels where four suffice makes the one real
finding — concentration matters, spread does not — harder to see. And (b) is fully
answered by *keeping* the redundancy result: it is stated in the main text in one
sentence with the number attached, and demonstrated in the SI. Nothing is buried; it is
promoted from "one of six panels" to "an explicit stated finding".

On (c): the Caco2 inconsistency is an argument for demoting Threshold, not retaining it.
A strategy that is Gaussian-at-2-sigma on one dataset and a subset-corruption on another
cannot serve as a fixed mechanism in a generalisation claim. Report it, do not lean on it.

On (d): post-hoc selection is only damaging when it is concealed or when the criterion is
"it gave a better result". Here the criterion is redundancy, stated openly, and it does
not change any conclusion — all six give the same model and representation rankings.
Disclose the ordering of decisions in one sentence and the objection evaporates.

On (e): the author's position is exactly why to cut. If uniqueness is not the claim, the
six strategies are overhead in service of a claim not being made. Removing overhead that
serves no argument is the whole point.

On (f) — the serious one: four is principled, not middling. Two mechanisms is the
hypothesis; two implementations per mechanism is the minimum needed to show that the
mechanism, and not the implementation, is what carries the effect. Going to one strategy
would make RQ3 unanswerable. Going to six confounds mechanism with implementation and
with dose. 2 x 2 x dose is the smallest design that separates all three.

**Decision stands: four in the main text — Gaussian, ValueProportional, Outlier,
Quantile — Threshold and Heteroscedastic to SI with an explicit redundancy table, all
results plotted against effective noise SD.**
