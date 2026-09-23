# Results and discussion — replacement text

Written 2026-09-21. **This document supersedes §R1 through §R7 of `PAPER_REVISION_GUIDE_FINAL.md`.**

Every block was written after reading the figure it carries, as an image, rather than from the
figure's caption or from a CSV. The visual readings are kept in full under each block's author
notes, including every place a caption disagrees with the drawing on disk.

Each block carries its own `\begin{figure}` input, with `\includegraphics`, `\caption` and
`\label`, placed where the text refers to it. The file names are exactly the PNG names in
`results/decisions_arc_20260916/figures/`.

Everything in a ```latex fence pastes into the manuscript. Everything under an **Author notes**
heading is for you and does not go into the paper.

## The four movements

| Block | Replaces | Figure it carries |
|---|---|---|
| §R1 Variance decomposition | `paper.tex:377–430` | F2, F3 |
| §R2 What label noise costs | `paper.tex:431–467` | F4a, F4c, R16 |
| §R3 Does making a model probabilistic help | `paper.tex:468–497` | R17 |
| §R4 The kind of noise | `paper.tex:460–467` | F4b, R19, R15 |
| §R5 Does it hold on measured labels | `paper.tex:542–556` | F8, R9 |
| §R6 Uncertainty under label noise | `paper.tex:498–541` | F6 |
| §R7 Whether uncertainty identifies the corrupted labels | no predecessor | **F7 does not exist — see §R7's notes** |

---

# §R1, §R2, §R3 — rewritten from the figures, 2026-09-21

Replaces the three blocks at `PAPER_REVISION_GUIDE_FINAL.md:1276–1617`. Every number below was read out
of a file in `results/decisions_arc_20260916/` this session, or out of
`results/master_tuned_hyperparameters.json` and `results/master_tuned_hyperparameters_lab.json`. The
ideas come from the figure readings in `scratchpad/rewrite/figures/`, which were read before any sentence
was written.

---

## MOVEMENT 1 — how the choice of model and the choice of representation divide the outcome

### §R1. Variance decomposition *(replaces `paper.tex:377–430`)*

```latex
\subsection{Variance decomposition}

The variance in predictive accuracy and in noise robustness on the QM9 HOMO--LUMO gap divides between
model architecture, molecular representation, the pairing of the two, and a residual
(Figure~\ref{fig:variance}). This decomposition covers the three noise conditions that ran on every
model and every representation: Gaussian, grouped-wider and grouped-shifted. The other four noise
conditions ran on a named subset of model-and-representation pairings and are not decomposed. For
predictive accuracy, model architecture and molecular representation account for comparable shares of
the variance in $R^2$, close to a third and a quarter under Gaussian noise. For noise robustness,
measured as \mbox{AUC$_{norm}$} (higher values indicate greater robustness), model architecture accounts
for about half the variance and molecular representation for under a tenth. Which of the two choices
dominates depends on the outcome being asked about. Label noise reaches the labels and not the features,
so molecular representation may have nothing extra to lose under noise.

\mbox{AUC$_{norm}$} is the area under a configuration's $R^2$ across the noise levels, divided by that
same configuration's clean $R^2$. Despite the name, it is not an area under a receiver operating
characteristic curve. A value of 1.0 is no accuracy lost at any noise level, and 0.5 is half the clean
accuracy gone. A value above one means a configuration scored higher with noise added than without it,
which happens where its clean $R^2$ is small. Because \mbox{AUC$_{norm}$} integrates every noise level, a
configuration that falls early and then flattens can reach the value of one that holds and then collapses.

The share of \mbox{AUC$_{norm}$} variance that model architecture takes is one number for thirteen base
models, and it hides a difference between families. Every one of the six neural models spans more
\mbox{AUC$_{norm}$} across the six representations than any of the other seven does. One cell of the grid
in Figure~\ref{fig:grid} is one base model on one representation, read as \mbox{AUC$_{norm}$} under one
noise condition. The four plain and fully Bayesian networks give up between 0.04 and 0.06 of
\mbox{AUC$_{norm}$} on ECFP4 against PDV. None of the seven tree and kernel models moves by more than a
hundredth of \mbox{AUC$_{norm}$} between those same two representations. The two variational networks are
the exception within the neural group, holding nearly the same \mbox{AUC$_{norm}$} on both
representations. Molecular representation is a robustness decision for a neural model, while for every
other family it is very largely an accuracy decision.

Set against the thirteen base models, the six representations are close together in \mbox{AUC$_{norm}$}.
Taking the median over the seven models that are not neural networks, the six representations differ by
at most 0.022 in \mbox{AUC$_{norm}$} on the gap and 0.061 on Caco-2. Those same seven models differ from
each other by 0.041 of \mbox{AUC$_{norm}$} on the gap and by 0.300 on Caco-2. No representation has the
highest median \mbox{AUC$_{norm}$} on every dataset, so every table names one representation rather than
averaging across them. ECFP4, the representation the model tables are computed at, spreads the thirteen
base models furthest apart in \mbox{AUC$_{norm}$}. It was chosen for that spread and not because it
scores highest.

Under grouped-shifted noise, which gives every molecule in a scaffold family the same offset, the
residual rises to over half the variance in \mbox{AUC$_{norm}$}. For predictive accuracy under the same
condition it rises to three quarters of the variance in $R^2$. The residual is the variation between ten
replicates of one configuration, differing only in seed. Under Gaussian and grouped-wider noise, at the
same delivered amount of noise, it sits near a fifth of the variance in \mbox{AUC$_{norm}$}. Which draw
of scaffold offsets a run receives may account for more of the variance in \mbox{AUC$_{norm}$} than model
architecture does. Why the residual rises further for predictive accuracy than for robustness, we cannot
say. Where the labels may be noisy by an unknown amount, this suggests that effort is better spent on
model architecture, at the cost of the accuracy molecular representation determines.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F2_variance_decomposition.png}
\caption{Share of the variance in each outcome explained by model architecture, molecular representation,
their pairing, and the residual, on the QM9 HOMO--LUMO gap. a) robustness, as \mbox{AUC$_{norm}$}; b)
predictive accuracy, as $R^2$ at a noise level of 1.0, one spread of the clean training labels. The bottom
axis is the noise condition, the side axis the share of variance.}
\label{fig:variance}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F3_model_by_representation.png}
\caption{Robustness (\mbox{AUC$_{norm}$}) of the thirteen base models on the QM9 HOMO--LUMO gap under a)
Gaussian and b) grouped-shifted noise. Rows are models, columns the six representations. Colour is fixed
across both panels, its bright end the higher \mbox{AUC$_{norm}$}. Grouped-wider repeats panel a) and is
in an additional file. Values are medians over ten replicates, excluded configurations in Additional
file~5.}
\label{fig:grid}
\end{figure}
```

**Measured on the block above, after the house-style pass of 2026-09-21: 653 words of prose in five
paragraphs of 7, 5, 6, 6 and 7 sentences.** Median sentence 22 words, longest 30, shortest 11, and
nothing under five. Five of the 31 sentences carry a decimal, which is 16 per cent and over the
house-style band of 12. Two of those five are the definition of \mbox{AUC$_{norm}$}, where 1.0 and 0.5
are what the sentences are about rather than results; excluding that paragraph it is 3 of 26, or 12 per
cent. **To get the whole block inside the band, the sentence to cut is "Those same seven models differ
from each other by 0.041 of \mbox{AUC$_{norm}$} on the gap and by 0.300 on Caco-2" — but that is the raw
pair the sentence before it is compared against, so cutting it breaks house-style rule 9.**

**What changed, and why.**

- **Paragraph 3 now carries what Figure~\ref{fig:grid} shows rather than restating paragraph 1.** The
  figure's content is that the neural loss is not spread over the grid: it sits on ECFP4 for the four
  plain and fully Bayesian networks, and the two variational networks do not show it. From
  `auc_norm_qm9.csv`, QM9 under Gaussian noise: NN-$\alpha$ 0.886 on ECFP4 against 0.940 on PDV,
  BNN-$\alpha$ 0.881 against 0.944, NN-$\beta$ 0.892 against 0.935, BNN-$\beta$ 0.897 against 0.943. The
  two variational networks move 0.009 and 0.019 between the same two representations. The seven tree and
  kernel rows move at most 0.011, which is RF.
- **One idea from the two scatter drawings is in the notes and not in the prose, and it is yours to
  promote.** Every one of the six neural configurations keeps more of its clean accuracy on PDV than on
  ECFP4, and RF, QRF, XGBoost, NGBoost and SVM do not — the support vector machine is level to four
  decimals. Rank correlation over the thirteen base models, ECFP4 against PDV, is 0.654 with $p = 0.015$;
  against MHG-GNN it is 0.505 with $p = 0.078$. Both recomputed here from `auc_norm_qm9.csv`. The guide's
  line 2524 proposes moving only the 0.505 into a Methods sentence, and those two numbers bracket the
  answer rather than one of them being it.
- **The Sort \& Slice sentence was wrong and is out of the prose.** The old text said it leads on three of
  four datasets "under both conditions that run the full roster". Medians over the seven non-neural
  models: under Gaussian it leads on QM9, logD and hERG $K_i$; under grouped-shifted on the same three;
  under grouped-wider on logD alone, where ChemBERTa leads QM9, PDV leads Caco-2 and MHG-GNN leads hERG
  $K_i$. The refusal to rank is kept as a sentence and the count is dropped.
- **Two sentences of the F2 caption are cut.** The slot marked "no value" has no referent: `anova_eta2.csv`
  has six rows, three conditions by two outcomes, and censoring is not on the bottom axis at all. The
  claim that the legend sits beside the panels is true of the current code and false of the PNG on disk,
  so the caption no longer says where the legend is.
- **Three clauses of the F3 caption are cut or corrected.** No grey cell exists on either panel — 78 of 78
  are filled — so the sentence explaining grey cells is replaced by one naming where the exclusions live.
  "Every value on this dataset falls inside it" is false: the minimum over all seven QM9 conditions in
  `auc_norm_qm9.csv` is 0.809, below the colour floor of 0.83, and the caption now says "in these
  panels". "The three conditions that run on a named subset of pairs" is four: censoring, Student-t,
  outlier and Laplace. The clause saying the range is printed on the colour bar is gone, because commit
  `93fe179` takes the range off the label.

- **House-style pass, 2026-09-21.** "Ladder of levels", "the model term" and "the six neural
  configurations" are out: a configuration is one cell of the grid everywhere else in this document, and
  the six in question are models. Every share now names what it is a share of, and every spread now
  carries \mbox{AUC$_{norm}$}. The shared-offset condition is named grouped-shifted at its first use and
  described in the same clause. "It reaches the targets" is now "the labels". One sentence naming the
  four conditions that are not decomposed is added to the first paragraph.

**Numbers behind the paragraphs.**

`anova_eta2.csv`, one row per condition and outcome, thirteen base models by six representations by ten
replicates, 780 values each:

| condition | outcome | model | rep | pairing | residual |
|---|---|---|---|---|---|
| Gaussian | robustness | 49.5 | 9.2 | 20.4 | 20.8 |
| Grouped, wider | robustness | 50.5 | 8.6 | 19.2 | 21.6 |
| **Grouped, shifted** | robustness | 30.5 | 6.6 | 6.6 | **56.3** |
| Gaussian | accuracy | 29.1 | 26.6 | 13.7 | 30.6 |
| Grouped, wider | accuracy | 36.3 | 24.7 | 11.5 | 27.5 |
| **Grouped, shifted** | accuracy | 7.1 | 10.2 | 6.1 | **76.6** |

Best minus worst \mbox{AUC$_{norm}$} over the six representations, QM9 under Gaussian noise, one row per
base model, from `auc_norm_qm9.csv`: NGBoost 0.015, SVM 0.018, LightGBM 0.024, XGBoost 0.025, RF 0.026,
Gaussian process 0.027, QRF 0.030, then VBLL-$\alpha$ 0.039, NN-$\beta$ 0.043, BNN-$\beta$ 0.053,
NN-$\alpha$ 0.054, VBLL-$\beta$ 0.059, BNN-$\alpha$ 0.063. Six neural rows, seven others, and the two
groups do not overlap.

Median \mbox{AUC$_{norm}$} per representation over the seven non-neural models, under Gaussian noise:

| representation | QM9 | logD | Caco-2 | hERG $K_i$ |
|---|---|---|---|---|
| ECFP4 | 0.946 | 0.891 | 0.841 | 0.842 |
| PDV | 0.945 | 0.913 | 0.856 | 0.842 |
| MHG-GNN | 0.934 | 0.900 | 0.854 | 0.847 |
| Avalon | 0.955 | 0.881 | 0.795 | 0.856 |
| ChemBERTa | 0.953 | 0.887 | 0.850 | 0.854 |
| Sort \& Slice | 0.956 | 0.921 | 0.842 | 0.879 |

Spread across the six representations: 0.022, 0.040, 0.061, 0.037. Spread across the seven models on the
same four datasets: 0.041, 0.068, 0.300, 0.254. Spread across the thirteen base models at one
representation, QM9 under Gaussian: ECFP4 0.098, Avalon 0.078, Sort \& Slice 0.067, ChemBERTa 0.054,
MHG-GNN 0.048, PDV 0.040. Over the same thirteen models, a representation is some model's worst of the
six on MHG-GNN 6 times, ECFP4 4 times, Sort \& Slice twice, Avalon once, PDV and ChemBERTa never.

🔴 **TODO — recompute after the matched-settings re-run.** Model is a factor in this decomposition, four
of its levels change, and `rf300` is added as a fifth.

⚠️ **Two things this subsection must not say.** The submitted paper's §4.1 says the pairing term is the
largest source of variance for accuracy under all six strategies. On this run it is the largest in none
of the six condition-and-outcome rows. And the old "83.6\% and 77.4\% residual" belongs to noise
strategies that no longer exist.

⚠️ **Two numbers in the fourth paragraph are medians taken across the six representations, which
house-style rule 11 forbids in the text.** "0.041 of \mbox{AUC$_{norm}$} on the gap and by 0.300 on
Caco-2" is the spread across the seven models that are not neural networks, where each model's value is
first medianed over the six representations. The same operation is behind the §R2 table of medians per
non-neural model. Neither can be rewritten without recomputing the spread inside one named
representation, which is a change to `scripts/run_paper_analysis.py` and a re-run, not an edit here. The
values are left exactly as measured and the sentence is not the place to fix it.

⚠️ **"None of the seven tree and kernel models moves by more than a hundredth" disagrees with the note
below it.** The note says those seven move at most 0.011 between ECFP4 and PDV, which is above a
hundredth. The claim is left at the value it was written with and needs your ruling: either the prose
says 0.011, or the note is the one that is wrong.

---

## MOVEMENT 2 — what the choice of model buys, and whether probabilistic machinery helps

### §R2. What label noise costs *(replaces `paper.tex:431–467`)*

```latex
\subsection{Robustness and clean accuracy}

The variance decomposition says whether model architecture or molecular representation moves the
outcome; it does not say how far any one model falls. The eight models with the highest
\mbox{AUC$_{norm}$} on the QM9 HOMO--LUMO gap all lose accuracy as the labels are corrupted, as seen
in Figure~\ref{fig:curves}. The loss stays small until the added noise reaches about half the spread
of the clean training labels. The HOMO--LUMO gap is computed rather than measured, so a noise level
on it has no equivalent in published assay error. Where a curve starts on the clean labels does not
predict how steeply it then falls.

NGBoost is the clearest case of that split. Of the thirteen base models under Gaussian noise,
NGBoost has the highest \mbox{AUC$_{norm}$} on all six representations. It ranks between eleventh
and thirteenth of thirteen for clean $R^2$ on every one of those six. Its clean $R^2$ runs from
0.706 on ECFP4 to 0.865 on PDV, the widest range of the thirteen, while its \mbox{AUC$_{norm}$}
varies over the narrowest. What molecular representation buys NGBoost is accuracy on clean labels,
and it buys almost no robustness. Read on its own, a robustness column would make NGBoost the model
to recommend here, which it is not. Each model's clean $R^2$ is printed beside its
\mbox{AUC$_{norm}$}, as seen in Figure~\ref{fig:robustness}, so that a share is never read without
the accuracy it is a share of.

LightGBM is the case a single summary number cannot show. It begins at the top of those eight
curves, holds with the rest to a noise level of 1.0, then ends below every other curve. Its
\mbox{AUC$_{norm}$} ranks fifth of the thirteen base models at ECFP4, because it collapses after
most of the area has been accumulated. Where the accuracy went has to be read off the curves, which
is the one thing \mbox{AUC$_{norm}$} does not report.

The split between accuracy and robustness could rest on NGBoost alone, so we tested it across all
thirteen base models. Across the thirteen base models on the HOMO--LUMO gap at ECFP4, clean $R^2$
and \mbox{AUC$_{norm}$} rank in opposite directions at a Spearman correlation of $-0.18$, which does
not reach significance. We also rescaled both quantities inside each dataset and repeated the test
over the seventy-five pairings of model and representation that ran on all four datasets. Across
those seventy-five pairings, clean $R^2$ and \mbox{AUC$_{norm}$} rank in opposite directions at a
Spearman correlation of $-0.35$. That correlation is taken over pairings drawn from four datasets;
it says the orderings disagree rather than by how much on any one endpoint. Nine of the twelve
pairings that rank highest on both are a Gaussian process or a forest, one pairing to a row of
Table~\ref{tab:pairs}. The two orderings disagree wherever they were tested, and not only at
NGBoost.

Under Gaussian noise, twenty-one of the twenty-four combinations of dataset and representation carry
all thirteen base models. The other three carry fewer than thirteen: ChemBERTa is missing models on
Caco-2 and on hERG $K_i$, and MHG-GNN is missing one on Caco-2. Across those twenty-one
combinations, the two forests are the only models whose \mbox{AUC$_{norm}$} rank never falls below
eighth of thirteen. The Gaussian process never falls below ninth of thirteen, and its median rank on
clean $R^2$ over those twenty-one combinations is first of thirteen. Among the seven models that are
not neural networks, LightGBM and XGBoost are the bottom two on Caco-2 and on hERG $K_i$. Those two
sit in the middle of the seven on the computed property, under every condition that runs all
thirteen base models. Why they lose so much more on the two measured endpoints with the lowest clean
$R^2$, we cannot say.

On Caco-2 and hERG $K_i$, four combinations of model, representation and noise condition retained
more accuracy with noise added than without it. All four began from a clean $R^2$ between 0.325 and
0.371, against the floor of 0.30 below which a replicate is excluded. We report them rather than
patching the measure, because a slope that is not divided by the clean $R^2$ cannot be compared
across models that start from different clean $R^2$. The plain forest and the quantile forest give
up a median rank on clean $R^2$ of eighth of thirteen. This suggests that either forest is the safer
choice when the amount of label noise is unknown.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F4a_models_under_noise.png}
\caption{Predictive accuracy against label noise on the QM9 HOMO--LUMO gap under Gaussian noise, for
the eight models with the highest \mbox{AUC$_{norm}$} at ECFP4. a) ECFP4; b) PDV, drawn for the same
eight models, so panel b) is not a separate selection. Bottom axis: the noise level, as a fraction
of the spread of the clean training labels. Side axis: $R^2$ on held-out molecules, as the median
over ten replicates. The dashed vertical line marks the noise level every table in this paper
reports at.}
\label{fig:curves}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F4c_robustness_grid.png}
\caption{Robustness (\mbox{AUC$_{norm}$}) of all thirteen base models on the QM9 HOMO--LUMO gap at
ECFP4, under the three noise conditions that ran on every model: Gaussian, grouped-wider and
grouped-shifted. Rows are models, ordered by \mbox{AUC$_{norm}$} under Gaussian noise; values are
the median over ten replicates, and higher values indicate more performance retained under noise.
The first column is each model's clean $R^2$ and is left uncoloured; it is the quantity the other
three columns are a fraction of. Censoring, Student-$t$, outlier and Laplace noise are absent, each
having run on a named subset of model-and-representation pairings.}
\label{fig:robustness}
\end{figure}
```

**Measured on the block above, after the house-style pass of 2026-09-21: 721 words of prose in six
paragraphs of 5, 7, 4, 7, 7 and 4 sentences.** Median sentence 22 words, longest 30, shortest 13. Three
of the 34 sentences carry a decimal, which is 9 per cent and inside the house-style band of 12. **It is
121 words over the 600 you asked for.** The cut I would make is the third paragraph, on LightGBM, which
is 86 words and is the only paragraph here whose finding also appears in §R5 — and losing it means losing
the one worked case of what \mbox{AUC$_{norm}$} hides. The alternative cut is the second half of the
fifth paragraph, which is the LightGBM and XGBoost result on the two measured endpoints, and §R5's last
paragraph carries that too.

**What changed, and why.**

- **Paragraph 1 no longer quotes a value off Figure~\ref{fig:curves}, because no file holds one.** F4a is
  drawn from a frame `scripts/run_paper_analysis.py` keeps in memory at line 362 and never writes out, so
  the per-level $R^2$ values on those curves are in no CSV in the harvest.
  `model_accuracy_by_level.csv` looks like the backing file and is not: its `n_representations` column is
  6 and its `mean_r2` is a mean across all six representations, so quoting it would be both wrong and an
  average across representations. Only the seven noise levels can be taken from it. **Everything the
  paragraph says about the curves is a shape, not a value.**
- **LightGBM's late collapse is now its own paragraph, and it is what earns the sentence about what
  \mbox{AUC$_{norm}$} cannot show.** Read off the drawing it starts at the top of the group and crosses
  below every other line between noise levels 1.0 and 1.5; `auc_norm_qm9.csv` puts its
  \mbox{AUC$_{norm}$} at 0.938, fifth of thirteen at ECFP4, which is where the collapse disappears. The
  old version made the same point about the measure with no case attached to it.
- **Figure~\ref{fig:robustness} is F4c and it is new to this block.** It is the drawing that puts the
  clean-$R^2$ column beside the robustness columns, which is this subsection's whole argument: NGBoost is
  the top row on the lowest clean $R^2$ in the grid, and NN-$\beta$ is near the bottom on the highest.
  **F4c has no caption anywhere in the guide and `RERUN_PLAN.md` §14.25 puts it in the additional files,
  so promoting it is your call.** The caption above is written from the drawing.
- **The orienting sentences naming the axes are gone from the prose and live in the captions.** Both
  target-journal papers in the reference set do it that way, which is the amendment the voice card makes
  to house-style rule 13.
- **"Nine of the twelve" is checked.** Ordering `standout_pairs.csv` by `combined_scaled`, the top twelve
  are the Gaussian process at Sort \& Slice, MHG-GNN, PDV, ECFP4, ChemBERTa and Avalon; RF at Sort \&
  Slice and MHG-GNN; QRF at Sort \& Slice; SVM at MHG-GNN; NN-$\alpha$ at MHG-GNN; and NN-$\beta$ at Sort
  \& Slice.

- **House-style pass, 2026-09-21.** "The roster", "mid-table", "the ladder", "the opposite defect" and
  "clean accuracy" as a name for clean $R^2$ are out. LightGBM's rank is now fifth of thirteen at ECFP4
  rather than mid-table, the Gaussian process's accuracy claim now names clean $R^2$ and its denominator,
  the four cells that scored above one now name Caco-2 and hERG $K_i$, and the closing recommendation
  names the plain and the quantile forest. The second and fourth paragraphs no longer end on a figure
  pointer and a table pointer.

**Numbers behind the paragraphs.**

NGBoost on QM9 under Gaussian noise, from `auc_norm_qm9.csv`. It is the most robust of the thirteen base
models at every one of the six representations:

| representation | clean $R^2$ | \mbox{AUC$_{norm}$} | clean-accuracy rank of 13 |
|---|---|---|---|
| ECFP4 | 0.706 | 0.979 | 13 |
| Sort \& Slice | 0.748 | 0.975 | 13 |
| ChemBERTa | 0.762 | 0.980 | 13 |
| Avalon | 0.786 | 0.982 | 13 |
| MHG-GNN | 0.852 | 0.967 | 11 |
| PDV | 0.865 | 0.975 | 12 |

Its clean-$R^2$ range of 0.159 is the widest of the thirteen, against 0.114 for the next and 0.050 for
the narrowest. Its \mbox{AUC$_{norm}$} range of 0.015 is the narrowest, against 0.063 for the widest.

The decoupling, recomputed here. Within QM9 at ECFP4 under Gaussian noise, clean $R^2$ against
\mbox{AUC$_{norm}$} over thirteen base models: Spearman $-0.176$, $p = 0.566$. Under grouped-shifted at
the same representation: $-0.500$, $p = 0.082$. Across datasets, each quantity rescaled from worst to best
within its own dataset and then averaged over the four, over the 75 comparable rows of
`standout_pairs.csv`: Spearman $-0.350$, $p = 0.0021$. Those two are different measurements and
`RERUN_PLAN.md` §14.15b warns they must not be conflated.

Ranks over the 21 dataset-and-representation combinations that carry all thirteen base models under
Gaussian noise, recomputed from `auc_norm_qm9.csv` and `auc_norm_assay.csv`:

| model | worst robustness rank | median robustness rank | median clean-accuracy rank |
|---|---|---|---|
| RF | 8 | 2 | 8 |
| QRF | 8 | 3 | 8 |
| Gaussian process | 9 | 7 | 1 |
| NGBoost | 11 | 3 | 12 |
| SVM | 11 | 6 | 5 |
| everything else | 12 or 13 | 7 to 11 | 3 to 13 |

Median \mbox{AUC$_{norm}$} over the six representations, one row per non-neural model, under Gaussian
noise. LightGBM and XGBoost are the bottom two on Caco-2 and on hERG $K_i$ and mid-table on QM9:

| | QM9 | logD | Caco-2 | hERG $K_i$ |
|---|---|---|---|---|
| NGBoost | 0.977 | 0.925 | 0.859 | 0.855 |
| RF | 0.961 | 0.937 | 0.863 | 0.900 |
| QRF | 0.956 | 0.927 | 0.903 | 0.891 |
| XGBoost | 0.945 | 0.869 | **0.709** | **0.720** |
| LightGBM | 0.944 | 0.871 | **0.603** | **0.646** |
| Gaussian process | 0.936 | 0.902 | 0.852 | 0.857 |
| SVM | 0.936 | 0.886 | 0.842 | 0.851 |

`d6_auc_above_one.csv` holds four rows: VBLL-$\beta$ on MHG-GNN under grouped-wider on Caco-2 at 1.134
from a clean $R^2$ of 0.331, VBLL-$\alpha$ on ChemBERTa under Gaussian on hERG $K_i$ at 1.088 from 0.356,
BNN-$\alpha$ on PDV under Gaussian on Caco-2 at 1.081 from 0.325, and VBLL-$\alpha$ on Avalon under
grouped-wider on Caco-2 at 1.052 from 0.371.

🔴 **TODO — recompute after the matched-settings re-run.** Every rank in this subsection is a rank against
other models, and five models change or appear.

---

### §R3. Does making a model probabilistic help *(replaces `paper.tex:468–497`)*

🔴 **The counts in the third paragraph are the ones the re-run exists for.** They are marked in the LaTeX
as a comment, so the block pastes and compiles, and the comment names the files that settle them.

```latex
\subsection{Probabilistic and deterministic counterparts}

Those comparisons set one model family against another, and none of them says whether a given
model's own probabilistic form changes what label noise costs it. A model that reports a
distribution rather than a point value may lose less of its clean $R^2$ as the labels are
corrupted. We compared three families across the noise levels: two neural architectures and the
forests (Figure~\ref{fig:variants}). Each neural architecture enters as a plain network, a Bayesian
network, and a Bayesian network with a variance head. NGBoost has no deterministic counterpart
among the thirteen base models and is left out of this comparison. Boosting against bagging is not
the same model with probabilistic machinery added. NGBoost's settings are the ones its own authors
report \citep{Duan2020}. Every comparison in this subsection is a model against its own
probabilistic form, and nothing else.

A comparison of this kind is readable only where the two models carry the same settings. Each plain
network and its fully Bayesian counterpart now run at one shared setting on all four datasets,
chosen by the worse of the two models in the pair. The plain forest is built at the quantile
forest's own count of 300 trees, so that pair differs by the quantile machinery alone. The two
networks that add a variance head have no tuned setting on any dataset and run at their defaults, so
that pair is reported as unmatched rather than scored.

% TODO after the matched-settings re-run. The counts in this paragraph come from the 16 September
% harvest, which predates commit 85bc2dc, so the plain networks and their Bayesian counterparts did not
% yet share a setting. Recompute from d10_probabilistic.csv and base_against_variant.csv against the
% settings now in results/master_tuned_hyperparameters.json and
% results/master_tuned_hyperparameters_lab.json.
Making a network Bayesian moved \mbox{AUC$_{norm}$} up more often than down on both architectures
(Additional file~8, one column per representation and one row per pair). One comparison is one pair
at one representation under one noise condition, tested with a signed-rank test over the ten
replicates. For NN-$\alpha$ against BNN-$\alpha$ it rose significantly in nine of eighteen
comparisons and fell in none. For NN-$\beta$ against BNN-$\beta$ it rose in six of eighteen and fell
in three. Giving the Gaussian process a per-molecule observation-noise term moved
\mbox{AUC$_{norm}$} the other way, lowering it in fourteen of twenty-seven comparisons. Its clean
$R^2$ fell as well. The quantile forest and the plain forest are indistinguishable until the added
noise passes about half the spread of the clean training labels. Beyond that level the quantile
forest ends lower. Which way a probabilistic form moves \mbox{AUC$_{norm}$} depends on the family it
is added to.

No probabilistic counterpart differed from its base model by more than 0.011 of \mbox{AUC$_{norm}$}
at the median of its comparisons. Over the same six representations the thirteen base models
themselves run from 0.871 to 0.935 in \mbox{AUC$_{norm}$} under grouped-shifted noise. Choosing a
different model family therefore moves \mbox{AUC$_{norm}$} further than making a given model
probabilistic does. This suggests that a probabilistic form is worth choosing for its uncertainty
estimate rather than for robustness to label noise.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R17_variant_families_ecfp4_gaussian.png}
\caption{Predictive accuracy under Gaussian label noise on the QM9 HOMO--LUMO gap at ECFP4, for each
model against its own probabilistic counterpart. a) NN-$\alpha$ as a plain network, as BNN-$\alpha$,
and as BNN-$\alpha$ with a variance head; b) the same three for NN-$\beta$; c) the plain forest at
300 trees against the quantile forest. The bottom axis is the noise added to the training labels, as
a fraction of the spread of the clean training labels, and the side axis is $R^2$ on held-out
molecules, taken as the median over ten replicates. The Gaussian process has no deterministic
counterpart among the thirteen base models and has no panel.}
\label{fig:variants}
\end{figure}
```

**Measured on the block above, after the house-style pass of 2026-09-21: 464 words of prose in four
paragraphs of 7, 4, 7 and 4 sentences.** Median sentence 21 words, longest 30, shortest 14. Two of the 22
sentences carry a decimal, which is 9 per cent.

**What the house-style pass of 2026-09-21 changed.** "The first architecture" and "the second" are now
NN-$\alpha$ against BNN-$\alpha$ and NN-$\beta$ against BNN-$\beta$, and "it rose in six" now carries
its denominator of eighteen. The signed-rank test is named in the sentence that says what one comparison
is. "Panel c)" is out of the prose and the quantile forest's 300 trees are in it. "The ladder" and "the
label spread" are said out. Additional file~8 moved into the finding sentence as a parenthesis, and the
paragraph now ends on a consequence.

**Which pairs are matched, on which dataset.** Read this session from
`results/master_tuned_hyperparameters.json` for QM9 and `results/master_tuned_hyperparameters_lab.json`
for logD, Caco-2 and hERG $K_i$. A pair counts as matched when both models carry byte-identical settings
at all six representations.

| pair | QM9 | logD | Caco-2 | hERG $K_i$ |
|---|---|---|---|---|
| NN-$\alpha$ against BNN-$\alpha$ (`dnn`, `dnn_bnn_full`) | matched | matched | matched | matched |
| NN-$\beta$ against BNN-$\beta$ (`mlp`, `mlp_bnn_full`) | matched | matched | matched | matched |
| NN-$\alpha$ against VBLL-$\alpha$ (`dnn`, `dnn_bnn_full_variational`) | matched | differs at all six | differs at all six | matched |
| NN-$\beta$ against VBLL-$\beta$ (`mlp`, `mlp_bnn_full_variational`) | differs at all six | differs at all six | no entry at all | no entry at all |
| BNN-$\alpha$ against its variance head (`dnn_bnn_full`, `dnn_bnn_full_mve`) | no entry | no entry | no entry | no entry |
| BNN-$\beta$ against its variance head (`mlp_bnn_full`, `mlp_bnn_full_mve`) | no entry | no entry | no entry | no entry |
| `rf300` against QRF | matched by construction on all four |  |  |  |
| Gaussian process against its heteroscedastic form | neither has a tuned path, on all four |  |  |  |

Three things that table is saying, each checked in code this session:

1. **The two variance-head networks have no entry in either file.** `models/tuning_rosters.py:195` gives
   them their own roster keys, and the comment above those two lines says why: `search_family` stripped
   `_bnn_full`, `_bnn_full_variational` and `_bnn_full_variational_hetero` and not `_bnn_full_mve`, so
   every sweep recorded them blocked and exited 0. They train at the shared spec default while the model
   they are compared against carries a tuned entry. **That is the one pair in the roster whose comparison
   the re-run alone does not clean.**
2. **`rf300` is matched to QRF by construction and not by tuning.** Neither file carries a forest at all,
   so all three forests read `models/model_defaults.py`, where the `rf300` entry at line 100 is the `qrf`
   entry with the quantile machinery removed: 300 trees, `min_samples_leaf` 5, `max_features` 0.3,
   bootstrap on. `rf` keeps its 100 trees, so `rf` against QRF was reading a 200-tree difference as well.
3. **Neither Gaussian process has a tuned path.** `models/tuning_rosters.py` maps `heteroscedastic_gp` to
   `None` and records that `train_heteroscedastic_gp` never calls `load_best_hyperparameters`, and
   `gauche_rbf` has no entry in either file. That pair is therefore unaffected by the tuning defect, and
   its numbers are the only ones in the block the re-run does not move.

**What the re-run changes, in one place.** The 16 September harvest predates commit `85bc2dc` of
2026-09-21. Before that commit `dnn` had no tuned entry at all while `dnn_bnn_full` was tuned to 64 and
32, so the NN-$\alpha$ comparison was a width comparison as well as a machinery comparison. Every count
in the third paragraph rests on that harvest.

**The counts as they stand, recomputed from `d10_probabilistic.csv` this session.** Each row of that file
is one signed-rank test over ten replicates, within one representation and one noise condition.

| pair | comparisons | significantly up | significantly down | median change in \mbox{AUC$_{norm}$} |
|---|---|---|---|---|
| NN-$\alpha$ → BNN-$\alpha$ | 18 | 9 | 0 | $+0.007$ |
| NN-$\beta$ → BNN-$\beta$ | 18 | 6 | 3 | $+0.011$ |
| BNN-$\alpha$ → variance head | 18 | 2 | 1 | $+0.001$ |
| BNN-$\beta$ → variance head | 18 | 5 | 0 | $+0.006$ |
| BNN-$\alpha$ → VBLL-$\alpha$ | 18 | 7 | 8 | $+0.003$ |
| BNN-$\beta$ → VBLL-$\beta$ | 18 | 7 | 1 | $+0.011$ |
| Gaussian process → heteroscedastic form | 27 | 5 | 14 | $-0.002$ |
| RF → QRF | 18 | 5 | 12 | $-0.008$ |

`base_against_variant_summary.csv` agrees and adds the clean-accuracy column: the variance head wins
clean $R^2$ on 16 of 16 paired cells for BNN-$\beta$ and on 3 of 18 for BNN-$\alpha$, and the two
heteroscedastic variational networks win clean $R^2$ on 18 of 18 while winning \mbox{AUC$_{norm}$} on 0 of
18, which that file calls a split.

**The 0.871 to 0.935 in the fourth paragraph.** That is the median over ten replicates for each of the
thirteen base models on QM9 under grouped-shifted noise, taken over the six representations, from
`auc_norm_qm9.csv`. NGBoost holds the top at 0.935 and NN-$\beta$ the bottom at 0.871. The old text said
"about six times the largest of those median changes", which is a ratio a reader cannot check against a
table; the raw pair is printed instead and the multiple is gone.

⚠️ **Both numbers in the fourth paragraph are medians taken across the six representations, which
house-style rule 11 forbids in the text.** The 0.871 and the 0.935 are per-model medians over the six
representations. The 0.011 is the largest of the per-pair median changes in `d10_probabilistic.csv`,
each of which is a median over eighteen comparisons spanning the same six representations and three
noise conditions. Fixing this needs the same quantity recomputed inside one named representation, which
is a change to `scripts/run_paper_analysis.py` and a re-run. The values are left exactly as measured.

**The figure needs `rf300` drawn in place of `rf`, and the code does not know that name yet.**
`grep rf300 scripts/figlib_config.py scripts/figlib_figures.py` returns nothing, so the re-run draws
panel c) with the 100-tree forest and the caption above is wrong until the figure library is taught the
name. The guide already carries this as a TODO under "Every label the draft text points at".

⚠️ **The submitted paper reports these substitutions as $+0.056$ to $+0.124$ and all significant.** Those
are noise degradation slopes on a retired measure and a retired noise scale. On \mbox{AUC$_{norm}$} the
same substitutions are worth about a hundredth. The direction for the quantile forest is unchanged.

---

## AUTHOR NOTES

**1. Three figures the task listed are carried as ideas rather than as pasted figure blocks: R6, both R18
scatters, and R16.** R6 is Figure~\ref{fig:grid} at a different colour range, R18 is the PDV-against-ECFP4
crossing recorded in §R1's notes, and R16 is the $-0.18$ in §R2's fourth paragraph. `RERUN_PLAN.md` §14.25
has all three in the additional files, and house-style rule 38 asks for one figure per finding in the
text. **None of the three has a LaTeX caption block anywhere in the guide today.** If you want any of them
promoted, name it and I will write the caption from the drawing.

**2. F4c is drawn into §R2 and it has never had a caption.** I wrote one from the PNG. It overlaps §R4,
which owns the kind of noise, on its three condition columns. The alternative is to leave §R2 with
Figure~\ref{fig:curves} alone and let the clean-$R^2$ column live only in Table~\ref{tab:robustness}.

**3. Five caption clauses are cut or corrected against the drawings on disk**, listed under §R1 and §R2.
Two of them — F2's four bars and F4a's two panels — describe what the code writes now rather than what the
PNG shows, so those captions are right only after
`sbatch slurm_scripts_analysis/run_paper_analysis.sh` has run.

**4. One figure defect that reaches no block here but reaches §R4.** On `R19_deep_conditions_qm9.png` the
columns of panels a) and b) are shifted by one: the column labelled Student-t holds censoring, and the
real Laplace column is drawn outside the grid. `scripts/figlib_figures.py:1867` records the fix, so the
re-run corrects it, and the PNG on disk must not be read until then.

**6. Tone pass of 2026-09-21, run over the LaTeX in every block of this document.** Your own Results
at `paper.tex:368-560` was read in full and all 106 of its sentences classified; the spec that came out
of that is in the scratchpad at `tone/voice_spec.md`. Every sentence in every block was then taken in
order, not sampled. Measured after: 241 prose sentences, median 19 words, longest 30, none under five.
Zero sentences open on a figure or table name, against zero in yours. Six open on "We", all of them on
a choice rather than a finding, which is what you do. "as shown in" does not appear; "as seen in" does.
No banned word is present.

What the pass could NOT fix, and what is still yours:

- The four numbers flagged with ⚠️ above, all medians taken across the six representations. Fixing them
  needs `scripts/run_paper_analysis.py` changed and an analysis re-run.
- Nothing in these blocks says how many noise levels \mbox{AUC$_{norm}$} integrates over. It belongs in
  Methods, stated once.
- The significance threshold and any multiple-comparison correction are named nowhere, though
  "significantly" is used in §R3 and "does not reach significance" in §R2.
- "base model" is used from §R1 onward with no definition. One Methods sentence has to say which
  thirteen it covers and what makes a model a variant rather than a base.

**5. R15 draws two single-panel figures and both are lettered a).** `scripts/figlib_figures.py:957`
hard-codes it, so the re-run reproduces it. Not in these blocks, but it is a paste-time defect wherever
R15 lands.

---

# §R4 and §R5 — replacement text

Two Results blocks, raw LaTeX, ready to paste into `PAPER_REVISION_GUIDE_FINAL.md` over the
existing §R4 (guide line 1618) and §R5 (guide line 1763). Figure blocks are included where the
text points at them. The author-notes block at the end is not paper text.

---

## §R4. The kind of noise *(replaces `paper.tex:460–467` and has no real predecessor)*

```latex
\subsection{The kind of noise, at a matched amount}

Every noise result so far was taken one kind of noise at a time. None of them says whether the kind
of noise itself changes the answer. We asked for error equal to half the spread of the clean labels.
The six conditions that can be held to a set amount delivered between 0.303 and 0.309 of a requested
0.308, in label units. Censoring cannot be held to a set amount, since its level counts the fraction
of labels clipped rather than a fraction of the label spread. Where two of the six matched
conditions differ in outcome, the difference is one of pattern and not of amount.

Changing the shape of one label's error made no difference we could measure. On the HOMO--LUMO gap
at ECFP4, seven models ran Gaussian, Laplace, Student-$t$ and outlier noise
(Table~\ref{tab:robustness}). None of the three other shapes moved the median \mbox{AUC$_{norm}$}
of those seven models by as much as a hundredth against Gaussian noise. No difference among them
reached significance, and each is smaller than the disagreement between ten replicates that differ
only in seed. \citet{Heid2023} reported no difference between uniform, hyperbolic and bimodal errors
drawn at a matched standard deviation. Their error was spread evenly over the training labels, while
ours fell on a tenth of the training molecules as well. The answer did not change in either case.
Little rests on which shape of error is injected, provided the amount is matched.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F4b_rf_across_noise_conditions.png}
\caption{Predictive accuracy of the random forest under each noise condition, one line per
condition, on the QM9 HOMO--LUMO gap at ECFP4. The bottom axis is the noise added to the training
labels, as a fraction of their clean spread. The side axis is median held-out R$^2$ over ten
replicates. Censoring has no line, as it cannot share this bottom axis.}
\label{fig:conditions}
\end{figure}

The two conditions that do separate break a different assumption. Grouped-shifted gives every
scaffold family one offset of its own, added to every label in that family. On the HOMO--LUMO gap at
ECFP4 it moved the median \mbox{AUC$_{norm}$} of nineteen models from 0.926 under Gaussian noise to
0.905. That is the largest move of the fifteen comparisons between conditions at that
representation. Grouped-shifted is in five of the six comparisons that reached significance.
Grouped-wider widens a scaffold family's errors without moving its mean, and it moved the same
median from 0.926 to 0.927. One line in Figure~\ref{fig:conditions} is the random forest under one
condition, drawn against the amount of noise added to the training labels. The grouped-shifted line
is the one that falls away, while the other five lie on one another. Molecules in one scaffold
family hold 0.78 of grouped-shifted's injected error in common, against 0.14 under Gaussian noise
and 0.11 under grouped-wider (Figure~\ref{fig:noise_conditions}). These conditions separate on
independent, zero-mean error against correlated or shifted error, rather than on Gaussian against
non-Gaussian error.

Clipping labels at an assay limit cost more \mbox{AUC$_{norm}$} than any of the six matched
conditions. Censoring ran on five named pairs of model and representation, and on four of them on
hERG $K_i$. It scored below that same pair's \mbox{AUC$_{norm}$} under Gaussian noise on the
HOMO--LUMO gap and on each of the three assay endpoints. Caco-2 is where it costs most: the five
pairs there run from 0.198 to 0.492 under censoring, against 0.815 to 0.865 under Gaussian noise.
Those pairs were chosen to measure the size of the effect, so nothing here says which model resists
clipping best. Clipping removes information from a label rather than adding error to it, which may
be why it costs more than any amount of ordinary scatter.

Model architecture decides how much a model loses to a shared offset far more than molecular
representation does. Reading each model as the median of its six representations, seventeen of
nineteen models lost \mbox{AUC$_{norm}$} under grouped-shifted against Gaussian noise on the
HOMO--LUMO gap. The same count was fifteen of nineteen models on logD and on Caco-2, and eighteen of
nineteen on hERG $K_i$. Across the nineteen models that loss spans 0.055 of \mbox{AUC$_{norm}$} on
the HOMO--LUMO gap and 0.225 on hERG $K_i$. Across the six representations, each read as the median
over the nineteen models, it spans 0.026 and 0.043 of \mbox{AUC$_{norm}$} on those same two
datasets. Grouped-shifted is also the condition whose replicates disagree most, at 0.093 of
\mbox{AUC$_{norm}$} for the random forest at ECFP4, against 0.012 to 0.018 under the other five
conditions. This suggests that model architecture, rather than molecular representation, is the
choice to weigh once the errors are shared within a scaffold family.

The kind of noise does not change which model to choose. On the HOMO--LUMO gap at ECFP4 the model
rankings agree across the six matched conditions (Kendall's $W = 0.937$, where 1 is complete
agreement and 0 is none). That agreement rests on the seven models that ran every condition at one
representation, so it does not speak for the other twelve. A model chosen under one condition keeps
its standing under the others. How much accuracy that model loses is set by the pattern of the noise
rather than by its shape.
```

---

## §R5. Does it hold on measured labels *(replaces `paper.tex:542–556`)*

```latex
\subsection{Robustness on the three assay datasets}

Every result so far comes from a computed property, whose labels carry no measurement error of their
own. We repeated the same grid of model architectures, molecular representations and noise
conditions on three assay endpoints: a distribution coefficient, an efflux ratio and a binding
affinity. On those endpoints the injected error sits on top of measurement error already in the
labels, which we cannot separate from it. Repeat measurements of the same compound and target
disagree by about 0.54 log units for pK$_i$ \citep{Kalliokoski2013, Kramer2012}. The hERG $K_i$
labels have a spread of 0.915 log units, so a noise level of 0.6 of that spread is about one unit of
that laboratory error. Each model's \mbox{AUC$_{norm}$} under each noise condition on the three
endpoints is given with its clean R$^2$ beside it (Figure~\ref{fig:assay}). \mbox{AUC$_{norm}$} is a
share of a model's own clean R$^2$, so neither number means much without the other.

\begin{figure*}[p]
\centering
\includegraphics[width=\textwidth]{F8_assay_datasets_logd.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_caco2.png}\\[2pt]
\includegraphics[width=\textwidth]{F8_assay_datasets_herg.png}
\caption{Robustness (\mbox{AUC$_{norm}$}) on the three assay datasets, on the ECFP4 representation,
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

The ordering of the noise conditions holds on measured labels. On the HOMO--LUMO gap,
grouped-shifted had the lowest \mbox{AUC$_{norm}$} of the six matched conditions on 19 of the 21
model--representation pairs that ran all six. On the assay endpoints it was lowest on 15 of 19 pairs
on logD, 13 of 17 on Caco-2 and 15 of 16 on hERG $K_i$. Caco-2 is where it holds least, and four of
those seventeen pairs put some other condition lowest. A ranking of noise conditions established on
a computed property carries to an assay endpoint.

What does not carry is the ranking of the models themselves. The two rankings were compared at 83
combinations of molecular representation, noise condition and assay dataset (a rank correlation of 1
is the same ordering on both). The two rankings agree at the five per cent level on 16 of the 28
combinations on logD. On Caco-2 that is 2 of 28 combinations, and on hERG $K_i$ 2 of 27. NGBoost is
first of nineteen on the HOMO--LUMO gap at ECFP4 under Gaussian noise, and third, fourth and seventh
on logD, Caco-2 and hERG $K_i$ (Figure~\ref{fig:transfer}). LightGBM is sixth there and nineteenth
of nineteen on both Caco-2 and hERG $K_i$. VBLL-$\alpha$ is tenth on the HOMO--LUMO gap and first on
both of those two endpoints. A study run on the computed property alone would have reported LightGBM
as a robust model.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{R9_rank_transfer_ecfp4.png}
\caption{Where each model ranks for robustness on the QM9 HOMO--LUMO gap and on each of the three
assay datasets, at ECFP4 under Gaussian noise. Rows are models, ordered by their rank on the
HOMO--LUMO gap with the most robust at the top. The bottom axis is the rank by
\mbox{AUC$_{norm}$} within a dataset, where 1 is the most robust of the nineteen models ranked.
Colour and shape together name the dataset. The thirteen base models are drawn and the six variant
models are not, so some ranks on the axis belong to a model that has no row.}
\label{fig:transfer}
\end{figure}

The model family is what carries, rather than the pairing of a model with a representation. The two
bagged forests, the plain forest and the quantile forest, are within the top five of nineteen models
on all four datasets. The plain forest ranks second on the HOMO--LUMO gap, first on logD, fifth on
Caco-2 and third on hERG $K_i$. The boosted trees do not carry: XGBoost ranks fourth of nineteen on
the HOMO--LUMO gap and fourteenth and fifteenth on Caco-2 and hERG $K_i$. The rank correlation also
depends on which molecular representation a model is paired with, so it is read one representation
at a time. All but nine of the 117 cells in Figure~\ref{fig:assay} are a median over five scaffold
folds, and those nine rest on three or four. Their remaining folds fell below the clean-accuracy
gate. Checking a pairing on an assay endpoint means running that grid on that endpoint. This
suggests that a model family can be chosen on the computed property, while the pairing is better
checked on the endpoint it will be used on.
```

---

## Author notes — not paper text

**Lengths, counted on the text between the figure blocks.** §R4 is 797 words in six paragraphs of
6, 7, 7, 6, 7 and 6 sentences. §R5 is 563 words in four paragraphs of 7, 5, 7 and 7 sentences. §R4
runs a median sentence of 21 words and §R5 of 23, the longest in either is 28 and the shortest is
10, so no sentence breaks either the 30-word ceiling or the five-word floor. Both blocks grew in
this pass, §R4 by 44 words and §R5 by 13, and that is 97 and 63 over their targets. The growth is
four sentences: a consequence ending for two paragraphs that ended on a measurement, the scope
clause on the per-dataset counts, and the split of one 33-word sentence. The two sentences I would
cut first are §R4's Heid comparison and its sentence on why clipping may cost more, and both are
there because a house-style rule asks for them.

🔴 **One rule I could not meet, and the fix is a decision.** House style caps Results sentences
carrying a decimal at about 12 per cent. §R4 runs 9 of 39 sentences, which is 23 per cent, and §R5
runs 2 of 26, which is 8. Four decimal sentences came out on the way here: the three shape medians
became "none of the three other shapes moved the median by as much as a hundredth" with a pointer
to Table~\ref{tab:robustness}, the 0.028 became "that is the largest move" because the pair above it
is printed, the three censoring medians became one count and one range, and the pooled 0.36 became
a count. Getting §R4 to about 12 per cent means moving five more things into tables: the delivered
amounts, the group shares, the censoring range, and the two per-dataset spans. The cost is that the
paper's differentiator then has no number of its own in the running text, which Landrum's noise
paper does not do either. **Yours to call.**

**One sentence from the old §R4 is gone.** "This is the ordering of §\ref{sec:variance}, arrived at
on a different outcome and on a different set of runs." It restated §R1's finding, and cutting it
bought 20 words. If you want the cross-reference back it goes at the end of the fifth paragraph.

**(a) The 0.127 and the 0.095 are gone.** Both were a median taken across the computed property and
the three assay datasets together, which house-style rules 11 and 12 forbid. They are replaced by
two things, both per dataset, in §R4's fifth paragraph. A count: 17 of 19 models lose
\mbox{AUC$_{norm}$} to the shared offset on the HOMO--LUMO gap, 15 of 19 on logD, 15 of 19 on
Caco-2, 18 of 19 on hERG K$_i$. A span, per dataset, of how much each of the 19 loses: 0.055
HOMO--LUMO gap, 0.163 logD, 0.251 Caco-2, 0.225 hERG K$_i$, against 0.026, 0.019, 0.036 and 0.043
for the same quantity read across the six representations. Computed this session from
`auc_norm_qm9.csv` and `auc_norm_assay.csv`: per model, the median over replicates of
\mbox{AUC$_{norm}$} under grouped-shifted minus \mbox{AUC$_{norm}$} under Gaussian, then the range
across models within one dataset. The text quotes only the HOMO--LUMO gap and hERG K$_i$ spans; the
other two belong in a table if you want all four.

🔴 **Those counts and spans still average over the six representations, and rules 11 and 12 forbid
that too.** Each model's value is the median of its six representations before the count is taken;
I re-ran it this session and the counts and spans only come out at 17/15/15/18 and 0.055/0.225 that
way. I have written the pooling into the sentence rather than hiding it, and left the numbers
alone. The per-representation replacement, at ECFP4, which is the representation the rest of §R4 is
read at: **18 of 19 models lose on the HOMO--LUMO gap, 15 of 19 on logD, 14 of 19 on Caco-2, 18 of
19 on hERG K$_i$; the spans are 0.054, 0.183, 0.293 and 0.287.** From `auc_norm_qm9.csv` and
`auc_norm_assay.csv`, `rep=ecfp4`, `condition` in `gaussian` and `grouped_shifted`, this session.
Swapping those six numbers in is a two-line edit and it takes the block clear of the rule. **Yours
to call, because it changes reported values.**

**The old worst-and-best pair is also gone.** "$-0.101$ for the Gaussian process to $+0.025$ for the
heteroscedastic variational network" was the same cross-dataset median. Per dataset the worst is a
different model every time: NN-$\beta$ at $-0.053$ on the HOMO--LUMO gap, the Gaussian process at
$-0.091$ on logD, the quantile forest at $-0.172$ on Caco-2, BNN-$\beta$ at $-0.210$ on hERG K$_i$.
That is itself a finding and it is not in the text — say the word and it becomes one sentence in the
fifth paragraph.

**(b) The matched-settings re-run, commit `85bc2dc`.** Four of the five models are in these numbers
and one is not.

- `dnn`, `dnn_bnn_full`, `mlp` and `mlp_bnn_full` are rows in `auc_norm_qm9.csv` and
  `auc_norm_assay.csv`. They are inside every 19-model and 13-model count in both blocks: §R4's
  0.926-to-0.905 move, the 17/15/15/18-of-19 counts, the per-dataset spans, and all of §R5's
  ranks and the 83 rank comparisons. 🔴 **TODO — recompute both blocks after the re-run.** The file
  that settles it is `auc_norm_qm9.csv` and `auc_norm_assay.csv` from the next harvest.
- `rf300` is in none of them, because it did not exist when this harvest was written. It will add a
  twentieth row, so every "of nineteen" count in both blocks changes to "of twenty".
- **Three numbers are clear of the re-run.** Kendall's $W$ = 0.937 covers seven models
  (`d3_kendall_w.csv`: `dnn_bnn_full_mve`, `dnn_vbll_hetero`, `gauche_rbf`, `het_gp_rbf`, `ngboost`,
  `rf`, `svm`), none of the five. The Laplace, Student-$t$ and outlier medians in §R4's second
  paragraph cover the same seven. The censoring pairs are `rf`/ECFP4, `het_gp_rbf`/ECFP4,
  `gauche_rbf`/PDV, `ngboost`/PDV and `dnn_bnn_full_mve`/PDV, none of the five.

**Every number, and the file it came from.** All in `results/decisions_arc_20260916/`, all opened
this session.

| number | file |
|---|---|
| asked 0.308, delivered 0.303–0.309, worst off by 0.005, censoring 0.141 | `figures/F1_delivered_dose.csv` |
| ⚠️ that row is measured on 2,000 constructed labels, not on a dataset | `f1_noise_conditions` in `scripts/figlib_figures.py`, read this session |
| group share 0.78, 0.14, 0.11 | `figures/F1_group_share.csv` |
| Laplace $-0.002$, Student-$t$ $+0.002$, outlier $+0.001$, none significant, 7 models | `d3_condition_pairs.csv` |
| 0.926 → 0.905, 19 models, $p = 7.6\times10^{-6}$; 0.926 → 0.927 for grouped-wider; 6 of 15 significant | `d3_condition_pairs.csv` |
| RF at ECFP4, replicate spread 0.093 grouped-shifted against 0.012–0.018 | `auc_norm_qm9.csv` |
| Kendall's $W$ 0.937, 7 models, 6 conditions, ECFP4 | `d3_kendall_w.csv` |
| censoring 0.198–0.492 against Gaussian 0.815–0.865 on Caco-2, and below Gaussian on all 19 pair-and-dataset combinations that ran | `auc_norm_qm9.csv`, `auc_norm_assay.csv`, paired on shared pairs |
| 17/15/15/18 of 19, and the four spans | `auc_norm_qm9.csv`, `auc_norm_assay.csv` |
| 19 of 21, 15 of 19, 13 of 17, 15 of 16 pairs with grouped-shifted lowest | `auc_norm_qm9.csv`, `auc_norm_assay.csv` |
| 16 of 28 on logD, 2 of 28 on Caco-2, 2 of 27 on hERG K$_i$ significant, 83 combinations in all | `d9_rank_agreement.csv` |
| every rank in §R5, including VBLL-$\alpha$ at 10, 1 and 1 | `d9_rank_transfer.csv`, `rep=ecfp4`, `condition=gaussian` |
| 9 of 117 cells on three or four folds | `auc_norm_assay.csv` (`n_replicates`), reasons in `excluded_assay.csv` |

**What the figure readings changed in the captions.**

- F8: "the second lowest clean R$^2$ on that panel" was wrong and is now "the lowest". VBLL-$\alpha$
  is at 0.358 on Caco-2 and the next lowest of the thirteen is 0.375.
- F8: "Each value is the median over the five scaffold folds" and "one fit per cell with the seed
  pinned" contradicted each other. Both are replaced by the median over the folds that met the gate,
  with the nine thin cells named.
- F8: "every model" is now "the thirteen base models", because the figure draws 13 of the 19 rows
  the file holds.
- F8: the grey-cell sentence is cut. No cell on any of the three panels is grey.
- F8: "Caco-2 efflux" is now "Caco-2", which is what the drawing prints and what nine other figures
  use. 🔴 If you want "efflux" in the paper it has to go into `DATASET_LABELS` in
  `scripts/figlib_config.py`, not just the caption.
- R9 and F4b had no caption block anywhere in the guide. Both are written above from the drawings.
- F4b's caption says the key is beside the panel and the drawing on disk has it below, because
  `93fe179` moved it after the harvest. The caption describes what the re-run draws.

🔴 **F4b goes in the paper in this draft, and your 2026-09-18 call was that it stays an additional
file.** I put it in because §R4 is the subsection carrying the paper's differentiator and it was the
only Results subsection with no figure, which is the guide's own argument at its figure-promotion
table. If you keep the additional-file call, the second paragraph's last sentence becomes "Every
condition is given model by model in Table~\ref{tab:robustness}", the two pointers to
Figure~\ref{fig:conditions} come out of the second and third paragraphs, and the `\begin{figure}`
block above is deleted. Nothing else in §R4 moves.
- R9's caption carries the one thing a reader cannot see: the ranks were computed over all nineteen
  models and only thirteen are drawn, so the axis reaches 19 and six ranks have no row.

**R19 is not cited in either block, and here is why.** On the drawing on disk, panels a) and b) have
their column labels shifted by one: the column labelled Student-$t$ holds censoring, the one
labelled outlier holds Student-$t$, the one labelled Laplace holds outlier, and the real Laplace
column is drawn outside the grid. A reader of that PNG reads the random forest's censoring value as
its Student-$t$ value. `scripts/figlib_figures.py:1867` records the fix, so the re-run corrects it.
The Laplace, Student-$t$ and outlier evidence in §R4's second paragraph comes from
`d3_condition_pairs.csv` instead, which is what T4 carries.

**R15 is not cited either.** §R4's last paragraph carries Kendall's $W$ alone, which is the guide's
own plan. Note that the two are not the same claim: Kendall's $W$ is agreement across conditions,
and R15's bottom axis is the noise level. The drawings show the ranking moving a good deal along the
axis R15 holds — the plain forest goes from rank 8 on clean labels to rank 1 at level 1.5. If you
want a sentence about that it needs its own figure, and R15 as drawn cannot back the "barely moves"
wording in the guide.

🔴 **The delivered amount is a property of the injector, not a measurement on any dataset.**
`f1_noise_conditions` builds a three-mode distribution of 2,000 labels and doses every condition
against its spread, so 0.303 to 0.309 of a requested 0.308 says the injector honours the request and
not that QM9 and the three assay sets each received the same amount. §R4's first paragraph is
written so that it does not claim more than that, but a referee will ask. One sentence in Methods
saying the check was run on a constructed distribution would close it, and running the check on each
dataset's own labels would close it better, at the cost of one local script and no cluster time.

**paper.tex:560's "QRF consistently less robust" stays wrong.** The quantile forest is ahead of the
plain forest on Caco-2 and hERG K$_i$ under Gaussian noise at ECFP4, at 0.903 against 0.865 and
0.902 against 0.898. §R5's fourth paragraph now treats the two as one family instead, and names both
of them.

### What the house-style pass changed, on top of the above

- **Censoring, §R4's fourth paragraph.** The three medians — 0.856 to 0.459 on Caco-2, 0.863 to
  0.555 on hERG K$_i$, 0.951 to 0.813 on the HOMO--LUMO gap — were each a median across the two
  representations the five pairs sit on, ECFP4 and PDV. Rule 11 forbids a number averaged across
  representations. They are replaced by a count, which rule 11 does allow: censoring scored below
  that same pair's Gaussian \mbox{AUC$_{norm}$} on **all 19** pair-and-dataset combinations that
  ran, and by the Caco-2 range, 0.198 to 0.492 against 0.815 to 0.865. Computed this session from
  `auc_norm_qm9.csv` and `auc_norm_assay.csv`. **logD was missing from the old sentence** and the
  count now covers it.
- **The median rank correlation of 0.36, §R5's third paragraph, is gone.** It was a median over all
  83 combinations, which pools the three assay datasets and the six representations at once. It is
  replaced by the same 20 significant combinations split per dataset: 16 of 28 on logD, 2 of 28 on
  Caco-2, 2 of 27 on hERG K$_i$, from `d9_rank_agreement.csv` this session. 16 + 2 + 2 = 20, so no
  value changed, only the scope it is reported at.
- **The 0.60 at ECFP4 and 0.00 at Sort \& Slice, §R5's fourth paragraph, are gone** for the same
  reason: each was a median over the three assay datasets within one representation. The sentence
  now states the dependence without a number. 🔴 **If you want a number back there it has to be one
  representation on one dataset, and I have not chosen which — yours to call.**
- **"the first variational network" is now VBLL-$\alpha$**, which is what `MODEL_LABELS` in
  `scripts/figlib_config.py` prints on the figure. The model is `dnn_vbll`: rank 10 on the
  HOMO--LUMO gap, rank 1 on Caco-2 and rank 1 on hERG K$_i$ at ECFP4 under Gaussian noise
  (`d9_rank_transfer.csv`), and clean R$^2$ 0.358 on Caco-2, the lowest of the thirteen base models
  (`auc_norm_assay.csv`). A reader could not find a row called "the first variational network".
- **"configurations" is now "models"** in the fifth paragraph. `d3_condition_pairs.csv` pairs on
  `model` and the count is 19 models, the same 19 the rest of both blocks ranks.
- **Coined nouns removed:** "that paired fall" (now "that is the largest move", with both endpoints
  still printed above it), "the axis along which these conditions separate" (now "what separates
  these conditions"), "Budget the accuracy it will lose" (now folded into the one recommendation
  sentence that ends §R4), "the rest of the roster" (now "the other twelve"), "concordance" (now
  "agreement"), "the whole model roster" in the F8 caption (now "every model").
- **Things arriving pre-known are introduced:** "the grid" is now "the same grid of models,
  representations and noise conditions"; "the pairing" is now "the pairing of model with
  representation"; "the clean-accuracy gate" is glossed in the F8 caption as the lowest clean R$^2$
  a fold may have and still be scored; "a contaminated tenth" is now "a tenth of the training
  molecules", which says what the tenth counts.
- **Paragraph endings.** §R4's second paragraph ended on a figure pointer and its fifth on a
  measurement, and §R5's first ended on a description of the figure's layout. All three now end on
  a consequence sentence, which is the rule the house style checks by reading last sentences alone.
- **Captions.** Five caption sentences ran past 30 words and are split. The F8 caption's "coloured
  on the bar, whose bright end" was ungrammatical and is now two sentences. R9's caption gained one
  sentence saying each rank comes from a single median value with no spread drawn, which is the
  replicate question every other caption answers.

🔴 **Two claims I did not touch, because fixing them means changing a number or a finding.**

1. **"That is the largest move of the fifteen comparisons between conditions at that
   representation."** `d3_condition_pairs.csv` gives fifteen comparisons at ECFP4, and the
   Gaussian-to-grouped-shifted move of $-0.0276$ is **not** the largest of them. Grouped-shifted
   against Student-$t$ is $+0.0355$, against outlier $+0.0326$ and against Laplace $+0.0306$, all
   larger. Those three are medians over seven models rather than nineteen, so they may be what the
   sentence means to exclude — but as written the claim is wrong. It is the largest move **against
   Gaussian noise**, and adding those two words fixes it if that is what was meant.
2. **Figure~\ref{fig:noise_conditions} has no figure block in this file and no caption anywhere in
   the guide**, unlike F4b, F8 and R9. The group shares 0.78, 0.14 and 0.11 point at it. Either the
   block exists elsewhere in `PAPER_REVISION_GUIDE_FINAL.md` under that label, or the reference is
   dangling and the shares need a home.

---

# §R6 and §R7 — replacement text

Written 2026-09-21. §R6 replaces `paper.tex:498–541`. §R7 is new and has no counterpart in the
submitted paper. Both are raw LaTeX and paste straight in. Every number below was traced this run to a
named file in `results/decisions_arc_20260916/`; the traces are in the author-notes block at the end.

---

### §R6. Uncertainty under label noise *(replaces `paper.tex:498–541`)*

```latex
\subsection{Uncertainty under label noise}

Everything to this point measures how much accuracy a model keeps when its training labels are
corrupted. How precise a model reports itself to be is a separate question. Every uncertainty
reported here is computed from the fitted model at a molecule's representation, never from the label
that molecule is scored against. None of it is calibrated after fitting. One result runs against
expectation. A model can grow more certain as its labels grow more wrong, and censoring is the one
noise condition where that happens.

\citet{Kolmar2021} found that a Gaussian process's mean predicted uncertainty rises with the label
noise in its training data. Six of the seven noise conditions add error to a label, and that rise
holds under all six. It holds on the QM9 HOMO--LUMO gap and on each of the three assay datasets
taken separately. Within each dataset, between 92 and 100 per cent of the combinations of model,
representation and fold give a rising slope. The median slope over those same combinations runs from
$+0.30$ to $+0.42$ label units of predicted uncertainty per unit of noise level. That reproduces
Kolmar's result on a computed quantum property and on three assay endpoints.

Censoring is the seventh condition, and under it the sign reverses on every model, on the QM9
HOMO--LUMO gap and on all three assay datasets. Of those same combinations, fewer than 5 per cent
rise. Censoring replaces every measurement beyond the limit of the assay with that limit, so it
removes information from a label instead of adding error to it. It is also the only one of the seven
conditions that narrows the spread of the training labels. Pulling the training labels together may
leave a model fitted to them reporting itself more certain. The runs recorded only the spread of the
clean training labels, so that mechanism was not measured here.

Some models report their uncertainty in two parts, the part attributed to the labels and the part
attributed to the model itself. Six of the thirteen models that report an uncertainty can be asked
for that division. Whether each part varies from molecule to molecule or is a single number for the
whole fit is given per model and noise condition (Table~\ref{tab:uncertainty}). NGBoost fits one
distribution and carries no separate term for the part it would attribute to itself. The two plain
Gaussian processes and the two plain variational networks report one observation-noise number per
fit, and the two plain Bayesian networks predict a mean alone. Those seven models are named rather
than scored, because a slope drawn through a constant describes the fit and not the molecules.

Both parts are drawn against the amount of noise added to the training labels, on the QM9
HOMO--LUMO gap at ECFP4 under Gaussian noise (Figure~\ref{fig:decomposition}). Only the Bayesian
network with a variance output head separates them, raising the part it attributes to its labels
while the other stays flat. The quantile forest raises both parts together, which is what one
bootstrap driving both of them would look like. The heteroscedastic Gaussian process starts with the
larger value on the part it attributes to itself and ends with the larger value on the other.
Reporting two parts is not the same as separating them, and only one of the three models drawn does
both.

The same separation can be counted rather than read off one figure. Across the QM9 HOMO--LUMO gap
and the three assay datasets, 117 combinations of dataset, model and representation have both parts
varying from molecule to molecule. The part attributed to the labels rises while the other holds in
78 of those 117 under Gaussian noise. It does so in 58 of the 117 when a whole scaffold family
shares an offset. Both parts rise together in 31 of the 117 under Gaussian noise and in 52 of the
117 under the shared offset. Treating label noise as independent across molecules overstates how
often a model can separate the two parts of its uncertainty.

Ranking predictions by how wrong they are asks less of a model than separating the two parts, and
almost every model manages it. Within one fixed amount of noise, 1,003 combinations of dataset,
model, representation and noise condition carry a named condition and can be scored. Those
combinations span the QM9 HOMO--LUMO gap and the three assay datasets. Of those 1,003 combinations,
95 per cent give a positive Spearman correlation between a model's total predicted uncertainty and
its out-of-fold error (higher values indicate better ordering). That correlation is above 0.3 in 5
per cent of the same 1,003 combinations. Taken as a median across those datasets, representations
and conditions, the two plain Bayesian networks sit lowest of the thirteen models, at 0.09 and 0.07.
A positive correlation between predicted uncertainty and error is the floor for an uncertainty
estimate rather than evidence that the estimate is informative.

The quantile forest is where those two abilities sit furthest apart. It holds the highest median
correlation of the thirteen models, again taken across datasets, representations and conditions. The
separation of its two parts fails in all 99 combinations that can be read. Its \mbox{AUC$_{norm}$}
also sits below that of the plain forest it is built from. That holds in 13 of the 18 combinations
of representation and noise condition on the QM9 HOMO--LUMO gap. A Bayesian network with a variance
output head separates the two parts, and it costs clean R$^2$ to use one. The heteroscedastic
Gaussian process reaches the higher clean R$^2$ in 111 of the 113 combinations of assay dataset,
representation and noise condition where both were run. Where the two parts have to be read apart, a
variance-head Bayesian network is preferable to a forest, at that cost in clean R$^2$.
```

**964 words, eight paragraphs of 5, 6, 6, 6, 5, 6, 7 and 7 sentences, running 86 to 149 words each.
Median sentence 20 words, longest 30, shortest 7. 3 of 48 sentences carry a decimal, which is 6 per
cent. Longer than the 700 asked for, and longer than the draft before this one, because seven
sentences that ran past thirty words were split rather than cut and two paragraphs gained a closing
sentence a reader can act on. The paragraph I would cut first is still the fifth, which reads the
figure, and cutting it brings the block to 853 words.**

---

### F6 — the figure §R6 carries

Corrected against the PNG on disk (`results/decisions_arc_20260916/figures/F6_decomposition.png`,
written 17 September 03:45). Three changes from the block in the guide's `THE FIGURES` section: the
panels are named, because the drawing has three and the guide's "one panel per model" reads as all of
them; the pointer to Table~\ref{tab:uncertainty} no longer claims to list every model that is not
drawn, because it does not; and the fourth model that could have been drawn is named as not drawn
(see the author notes).

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{F6_decomposition.png}
\caption{Two parts of the predicted uncertainty against the amount of noise added to the training
labels, on the QM9 HOMO--LUMO gap at ECFP4 under Gaussian noise. Orange is the part a model
attributes to its labels and blue the part it attributes to itself. The three panels are the
quantile forest, the heteroscedastic Gaussian process and the Bayesian network with a variance
output head. The bottom axis is the noise level, as a fraction of the spread of the clean training
labels. The side axis is the mean predicted uncertainty over one out-of-fold pass, in eV, on a scale
set per panel. One mark is the median over the scored folds and the band spans the lowest fold to
the highest. A rising orange line under a flat blue one is the separation being looked for, and two
lines climbing together is its failure. A part that is one number per fit is not drawn.
Table~\ref{tab:uncertainty} says for every model whether each part varies from molecule to molecule
or is one number per fit. A fourth model whose two parts both vary from molecule to molecule, the
second Bayesian network with a variance output head, is also not drawn.}
\label{fig:decomposition}
\end{figure}
```

*199 words in ten sentences, against the 70–130 that Kolmar's and Dablander's captions run, because
this one carries the per-fit rule and the dropped fourth model as well as the five things every
caption answers. The sentence that could go is the one about the marks and the band, which
Table~\ref{tab:uncertainty} also states.*

---

### §R7. Whether uncertainty identifies the corrupted labels

```latex
\subsection{Whether uncertainty identifies the corrupted labels}

Ranking predictions by how wrong they are is not the same as naming which labels were corrupted.
Under four of the seven noise conditions every molecule receives the same noise scale, so there is
nothing to find by construction. The question is defined only where some molecules are corrupted
more than others: widening a scaffold family's errors, a larger draw for a random tenth of the
labels, and censoring. The correlation between a molecule's predicted uncertainty and the size of
the error injected into its label is near zero under the six conditions that add error. That holds
on the QM9 HOMO--LUMO gap and on each of the three assay datasets. A model could see the injected
error through the label it trained on. That near-zero correlation is the check against it. No
scoring model ever saw the draw made for the molecule it scores, so the reading here is a working
control rather than a null result.

We asked whether dividing a molecule's out-of-fold error by its predicted uncertainty ranks the
corrupted labels better than the error alone does. Under the six conditions that add error, the
error alone ranks them at a Spearman correlation of $+0.32$ to $+0.40$. Each of those two figures is
a median within one condition, taken across datasets, models and representations. Dividing by the
predicted uncertainty moves that correlation by less than 0.002 in every one of the six. Censoring
is the exception on both counts. Under censoring, the error alone ranks the censored labels at
$-0.02$ in Spearman correlation and the ratio at $+0.01$. That difference is an order of magnitude
larger than the change under any of the other six conditions. Both of those numbers are near zero,
so censoring gives a difference between two rankings that both fail.

That direction is reported with no test behind it. The band of correlations from shuffled labels
that it would have to beat was not computed on these runs. That band is not centred on zero, so the
size of the difference cannot be judged against zero by eye. Censoring reached only three pairings
of model and representation on the QM9 HOMO--LUMO gap, against seven models at three representations
on each assay dataset. Censoring is also the condition where predicted uncertainty tracks
out-of-fold error least well. It is the one condition where predicted uncertainty runs mildly
against the injected corruption rather than with it. That pair of results has no explanation in
these runs. A model's own uncertainty is not a way to find the corrupted labels in a training set.
This suggests that a training set is better audited by sorting on prediction error, with the
predicted uncertainty as a tie-break at most.
```

**453 words, three paragraphs of 7, 8 and 8 sentences, running 144 to 161 words each. Median sentence
20 words, longest 30, shortest 9. 4 of 23 sentences carry a decimal, which is 17 per cent — above the
12 per cent house style sets, and the two sentences to drop if you want it under are the one giving
the $+0.32$ to $+0.40$ range and the one saying what those two figures are a median of.**

---

## AUTHOR NOTES

### 1. F7 does not exist on disk, and a re-run will not make it

`results/decisions_arc_20260916/figures/notes_for_the_text.md` ends on the line you quoted, calling
the censoring enrichment curve the strongest possible answer to the paper's title. There is no
`F7_enrichment.png` in that directory, nor `F7_error_retention.png`, nor `F7_uncertainty_grid.png`. I
listed the directory this run: 21 PNGs, none of them an F7.

The drawing code is not missing. `scripts/figlib_figures.py:1260` defines `f7_uncertainty` and
`:1359` defines the enrichment curve itself, and `scripts/test_figure_slots.py:106` still exercises
both. What is missing is the call. `scripts/run_paper_analysis.py:546` is `_draw_uncertainty`, and its
docstring reads **"F6 only. F7 IS DROPPED -- the author's call, 2026-09-13."** The function draws F6
and returns. So the note in `notes_for_the_text.md` is written by the decision step, which still runs
and still fires, while the drawing step it names was removed three days earlier.

Three things have to happen to get that PNG, and the first is yours:

1. **Your word to reinstate it.** The reasons recorded for the cut on 2026-09-13 are in that
   docstring: the bottom axis described a person auditing a dataset, NGBoost lay on the random
   diagonal past 60 per cent because its model-doubt half is one number per fit, and the same finding
   was already a number in T6.
2. **One edit in `_draw_uncertainty`**, calling `FIG.f7_uncertainty('7B', out, rep, condition,
   enrichment=tables['_unc_enrichment'], q4=q4)`. The enrichment table is already built in the same
   streaming pass and carried in memory under `_unc_enrichment`
   (`scripts/run_paper_analysis.py:322`), and the leading underscore keeps it out of the CSV sweep on
   purpose. It is not in the harvest, so the figure cannot be drawn from what is on disk — it needs
   the analysis to run.
3. **The analysis re-run**, which is one submission and about half an hour:

   ```bash
   cd $QSAR && git pull && sbatch slurm_scripts_analysis/run_paper_analysis.sh
   ```

Optionally the enrichment band as well. `figlib_uncertainty.py:132` sets `DEFAULT_NULLS` and leaves
`delta_auc` out of it; `RERUN_PLAN.md` §14.6 row 1 names that statistic as this figure's trigger. Pass
`nulls=DEFAULT_NULLS + (('delta_auc', 'aucdelta_'),)` to compute it, at the cost of another
permutation pass over every cell.

### 2. The note that calls F7 the strongest answer was computed off the wrong band

This is the reason I did not put the censoring enrichment result in §R7 as a finding.

`d7_q4.csv` in this harvest has no `adds_signal` column. `figlib_uncertainty.py:155–205` records that
until 2026-09-17 `outside_null` carried the band for the correlation between out-of-fold *error* and
injected noise, with the uncertainty nowhere in it, and that under censoring a clipped label simply is
a large error. `figlib_decisions.py:959–968` reads `adds_signal` and falls back to `outside_null`,
printing a warning, which is what happened here.

Two counts I made on `d7_q4.csv` this run, both against that wrong band:

- 2,417 rows of 20,699 have no observed value and no band at all, and every one of them is recorded as
  firing. 345 of those are censoring rows. `figlib_uncertainty.py:196–201` is the fix and it requires
  both ends of the comparison to exist.
- Among the rows that do carry both, censoring fires in 1,925 of 2,070 against 3.4 to 23.3 per cent
  for the other six conditions. **Do not quote that 93 per cent.** It is the error band, not the
  uncertainty's.

What is safe to quote, and what §R7 uses, is the difference itself, which needs no band to be printed:
median `rho_error`, `rho_ratio` and `rho_delta` by condition from `d7_q4.csv`. Under censoring they are
$-0.024$, $+0.006$ and $+0.014$. Under the other six, `rho_error` is $+0.32$ to $+0.40$ and
`rho_delta` is $+0.0001$ to $+0.0018$.

`STATS_CACHE_GENERATION` is already bumped to 1 (`figlib_uncertainty.py:471`), so the re-run will not
hit the old cache.

### 3. The two fractions you asked me to verify

The guide's §R6 says the split separates "in 78 of the 109 comparisons that can be read at all under
Gaussian noise, and in 58 of 110 under grouped-shifted". Recomputed from `unc_slopes.csv` on the 237
combinations of dataset, model and representation that ran all three full-grid conditions:

| condition | separates | both rise | neither moves | cannot be read |
|---|---|---|---|---|
| Gaussian | 78 | 31 | 8 | 120 |
| Grouped-wider | 74 | 37 | 6 | 120 |
| Grouped-shifted | 58 | 52 | 7 | 120 |

**Both numerators are right and both denominators are wrong.** 109 and 110 are separates plus
both-rise. The combinations that can be read at all are 117 in every condition — the same 117, because
what makes a combination readable is whether both parts vary per molecule, which does not depend on
the condition. The 8 and the 7 that go missing are the cells where neither part moves clearly, and
dropping them silently is what makes the two denominators differ by one when nothing differs.

§R6 above prints 78 of 117 and 58 of 117, with the both-rise counts of 31 and 52 in the sentence
after them, each against the same denominator of 117.

### 4. Everything else in §R6 and §R7, traced

- **Six of thirteen models can be asked.** `unc_slopes.csv` carries 13 model names. Readable: QRF,
  GP (het.), BNN-$\alpha$ (var. head), BNN-$\beta$ (var. head), VBLL-$\alpha$ (het.),
  VBLL-$\beta$ (het.). Not readable: NGBoost, GP, GP (Tanimoto), VBLL-$\alpha$, VBLL-$\beta$,
  BNN-$\alpha$, BNN-$\beta$. That matches `scripts/uncertainty_decomposition.py:94–180` row for row.
- **The quantile forest, 0 of 99.** `unc_slopes.csv`: 90 both-rise, 9 neither-moves, 0 separates,
  across every dataset, representation and condition it ran.
- **92 to 100 per cent rising, median slope +0.30 to +0.42.** `unc_q5.csv`, held-out split, one value
  per dataset, model, representation, condition, fold and component. Lowest share is Caco-2 under the
  outlier condition at 91.9 per cent; QM9 under Laplace, outlier and Student-t is 100 per cent.
- **Censoring, fewer than 5 per cent rising, median $-0.40$.** Same file. By dataset: Caco-2 0.4 per
  cent, logD 1.9, QM9 4.3, hERG K$_i$ 4.4.
- **95 per cent positive, 5 per cent above 0.3, of 1,003.** `d7_q6.csv`, the 1,003 rows carrying a
  named condition. Median 0.168. QRF highest by model at 0.257, BNN-$\beta$ lowest at 0.070.
- **QRF below RF in 13 of 18.** `d10_probabilistic.csv`, the `rf`/`qrf` rows, 6 representations by 3
  conditions on QM9. QRF is below under Gaussian and under grouped-shifted at all six
  representations, and ahead under grouped-wider at five of six.
- **111 of 113.** `auc_norm_assay.csv`, `baseline_r2`, the two variance-head Bayesian networks
  against the heteroscedastic Gaussian process, matched on dataset, representation and condition: 67
  of 68 for the $\alpha$ base and 44 of 45 for the $\beta$ base. Widening it to all four networks whose
  split can be read gives 221 of 224, because the two heteroscedastic variational networks are lower
  still. Against the quantile forest instead, the variance-head networks are lower in only 44 of 91,
  so the cost sentence has to name the Gaussian process and not a forest.
- **The censoring pairings.** `d7_q4.csv`: three on QM9 (BNN-$\alpha$ (var. head), GP and NGBoost, all
  at PDV) and 21 on each assay dataset, being 7 models at 3 representations.

### 5. A statistic in this harvest that computed nothing

`unc_q7_group_error.csv` has 20,699 rows and every one carries the reason **"no canonical_smiles on the
row"**. Its `group_share` column is empty throughout, so the share of a model's error carried by a whole
scaffold family — the test `RERUN_PLAN.md` §14.11y asks for, and the one thing that would say directly
whether the grouped-shifted condition concentrates error inside a scaffold family — is unmeasured in
this harvest. Nothing in §R6 or §R7 rests on it. It wants a look before the re-run, because a re-run
that does not carry `canonical_smiles` onto the per-molecule rows will produce the same empty column.

### 6. Two caption defects in F6 that a re-run will not fix

- **A fourth drawable model is silently dropped.** At QM9, ECFP4, Gaussian,
  `d8_component_slopes.csv` has four models with both parts per molecule: QRF, GP (het.),
  BNN-$\alpha$ (var. head) and BNN-$\beta$ (var. head). Three are drawn.
  `scripts/figlib_figures.py:1105` takes `C.sort_models(...)[:max_panels]` with `max_panels=6`
  **before** dropping the models whose second part is one number per fit, so BNN-$\beta$ (var. head)
  is cut by the panel cap and then never appears in any not-drawn list. It separates, at slopes
  $+1.051$ and $+0.144$, so the panel that is missing agrees with the panel that carries the finding.
  The caption above therefore names the three drawn panels, names the fourth model as not drawn, and
  does not claim a complete not-drawn list. Fixing the order of those two operations is a one-line
  change and would let the caption say it.
- **The drawn title vanishes on the re-run.** The PNG carries `QM9 (HOMO–LUMO gap), ECFP4, Gaussian`
  above the panels; `scripts/figlib_figures.py:1201` says `NO FIGURE TITLE` (you, 2026-09-17).
  Everything in that title is in the caption above, so nothing is lost.

### 7. What §R6 drops from the submitted text, and why

- *"GPs and NGBoost gave the strongest correlations"* (`paper.tex:502`) was a per-sample correlation
  pooled across noise levels, which measures the population trend rather than per-molecule detection.
  On this harvest the quantile forest has the highest median correlation between predicted uncertainty
  and out-of-fold error.
- *"for VBLL both the aleatoric and epistemic components increased"* (`paper.tex:541`) is now the
  quantile forest's result. The heteroscedastic variational networks separate; the plain ones cannot
  be asked, because their observation-noise term is one number per fit.
- mol2vec is gone from the table §R6 replaces, and so is ECE.

### 8. Two things only you can decide

- **Whether §R7 stays prose or gets F7 back.** It is written above as prose and stands on its own. If
  you reinstate F7, §R7's second paragraph becomes the figure's warrant rather than the whole
  finding, and the paragraph shortens by about 60 words. The uncertainty half of the paper is
  currently one figure and one table, and the title is about uncertainty.
- **Whether §R7's closing recommendation is one you want to make.** It tells a reader auditing a
  training set to sort by prediction error and treat the uncertainty as a tie-break. That is what the
  numbers support and it is the third recommendation in the Results, which is the cap house style
  rule 11 sets for the whole paper.
