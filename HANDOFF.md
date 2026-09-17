# Handoff — finishing the revision guide: the Results and the Introduction

Written 2026-09-17. Branch `additional_reps`. **This is the only handoff. Everything older was deleted
because six stacked handoffs in one file sent five chats off to do work from 7 September.**

Read `CLAUDE.md` first.

---

## The job, in one line

Write the Results and discussion, and revise the Introduction, into `PAPER_REVISION_GUIDE_FINAL.md` as
replacement text — following the format of `PAPER_REVISION_GUIDE.md`, which is the one the author says
worked.

The Methods are already done. They are in the same file under **PART ONE REWRITTEN — METHODS**.

---

## The author's brief. This is the specification — read it before anything else

Her words, 2026-09-16:

> This one should be based on the actual results which are like mostly in (there's a few things that will
> need to be left blank with TODOs). It will involve reading the figure generation script, HEAVILY reading
> the re-run plan. I like how I phrased things in paper.tex. Obviously a lot will change. Oh yea and you
> have to really go through my results. I did a lot of analysis while I was editing the figures, so a lot
> of the stuff is in the rerun plan, but you can't just take that for granted you need to dig deep.
>
> I have those set of questions that I want to answer in the rerun plan. I may explicitly ask a few of them
> in the paper text, but I want to naturally work most of them in. The takeaways should be the main point
> of my paper. I have the figures and the tables as proof of my results. Once I briefly explain what the
> reader is seeing I want to focus on the importance. I don't want many numbers in the text itself, that's
> what tables are for.
>
> I'm going to need the latex style inputs of the figures themselves. I'll upload the PNGs to my overleaf
> with those names. The captions need to be included as well.
>
> This is going to be the hardest part for you, don't go overboard. In fact, flag things that could be
> taken out. Things that don't contribute to the main points of my argument and are making my paper more
> complex. I want a clear story. Not a bunch of random facts. There should be a flow. It should start broad
> with the anova, saying generally, how does the choice of model and representation affect accuracy and
> noise robustness. Then I'll go deeper into specific models, things like NN variants. Then I get into
> different noise conditions and different datasets, seeing if my conclusions for which specific models,
> reps, and model/rep pairs did well on experimental datasets (as opposed to QM9 which is clean). Then I go
> into uncertainty, and seeing how uncertainty and noise play together and how the choice of model and how
> it estimates uncertainty plays a huge role in whether or not it works with noise, and what kind of noise
> it works for.
>
> Something to emphasize is the use of non-Gaussian noise, and hopefully finding conclusions that differ
> between gaussian and non-gaussian noise. Most papers when they look at noise model it as gaussian. I want
> to be able to say no and you're missing conclusions if you do that. That's what sets this paper apart.
> Hopefully. Or I find amazing uncertainty tracking. Let the results guide the final takeaway.
>
> This chat is responsible for the results section, but I'm intentionally leaving the abstract/conclusion
> till the end. That will all come from the results, and I may extend this chat later to do that because
> again, the focus should be the main takeaways.
>
> The introduction hopefully won't change all too much, but given my change in methods plus a lot of new
> references (esp with noise) in the re-run plan it will definitely need some edits. A FULL audit of the
> rerun plan is necessary here. Actually the introduction will change a lot wrt noise conditions. I'll need
> to focus on a few papers that I'm using to justify why I made the noise conditions the way I did. And I
> haven't seen other cheminformatics papers do this so I do need an emphasis on this.
>
> Remember we're following the format of the old revision guide, which is vaguely figure out what's wrong,
> suggest replacements, HUGE focus on answering the main questions and flow of the paper and the story I'm
> telling with it.
>
> You really like complicating things but your biggest challenge is going to be distilling information in
> the correct form.

---

## Step one, and it is most of the work: read the reference papers

**Do not start drafting.** The author asked for this explicitly and twice:

> I am begging you, the new chats need to use dynamic workflows and actually read these papers to verify
> structure, how things are explained, language, and honestly most importantly the KIND of content. How
> much detail? How many numbers go in the text? What kind of information is presented? What goes in the
> captions? How much do the results state what's in the figures vs a quick summary, how do the other papers
> flow? How do other papers discuss uncertainty? How long are the paragraphs? How much content is put into
> it?

They are at `/Users/apunt/Documents/ReferencePapers`, eighteen files in four folders:

| folder | what is in it |
|---|---|
| `JChem Papers` | three PDFs — the target journal, so these set the conventions the paper must meet |
| `NatureML Papers` | nine PDFs from Nature Machine Intelligence |
| `Other Papers` | five, including two chemistry preprints and the no-free-lunch paper |
| `Theses` | Markus Dablander's DPhil thesis |
| root | one review PDF on uncertainty quantification in QSAR, experimental noise and calibration |

Run this as a dynamic workflow: one agent per paper or per small group, each reading for the same fixed
list of questions, then one agent reconciling them into a short style specification. What comes back has to
be concrete enough to check a draft against — paragraph length in sentences, how many numbers appear in a
results paragraph, whether figures are described or summarised, what a caption carries that the text does
not, how an uncertainty result is framed. **Keep that specification in front of you while drafting and
check each drafted paragraph against it.**

The journal is JCheminf. It uses a combined "Results and discussion" and then "Conclusions". There is never
a standalone Discussion section — do not propose one.

---

## The format to follow

`PAPER_REVISION_GUIDE.md`, 645 lines, is the shape the author says worked. Its order:

1. **The current main picture — read this first.** The spine of the paper in two numbered points.
2. **The spine.** The same argument at length, with the table that makes it concrete.
3. **Your main points: where each is reflected.** One row per claim, and where in the paper it lands.
4. **The big picture: N whole units to rebuild, not patch.** A numbered list, one line each.
5. **FULL REBUILDS.** Numbered sections, each with the `paper.tex` lines it replaces and the replacement
   text written out.
6. **Mechanical fixes**, **figure audit**, **verify in data or code**, **suggested priority order**.

Copy that shape. Find what is wrong, propose the replacement, and keep the weight on the argument and the
flow rather than on a list of corrections. **Do not go past it and do not complicate it.**

---

## Non-negotiables

- **`paper.tex` is never edited.** It is a read-only download from the author's Overleaf project and
  nothing written into it reaches the paper. Read it for the voice and for line numbers. Everything you
  write goes into `PAPER_REVISION_GUIDE_FINAL.md`.
- **Never start a new file.** State goes in `RERUN_PLAN.md` (what gets run) or `NOISE_DESIGN.md` (what the
  noise is). This document is the exception and it replaces itself rather than growing a sibling.
- **No number from memory, and none from a document.** `RERUN_PLAN.md` records a great deal of analysis the
  author did while editing the figures, and it is the best map you will get — but it is a dated claim, not
  a fact, and it contains conclusions that were later withdrawn. Open the file under
  `results/decisions_arc/` and confirm the number in your own session before it goes in a sentence.
- **Never average across representations.** Representation is a measured factor. Every table is one
  representation, named in its title.
- **Never write an AI verdict into her documents.** Later it is indistinguishable from a measured result.
- **If something is unverified, say so in the same sentence as the claim.**
- **Banned words**: "arm", "stage", "retention area", "descriptor vector", "read-across". Say *noise
  condition*; **AUC_norm**; **PDV**. The three measured datasets are the **assay datasets** in the paper,
  never "validation" — that word collides with the early-stopping split inside every fold.

### The phrasing rules, and they are hard rules

`~/Documents/ai_phrasing.rtf` is the author's own record of what she stripped out of this paper once. Read
it. The pattern it identifies:

- No "indicates that", "suggests that", "demonstrates that" — write "shows", or drop the connector and
  state the thing.
- No "emerged as" for a result — it "was", it "produced", it "achieved".
- No "warranted investigation", no "The critical insight is", no "Several avenues emerge from these
  findings", no "providing actionable guidance".
- Remove the interpretive layer between the data and the conclusion. Give the finding, not a sentence about
  the finding.

Her own prose at `paper.tex:375–545` is the model: first person plural, one idea per sentence, nothing over
30 words, plain connectives, informal where it works ("On the flip side", "At the end of the day"), no bold
in body text, `\citet` when the authors are the subject and `\citep` otherwise. Six lines of `paper.tex` still carry a
banned construction, checked 2026-09-17: 380 "demonstrates that", 383 "translates to", 460 "indicates
that", 462 and 540 "suggests that", 493 "It appears that". An earlier note in this repository listed 264
and 433 as well and neither carries one — do not repeat that pair. Copy her sentence shape from her prose
and not the connector.

She has a PhD in this. Never explain the chemistry or the statistics.

---

## PART A — THE RESULTS AND DISCUSSION

### The shape of the argument. It is the author's and it is not negotiable

1. **Broad.** How does the choice of model and the choice of representation affect accuracy, and how do
   they affect noise robustness? Two different answers.
2. **Deeper into models**, the neural variants in particular — does making a model probabilistic change how
   noise hurts it. Plain network against Bayesian network against the variance head; forest against
   quantile forest.
3. **Noise conditions and the other datasets.** Do the conclusions about which models, which
   representations and which pairings survive when the noise changes shape, and when the labels are real
   measurements rather than computed ones?
4. **Uncertainty.** How uncertainty and noise play together, how a model's way of estimating uncertainty
   decides whether it survives noise at all, and which kind of noise it survives.

**The takeaways are the paper. The figures and tables are the proof.** Explain briefly what the reader is
looking at, then spend the words on why it matters. Keep numbers out of the running text — a number belongs
in a sentence only when the sentence turns on it.

### The seven questions

`RERUN_PLAN.md` §7.0, at line 8540, holds Q1 to Q7 with the statistic that answers each, what it is
computed over, and what the design had to supply. **Work most of them in naturally** rather than as
headings; the author may put one or two as explicit questions.

- Q1, whether robustness is decided by model, representation or their pairing, is movement 1 and is asked
  outright.
- Q3, what model choice buys at a realistic amount of error, is movement 2.
- Q2, whether the kind of noise matters or only the amount, is the spine of movement 3 and is asked
  outright, because it is the paper's differentiator.
- Q5, Q6, Q4 and Q7 are movement 4, in that order. Q5 is whether a model gets less sure and was already
  known. Q6 is whether uncertainty still ranks predictions by error. Q4 is whether it points at the
  corrupted labels, and is the hard one. Q7 is whether it tracks some kinds of noise better than others,
  and it is what makes Q4's answer interesting rather than negative.

**Q4, Q5 and Q6 are three different questions and earlier drafts of this paper fused them.** §7.0 also
records that Q4 is only defined for the three conditions that give some molecules more noise than others,
so a table ranking all seven on the Q4 statistic would be ranking four undefined cells against three real
ones.

### The thing that has to carry through all four movements

Most work in this area models label error as independent Gaussian noise. The author's claim is that doing
so costs you conclusions, and the run supports a sharper version of that than "non-Gaussian matters".

**Read `RERUN_PLAN.md` §14.17 and §14.17a before you decide how to phrase it, then re-derive every number
from `results/decisions_arc/` yourself.** The shape recorded there is that changing the shape of an
individual error costs little, while correlating errors within a scaffold family, or censoring at an assay
limit, costs a great deal — and costs more on the assay datasets than on QM9. If that survives your own
reading, the claim is about **independence and zero mean rather than Gaussianity**, which is stronger,
because those are the two assumptions nobody tests. If it does not survive, say so and write what did.

**A claim that the tail shape matters would be contradicted by her own table.** Check before you write it.

`NOISE_DESIGN.md` §5.1e holds the measurement that makes a comparison between conditions fair — every
dose-matched condition delivers the same expected amount of corruption, so a difference between two of them
is a difference of pattern and not of size. That is what lets the comparison mean anything and it belongs
in the argument.

### Read these, in this order

1. `results/decisions_arc/DECISIONS.md` — thirteen questions, each with the number that settles it and
   whether it fired. This is the spine of the figure set: every figure exists because a decision fired.
2. `results/decisions_arc/figures/captions.md` and `notes_for_the_text.md` — the generated captions, and
   the findings the author ruled were sentences rather than pictures.
3. `results/decisions_arc/what_is_missing.csv` — what cannot be said yet, in terms of the figure slot that
   wanted it, each row marked `runnable` or `decision`. **Read it before writing "we did not measure"
   about anything** — most of those rows are a design decision, not a queue to clear.
4. `RERUN_PLAN.md`: §7.0 (the questions), §13.16 (the reporting levels), §14.5 (the figure slots), §14.7
   (the tables), **§14.15, which overrides every earlier §14 subsection it disagrees with and lists four
   claims this repository stated and then retracted**, **§14.17 and §14.17a** (what each noise condition
   bought), §14.24 (the colour ranges and the F9 cut), **§14.25 (the live figure list)**.
5. The figure code: `scripts/figlib_metrics.py` for what AUC_norm and the variance decomposition actually
   compute — in particular that AUC_norm is integrated **per replicate** and then aggregated, which the
   submitted paper's numbers are not; `scripts/figlib_decisions.py` for what each decision tests;
   `scripts/figlib_figures.py` for what each figure draws; `scripts/run_paper_analysis.py` for what is
   drawn at all.

**Do not take `RERUN_PLAN.md` at face value.** It is a working log. §14.15 exists because of exactly that.
If a claim lives only in prose in the plan, treat it as a lead and go to the results file.

### The figures

`RERUN_PLAN.md` §14.25 is the live list and it is the one to work from. Seven figures in the paper:

| slot | file | what it answers |
|---|---|---|
| F1 | `F1_noise_conditions.png` | what each condition does to a label distribution — belongs with the Methods |
| F2 | `F2_variance_decomposition.png` | model, representation, or the pairing |
| F3 | `F3_model_by_representation.png` | which model on which representation |
| F4a | `F4a_models_under_noise.png` | what label noise costs you |
| R17 | `R17_variant_families_ecfp4_gaussian.png` | does making a model probabilistic help |
| F6 | `F6_decomposition.png` | does noisy training make a model less sure |
| F8 | `F8_assay_datasets_logd.png`, `_caco2.png`, `_herg.png` | does it hold on assay data |

Additional files: F4b, F4c, R6 (two files), R9, R15 (two files), R18 (two files), R19. **F7 was cut on 13
September and F9 on 16 September** (§14.24) — the author's calls, both recorded, neither to be reopened.
R10 and R15b are sentences rather than figures and their text is in the guide's §L1.

**⚠️ The consequence of cutting F9, written into `scripts/run_paper_analysis.py:579–581` and §14.24:** the
uncertainty side of the paper is now one figure, F6, and one table, T6 — and the paper's title is about
uncertainty. The Q4 result has to be carried in prose, from `notes_for_the_text.md` and T6. That is flagged
as a thing to watch, not resolved. If your reading of the results says the paper needs a picture there,
put the case to the author with what it would cost. Do not draw one on your own authority.

**Deliver, for every figure in the paper, a complete LaTeX block ready to paste**: `\begin{figure}`,
`\includegraphics[width=\textwidth]{<the exact file name above>}`, the caption, and the `\label` the text
then references. Follow `paper.tex`'s convention — `htbp`, `\label{fig:...}`, `figure*` for full-width.
F8 is three files and needs a decision about whether it is one figure with three panels or three figures;
put that to the author.

**The captions in `captions.md` are generated and several are far too long for a journal.** They were
written to carry what was deliberately kept out of the figure titles — what a grey cell means, what a
whisker is, what is deliberately not drawn. Those clauses exist because a reader would otherwise over-read
the picture. Rewrite each in the author's voice, keep every factual clause, trim the rest, and say in your
notes what you cut from each. Her caption style: one sentence naming what the figure shows and on what,
then lettered panels `a)` `b)` `c)`, then the reading notes. The reference papers will tell you how long a
caption in this literature actually runs — check.

### The tables

In `results/decisions_arc/tables/`, as both `.csv` and `.tex`. **Read the `.tex` files — do not retype a
table.**

T1 metrics, T2 noise conditions, T3 variance decomposition, T4 robustness (one file per dataset at ECFP4),
T5 probabilistic transformations, T6 uncertainty, T7 rank transfer (one file per representation), T8
pairings across datasets. T1 and T2 are Methods. The paper takes T1, T2, T4 at ECFP4, and T6; T3, T5, T7
and T8 are additional files.

**T4 and T7 exist once per dataset or representation by design** — averaging over either is the thing the
whole figure set was rebuilt to prevent. The paper takes one of each and the rest are additional files.
Where a table needs a column the generated one does not have, say so and name the script.

### Flag what to cut. This is asked for explicitly

> In fact, flag things that could be taken out. Things that don't contribute to the main points of my
> argument and are making my paper more complex. I want a clear story. Not a bunch of random facts.

Produce a short numbered list at the end: **what you would cut, what the paper loses, and what it was
protecting against.** Candidates to weigh without deciding for her:

- Six rank-transfer tables when the finding is one number.
- R16, the decoupling figure — its own caption warns that part of what it shows is arithmetic, because
  AUC_norm divides the clean baseline out. The finding is real and lives in §14.15b; the figure may be
  doing it a disservice.
- R18, the representation-against-representation scatter — it is the evidence for holding one
  representation constant in the main text, which is a Methods decision rather than a result.
- R15, the rank ladder — two charts to say the ranking barely moves.
- R19, the depth conditions as a figure, when the matched-pair comparison is the finding and is a table.
- Every sentence that reports a statistic nobody asked a question about.

Anything you cut, say where it goes — additional file, or gone.

---

## PART B — THE INTRODUCTION

`paper.tex:178–188`, six paragraphs. It is the part of the paper that survives best and the author expects
it to change least. Three things force it anyway.

**1. The Methods underneath it changed.** Six noise strategies became seven conditions, eleven noise levels
became seven, the representation set changed, and the paper has an uncertainty subsection it did not have.
Any sentence that sets up the old design has to move.

**2. The claim has sharpened.** The Introduction currently sets representation choice up as the open
question. Read the decomposition yourself and write the Introduction to motivate the question the paper can
now answer.

**3. The noise conditions need justifying, and this is most of the job.** Her words:

> Actually the introduction will change a lot wrt noise conditions. I'll need to focus on a few papers that
> I'm using to justify why I made the noise conditions the way I did. And I haven't seen other
> cheminformatics papers do this so I do need an emphasis on this.

### The reference audit

**Read `NOISE_DESIGN.md` §3.1 to §3.6 and §4b in full.** §4b is a primary-source layer — verified verbatim
quotes, each with a note saying what may and may not be claimed from it. §4a reconciles two earlier
literature passes that disagreed and says which to trust. **`NOISE_DESIGN.md:638` is a list headed "Numbers that must NOT
enter the paper" — read it before you quote anything and honour it.**

What is there and what it is for:

- **Krüger & Overington (2012)** rejected normality of bioactivity differences and fitted a Laplace. §3.1
  warns that the paper never uses the words "heavy-tailed" or "Gaussian", so cite what they did, not what
  it implies.
- **Kalliokoski et al. (2013)**, 16,844 repeat pairs, could only fit a Gaussian after truncating. §3.1 also
  carries the honest limitation: matching the real tail needs a Student-*t* with about one degree of
  freedom, which has no finite variance and therefore cannot be dose-matched at all. **That limitation
  belongs in the paper.**
- **Kramer et al. (2012)** and §3.2: error does not depend on the measured value. This disposes of the
  value-proportional and threshold strategies the submitted paper used, and it is why the outlier condition
  selects at random rather than by size. The Introduction is where that premise gets stated.
- **Bentz et al. (2013)** and §3.3: most measurement variance is between laboratories. This is the source
  behind the grouped conditions and it is what the paper's sharpest finding rests on. The Introduction
  never sets it up.
- **Hayeshi et al. (2008)**, **Chen et al. (2017)**: inter-laboratory Caco-2.
- **Svensson et al. (2025)** and §3.5: censoring is the most prevalent real mechanism, with a quarter to
  two thirds of labels censored in industrial assays. The paper's worst result is under censoring, so the
  Introduction should say it is the most common mechanism before the Results say it is the most damaging.
- **Heid et al. (2023)** is already cited and §3.6 records exactly what they published, so the paper's
  relationship to it can be stated precisely rather than gestured at.
- **Alvarez Baron et al. (2025)**, **Niu et al. (2024)**, **Sato et al. (2018)**, **Wenlock et al. (2011)**,
  **Prieto et al. (2010)**, **Avdeef (2019)** — assay-error anchors, each with its scope in §4b.

On the uncertainty side, `RERUN_PLAN.md` §14.6 row 2 names four papers as the field standard that are not
in `citations.bib`: Scalia et al. 2020, Hirschfeld et al. 2020, Tran et al. 2020, and the Kendall & Gal
decomposition the Methods now gives as a display equation. `research_archive/f692d614/` holds three of them
as PDFs.

**The bibliography is in three places and they have drifted.** `citations.bib` carries 221 entries and
`paper.tex` cites 51. `refs.bib` is untracked in git. `paper.tex` points `\bibliography` at
`sn-bibliography`, which lives in the Overleaf project and is **not in this checkout** — so you cannot tell
from here whether it holds a key. Report what is missing from the two local files and flag every key you
add as needing an Overleaf check. `\citep{avalon}` is in neither: Gedeck, Rohde & Bartels, *J. Chem. Inf.
Model.* 46(5):1924–1936, 2006. Run `python scripts/check_bib_and_docs.py` before you finish.

### Two sentences that are now wrong

- **`paper.tex:188`, the second of the three aims.** It promises to compare probabilistic models against
  deterministic ones and to say whether their per-sample uncertainty tracks which labels were corrupted.
  Both halves are measured now and the second has a conditional answer — it depends on the kind of noise.
  The sentence promises a yes or no and the paper delivers something better than one.
- **The Deng (2023) sentence at `paper.tex:186`.** It sets representation choice up as the open question.
  Keep it, because it motivates the work, but do not lean on it as though the answer were expected.

The Introduction is where a claim about the literature is easiest to write as a sentence about the claim.
Write what the cited paper measured instead.

---

## What to leave as TODO, with the file that would settle it

Mark these rather than guessing around them.

- **Anything resting on QM9's heteroscedastic Gaussian process.** The lengthscale fix is commit `83228f3`,
  pushed 13 September at 21:55, whose own message says 1,717 rows already on disk are a different fit and
  are a resubmission. The results were harvested nine hours later, so those rows are probably a mixture of
  pre-fix and post-fix fits. No number and no cross-condition comparison for that model until the author
  establishes from the job logs which rows were fitted on which code. `d3_condition_spread.csv` gives it
  the largest cross-condition spread of any model, which is the row most likely to be the artefact rather
  than a finding. **Do not resolve this yourself and do not soften it.**
- **Avalon wherever it depends on the two NN-α tuned settings**, which were ranked over five
  representations and used on six.
- **The fourteen combinations short a noise level** in `what_is_missing.csv`.
- **Anything the uncertainty runs have not delivered** for a pair named in `uncertainty_pairs.json`.

---

## Not yours

- **The abstract, the conclusion and the scientific-contribution statement.** The author is doing them last,
  deliberately, because they come off the Results. Do not draft them, and do not write a Results paragraph
  that only works once the conclusion exists. She may extend the chat to do them afterwards.
- **The Methods.** Done, in Part One Rewritten of the same file. Read §M0 and the third-pass section before
  you write a sentence about what was run. QM9 and the three assay datasets are two implementations of one
  design and they differ in many load-bearing ways, so a flat sentence about "the study" is usually false
  on one side. The guide gives the count as thirteen at line 84 and as "far more than twelve" in §4.5, so
  read §M0 and count for yourself rather than quoting either.
- **`paper.tex`.** Inert. Read it, never write it.

---

## What you owe back

1. Replacement text for the Results and discussion, in the four-movement order, in the format of
   `PAPER_REVISION_GUIDE.md`, into `PAPER_REVISION_GUIDE_FINAL.md`.
2. Replacement text for the Introduction, paragraph by paragraph, marked against the `paper.tex` lines it
   replaces.
3. A LaTeX block and a trimmed caption for every figure in the paper, with its `\label`.
4. The numbered cut list — what could come out and what the paper loses.
5. A table of every reference added: key, what it supports, which `NOISE_DESIGN.md` or `RERUN_PLAN.md`
   section it came from, and whether it is already in `citations.bib`, in `refs.bib`, or in neither — plus
   the list of keys that can only be checked against the Overleaf bibliography.
6. The style specification you extracted from the reference papers, so the author can see what you matched
   the draft against.
7. `python scripts/check_bib_and_docs.py` passing.
8. One entry in `RERUN_PLAN.md` §14 recording what you settled and what you left open. **Do not start a new
   file.**
9. If a number in `RERUN_PLAN.md` is contradicted by the results, correct it there with the file the
   correction came from, the way §14.15a does.

---

## The state of the repository, verified 2026-09-17

Read in this session, from the files named. Re-check anything you are about to write from.

**The experiments are in.** `results/decisions_arc/d0_coverage.csv` is built from the result files
themselves and holds 1,565 combinations — one per dataset, model, representation and noise condition.
1,551 are complete. All four datasets carry 109 complete combinations on each of Gaussian, grouped-wider
and grouped-shifted, 18 to 24 on each of Laplace, outlier and Student-*t*, and 5 on censoring.

**The fourteen that are short a noise level** are all on QM9: five `het_gp_rbf` under Laplace, outlier and
Student-*t*, and nine `mlp_vbll_hetero` under the same three, missing only the clean level — which is what
drops them from every AUC_norm figure, because the metric is a ratio to the clean score.

**Two combinations carry a collapsed Gaussian-process fit** and the analysis filters them before any number
is drawn. Per-combination detail is in `d0_coverage.csv` under `gp_collapsed`.

**The uncertainty runs have landed on all four datasets**, seven conditions and seven models —
`uncertainty_pairs.csv` holds 3,156 rows and `d7_q4.csv` holds 20,699.

**The figures in `results/decisions_arc/` may be behind the code.** They are regenerated by
`scripts/run_paper_analysis.py` on the cluster. Two commits landed after the set of 16 September at 22:04 —
`87643a8` and `9d2cd2e` — and two more after the set of 17 September at 04:00 — `69815b6`, which fixes F4a
drawing one panel under a caption that promised two, and `93fe179`. **Check the timestamp on the PNG
against `git log` on the figure scripts before you write a caption from a picture.** The re-run is
`sbatch slurm_scripts_analysis/run_paper_analysis.sh` and takes about sixteen minutes.

**Sections 14.20, 14.21 and 14.22 each appear twice in `RERUN_PLAN.md`**, and
`scripts/run_paper_analysis.py:581` cites §14.22 for the F9 cut when the record is in §14.24. Nothing was
renumbered, because other documents cite those numbers. **§14.25 is the live figure list.**

**The author runs every command.** There is no cluster access from this side. Anything needed from ARC is
one block to paste, not a conversation, and several questions batched into one block. Prefer plain `sacct`,
`squeue`, `ls` and `tail` over a script written five minutes ago. Code reaches the cluster only when she
runs `bash scripts/pull_safely.sh` against a commit that is already pushed — a fix that is not pushed does
not exist.
