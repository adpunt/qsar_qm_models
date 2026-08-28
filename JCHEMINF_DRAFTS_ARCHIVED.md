# Drafts rejected — archived for reference

User reviewed these on 2026-06-02 and rejected them. Kept here as a record of what was tried. **Do not re-apply.** Replacement drafts will be written separately after the Nature MI corpus review.

---

## DRAFT 1 — Scientific Contribution paragraph (was in abstract, L127–129)

> **Scientific contribution.** This study is the first to systematically partition QSAR noise robustness into model-architecture and molecular-representation contributions across six controlled noise strategies, demonstrating that architecture, not representation, is the dominant driver and that a model's relative robustness transfers across noise types, datasets, and representations. We further show that uncertainty estimates track per-sample injected noise only for models that learn an explicit noise or scale parameter (NGBoost, Gaussian processes), and only when paired with fingerprint-based representations. We release NoiseInject, an open-source benchmarking framework that operationalizes these protocols on any scikit-learn- or PyTorch-compatible model so that practitioners can pre-select robust architectures without re-running full noise sweeps on each new dataset.

**User feedback:** "Horrendous. Too specific, no one cares about doing that exact process, there's absolutely no concept of big picture or concern for what is actually out there." Reads like a methods recap rather than a contribution to the field. Missing the broader claim about why this work matters relative to prior QSAR / LNL work.

**Word count:** 107 words. Sentence count: 3 (within JCheminf max of 3).

---

## DRAFT 2 — Authors' contributions (Declarations, L619–623)

> *To be finalised.* AP designed the study, implemented the NoiseInject framework, ran all experiments, and drafted the manuscript. TH and SW contributed to the experimental design and provided industry-perspective input on noise sources in QSAR data. GM supervised the project, contributed to the experimental design, and revised the manuscript. All authors read and approved the final manuscript.

**User feedback:** Rejected. Reasons unstated but will be informed by the Nature MI corpus review of how this section is actually written.

---

## DRAFT 3 — Use of large language models (Methods, L350)

> The original noise-injection testing pipeline was developed manually. Extensions of that framework, including the NoiseInject benchmarking package and the validation experiments on additional datasets, were developed with the assistance of large language models, primarily for code drafting and refinement. Figure-generation scripts were also AI-assisted. The manuscript text was written manually, with large language models used for grammar correction, copy-editing, and consistency checks of the prose against the underlying experimental data. The bibliography was assembled manually and edited with the assistance of large language models. All AI-generated content was reviewed and verified by the authors, who take full responsibility for the accuracy of the methods, results, and citations reported in this manuscript.

**User feedback:** Rejected. Built without reference to how the journal community actually writes these disclosures. Need real evidence from published articles.

---

## DRAFT 4 — Seven trimmed caption titles (full set)

All seven carry `% TITLE TRIMMED to $\leq$15 words --- review wording.` in source. Originals are recoverable from `paper.tex.before_jcheminf_fixes` if needed.

| Line | Object | Trimmed first sentence |
|------|--------|------------------------|
| 332  | Fig 1 | "Effect of each noise injection strategy on the HOMO--LUMO label distribution at $\sigma = 0.5$." |
| 384  | Fig 2 | "ANOVA variance decomposition of predictive performance and noise robustness by noise strategy." |
| 412  | Table | "Noise degradation slope (NDS) by model on PDV, ranked across six strategies." |
| 463  | Fig 5 | "Neural network family comparison under Gaussian noise on PDV." |
| 477  | Table | "Strongest and weakest model--representation combinations for uncertainty--noise correlation on the QM9 HOMO--LUMO gap." |
| 521  | Fig 7 | "Noise degradation slope (NDS) heatmaps for three external validation datasets." |
| 532  | Fig 8 | "Comparison of noise robustness on QM9 and three external validation datasets." |

**User feedback:** Caption titles trimmed to satisfy the 15-word cap but supervisor specifically wants the **main idea of the table or plot** to be in the caption title. The current trims are descriptive-only (they say *what is shown*) and drop the *takeaway*. The previous full captions had `\textbf{Takeaway:} ...` sentences in the legend that captured the main idea but those are now buried in the legend, not the title. Need rewrite informed by the Nature MI corpus.

---

## What's next (the work the user actually wants done)

1. Read 9 Nature Machine Intelligence articles (paths under `~/Downloads/s42256-*.pdf`).
2. For each article, extract:
   - Author Contributions section text and structure
   - Scientific Contribution / novelty statement at end of abstract (or wherever it lives)
   - LLM / AI disclosure (if any)
   - Caption titles for figures and tables (and whether they include the main idea)
3. Synthesize across the corpus. Name the articles. Cite the lines.
4. Use the synthesis to rewrite the four drafts above with the actual writing norms in mind.
5. Especially: rewrite Scientific Contribution so it speaks to the field-level claim, not the local protocol details.

Do NOT proceed to rewrites until the literature review is done.
