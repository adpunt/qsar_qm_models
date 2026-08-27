# Journal of Cheminformatics — literature review evidence (consolidated)

**File location:** `/Users/apunt/repos/qsar_qm_models/JCHEMINF_LITERATURE_REVIEW.md`

**Status:** Three rounds of evidence collection complete. No rewrites have been applied to `paper.tex` yet — the user must approve direction before any draft hits the manuscript.

**Companion files in the same directory:**
- `JCHEMINF_DRAFTS_ARCHIVED.md` — the four rejected drafts (Sci Contrib, Authors' Contributions, LLM disclosure, caption titles) preserved for reference, not for re-use.
- `JCHEMINF_GUIDELINES_VERBATIM.md` — verbatim JCheminf research-article submission guidelines.
- `JCHEMINF_GENERAL_FORMATTING_VERBATIM.md` — verbatim BMC General Formatting guidelines.
- `JCHEMINF_FIXES_LOG.md` — history of every edit applied to `paper.tex` to date.
- `JCHEMINF_STATUS.md` — forward-looking tracker of done / TODO items.

**Anti-hallucination guarantee:** every quote in this file is verbatim from a source I (or a sub-agent I dispatched) actually fetched. PMC URLs are listed; the user can verify each quote against the live page. Where a piece of evidence is missing, it is flagged "absent" or "not found" — not paraphrased as if quoted.

---

## Part 1 — Scientific Contribution statement evidence (10 confirmed verbatim examples)

### JCheminf editor guidance (verbatim)

From Bajorath (2024), *J Cheminform* 16, 4 — "Are new ideas harder to find? A note on incremental research and Journal of Cheminformatics' Scientific Contribution Statement". DOI `10.1186/s13321-024-00803-6`. PMC: PMC10789001.

> "The authors should use a maximum of three sentences to specifically highlight the scientific contributions that advance the field and what differentiates their contribution from prior work on this topic."

Editorial frames the statement as an "opportunity to highlight … rather than a burden." Contains **no verbatim example statements** itself.

### Example 1 — MotifMol3D (Hu et al. 2025)

- **Article:** Learning motif features and topological structure of molecules for metabolic pathway prediction
- **DOI:** `10.1186/s13321-025-00994-6` · **PMC:** PMC12013036
- **Authors:** Jianguo Hu, Yiqing Zhang, Jinxin Xie, Zhen Yuan, Zhangxiang Yin, Shanshan Shi, Honglin Li, Shiliang Li
- **Topic:** Motif-aware 3D GNN for metabolic pathway prediction
- **3 sentences · ~62 words**

> "MotifMol3D integrates motif information, graph neural networks, and 3D structural data to enhance feature extraction for small-sample molecules, improving the precision and interpretability of metabolic pathway predictions. The model outperforms state-of-the-art approaches in precision, recall, and F1 score. This work reveals how motif information characterizes pathway-specific molecules, offering novel insights into molecular properties within metabolic pathways."

**Pattern:** artifact + what-it-does → empirical claim → field-level insight.

### Example 2 — QSPRpred (van den Maagdenberg et al. 2024)

- **Article:** QSPRpred: a Flexible Open-Source Quantitative Structure-Property Relationship Modelling Tool
- **DOI:** `10.1186/s13321-024-00908-y` · **PMC:** PMC11566221
- **Topic:** Open-source QSPR software toolkit
- **3 sentences · ~79 words**

> "QSPRpred aims to provide a complex, but comprehensive Python API to conduct all tasks encountered in QSPR modelling from data preparation and analysis to model creation and model deployment. In contrast to similar packages, QSPRpred offers a wider and more exhaustive range of capabilities and integrations with many popular packages that also go beyond QSPR modelling. A significant contribution of QSPRpred is also in its automated and highly standardized serialization scheme, which significantly improves reproducibility and transferability of models."

**Pattern:** artifact + scope → explicit comparison ("In contrast to similar packages") → specific methodological advance.

### Example 3 — MolPROP (Rollins et al. 2024)

- **Article:** MolPROP: Molecular Property prediction with multimodal language and graph fusion
- **DOI:** `10.1186/s13321-024-00846-9` · **PMC:** PMC11112823
- **Topic:** Multimodal language + graph fusion for molecular property prediction
- **2 sentences · ~63 words**

> "This work explores a novel multimodal fusion of learned language and graph representations of small molecules for the supervised task of molecular property prediction. The MolPROP suite of models demonstrates that language and graph fusion can significantly outperform modern architectures on several regression prediction tasks and also provides the opportunity to explore alternative fusion strategies on classification tasks for multimodal molecular property prediction."

**Pattern:** novelty claim first → empirical claim + opens-future-work clause.

### Example 4 — Syk-inhibitor QSAR+RL (Zavadskaya et al. 2025)

- **Article:** Integrating QSAR modelling with reinforcement learning for Syk inhibitor discovery
- **DOI:** `10.1186/s13321-025-00998-2` · **PMC:** PMC11998205
- **Topic:** QSAR-guided reinforcement learning for kinase inhibitor design
- **2 sentences · ~42 words**

> "The study presents the first application of QSAR-guided reinforcement learning for Syk inhibitor discovery, yielding structurally novel candidates with predicted high potency. The presented methodology can be adapted for other therapeutic targets, potentially accelerating the drug development process."

**Pattern:** insight-led with explicit "first" novelty claim → forward utility. Generic prior-work positioning.

### Example 5 — Toxicokinetic & physicochemical benchmark (Gadaleta et al. 2024)

- **Article:** Comprehensive benchmarking of computational tools for predicting toxicokinetic and physicochemical properties of chemicals
- **DOI:** `10.1186/s13321-024-00931-z` · **PMC:** PMC11674477
- **Topic:** Benchmark of in silico PC/TK prediction tools
- **2 sentences · ~54 words**

> "The present manuscript provides an overview of the state-of-the-art available computational tools for predicting the PC and TK properties of chemicals. The results here offer valuable guidance to researchers, regulatory authorities, and the industry in identifying robust computational tools suitable for predicting relevant chemical properties in the context of chemical design, toxicity and environmental fate assessment."

**Pattern:** artifact-led (manuscript as subject) + stakeholder utility appeal (typical for benchmark papers).

### Example 6 — BERT + Bayesian active learning (Masood et al. 2025)

- **Article:** Molecular property prediction using pretrained-BERT and Bayesian active learning: a data-efficient approach to drug design
- **DOI:** `10.1186/s13321-025-00986-6` · **PMC:** PMC12020163
- **Topic:** BERT + Bayesian active learning for low-data molecular property prediction
- **3 sentences · ~57 words**

> "We demonstrate that high-quality molecular representations fundamentally determine active learning success in drug discovery, outweighing acquisition strategy selection. We provide a framework that integrates pretrained transformer models with Bayesian active learning to separate representation learning from uncertainty estimation—a critical distinction in low-data scenarios. This approach establishes a foundation for more efficient screening workflows across diverse pharmaceutical applications."

**Pattern:** insight-led empirical claim first ("representations outweigh acquisition strategy"), then artifact, then forward-looking utility.

*Caveat on Example 6: The PMC fetch returned the final phrase as "pharmaceutical articles", probably an OCR artefact. Verify on the PMC page directly before re-quoting.*

### Example 7 — ADMET feature-representation benchmark (Kamuntavičius et al. 2025)

- **Article:** Benchmarking ML in ADMET predictions: the practical impact of feature representations in ligand-based models
- **DOI:** `10.1186/s13321-025-01041-0` · **PMC:** PMC12281724
- **Topic:** ADMET ML benchmarking, feature-representation impact
- **4 sentences · ~76 words** *(slightly over the 3-sentence target)*

> "This study provided a structured approach to feature selection. We improve model evaluation by combining cross-validation with statistical hypothesis testing, making results more reliable. The methodology used in our study can be generalized beyond feature selection, boosting the confidence in selected models which is crucial in a noisy domain such as the ADMET prediction tasks. Additionally, we assess how well models trained on one dataset perform on another, offering practical insights for using external data in drug discovery."

**Pattern:** methodology-led, multi-claim. Explicit framing as evaluation-rigor contribution ("statistical hypothesis testing", "structured approach"). **The closest stylistic match to the user's noise-benchmarking paper.**

### Example 8 — Auxiliary-task GNN adaptation (Dey & Ning 2024)

- **Article:** Enhancing molecular property prediction with auxiliary learning and task-specific adaptation
- **DOI:** `10.1186/s13321-024-00880-7` · **PMC:** PMC11270959
- **Topic:** Auxiliary-task gradient surgery (RCGrad) for fine-tuning pretrained molecular GNNs
- **3 sentences · ~64 words**

> "We introduce a novel framework for adapting pretrained GNNs to molecular tasks using auxiliary learning to address the critical issue of negative transfer. Leveraging novel gradient surgery techniques such as RCGrad, the proposed adaptation framework represents a significant departure from the dominant pretraining fine-tuning approach for molecular GNNs. Our contributions are significant for drug discovery research, especially for tasks with limited data, filling a notable gap in the efficient adaptation of pretrained models for molecular GNNs."

**Pattern:** artifact-led ("we introduce a novel framework"). Explicit "novel" twice. Generic prior-work comparator ("dominant pretraining fine-tuning approach"). More assertive register.

### Example 9 — CPSign conformal-prediction software (Arvidsson McShane et al. 2024)

- **Article:** CPSign: conformal prediction for cheminformatics modeling
- **DOI:** `10.1186/s13321-024-00870-9` · **PMC:** PMC11214261
- **Topic:** Conformal prediction software framework with empirical SOTA comparison
- **2 sentences · ~62 words**

> "CPSign provides a single software that allows users to perform data preprocessing, modeling and make predictions directly on chemical structures, using conformal and probabilistic prediction. Building and evaluating new models can be achieved at a high abstraction level, without sacrificing flexibility and predictive performance—showcased with a method evaluation against contemporary modeling approaches, where CPSign performs on par with a state-of-the-art deep learning based model."

**Pattern:** artifact-led with named tool. Empirical comparison ("on par with a state-of-the-art deep learning based model") but generic comparator.

### Example 10 — UMAP clustering splits (Guo et al. 2025)

- **Article:** UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines
- **DOI:** `10.1186/s13321-025-01039-8` · **PMC:** PMC12153141
- **Topic:** New data-splitting protocol for virtual screening benchmarks
- **3 sentences · ~64 words**

> "This work advances the field by introducing UMAP clustering as a robust splitting method for molecular datasets, improving over traditional methods like Butina clustering and especially scaffold splits. It offers a new evaluation framework to benchmark AI models under more realistic conditions, fostering progress in molecular property prediction. The findings also show how inappropriate the use of ROC AUC for virtual screening (VS) continues to be, despite its popularity, emphasizing the need for context-specific evaluation metrics."

**Pattern:** method-led with two **named prior-work comparators** (Butina clustering, scaffold splits) — the only paper in the sample to name competitors explicitly. Embedded critique of common practice (ROC AUC misuse). **Also relevant to the user's paper: this is exactly the "improve how the field evaluates X" rhetorical move.**

### Example 11 — Grigorev et al. 2026 (logP transformer)

- **DOI:** `10.1186/s13321-026-01160-2` · **PMC:** PMC13041340
- **Topic:** Graphormer-based logP prediction with curated dataset
- **3 sentences · ~58 words**

> "This paper presents two key scientific contributions. First, we have collected and carefully curated a large and diverse dataset of molecules with measured logP values, comprising over 42 000 compounds. Second, we propose a Graphormer-based model with a task-specific fine-tuning architecture for logP prediction, tailored to leverage representations learned from reaction data. This model demonstrates high performance in benchmark studies on both established literature data and the newly compiled dataset."

**Pattern:** insight-led, explicit "two key scientific contributions" framing. Dataset + method doublet — common QSAR template.

### Example 12 — Esaki & Ikeda 2026 (data curation review)

- **DOI:** `10.1186/s13321-026-01174-w` · **PMC:** PMC13059370
- **Topic:** Review/synthesis of QSAR data curation practices
- **~2 sentences with enumeration · ~125 words**

> "Scientific Contribution: This article does not introduce a new algorithm but provides a practice-oriented, structured synthesis of data curation in cheminformatics. We (i) formulate a two-pillar framework that treats structural curation and experimental-condition curation as equally important components of cheminformatics workflows; (ii) consolidate scattered best practices into concrete workflows, checklists, and decision maps for building \"QSAR-ready\" and condition-aware datasets; and (iii) integrate endpoint-specific case studies showing that rigorous curation materially improves predictive performance and reproducibility. We also identify open challenges and research directions for scaling and automating curation, including the use of workflow technologies and large language models, and for establishing community standards for condition metadata."

**Pattern:** defensive insight-led opening ("does not introduce a new algorithm but…"); explicit i/ii/iii enumeration. **Directly relevant template for the user's paper** — gives permission to lead with the synthesis claim rather than an artifact.

### Example 13 — Mun & Fazli 2026 (CheMLT-F multitask transformer)

- **DOI:** `10.1186/s13321-026-01199-1` · **PMC:** PMC13217713
- **Topic:** Unified multitask transformer with scaffold-aware vs random splits
- **3 sentences · ~95 words**

> "We introduce a unified transformer architecture that jointly models molecular and protein sequences across hundreds of pharmacologically relevant endpoints spanning toxicity, physicochemical properties, and drug–target interactions. A tailored training strategy that combines partial encoder freezing, global–local loss balancing, and weighted task sampling reduces trainable parameters and deployment complexity while preserving strong cross-domain generalization. Comprehensive evaluation across 13 public datasets, including scaffold-aware and random data splits, demonstrates competitive accuracy with substantially lower operational overhead than maintaining numerous single-task models, establishing a scalable foundation for extensible and holistic predictive modeling in computational drug discovery."

**Pattern:** artifact-led with explicit evaluation-rigor framing (scaffold-aware vs random splits). Relevant template for robustness-benchmarking framing.

### Example 14 — Domnjuk et al. 2026 (lightweight GNN for 31P NMR)

- **DOI:** `10.1186/s13321-026-01178-6` · **PMC:** PMC13112750
- **Topic:** GNN benchmarking + interpretability for NMR shift prediction
- **4 sentences · ~92 words**

> "The proposed lightweight GNN based on the Metalayer framework improves the state-of-the-art 31P NMR shift prediction. By systematically varying the training-set size and benchmarking against multiple state-of-the-art models, we provide a standardized performance comparison that was previously lacking for 31P NMR shift prediction. Using GNNExplainer and targeted feature ablations, we relate the model's predictions to specific molecular substructures and input features. By quantitatively verifying that the model learns fundamental physical trends like substituent increments and identifying specific error sources, we provide a level of chemical interpretability and validation that goes beyond prior black-box approaches."

**Pattern:** artifact + benchmarking + interpretability. Phrase **"a standardized performance comparison that was previously lacking"** is the exact gap-filling move the user's noise paper could mirror. Strong "we provide... we provide..." parallel structure.

### Example 15 — Parrondo-Pizarro et al. 2025 (AssayInspector for ADME)

- **DOI:** `10.1186/s13321-025-01103-3` · **PMC:** PMC12574261
- **Topic:** Diagnosing distributional misalignment in ADME benchmarks
- **3 sentences · ~72 words**

> "By systematically analyzing public ADME datasets, we uncovered substantial distributional misalignments and annotation discrepancies between benchmark and gold-standard sources. These challenges were shown to undermine predictive modeling, as naive integration or standardization often degraded performance. To address them, we present AssayInspector, a tool that enables both consistency assessment and informed data integration, providing a foundation for more reliable predictive modelling in drug discovery."

**Pattern:** problem-found → consequence → tool. **Closest in tone to the user's noise-robustness story.** The phrase "these challenges were shown to undermine predictive modeling" maps directly onto the noise/label-quality framing.

### Example 16 — Lou et al. 2025 (Multi-MoleScale)

- **DOI:** `10.1186/s13321-025-01126-w` · **PMC:** PMC12797713
- **3 sentences · ~62 words**

> "We introduce Multi-MoleScale, a novel multi-scale framework that integrates Graph Contrastive Learning (GCL) with sequence-based models such as BERT. This innovative dual approach effectively combines molecular graph structures with sequence information, significantly enhancing predictive accuracy. By capturing both intrinsic graph-based features and contextual relationships within molecular sequences, Multi-MoleScale enables the differentiation between relevant and irrelevant molecular features."

**Pattern:** artifact-led with explicit "novel"/"innovative" puffery — a **weaker example**, included as a counterpoint to the more disciplined 2026 statements above.

### Counter-example — no SC paragraph

- **Nelen et al. 2025**, "Matched pairs demonstrate robustness against inter-assay variability"
- DOI `10.1186/s13321-025-00956-y` · PMC: PMC11748845
- Abstract has no labelled "Scientific contribution" section. **Compliance not universal.**

### Cross-paper synthesis (Sci Contrib, n = 10)

- **Word counts:** 42 (shortest) to 79 (longest). Median ~63. All within the spirit of the 3-sentence guideline; 1 of 10 (Kamuntavičius) used 4 sentences.
- **Opening style:**
  - **Artifact-led** (names a tool / framework / manuscript / study as subject): MotifMol3D, QSPRpred, MolPROP, Gadaleta (manuscript), Dey & Ning (framework), CPSign, Guo (work) → **7 of 10**.
  - **Insight/claim-led** (leads with empirical claim): Zavadskaya, Masood, Kamuntavičius → **3 of 10**.
- **Prior-work positioning:**
  - **Named comparator:** Guo (Butina + scaffold) — **1 of 10**.
  - **Generic** ("state-of-the-art", "similar packages", "dominant fine-tuning approach", "traditional methods"): MotifMol3D, QSPRpred, Gadaleta, Dey & Ning, CPSign — **5 of 10**.
  - **No prior-work positioning at all** in the SC paragraph: MolPROP, Zavadskaya, Masood, Kamuntavičius — **4 of 10**.
- **Novelty wording:**
  - **Explicit** ("first", "novel"): Zavadskaya, Dey & Ning, MolPROP — **3 of 10**.
  - **Implicit** ("advances", "improve", "outweighing", "departure from", "on par with"): **7 of 10**.
- **For methodology / benchmarking papers specifically** (Gadaleta, Kamuntavičius, CPSign, Guo — most relevant to the user's paper):
  - All open with the artifact / study as grammatical subject.
  - All invoke stakeholder utility ("researchers, regulators, industry", "the field", "users", "drug discovery").
  - None use "first"; only Guo names specific comparators.
  - **Common rhetorical move: evaluation-rigor framing** — statistical hypothesis testing (Kamuntavičius), method evaluation against SOTA (CPSign), context-specific metrics + ROC AUC critique (Guo), comprehensive benchmark coverage (Gadaleta).
  - **This is the most reusable move for the user's paper:** "we improve how the field evaluates X" rather than "we invented X".

---

## Part 2 — LLM / AI disclosure evidence (5 verbatim examples + publisher policy)

### Springer Nature / BMC publisher policy

JCheminf inherits these policies (BMC = Springer Nature). Confirmed via Nature Portfolio and Springer Nature editorial policy pages.

- **LLMs cannot be authors.**
- **Disclosure required** for AI used in creative/editorial roles: drafting, rewriting, summarising, content generation.
- **Disclosure NOT required** for minor copy-editing (grammar, spelling, punctuation, tone, readability, formatting).
- **Disclosure location:** Methods section, or alternative section if no Methods. (In published practice, see below: most authors put it in back-matter subsections, not Methods.)
- **Generative AI images NOT permitted** (except as the research subject).
- **Peer reviewers** must NOT upload manuscripts into LLMs.

### Example A — Raymond et al. 2026 (BoUTS feature selection)

- **DOI:** `10.1186/s13321-025-01096-z` · **PMC:** PMC12896148
- **Location:** dedicated back-matter subsection titled **"Language model"**

> "During the preparation of this work, the authors used GitHub Copilot as a programming assistant and ChatGPT4 to generate simple functions. GitHub Copilot was also used to assist in LaTeX typesetting. ChatGPT4 and Claude2 were used to review the manuscript's early drafts and brainstorm ideas, including the name 'BoUTS.'"

**Disclosed scope:** code (Copilot + ChatGPT-4), LaTeX (Copilot), draft review and brainstorming including naming (ChatGPT-4 + Claude-2). **The only example in the sample that discloses code-level AI use.**

### Example B — Shah, Bi & Yang 2025 (food-effects ML review)

- **DOI:** `10.1186/s13321-025-01131-z` · **PMC:** PMC12801895
- **Location:** dedicated back-matter subsection titled **"Use of a generative AI tool"**

> "In preparing this work, the author used ChatGPT 4 to improve the clarity and readability of the English text. After using ChatGPT 4, the author thoroughly reviewed and edited the content as needed. The author takes full responsibility for the content of the publication."

**Disclosed scope:** ChatGPT-4 for clarity/readability of English only. Explicit responsibility statement. Almost-verbatim Elsevier/Springer template wording.

### Example C — Palmacci et al. 2025 (E-GuARD assay-interference detection)

- **DOI:** `10.1186/s13321-025-01014-3` · **PMC:** PMC12042382
- **Location:** dedicated back-matter subsection titled **"Declaration of generative AI and AI-assisted technologies in the writing process"**

> "While preparing this work, the authors used ChatGPT to improve the manuscript's readability. After using this tool/service, the authors reviewed and edited the content as needed and took full responsibility for the content of the publication."

**Disclosed scope:** ChatGPT (unversioned) for readability. Standard template responsibility statement.

### Example D — Steinbeck 2025 (Perspective on open science in cheminformatics)

- **DOI:** `10.1186/s13321-025-00990-w` · **PMC:** PMC11969984
- **Location:** subsection titled **"Disclosure"**

> "Large language models were used to assist the author in researching the timeline, generating ideas, and generating text suggestions on individual historical events, toolkits, and databases. The complete text of this article reflects the author's in-depth knowledge of the subject matter and methods of the work described here."

**Disclosed scope:** unnamed LLMs for research / ideation / text suggestions on historical content. Substantive use, disclosed accordingly.

### Example E — Koyama et al. 2026 (ChemGLaM compound-protein interactions)

- **DOI:** `10.1186/s13321-026-01155-z` · **PMC:** PMC12922394
- **Location:** **Acknowledgements**

> "OpenAI ChatGPT was used to improve the wording of some paragraphs, but not to generate new content."

**Disclosed scope:** ChatGPT for wording only; explicit denial of content generation.

### Cross-paper synthesis (LLM disclosure, n = 5)

- **Where it lives in the article:** 4 different placements across 5 papers. **None in Methods**, despite the policy. Distribution: dedicated back-matter subsection (3 papers), generic "Disclosure" subsection (1), Acknowledgements (1).
- **Subsection heading variants observed:** "Language model" (Raymond) · "Use of a generative AI tool" (Shah) · "Declaration of generative AI and AI-assisted technologies in the writing process" (Palmacci) · "Disclosure" (Steinbeck).
- **Length:** 1 sentence (Koyama; Steinbeck working content) to multi-sentence detailed enumeration (Raymond names 3 tools and 3 tasks).
- **Tool naming:** mixed. Specific versions (ChatGPT 4, Claude2, GitHub Copilot) common; fully generic ("a large language model") also accepted (Steinbeck).
- **Task specificity:** authors consistently say *what* the AI did, not just *that* it was used. Common task verbs: "improve readability / clarity", "generate simple functions", "LaTeX typesetting", "brainstorm ideas / review drafts", "researching the timeline / generating ideas / generating text suggestions".
- **Responsibility statement:** 3 of 5 include explicit "authors take full responsibility" sentence — Elsevier/Springer template wording diffused into JCheminf author pool.

### Search trail (for honesty)

Agent 3 ran PMC full-text and Google Scholar searches with phrase combinations: `"J Cheminform" "ChatGPT was used"`, `"OpenAI ChatGPT was used"`, `"used ChatGPT"`, `"Large language models were used"`, `"AI-assisted technologies"`, `"Declaration" "ChatGPT"`, `"generative AI tool"`, `"preparing this work" ChatGPT`, `"GPT-4" "writing"`, `"Copilot" code`. Checked ~6 other LLM-as-research-subject papers (PMC12490122, PMC12323263, PMC11629536, PMC12255981, PMC12828956, PMC12613558) — none carried an author writing-tool disclosure, consistent with their LLM use being part of the research methodology rather than the writing process.

**Disclosures are rare in JCheminf:** of approximately 158 PMC-indexed J Cheminform articles touching LLMs/generative AI/ChatGPT, only about 5 currently carry an author disclosure. Most JCheminf authors either don't use AI for content work, or use it only for copy-editing (which the Springer Nature policy doesn't require disclosing).

### Minimum-acceptable disclosure template (synthesised from the 5 examples)

Based on observed JCheminf practice, a defensible disclosure has four elements:

1. **Trigger heading** — either a dedicated back-matter subsection ("Use of generative AI", "Declaration of generative AI and AI-assisted technologies in the writing process", "Language model") or place it in Acknowledgements or a "Disclosure" subsection. (Methods is what the policy literally says, but published practice is back matter.)
2. **What tool** — name the product and ideally the version. Generic "a large language model" is acceptable but specific is more common.
3. **What tasks** — concrete: "improve readability", "generate simple functions", "LaTeX assistance", "brainstorm ideas". Skip the copy-editing-only items if you're trying to stay minimal (policy doesn't require disclosing those).
4. **Responsibility statement** — "The author(s) reviewed and edited the content and take full responsibility for the publication." Standard template wording; 3 of 5 examples include it.

Template:
> **Declaration of generative AI in the writing process.** During the preparation of this work, the author(s) used [TOOL + version] to [SPECIFIC TASKS]. After using this tool, the author(s) reviewed and edited the content as needed and take full responsibility for the content of the publication.

---

## Part 3 — Caption-title audit (78 captions across 8 JCheminf articles)

### Audited articles

| # | Article | DOI | PMC | Year |
|---|---------|-----|-----|------|
| 1 | MotifMol3D — Hu et al. | `10.1186/s13321-025-00994-6` | PMC12013036 | 2025 |
| 2 | QSPRpred — van den Maagdenberg et al. | `10.1186/s13321-024-00908-y` | PMC11566221 | 2024 |
| 3 | MolPROP — Rollins et al. | `10.1186/s13321-024-00846-9` | PMC11112823 | 2024 |
| 4 | Matched pairs robustness — Nelen/Tetko et al. | `10.1186/s13321-025-00942-4` | PMC11748845 | 2025 |
| 5 | Effect of Noise on QSAR Predictive Limit — Kolmar & Grulke | `10.1186/s13321-021-00571-7` | PMC8613965 | 2021 |
| 6 | Quantized GNNs — Rasool et al. | `10.1186/s13321-025-01018-z` | PMC12108020 | 2025 |
| 7 | ADMET feature benchmark — Kamuntavičius et al. | `10.1186/s13321-025-01041-0` | PMC12281724 | 2025 |
| 8 | CPSign — Arvidsson McShane et al. | `10.1186/s13321-024-00870-9` | PMC11214261 | 2024 |

Most relevant to the user's paper: Kolmar & Grulke 2021 (noise / QSAR predictive limit), Kamuntavičius 2025 (ADMET benchmarking), Rasool 2025 (GNN benchmarking), Tetko 2025 (robustness).

### Headline finding

**77 of 78 caption titles are descriptive-only. Zero contain a main finding. One is hybrid.**

| Style | Count | % |
|-------|-------|---|
| Descriptive-only | 77 | 98.7% |
| Hybrid (descriptive + capability claim) | 1 | 1.3% |
| Contains-main-finding | 0 | 0.0% |

**Caption word count distribution:** median ~11 words; range 2 to 31; **23 of 78 (29%) exceed BMC's 15-word title cap.** JCheminf authors routinely blow past that guideline.

### Most relevant per-article finding (Kolmar & Grulke 2021 — the user's nearest topic match)

13/13 caption titles are pure descriptive labels — including the central RMSE-vs-noise figures (Fig 3, Fig 4) which are 31 words each and use the form "Plots showing RMSE versus the amount of random error added for the [datasets], for the [algorithms]." The headline finding (e.g. which algorithm degrades most slowly) is **not** in the caption title.

### Per-article details

**Paper 1 — MotifMol3D (10/10 descriptive-only):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "The structure of MotifMol3D framework for metabolic pathway prediction." | 9 |
| Fig 2 | "Distribution of molecules in each metabolic pathway." | 7 |
| Fig 3 | "Construction (A) and distribution (B) of the motif dictionary" | 8 |
| Fig 4 | "Model performance under different motif numbers" | 6 |
| Fig 5 | "Ablation study." | 2 |
| Fig 6 | "Compounds in the degradation pathway of pinene, camphor and geraniol." | 10 |
| Fig 7 | "Compounds in the degradation pathway of flavonoids." | 8 |
| Table 1 | "Comparison results of different methods on the metabolic pathway dataset" | 10 |
| Table 2 | "The top 7 motifs within each pathway" | 7 |
| Table 3 | "Results of the external validation" | 5 |

**Paper 2 — QSPRpred (5/5 descriptive-only; Fig 3 and Fig 4 exceed 15 words):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Visualization of the QSPRpred workflow." | 5 |
| Fig 2 | "Graphical overview of the QSPRpred package architecture." | 7 |
| Fig 3 | "Coefficients of determination (R2) calculated for each replica in different benchmarking runs conducted in Experiment 1" | 16 |
| Fig 4 | "Coefficients of determination (R2) calculated for each replica in different benchmarking runs conducted in Experiment 2" | 16 |
| Table 1 | "Comparison of QSPR modelling tools (adapted from Mervin et al. [53])" | 11 |

**Paper 3 — MolPROP (8/8 descriptive-only):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Graphic of the MolPROP architecture." | 5 |
| Fig 2 | "Latent Embedding Visualization of the MolPROP ESOL Regression Model." | 9 |
| Fig 3 | "Latent Embedding Visualization of the MolPROP BACE Classification Model." | 9 |
| Fig 4 | "Latent Embedding Visualization of the MolPROP ClinTox Classification Model." | 9 |
| Table 1 | "The searched MolPROP hyperparameters spaces" | 5 |
| Table 2 | "MolPROP and baseline model performance on regression tasks" | 9 |
| Table 3 | "MolPROP and baseline model performance on classification tasks" | 9 |
| Table 4 | "MolPROP ablation experiments on regression and classification tasks" | 9 |

**Paper 4 — Matched pairs (6/6 descriptive-only):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Schematic Overview of ΔΔpChEMBL Calculation for Matched Molecular Pairs" | 9 |
| Fig 2 | "Hexbin plots (left) and histograms (right) for the ΔΔpChEMBL IC50 data." | 11 |
| Fig 3 | "Hexbin plots (left) and histograms (right) for the unpruned ΔΔpChEMBL Ki data." | 12 |
| Fig 4 | "Hexbin plots (left) and histograms (right) for the unpruned ΔΔpChEMBL Ki data." | 12 |
| Table 1 | "Pairwise metrics and dataset characteristics for the matched pairs of structural analogs" | 12 |
| Table 2 | "Comparison of matched pair data with the original manuscript results" | 10 |

**Paper 5 — Kolmar & Grulke (13/13 descriptive-only; 6 of 13 exceed 15 words):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Graphical representation of experimental error and prediction error for an arbitrary dataset." | 12 |
| Fig 2 | "The modeling workflow and machine learning pipeline which is used in this work." | 13 |
| Fig 3 | "Plots showing RMSE versus the amount of random error added for the g298_atom and Tox134 datasets, for the Ridge regression, k-Nearest Neighbors (KNN), Support Vector Regression (SVR), and Random Forest (RF) algorithms." | 31 |
| Fig 4 | "Plots showing R2 versus the amount of random error added for the g298_atom and Tox134 datasets, for the Ridge regression, k-Nearest Neighbors (KNN), Support Vector Regression (SVR), and Random Forest (RF) algorithms." | 31 |
| Fig 5 | "Plots showing RMSE versus the amount of added error to the Solv and Tox134 datasets, using Gaussian Process algorithm." | 18 |
| Fig 6 | "Plots showing prediction error versus amount of added error to the G298_atom and Tox134 datasets, using the Gaussian Process algorithm." | 20 |
| Table 1 | "Datasets used in this work, with the number of molecules, endpoint, endpoint units, range, and reference for each" | 18 |
| Table 2 | "Algorithms used in this work and their respective hyperparameter optimization spaces" | 11 |
| Table 3 | "Slopes mnoise and mtrue for each dataset and algorithm" | 9 |
| Table 4 | "Ratios of mnoise/mtrue for each dataset and algorithm" | 8 |
| Table 5 | "Ratios of mnoise/mtrue without Principal Component Analysis" | 7 |
| Table 6 | "Ratios of m to mtrue for the Gaussian Process algorithm" | 10 |
| Table 7 | "Slopes of mean σŷ and σŷ 95% CI versus σ for the Gaussian Process algorithm." | 14 |

**Paper 6 — Rasool quantized GNNs (13/13 descriptive-only):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Overview of predicting molecular properties using the DoReFa-Net algorithm on GNNs" | 11 |
| Fig 2 | "Training and validation loss curves for GNN models trained on different datasets over time" | 14 |
| Fig 3 | "Histograms of weight distribution at layer "conv1" across different quantization levels." | 11 |
| Fig 4 | "Scatter plot of the baseline and quantized models with RMSE metric for the ESOL dataset" | 15 |
| Fig 5 | "Scatter plot of the baseline and quantized models with RMSE metric for the FreeSolv dataset" | 15 |
| Fig 6 | "Scatter plot of the baseline and quantized models with RMSE metric for the Lipo dataset" | 15 |
| Fig 7 | "Scatter plot of the baseline and quantized models with RMSE metric for the QM9 dataset" | 15 |
| Table 1 | "Quantization studies on GNN models" | 5 |
| Table 2 | "SMILES notation, graph visualization, and target values of the first molecule in each dataset" | 14 |
| Table 3 | "RMSE scores for the baseline model and existing literature" | 9 |
| Table 4 | "Full-precision vs. quantized model performance at varying bitwidths" | 8 |
| Table 5 | "GCN framework used in experiments for each domain-specific dataset" | 9 |
| Table 6 | "GIN framework used in experiments for each domain-specific dataset" | 9 |

**Paper 7 — Kamuntavičius ADMET benchmark (18/18 descriptive-only; 9 of 18 exceed 15 words):**

| # | Verbatim first sentence | Words |
|---|---|---|
| Fig 1 | "Model comparison, p-value heatmap according to the Nemenyi test of all 11 features trained individually." | 15 |
| Fig 2 | "Model comparison, p-value heatmap according to the Nemenyi test of 9 features (previous 11 minus rdkit_desc and mordred) trained in combination with rdkit_desc." | 24 |
| Fig 3 | "CatBoost model configuration comparison, p-value heatmap according to the Nemenyi test comparing four differently optimized models." | 17 |
| Fig 4 | "Comparison of baseline (CatBoost default hyperparameters + rdkit_desc features) versus fully optimized (CatBoost optimized hyperparameters + optimized features) on the test set of each dataset." | 24 |
| Fig 5 | "Comparison of baseline (CatBoost default hyperparameters + rdkit_desc features) versus fully optimized (CatBoost optimized hyperparameters + optimized features) on the test set of each dataset." | 24 |
| Fig 6 | "Correlation between the property values for overlapping compounds across the datasets for A hPPB, B HLM, and C solubility." | 19 |
| Fig 7 | "Correlation between the measured and predicted property values for models trained on data from one source and tested data from another across." | 22 |
| Fig 8 | "Performance of models trained with increasing amounts of Biogen data by itself, or combined with TDC or NIH data." | 19 |
| Table 1 | "Dataset descriptions" | 2 |
| Table 2 | "CatBoost hyperparameter grid used in the random search with 20 iterations" | 11 |
| Table 3 | "Maximum combined dataset compositions for transferability experiments" | 7 |
| Table 4 | "Average model rankings in the single-fold experiment" | 7 |
| Table 5 | "Feature performance comparison in the single-fold evaluation experiment, using a catboost model with default hyperparameters" | 16 |
| Table 6 | "Performance (average rank) of iteratively added feature representations in regression datasets" | 11 |
| Table 7 | "Performance (average rank) of iteratively added feature representations in binary classification datasets" | 11 |
| Table 8 | "Performance of four different CatBoost model configurations, with either default or optimized hyperparameters as well as various feature representation combinations" | 19 |
| Table 9 | "Test set evaluation of model configurations at four different degrees of optimization" | 12 |
| Table 10 | "Performance of models at various levels of optimization when trained on data from one source (TDC or NIH) and evaluated on data from a different source (Biogen)" | 26 |

**Paper 8 — CPSign (9 captions, 1 hybrid — the only non-descriptive in the entire 78-caption audit):**

| # | Verbatim first sentence | Words | Style |
|---|---|---|---|
| Fig 1 | "Predictions made with the Lipophilicity model from the evaluation using different significance levels." | 13 | D |
| Fig 2 | "Figure showing features from conformal classifiers." | 6 | D |
| Fig 3 | "Figures showing features from conformal regression predictors." | 7 | D |
| Fig 4 | **"Using the Signatures descriptor allows to map feature importance back to the atoms they originate from..."** | 16 | **H** |
| Fig 5 | "A shows the general workflow of working with CPSign." | 9 | D |
| Fig 6 | "Boxplots for aggregating the results from all tested datasets from the evaluation..." | 12 | D |
| Table 1 | "The 16 classification data sets used in the evaluation, taken from the MoleculeNet benchmark datasets" | 15 | D |
| Table 2 | "The 18 regression data sets used in the evaluation, were the three first datasets comes from the MoleculeNet..." | 18 | D |
| Table 3 | "Runtime comparison summarized across all datasets, were each runtime is calculated to be relative to the CPSign method..." | 18 | D |

### Recommendation re: supervisor's main-idea-in-title preference

**The supervisor's preference is not standard JCheminf practice.** Across 78 captions in 8 articles, zero contain a numerical result or head-to-head claim in the title. Three viable paths:

1. **Follow the supervisor.** Stylistically distinct but scientifically legitimate (this is Nature/Cell-style results-first titling). Tell reviewers "the supervisor preferred it" if asked.
2. **Push back with this audit.** Show that JCheminf community defaults to descriptive labels.
3. **Compromise on hybrid pattern.** Descriptive subject + short interpretive clause — e.g. *"RMSE versus added noise for four QSAR algorithms, showing RF degrades most slowly."* This stays close to convention while inserting the takeaway. Only one paper in the audit (CPSign Fig 4) does this.

---

## Part 4 — Authors' Contributions (gap — no JCheminf-specific evidence yet)

**Status: I have NOT yet pulled JCheminf-specific Authors' Contributions sections.** What follows is the Nat MI corpus finding from earlier (caveat: Nat MI is not JCheminf, and BMC conventions for CRediT taxonomy may differ).

### Nat MI corpus (6 of 9 papers reviewed before user re-direction)

| Paper | Words | Authors | Style notes |
|---|---|---|---|
| Dong & Rudin 2020 | 25 | 2 | Initials, free text, no closing sentence |
| Zeng/Xiang et al. 2022 | 49 | 7 | Initials, free text, no "all authors approved" |
| Ektefaie et al. 2024 | 63 | 7 | Initials, free text, "All authors contributed to writing" closing |
| de Almeida et al. 2025 | 51 | 14 | Initials, free text, no formal closing |
| Mataraso et al. 2025 | 102 | 20 | Initials, free text, "All authors contributed to editing and revising" closing |
| Steyaert et al. 2023 (Perspective) | absent | 7 | Perspective has no Author Contributions section |

**Common patterns (Nat MI):** initials only, free-text prose, NOT CRediT taxonomy categories, 25–100 words.

**Gap:** I have not yet checked JCheminf-specific norms. BMC journals frequently request CRediT taxonomy declarations rather than free-text — need to verify. **Will not draft Authors' Contributions until JCheminf-specific evidence is in.**

---

## Part 5 — Two candidate Sci Contrib rewrites for the user to choose between

These are built from the user's actual paper claims (Intro at `paper.tex` L153–162) plus the pattern evidence above. Both 3 sentences, ~75–95 words, lead with the field-level claim.

### Candidate A — Artifact-led (QSPRpred / MotifMol3D / CPSign / Gadaleta style)

> "NoiseInject benchmarks QSAR noise robustness across six controlled noise strategies, revealing that model architecture, not molecular representation, is the dominant driver of robustness. In contrast to prior noise studies that fixed model or representation, NoiseInject's full cross-product shows that a model's relative noise ranking transfers across noise types, datasets, and representations. We further show that per-sample uncertainty tracks injected noise only for models that learn an explicit noise parameter — providing a principled criterion for selecting probabilistic models in noisy regimes."

### Candidate B — Insight-led (MolPROP / Masood / Kamuntavičius style)

> "This work shows that model architecture, not molecular representation, is the dominant driver of QSAR noise robustness, and that a model's relative robustness transfers across noise types and datasets. We further demonstrate that meaningful per-sample uncertainty quantification requires architectures that learn an explicit noise term (NGBoost, Gaussian Processes), rather than relying on generic Bayesian transformations. We release NoiseInject, an open-source framework that operationalises these findings for any scikit-learn- or PyTorch-compatible model."

Neither has been written into `paper.tex`. **Direction must be approved before any draft is applied.**

---

## Part 6 — Honest remaining gaps

1. **No JCheminf-specific Authors' Contributions example yet.** Need to audit ~5 recent JCheminf primary-research articles' back-matter to confirm whether CRediT taxonomy is required or free-text prose is acceptable, and to capture style/length conventions.
2. **JCheminf in-article LLM-disclosure sample is 5 of an estimated ~158 candidate articles.** That's enough to identify the dominant template (Elsevier/Springer responsibility-statement wording, placed in back matter not Methods), but the sample size is small. If you want more breadth, another agent pass could push to 10+.
3. **Caption-title audit covers 8 papers / 78 captions.** Style finding (98.7% descriptive-only) is robust, but if you want tighter bounds, an extended 20-paper audit would help.
4. **PMC fetch of Masood 2025 (Example 6) returned "pharmaceutical articles" as the closing phrase**, likely an OCR artefact for "applications". Verify directly on PMC12020163 before quoting.

---

## Sources

### JCheminf editorial
- [Editorial: Are new ideas harder to find? — Bajorath, J Cheminform 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10789001/)

### Scientific Contribution verbatim examples (10)
- [MotifMol3D — Hu et al. 2025 (PMC12013036)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12013036/)
- [QSPRpred — van den Maagdenberg et al. 2024 (PMC11566221)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11566221/)
- [MolPROP — Rollins et al. 2024 (PMC11112823)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11112823/)
- [Syk QSAR+RL — Zavadskaya et al. 2025 (PMC11998205)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11998205/)
- [TK/PC benchmark — Gadaleta et al. 2024 (PMC11674477)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11674477/)
- [BERT + Bayesian AL — Masood et al. 2025 (PMC12020163)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12020163/)
- [ADMET feature benchmark — Kamuntavičius et al. 2025 (PMC12281724)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12281724/)
- [Auxiliary learning GNN — Dey & Ning 2024 (PMC11270959)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11270959/)
- [CPSign — Arvidsson McShane et al. 2024 (PMC11214261)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11214261/)
- [UMAP splits — Guo et al. 2025 (PMC12153141)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12153141/)
- [Counter-example: Matched pairs — Nelen et al. 2025 (PMC11748845)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11748845/)

### LLM disclosure verbatim examples (5)
- [BoUTS — Raymond et al. 2026 (PMC12896148)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12896148/)
- [Food-effects ML review — Shah et al. 2025 (PMC12801895)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12801895/)
- [E-GuARD — Palmacci et al. 2025 (PMC12042382)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12042382/)
- [Open-science Perspective — Steinbeck 2025 (PMC11969984)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11969984/)
- [ChemGLaM — Koyama et al. 2026 (PMC12922394)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12922394/)

### Caption-title audit (additional articles, 78 captions)
- [Kolmar & Grulke 2021 noise paper (PMC8613965)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8613965/)
- [Quantized GNNs — Rasool et al. 2025 (PMC12108020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12108020/)

### Policy
- [JCheminf research-article submission guidelines](https://jcheminf.biomedcentral.com/submission-guidelines/preparing-your-manuscript/research)
- [Springer Nature editorial policies (AI/LLM)](https://www.springernature.com/gp/policies/editorial-policies)
- [Nature Portfolio AI policy](https://www.nature.com/nature-portfolio/editorial-policies/ai)
