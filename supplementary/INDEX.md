# Supplementary items: the index

One row is one supplementary table or figure (an *item*). The name in the first column is
permanent. It is the file name in `items/`, the LaTeX label (`supp:<name>`), and the name the
thesis copy keeps. Numbers are never used as names.

`scripts/build_additional_files.py` reads the table below. It prints the Additional files in
the order of the **Paper file** column. Rows that share a number print as one Additional file,
in the order they appear here. A row with `—` in that column is left out of the paper build.

**Status** says what the content is:
- `current`: built from the study as it stands now.
- `stale`: built from the earlier version of the study (six noise strategies, NDS, mol2vec,
  R² ≤ 0.6 gate). Kept so the numbering holds until it is replaced or dropped.
- `planned`: the paper text cites it but no item file exists yet.

**Cited in paper.tex** records where the text points at it, checked against paper.tex on
2026-09-30. Re-check after every Overleaf download: paper.tex changes and this column does not.

The thesis keeps its own copies and its own index at `KIRBy/thesis_appendix/noise/INDEX.md`.

| Name | Kind | Title | Made by | Paper file | Status | Cited in paper.tex | Notes |
|---|---|---|---|---|---|---|---|
| hyperparameters_default | table | Default hyperparameters for every model configuration | scripts/generate_supp_table1.py | 1 | current | 214, 228, 708 | Table A |
| hyperparameters_tuned | table | Tuned settings that replace a default, per dataset | scripts/generate_supp_table1.py | 1 | current | 214 | Table B |
| pdv_descriptors | table | The 200 descriptors that make up PDV | scripts/generate_supp_table1.py | 1 | current | 210 ("Additional file 1, Table C") | Table C |
| model_redundancy | table | Spearman correlations between models on clean R²; the ANOVA filter | scripts/run_paper_analysis.py (T13) | — | stale | 349 ("Additional files 2--4") | Superseded 2026-09-28 by the family ANOVA; the correlation filter was dropped; Dropped from the paper 2026-09-30: stale and not cited |
| representation_redundancy | table | Spearman correlations between representations on clean R²; the ANOVA filter | scripts/run_paper_analysis.py (T13) | — | stale | 349 | Superseded 2026-09-28 by the family ANOVA; the correlation filter was dropped; Dropped from the paper 2026-09-30: stale and not cited |
| model_icc | table | ICC(1,1) between models on clean R² | scripts/run_paper_analysis.py (T13) | — | stale | 349 | Superseded 2026-09-28 by the family ANOVA; the correlation filter was dropped; Dropped from the paper 2026-09-30: stale and not cited |
| family_pairs_models | table | Candidate model families: pairs within split-to-split variation | scripts/run_paper_analysis.py (T13) | 3 | current | Methods, ANOVA paragraph | Made 2026-09-28 |
| family_pairs_representations | table | Candidate representation groups: pairs within split-to-split variation | scripts/run_paper_analysis.py (T13) | 5 | current | Methods, ANOVA paragraph |  |
| family_tukey | table | Tukey HSD between the members of each model family | scripts/run_paper_analysis.py (T13) | 4 | current | Methods, ANOVA paragraph |  |
| excluded_configurations | table | Configurations excluded from the robustness analysis | scripts/run_paper_analysis.py (T16) | 2 | current | Methods (AUC_norm), fig:grid and fig:assay captions | Rewritten 2026-09-30 from the current study; lists only folds with no AUC_norm |
| ecfp4_overview | figure | Global overview of noise robustness on ECFP4 | scripts/generate_paper_figures_v2.py (old study) | — | stale | not cited | Figure is fig1_supp_ecfp4_overview.png; Dropped from the paper 2026-09-30: stale and not cited |
| clean_r2_vs_robustness | figure | Clean R² against NDS, PDV, Gaussian | scripts/generate_paper_figures_v2.py (old study) | — | stale | not cited | Figure is fig3_ranking_consistency.png; Dropped from the paper 2026-09-30: stale and not cited |
| bayesian_transformation_tests | table | Wilcoxon tests of the Bayesian transformations by representation | scripts/generate_supp_tables.py (old study) | — | stale | not cited | paper.tex:635 cites Additional file 8 for variant_models_assay, not for this; Dropped from the paper 2026-09-30: stale and not cited |
| uncertainty_metrics_ecfp4 | table | Uncertainty metrics for probabilistic models on ECFP4 | scripts/generate_supp_tables.py (old study) | — | stale | not cited | Dropped from the paper 2026-09-30: stale and not cited |
| assay_variance_decomposition | figure | ANOVA variance decomposition on the three assay datasets | scripts/generate_paper_figures_v2.py (old study) | — | stale | 694 (commented out) | Figure is fig_validation_anova.png; Dropped from the paper 2026-09-30: stale and not cited |
| rf_vs_qrf_assay | table | RF against QRF on the three assay datasets | scripts/generate_supp_tables.py (old study) | — | stale | 696 (commented out) | Dropped from the paper 2026-09-30: stale and not cited |
| variant_models_assay | table | The six variant models on the three assay datasets, beside Figure F8 | scripts/run_paper_analysis.py (T17) | 8 | current | fig:assay caption | Made 2026-09-30 as a table |
| counterparts_assay | table | Deterministic against probabilistic counterparts on logD, Caco-2, hERG K_i | scripts/run_paper_analysis.py | — | stale | 754–755 (comment: "go to an Additional file") | Replaced 2026-09-30 by counterparts_logd, counterparts_caco2, counterparts_herg (one item is one table) |
| variance_intervals | table | 95% intervals for every share in the decomposition tables | scripts/run_paper_analysis.py (T3i) | 6 | current | tab:variance, tab:three_outcomes_* and tab:variance_clean captions ("Additional file 6") | Made 2026-09-30 |
| robustness_grid_qm9_grouped_wider | figure | Robustness grid on QM9 under grouped-wider noise, the condition fig:grid leaves out | scripts/run_paper_analysis.py (F3b) | 7 | current | fig:grid caption | Made 2026-09-30 |
| counterparts_logd | table | Base models against their probabilistic counterparts, logD | scripts/run_paper_analysis.py (T11) | 9 | current | Counterparts subsection | Made 2026-09-30 |
| counterparts_caco2 | table | Base models against their probabilistic counterparts, Caco-2 | scripts/run_paper_analysis.py (T11) | 9 | current | Counterparts subsection | Made 2026-09-30 |
| counterparts_herg | table | Base models against their probabilistic counterparts, hERG K_i | scripts/run_paper_analysis.py (T11) | 9 | current | Counterparts subsection | Made 2026-09-30 |
