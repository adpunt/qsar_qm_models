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
2026-09-27. Re-check after every Overleaf download: paper.tex changes and this column does not.

The thesis keeps its own copies and its own index at `KIRBy/thesis_appendix/noise/INDEX.md`.

| Name | Kind | Title | Made by | Paper file | Status | Cited in paper.tex | Notes |
|---|---|---|---|---|---|---|---|
| hyperparameters_default | table | Default hyperparameters for every model configuration | scripts/generate_supp_table1.py | 1 | current | 214, 228, 708 | Table A |
| hyperparameters_tuned | table | Tuned settings that replace a default, per dataset | scripts/generate_supp_table1.py | 1 | current | 214 | Table B |
| pdv_descriptors | table | The 200 descriptors that make up PDV | scripts/generate_supp_table1.py | 1 | current | 210 ("Additional file 1, Table C") | Table C |
| model_redundancy | table | Pairwise Spearman correlations between model NDS profiles | scripts/generate_supp_tables.py (old study) | 2 | stale | 349 ("Additional files 2--4") | paper.tex:348 TODO says these tables are not in the additional files; they are, but under NDS |
| representation_redundancy | table | Pairwise Spearman correlations between representation NDS profiles | scripts/generate_supp_tables.py (old study) | 3 | stale | 349 | |
| model_icc | table | ICC(1,1) for model pairs | scripts/generate_supp_tables.py (old study) | 4 | stale | 349 | |
| excluded_configurations | table | Configurations excluded from the robustness analysis | scripts/generate_supp_tables.py (old study) | 5 | stale | 324, 531 | Text says excluded at clean R² < 0.3; this table uses R² ≤ 0.6 |
| ecfp4_overview | figure | Global overview of noise robustness on ECFP4 | scripts/generate_paper_figures_v2.py (old study) | 6 | stale | not cited | Figure is fig1_supp_ecfp4_overview.png |
| clean_r2_vs_robustness | figure | Clean R² against NDS, PDV, Gaussian | scripts/generate_paper_figures_v2.py (old study) | 7 | stale | not cited | Figure is fig3_ranking_consistency.png |
| bayesian_transformation_tests | table | Wilcoxon tests of the Bayesian transformations by representation | scripts/generate_supp_tables.py (old study) | 8 | stale | not cited | paper.tex:635 cites Additional file 8 for variant_models_assay, not for this |
| uncertainty_metrics_ecfp4 | table | Uncertainty metrics for probabilistic models on ECFP4 | scripts/generate_supp_tables.py (old study) | 9 | stale | not cited | |
| assay_variance_decomposition | figure | ANOVA variance decomposition on the three assay datasets | scripts/generate_paper_figures_v2.py (old study) | 10 | stale | 694 (commented out) | Figure is fig_validation_anova.png |
| rf_vs_qrf_assay | table | RF against QRF on the three assay datasets | scripts/generate_supp_tables.py (old study) | 11 | stale | 696 (commented out) | |
| variant_models_assay | figure | The six variant models on the three assay datasets, beside Figure F8 | — | — | planned | 635 ("Additional file 8") | No figure found yet |
| counterparts_assay | table | Deterministic against probabilistic counterparts on logD, Caco-2, hERG K_i | scripts/run_paper_analysis.py | — | planned | 754–755 (comment: "go to an Additional file") | Tables exist: results/decisions_arc/tables/T11_counterparts_{logd,caco2,herg}.tex |
