# NoiseInject Revision — All-Fronts Status

**Last updated: 2026-08-17.** Master admin doc for the paper revision. This tracks *everything* across the whole effort so no thread is dropped. The detailed find/replace prose lives in `REVISION_GUIDE.md`; this file is the map.

## Ground rules (do not violate)
- **I never edit `paper.tex`** — the author makes every paper edit. I only edit `REVISION_GUIDE.md` and this file.
- **Every finding must be reported for QM9 AND the validation datasets** (LogD, Caco-2 Efflux, hERG Ki), not QM9 alone.
- **No numbers from memory** — every value traces to a file the pipeline wrote or a computation actually run.
- **Never average across the 6 noise strategies** to make a claim unless all 6 are shown.
- Local `paper.tex` is a June-30 snapshot and has diverged from the live Overleaf in the Results section — anchor edits to the live text.

---

## 1. Two headline changes
- **C1 — metric:** old "Noise Degradation Slope (NDS)" → **AUC_norm** (normalised area under the R²-retention curve, higher = more robust). Sole metric. No Weibull.
- **C2 — uncertainty:** measured **within each σ** (and per strategy). This **reversed** the old per-sample finding → it is now a **NULL** (see §3, F1).

---

## 2. Paper section-by-section status

| Section | Status | Notes |
|---|---|---|
| Abstract | ✅ drafted in guide | symbol-free; robustness = "proportion retained"; **one-line null** for uncertainty. |
| Scientific Contribution | ✅ drafted | per-sample uncertainty sentence **deleted entirely**; 2-sentence contribution. |
| Introduction (RQs) | ✅ drafted | single Q2 sentence reworded (as a question — compatible with the null). |
| Methods — metric def | ✅ drafted | paste-ready AUC_norm LaTeX; trapezoidal rule confirmed standard (no cite needed). |
| Methods — excluded configs | ✅ | "48 configs, baseline R²<0.3", convention-matched `(Additional file~5)`. |
| Results 4.1 — ANOVA | ⚠ table STALE | text has 83.6/77.4 (correct); **table body L396–401 still old (48.7/41.0)** → replace (guide has correct table). Residual-dominance confirmed REAL (§3 F7). |
| Results 4.2 — robustness ranking | partial | model table + Kendall W OK; fix W 0.92→**0.9121**, SVM is 5th not top-2. **Rep claims need rework** (§3 F3). |
| Results 4.3 — uncertainty | 🔴 REWRITE | currently built on the dead 0.485 detector. Guide has a placeholder. **New framing available** (§3 F1/F5/F6). Pending author decision. |
| Results 4.4 — validation | ⚠ open | NDS→AUC_norm; add decoupling nuance (§3 F4); **validation ANOVA (Add. file 10) is a saturated 1-obs-per-cell design, residual=0 → refit needed**. |
| Conclusion | partial | fix W; drop per-sample detection sentences; rework PDV rep claims; delete duplicated L381. |
| Back matter | open | Abbreviations: drop "NDS"; 11 stale "NDS" mentions (§4 T10). |

---

## 3. Master findings log (QM9 + validation)

| # | Finding | QM9 | Validation | Verdict |
|---|---|---|---|---|
| F1 | Per-sample uncertainty tracks injected noise | NULL (max within-σ |ρ|=0.129 across 143 combos; 0.485 was a strategy-pooling artifact) | pending (workflow) | **NULL** |
| F2 | Population uncertainty rises with σ (Kolmar link) | holds | pending | HOLDS |
| F3 | Representation robustness | reps barely separate except under **threshold (spread 0.090) & valprop (0.070)**; outlier 0.013 / hetero 0.021 / quantile 0.029 / legacy 0.040. No universal best rep. | rep-separating strategy is **dataset-specific**: threshold on LogD, valprop/legacy on hERG, **legacy/quantile/outlier on Caco-2** (spread 0.22!). | **BOTH, but which strategy separates depends on dataset** |
| F4 | Robustness vs clean-data accuracy (coupling) | NOT locally verifiable (baseline CSV absent); server .out gave within-Gaussian +0.046 | **SIGN-FLIPS by dataset**: hERG **+0.30** (coupled), LogD **−0.31** (anti-coupled, "more to lose"), Caco-2 **+0.10** (~none) | **dataset-specific, sign-flipping** — earlier "+0.525 ADME coupled" was a POOLED artifact (my error) |
| F5 | Uncertainty predicts a model's own ERROR | yes: QRF 0.270, NGBoost 0.260, GP 0.184 (table4, legacy) | **YES — QRF is the strongest error-tracker on ALL 3 datasets** (pooled Unc-Error ρ: hERG 0.20–0.29, Caco2 0.24–0.29, LogD 0.15–0.24). All combos ρ>0 (min 0.077). NGBoost pooled weaker (~0.08–0.13) but recovers at σ=0 (0.16–0.36). | **REPLICATES: QRF best error-ranker both studies** |
| F6 | Calibration | NGBoost best-calibrated (ECE 0.111); QRF ECE 0.163; GP 0.231; cov@2σ ~0.95 | **REPLICATES: NGBoost has the LOWEST ECE on validation too** (hERG 0.13–0.18, Caco2 0.23–0.26, LogD 0.24–0.29 vs QRF/GP higher). Validation GP=RBF-on-PDV (NOT QM9's Tanimoto gauche) → over-wide intervals, high ECE (0.32–0.38). cov@2σ 0.83–0.98 (near nominal). **VBLL NOT broken** — QM9 ECE~5 was raw-units scale artifact; no validation BNN/VBLL data. |
| F5/F6 | **Ranking vs magnitude trade-off** | QRF wins Unc-Error, NGBoost wins ECE | same: QRF best error-RANKER, NGBoost best-CALIBRATED magnitude | **NEW cross-study headline — holds on QM9 + all 3 validation sets** |
| F7 | ANOVA residual-dominance (Outlier 83.6 / Hetero 77.4) | REAL, robust to roster(7)/SS-type; = run-to-run variance (cell-means Model 55/71%) | n/a | CONFIRMED |
| F8 | NGBoost most robust; NN-β (mlp) least | holds (Kendall W=0.9121) | mixed (dataset-dependent) | QM9 holds |
| F9 | PDV best clean-data accuracy | YES (baseline 0.857, highest rep) | mid-pack on ADME | QM9 |
| F10 | Binary PDV | worse than continuous PDV (local hERG/LogD test) → **DROPPED** | n/a | DROPPED |
| F11 | "Noise types behave differently" | supported on 3 axes: model spread, representation, uncertainty quality | partially | **emerging organizing theme** |

---

## 4. Thread tracker (open items)

| T | Thread | Status |
|---|---|---|
| T3 | Replace stale ANOVA table body (paper L396–401) | LOGGED — paste guide table |
| T4 | Decoupling: keep on QM9, add ADME nuance | DONE (nuance to write) |
| T5/T7 | Rewrite rep claims L462/L493/L567 → F3 shape | DECIDE |
| T6 | Noise-types-differ (representation) | DECIDE how prominent |
| T8 | Cross-strategy averaging (9-row table) | model table KEEP; rep/validation/ICC fix |
| T9 | Kendall W 0.92→0.9121; SVM not top-2 | LOGGED |
| T10 | 11 stale "NDS" lines | rename-only (L387/464/556); recompute (L470/475 Wilcoxon, L495 top-10); content-rework (L462/493/567/573) |
| T12 | Delete duplicated L381 | LOGGED |
| T14 | valprop corrupted in direct dump (catastrophic filter skipped) | KNOWN — guard/filter |
| T16 | **Uncertainty section reframe** → "predicts error not noise; NGBoost/QRF/GP trustworthy" — QM9-ONLY, scope it. **NOT** "VBLL broken" (scale artifact) | NEW — DECIDE |
| T21 | **DONE — validation uncertainty now processed.** Data was LOCAL all along (`KIRBy/tests/results_server/validation_rerun/<rep>_<dataset>/<dataset>/MODEL_REP_uncertainty_values.csv`), not server-only: 3 datasets × 4 reps, models QRF+NGBoost everywhere, GP on PDV only (raw label units, Gaussian/legacy only, no per-sample injected_noise). Added `load_validation_uncertainty()` + `create_validation_uncertainty_table()` to `generate_paper_figures_v2.py`, wired into main(); emits `table_validation_uncertainty.csv` (27 rows). Ran locally → F5/F6 now have validation results (see F5/F6 rows). | ✅ DONE |
| T22 | Validation coverage thin: only **4 reps** (missing mol2vec/morgan/smiles/pdv), **7 classic models** (no BNN/VBLL/MLP), continuous_pdv missing dnn+svm, hERG under-filled (153/156) | GAP — decide add limitations paragraph vs fill |
| T23 | QM9 QRF robustness asymmetry (in uncertainty table, absent from robustness table) | GAP — confirm on server |
| T17 | **Validation ANOVA (Add. file 10) refit** (saturated, residual=0) | OPEN — needs per-fold replicates |
| T18 | **Reproducible `scripts/deep_analysis.py`** — ✅ BUILT & runs (pandas/numpy/scipy). Covers QM9+validation, valprop-filtered; emits 10 `deep_*.csv` (rep/model robustness by strategy, baseline-vs-robustness, uncertainty by model/strategy, gaps). | DONE |
| T20 | VBLL "broken" | RESOLVED — scale artifact (ECE in raw error units), not a failure. VBLL median ECE ~0.24. |
| T19 | Figures: re-run robustness figs; **new uncertainty figure** (predicts-error + calibration); optional rep×noise-type figure | OPEN |
| T20 | VBLL-broken root cause (real failure vs scale bug) | OPEN — investigate |

---

## 5. Data & reproducibility status
- **Local (verified):** QM9 robustness per model×rep×strategy (`table2_supp_auc_all_reps.csv`); QM9 uncertainty (`table4_supp_uncertainty_by_strategy_rep.csv`); validation robustness w/ baseline (`table_validation_auc_full.csv`); validation uncertainty (`table_validation_probabilistic.csv` — columns being verified); KIRBy raw SMILES+targets (`KIRBy/tests/data_cache/`).
- **Server dump done:** `qm9_auc_with_baseline.csv` (per model×rep×strategy auc_norm + baseline_r2) — **valprop column corrupted**; may/may not be scp'd local yet.
- **Not local (server only):** raw QM9 per-run `anova_*.csv`.
- **Reproducibility problem:** tables are scattered across ad-hoc CSVs. Fix = `scripts/deep_analysis.py` (T18) as the single source.

## 6. Server / infra
- Host `gateway.arc.ox.ac.uk` → hop to `arc-login` (SLURM lives there, not the gateway).
- Env: `conda activate env_test` (fast) — avoid `. setup.sh` (reinstalls). Account `--account=stat-cadd`. `devel`=10min, use `short`/`long` for real jobs. Helper: `KIRBy/tests/slurm_scripts/where_to_submit.sh`.
- Pending server job: value-proportional-filtered baseline dump + any recomputes (Wilcoxon Δ, top-10) — all derivable once `qm9_auc_with_baseline.csv` (filtered) is local.

## 7. Open decisions for the author
1. Uncertainty section reframe (T16) — biggest upgrade: null → "useful for triage, not noise; NGBoost/QRF trustworthy; VBLL broken."
2. Representation rewrite (T5/T6/T7) — noise-type-dependent, both studies.
3. Make "different noise types behave differently" (F11) an organizing theme?
4. Validation-section decoupling nuance (F4).
5. Whether to add figures/tables for F3, F5, F6.

---

# 8. MASTER CROSS-REFERENCE — script ↔ revision (grounded 2026-08-17, workflow wwcgxxd3i)

Every revision TODO cross-checked against `scripts/generate_paper_figures_v2.py` (read in full) and local data. 33 items, all file:line-grounded. **Three standing gaps the author flagged:** (1) of ~15 script-side bullets only the validation-uncertainty functions were added — baseline_r2 save, per-strategy decoupling table, rep×strategy table, mean fixes, validation-ANOVA refit, valprop self-guard, and deep_analysis fold-in are all untouched code; (2) the validation-uncertainty addition is **Gaussian-only** and adds ZERO strategy-varying analysis; (3) validation is more covered but not *properly* — robustness tables still collapse strategy+datasets, ANOVA still residual-0 saturated, coverage gaps remain (4 reps, no BNN/VBLL).

## 8A. Over-averaging in the script (the "banned means")
The pass separated **cited** (load-bearing) from **latent** (written but never `\ref`'d in paper.tex). A fully-disaggregated sibling already exists for most.

| Artifact | file:line | Averages away | Cited in paper? | Fix |
|---|---|---|---|---|
| `fig_validation_combined.png` Panel A | L1713/1720 | reps × 6 strategies → one bar/model×dataset; QM9 col = mean over rep×strategy | **YES (L551)** | re-facet by strategy OR relabel as explicit strategy-mean + re-point claims to `fig_validation_overview` (strategies shown). Also caption still says "NDS". |
| `fig_validation_combined.png` Panel B | L1730/1731 | reps × strategies × **3 datasets** → 1 pt/model | **YES (L556 caption)** | split external axis per dataset OR demote to per-dataset from `table_validation_auc_full.csv`. Caption "NDS" too. |
| `table_validation_auc.csv` | ~~L1491-1497~~ | strategy collapsed + MEAN across 3 datasets | No `\ref` | ✅ **DONE (D2, 2026-08-19)** — collapsed pivot no longer emitted in v2; `table_validation_auc_full.csv` is the only validation auc table. |
| `table_validation_probabilistic.csv` | ~~L1690~~ | rep × strategy per dataset (RF vs QRF) | **YES — it is Additional file 11**, cited at paper L560. My earlier "No `\ref`" was wrong. | ✅ **DONE (D2, 2026-08-19)** — rebuilt as dataset × 6 strategies with rep held at `PRIMARY_REP` (continuous_pdv); 18 rows, nothing averaged. **Paper consequence:** L560 "QRF was consistently less robust than RF on every external data set" is an averaging artifact — QRF is ahead on Caco-2 under outlier (+0.160), quantile (+0.034), threshold (+0.017) and hetero (+0.004). Needs a §8D edit. |
| `table_supp_icc.csv` | L2454 | 6 strategies before ICC | supp only | keep as supp (ICC needs a per-rep scalar) OR compute per-strategy — DISCUSS. |
| `table4_uncertainty_metrics.csv` (`all`) | ~~L3386~~ | reps pooled (legacy only) | No (per-rep `_<rep>` + table4c cited) | ✅ **DONE (D2, 2026-08-19)** — `all` no longer emitted in v2; per-rep tables kept. (Its worst column, ECE, is gone from every table — see REVISION_GUIDE §"Metric removal — ECE".) |
| `table2_*_pdv` MEAN/STD/Mean_Rank | L3271/3291/3298 | 6 strategies (but all 6 ARE columns) | table cited | **OK per GR8** (all 6 shown) — keep MEAN as transparent summary; Kendall W justifies. |

## 8B. Analyses missing from the script (must ADD, then retire side scripts)
- **B1 `save_qm9_baseline_table`**: `calculate_robustness` computes `baseline_r2` for all 6 strategies (L1848) but **auc_df is never dumped**; only a 5-model PDV/Gaussian slice reaches `table3` (L3328). Add a `table_qm9_baseline_r2.csv` (long, all strategies). Trivial — value already in memory. Backs F9/T13/T15. *(blocked: needs ARC run for raw QM9)*
- **B2 `create_decoupling_tables`**: the ONLY baseline↔robustness analysis is `create_figure3` (L2911) — figure-only, PDV+legacy, single Spearman ρ in a textbox, no CSV, no per-strategy/per-dataset. Add per-strategy (QM9) + per dataset×strategy (validation) Pearson+Spearman table. The +0.525 is a **pooled artifact** — report only as an explicitly-labelled pooled row. Backs T4/F4.
- **B3 `create_rep_strategy_auc_table`**: **no table has representation as the aggregation unit.** Rep-separation claim (T6) has no backing table in the script — only orphaned `deep_qm9_rep_robustness_by_strategy.csv`. Add rep×strategy (all 6 shown) + per-strategy rep-spread. Backs T5/T6/T7/F3.
- **B4 fold `deep_analysis.py` in + delete it**: it's a standalone entrypoint (not in any regen path); its numbers (F3/F4/T6) are uncertified and its QM9 baseline row is **n=0 GAP locally**. Folding lets it read in-memory `auc_df` (correctly valprop-filtered). This is the reproducibility fix. **DR3.**

## 8C. Validation
- **VAL-FLOW**: validation DOES flow through all 4 analyses (robustness/ANOVA/probabilistic/uncertainty) — but each narrows the design (ANOVA legacy-only+fold-collapsed; probabilistic RF-vs-QRF only; uncertainty Gaussian+3 models).
- **VAL-ANOVA-REFIT (T17)**: residual η²=0 is a **saturation artifact** — `calculate_validation_auc` (L1384) averages over the 5 folds before auc_norm → 1 obs/cell. **Fix is a SCRIPT EDIT, no rerun** — per-fold replicates are local in `all_results.csv` (fold ∈ {0..4}). Compute auc_norm per fold → genuine within-cell variance.
- **VAL-UNC not actually emitted**: `create_validation_uncertainty_table` is wired (L4173) but `table_validation_uncertainty.csv` **does not exist locally** — never run through the pipeline. My "27 rows DONE" was an in-memory spot-check. Must run the script.
- **VAL-STALE-GP-MISSING**: GP is in raw KIRBy (Caco-2 baseline 0.524, LogD 0.784, passes gate) but **absent from local `table_validation_auc_full.csv`** (outputs predate GP being added 4 Mar). A fresh run adds GP and changes the validation ANOVA roster.
- **VAL-UNC-GAUSSIAN-ONLY / MODEL-COVERAGE**: uncertainty is Gaussian-only (KIRBy saved no strategy/injected_noise col) and only QRF+NGBoost(4 reps)+GP(PDV) — no BNN/VBLL. Frame as a targeted F5/F6 replication + limitations paragraph; filling needs a KIRBy rerun (likely decline).

## 8D. Paper-text edits backed by a CURRENT script output (mechanical once agreed)
- **T3** ANOVA table body (paper L396-401): every Robustness cell is NDS-stale. Correct values in `table1_anova_summary.csv`: Gaussian 43.8/5.2/16.9/34.2; Quantile 36.8/4.4/15.1/43.7; Threshold 54.7/7.9/22.6/14.8; Hetero 14.0/0.7/8.0/**77.4**; Value-prop 52.5/6.0/19.9/21.6; Outlier 10.3/0.2/5.9/**83.6**. Bold factor flips for Gaussian/Quantile/Hetero.
- **T10** caption L387 "NDS"→AUC_norm; L464/L556 word-swaps.
- **T9** W 0.92→0.9121 at L573; SVM is 5th not top-2 (L460/L573).
- **T10-Wilcoxon** (L470-483): recompute Δ from `table3_wilcoxon_tests.csv` — **VBLL-α flips to NON-significant (p=0.25)**; changes a claimed result.
- **T10-top10** (L495): "no NN in top-10" is **FALSE** under AUC_norm (dnn_bnn_full/morgan = 0.930 is #1; NGBoost ×8). Restate or restrict roster (check the 0.930 outlier first).
- **T10-PDV** (L462/L493/L567): PDV is mid-cluster under AUC_norm, not most/least; the 91%/72% numbers don't reproduce. Consolidate to F3 framing.

## 8E. Data / reproducibility
- QM9 raw `anova_*.csv` **server-only** → full regen (and B1/B2 QM9 legs) need ARC. Validation raw is local.
- **valprop guard (T14)**: main pipeline is SAFE — `filter_catastrophic_iterations` (L4058) precedes `calculate_robustness` (L4141); local valprop column is clean. Corruption is confined to direct dumps (`qm9_auc_with_baseline.csv`) that skip the filter. **T14-HARDEN (discuss)**: add the filter inside `calculate_robustness` so every caller is safe by construction.
- `qm9_auc_with_baseline.csv` (baseline carrier) is **absent locally + valprop-corrupted** → regenerate through the filtered pipeline on ARC, not the direct dump.

## 8F. DECISIONS FOR THE AUTHOR (resolve before editing paper.tex)
1. **fig_validation_combined (cited, over-averaged):** re-facet by strategy, or relabel as explicit strategy-mean and re-point L551 claims to `fig_validation_overview`?
2. ~~**Latent averaged tables**~~ ✅ **RESOLVED (D2, 2026-08-19).** `table_validation_auc` and `table4 all` dropped; `_probabilistic` was NOT latent (it is Additional file 11) and is rebuilt as dataset × 6 strategies at `PRIMARY_REP`. Implemented in v2 only — **still open:** `run_figures.sh` invokes v1 (`generate_paper_figures.py`), so either re-point it at v2 or mirror the change.
3. **deep_analysis.py:** fold A/B/C into the main script and `git rm` it? Keep `deep_` CSV prefix or rename to `table_qm9_*`?
4. **QM9 decoupling correlation:** compute across all reps within each strategy, or PDV-only? (pooled +0.525 → labelled pooled row only.)
5. **rep×strategy table roster:** ANOVA rep set or all 9 reps?
6. **validation ANOVA refit:** per-fold within legacy only (mirrors QM9), or per-strategy all-6 side-by-side?
7. **validation uncertainty:** accept Gaussian-only + limitations paragraph, or KIRBy rerun for per-strategy/BNN-VBLL? (recommend accept + caveat.)
8. **T14-HARDEN:** self-guard `calculate_robustness`, or forbid direct dumps?
9. **SVM wording; VBLL-α now non-significant; "no NN in top-10" false:** confirm the reframes.
10. **QM9 baseline provenance:** run ARC dump to certify +0.046 / 0.671 / 0.857 locally, or cite as provisional?
