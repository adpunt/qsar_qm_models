# DISCUSSION TRACKER — must agree BEFORE any script change

Rule: nothing in `scripts/generate_paper_figures_v2.py` gets changed until the relevant item below is **DECIDED**. Status updated every chat.
Legend: ⬜ not discussed · 🔶 discussing · ✅ decided (decision recorded) · ⏸ parked

> **2026-08-21 — thread register moved.** The single master list of every open thread across all four
> state docs now lives in `immediate_next_steps.md` §H, with the three colliding `T`-registers
> namespaced (`INS-T*`, `RG-T*`, `RS-T*`). §H also records four internal contradictions (§H2) that must
> be resolved rather than averaged over. **This tracker keeps its D1–D11 decision rows as the
> authority on script changes; §H is the index.**
>
> Session in progress: step-1 triage close-out — defining the paper's key takeaways from the results,
> then deriving the figure/table plan and the pre-compute decision list from them.

Last updated: 2026-08-21

**D1 finding (2026-08-19):** Analysed `table_validation_auc_full.csv`. Model robustness DIFFERS materially across the 6 strategies — NOT Gaussian-only. Kendall W across strategies: LogD 0.78, hERG 0.76, Caco-2 0.59. "Most robust model" changes by noise type (SVM wins outlier/quantile/threshold; NGBoost wins Gaussian/valprop/hetero; LGB wins hERG-outlier). Magnitudes swing hugely (XGBoost Caco-2: −0.19 outlier → +0.90 hetero). Hetero = near-ceiling, threshold/valprop = stress tests. ⇒ cannot collapse to Gaussian; strategy dimension must be visible. Next: decide HOW to show it in the figure.
**D1 validity check (2026-08-19):** validation auc is CLEAN — 0 artifacts, 0 valprop corruption, 0 baseline leakage. 21 negative "collapse" cells are REAL (17 = XGBoost under threshold/valprop/outlier/quantile), not invalid. 1 true NaN (dnn/hERG/valprop). Clear findings: XGBoost collapses (negative) under harsh noise on Caco-2+hERG; SVM = harsh-noise champion (outlier/quant/thresh); NGBoost = mild-noise (Gauss/hetero) + LogD; hetero = ceiling (no discrimination); LogD barely discriminates. Tables saved scratchpad/valid_auc_*.csv.
**D1 → METRIC ISSUE (2026-08-19, raises new item):** User cares about ABSOLUTE R², not retention. Computed absolute mean-R²-under-noise from raw KIRBy all_results.csv (gated baseline≥0.3, over usable reps). AUC_norm (retention) and absolute-R² rankings DISAGREE. SVM = AUC_norm "harsh-noise champion" but near-BOTTOM in real R² (Caco-2 0.17, negative under Gaussian) — retention mirage confirmed. High-baseline ensembles (GP[PDV-only], LightGBM, RF, NGBoost) hold best actual R²; NGBoost good in both views; XGBoost worst in both. ⇒ NEW cross-cutting question (add as **D11**): report robustness as retention (AUC_norm) vs absolute R² vs both (baseline+retention). Affects metric C1, not just the figure. Tables: scratchpad/absR2gated_*.csv.

**D2 finding (2026-08-19) — the "latent" premise is WRONG for one of the three:**
- `table_validation_auc.csv` — confirmed **latent**: no `\ref` in paper.tex, no paper value traces to it, nothing in either script reads it back. Fully reconstructible from `table_validation_auc_full.csv` (max abs diff on MEAN = 2.2e-16, 26 rows vs 465). Safe to stop emitting / demote to a `diag_` name.
- `table_validation_probabilistic.csv` — **NOT latent. It is Additional file 11**, cited at paper L560 ("QRF was consistently less robust than RF on every external data set"). Its 3 rows average over 4 reps × 6 strategies. Disaggregated from `_full` (72 paired rf/qrf cells): **RF wins 61/72; QRF wins 11**, 10 of them on Caco-2 — QRF wins **all 4 reps under quantile** (mean Δ −0.091, worst −0.245) and 2/4 under outlier (mean Δ −0.061). Per dataset×strategy means: Caco-2 quantile −0.091, outlier −0.061 (QRF ahead); every hERG and LogD cell RF-ahead. ⇒ **"consistently" is an averaging artifact**; the table must carry the strategy axis and the L560 sentence needs a §8D paper edit. Caption also still says "NDS". (The BNN pairs this function would add never fire — validation has only dnn/lgb/ngboost/qrf/rf/svm/xgboost; same root cause as the D1 L551 "BNN variants" claim.)
**D2 options drafted (2026-08-19)** — sample CSVs in `scratchpad/PROPOSED_validation_probabilistic_{A,B,A_withbaseline}.csv`:
- **A** = dataset × strategy, reps averaged, 21 rows incl. an explicitly labelled `POOLED (all 6)` row per dataset + an "RF better in n/4" column. Shows Caco-2 quantile Δ=+0.091 and outlier Δ=+0.061 (QRF ahead).
- **B** = dataset × strategy × rep, 72 rows; only version that shows MHG-GNN/quantile (QRF 0.622 vs RF 0.376, Δ=+0.245) and PDV/outlier (Δ=+0.160).
- **A+** = A plus baseline R² columns — but `baseline_r2` is **constant across strategies** (σ=0 baseline), so the columns repeat: state baseline once per dataset/model instead. Relevant to D11.
- Also found at L560: the **second** sentence ("representation has a minimal effect … on all three external datasets") is a separate averaging artifact — Caco-2 quantile rep spread for QRF is 0.366→0.622 (matches F3 "Caco-2 spread 0.22").
- `table4_uncertainty_metrics.csv` ("all") — confirmed **not the source** of `tab:top_unc_noise` (that is table4c) nor of Additional file 9 (ECFP4): value-match against paper.tex gives ecfp4 19/21 hits vs "all" 14/21. Legacy-only, so no strategy averaging; the averaging is **across reps**, and it pools raw-unit ECE artifacts (VBLL-β 9.72, VBLL-α 6.05) with real values into a meaningless column. Per-rep siblings exist for all 8 reps plus `table4_supp_uncertainty_by_strategy_rep.csv`.

**✅ D2 — DECIDED & IMPLEMENTED (2026-08-19)** in `scripts/generate_paper_figures_v2.py`. All three artifacts settled:
1. `table_validation_auc.csv` — **no longer emitted** (author: "too much to report"). The collapsed pivot block is deleted; `table_validation_auc_full.csv` (465 rows, keeps strategy + baseline) is now the only validation auc table, and the no-`dataset`-column fallback writes to `_full` too. Docstring updated.
2. `table4_uncertainty_metrics.csv` (rep-pooled "all") — **no longer emitted**. `reps_to_compute` is now just the available reps; `'all'` survives only for the degenerate case where the data has no `rep` column (nothing pooled). Per-rep `table4_uncertainty_metrics_<rep>.csv` unchanged.
3. `table_validation_probabilistic.csv` — **rebuilt as Option C**: one row per dataset × noise strategy with the representation held at `PRIMARY_REP` (**continuous_pdv**, author-confirmed). 18 rows, all 6 strategies visible, columns now include `Strategy` and `Representation`. Datasets ordered by `_order_validation_datasets` (LogD → Caco-2 → hERG).

**Verification (2026-08-19):** module imported and the new probabilistic block exercised against the real helpers on local data → exactly **18 rows**, rep=continuous_pdv, all 6 strategies present. **No-averaging check passed**: every (dataset, strategy, model) cell has exactly 1 underlying row, so the `.mean()` calls reduce over a single value. Script compiles. Result reproduces the sample: QRF ahead on Caco-2 under outlier (+0.160), quantile (+0.034), threshold (+0.017), hetero (+0.004); RF ahead in all 14 other cells.

**✅ Carried-forward v1/v2 question — RESOLVED, no change needed (2026-08-19). Author: "use v2".** My earlier "blocking" framing was WRONG: `slurm_scripts_analysis/run_figures_v2.sh` already exists (tracked, commit 1987913, 2026-07-31) and already runs `generate_paper_figures_v2.py --output-dir ../results/paper_figures_v2`. D2 takes effect on the next `sbatch run_figures_v2.sh`; nothing to mirror into v1.
- **v1 is the pre-metric-change script and is dead:** `generate_paper_figures.py` has 216 `nds` mentions and **0** `auc_norm`; v2 has 122 `auc_norm`. So v1 is not an alternative version, it is the NDS-era one. It writes to `results/paper_figures/`; v2 writes to `results/paper_figures_v2/` ⇒ **`results/paper_figures/` is stale NDS output, do not read numbers from it.**
- **⚠ REMAINING TRAP:** `run_figures.sh` (v1, NDS) is still present and still tracked, and MEMORY.md's "Re-run Figure Generation" line points at it. Running it silently produces NDS tables. Open tidy-up: `git rm` v1 + `run_figures.sh`, or keep them archived? (Author decision pending — separate from D2.)
- Both SLURM scripts pass `--validation-dir .../KIRBy/tests/results/alternative_full`. v2's `load_validation_data` (L1118) excludes `herg_fluid`, `openadmet_caco2`, `openadmet_logd` and keeps subdirs `caco2`, `herg`, `logd` — so `alternative_full` IS correct for the loader. **Supersedes the earlier tracker claim that "canonical validation data = validation_rerun"**: `validation_rerun/` subdirs are named `<rep>_<dataset>` (e.g. `ecfp4_caco2`) which this loader would read as dataset names. Those three `alternative_full` subdirs are EMPTY locally — they hold data on ARC only.

**D2 → OPTION C proposed by author (2026-08-19): six strategies × 3 datasets with ONE representation held constant (18 rows, nothing averaged).**
- **Feasible for any rep** — coverage is complete 18/18 dataset×strategy cells for rf AND qrf on all four reps (continuous_pdv, ecfp4, mhggnn, sns). No gaps to work around.
- Mean baseline R² per rep is close: sns 0.583, ecfp4 0.572, mhggnn 0.558, continuous_pdv 0.548 — no rep is "obviously best".
- **The choice does move the numbers**, so it must be made on a stated principle, not on the answer: QRF wins 4/18 cells on continuous_pdv, 3/18 on mhggnn, 2/18 on ecfp4, 2/18 on sns. Qualitative story identical in all four (RF ahead nearly everywhere; the exception is a Caco-2 pocket).
- **Recommendation: continuous_pdv**, because `PRIMARY_REP = 'continuous_pdv'` (script L78) already governs every other main figure including the validation heatmap (L1513) — a principle fixed before this question arose. Caption states the rep; keep the 4-rep `table_validation_auc_full.csv` as the auditable backing file.
- Sample tables: `scratchpad/PROPOSED_validation_probabilistic_C_{continuous_pdv,sns}.csv`.

**✅ ECE — DECIDED & DONE (2026-08-19): complete removal.** Author call: strip ECE from the paper and from the figure-generation scripts, full deletion not commented out. Implemented in `generate_paper_figures_v2.py`, `generate_paper_figures.py` (the script `run_figures.sh` actually invokes) and `deep_analysis.py` — zero occurrences remain in all three, all compile, `deep_analysis.py` re-run clean. Paper-side removal list written into REVISION_GUIDE.md §"Metric removal — ECE" (7 paper.tex locations) + tracker row T24; the guide's own Replace/With blocks that would have re-introduced ECE (availability paragraph, Additional file 9 caption, metrics-summary row, NoiseInject framework paragraph) were fixed too. Coverage at 1σ/2σ now carries calibration alone. NOT touched: `uncertainty_analysis.py`, `phase2_analysis.py`, `generate_figures.py` (legacy, no paper artefacts) — say if these should go too.

**ECE evidence (background to the above):** ECE appears in paper.tex only at L234–238 (definition), L300–302 (metrics-summary row), L503–528 (one column of `tab:top_unc_noise`), L587 (abbreviation), L667 (Additional file 9 caption). **No prose sentence reports an ECE value.** Two concerns: (a) the implementation compares predicted sd against mean |error|, which differ by a factor 0.798 even when calibration is perfect — simulated a flawless model, ECE = 0.198 at label scale ×1 and 2.031 at scale ×10, so the score has a floor that grows with the numbers' size; (b) coverage at 1σ/2σ is already reported, is scale-free, and carries the same information. ⇒ possible "drop ECE, keep coverage".
**⚠ Reopens T20:** `table4_supp_uncertainty_by_strategy_rep.csv` (Gaussian) shows VBLL on mhggnn/mol2vec with coverage 1σ = 0.27–0.45 and 2σ = 0.39–0.58 against targets 0.68/0.95. Low coverage despite huge intervals ⇒ errors exceed even the inflated uncertainty; this is **not** only a units artifact, so "T20 RESOLVED — scale artifact" is at best incomplete. Raw QM9 uncertainty CSVs are server-only; needs an ARC check.

**D11 → AUTHOR DIRECTION (2026-08-20): "I don't think we need a whole new metric. Just rank R² under a set noise level (we need to determine this level)."** ⇒ no `delivered`/composite metric. Report plain R² read off at ONE σ. Remaining question = which σ (+ whether auc_norm survives alongside).

**D11 σ-SELECTION EVIDENCE (2026-08-20, all re-derived locally, nothing from memory):**
- **Source**: per-σ validation data IS local after all — `KIRBy/tests/results/validation_rerun/<rep>_<dataset>/<dataset>/all_results.csv`, 12 files (4 reps × 3 datasets), 48510 rows. (Earlier note "alternative_full subdirs empty locally" still true; `validation_rerun` is the local per-σ source.) Fold-averaged, then gated baseline≥0.3 to match `calculate_robustness` → 709 configs / 7799 rows.
- **Grid is 0.0–1.0 step 0.1, IDENTICAL for all 6 strategies AND for QM9** (`--sigma 0.0 0.1 … 1.0` in slurm_scripts_*). One σ covers both studies. 10 candidates.
- **Separation** (IQR of R² across models within dataset×rep×strategy) vs **baseline-echo** (Spearman of R²@σ against R²@0) vs **usability**:

| σ | separation | ρ vs baseline | %R²>0 | %diverged(R²<−1) |
|---|---|---|---|---|
| 0.0 | 0.068 | — | 100.0 | 0.0 |
| 0.2 | 0.069 | 0.921 | 99.4 | 0.3 |
| 0.4 | 0.075 | 0.860 | 99.2 | 0.3 |
| 0.5 | 0.086 | 0.833 | 98.6 | 0.3 |
| 0.6 | 0.124 | 0.810 | 94.2 | 0.7 |
| 0.7 | 0.142 | 0.791 | 92.1 | 0.4 |
| **0.8** | **0.177** | **0.760** | **87.7** | **0.7** |
| 0.9 | 0.221 | 0.737 | 82.9 | 1.0 |
| 1.0 | 0.257 | 0.708 | 79.1 | 3.0 |

- **σ<0.5 is wasted**: separation (0.066–0.086) ≈ separation at σ=0 (0.068) and ρ vs baseline ≥0.83 ⇒ you'd be re-publishing the baseline ranking.
- **Ranking is locally stable everywhere** (adjacent-σ ρ ≥ 0.947; 0.7↔0.8 = 0.982, 0.8↔0.9 = 0.978) ⇒ the choice is not knife-edge. But it DOES matter across the range (σ=0.2 vs 1.0 ρ=0.801).
- **RECOMMENDATION: σ = 0.8.** Separation 2.6× baseline spread; ρ vs baseline 0.760 (new info, not unrecognisable); 87.7% still R²>0 (a ranking of all-negative models is meaningless); 0.7% diverged vs 3.0% at σ=1.0; has neighbours BOTH sides so stability is demonstrable (σ=1.0 is the endpoint and cannot show this); ρ(0.8, 1.0)=0.968 so little is lost vs the harshest setting. σ=1.0 = defensible alternative ("max noise tested", most separation) but 21% negative + endpoint.
- **⚠ HETERO NEVER SEPARATES AT ANY σ** — IQR flat 0.061–0.071 from σ=0 to σ=1.0; median R² only 0.485→0.400. Not fixable by σ choice; it is a property of the strategy. Per-strategy separation at σ=0.8: threshold 0.390, valprop 0.219, legacy 0.139, quantile 0.128, outlier 0.122, **hetero 0.063**. ⇒ the hetero column of any R²@σ ranking ≈ the baseline ranking and needs an explicit caveat sentence. (Consistent with the existing "hetero = ceiling tier" finding.)
- **⚠ OPEN, blocks implementation:** does `auc_norm` REMAIN alongside R²@σ, or is it replaced? Author has not said. No script change until answered.
- **⚠ CORRECTION to the D1/D11 "SVM mirage" claim (line 10 above):** does NOT reproduce. On `table_validation_auc_full.csv` restricted to the 45 cells where all 7 models are present, SVM ranks **#1 on BOTH** retention (0.900) and absolute R² under noise (0.574). The old claim came from an UNGATED raw-KIRBy computation where SVM's single divergent PDV/hERG cell (baseline −1.9e7) dragged the mean down. The genuine mirage is **NGBoost** (retention rank 2, absolute rank 5, lowest baseline 0.536); the reverse case is **LightGBM** (retention rank 4, absolute rank 2). Worked example — LogD/ECFP4/threshold: NGBoost auc_norm 0.971 vs LGB 0.864, but absolute R² 0.551 vs 0.647.
- **Baseline↔retention Spearman = 0.179** (model means, 45 common cells) — near-independent. Cuts both ways: it justifies auc_norm as carrying new information, AND is why it cannot stand alone. Paper currently presents this independence as the "decoupling" FINDING (L460, L577); those two readings are incompatible — needs a §8D decision.
**⚠⚠ D11 — σ UNITS PROBLEM (2026-08-20, author challenge "0.8 seems really extreme" → correct, and it exposed a design confound):**
- **σ is in RAW LABEL UNITS, not scaled.** `NoiseInject/noiseInject/core.py:68` `_legacy`: `y_noisy = y + σ·N(0,1)`. In `KIRBy/tests/alternative_data_noise_robustness.py:750-760` only **X** gets `StandardScaler`; `y_train` goes to `injector.inject(y_train, sigma)` RAW. Same grid on QM9 (`--sigma 0.0 … 1.0` in slurm_scripts_*).
- **⇒ ONE σ IS NOT ONE NOISE CONDITION.** Label SD back-computed as `RMSE²/(1−R²)` from σ=0 rows (n=1219–1362 per validation dataset; QM9 from 292 local `mol2vec_investigate_*.csv` rows):

| dataset | label SD | σ=0.5 | σ=0.8 | σ=1.0 |
|---|---|---|---|---|
| OpenADMET-LogD | 1.191 | 0.42× SD | 0.67× SD | 0.84× SD |
| ChEMBL-hERG-Ki | 0.905 | 0.55× SD | 0.88× SD | 1.10× SD |
| OpenADMET-Caco2_Efflux | 0.434 | 1.15× SD | **1.84× SD** | 2.30× SD |
| QM9 | 1.051 (IQR 1.041–1.065 ⇒ effectively standardised) | 0.48× SD | 0.76× SD | 0.95× SD |

- At σ=0.8 three datasets sit at 0.67–0.88× SD but **Caco-2 sits at 1.84× SD — noise ~2× the signal spread.** This is the mechanism behind Caco-2 supplying most of the negative-R²/collapse cells (ties to the D1 "XGBoost collapses on Caco-2" finding — partly a σ-scaling artifact, not purely model fragility).
- **⇒ AFFECTS THE PAPER AS IT STANDS, not just D11:** any cross-dataset comparison at a fixed σ is comparing different amounts of noise. Needs its own decision + a §8D sentence.
- **THE FIX ALREADY EXISTS IN-HOUSE BUT WAS NOT USED:** `KIRBy/src/kirby/noise_spec.py` expresses level as `target_effective_noise` (mean |Δy| as a fraction of label SD) and binary-searches the raw σ per dataset. Its own docstring: *"This is what makes a level comparable across datasets; the four ad-hoc definitions never were."* The paper run hardcodes raw `SIGMA_LEVELS` (`alternative_data_noise_robustness.py:79`). Options: (a) keep raw σ + report the ×SD ratio in every caption, (b) re-run calibrated per dataset (expensive, needs ARC), (c) report σ but restrict cross-dataset claims.
- **σ=0.8 recommendation is ON HOLD** pending literature anchoring — it was derived from statistical discrimination ONLY and never asked what σ physically means.
- **Literature workflow running** (`wqrlyif68`, run `wf_b419480e-a74`): published experimental-error SDs for logD7.4 / Caco-2 efflux / hERG pKi / QM9-DFT, + what convention other noise-injection papers use (absolute vs ×label-SD vs SNR). If experimental SD for pKi ≈ 0.5 log units then σ=0.5 on hERG ≈ "double the noise the public data already carries" — a far stronger justification than spread statistics.
**D11 → AUTHOR DIRECTION 2 (2026-08-20): "So it should be a different sigma per dataset" — YES, and it costs NOTHING to run.** Define **r = injected noise SD ÷ label SD** (plain meaning: how big the noise is relative to the natural spread of the labels). The existing 0.0–1.0 grid ALREADY contains matched levels, so this is a read-off, not a re-run.
- **Matched picks from the existing grid** (nearest available σ to r × labelSD):

| target r | LogD (SD 1.191) | hERG (SD 0.905) | Caco-2 (SD 0.434) | QM9 (SD 1.051) |
|---|---|---|---|---|
| 0.5 | σ=0.6 (0.50×) | σ=0.5 (0.55×) | σ=0.2 (0.46×) | σ=0.5 (0.48×) |
| **0.7** | **σ=0.8 (0.67×)** | **σ=0.6 (0.66×)** | **σ=0.3 (0.69×)** | **σ=0.7 (0.67×)** |

  r=0.7 is the tightest match available (0.66–0.69 across all four). r=0.5 also clean. r≥0.85 impossible — LogD caps at 1.0/1.191 = 0.84.
- **Matching DOES equalise the damage (validated):** % of baseline R² retained — fixed σ=0.8 gives LogD 90.8 / hERG 65.6 / **Caco-2 53.1**; matched r=0.7 gives 90.8 / 82.1 / **86.1**. ⇒ confirms the fixed-σ Caco-2 collapse was a UNITS artifact, not a dataset property.
- **⚠ THE COST — matching kills Caco-2 as a discriminator.** Model separation (IQR of R² across models): Caco-2 falls 0.200 → **0.057** at r=0.7, and 0.057 ≈ the σ=0 separation (0.068) ⇒ no signal. hERG falls 0.169 → 0.102. LogD unchanged (0.162, since σ=0.8 is its matched level). **Caco-2's apparent discriminating power in current results comes from being hit ~3× harder than the others.**
- **⚠ Caco-2 GRANULARITY:** labelSD 0.434 ⇒ each 0.1 grid step moves r by 0.23. Only ~4 usable levels (0.23/0.46/0.69/0.92) vs 10 for LogD. r=0.7 landing at 0.69 is luck. A calibrated re-run (`noise_spec.calibrate_sigma`) would be needed for finer control.
- **STILL OPEN:** whether the fixed-σ discrimination being given up was ever meaningful — depends on whether r≈0.7 is already far above real experimental noise (→ literature workflow).
**D11 → AUTHOR DIRECTION 3 (2026-08-20): "Every dataset has a different threshold. Determine the threshold on percentage of models falling above/below."** ⇒ drop the label-SD ratio arithmetic; set σ per dataset empirically from where models start failing. Computed on the same gated set (baseline≥0.3, fold-averaged, 709 configs).
- **Rule tested: σ at which fewer than X% of model configs still clear an R² threshold.**

| rule | Caco-2 | hERG | LogD |
|---|---|---|---|
| R²>0.3, <75% remain | **σ=0.5** (55%) | **σ=0.6** (69%) | NEVER (88% at σ=1.0) |
| R²>0.3, <50% remain | σ=0.6 (38%) | σ=0.9 (42%) | NEVER (88%) |
| R²>0.3, <25% remain | σ=1.0 (23%) | NEVER (41%) | NEVER (88%) |
| R²>0, <75% remain | σ=0.9 (67%) | NEVER (81%) | NEVER (98%) |

- **RECOMMENDED RULE: R²>0.3, fewer than 75% remain ⇒ Caco-2 σ=0.5, hERG σ=0.6.** Both inside the grid with headroom either side, neither in the collapse zone.
- **⚠ LogD REACHES NO THRESHOLD AT ALL** — 88.1% of configs still clear R²=0.3 and 97.7% still clear R²=0 at σ=1.0. LogD is insensitive to the entire noise range run. (Independently matches the earlier D1 note "LogD barely discriminates".) Options: assign σ=1.0 as its most-degraded available point, or report "no threshold reached" as the finding.
- **Threshold choice is constrained, not free:** R²=0 is useless (nothing crosses it but Caco-2 at σ=0.9); R²=0.5 is useless the other way (hERG has only 29.6% of configs above 0.5 at σ=0 — already "failed" before any noise). **R²=0.3 is the only workable threshold**, and it coincides with `ROBUSTNESS_BASELINE_THRESHOLD`.
- **⚠ OPEN:** these percentages POOL the 6 strategies (hetero barely degrades, threshold/valprop degrade hard) ⇒ crossing σ will differ by strategy. Needs per-strategy breakout before it goes in the paper, per the no-averaging rule.
- **STATUS of the earlier σ work:** the fixed-σ=0.8 recommendation and the label-SD ratio (r) matching are SUPERSEDED as the selection method. The label-SD table is still worth keeping as a CAVEAT (it explains WHY Caco-2 fails earliest: at a given raw σ it receives ~2.7× the noise-to-signal of LogD), but it is no longer how σ is chosen.
**✅ D11 → σ = 0.6 CHOSEN (2026-08-20, author: "Why don't we just use 0.6?") — and the literature backs it.** 31-agent verified search (workflow `wf_b419480e-a74`, full output `tasks/wqrlyif68.output`). Published experimental error, in the SAME log units as the labels: hERG pKi **SD 0.54** (Kramer 2012 *JMC* 55:5165); pIC50 **0.68** (Kalliokoski 2013 *PLoS ONE* 8:e61007, 20356 pairs); hERG pIC50 RMSD **0.737** (Sato 2018); logD **MAE 0.48** after curation / 0.7 before, repeat-test worst case (Niu 2024 *Sci Data* 11:985 Table 5, VERIFIED). ⚠ **The Bruneau & McElroy 2006 "0.27" figure FAILED verification — do not use** (ACS 403, Unpaywall reports closed, abstract contains neither 0.27 nor 307); Caco-2 log10 **~0.43** (Hayeshi 2008 10-lab, via Chen 2017). **All estimates fall 0.27–0.74, clustered ~0.5 ⇒ σ=0.6 ≈ ONE UNIT OF REAL EXPERIMENTAL ERROR for every endpoint.**
- **Units verified:** injection is `y + σ·N(0,1)` in RAW label units, and ALL THREE validation labels are logs — logD natively, hERG = pKi, Caco-2 via `log_transform=True` (`alternative_data_noise_robustness.py:1256`). So σ is directly comparable to a published log-unit assay error.
- **⇒ THE LABEL-SD MATCHING IS SUPERSEDED.** Assay error in log units is roughly constant across endpoints *regardless* of label spread, so a FIXED σ is more defensible than per-dataset matching. Ratios survive only to explain why Caco-2 degrades first.
- **Caveats carried into REVISION_GUIDE §"Noise magnitude":** (1) QM9 is computed DFT data — no experimental error exists, justification does not transfer; (2) public data ALREADY carries this error, σ=0.6 DOUBLES it (√(0.54²+0.6²)≈0.81) — say "an additional unit", never "we add realistic noise to clean data"; (3) Caco-2 evidence weakest — cite Hayeshi 2008 not the Fagerholm vendor preprint; (4) don't swap intra-lab (0.17–0.22) for inter-lab (0.54–0.68) figures.
- **⚠ Quotes NOT yet available** — verbatim-quote extraction + independent re-check running (`wf_a08b90aa-22c`). Until it lands: **cite, do not quote.** Author has paywalled DOIs for manual access (listed in guide).

**D11 → RANKING AT σ=0.6 vs CLEAN R² vs auc_norm (2026-08-20, rep=PDV, gate≥0.3).** CSVs: `scratchpad/rank_sigma06_pdv.csv`, `rank_sigma06_all_reps.csv`. Full tables in REVISION_GUIDE §"Ranking at σ=0.6". Headlines:
- **clean R² ↔ auc_norm Spearman is ~0 in 14 of 18 dataset×strategy cells** (range −0.36 to +0.49). auc_norm does not rank models by accuracy. Reproduces the 0.179 model-mean result at 13 models.
- **Under mild noise (hetero/legacy/outlier/quantile) R²@0.6 ≈ clean ranking** (ρ 0.91–1.00 on LogD/hERG) ⇒ σ=0.6 of those noise types does not change who wins.
- **Under STRESS noise it does:** R²@0.6 vs clean drops to **0.26** (hERG threshold) and **0.20** (hERG valprop). ⇒ threshold + valprop are where robustness actually decides the winner. Strong support for the 3-tier strategy framing.
- **Top-3 disagreement, Gaussian/LogD:** R²@0.6 → DNN, GP, MLP; clean → DNN, GP, SVM; **auc_norm → NGBoost, MLP-VBLL-Full, BNN-Full — NO overlap with the accuracy podium**, and all low-baseline (NGBoost clean 0.661 vs DNN 0.797). Clearest demonstration yet that retention crowns models for having less to lose.
- **⚠ DATA PROVENANCE:** these come from `validation_rerun` (13 models, 48510 rows) — the LOCAL per-σ source. The paper pipeline is fed `alternative_full` (7 models) on ARC. **Different runs.** Must be reproduced from whichever directory the final figures use.
- **⚠ REOPENS A CLOSED ASSUMPTION:** `validation_rerun` CONTAINS MLP, BNN-Full, VBLL-Full, MLP-BNN-Full, MLP-VBLL-Full and GP. This contradicts (a) the standing memory note "KIRBy has no MLP/VBLL models" and (b) the D2 finding that the BNN comparison pairs "never fire on validation" — both true of `alternative_full`, NOT of `validation_rerun`. **May unblock BNN/VBLL validation comparisons previously written off.** Do not act until the directory question is settled. (MLP diverges on hERG/PDV: R² −41.9 at σ=0.6.)

- **Two DIFFERENT jobs for σ, keep them separate in the paper:** σ chosen to DISCRIMINATE between models (statistical, → 0.8) vs σ chosen to MIMIC REAL EXPERIMENTAL NOISE (physical, → TBD from literature). Either is legitimate; the paper must say which it is doing.

- **⚠ QM9 CANNOT BE CHECKED LOCALLY**: no QM9 table carries baseline_r2 (`table2_auc_by_strategy_pdv.csv`, `table2_supp_auc_all_reps.csv` are retention-only) and there are 0 local `anova_*.csv`. Whether the NGBoost-style reversal exists in the MAIN study is currently an ASSUMPTION. ⇒ concrete justification for **D5** (baseline dump), which should run before any D11 paper text is written.

| # | Item (script change needing a decision) | §8 ref | Status | Decision |
|---|------------------------------------------|--------|--------|----------|
| D1 | `fig_validation_combined` (CITED L551) averages away all 6 strategies + 3 datasets — re-facet by strategy vs. relabel-as-mean + re-point claims to `fig_validation_overview` | 8A / 8F-1 | 🔶 discussing | — |
| D2 | Latent averaged tables (`table_validation_auc`, `table_validation_probabilistic`, `table4_uncertainty_metrics` "all") — stop emitting / rely on disaggregated siblings? | 8A / 8F-2 | ✅ decided + implemented (v2) | Drop `table_validation_auc.csv` and rep-pooled `table4_uncertainty_metrics.csv`; rebuild `table_validation_probabilistic.csv` as dataset × 6 strategies at PRIMARY_REP (continuous_pdv), 18 rows. v1 mirror still open. |
| D3 | `table_supp_icc` averages across 6 strategies before ICC — keep as supp, or compute per-strategy? | 8A | ⬜ | — |
| D4 | `table2_*_pdv` MEAN/STD/Mean_Rank across strategies — KEEP (all 6 shown, GR8-compliant)? Just confirm. | 8A | ⬜ | — |
| D5 | ADD `save_qm9_baseline_table` (dump baseline_r2 per model×rep×strategy) | 8B-B1 | ⬜ | — |
| D6 | ADD `create_decoupling_tables` (per-strategy QM9 + per dataset×strategy validation; pooled only as labelled row) — roster: all reps vs PDV-only? | 8B-B2 / 8F-4 | ⬜ | — |
| D7 | ADD `create_rep_strategy_auc_table` (rep×strategy, all 6 shown) — roster: ANOVA reps vs all 9? | 8B-B3 / 8F-5 | ⬜ | — |
| D8 | Fold `deep_analysis.py` into main script + `git rm` it — and CSV naming (`deep_` vs `table_qm9_*`) | 8B-B4 / 8F-3 | ⬜ | — |
| D9 | Validation ANOVA refit (per-fold replicates, local) — scope: Gaussian-only (mirror QM9) vs all 6 side-by-side | 8C / 8F-6 | ⬜ | — |
| D10 | `calculate_robustness` self-guard against catastrophic iterations (T14-HARDEN) — add internal filter vs keep caller-side | 8E / 8F-8 | ⬜ | — |
| D11 | **METRIC: retention (AUC_norm) vs absolute R² under noise.** AUC_norm rewards holding onto a bad baseline (mirage = **NGBoost**, NOT SVM — see correction above). Cross-cutting — touches C1, every robustness table/figure, both studies. | new (from D1) | ✅ **METRIC DECIDED** (2026-08-20) — figure work specced | **σ = 0.6 CHOSEN** (author, 2026-08-20), justified by published assay error ~0.4–0.74 log units across all endpoints — see the literature block above and REVISION_GUIDE §"Noise magnitude" + §11 (verified quotes). Tables built (guide §10.4). **(a) ANSWERED — `auc_norm` STAYS.** Author's framing: *"auc_norm needs to take into account baseline performance at higher sigmas. The r2 at sigma=0.6 is a bit of a sanity check on auc_norm. Obviously other noise levels exist, but it catches a few cases."* ⇒ NOT a metric replacement; the fix is to make baseline visible beside auc_norm. Figure changes specced in **REVISION_GUIDE §12** (fig1 Panel B baseline column + σ=0.6 line in Panel A; fig3 colour-by-R²@0.6 + the y-zoom decision; validation baseline strip; one added panel on fig3 — no new figures, no metric removed). **STILL AWAITING**: (b) fig3 y-axis — keep the zoom or go [0,1] (author call; the zoom currently argues for the retired "decoupling" claim); (c) which validation directory — `alternative_full` (7 models) vs `validation_rerun` (13) — blocks §12.3; (d) how the QM9 half is framed (computed data, no experimental error to anchor σ to). D1 must settle before §12.3. |

**D11 winner charts (2026-08-19) — model & rep, R² vs auc_norm, per strategy × dataset + QM9:**
- MODELS flip winner by strategy; REPS do not — each dataset has one dominant rep across all 6 strategies (LogD→SNS/MHG-GNN, Caco2→PDV, hERG→MHG-GNN). Rep winner is dataset-specific, not strategy-specific (= F3).
- R²-vs-retention DISAGREEMENT is a MODEL-level problem (SVM/NGBoost); reps mostly agree (aggregating over models smooths the mirage).
- Retention story consistent QM9↔validation: SVM wins outlier/quantile by auc_norm on BOTH; NGBoost wins mild on both. Validation PROVED SVM's auc_norm win is a mirage (real R² near-bottom) ⇒ QM9 likely same but **UNCONFIRMABLE locally** (QM9 baseline/R² server-only). Strongest reason to run D5 baseline dump.
- QM9 rep winner by auc_norm = mol2vec (low-baseline rep = same mirage at rep level). Caveats: GP is PDV-only (inflates GP in model-R² col + PDV in Caco2 rep row).

## Not script CHANGES, but must happen (tracked so they aren't lost)
- **R1 — run the script** to actually emit `table_validation_uncertainty.csv` (wired but never run) and pick up missing GP validation rows. *(regeneration, not a code edit)*
- **R2 — one ARC regen** after D5–D8 land, since QM9 raw `anova_*.csv` is server-only (certifies QM9 baseline/decoupling numbers).
- **Paper-text edits (§8D)** — tracked separately in REVISION_STATUS §8D; those are author edits to paper.tex, not script changes.

## Decisions log
_(append here as items are decided)_

**D11 PATTERN SYNTHESIS (2026-08-19, workflow wz5d8h5jy, 5 grounded lenses):**
- **TWO INDEPENDENT AXES.** (1) baseline quality → absolute R² under noise (Pearson +0.75; "good stays good" in real terms). (2) inductive-bias × noise-type → retention (AUC_norm), which is ORTHOGONAL to baseline within a dataset (Spearman ~0). AUC_norm measures only axis 2 and hides axis 1 — the root of every mirage. (Pooled baseline↔auc +0.45 = dataset-difficulty confound.)
- **Trending models are all flattered:** NGBoost tops retention only b/c LOWEST baseline (abs R² near-bottom: LogD #7/8, hERG #5/6); GP real-but-PDV-only + gated out of hERG (survivorship); SVM rep-dependent + sign-flips negative on PDV threshold/valprop. Genuinely-stay-good = RF, LightGBM (high baseline).
- **STRATEGY TIERS (universal, all 4 datasets):** ceiling = Hetero+Outlier (retention>0.9); mild = Gaussian+Quantile; STRESS = Threshold+ValProp (destroy high-|y| tail). Strategy is selective by model TYPE on validation: threshold/valprop kill boosting (XGBoost worst)+QRF, kernels shrug off; Gaussian hits DNN+SVM; outlier only bites low-baseline Caco-2.
- **DATASET-SIZE HEADLINE:** boosting-fragility is validation-only — vanishes on big clean QM9 (there NNs are the laggard). Model choice matters far more on small real data. Rep matters ~8× more on validation than QM9.
- **XGBoost = the "good≠stays-good" exception:** passes baseline gate but collapses to negative R² under threshold/valprop on hard datasets.
- **Implication for D11:** the two axes are independent ⇒ reporting retention ALONE (or absolute alone) is insufficient; leans toward BOTH (baseline + retention, or baseline + absolute-under-noise). QM9 abs-R² mirage is INFERRED not confirmed → needs D5 server dump.
Full lens output: tool-results/bnk32v38d.txt.

**D11 → PAPER-LEVEL FINDINGS (2026-08-19, candidates for guide/§8D — DISCUSS before writing to paper):**
1. **"Decoupling" (L460, L577) is the retention MIRAGE, not a result** — absolute R² under noise tracks baseline (+0.75); only the retention FRACTION is decoupled, and NGBoost/SVM top it because baselines are LOW. Biggest correction. Currently paper presents mirage AS the finding.
2. **AUC_norm (hence the whole ANOVA + ranking + Kendall W) measures FRAGILITY, blind to delivered performance.** State what auc_norm does/doesn't capture; report BOTH axes (→ D11).
3. **ANOVA residual-dominance (Outlier 83.6 / Hetero 77.4) mechanism = they are CEILING strategies** (nothing degrades → no model variance → replicate noise fills residual). Paper says "run-to-run variance" (L380) w/o the why. Add mechanism.
4. **⚠ L573 SVM claim WRONG:** "SVM maintained consistent NDS across all representations" — data says SVM is the MOST rep-dependent (MHG-GNN 0.96, collapses PDV/SNS). Must change.
5. **NGBoost false-positive** — note it's retention-flattered (low baseline), not genuinely robust.
6. **NEW findings not in paper:** (a) model-family × noise-type selectivity — threshold/valprop kill boosting(XGBoost)+QRF, kernels shrug off; Gaussian hits DNN/SVM; outlier only bites low-baseline Caco-2. (b) fragility is DATASET-SIZE-dependent — boosting fragile on small real data, fine on QM9 (NNs fragile there instead) → "model choice matters more on small real data." (c) floor-vs-ceiling model profiles: RF/LightGBM = high floor/safe default; GP/SVM = high ceiling/low floor (rep-dependent). (d) easy datasets unreliable for ranking (hERG↔LogD Spearman +0.09; hERG↔Caco-2 +0.71–0.89). (e) rep matters ~8× more on validation than QM9.
7. **Strategy 3-tier** partly in paper (L383 threshold/valprop harshest) but MISSING: clean tiers, hetero-as-ceiling, high-|y|-tail mechanism.
All grounded in workflow wz5d8h5jy (tool-results/bnk32v38d.txt). QM9 absolute-R² claims INFERRED (need D5 server dump to confirm).

**D-FIG PROPOSALS + REP CHOICES (2026-08-19, workflow wb1ogemra) — guide §9 written; these need author decisions:**

*THE PIVOTAL DECISION (recurs everywhere) — which rep is held constant:*
- **QM9 primaries** (tab:auc_ranking, fig1 Panel B, table5, fig3): PDV (current, highest baseline 0.857, consistent) vs ECFP4 (more conventional). One-line change; ECFP4 twins already exist.
- **VALIDATION figures (F6/F7/F8): PDV is RULED OUT** — SVM/DNN were never run on continuous_pdv externally (only 5 tree models). Full 7-model roster only on ECFP4/SNS/MHGGNN. Rec: **ECFP4** (matches QM9 fingerprint, most interpretable) + one sentence noting PDV was tree-only externally. ⇒ paper would hold PDV for QM9, ECFP4 for validation.

*Figure/table changes proposed (all hold ≥1 dim constant, no 6-strategy averaging):*
1. tab:auc_ranking — ADD leading Baseline R² column (+ optional Delivered=baseline×auc). Plumbing exists (auc_df.baseline_r2). Backs F2/F5. [rep TBD]
2. fig3 (decoupling fig) — ADD delivered-R²-vs-baseline panel beside the flat retention panel. Backs F1. [rep + σ; QM9 delivered may need ARC dump OR proxy baseline×auc]
3. fig1 Panel B — ADD baseline-R² strip left of the model×strategy heatmap. Backs F2. [rep TBD]
4. table5(delivered) + tab:auc_ranking(retention) — present as explicit companion pair. Backs F2/F5. [table5 Gaussian-only vs expand]
5. fig_validation_overview — switch/add ECFP4 (PDV can't show SVM/DNN). Backs F6/F8. [ECFP4 replace vs companion]
6. fig_validation_combined Panel A — RETIRE or rebuild rep-held + strategy-faceted (over-averages strategy+rep). 
7. fig_validation_combined Panel B — split the merged external mean into 3 per-dataset series. Backs F7/F9.
8. NEW validation floor/ceiling table (MEAN/STD/FLOOR/CEILING/baseline) on ECFP4. Backs F8.
9. NEW N-vs-model-spread figure (spread collapses as N grows). Backs F7.
10. table2_auc_by_strategy_pdv — reorder columns into 3 tiers (ceiling|mild|stress). Backs F11. [rep TBD]

*⚠ CONFLICTS (do NOT publish until resolved — flagged in guide §9):*
- **F4 SVM:** "L573 wrong / SVM rep-dependent" does NOT reproduce in aggregate CSVs (SVM = smallest rep-spread, positive on PDV). My earlier "L573 WRONG" was over-claimed. Likely only NDS→AUC_norm fix. Needs per-config check before any reversal.
- **F9:** "+0.09 hERG↔LogD" does not reproduce (aggregate +0.79–0.93). Report grounded aggregate + ceiling-tier caveat.
- Agent error: "valprop never run on validation" is FALSE — valprop IS present. Ignored.
- Validation ANOVA (Add. file 10) saturated (residual 0, wrongly incl. QRF) — don't cite its rep η² (T17).

**CORRECTIONS (2026-08-19, verified from raw KIRBy all_results.csv):**
- ❌ "PDV RULED OUT / SVM+DNN never run on PDV" was an AGENT ERROR. TRUTH: all 8 models × 4 reps run (990 rows each), NO data gap. SVM/DNN ARE on PDV; they DIVERGE on specific cells (SVM PDV-hERG baseline −1.9e7; DNN on MHG-GNN + PDV-hERG) → baseline-gated. Model-STABILITY issue, not missing data. GP is PDV-only BY DESIGN (RBF on continuous; never run on fingerprints), weak on hERG (0.04). ⇒ ECFP4 preferred for validation figs for STABILITY (fewest divergences), not availability.
- ✅ **F4 SVM conflict RESOLVED — do NOT reverse L573.** SVM baselines: LogD (ECFP4 0.59/MHG 0.75/PDV 0.65/SNS 0.80), Caco-2 (0.36/0.46/0.43/0.40) — all consistent & positive. The ONLY collapse is SVM/PDV/hERG (−1.9e7 divergence, one cell). So SVM is genuinely rep-CONSISTENT on all usable configs; the "sign-flip" was that single divergent hERG-PDV cell bleeding through raw aggregates. L573 fix = NDS→AUC_norm only, + optional one-line caveat that SVM/DNN can numerically diverge on specific small-hard-dataset × rep combos. No deep per-config archaeology needed.
- Broader: DNN + SVM are numerically UNSTABLE on small hard datasets with some reps (divergence, not noise-sensitivity) — a real but SEPARATE caveat from noise robustness; worth one honest sentence.

**DATA + KERNEL GROUND TRUTH (2026-08-19, read from code + all result dirs):**
- **Canonical validation data = `validation_rerun`** (8 models × 4 reps × 3 paper endpoints, 28710 rows). `openadmet_caco2/logd` = 5-model partials; `herg` = 2-model stub; `alternative_full` = hERG-FLuID (classification, not hERG-Ki). All superseded. Pipeline already uses validation_rerun.
- **GP = ONE model** (`Gauche` ExactGP, models.py L1690-1789). Kernel is a param; label = `gauche_rbf` if RBF else `gauche` (L1787). Tanimoto only defined on fingerprints → RBF used on continuous PDV. NOT two models.
- **GP on validation = RBF/PDV-only in EVERY dir.** Tanimoto-GP on validation fingerprints was NEVER run (exists on QM9 only). Real gap → re-run `gauche` on validation ECFP4/SNS to fill.
- **SVM = RBF always** (models.py L1454 default; tuning ∈ rbf/poly/sigmoid; KIRBy L914 fixed rbf). NEVER Tanimoto. ⇒ SVM cross-rep is a FAIR same-kernel comparison; L573 stands.
- **⚠ PAPER ERROR L197 / Add. file 12:** claims "Tanimoto kernel for SVM on binary reps" — code does NOT do this (SVM is RBF throughout). Fix the paper (likely copied from GP's kernel scheme).
- Open re-run decisions: (a) run gauche(Tanimoto) on validation fingerprints to make GP a fair cross-rep validation model? (b) optionally RBF-GP on fingerprints (QM9+val) to isolate kernel from rep. (c) fix SVM L197 claim (no re-run needed — code is RBF).

**GP RE-RUN — KIRBy edit DONE + commands (2026-08-19):**
- ✅ Edited `KIRBy/tests/alternative_data_noise_robustness.py`: added `--gp-kernel` (default rbf) + `--gp-reps` (default PDV, backward-compat); GP auto-named `GP`(rbf)/`GP-Tanimoto`. Compiles clean.
- Validation run (ARC): `python alternative_data_noise_robustness.py --datasets all --models GP --gp-reps ECFP4 PDV SNS MHG-GNN-pretrained --gp-kernel rbf` (RBF-GP all reps, replaces old PDV-only GP). Diagnostic: `--models GP-Tanimoto --gp-reps ECFP4 --gp-kernel tanimoto`.
- QM9 run (ARC): loop 6 `--noise-strategy`, `-m gauche --kernel rbf -r <ANOVA reps minus pdv> -n 10000 -b 20 --sigma 0..1 -s scaffold -f results/anova_<strat>_gauche_rbf.csv`. Confirm exact rep tokens.
- ⏳ FOLLOW-UPS to land GP in ANOVA (after re-run): (1) remove `gauche_rbf` from `ANOVA_MODELS_EXCLUDE` (generate_paper_figures_v2.py L131). (2) add `'GP-Tanimoto'→'gauche'` to val_model_map. Both are figure-script edits (new D-items).
- SVM L197 paper fix logged in guide §9.8 (no re-run; RBF-everywhere; ANOVA inclusion unchanged).

## 🟢 ACTIVE JOBS — GP re-run (submitted 2026-08-19, partition=long)
QM9 gauche_rbf (`slurm_scripts_gauche_rbf/`, order = strategy × rep from submit_all.sh):
| Job | strategy | rep | expected output |
|---|---|---|---|
| 12822669 | legacy | ecfp4 | results/anova_legacy_ecfp4_gauche_rbf.csv |
| 12822670 | legacy | smiles | results/anova_legacy_smiles_gauche_rbf.csv |
| 12822671 | legacy | mhggnn | results/anova_legacy_mhggnn_gauche_rbf.csv |
| 12822672 | legacy | mol2vec | results/anova_legacy_mol2vec_gauche_rbf.csv |
| 12822673 | valprop | ecfp4 | results/anova_valprop_ecfp4_gauche_rbf.csv |
| 12822674 | valprop | smiles | results/anova_valprop_smiles_gauche_rbf.csv |
| 12822675 | valprop | mhggnn | results/anova_valprop_mhggnn_gauche_rbf.csv |
| 12822676 | valprop | mol2vec | results/anova_valprop_mol2vec_gauche_rbf.csv |
| 12822677 | quantile | ecfp4 | results/anova_quantile_ecfp4_gauche_rbf.csv |
| 12822678 | quantile | smiles | results/anova_quantile_smiles_gauche_rbf.csv |
| 12822679 | quantile | mhggnn | results/anova_quantile_mhggnn_gauche_rbf.csv |
| 12822680 | quantile | mol2vec | results/anova_quantile_mol2vec_gauche_rbf.csv |
| 12822681 | threshold | ecfp4 | results/anova_threshold_ecfp4_gauche_rbf.csv |
| 12822682 | threshold | smiles | results/anova_threshold_smiles_gauche_rbf.csv |
| 12822683 | threshold | mhggnn | results/anova_threshold_mhggnn_gauche_rbf.csv |
| 12822684 | threshold | mol2vec | results/anova_threshold_mol2vec_gauche_rbf.csv |
| 12822685 | outlier | ecfp4 | results/anova_outlier_ecfp4_gauche_rbf.csv |
| 12822686 | outlier | smiles | results/anova_outlier_smiles_gauche_rbf.csv |
| 12822687 | outlier | mhggnn | results/anova_outlier_mhggnn_gauche_rbf.csv |
| 12822688 | outlier | mol2vec | results/anova_outlier_mol2vec_gauche_rbf.csv |
| 12822689 | hetero | ecfp4 | results/anova_hetero_ecfp4_gauche_rbf.csv |
| 12822690 | hetero | smiles | results/anova_hetero_smiles_gauche_rbf.csv |
| 12822691 | hetero | mhggnn | results/anova_hetero_mhggnn_gauche_rbf.csv |
| 12822692 | hetero | mol2vec | results/anova_hetero_mol2vec_gauche_rbf.csv |

Validation (KIRBy, → tests/results/validation/):
| 12822693 | RBF-GP, all 4 reps × 3 datasets (--models GP) |
| 12822694 | Tanimoto-GP, ECFP4 × 3 datasets (--models GP-Tanimoto) |

Range: **12822669–12822694** (26 jobs). Monitor: `sacct -j 12822669-12822694 --format=JobID,JobName%24,State,Elapsed,MaxRSS`.
POST-COMPLETION follow-ups (already logged): remove `gauche_rbf` from ANOVA_MODELS_EXCLUDE (generate_paper_figures_v2.py L131); add `'GP-Tanimoto'→'gauche'` to val_model_map.

**JOB PARTITION UPDATE (2026-08-24):** scontrol WORKS (with TimeLimit lowered ≤ medium's 48h max). GP is SLOW — running jobs at 12–23.5h, none completed at check time; 72h ceiling was justified. Moved FAST reps (ecfp4/smiles) of the pending set to medium/48h: 12822681,682,685,686,689,690. KEPT slow reps (mhggnn/mol2vec) on long/72h: 12822683,684,687,688,691,692 (48h wall too risky). Running (long): 671,673,677,678,679. Completed: 669,670,672,674,675,676,680. Note: local edited scripts for threshold/outlier/hetero were set to medium/48h but jobs moved via scontrol instead — the mhggnn/mol2vec local scripts should be reverted to long/72h if ever re-submitted from file.
