# Uncertainty–Noise Correlation: Diagnosis & Re-Analysis Plan

> For a fresh chat. Read top to bottom before touching code. The decision has been made:
> **re-do the uncertainty–noise correlation as a within-σ analysis at σ = 0, 0.3, 0.6.**
> This plan explains the bug, the exact re-analysis, the SLURM/server path, and how the
> fresh chat should work (multiple agents, verify everything, do not trust memory).

---

## 0. How the fresh chat must work (READ FIRST)

- **Use multiple agents to gather ground truth in parallel.** Before writing any code, spawn
  several agents to independently establish: (a) what raw uncertainty data exists on the
  server and its exact columns/σ grid, (b) the exact code paths that compute the correlation,
  (c) the SLURM/partition conventions actually in use, (d) how the analysis script is invoked.
  Do not serialize this — fan out.
- **Minimal reliance on memory; zero tolerance for hallucination.** Treat MEMORY.md and any
  recalled fact as a hint, not truth. Verify every file path, column name, σ value, partition
  name, and function line number against the actual files/code before acting. If a fact can't
  be verified from the repo or server, say so — do not invent it.
- **No token budget cap.** Be exhaustive. Read the whole relevant code, check every data file,
  cross-check numbers. Thoroughness beats brevity here.
- **Work with the user (Adelaide) at the checkpoints marked ⛔.** Do not code past them alone.

---

## 1. What the metric is supposed to show

Table 7 (`tab:top_unc_noise`) claims a model's **per-sample** predicted uncertainty tracks the
**per-sample** injected label noise — i.e. within a given noise level, uncertainty is higher on
the labels that were corrupted more. This is pitched as going *beyond* Kolmar et al.'s
population-level result (mean uncertainty rises with the overall amount of noise).

## 2. What the metric actually computes (the bug)

"Unc-Noise ρ" comes from `scripts/generate_paper_figures.py`, in TWO spots with the same flaw:

- **Block A** — main per-rep table (`table4_uncertainty_metrics_*.csv`): loop ~line 3707;
  `unc_legacy = unc_df[unc_df['strategy']=='legacy']` (line 3690, Gaussian only);
  `model_data` grouped by rep + model only (line 3722); Spearman at line **3756**.
- **Block B** — by-strategy supp table (`table4_supp_uncertainty_by_strategy_rep.csv`): loop
  ~line 3860; groups by strategy + rep + model; Spearman at line **3896** (identical form).

In BOTH, `model_data` stacks **every σ level (0.0 … 1.0) and every iteration together**. There
is no `groupby('sigma')`. The Spearman runs over all (sample × σ × iteration) rows at once.

`injected_noise` is genuinely per-sample and fine: `fix_injected_noise()` (line 981) recomputes
it per (model, rep, sigma, iteration) as the residual of `y_true_noisy ~ y_true_original`,
i.e. ≈ the per-sample noise draw εᵢ; magnitude scales with σ.

### Why σ-pooling inflates it
Uncertainty rises with σ, and |injected_noise| rises with σ. Pooling σ makes the points a
staircase (each σ a cloud up-and-to-the-right of the last); Spearman reports the staircase as a
strong positive value. **That staircase IS the population-level Kolmar effect** — not per-sample
discrimination. A model can score 0.56 pooled with zero within-σ signal.

## 3. The deeper caveat — set expectations before running

On **Gaussian** noise, within-σ per-sample tracking is expected to be ≈ 0 for every model:
uncertainty is a function of the **features** x (GP: posterior variance from kernel distance +
a global scalar noise; NGBoost: a learned scale σ(x)), while εᵢ is an **independent random draw
on the labels**, uncorrelated with x. Nothing in the model's input can predict which sample drew
a bigger εᵢ. So do not be surprised if Gaussian within-σ ρ ≈ 0 — that is the correct, honest
result, and it confirms the current numbers were the population trend.

A genuine per-sample signal, if it exists, lives in the **value-dependent strategies** (outlier,
quantile, threshold, value-proportional), where noise is tied to label magnitude → correlated
with features → localizable. The claim there is softer ("uncertainty flags the systematically
noisier regions"), but it is real per-sample detection.

**⛔ Checkpoint decision with user:** run the within-σ analysis on Gaussian only (matches current
Table 7), or on Gaussian **plus** the value-dependent strategies (recommended — it's cheap to add
and it's the only place a positive per-sample result can appear). Default assumption for this
plan: include the value-dependent strategies.

## 4. The re-analysis to run

For each (strategy, rep, model), for each **σ ∈ {0.0, 0.3, 0.6}** separately:
`ρ_σ = spearman( per_sample_uncertainty , |injected_noise| )` over that σ's rows only
(require ≥100 finite points; keep the p-value and n).

Report per-σ ρ, plus keep the OLD pooled ρ alongside for comparison during review.

Notes on the three σ levels:
- **σ = 0.0 is the clean control.** No noise is injected, so |injected_noise| ≈ 0 with ~no
  variation and ρ is expected to be undefined/≈0. Keep it as the null floor; flag if it is not
  near zero (that would signal a data or `injected_noise` problem).
- **σ = 0.3 (moderate) and σ = 0.6 (high)** are the real tests. Confirm both exist in the raw
  data (the σ grid is 0.0–1.0 in 0.1 steps, so they should).
- Aggregation across the three σ: to be decided with the user (report each σ separately vs a
  mean of 0.3 and 0.6). Default: report each σ separately — no averaging that re-hides the
  effect.

## 5. Investigation phase (agents — do this before any coding)

Spawn agents in parallel to establish, from the server/repo (not memory):
1. **Data inventory.** Locate the raw `*_uncertainty_values.csv` files on the server. Confirm
   they contain columns: `sigma`, `injected_noise`, an uncertainty column
   (`y_pred_std_calibrated`/`y_pred_std`), `y_true_noisy`, `y_true_original`, `model`, `rep`,
   `iteration`, and a strategy (in-file or via filename). Confirm σ = 0.0, 0.3, 0.6 are present
   for the key model×rep cells (GP/SNS, NGBoost/PDV, BNN-β/PDV) and for the value-dependent
   strategies. **This determines whether we can re-analyze existing data or must re-run
   experiments.**
2. **Code path.** Confirm the two Spearman sites (lines ~3756 and ~3896) and the table writers
   (`table4c_*`, `table4_uncertainty_metrics_*`, `table4_supp_*`). Identify the cleanest place
   to add per-σ grouping (likely a shared helper).
3. **Run mechanics.** Determine exactly how the analysis script is invoked (entry point, args
   like `--qm9-dir`, expected working dir) and how long/how much memory it needs.
4. **SLURM conventions.** Read existing SLURM scripts in the repo (e.g. `slurm_scripts_*`) to
   copy the real partition names, account, module loads, conda env, and paths actually used —
   do not guess these.

⛔ Report the inventory back to the user before writing the SLURM job — if σ=0.3/0.6 per-sample
uncertainty is missing for the strategies we want, we need to re-RUN uncertainty experiments
(heavier), not just re-analyze.

## 6. Two possible execution paths (investigation decides which)

- **Path 1 — Re-analysis only (light).** If the raw per-sample uncertainty + `injected_noise`
  already exist for σ=0/0.3/0.6 across the wanted strategies: just modify the two correlation
  sites to compute per-σ Spearman, re-run the analysis script over existing CSVs, regenerate the
  tables. A SLURM job is used mainly to run on a compute node (data is large / login-node
  unfriendly), not for heavy compute.
- **Path 2 — Re-run uncertainty experiments (heavier).** If the needed σ levels or per-sample
  columns are missing: re-run the uncertainty-saving experiments (`process_and_train.py` with the
  uncertainty flag) at σ=0/0.3/0.6 for the model×rep×strategy cells of interest, then analyze.
  Bigger SLURM job; needs the partition/queue check the user mentioned.

## 7. SLURM / server specifics (to be confirmed by agents, not memory)

- Target cluster: **gateway.arc.ox.ac.uk**, run under **stat-cadd**
  (`/data/stat-cadd/scat9264/qsar_qm_models`), per the user's preference to run on stat-cadd.
- Verify the server repo is on the correct branch before submitting (history of wrong-branch
  runs — check `git branch`).
- **Help the user check whole-partition queue load before picking a partition** (not just their
  own jobs). Give them: `sinfo -o "%P %a %l %C %t %D"` (per-partition CPU load A/I/O/T + state),
  a per-partition running-vs-pending count via `squeue -t R/-t PD`, and crucially
  `sbatch --test-only [-p <part>] run_figures.sh` — estimates when THIS job would start on a
  given partition without queuing it. Compare partitions and pick the earliest start. The script
  should make the partition easy to swap. Copy account/env/module lines from an existing working
  SLURM script in the repo rather than authoring from scratch.
- SLURM scripts go to the server via SCP, not git (repo convention).
- The user will review the script and confirm the partition before submission (⛔).

## 8. Verification / sanity checks

- Re-derive the OLD pooled ρ and confirm it reproduces current Table 7 (0.56 GP/SNS, 0.47
  NGBoost/PDV, 0.38 BNN-β/PDV) — proves we're on the right code path.
- Gaussian within-σ ρ for GP should come back ≈ 0. If it's high, STOP and investigate the
  grouping or the `injected_noise` column before trusting anything.
- σ=0 control should be ≈ 0/undefined. If not, the `injected_noise` recomputation is suspect.
- Check ≥100 points per σ slice so Spearman isn't run on thin data; log any dropped cells.
- Use |injected_noise| (it's a signed regression residual); confirm its magnitude grows with σ
  per group.

## 9. Downstream paper edits (after results are in)

Depends on what the within-σ numbers show:
- If Gaussian ≈ 0 and value-dependent strategies also ≈ 0 → reframe the whole section to
  population-level (drop "per-sample / beyond the population level"); still a valid Kolmar
  extension across models and representations.
- If value-dependent strategies show a real within-σ signal → keep a (carefully worded)
  per-sample claim scoped to those strategies; rebuild Table 7 around the within-σ numbers.
- Either way: update Table 7, the two uncertainty-subsection paragraphs, and the
  abstract/scientific-contribution/conclusion "per-sample" sentences.

## 10. Open questions to settle with the user

1. Gaussian only, or Gaussian + value-dependent strategies? (§3 — recommend include them.)
2. Report each σ separately, or aggregate 0.3 & 0.6? (§4 — recommend separately.)
3. Path 1 (re-analyze) vs Path 2 (re-run) — decided by the §5 data inventory.
4. Partition/account for the SLURM job (user to choose).
5. Keep the pooled number anywhere (as an explicit population-level column) or drop it?

## 11. One-line summary
The Unc-Noise ρ pools all σ levels, so it measures the population-level (Kolmar) trend, not
per-sample tracking. Re-do it as a within-σ Spearman at σ = 0, 0.3, 0.6 (0 = clean control),
on Gaussian and ideally the value-dependent strategies; investigate the server data first with
multiple agents, verify everything against real files, run via a stat-cadd SLURM job, and expect
Gaussian to collapse to ≈ 0 — which decides how the paper's uncertainty claim gets reworded.
