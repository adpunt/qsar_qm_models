#!/usr/bin/env python3
"""deep_analysis.py — one reproducible script that regenerates every robustness
and uncertainty table we care about, for BOTH studies, side by side:

  * QM9 (main study): HOMO-LUMO gap prediction.
  * Validation (external, experimental): OpenADMET-LogD, OpenADMET-Caco2_Efflux,
    ChEMBL-hERG-Ki.

Every finding is emitted for QM9 AND validation separately. Where a metric is not
available locally the script writes an explicit "GAP: ... NEEDS ..." row rather
than fabricating a number.

Design / data provenance
------------------------
QM9 robustness:
  - Raw per-run anova_*.csv are SERVER-ONLY. If they happen to be present in the
    results dir, we reuse generate_paper_figures_v2.load_anova_data +
    filter_catastrophic_iterations + calculate_robustness (the real pipeline,
    which is what protects valprop from catastrophic iterations).
  - Otherwise we degrade gracefully to the local aggregate
    table2_supp_auc_all_reps.csv, which is ALREADY pipeline-filtered (valprop is
    clean there — verified min auc_norm ~0.41, no ~-1e29 values). We melt it to
    long (model, rep, strategy, auc_norm).
  - qm9_auc_with_baseline.csv is a server dump that carries baseline_r2 but whose
    'valprop' strategy is CORRUPTED (catastrophic ~-1e29 because a catastrophic-
    iteration filter was skipped). If present we USE it for baseline but GUARD
    valprop (drop rows failing a sanity floor). If absent, QM9 baseline-vs-
    robustness is reported as a GAP.

Validation robustness:
  - table_validation_auc_full.csv (dataset, model, rep, strategy, auc_norm,
    baseline_r2). Already aggregated + has baseline, so correlations work per
    dataset.

Uncertainty quality:
  - QM9: table4_supp_uncertainty_by_strategy_rep.csv (Unc-Error rho,
    Coverage 1sigma/2sigma). DNN/MLP (~zero uncertainty) are already excluded
    from this table; we additionally guard on Mean Uncertainty >= 1e-3.
  - Validation: table_validation_probabilistic.csv only holds an RF-vs-QRF
    AUC_norm comparison per dataset — NO per-model calibration (coverage /
    Unc-Error rho). Reported as a GAP.

Outputs: results/paper_figures_v2/deep_*.csv (+ stdout summary).
Dependency-light: pandas / numpy / scipy only.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent                     # .../scripts
REPO = HERE.parent                                         # repo root
RESULTS_DIR = REPO / "results" / "paper_figures_v2"

# Mirror generate_paper_figures_v2 thresholds so the local path matches the pipeline.
CATASTROPHIC_R2_THRESHOLD = -0.5      # per-iteration R2 below this = training failure
ROBUSTNESS_BASELINE_THRESHOLD = 0.3   # baseline R2(0) below this => config excluded
MIN_MEAN_UNCERTAINTY = 1e-3           # models below this have ~zero uncertainty
BASELINE_SANITY_FLOOR = -10.0         # auc_norm below this = corrupted (valprop dump)

# Canonical raw strategy names <-> display labels (from generate_paper_figures_v2).
STRATEGY_LABELS = {
    'legacy': 'Gaussian',
    'valprop': 'Value-Prop.',
    'quantile': 'Quantile',
    'threshold': 'Threshold',
    'outlier': 'Outlier',
    'hetero': 'Heteroscedastic',
}
LABEL_TO_RAW = {v: k for k, v in STRATEGY_LABELS.items()}
STRATEGY_ORDER = ['legacy', 'outlier', 'quantile', 'threshold', 'valprop', 'hetero']

VALIDATION_DATASETS = ['OpenADMET-LogD', 'OpenADMET-Caco2_Efflux', 'ChEMBL-hERG-Ki']

GAPS = []  # collected machine-readable gaps


def _gap(what, needs):
    msg = f"GAP: {what} NEEDS {needs}"
    GAPS.append({'gap': what, 'needs': needs})
    print("  " + msg)
    return msg


def _to_raw_strategy(s):
    """Map either a raw name or a display label to the canonical raw name."""
    if s in STRATEGY_LABELS:
        return s
    return LABEL_TO_RAW.get(s, s)


# --------------------------------------------------------------------------- #
# QM9 robustness loading
# --------------------------------------------------------------------------- #
def load_qm9_robustness(results_dir):
    """Return (long_df[model,rep,strategy,auc_norm,baseline_r2 or NaN], provenance).

    Prefers raw anova_*.csv via the real pipeline; degrades to the local
    pipeline-filtered aggregate table2_supp_auc_all_reps.csv otherwise.
    """
    results_dir = Path(results_dir)
    raw_files = sorted(results_dir.glob("anova_*.csv"))
    raw_files = [f for f in raw_files if '_uncertainty_values' not in f.name]

    if raw_files:
        # --- Raw path: reuse the real pipeline (guarded import) ---
        try:
            sys.path.insert(0, str(HERE))
            import generate_paper_figures_v2 as g  # noqa: E402
            print(f"  QM9: found {len(raw_files)} raw anova_*.csv -> using pipeline "
                  "(load_anova_data + filter_catastrophic_iterations + calculate_robustness)")
            raw = g.load_anova_data(results_dir)
            filtered, log = g.filter_catastrophic_iterations(raw)
            print(f"  QM9: catastrophic filter removed {len(raw) - len(filtered)} rows "
                  f"(protects valprop); {len(log)} catastrophic R2 values logged")
            robust, _excl = g.calculate_robustness(filtered)
            robust = robust[['model', 'rep', 'strategy', 'auc_norm', 'baseline_r2']].copy()
            return robust, 'raw_pipeline (anova_*.csv + catastrophic filter)'
        except Exception as e:  # pragma: no cover - degrade gracefully
            print(f"  QM9: raw pipeline failed ({e}); falling back to aggregate CSV")

    # --- Aggregate path ---
    agg_path = results_dir / "table2_supp_auc_all_reps.csv"
    if not agg_path.exists():
        _gap("QM9 robustness aggregate table2_supp_auc_all_reps.csv",
             "server dump of table2_supp or raw anova_*.csv")
        return pd.DataFrame(columns=['model', 'rep', 'strategy', 'auc_norm', 'baseline_r2']), 'MISSING'

    agg = pd.read_csv(agg_path)
    strat_cols = [c for c in agg.columns if c in LABEL_TO_RAW]
    long = agg.melt(id_vars=['model', 'rep'], value_vars=strat_cols,
                    var_name='strategy_label', value_name='auc_norm')
    long['strategy'] = long['strategy_label'].map(LABEL_TO_RAW)
    long = long.dropna(subset=['auc_norm'])
    long['baseline_r2'] = np.nan  # aggregate carries no baseline

    # Try to attach baseline from server dump (guard corrupted valprop).
    baseline_path = results_dir / "qm9_auc_with_baseline.csv"
    if baseline_path.exists():
        bl = pd.read_csv(baseline_path)
        n0 = len(bl)
        # Guard: drop corrupted rows (valprop catastrophic ~-1e29).
        bl = bl[bl['auc_norm'] > BASELINE_SANITY_FLOOR].copy()
        n_guard = n0 - len(bl)
        if n_guard:
            print(f"  QM9: qm9_auc_with_baseline.csv present -> attaching baseline_r2; "
                  f"GUARDED {n_guard} corrupted rows (valprop catastrophic)")
        bl['strategy'] = bl['strategy'].map(_to_raw_strategy)
        bl = bl[['model', 'rep', 'strategy', 'baseline_r2']].drop_duplicates()
        long = long.drop(columns='baseline_r2').merge(
            bl, on=['model', 'rep', 'strategy'], how='left')
        prov = 'aggregate table2_supp (pipeline-filtered) + qm9_auc_with_baseline (valprop-guarded)'
    else:
        print("  QM9: qm9_auc_with_baseline.csv ABSENT -> baseline_r2 unavailable")
        prov = 'aggregate table2_supp_auc_all_reps.csv (pipeline-filtered; NO baseline)'

    return long[['model', 'rep', 'strategy', 'auc_norm', 'baseline_r2']], prov


# --------------------------------------------------------------------------- #
# Validation robustness loading
# --------------------------------------------------------------------------- #
def load_validation_robustness(results_dir):
    path = Path(results_dir) / "table_validation_auc_full.csv"
    if not path.exists():
        _gap("validation robustness table_validation_auc_full.csv", "KIRBy re-run")
        return pd.DataFrame(columns=['dataset', 'model', 'rep', 'strategy', 'auc_norm', 'baseline_r2'])
    v = pd.read_csv(path)
    v['strategy'] = v['strategy'].map(_to_raw_strategy)
    return v


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _agg(df, group_cols, val='auc_norm'):
    g = (df.groupby(group_cols)[val]
           .agg(['mean', 'std', 'count'])
           .reset_index()
           .rename(columns={'mean': f'{val}_mean', 'std': f'{val}_std', 'count': 'n'}))
    return g


# --------------------------------------------------------------------------- #
# Correlation helper (baseline vs robustness)
# --------------------------------------------------------------------------- #
def baseline_corr(df, label):
    """Pearson & Spearman corr between baseline_r2 and auc_norm."""
    sub = df.dropna(subset=['baseline_r2', 'auc_norm'])
    if len(sub) < 3:
        return {'scope': label, 'n': len(sub), 'pearson_r': np.nan,
                'pearson_p': np.nan, 'spearman_r': np.nan, 'spearman_p': np.nan}
    pr, pp = stats.pearsonr(sub['baseline_r2'], sub['auc_norm'])
    sr, sp = stats.spearmanr(sub['baseline_r2'], sub['auc_norm'])
    return {'scope': label, 'n': len(sub), 'pearson_r': pr, 'pearson_p': pp,
            'spearman_r': sr, 'spearman_p': sp}


# --------------------------------------------------------------------------- #
# Uncertainty quality (QM9)
# --------------------------------------------------------------------------- #
def load_qm9_uncertainty(results_dir):
    path = Path(results_dir) / "table4_supp_uncertainty_by_strategy_rep.csv"
    if not path.exists():
        _gap("QM9 uncertainty table4_supp_uncertainty_by_strategy_rep.csv", "server figure run")
        return pd.DataFrame()
    u = pd.read_csv(path)
    u = u.rename(columns={
        'Unc-Error ρ': 'unc_error_rho',
        'Coverage 1σ': 'coverage_1sig',
        'Coverage 2σ': 'coverage_2sig',
        'Mean Uncertainty': 'mean_unc',
        'Strategy': 'strategy_label', 'Rep': 'rep', 'Model': 'model',
    })
    # Guard: drop ~zero-uncertainty models (DNN/MLP already excluded upstream).
    n0 = len(u)
    u = u[u['mean_unc'] >= MIN_MEAN_UNCERTAINTY].copy()
    if len(u) != n0:
        print(f"  QM9 uncertainty: dropped {n0 - len(u)} rows with mean_unc < {MIN_MEAN_UNCERTAINTY}")
    u['strategy'] = u['strategy_label'].map(_to_raw_strategy)
    return u


def summarize_uncertainty(u, by):
    cols = ['unc_error_rho', 'coverage_1sig', 'coverage_2sig']
    g = (u.groupby(by)[cols]
           .mean()
           .reset_index())
    g['n'] = u.groupby(by).size().values
    return g


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    rd = RESULTS_DIR
    print("=" * 78)
    print("deep_analysis.py — QM9 (main) + Validation (LogD / Caco2_Efflux / hERG-Ki)")
    print("=" * 78)
    print(f"results dir: {rd}")

    outputs = {}

    # ---- Catastrophic filter reference (valprop protection) ----------------
    print("\n[0] Catastrophic-iteration filter reference")
    cat_path = rd / "filtered_catastrophic_iterations.csv"
    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        by_strat = cat.groupby('strategy').size().rename('n_rows_flagged').reset_index()
        by_strat['strategy'] = by_strat['strategy'].map(_to_raw_strategy)
        iters = (cat[['model', 'rep', 'strategy', 'iteration']].drop_duplicates()
                 .groupby('strategy').size().rename('n_iterations_dropped').reset_index())
        iters['strategy'] = iters['strategy'].map(_to_raw_strategy)
        cat_ref = by_strat.merge(iters, on='strategy', how='outer')
        print(cat_ref.to_string(index=False))
        print(f"  -> valprop protected: {int(cat_ref.loc[cat_ref.strategy=='valprop','n_rows_flagged'].sum())} "
              "catastrophic rows would corrupt valprop if not filtered")
        outputs['deep_catastrophic_filter_reference.csv'] = cat_ref
    else:
        _gap("filtered_catastrophic_iterations.csv", "server figure run")

    # ---- Load robustness ---------------------------------------------------
    print("\n[1] Loading robustness data")
    qm9, qm9_prov = load_qm9_robustness(rd)
    print(f"  QM9 provenance: {qm9_prov}")
    print(f"  QM9 rows: {len(qm9)}  models={qm9['model'].nunique()} reps={qm9['rep'].nunique()} "
          f"strategies={sorted(qm9['strategy'].dropna().unique())}")
    val = load_validation_robustness(rd)
    print(f"  Validation rows: {len(val)}  datasets={sorted(val['dataset'].unique())}")
    print(f"  Validation reps={sorted(val['rep'].unique())} "
          f"strategies={sorted(val['strategy'].unique())}")

    qm9_valprop_ok = 'valprop' in set(qm9['strategy'].dropna().unique())
    print(f"  valprop present & non-corrupt -> QM9: {qm9_valprop_ok} "
          f"(min auc_norm={qm9.loc[qm9.strategy=='valprop','auc_norm'].min():.3f}), "
          f"Validation: {'valprop' in set(val['strategy'].unique())} "
          f"(min auc_norm={val.loc[val.strategy=='valprop','auc_norm'].min():.3f})")

    # ---- (A) Per-strategy REPRESENTATION robustness ------------------------
    print("\n[2] Per-strategy REPRESENTATION robustness (mean auc_norm; higher=more robust)")
    if len(qm9):
        qm9_rep = _agg(qm9, ['rep', 'strategy'])
        outputs['deep_qm9_rep_robustness_by_strategy.csv'] = qm9_rep
        print("  QM9 (rep x strategy) — top per strategy:")
        piv = qm9_rep.pivot_table(index='rep', columns='strategy', values='auc_norm_mean')
        piv = piv[[c for c in STRATEGY_ORDER if c in piv.columns]]
        print(piv.round(3).to_string())
    if len(val):
        val_rep = _agg(val, ['dataset', 'rep', 'strategy'])
        outputs['deep_validation_rep_robustness_by_strategy.csv'] = val_rep
        print("\n  Validation (dataset x rep x strategy) — mean auc_norm by rep (pooled strategies shown per dataset):")
        for ds in VALIDATION_DATASETS:
            sub = val[val.dataset == ds]
            if not len(sub):
                print(f"    {ds}: no rows")
                continue
            rp = sub.groupby('rep')['auc_norm'].mean().sort_values(ascending=False)
            print(f"    {ds}: " + ", ".join(f"{r}={v:.3f}" for r, v in rp.items()))
    holds = "BOTH" if len(qm9) and len(val) else ("QM9 only" if len(qm9) else "validation only")
    print(f"  -> per-strategy representation robustness available on: {holds}")

    # ---- (B) Per-strategy MODEL robustness ---------------------------------
    print("\n[3] Per-strategy MODEL robustness (mean auc_norm)")
    if len(qm9):
        qm9_model = _agg(qm9, ['model', 'strategy'])
        outputs['deep_qm9_model_robustness_by_strategy.csv'] = qm9_model
        piv = qm9_model.pivot_table(index='model', columns='strategy', values='auc_norm_mean')
        piv = piv[[c for c in STRATEGY_ORDER if c in piv.columns]]
        print("  QM9 (model x strategy):")
        print(piv.round(3).to_string())
    if len(val):
        val_model = _agg(val, ['dataset', 'model', 'strategy'])
        outputs['deep_validation_model_robustness_by_strategy.csv'] = val_model
        print("\n  Validation (model, mean auc_norm pooled over reps, per dataset):")
        for ds in VALIDATION_DATASETS:
            sub = val[val.dataset == ds]
            if not len(sub):
                continue
            mr = sub.groupby('model')['auc_norm'].mean().sort_values(ascending=False)
            print(f"    {ds}: " + ", ".join(f"{m}={v:.3f}" for m, v in mr.items()))
    print(f"  -> per-strategy model robustness available on: {holds}")

    # ---- (C) Baseline vs robustness correlation ----------------------------
    print("\n[4] Baseline R2 vs robustness (auc_norm) correlation")
    corr_rows = []
    # QM9
    if len(qm9) and qm9['baseline_r2'].notna().any():
        corr_rows.append({'study': 'QM9', **baseline_corr(qm9, 'QM9 (all configs)')})
    else:
        note = _gap("QM9 baseline_r2 for baseline-vs-robustness correlation",
                    "server dump qm9_auc_with_baseline.csv (valprop-guarded) OR raw anova_*.csv")
        corr_rows.append({'study': 'QM9', 'scope': 'QM9 (all configs)', 'n': 0,
                          'pearson_r': np.nan, 'pearson_p': np.nan,
                          'spearman_r': np.nan, 'spearman_p': np.nan, 'note': note})
    # Validation: pooled + per dataset
    if len(val):
        c = baseline_corr(val, 'Validation (pooled all datasets)')
        c['study'] = 'Validation'
        corr_rows.append(c)
        for ds in VALIDATION_DATASETS:
            sub = val[val.dataset == ds]
            if len(sub):
                c = baseline_corr(sub, f'Validation ({ds})')
                c['study'] = 'Validation'
                corr_rows.append(c)
    corr_df = pd.DataFrame(corr_rows)
    outputs['deep_baseline_vs_robustness.csv'] = corr_df
    print(corr_df.to_string(index=False))
    q_has = bool(len(qm9) and qm9['baseline_r2'].notna().any())
    print(f"  -> baseline-vs-robustness available on: "
          f"{'BOTH' if q_has and len(val) else 'validation only (QM9 = GAP)'}")

    # ---- (D) Uncertainty quality -------------------------------------------
    print("\n[5] Uncertainty quality")
    u = load_qm9_uncertainty(rd)
    if len(u):
        by_model = summarize_uncertainty(u, 'model')
        by_strat = summarize_uncertainty(u, 'strategy')
        by_strat['strategy'] = pd.Categorical(by_strat['strategy'], STRATEGY_ORDER, ordered=True)
        by_strat = by_strat.sort_values('strategy')
        outputs['deep_qm9_uncertainty_by_model.csv'] = by_model
        outputs['deep_qm9_uncertainty_by_strategy.csv'] = by_strat
        print("  QM9 uncertainty by MODEL (Unc-Error rho higher=better; Cov2sig target~0.95):")
        print(by_model.round(3).to_string(index=False))
        print("\n  QM9 uncertainty by STRATEGY:")
        print(by_strat.round(3).to_string(index=False))
    # Validation uncertainty
    prob_path = rd / "table_validation_probabilistic.csv"
    if prob_path.exists():
        prob = pd.read_csv(prob_path)
        outputs['deep_validation_uncertainty_rfqrf.csv'] = prob
        print("\n  Validation probabilistic table (ONLY an RF-vs-QRF AUC_norm robustness "
              "comparison — no coverage/Unc-Error rho):")
        print(prob.to_string(index=False))
        _gap("Validation per-model calibration (Unc-Error rho, Coverage) for "
             "LogD / Caco2_Efflux / hERG-Ki",
             "KIRBy uncertainty run (only RF-vs-QRF robustness comparison exists locally)")
    else:
        _gap("Validation uncertainty table_validation_probabilistic.csv", "KIRBy uncertainty run")
    print("  -> uncertainty quality available on: QM9 only (validation calibration = GAP)")

    # ---- Gaps table --------------------------------------------------------
    outputs['deep_gaps.csv'] = pd.DataFrame(GAPS) if GAPS else pd.DataFrame(columns=['gap', 'needs'])

    # ---- Write -------------------------------------------------------------
    print("\n[6] Writing outputs to", rd)
    for name, df in outputs.items():
        p = rd / name
        df.to_csv(p, index=False)
        print(f"  wrote {name:52s} ({len(df)} rows)")

    print("\nDONE. Studies covered: QM9=%s  Validation=%s  valprop-filtered=%s" % (
        bool(len(qm9)), bool(len(val)), qm9_valprop_ok and ('valprop' in set(val['strategy'].unique()))))
    return outputs


if __name__ == "__main__":
    main()
