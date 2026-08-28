"""
Threshold investigation for noise-robustness scalars.

Goal: decide, on the REAL paper data, which single scalar best summarizes a
(curved) R²(σ) degradation curve while being (a) distinct from raw performance,
(b) structurally meaningful (carries model/strategy signal), (c) not censored.

Scope for this pass: PDV family only (continuous_pdv + pdv), baseline R² > 0.5
(removes negative-baseline junk). Nothing is pooled across noise strategies —
every (model, rep, strategy) is its own degradation curve.

Run locally after scp-ing the PDV anova_*.csv into DATA_DIR, or on the server
against ../results.

Usage:  python threshold_investigation.py [DATA_DIR]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

# Reuse the vetted final-data loader (glob anova_*.csv, parse strategy/model,
# normalize names, dedup, average iterations -> one row per model/rep/strat/sigma)
from noise_metrics_analysis import load_final_paper_data

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "../results"
REPS = {'continuous_pdv', 'pdv'}
BASELINE_MIN = 0.5
SIGMA_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ----------------------------------------------------------------------------
# Per-config metric construction
# ----------------------------------------------------------------------------
def sigma_at_retention(sig, ret, target):
    """σ where retention first drops below target (linear interp). NaN if never."""
    below = np.where(ret <= target)[0]
    if len(below) == 0:
        return np.nan                      # censored
    j = below[0]
    if j == 0:
        return sig[0]
    x0, x1, y0, y1 = sig[j-1], sig[j], ret[j-1], ret[j]
    return x1 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def config_metrics(sig, r2, base):
    ret = r2 / base
    srange = sig.max() - sig.min()
    ret_pos = np.clip(ret, 1e-3, None)
    m = {}
    # sigma_X family (the threshold sweep)
    for X in SIGMA_THRESHOLDS:
        m[f'sigma_{int(X*100)}'] = sigma_at_retention(sig, ret, X)
    # whole-curve, threshold-free
    m['auc_norm'] = trapezoid(ret, sig) / srange if srange > 0 else np.nan
    m['retention_06'] = float(ret[np.isclose(sig, 0.6)][0]) if np.any(np.isclose(sig, 0.6)) else np.nan
    # linear reference
    try:
        sl, ic, _, _, _ = stats.linregress(sig, r2)
        m['nsi'] = sl
        m['nsi_relative'] = sl / ic if ic != 0 else np.nan
    except Exception:
        m['nsi'] = m['nsi_relative'] = np.nan
    # parametric shape
    try:
        p, _ = curve_fit(lambda s, lam: np.exp(-lam*s), sig, ret_pos,
                         p0=[1.0], maxfev=10000, bounds=(0, np.inf))
        m['decay_lambda'] = float(p[0])
    except Exception:
        m['decay_lambda'] = np.nan
    try:
        p, _ = curve_fit(lambda s, tau, beta: np.exp(-np.power(np.clip(s, 0, None)/tau, beta)),
                         sig, ret_pos, p0=[0.5, 1.5], maxfev=10000,
                         bounds=([1e-3, 0.1], [np.inf, 10.0]))
        m['weibull_tau'], m['weibull_beta'] = float(p[0]), float(p[1])
    except Exception:
        m['weibull_tau'] = m['weibull_beta'] = np.nan
    return m


# ----------------------------------------------------------------------------
# Meaningful-structure diagnostic: one-way eta^2 of a metric onto a factor
# ----------------------------------------------------------------------------
def eta_squared(df, value_col, factor_col):
    """Fraction of the metric's variance explained by a design factor."""
    d = df[[value_col, factor_col]].dropna()
    if len(d) < 4 or d[factor_col].nunique() < 2:
        return np.nan
    grand = d[value_col].mean()
    ss_tot = ((d[value_col] - grand) ** 2).sum()
    if ss_tot == 0:
        return np.nan
    ss_between = 0.0
    for _, g in d.groupby(factor_col):
        ss_between += len(g) * (g[value_col].mean() - grand) ** 2
    return ss_between / ss_tot


def main():
    print("="*84)
    print("THRESHOLD INVESTIGATION — noise-robustness scalar selection")
    print(f"  reps={sorted(REPS)}  baseline>{BASELINE_MIN}  (no pooling across strategies)")
    print("="*84)

    raw = load_final_paper_data(DATA_DIR)
    raw = raw[raw['representation'].isin(REPS)].copy()

    rows = []
    for (model, rep, strat), g in raw.groupby(['model', 'representation', 'strategy']):
        g = g.sort_values('sigma')
        sig = g['sigma'].values.astype(float)
        r2 = g['r2'].values.astype(float)
        if len(sig) < 4 or not np.any(np.isclose(sig, 0.0)):
            continue
        base = r2[np.isclose(sig, 0.0)][0]
        if base <= BASELINE_MIN:
            continue
        row = {'model': model, 'rep': rep, 'strategy': strat, 'baseline': base}
        row.update(config_metrics(sig, r2, base))
        rows.append(row)

    M = pd.DataFrame(rows)
    print(f"\nConfigs after cleaning: {len(M)}  "
          f"(models={M['model'].nunique()}, strategies={sorted(M['strategy'].unique())})\n")

    candidates = [c for c in M.columns if c not in ('model', 'rep', 'strategy', 'baseline')]

    # ---- Part 1: sigma_X threshold sweep ----
    print("="*84)
    print("PART 1 — sigma_X THRESHOLD SWEEP")
    print("  censor% = undefined (never crosses) | disc = IQR of defined values")
    print("  |rho_base| low=decoupled | eta2_model/strat = meaningful structure")
    print("="*84)
    print(f"{'threshold':>10} {'censor%':>8} {'disc(IQR)':>10} {'|rho_base|':>11} "
          f"{'eta2_model':>11} {'eta2_strat':>11}")
    print("-"*74)
    for X in SIGMA_THRESHOLDS:
        col = f'sigma_{int(X*100)}'
        vals = M[col]
        censor = vals.isna().mean() * 100
        defined = vals.dropna()
        iqr = (defined.quantile(0.75) - defined.quantile(0.25)) if len(defined) > 3 else np.nan
        rho_b = abs(stats.spearmanr(M[col], M['baseline'], nan_policy='omit').correlation)
        em = eta_squared(M, col, 'model')
        es = eta_squared(M, col, 'strategy')
        print(f"{'ret<='+format(X,'.2f'):>10} {censor:>7.1f}% {iqr:>10.3f} {rho_b:>11.3f} "
              f"{em:>11.3f} {es:>11.3f}")

    # ---- Part 2: full candidate scorecard ----
    print("\n" + "="*84)
    print("PART 2 — FULL CANDIDATE SCORECARD")
    print("  Good scalar: LOW |rho_base|, HIGH eta2_model, some eta2_strat, LOW censor%")
    print("="*84)
    print(f"{'metric':>16} {'|rho_base|':>11} {'eta2_model':>11} {'eta2_strat':>11} "
          f"{'eta2_rep':>9} {'censor%':>8}")
    print("-"*78)
    score_rows = []
    for c in candidates:
        rho_b = abs(stats.spearmanr(M[c], M['baseline'], nan_policy='omit').correlation)
        em = eta_squared(M, c, 'model')
        es = eta_squared(M, c, 'strategy')
        er = eta_squared(M, c, 'rep')
        censor = M[c].isna().mean() * 100
        score_rows.append({'metric': c, 'rho_base': rho_b, 'eta2_model': em,
                           'eta2_strat': es, 'eta2_rep': er, 'censor': censor})
        print(f"{c:>16} {rho_b:>11.3f} {em:>11.3f} {es:>11.3f} {er:>9.3f} {censor:>7.1f}%")

    S = pd.DataFrame(score_rows)

    # ---- Part 3: composite desirability score ----
    # Reward decoupling + model structure; penalize censoring. All rank-normalized
    # so units don't matter. (strategy structure reported but not scored — it's a
    # property of the data, not always desired in the summary scalar.)
    print("\n" + "="*84)
    print("PART 3 — COMPOSITE DESIRABILITY (rank-normalized, higher=better)")
    print("  = decoupling_rank + model_structure_rank - censor_penalty")
    print("="*84)
    S['r_decouple'] = (1 - S['rho_base']).rank(pct=True)
    S['r_model'] = S['eta2_model'].rank(pct=True)
    S['r_censor'] = (1 - S['censor'] / 100).rank(pct=True)
    S['desirability'] = (S['r_decouple'] + S['r_model'] + S['r_censor']) / 3
    S = S.sort_values('desirability', ascending=False)
    print(f"{'metric':>16} {'desir.':>8} {'|rho_base|':>11} {'eta2_model':>11} {'censor%':>8}")
    print("-"*60)
    for _, r in S.iterrows():
        print(f"{r['metric']:>16} {r['desirability']:>8.3f} {r['rho_base']:>11.3f} "
              f"{r['eta2_model']:>11.3f} {r['censor']:>7.1f}%")

    # ---- Part 4: rank agreement among the survivors ----
    print("\n" + "="*84)
    print("PART 4 — RANK AGREEMENT among top candidates (do they order configs alike?)")
    print("="*84)
    top = S.head(6)['metric'].tolist()
    corr = pd.DataFrame(index=top, columns=top, dtype=float)
    for a in top:
        for b in top:
            corr.loc[a, b] = stats.spearmanr(M[a], M[b], nan_policy='omit').correlation
    print(corr.round(3).to_string())

    out = Path(DATA_DIR) / "metric_comparison_analysis"
    out.mkdir(exist_ok=True, parents=True)
    M.to_csv(out / "threshold_investigation_configs.csv", index=False)
    S.to_csv(out / "threshold_investigation_scorecard.csv", index=False)
    print(f"\n✓ Saved per-config metrics + scorecard to {out}")


if __name__ == "__main__":
    main()
