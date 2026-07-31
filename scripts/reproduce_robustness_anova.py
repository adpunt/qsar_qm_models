#!/usr/bin/env python3
"""Standalone, pipeline-free reproduction of the ROBUSTNESS ANOVA that populates
the Robust_* columns of results/paper_figures_v2/table1_anova_summary.csv.

It does NOT import generate_paper_figures_v2. It re-derives everything from the
raw anova_*.csv files and then computes eta^2 TWO ways:
  (a) exactly matching the pipeline's weighted-marginal SS decomposition, and
  (b) a statsmodels anova_lm cross-check at Type I / II / III.
Finally it prints each strategy's four-tuple next to the published CSV value with
OK/MISMATCH and the absolute difference.

USAGE (on the ARC login node, where the raw anova_*.csv live):
    python3 reproduce_robustness_anova.py \
        --results-dir /data/stat-cadd/scat9264/qsar_qm_models/results \
        --csv         /data/stat-cadd/scat9264/qsar_qm_models/results/paper_figures_v2/table1_anova_summary.csv

Self-test (no raw data needed; fabricates schema-correct files and checks the
algebra path runs and code-SS == Type I on a balanced design):
    python3 reproduce_robustness_anova.py --selftest
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# ---- constants copied verbatim from generate_paper_figures_v2.py -------------
GLOBAL_MODELS_EXCLUDE = {
    'conformal_rf','conformal_qrf','conformal_dnn','conformal_rf_split',
    'conformal_qrf_split','conformal_dnn_split','dnn_bnn_variational',
    'mlp_bnn_variational','dnn_bnn_last','mlp_bnn_last','flexible_dnn',
    'flexible_dnn_256_128_64','flexible_dnn_512_256',
}
ANOVA_MODELS_EXCLUDE = {'qrf','gauche','gauche_rbf'} | GLOBAL_MODELS_EXCLUDE
ANOVA_REPS_EXCLUDE = {'sns','randomized_smiles','random_smiles','pdv','morgan'}
EXCLUDED_MODELS = {'graph_gp','gcn','gin','ginct','gin2d'}
ROBUSTNESS_BASELINE_THRESHOLD = 0.3
CATASTROPHIC_R2_THRESHOLD = -0.5
MIN_CELL_ITERS = 5
VALID_STRATEGIES = {'legacy','outlier','quantile','threshold','hetero','valprop',
                    'heteroscedastic','value_proportional'}
STRATEGY_NORMALIZE = {'heteroscedastic':'hetero','value_proportional':'valprop'}
BNN_NAME_MAP = {'bnn_full':'dnn_bnn_full','bnn_last':'dnn_bnn_last',
                'bnn_variational':'dnn_bnn_variational',
                'bnn_full_variational':'dnn_vbll',
                'dnn_bnn_full_variational':'dnn_vbll',
                'mlp_bnn_full_variational':'mlp_vbll'}
STRAT_ORDER = ['legacy','valprop','quantile','threshold','hetero','outlier']
STRAT_LABEL = {'legacy':'Gaussian','valprop':'Value-Prop.','quantile':'Quantile',
               'threshold':'Threshold','hetero':'Heteroscedastic','outlier':'Outlier'}


def load_anova_data(results_dir):
    """Faithful port of generate_paper_figures_v2.load_anova_data (L552)."""
    results_dir = Path(results_dir)
    all_data = []
    files = sorted(results_dir.glob("anova_*.csv"), key=lambda p: p.stat().st_mtime)
    print(f"  load_anova_data: {len(files)} anova_*.csv files")
    for f in files:
        if '_uncertainty_values' in f.name:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  WARN could not read {f.name}: {e}"); continue
        rest = f.stem[len('anova_'):]
        strat = None
        for s in sorted(VALID_STRATEGIES, key=len, reverse=True):
            if rest.startswith(s + '_'):
                strat = STRATEGY_NORMALIZE.get(s, s); break
        if strat:
            df['strategy'] = strat
        if 'model' in df.columns and len(df):
            cm = df['model'].iloc[0]; fn = f.stem
            if cm in ('dnn','mlp'):
                if '_bnn_full_variational' in fn: df['model'] = cm+'_bnn_full_variational'
                elif '_bnn_last' in fn: df['model'] = cm+'_bnn_last'
                elif '_bnn_full' in fn: df['model'] = cm+'_bnn_full'
        all_data.append(df)
    if not all_data:
        return None
    c = pd.concat(all_data, ignore_index=True)
    if 'model' in c.columns:
        c['model'] = c['model'].map(lambda m: BNN_NAME_MAP.get(m, m))
    if 'representation' in c.columns and 'rep' not in c.columns:
        c = c.rename(columns={'representation':'rep'})
    c = c[~c['model'].isin(GLOBAL_MODELS_EXCLUDE)]
    dedup = ['model','rep','strategy','sigma','iteration']
    if all(x in c.columns for x in dedup):
        c = c.drop_duplicates(subset=dedup, keep='last')
    print(f"  loaded {len(c)} rows")
    return c


def filter_catastrophic(df):
    """Port of filter_catastrophic_iterations (L821): drop whole iterations
    that contain any r2 < CATASTROPHIC_R2_THRESHOLD."""
    mask = df['r2'] < CATASTROPHIC_R2_THRESHOLD
    if mask.sum() == 0:
        return df
    bad = df[mask][['model','rep','strategy','iteration']].drop_duplicates()
    m = df.merge(bad, on=['model','rep','strategy','iteration'], how='left', indicator=True)
    out = m[m['_merge']=='left_only'].drop(columns='_merge')
    print(f"  catastrophic filter removed {len(df)-len(out)} rows")
    return out


def retention_auc_norm(sig, r2, base):
    srange = sig.max() - sig.min()
    return float(trapezoid(r2/base, sig)/srange) if srange > 0 else np.nan


def build_auc_rows(strategy_df):
    """Port of run_robustness_anova row-builder (L1914-1926): per-iteration
    auc_norm with the per-iteration base>=0.3 gate."""
    rows = []
    for (model, rep, it), g in strategy_df.groupby(['model','rep','iteration']):
        g = g.sort_values('sigma')
        if len(g) < 3:
            continue
        sig = g['sigma'].values.astype(float); r2 = g['r2'].values.astype(float)
        b = r2[np.isclose(sig, 0.0)]
        if len(b) == 0 or b[0] < ROBUSTNESS_BASELINE_THRESHOLD:
            continue
        a = retention_auc_norm(sig, r2, float(b[0]))
        if np.isfinite(a):
            rows.append({'model':model,'rep':rep,'iteration':it,'auc_norm':a})
    return pd.DataFrame(rows)


def apply_roster(df):
    """Port of _metric_two_way_anova gating (L1857-1875)."""
    df = df.dropna(subset=['auc_norm'])
    df = df[~df['rep'].isin(ANOVA_REPS_EXCLUDE)]
    df = df[~df['model'].isin(ANOVA_MODELS_EXCLUDE)]
    reps = df['rep'].unique()
    cs = df.groupby(['model','rep']).size()
    valid = [m for m in df['model'].unique()
             if all(cs.get((m,r),0) >= MIN_CELL_ITERS for r in reps)]
    return df[df['model'].isin(valid)], sorted(valid), sorted(reps)


def code_eta2(df, col='auc_norm'):
    """Port of _metric_two_way_anova SS block (L1879-1906)."""
    gm = df[col].mean()
    tot = ((df[col]-gm)**2).sum()
    if tot == 0:
        return None
    mm = df.groupby('model')[col].mean(); mc = df.groupby('model').size()
    ss_m = float((mc*(mm-gm)**2).sum())
    rm = df.groupby('rep')[col].mean(); rc = df.groupby('rep').size()
    ss_r = float((rc*(rm-gm)**2).sum())
    im = df.groupby(['model','rep'])[col].mean(); ic = df.groupby(['model','rep']).size()
    ss_i = 0.0
    for (m,r),cnt in ic.items():
        ss_i += cnt*(im[(m,r)] - (mm[m]+rm[r]-gm))**2
    ss_res = tot-ss_m-ss_r-ss_i
    return np.array([ss_m,ss_r,ss_i,ss_res])/tot*100


def sm_eta2(df, typ, col='auc_norm'):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    fit = smf.ols(f'{col} ~ C(model)*C(rep)', data=df).fit()
    a = anova_lm(fit, typ=typ)
    tot = a['sum_sq'].sum()
    return np.array([a.loc['C(model)','sum_sq'], a.loc['C(rep)','sum_sq'],
                     a.loc['C(model):C(rep)','sum_sq'], a.loc['Residual','sum_sq']])/tot*100


def read_csv_targets(csv_path):
    t = pd.read_csv(csv_path)
    out = {}
    for _, r in t.iterrows():
        lab = r['Strategy']
        out[lab] = np.array([r['Robust_Model_η²'], r['Robust_Rep_η²'],
                             r['Robust_Interaction_η²'], r['Robust_Residual_η²']])
    return out


def run(results_dir, csv_path):
    df = load_anova_data(results_dir)
    if df is None:
        print("ERROR: no anova_*.csv found under", results_dir); sys.exit(1)
    df = df[~df['model'].isin(EXCLUDED_MODELS)]
    df = filter_catastrophic(df)
    targets = read_csv_targets(csv_path) if csv_path and Path(csv_path).exists() else {}
    print("\n" + "="*100)
    print(f"{'Strategy':16s} {'term':6s} {'code':>8s} {'TypeI':>8s} {'TypeII':>8s} "
          f"{'TypeIII':>8s} {'CSV':>8s} {'|code-CSV|':>11s}  verdict")
    print("="*100)
    terms = ['Model','Rep','Interact','Resid']
    for s in STRAT_ORDER:
        sdf = df[df['strategy']==s]
        auc = build_auc_rows(sdf)
        rdf, valid, reps = apply_roster(auc)
        if len(rdf) < 10:
            print(f"{STRAT_LABEL[s]:16s} INSUFFICIENT ({len(rdf)} rows)"); continue
        c = code_eta2(rdf)
        try: t1 = sm_eta2(rdf,1)
        except Exception: t1 = np.full(4,np.nan)
        try: t2 = sm_eta2(rdf,2)
        except Exception: t2 = np.full(4,np.nan)
        try: t3 = sm_eta2(rdf,3)
        except Exception: t3 = np.full(4,np.nan)
        tgt = targets.get(STRAT_LABEL[s], np.full(4,np.nan))
        cell = rdf.groupby(['model','rep']).size()
        balanced = len(set(cell.values))==1
        print(f"\n-- {STRAT_LABEL[s]}  roster={len(valid)} reps={len(reps)} "
              f"rows={len(rdf)} balanced={balanced}")
        for i,term in enumerate(terms):
            d = abs(c[i]-tgt[i]) if np.isfinite(tgt[i]) else np.nan
            verdict = ('OK' if (np.isfinite(d) and d<0.05) else
                       'MISMATCH' if np.isfinite(d) else 'no-CSV')
            print(f"{'':16s} {term:6s} {c[i]:8.3f} {t1[i]:8.3f} {t2[i]:8.3f} "
                  f"{t3[i]:8.3f} {tgt[i]:8.3f} {d:11.4f}  {verdict}")


def selftest():
    """Fabricate schema-correct raw files, run the full path, assert code==TypeI
    on a balanced strategy. No real data required."""
    import tempfile, statsmodels  # noqa
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    models = ['rf','svm','xgboost','lgb','ngboost']
    reps = ['ecfp4','smiles','continuous_pdv']
    sig = np.round(np.arange(0,1.01,0.1),1)
    for strat in ['legacy']:
        for rep in reps:
            for model in models:
                rows=[]
                base = rng.uniform(0.6,0.95)
                slope = rng.uniform(0.3,0.8)
                for it in range(10):
                    b = base + rng.normal(0,0.02)
                    for sg in sig:
                        r2 = b*np.exp(-slope*sg) + rng.normal(0,0.01)
                        rows.append(dict(sigma=sg,iteration=it,model=model,rep=rep,
                                         sample_size=10000,mae=0,mse=0,rmse=0,r2=r2,
                                         pearson_corr=0,params_source='x',loss_function='mse'))
                pd.DataFrame(rows).to_csv(d/f"anova_{strat}_{rep}_{model}.csv", index=False)
    df = load_anova_data(d)
    df = filter_catastrophic(df)
    auc = build_auc_rows(df[df['strategy']=='legacy'])
    rdf, valid, reps = apply_roster(auc)
    c = code_eta2(rdf); t1 = sm_eta2(rdf,1)
    print("selftest roster:", valid, "reps:", reps, "rows:", len(rdf))
    print("code eta2 :", np.round(c,4))
    print("TypeI eta2:", np.round(t1,4))
    print("max|code-TypeI| =", np.abs(c-t1).max(), "pp  (balanced -> expect ~0)")
    assert np.abs(c-t1).max() < 1e-6, "code-SS should equal Type I on balanced design"
    print("SELFTEST PASS")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.results_dir:
        run(a.results_dir, a.csv)
    else:
        ap.error("provide --results-dir (server) or --selftest")
