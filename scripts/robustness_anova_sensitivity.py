#!/usr/bin/env python3
"""Task E: sensitivity of the ROBUSTNESS two-way ANOVA eta^2 to analysis choices.

Run ON THE SERVER where the raw anova_*.csv live:
    /data/stat-cadd/scat9264/qsar_qm_models/results/

Reproduces table1's robustness columns and sweeps 4 defensible variations:
  (1) SS type: Type I (model-first & rep-first) / Type II / Type III / code-engine
  (2) Roster: balanced MIN>=5 / include NN-Bayes where present / all-with-any-data
  (3) Granularity: per-iteration rows vs per-(model,rep) cell means
  (4) Gate: R2(0)>=0.3 / >=0.6 / no gate

Usage:  python robustness_anova_sensitivity.py --qm9-dir /data/stat-cadd/scat9264/qsar_qm_models/results
Requires: pandas, numpy, scipy, statsmodels
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
from scipy.integrate import trapezoid
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ---- constants copied verbatim from generate_paper_figures_v2.py ----
GLOBAL_MODELS_EXCLUDE = {'conformal_rf','conformal_qrf','conformal_dnn',
    'conformal_rf_split','conformal_qrf_split','conformal_dnn_split',
    'dnn_bnn_variational','mlp_bnn_variational','dnn_bnn_last','mlp_bnn_last',
    'flexible_dnn','flexible_dnn_256_128_64','flexible_dnn_512_256'}
ANOVA_MODELS_EXCLUDE = {'qrf','gauche','gauche_rbf'} | GLOBAL_MODELS_EXCLUDE
ANOVA_REPS_EXCLUDE = {'sns','randomized_smiles','random_smiles','pdv','morgan'}
NN_BAYES = {'dnn_bnn_full','mlp_bnn_full','dnn_vbll','mlp_vbll'}
MIN_CELL_ITERS = 5
STRATS = ['legacy','quantile','threshold','valprop','hetero','outlier']
SMAP = {'legacy':'Gaussian','quantile':'Quantile','threshold':'Threshold',
        'valprop':'Value-Prop.','hetero':'Heteroscedastic','outlier':'Outlier'}
BNN_NAME_MAP = {'bnn_full':'dnn_bnn_full','bnn_last':'dnn_bnn_last',
    'bnn_full_variational':'dnn_vbll','mlp_bnn_full_variational':'mlp_vbll'}

def load_anova(results_dir):
    frames=[]
    for f in sorted(glob.glob(os.path.join(results_dir,'anova_*.csv'))):
        if '_uncertainty_values' in f: continue
        base=os.path.basename(f)[len('anova_'):-4]
        strat=next((s for s in sorted(STRATS,key=len,reverse=True)
                    if base.startswith(s)), None)
        if strat is None: continue
        try: d=pd.read_csv(f)
        except Exception: continue
        if 'representation' in d.columns: d=d.rename(columns={'representation':'rep'})
        d['strategy']=strat
        d['model']=d['model'].replace(BNN_NAME_MAP)
        frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    df=df.drop_duplicates(subset=['model','rep','strategy','sigma','iteration'],keep='last')
    return df

def auc_norm(sig,r2,base):
    ret=r2/base; sr=sig.max()-sig.min()
    return float(trapezoid(ret,sig)/sr) if sr>0 else np.nan

def per_iteration_rows(df, gate):
    rows=[]
    for (m,r,it),g in df.groupby(['model','rep','iteration']):
        g=g.sort_values('sigma')
        if len(g)<3: continue
        sig=g['sigma'].values.astype(float); r2=g['r2'].values.astype(float)
        b=r2[np.isclose(sig,0.0)]
        if len(b)==0: continue
        if gate is not None and b[0]<gate: continue
        a=auc_norm(sig,r2,float(b[0]))
        if np.isfinite(a): rows.append({'model':m,'rep':r,'iteration':it,'auc_norm':a})
    return pd.DataFrame(rows)

def apply_roster(mdf, mode):
    mdf=mdf[~mdf['rep'].isin(ANOVA_REPS_EXCLUDE)]
    mdf=mdf[~mdf['model'].isin(ANOVA_MODELS_EXCLUDE)]
    reps=mdf['rep'].unique(); cs=mdf.groupby(['model','rep']).size()
    models=mdf['model'].unique()
    if mode=='balanced_min5':
        keep=[m for m in models if all(cs.get((m,r),0)>=MIN_CELL_ITERS for r in reps)]
    elif mode=='nn_bayes_present':  # min5 for non-NN, any-data for NN-Bayes
        keep=[m for m in models if (m in NN_BAYES) or
              all(cs.get((m,r),0)>=MIN_CELL_ITERS for r in reps)]
    elif mode=='all_any_data':
        keep=[m for m in models if all(cs.get((m,r),0)>=1 for r in reps)]
    else: raise ValueError(mode)
    return mdf[mdf['model'].isin(keep)], sorted(keep)

def code_engine(df,col='auc_norm'):
    g=df[col].mean(); tot=((df[col]-g)**2).sum()
    if tot==0: return None
    mm=df.groupby('model')[col].mean(); mc=df.groupby('model').size()
    ssm=(mc*(mm-g)**2).sum()
    rm=df.groupby('rep')[col].mean(); rc=df.groupby('rep').size()
    ssr=(rc*(rm-g)**2).sum()
    ic=df.groupby(['model','rep']).size(); im=df.groupby(['model','rep'])[col].mean()
    ssi=sum(c*(im[(m,r)]-(mm[m]+rm[r]-g))**2 for (m,r),c in ic.items())
    return np.array([ssm,ssr,ssi,tot-ssm-ssr-ssi])/tot*100

def sm_engine(df,typ,order='model',col='auc_norm'):
    if df['model'].nunique()<2 or df['rep'].nunique()<2: return None
    f=f'{col} ~ C(model)*C(rep)' if order=='model' else f'{col} ~ C(rep)*C(model)'
    a=anova_lm(smf.ols(f,data=df).fit(),typ=typ); tot=a['sum_sq'].sum()
    ik='C(model):C(rep)' if 'C(model):C(rep)' in a.index else 'C(rep):C(model)'
    def gv(k): return a.loc[k,'sum_sq'] if k in a.index else 0.0
    return np.array([gv('C(model)'),gv('C(rep)'),gv(ik),a.loc['Residual','sum_sq']])/tot*100

def fmt(v): return "None" if v is None else "  ".join(f"{x:5.1f}" for x in v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--qm9-dir',default='/data/stat-cadd/scat9264/qsar_qm_models/results')
    a=ap.parse_args()
    df=load_anova(a.qm9_dir)
    print(f"Loaded {len(df)} rows from {a.qm9_dir}\n")
    hdr="Model   Rep    Int    Resid"

    # ---- Variation 1 (+ term order): SS type, DEFAULT roster/gran/gate ----
    print("="*70); print("VARIATION 1 — SS TYPE (roster=balanced_min5, per-iter, gate=0.3)")
    print("="*70)
    for s in STRATS:
        d=df[df['strategy']==s]; mdf=per_iteration_rows(d,0.3)
        mdf,roster=apply_roster(mdf,'balanced_min5')
        print(f"\n{SMAP[s]}  (n={len(mdf)}, {len(roster)} models: {roster})")
        print(f"          {hdr}")
        print(f"  code    {fmt(code_engine(mdf))}")
        print(f"  TypeI-M {fmt(sm_engine(mdf,1,'model'))}")
        print(f"  TypeI-R {fmt(sm_engine(mdf,1,'rep'))}")
        print(f"  TypeII  {fmt(sm_engine(mdf,2))}")
        print(f"  TypeIII {fmt(sm_engine(mdf,3))}")

    # ---- Variation 2: roster (code engine, per-iter, gate=0.3) ----
    print("\n"+"="*70); print("VARIATION 2 — ROSTER (code engine, per-iter, gate=0.3)")
    print("="*70)
    for s in STRATS:
        d=df[df['strategy']==s]; base=per_iteration_rows(d,0.3)
        print(f"\n{SMAP[s]}          {hdr}")
        for mode in ['balanced_min5','nn_bayes_present','all_any_data']:
            mdf,roster=apply_roster(base.copy(),mode)
            print(f"  {mode:16s} ({len(roster)}m) {fmt(code_engine(mdf))}")

    # ---- Variation 3: granularity (code engine, balanced, gate=0.3) ----
    print("\n"+"="*70); print("VARIATION 3 — GRANULARITY (code engine, balanced_min5, gate=0.3)")
    print("="*70)
    for s in STRATS:
        d=df[df['strategy']==s]; mdf=per_iteration_rows(d,0.3)
        mdf,roster=apply_roster(mdf,'balanced_min5')
        cell=mdf.groupby(['model','rep'],as_index=False)['auc_norm'].mean()
        print(f"\n{SMAP[s]}          {hdr}")
        print(f"  per-iteration   {fmt(code_engine(mdf))}")
        print(f"  cell-means      {fmt(code_engine(cell))}")

    # ---- Variation 4: gate (code engine, balanced, per-iter) ----
    print("\n"+"="*70); print("VARIATION 4 — GATE (code engine, balanced_min5, per-iter)")
    print("="*70)
    for s in STRATS:
        d=df[df['strategy']==s]
        print(f"\n{SMAP[s]}          {hdr}")
        for g,lab in [(0.3,'gate>=0.3'),(0.6,'gate>=0.6'),(None,'no-gate ')]:
            mdf=per_iteration_rows(d,g)
            mdf,roster=apply_roster(mdf,'balanced_min5')
            print(f"  {lab} ({len(roster)}m) {fmt(code_engine(mdf))}")

if __name__=='__main__':
    main()
