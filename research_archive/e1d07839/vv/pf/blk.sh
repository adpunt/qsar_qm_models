python3 - "$1" <<PYEOF
import sys, glob, pandas as pd
root = sys.argv[1]
f = glob.glob(f"{root}/**/*_uncertainty_values.csv", recursive=True)
if not f:
    print("FAIL: no uncertainty file written"); sys.exit(1)
d = pd.read_csv(f[0])
need = {'split','strategy','sigma','fold','uncertainty','noise_scale',
        'noise_pattern','injected_noise','oof_folds_ok'}
miss = need - set(d.columns)
if miss: print("FAIL: missing columns", miss); sys.exit(1)
print(f"  OK    {f[0].split('/')[-1]}: {len(d)} rows")
print(f"  splits    : {sorted(d['split'].unique())}")
print(f"  strategies: {sorted(d['strategy'].unique())}")
print(f"  folds     : {sorted(d['fold'].unique())}  (must be all 5, not just one)")
if 'train_oof' not in set(d['split']):
    print("FAIL: no out-of-fold training rows — question A cannot be answered"); sys.exit(1)
tr = d[d.split=='train_oof']
z = tr[tr.sigma==0.0]
if len(z) and z['injected_noise'].abs().max() != 0:
    print("FAIL: sigma=0 is not a clean control"); sys.exit(1)
print("  OK    sigma=0 control is exactly zero; out-of-fold training rows present")
pat = tr['noise_pattern']
if pat.nunique() <= 1 and d['strategy'].iloc[0] != 'legacy':
    print("FAIL: noise_pattern is constant - the question-B confound control is unusable"); sys.exit(1)
if tr['uncertainty'].notna().sum() == 0:
    print("FAIL: every out-of-fold uncertainty is blank"); sys.exit(1)
if (tr['oof_folds_ok'] < 5).any():
    print(f"WARN: some out-of-fold passes were truncated "
          f"(min {int(tr['oof_folds_ok'].min())}/5 inner folds)")
print(f"  OK    noise_pattern varies ({pat.nunique()} distinct); "
      f"{tr['uncertainty'].notna().sum()} finite out-of-fold uncertainties")
PYEOF
