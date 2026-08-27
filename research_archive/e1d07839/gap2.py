import glob, pandas as pd
base="/Users/apunt/repos/KIRBy/tests/results_server/validation_rerun"
for d in sorted(glob.glob(base+"/**/all_results.csv",recursive=True)):
    df=pd.read_csv(d)
    print("###",d, len(df))
    for c in ['model','rep','strategy','fold','dataset']:
        if c in df: print("  %s:"%c, sorted(df[c].unique()))
