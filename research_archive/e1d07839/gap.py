import os,glob
import pandas as pd
base="/Users/apunt/repos/KIRBy/tests/results_server"
for d in sorted(glob.glob(base+"/**/all_results.csv",recursive=True))+sorted(glob.glob(base+"/all_results.csv")):
    df=pd.read_csv(d)
    print("###",d, len(df))
    print("  models:",sorted(df['model'].unique()) if 'model' in df else df.columns.tolist())
    if 'rep' in df: print("  reps:",sorted(df['rep'].unique()))
    if 'strategy' in df: print("  strategies:",sorted(df['strategy'].unique()))
    if 'fold' in df: print("  folds:",sorted(df['fold'].unique()))
    if 'dataset' in df: print("  datasets:",sorted(df['dataset'].unique()))
