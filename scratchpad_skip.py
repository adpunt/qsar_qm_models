import pandas as pd, numpy as np
files={'gp':'gp','qrf':'qrf','ngboost':'ngboost','dnn_full_bnn':'dnn_full_bnn','dnn_lastlayer_bnn':'dnn_lastlayer_bnn','dnn_var_bnn':'dnn_var_bnn'}
for name in files:
    df=pd.read_csv(f'results/phase1_continuous_pdv_{name}_uncertainty_values.csv').rename(columns={'representation':'rep'})
    g=df.groupby(['model','rep','sigma','iteration'])
    sizes=g.size()
    n_total=len(sizes); n_small=(sizes<10).sum()
    print(f"{name:20s} modelcol={df['model'].iloc[0]:16s} groups={n_total} skipped(<10)={n_small} minsize={sizes.min()} rows={len(df)}")
