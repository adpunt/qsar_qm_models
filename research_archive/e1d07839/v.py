import pandas as pd, numpy as np, glob
from scipy import stats
fs = glob.glob('/Users/apunt/repos/KIRBy/tests/results/validation/*/all_results.csv')
df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print(df.shape, sorted(df.model.unique()), sorted(df.rep.unique()), sorted(df.strategy.unique()), sorted(df.dataset.unique()))
print('folds', sorted(df.fold.unique()), 'sigmas', sorted(df.sigma.unique()))
df.to_pickle('val.pkl')
