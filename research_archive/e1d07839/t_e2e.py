import numpy as np, sys, itertools, random
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
import alternative_data_noise_robustness as M
from pathlib import Path
rng=random.Random(0)
frag=['C','N','O','CC','CO','CN','c1ccccc1','C1CCCCC1','CCO','C(=O)O','CF','CCl','CBr','C#N','CS']
sm=[]
seen=set()
while len(sm)<300:
    s=''.join(rng.choice(frag) for _ in range(rng.randint(2,5)))
    from rdkit import Chem
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    cs=Chem.MolToSmiles(m)
    if cs in seen: continue
    seen.add(cs); sm.append(cs)
smiles=np.array(sm)
labels=np.random.RandomState(1).normal(3,1.2,size=len(sm))
out=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/e2e_out')
M.run_dataset('TINY', smiles, labels, out, model_filter=['NGBoost'], rep_filter=['ECFP4'],
              sigma_levels=[0.0,0.6], unc_strategies='all', oof_folds=5, strategies=['quantile'])
