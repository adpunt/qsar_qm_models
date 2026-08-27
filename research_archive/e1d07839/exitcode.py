import sys, numpy as np, importlib.util
from pathlib import Path
sys.argv=['x']
spec=importlib.util.spec_from_file_location('anr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# rep generation fails -> empty reps dict, exactly what happens when MHG-GNN fails
m.generate_representations=lambda sl, rep_filter=None: {}
m.assign_scaffold_groups=lambda s:(np.arange(len(s))%15,15)
m.N_FOLDS=3
out=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/emptyrun')
r=m.run_dataset('OpenADMET-LogD', np.array(['CCO']*60), np.random.randn(60), out,
                model_filter=['QRF'], rep_filter=['MHG-GNN-pretrained'],
                sigma_levels=[0.0,1.0], unc_strategies='all', oof_folds=3, strategies=['outlier'])
print("returned empty:", r.empty)
print("files written:", sorted(p.name for p in out.glob('*')))
